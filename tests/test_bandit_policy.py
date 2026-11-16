"""Software-only constrained optimization; no experiment parameter selection."""
import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.common.contracts import read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.bandit import (BanditConfig, _train_dataset, bootstrap_draws, empirical_upper,
    objective_and_gradient, optimize_gate, predict_bandit, train_bandit, load_bandit)
from cave_mem_v3.policy.supervised import SupervisedConfig, _numpy, fit_supervised
from cave_mem_v3.policy.calibration import calibrate_supervised
from cave_mem_v3.policy.gate import GateConfig, _fit_gate_dataset, fit_initial_gate
from cave_mem_v3.policy.dataset import write_policy_dataset
from cave_mem_v3.cli import main
import test_supervised_policy as support
from test_target_pair_inputs import make_prepared_fixture
from test_validation_partitions import manifest, collect


class BanditPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        support.SupervisedPolicyTests.setUpClass()
        helper = support.SupervisedPolicyTests()
        cls.data = helper.synthetic()
        cls.model = helper.fit(cls.data)
        cls.calibration = helper.calibration(cls.model)
        cls.gate = _fit_gate_dataset(cls.model, cls.calibration, cls.data, GateConfig(.9))
        cls.audits = {r["sample_id"]: {"status": "verified", "c0_actor_calls": 1.0,
            "source": "synthetic_test_only", "selected_response_hash": "fixture"} for r in cls.data["rows"]}

    @classmethod
    def tearDownClass(cls):
        support.SupervisedPolicyTests.tearDownClass()

    def config(self, **kwargs):
        return BanditConfig(route="paired", risk_limit=.9, actor_call_budget=10, iterations=60,
            bootstrap_samples=30, **kwargs)

    def train(self, config=None, data=None, model=None, calibration=None, gate=None, audits=None, dr=None):
        return _train_dataset(data or self.data, model or self.model, calibration or self.calibration,
            gate or self.gate, config or self.config(), self.audits if audits is None else audits, dr)

    def test_direct_objective_reproducible_same_capacity_and_no_risk_comparator(self):
        result = self.train()
        self.assertEqual(result, self.train())
        self.assertEqual(result["status"], "candidate")
        self.assertEqual(result["capacity"], self.model["capacity"])
        self.assertEqual(len(result["training"]["final_coefficients"]), len(self.gate["head"]["coefficients"]))
        self.assertGreater(result["training"]["history"][-1]["objective"], result["training"]["history"][0]["objective"])
        self.assertTrue(all(row["risk_dual"] == 0 for row in result["no_risk_comparator"]["history"]))
        self.assertEqual(result["cost_contract"]["delta"], "audited signed paired Remind minus Silent model_calls")

    def test_objective_kl_and_constraint_gradient_matches_finite_difference(self):
        np = _numpy()
        config = self.config()
        matrix = np.asarray([[1, -.9], [1, .2], [1, .7], [1, 1.1]])
        groups = ["a", "b", "b", "c"]
        draws = bootstrap_draws(sorted(set(groups)), config)
        theta, reference = np.asarray([.2, -.3]), np.asarray([-.1, .5])
        args = (reference, matrix, [1, -.2, .8, .4], [.1, .5, .3, .8], [-.2, .3, 1, .2], groups, draws, config)
        actual = objective_and_gradient(theta, *args, risk_dual=.3, cost_dual=.4)
        for i in range(2):
            step = np.zeros(2)
            step[i] = 1e-6
            high = objective_and_gradient(theta + step, *args, risk_dual=.3, cost_dual=.4)["lagrangian"]
            low = objective_and_gradient(theta - step, *args, risk_dual=.3, cost_dual=.4)["lagrangian"]
            self.assertAlmostEqual(actual["gradient"][i], (high - low) / 2e-6, places=6)
        baseline = objective_and_gradient(reference, *args)
        self.assertAlmostEqual(baseline["kl"], 0, places=12)

    def test_dual_updates_and_risk_constraint_reduce_reminders(self):
        np = _numpy()
        config = BanditConfig("paired", risk_limit=.05, actor_call_budget=10, iterations=500,
            learning_rate=.3, dual_learning_rate=1, kl_weight=.01, bootstrap_samples=20)
        groups = ["a", "b", "c", "d"]
        draws = bootstrap_draws(groups, config)
        args = ([0.0], np.ones((4, 1)), [1] * 4, [1] * 4, [0] * 4, groups, draws, config)
        constrained = optimize_gate(*args)
        control = optimize_gate(*args, risk_constraint=False)
        self.assertGreater(constrained["history"][1]["risk_dual"], 0)
        self.assertTrue(all(row["risk_dual"] >= 0 and row["cost_dual"] >= 0 for row in constrained["history"]))
        self.assertLess(constrained["history"][-1]["mean_remind_probability"], control["history"][-1]["mean_remind_probability"])
        self.assertLess(constrained["history"][-1]["risk_upper"], .05)

    def test_cost_dual_reacts_to_infeasible_budget_and_never_emits_candidate(self):
        config = replace(self.config(), actor_call_budget=0, iterations=5)
        result = self.train(config)
        self.assertEqual(result["status"], "infeasible")
        self.assertGreater(result["training"]["history"][1]["cost_dual"], 0)
        self.assertFalse(result["training"]["empirical_constraints_pass"])
        self.assertEqual(predict_bandit(result, self.data["rows"][0]["features"])["effective_probability"], 0)

    def test_bootstrap_uses_group_units_not_duplicate_rows(self):
        np = _numpy()
        config = self.config()
        groups = ["a", "b"]
        draws = bootstrap_draws(groups, config)
        first = empirical_upper([.1, .9], [[1], [3]], groups, draws, .1)
        second = empirical_upper([.1] * 10 + [.9], [[1]] * 10 + [[3]], ["a"] * 10 + ["b"], draws, .1)
        self.assertAlmostEqual(first["mean"], second["mean"])
        self.assertAlmostEqual(first["upper"], second["upper"])
        np.testing.assert_allclose(first["gradient"], second["gradient"])

    def test_masked_silent_rows_remain_in_denominator_cost_and_inference(self):
        data = copy.deepcopy(self.data)
        for row in data["rows"][20:]:
            row["features"]["v"] = "?"
            row["features"]["m"] = None
            row["labels"]["D"] = None
            row["label_masks"]["advantage"] = False
        result = self.train(data=data)
        self.assertEqual(result["denominators"]["ready"], 40)
        self.assertEqual(result["denominators"]["pre_treatment_eligible"], 20)
        self.assertEqual(sum(r["status"] == "masked_silent" for r in result["rows"]), 20)
        self.assertGreaterEqual(result["training"]["history"][0]["actor_calls_mean"], 1)
        self.assertLessEqual(result["training"]["history"][0]["mean_remind_probability"], .5)
        if result["status"] == "candidate":
            predicted = predict_bandit(result, data["rows"][-1]["features"])
            self.assertEqual(predicted["effective_probability"], 0)
            self.assertFalse(predicted["eligible"])

    def test_missing_cost_risk_or_reward_blocks_whole_candidate(self):
        for field in ("advantage", "cost", "audit"):
            data = copy.deepcopy(self.data)
            audits = copy.deepcopy(self.audits)
            if field == "advantage":
                data["rows"][0]["label_masks"]["advantage"] = False
            elif field == "cost":
                data["rows"][0]["costs"]["delta_valid"]["model_calls"] = False
            else:
                audits.pop(data["rows"][0]["sample_id"])
            result = self.train(data=data, audits=audits)
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["reason"], "incomplete_eligible_support")
            self.assertIsNone(result["training"])

    def test_single_route_consumes_estimates_with_worstcase_risk_and_cost(self):
        data = copy.deepcopy(self.data)
        dr_rows = []
        for row in data["rows"]:
            row["kind"] = "single"
            row["labels"] = {"Y": True, "R": 1}
            dr_rows.append({"sample_id": row["sample_id"], "dr_training_mask": True, "status": "available",
                "raw": {"estimated_dr_advantage": .4}})
        data = seal(data)
        dr = seal({"dataset_hash": data["record_hash"], "config": {"group_axis": "target_group"}, "rows": dr_rows})
        result = self.train(config=replace(self.config(), route="single"), data=data, dr=dr)
        self.assertTrue(all(row["risk_bound"] == 1 for row in result["rows"]))
        self.assertTrue(all(row["delta_actor_calls"] == 3 for row in result["rows"]))
        self.assertTrue(all(row["advantage"] == .4 for row in result["rows"]))
        self.assertNotIn("H", result["rows"][0])
        with self.assertRaises(ValueError):
            self.train(config=replace(self.config(), route="single"), data=data)

    def test_explicit_units_roles_and_signed_paired_costs(self):
        with self.assertRaises(ValueError):
            replace(self.config(), cost_unit="total_task_calls")
        data = copy.deepcopy(self.data)
        data["available_roles"] = ["validation/selection"]
        with self.assertRaises(ValueError):
            self.train(data=data)
        data = copy.deepcopy(self.data)
        for row in data["rows"]:
            row["costs"]["delta"]["model_calls"] = -1
        result = self.train(data=data)
        self.assertLess(result["training"]["history"][0]["actor_calls_mean"], 1)
        self.assertTrue(all(row["delta_actor_calls"] == -1 for row in result["rows"]))

    def test_public_cli_raw_artifact_audit_and_unavailable_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            from cave_mem_v3.interventions.collection import collect_target_pairs
            collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                actor_factory=fixture["actor_factory"], output_dir=root / "paired")
            write_policy_dataset([{"kind": "paired", "run": str(root / "paired"), "prepared": str(fixture["target_dir"])}], root / "train.json")
            source = collect(root / "cal", fixture, "calibration", "paired")
            write_policy_dataset([source], root / "cal.json", role="validation", validation_role="calibration")
            fit_supervised(root / "train.json", config=SupervisedConfig(gamma=1.0), output=root / "model.json")
            calibrate_supervised(root / "model.json", root / "cal.json", output=root / "bounds.json")
            fit_initial_gate(root / "model.json", root / "bounds.json", root / "train.json", config=GateConfig(.1), output=root / "gate.json")
            write_json(root / "config.json", asdict(self.config()))
            args = ["train-bandit", "--dataset", str(root / "train.json"), "--model", str(root / "model.json"),
                "--calibration", str(root / "bounds.json"), "--warm-gate", str(root / "gate.json"),
                "--config", str(root / "config.json"), "--output", str(root / "bandit.json")]
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("no actor")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            result = load_bandit(root / "bandit.json")
            self.assertEqual(result["status"], "unavailable")
            self.assertEqual(result["rows"][0]["cost_audit"]["c0_actor_calls"], 1)
            self.assertIsNone(result["training"])
            result["rows"][0]["cost_audit"]["c0_actor_calls"] = 0
            write_json(root / "bandit.json", seal(result))
            with self.assertRaisesRegex(ValueError, "audited"):
                load_bandit(root / "bandit.json")

    def test_real_excluded_selected_checkpoint_retains_cached_call_cost_audit(self):
        from cave_mem_v3.policy.bandit import _cached_call_audits
        from cave_mem_v3.policy.dataset import build_policy_dataset
        from cave_mem_v3.interventions.collection import collect_target_pairs
        for kwargs in ({"empty_bank": True}, {"target_max_calls": 1}):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = make_prepared_fixture(root / "inputs", **kwargs)
                collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                    actor_factory=fixture["actor_factory"], output_dir=root / "pair")
                data = build_policy_dataset([{"kind": "paired", "run": str(root / "pair"), "prepared": str(fixture["target_dir"])}])
                audits = _cached_call_audits(data)
                self.assertEqual(len(audits), 1)
                self.assertEqual(audits[data["rows"][0]["sample_id"]]["c0_actor_calls"], 1)

    def test_unknown_planned_frozen_state_blocks_and_rho_e_unknown_are_soft(self):
        data = copy.deepcopy(self.data)
        data["rows"][0]["features"] = None
        data["rows"][0]["label_masks"]["features"] = False
        data["rows"][0]["status"] = "pending"
        data["rows"][0]["reason"] = "preparation_missing"
        result = self.train(data=data)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["rows"][0]["reason"], "planned_frozen_state_unavailable")
        result = self.train()
        self.assertEqual(result["denominators"]["pre_treatment_eligible"], 40)
        self.assertTrue(all(r["features"]["rho"] == "?" and r["features"]["e"]["value"] == "?" for r in self.data["rows"]))

    def test_resealed_infeasible_and_nonfinite_prediction_candidates_refused(self):
        artifact = self.train()
        bad = copy.deepcopy(artifact)
        bad["training"]["history"][-1]["risk_upper"] = 1.0
        with self.assertRaises(ValueError):
            predict_bandit(seal(bad), self.data["rows"][0]["features"])
        bad = copy.deepcopy(artifact)
        bad["normalization"]["scale"][0] = 0
        with self.assertRaises(ValueError):
            predict_bandit(seal(bad), self.data["rows"][0]["features"])
