"""Full offline validation sessions; no real benchmark claims."""
import copy
import contextlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.frozen import freeze_policy, load_frozen_policy, policy_decision
from cave_mem_v3.policy.rollout import collect_policy_validation, summarize_policy_validation
from cave_mem_v3.target_data.sampling import RiskHazard
from cave_mem_v3.cli import main
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture
from test_validation_partitions import manifest
from test_single_execution import NoncloneableFixture


class PolicyRolloutTests(unittest.TestCase):
    def fixture(self, root, *, kind="always_remind", empty=False, calls=4, hazard=.99, safe=False, actor=ScriptedActor):
        fixture = make_prepared_fixture(root / "inputs", manifest=manifest(), empty_bank=empty)
        bundle_path = root / "bundle.json"
        config = {"kind": kind}
        if kind == "similarity":
            config["similarity_threshold"] = .99
        bundle = freeze_policy(fixture["manifest"], fixture["bank"], actor_config=actor().configuration(),
            encoder_identity=EncoderFixture.identity, study=StudySpec(max_steps=4, max_model_calls=calls),
            hazard=RiskHazard(low_risk_probability=hazard, high_risk_probability=hazard), policy_config=config, output=bundle_path)
        environments = []
        def env_factory(task):
            env = NoncloneableFixture(task_id=task["task_id"], safe_intent=safe)
            environments.append(env)
            return env
        return bundle_path, bundle, env_factory, environments

    def run_fixture(self, root, **kwargs):
        path, bundle, factory, envs = self.fixture(root, **kwargs)
        result = collect_policy_validation(path, env_factory=factory, actor_factory=kwargs.get("actor", ScriptedActor),
            encoder=EncoderFixture(), output_dir=root / "run")
        return result, path, bundle, envs

    def test_silent_and_remind_complete_full_tasks_and_reminder_is_once(self):
        class Observed(ScriptedActor):
            hints = []
            def propose_target(self, *args, **kwargs):
                self.hints.append(kwargs.get("reminder"))
                return super().propose_target(*args, **kwargs)
        for kind, expected, calls in (("silent", False, 1), ("always_remind", True, 4)):
            with tempfile.TemporaryDirectory() as tmp:
                Observed.hints = []
                root = Path(tmp)
                report, _, _, envs = self.run_fixture(root, kind=kind, actor=Observed)
                self.assertEqual(report["planned_tasks"], 1)
                self.assertEqual(report["observed_outcomes"], 1)
                row = report["tasks"][0]
                self.assertEqual(row["success"], expected)
                self.assertEqual(row["costs"]["model_calls"], calls)
                self.assertEqual(row["arm"], "silent" if kind == "silent" else "remind")
                self.assertEqual(len([hint for hint in Observed.hints if hint is not None]), int(expected))
                self.assertEqual(envs[-1].resets, 1)
                self.assertEqual(summarize_policy_validation(root / "run"), report)

    def test_no_memory_and_zero_remaining_calls_execute_cached_a0_and_keep_denominator(self):
        for options in ({"empty": True}, {"calls": 1}):
            with tempfile.TemporaryDirectory() as tmp:
                report, _, _, envs = self.run_fixture(Path(tmp), **options)
                row = report["tasks"][0]
                self.assertEqual(report["observed_outcomes"], 1)
                self.assertFalse(row["success"])
                self.assertEqual(row["arm"], "silent")
                self.assertEqual(row["costs"]["model_calls"], 1)
                self.assertEqual(envs[-1].actions, [{"type": "operate"}])

    def test_budget_exhaustion_uses_verifier_without_borrowing_reserved_slot(self):
        for calls, expected_calls in ((1, 1), (2, 1), (3, 2)):
            with tempfile.TemporaryDirectory() as tmp:
                report, _, _, envs = self.run_fixture(Path(tmp), kind="silent", safe=True, calls=calls)
                row = report["tasks"][0]
                self.assertTrue(row["outcome_valid"])
                self.assertEqual(row["costs"]["model_calls"], expected_calls)
                self.assertIn({"type": "prepare"}, envs[-1].actions)
                self.assertEqual(row["termination_reason"], "budget_exhausted")

    def test_no_checkpoint_tasks_have_terminal_outcome_and_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _, _, _ = self.run_fixture(Path(tmp), hazard=1e-9)
            row = report["tasks"][0]
            self.assertEqual(row["reason"], "no_checkpoint")
            self.assertTrue(row["outcome_valid"])
            self.assertIsNone(row["decision"])
            self.assertEqual(row["costs"]["model_calls"], 1)

    def test_pending_is_unknown_and_reentry_never_calls_factories(self):
        class Timeout(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                raise TimeoutError("unknown server response")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, path, _, _ = self.run_fixture(root, actor=Timeout)
            self.assertEqual(report["pending_tasks"], 1)
            self.assertIsNone(report["tasks"][0]["success"])
            self.assertEqual(report["tasks"][0]["costs"]["model_calls"], 2)
            never = Mock(side_effect=AssertionError("factory"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                collect_policy_validation(path, env_factory=never, actor_factory=never,
                    encoder=EncoderFixture(), output_dir=root / "run")
            never.assert_not_called()

    def test_unknown_fees_keep_outcome_but_utility_unknown(self):
        class Unknown(ScriptedActor):
            def propose(self, *args, **kwargs):
                response = super().propose(*args, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, bundle, factory, _ = self.fixture(root, kind="silent", actor=Unknown)
            bundle["study"]["inference_weights"] = {"api_cost": 1}
            write_json(path, seal(bundle))
            report = collect_policy_validation(path, env_factory=factory, actor_factory=Unknown,
                encoder=EncoderFixture(), output_dir=root / "run")
            self.assertEqual(report["observed_outcomes"], 1)
            self.assertEqual(report["valid_utilities"], 0)
            self.assertIsNone(report["tasks"][0]["utility"])

    def test_changed_remind_action_still_passes_environment_permission_validation(self):
        class Bad(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                response = super().propose_target(*args, **kwargs)
                return replace(response, action={"type": "forbidden"})
        with tempfile.TemporaryDirectory() as tmp:
            report, _, _, envs = self.run_fixture(Path(tmp), actor=Bad)
            row = report["tasks"][0]
            self.assertEqual(row["termination_reason"], "invalid_actor_output")
            self.assertEqual(envs[-1].actions, [])
            self.assertTrue(row["outcome_valid"])

    def test_test_role_is_refused_before_file_or_provider_access(self):
        never = Mock(side_effect=AssertionError("factory"))
        with self.assertRaisesRegex(ValueError, "test is locked"):
            collect_policy_validation("missing", env_factory=never, actor_factory=never,
                encoder=None, output_dir="missing", role="test", validation_role=None)
        with patch("cave_mem_v3.cli.load_config", side_effect=AssertionError("config")), contextlib.redirect_stderr(io.StringIO()):
            code = main(["collect-policy-validation", "--bundle", "missing", "--config", "missing", "--encoder-path", "missing",
                "--encoder-revision", "fixture", "--output", "missing", "--role", "test"])
        self.assertEqual(code, 2)

    def test_complete_reentry_no_calls_and_bundle_drift_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, path, bundle, _ = self.run_fixture(root)
            never = Mock(side_effect=AssertionError("factory"))
            self.assertEqual(collect_policy_validation(path, env_factory=never, actor_factory=never,
                encoder=EncoderFixture(), output_dir=root / "run"), report)
            bundle["policy_config"]["kind"] = "silent"
            write_json(path, seal(bundle))
            with self.assertRaisesRegex(ValueError, "package changed"):
                collect_policy_validation(path, env_factory=never, actor_factory=never,
                    encoder=EncoderFixture(), output_dir=root / "run")
            never.assert_not_called()

    def test_rules_and_same_task_seeds_are_frozen_before_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = []
            for kind in ("source", "applicability", "exposure", "hard_tristate", "similarity"):
                report, _, _, _ = self.run_fixture(root / kind, kind=kind)
                reports.append(report)
            self.assertEqual(reports[0]["tasks"][0]["arm"], "remind")
            self.assertEqual(reports[2]["tasks"][0]["arm"], "silent")
            self.assertEqual(reports[3]["tasks"][0]["arm"], "silent")
            self.assertEqual(len({r["tasks"][0]["decision"]["decision_seed"] for r in reports}), 1)
            self.assertEqual(len({r["tasks"][0]["decision"]["execution_seed"] for r in reports}), 1)

    def test_summary_rejects_resealed_decision_or_test_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_fixture(root)
            assignment = next((root / "run/sessions").glob("*/assignment.json"))
            value = read_json(assignment)
            value["decision"]["effective_probability"] = 0
            value["decision"] = seal(value["decision"])
            write_json(assignment, seal(value))
            with self.assertRaisesRegex(ValueError, "decision was changed"):
                summarize_policy_validation(root / "run")
            run = read_json(root / "run/run.json")
            run["role"] = "test"
            write_json(root / "run/run.json", seal(run))
            with self.assertRaisesRegex(ValueError, "test is locked"):
                summarize_policy_validation(root / "run")

    def test_local_decision_wall_cost_is_counted_once_and_reread_never_times_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, _, _, _ = self.run_fixture(root)
            row = report["tasks"][0]
            self.assertGreater(row["decision_costs"]["wall_seconds"], 0)
            self.assertAlmostEqual(row["costs"]["wall_seconds"], row["preparation_costs"]["wall_seconds"]
                + row["decision_costs"]["wall_seconds"] + row["continuation_costs"]["wall_seconds"], places=12)
            with patch("cave_mem_v3.policy.rollout.time.perf_counter", side_effect=AssertionError("no timing on summary")):
                self.assertEqual(summarize_policy_validation(root / "run"), report)

    def test_freeze_and_validation_cli_with_local_encoder_fixture(self):
        class LocalEncoder(EncoderFixture):
            def __init__(self, *args, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "inputs", manifest=manifest())
            for name, value in (("manifest", fixture["manifest"]), ("bank", fixture["bank"]),
                ("encoder", EncoderFixture.identity), ("hazard", {"low_risk_probability": .99, "high_risk_probability": .99}),
                ("policy", {"kind": "always_remind"})):
                write_json(root / (name + ".json"), value)
            (root / "model-dir").mkdir()
            config = Path(__file__).resolve().parents[1] / "configs/phase1/demo.toml"
            with contextlib.redirect_stdout(io.StringIO()), patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", LocalEncoder):
                self.assertEqual(main(["freeze-policy", "--config", str(config), "--manifest", str(root / "manifest.json"),
                    "--bank", str(root / "bank.json"), "--encoder-identity", str(root / "encoder.json"),
                    "--hazard", str(root / "hazard.json"), "--policy-config", str(root / "policy.json"), "--output", str(root / "bundle.json")]), 0)
                collect_args = ["collect-policy-validation", "--bundle", str(root / "bundle.json"), "--config", str(config),
                    "--encoder-path", str(root / "model-dir"), "--encoder-revision", "fixture", "--output", str(root / "run")]
                self.assertEqual(main(collect_args), 0)
                with patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", side_effect=AssertionError("encoder reload")), \
                     patch("cave_mem_v3.cli._actor", side_effect=AssertionError("actor reload")):
                    self.assertEqual(main(collect_args), 0)
                    changed = list(collect_args)
                    changed[changed.index("--encoder-revision") + 1] = "changed"
                    with contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(main(changed), 2)
                self.assertEqual(main(["summarize-policy-validation", "--run", str(root / "run")]), 0)
            self.assertEqual(summarize_policy_validation(root / "run")["observed_successes"], 1)

    def test_learned_fitted_bundles_are_self_contained_and_share_uncertainty_wrapper(self):
        import test_bandit_policy as support
        from cave_mem_v3.policy.frozen import validate_bundle
        support.BanditPolicyTests.setUpClass()
        try:
            helper = support.BanditPolicyTests()
            data, model, cal, gate = helper.data, helper.model, helper.calibration, helper.gate
            bandit = helper.train()
            base = support.support.SupervisedPolicyTests.base
            run = read_json(Path(base["sources"][0]["run"]) / "run.json")
            prepared = run["prepared_run"]
            baseline = freeze_policy(prepared["manifest"], run["bank"], actor_config=prepared["actor_config"],
                encoder_identity=prepared["pre_treatment"]["encoder_identity"], study=StudySpec(**prepared["study"]),
                hazard=RiskHazard(**prepared["risk_hazard"]), policy_config={"kind": "silent"})
            x = data["rows"][-1]["features"]
            item = seal({"pre_treatment_state": {"x": x}, "a0_hash": "fixture-a0", "task_id": "fixed-task"})
            identities = []
            for kind in ("supervised", "bandit", "no_risk_bandit"):
                bundle = copy.deepcopy(baseline)
                bundle["policy_config"] = {"kind": kind, "risk_threshold": .9, "minimum_advantage": 0}
                bundle["artifacts"] = {"model": model, "calibration": cal, "warm_gate": gate}
                if kind != "supervised":
                    bundle["artifacts"]["bandit"] = bandit
                bundle = seal(bundle)
                with patch("cave_mem_v3.policy.supervised.read_json", side_effect=AssertionError("raw files")), \
                     patch("cave_mem_v3.policy.calibration.load_policy_dataset", side_effect=AssertionError("raw dataset")), \
                     patch("cave_mem_v3.policy.bandit.train_bandit", side_effect=AssertionError("retraining")):
                    validate_bundle(bundle)
                    decision = policy_decision(bundle, item, "trace-unused")
                self.assertGreater(decision["effective_probability"], 0)
                self.assertGreater(decision["raw_probability"], 0)
                identities.append(bundle["record_hash"])
                guarded = copy.deepcopy(bundle)
                guarded["policy_config"]["minimum_advantage"] = 999
                guarded = seal(guarded)
                decision = policy_decision(guarded, item, "trace-unused")
                self.assertEqual(decision["effective_probability"], 0)
                self.assertEqual(decision["reason"], "benefit_not_supported")
                self.assertNotEqual(guarded["record_hash"], bundle["record_hash"])
            self.assertEqual(len(set(identities)), 3)
            bad = copy.deepcopy(bundle)
            bad["artifacts"]["bandit"]["no_risk_comparator"]["history"][-1]["actor_calls_upper"] = 999
            bad["artifacts"]["bandit"] = seal(bad["artifacts"]["bandit"])
            with self.assertRaises(ValueError):
                validate_bundle(seal(bad))
        finally:
            support.BanditPolicyTests.tearDownClass()

    def test_public_learned_freeze_with_insufficient_evidence_rolls_out_silent(self):
        from cave_mem_v3.policy.supervised import fit_supervised
        from cave_mem_v3.policy.calibration import calibrate_supervised
        from cave_mem_v3.policy.gate import GateConfig, fit_initial_gate
        from cave_mem_v3.policy.bandit import BanditConfig, train_bandit
        from cave_mem_v3.policy.supervised import SupervisedConfig
        from cave_mem_v3.policy.dataset import write_policy_dataset
        from cave_mem_v3.interventions.collection import collect_target_pairs
        from test_validation_partitions import collect
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "inputs", manifest=manifest())
            collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                actor_factory=ScriptedActor, output_dir=root / "pairs")
            write_policy_dataset([{"kind": "paired", "run": str(root / "pairs"), "prepared": str(fixture["target_dir"])}], root / "train.json")
            cal_source = collect(root / "cal", fixture, "calibration", "paired")
            write_policy_dataset([cal_source], root / "cal.json", role="validation", validation_role="calibration")
            fit_supervised(root / "train.json", config=SupervisedConfig(gamma=1.0), output=root / "model.json")
            calibrate_supervised(root / "model.json", root / "cal.json", output=root / "bounds.json")
            fit_initial_gate(root / "model.json", root / "bounds.json", root / "train.json", config=GateConfig(.1), output=root / "gate.json")
            train_bandit(root / "train.json", root / "model.json", root / "bounds.json", root / "gate.json",
                config=BanditConfig("paired", .1, 5), output=root / "bandit.json")
            for kind in ("supervised", "bandit", "no_risk_bandit"):
                path = root / (kind + ".bundle.json")
                freeze_policy(fixture["manifest"], fixture["bank"], actor_config=ScriptedActor().configuration(),
                    encoder_identity=EncoderFixture.identity, study=StudySpec(max_steps=4, max_model_calls=4),
                    hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
                    policy_config={"kind": kind, "minimum_advantage": 0, "risk_threshold": .1},
                    model_path=root / "model.json", calibration_path=root / "bounds.json", warm_gate_path=root / "gate.json",
                    bandit_path=root / "bandit.json" if kind != "supervised" else None, output=path)
                with patch("cave_mem_v3.policy.bandit.load_bandit", side_effect=AssertionError("raw training files")), \
                     patch("cave_mem_v3.policy.supervised.load_policy_dataset", side_effect=AssertionError("raw training files")):
                    report = collect_policy_validation(path, env_factory=fixture["env_factory"], actor_factory=ScriptedActor,
                        encoder=EncoderFixture(), output_dir=root / (kind + "-run"))
                self.assertEqual(report["tasks"][0]["arm"], "silent")
                self.assertEqual(report["tasks"][0]["decision"]["reason"], "uncertainty_unavailable")
                self.assertEqual(report["observed_outcomes"], 1)

    def test_shared_configuration_drift_stops_after_one_task_and_keeps_unknown(self):
        from cave_mem_v3.benchmarks.tasks import build_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            m = manifest()
            extra = {"task_id": "selection-b", "group_id": "selection-b-family", "role": "validation", "validation_role": "selection"}
            m = build_manifest(m["tasks"] + [extra], benchmark=m["benchmark"], revision=m["revision"])
            fixture = make_prepared_fixture(root / "inputs", manifest=m)
            class Drift(ScriptedActor):
                def propose(self, *args, **kwargs):
                    response = super().propose(*args, **kwargs)
                    self.error_rate = .25
                    return response
            freeze_policy(m, fixture["bank"], actor_config=ScriptedActor().configuration(), encoder_identity=EncoderFixture.identity,
                study=StudySpec(max_steps=4, max_model_calls=4), hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
                policy_config={"kind": "always_remind"}, output=root / "bundle.json")
            report = collect_policy_validation(root / "bundle.json", env_factory=fixture["env_factory"], actor_factory=Drift,
                encoder=EncoderFixture(), output_dir=root / "run")
            self.assertEqual(report["attempted_tasks"], 1)
            self.assertEqual(report["observed_outcomes"], 0)
            self.assertEqual(report["unknown_outcomes"], 2)
            self.assertEqual(report["error_tasks"], 1)
            never = Mock(side_effect=AssertionError("retry"))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                collect_policy_validation(root / "bundle.json", env_factory=never, actor_factory=never,
                    encoder=EncoderFixture(), output_dir=root / "run")

    def test_changed_run_protocol_or_capabilities_refused_before_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_fixture(root)
            original = read_json(root / "run/run.json")
            for field, value in (("protocol", "unreviewed"), ("actor_capabilities", {"stateless": False}), ("environment_capabilities", {})):
                folder = root / field
                run = copy.deepcopy(original)
                run[field] = value
                write_json(folder / "run.json", seal(run))
                with patch("cave_mem_v3.policy.rollout._session", side_effect=AssertionError("outcomes")):
                    with self.assertRaises(ValueError):
                        summarize_policy_validation(folder)
