"""Prospective software-fixture selection; no real research result."""
import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.frozen import freeze_policy
from cave_mem_v3.policy.selection import (SelectionConfig, create_selection_plan, load_selection_plan,
    collect_selection_plan, evaluate_selection_plan, select_policy, load_selection_lock, _group_interval)
from cave_mem_v3.target_data.sampling import RiskHazard
from cave_mem_v3.cli import main
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture
from test_validation_partitions import manifest


class PolicySelectionTests(unittest.TestCase):
    def fixture(self, root, *, tasks=3, policies=None, hazard=.99, risk=1, budget=5, config=None, study=None):
        m = manifest()
        extra = [{"task_id": "selection-" + str(i), "group_id": "selection-family-" + str(i),
            "role": "validation", "validation_role": "selection"} for i in range(tasks - 1)]
        m = build_manifest(m["tasks"] + extra, benchmark=m["benchmark"], revision=m["revision"])
        fixture = make_prepared_fixture(root / "inputs", manifest=m)
        candidates = []
        for identifier, kind in policies or [("silent", "silent"), ("remind", "always_remind")]:
            bundle_path = root / (identifier + ".bundle.json")
            freeze_policy(m, fixture["bank"], actor_config=ScriptedActor().configuration(), encoder_identity=EncoderFixture.identity,
                study=study or StudySpec(max_steps=4, max_model_calls=4), hazard=RiskHazard(low_risk_probability=hazard, high_risk_probability=hazard),
                policy_config={"kind": kind}, output=bundle_path)
            candidates.append({"candidate_id": identifier, "bundle": str(bundle_path), "rollout_dir": str(root / (identifier + "-slot"))})
        plan = create_selection_plan(candidates, audit_dir=root / "audit-slot", lock_path=root / "lock.json",
            config=config or SelectionConfig(risk, budget, bootstrap_samples=30), output=root / "plan.json")
        return fixture, plan

    def collect(self, root, fixture):
        return collect_selection_plan(root / "plan.json", env_factory=fixture["env_factory"], actor_factory=ScriptedActor, encoder=EncoderFixture())

    def test_raw_multitask_full_return_selects_feasible_and_reloads_immutable_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.assertEqual(self.collect(root, fixture)["status"], "collected")
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("read-only selection")):
                result = select_policy(root / "plan.json")
                self.assertEqual(load_selection_lock(root / "lock.json"), result)
            self.assertEqual(result["status"], "selected")
            self.assertEqual(result["winner"]["candidate_id"], "remind")
            self.assertEqual(result["test_access"], "locked_pending_final_evaluation_protocol")
            reports = {c["candidate_id"]: c for c in result["candidates"]}
            self.assertGreater(reports["remind"]["full_task_mean_utility"], reports["silent"]["full_task_mean_utility"])
            self.assertGreater(reports["remind"]["risk"]["upper"], 0)
            self.assertEqual(reports["silent"]["risk"]["upper"], 0)
            self.assertEqual(reports["remind"]["audit_denominator"]["planned"], 3)
            self.assertTrue(all(c["full_task_denominator"]["planned_tasks"] == 3 for c in result["candidates"]))
            before = (root / "lock.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "lock exists"):
                select_policy(root / "plan.json")
            self.assertEqual((root / "lock.json").read_bytes(), before)

    def test_insufficient_group_support_creates_explicit_no_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, tasks=1)
            self.collect(root, fixture)
            result = select_policy(root / "plan.json")
            self.assertEqual(result["status"], "no_selection")
            self.assertIsNone(result["winner"])
            self.assertTrue(all("full_task_groups_insufficient" in c["reasons"] for c in result["candidates"]))

    def test_zero_observed_harm_is_not_zero_uncertainty_and_strict_limit_selects_silent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, risk=0)
            self.collect(root, fixture)
            result = select_policy(root / "plan.json")
            self.assertEqual(result["winner"]["candidate_id"], "silent")
            remind = next(c for c in result["candidates"] if c["candidate_id"] == "remind")
            self.assertEqual(remind["risk"]["mean"], 0)
            self.assertGreater(remind["risk"]["finite_support_upper"], 0)
            self.assertIn("risk_upper_exceeds_limit", remind["reasons"])

    def test_infeasible_cost_budget_yields_no_winner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, budget=0)
            self.collect(root, fixture)
            result = select_policy(root / "plan.json")
            self.assertIsNone(result["winner"])
            self.assertTrue(all("actor_calls_upper_exceeds_limit" in c["reasons"] for c in result["candidates"]))

    def test_tie_break_is_predeclared_and_candidate_order_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root, policies=[("z-source", "source"), ("a-always", "always_remind")])
            self.assertEqual(plan["candidate_order"], ["a-always", "z-source"])
            self.collect(root, fixture)
            result = select_policy(root / "plan.json")
            self.assertEqual(result["winner"]["candidate_id"], "a-always")

    def test_plan_rejects_preexisting_outputs_duplicate_policy_or_nested_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, plan = self.fixture(root)
            candidate = plan["candidates"][0]
            def make(entries, audit, lock, output):
                return create_selection_plan(entries, audit_dir=audit, lock_path=lock, config=SelectionConfig(1, 5), output=output)
            entry = {"candidate_id": "a", "bundle": candidate["bundle_path"], "rollout_dir": str(root / "exists")}
            (root / "exists").mkdir()
            with self.assertRaisesRegex(ValueError, "before any"):
                make([entry], root / "audit2", root / "lock2", root / "plan2")
            entry["rollout_dir"] = str(root / "new")
            with self.assertRaisesRegex(ValueError, "duplicate frozen"):
                make([entry, {**entry, "candidate_id": "b", "rollout_dir": str(root / "other")}], root / "audit2", root / "lock2", root / "plan2")
            with self.assertRaisesRegex(ValueError, "contain"):
                make([entry], root / "new/sub", root / "lock2", root / "plan2")

    def test_missing_or_unbound_data_never_select_and_cannot_collect_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            report = evaluate_selection_plan(root / "plan.json")
            self.assertIsNone(report["winner"])
            path = Path(plan["candidates"][0]["rollout_dir"])
            path.mkdir()
            with self.assertRaisesRegex(ValueError, "not bound"):
                evaluate_selection_plan(root / "plan.json")
            never = lambda *args: (_ for _ in ()).throw(AssertionError("provider"))
            with self.assertRaisesRegex(ValueError, "already exists"):
                collect_selection_plan(root / "plan.json", env_factory=never, actor_factory=never, encoder=EncoderFixture())

    def test_tampered_binding_plan_bundle_and_locked_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            binding_path = Path(plan["audit_dir"]) / "binding.json"
            original = read_json(binding_path)
            bad = copy.deepcopy(original)
            bad["plan_hash"] = "another-plan"
            write_json(binding_path, seal(bad))
            with self.assertRaisesRegex(ValueError, "not bound"):
                evaluate_selection_plan(root / "plan.json")
            write_json(binding_path, original)
            result = select_policy(root / "plan.json")
            result["winner"]["candidate_id"] = "silent"
            write_json(root / "lock.json", seal(result))
            with self.assertRaisesRegex(ValueError, "original prospective"):
                load_selection_lock(root / "lock.json")
            bundle_path = Path(plan["candidates"][0]["bundle_path"])
            bundle = read_json(bundle_path)
            bundle["study"]["base_seed"] += 1
            write_json(bundle_path, seal(bundle))
            with self.assertRaisesRegex(ValueError, "changed after"):
                load_selection_plan(root / "plan.json")

    def test_equal_task_ranking_and_cluster_interval_mean_agree_with_unequal_groups(self):
        config = SelectionConfig(1, 5, bootstrap_samples=30)
        result = _group_interval([1, 1, 1, 0], ["a", "a", "a", "b"], config, .1, equal_task=True)
        self.assertEqual(result["mean"], .75)
        group = _group_interval([1, 1, 1, 0], ["a", "a", "a", "b"], config, .1)
        self.assertEqual(group["mean"], .5)

    def test_cli_selection_only_reads_registered_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, tasks=1)
            self.collect(root, fixture)
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("no actor")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["select-policy", "--plan", str(root / "plan.json")]), 0)
            self.assertEqual(read_json(root / "lock.json")["status"], "no_selection")

    def test_sample_zero_probability_contextual_rule_does_not_receive_global_silent_exemption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, policies=[("a-exposure", "exposure"), ("z-silent", "silent")], risk=0)
            self.collect(root, fixture)
            result = select_policy(root / "plan.json")
            self.assertEqual(result["winner"]["candidate_id"], "z-silent")
            exposure = next(c for c in result["candidates"] if c["candidate_id"] == "a-exposure")
            self.assertTrue(all(r["effective_probability"] == 0 for r in exposure["checkpoint_audit"]))
            self.assertFalse(exposure["risk"]["structural_zero_reminder_probability"])
            self.assertGreater(exposure["risk"]["upper"], 0)

    def test_missing_registered_candidate_blocks_otherwise_feasible_winner(self):
        import cave_mem_v3.policy.selection as selection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            original = selection._check_slot
            omitted = plan["candidates"][0]["rollout_dir"]
            def missing(path, *args):
                return False if str(path) == omitted else original(path, *args)
            with patch.object(selection, "_check_slot", side_effect=missing):
                result = evaluate_selection_plan(root / "plan.json")
            self.assertIsNone(result["winner"])
            self.assertFalse(result["registered_plan_complete"])
            self.assertTrue(all("registered_plan_incomplete" in c["reasons"] for c in result["candidates"]))

    def test_unknown_full_task_cost_prevents_any_qualified_winner(self):
        class Unknown(ScriptedActor):
            def propose(self, *args, **kwargs):
                response = super().propose(*args, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root, study=StudySpec(max_steps=4, max_model_calls=4, inference_weights={"api_cost": 1}))
            collect_selection_plan(root / "plan.json", env_factory=fixture["env_factory"], actor_factory=Unknown, encoder=EncoderFixture())
            result = select_policy(root / "plan.json")
            self.assertFalse(result["registered_plan_complete"])
            self.assertIsNone(result["winner"])
            self.assertTrue(all(c["full_task_denominator"]["observed_outcomes"] == 3 for c in result["candidates"]))
            self.assertTrue(all(c["full_task_denominator"]["valid_utilities"] == 0 for c in result["candidates"]))

    def test_reserved_paths_cannot_write_inside_protected_raw_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            old = plan["candidates"][0]
            bundle = copy.deepcopy(old["bundle"])
            bundle["protected_source_roots"] = [str(fixture["target_dir"].resolve())]
            write_json(root / "protected.bundle.json", seal(bundle))
            entry = {"candidate_id": "new", "bundle": str(root / "protected.bundle.json"),
                "rollout_dir": str(fixture["target_dir"] / "new-selection-slot")}
            with self.assertRaisesRegex(ValueError, "protected raw"):
                create_selection_plan([entry], audit_dir=root / "audit-new", lock_path=root / "new.lock",
                    config=SelectionConfig(1, 5), output=root / "new.plan")
            self.assertFalse((fixture["target_dir"] / "new-selection-slot").exists())

    def test_actual_served_identity_conflict_between_live_and_audit_is_refused(self):
        import cave_mem_v3.policy.selection as selection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            dataset, reason = selection._load_audit(plan)
            dataset["served_identity"]["fingerprint"] = "different-validation-backend"
            with patch.object(selection, "_load_audit", return_value=(dataset, reason)):
                with self.assertRaisesRegex(ValueError, "served model/fingerprint"):
                    evaluate_selection_plan(root / "plan.json")

    def test_same_configuration_unregistered_raw_pair_cannot_replace_plan_audit(self):
        from cave_mem_v3.target_data.collection import collect_target
        from cave_mem_v3.interventions.collection import collect_target_pairs
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            bundle = plan["candidates"][0]["bundle"]
            collect_target(bundle["manifest"], role="validation", validation_role="selection", spec=StudySpec(**bundle["study"]),
                env_factory=fixture["env_factory"], actor_factory=ScriptedActor, output_dir=root / "old-prepared",
                encoder=EncoderFixture(), bank=bundle["bank"], hazard=RiskHazard(**bundle["risk_hazard"]), prepare_before_action=True)
            collect_target_pairs(root / "old-prepared", bundle["bank"], env_factory=fixture["env_factory"],
                actor_factory=ScriptedActor, output_dir=root / "old-paired")
            self.collect(root, fixture)
            # Replace just the raw prepared header to model an old same-config run.
            # The slot sidecar remains correct, so raw-level binding must reject first.
            shutil.copyfile(root / "old-prepared/run.json", Path(plan["audit_dir"]) / "prepared/run.json")
            with patch("cave_mem_v3.policy.selection.build_policy_dataset", side_effect=AssertionError("outcome read")):
                with self.assertRaisesRegex(ValueError, "not bound to this prospective"):
                    evaluate_selection_plan(root / "plan.json")

    def test_partial_live_identity_blocks_paired_plan_even_with_known_values(self):
        class Partial(ScriptedActor):
            def _propose(self, *args, **kwargs):
                response = super()._propose(*args, **kwargs)
                return replace(response, fingerprint=None) if response.action['type'] == 'operate' else response
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = self.fixture(root)
            created = []
            def actor_factory():
                actor = Partial() if len(created) < 4 else ScriptedActor()
                created.append(actor)
                return actor
            collect_selection_plan(root / 'plan.json', env_factory=fixture['env_factory'], actor_factory=actor_factory, encoder=EncoderFixture())
            result = select_policy(root / 'plan.json')
            self.assertEqual({c['served_identity']['status'] for c in result['candidates']}, {'unknown', 'reported'})
            self.assertTrue(all(c['served_identity']['fingerprint'] for c in result['candidates']))
            self.assertIsNone(result['winner'])
            self.assertFalse(result['registered_plan_complete'])

    def test_unknown_paired_audit_identity_blocks_all_live_candidates(self):
        import cave_mem_v3.policy.selection as selection
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, plan = self.fixture(root)
            self.collect(root, fixture)
            dataset, reason = selection._load_audit(plan)
            dataset['served_identity']['status'] = 'unknown'
            with patch.object(selection, '_load_audit', return_value=(dataset, reason)):
                result = evaluate_selection_plan(root / 'plan.json')
            self.assertIsNone(result['winner'])
            self.assertFalse(result['registered_plan_complete'])
            self.assertTrue(all('actual_audit_served_identity_unknown' in c['reasons'] for c in result['candidates']))
