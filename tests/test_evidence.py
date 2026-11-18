import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cave_mem_v3.common.contracts import StudySpec, digest, read_json, write_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.memory_generation.bank import build_memory, freeze_bank, load_bank, validate_memory
from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.memory_generation.replay import (
    SOURCE_DECISION_RULE, SOURCE_EVIDENCE_SCHEMA, SOURCE_PLAN_SCHEMA, ReplaySpec,
    evaluate_evidence, require_current_evidence, score_bounds, validate_evidence, verify_candidate,
)
from cave_mem_v3.agents.runtime import run_episode
from test_candidates import repair, source_trace


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.trace = source_trace()
        self.candidate = repair(self.trace)
        self.study = StudySpec(max_steps=4, max_model_calls=4)
        self.spec = ReplaySpec(replay_pairs=2)

    def evidence(self, **kwargs):
        return verify_candidate(self.candidate, self.trace,
                                lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
                                study=self.study, replay_spec=self.spec, **kwargs)

    def mixed_correction_outcomes(self):
        """Inject one valid but unsuccessful continuation into a fixture pair."""
        corrected_count = 0

        class EarlyFinishActor(ScriptedActor):
            def propose(self, *args, **kwargs):
                response = super().propose(*args, **kwargs)
                return replace(response, action={"type": "finish"}, text='{"type":"finish"}')

        def runner(env, actor, **kwargs):
            nonlocal corrected_count
            if kwargs["first_action"] == {"type": "prepare"}:
                corrected_count += 1
                if corrected_count == 2:
                    actor = EarlyFinishActor()
            return run_episode(env, actor, **kwargs)

        return self.evidence(episode_runner=runner)

    def legacy_v5_fixture(self, evidence):
        """Encode the same fixture observations in the historical v5 format.

        Its expected decision is the old mean-only rule, including cases the
        new rule rejects. This does not rewrite any saved experimental file.
        """
        old = copy.deepcopy(evidence)
        plan = old["plan"]
        plan["schema_version"] = "source-replay-plan.v5"
        plan.pop("decision_rule")
        plan["plan_hash"] = digest({k: v for k, v in plan.items() if k != "plan_hash"})
        for row in old["pairs"]:
            row["schema_version"] = "source-replay-pair.v5"
            row["plan_hash"] = plan["plan_hash"]
            row["sample_id"] = digest([plan["plan_hash"], row["replicate"], row["seed"]])
            row["record_hash"] = digest({k: v for k, v in row.items() if k != "record_hash"})
        old["schema_version"] = "source-evidence.v5"
        old["decision_rule"] = "complete_valid_pairs_and_mean_difference_gt_acceptance_epsilon"
        old.pop("n_successful_corrections")
        old.pop("all_corrections_successful")
        old["v"] = 1
        old["reason"] = "supported_source_repair"
        old["evidence_hash"] = digest({k: v for k, v in old.items() if k != "evidence_hash"})
        return old

    def test_fixed_k_replay_and_bank_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = self.evidence(output_dir=Path(temporary) / "replays")
            self.assertEqual(evidence["v"], 1)
            self.assertEqual(evidence["schema_version"], SOURCE_EVIDENCE_SCHEMA)
            self.assertEqual(evidence["plan"]["schema_version"], SOURCE_PLAN_SCHEMA)
            self.assertEqual(evidence["decision_rule"], SOURCE_DECISION_RULE)
            self.assertTrue(evidence["all_corrections_successful"])
            self.assertEqual(evidence["n_successful_corrections"], 2)
            self.assertEqual(require_current_evidence(evidence, replay_pairs=2)["v"], 1)
            self.assertEqual(evidence["mean_difference"], 0.98)
            self.assertNotIn("lower_bound", evidence)
            self.assertEqual(evidence["independent_source_tasks"], 1)
            self.assertEqual(len(evidence["pairs"]), self.spec.replay_pairs)
            self.assertEqual(evidence["n_replay_pairs"], self.spec.replay_pairs)
            memory = build_memory(self.candidate, self.trace, evidence)
            self.assertEqual(memory.quality["mean_difference"], 0.98)
            self.assertEqual(memory.quality["decision_rule"], SOURCE_DECISION_RULE)
            path = Path(temporary) / "bank.json"
            bank = freeze_bank([memory], path, source_traces=[self.trace])
            self.assertEqual(load_bank(path), bank)
            with self.assertRaises(FileExistsError):
                freeze_bank([memory], path, source_traces=[self.trace])
            with self.assertRaises(FileExistsError):
                self.evidence(output_dir=Path(temporary) / "replays")

    def test_positive_mean_does_not_replace_two_successful_corrections(self):
        evidence = self.mixed_correction_outcomes()
        self.assertTrue(all(pair["valid"] for pair in evidence["pairs"]))
        self.assertGreater(evidence["mean_difference"], 0)
        self.assertEqual(evidence["n_successful_corrections"], 1)
        self.assertIs(evidence["all_corrections_successful"], False)
        self.assertEqual(evidence["v"], 0)
        self.assertEqual(evidence["reason"], "correction_not_consistently_successful")
        with self.assertRaises(ValueError):
            build_memory(self.candidate, self.trace, evidence)
        with self.assertRaises(ValueError):
            require_current_evidence(evidence, replay_pairs=2)

    def test_legacy_positive_mean_evidence_stays_valid_under_its_original_rule(self):
        legacy = self.legacy_v5_fixture(self.mixed_correction_outcomes())
        before = copy.deepcopy(legacy)
        self.assertEqual(validate_evidence(legacy, trace=self.trace)["v"], 1)
        self.assertEqual(legacy, before)
        self.assertEqual(evaluate_evidence(legacy["plan"], legacy["pairs"]), legacy)
        memory = build_memory(self.candidate, self.trace, legacy)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy-bank.json"
            bank = freeze_bank([memory], path, source_traces=[self.trace])
            self.assertEqual(load_bank(path), bank)
        with self.assertRaisesRegex(ValueError, "current"):
            require_current_evidence(legacy, replay_pairs=2)

    def test_new_plan_freezes_rule_and_rejects_mixed_schema_evidence(self):
        evidence = self.evidence()
        changed = copy.deepcopy(evidence["plan"])
        changed["decision_rule"] = "complete_valid_pairs_and_mean_difference_gt_acceptance_epsilon"
        changed["plan_hash"] = digest({k: v for k, v in changed.items() if k != "plan_hash"})
        with self.assertRaisesRegex(ValueError, "decision rule"):
            evaluate_evidence(changed, [])
        evidence["schema_version"] = "source-evidence.v5"
        with self.assertRaisesRegex(ValueError, "schema versions differ"):
            validate_evidence(evidence)

    def test_new_rule_preserves_general_k_and_pilot_checks_its_own_count(self):
        evidence = verify_candidate(self.candidate, self.trace,
            lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
            study=self.study, replay_spec=ReplaySpec(replay_pairs=3))
        self.assertEqual(evidence["n_replay_pairs"], 3)
        self.assertEqual(evidence["n_successful_corrections"], 3)
        self.assertEqual(require_current_evidence(evidence)["v"], 1)
        with self.assertRaisesRegex(ValueError, "count differs"):
            require_current_evidence(evidence, replay_pairs=2)

    def test_replay_labels_and_report_tampering_detected(self):
        evidence = self.evidence()
        bad = copy.deepcopy(evidence)
        bad["mean_difference"] = 100
        with self.assertRaises(ValueError):
            validate_evidence(bad, trace=self.trace)
        bad = copy.deepcopy(evidence)
        arm = bad["pairs"][0]["arms"]["corrected"]
        arm["utility"] = 0.1
        row = bad["pairs"][0]
        row["record_hash"] = digest({k: v for k, v in row.items() if k != "record_hash"})
        with self.assertRaises(ValueError):
            evaluate_evidence(bad["plan"], bad["pairs"])

    def test_missing_or_duplicate_replay_pair_never_passes(self):
        evidence = self.evidence()
        incomplete = evaluate_evidence(evidence["plan"], evidence["pairs"][:-1])
        self.assertEqual(incomplete["v"], "?")
        duplicated = copy.deepcopy(evidence["pairs"])
        duplicated[1] = duplicated[0]
        with self.assertRaises(ValueError):
            evaluate_evidence(evidence["plan"], duplicated)
        with self.assertRaises(ValueError):
            build_memory(self.candidate, self.trace, incomplete)

    def test_mean_must_exceed_registered_threshold(self):
        spec = ReplaySpec(replay_pairs=2, acceptance_epsilon=0.98)
        evidence = verify_candidate(self.candidate, self.trace,
                                    lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
                                    study=self.study, replay_spec=spec)
        self.assertEqual(evidence["mean_difference"], 0.98)
        self.assertEqual(evidence["v"], 0)
        with self.assertRaises(ValueError):
            build_memory(self.candidate, self.trace, evidence)

    def test_error_is_not_a_zero_reward_failure_or_dropped_pair(self):
        count = 0
        def runner(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 3:
                raise RuntimeError("simulated provider outage")
            return run_episode(*args, **kwargs)
        evidence = self.evidence(episode_runner=runner)
        self.assertEqual(len(evidence["pairs"]), self.spec.replay_pairs)
        self.assertEqual(evidence["v"], "?")
        self.assertEqual(evidence["reason"], "replay_contains_error_or_invalid_repair")

    def test_scored_invalid_correction_is_not_a_valid_repair(self):
        def runner(*args, **kwargs):
            arm = run_episode(*args, **kwargs)
            if kwargs.get("first_action") == {"type": "prepare"}:
                arm["termination_reason"] = "model_output_truncated"
            return arm
        evidence = self.evidence(episode_runner=runner)
        self.assertTrue(all(not row["valid"] for row in evidence["pairs"]))
        self.assertEqual(evidence["v"], "?")
        self.assertEqual(evidence["reason"], "replay_contains_error_or_invalid_repair")

    def test_repaired_or_substituted_first_action_cannot_confirm_candidate(self):
        def runner(env, actor, **kwargs):
            if kwargs["first_action"] == {"type": "prepare"}:
                kwargs["first_action"] = {"type": "operate"}
            return run_episode(env, actor, **kwargs)

        evidence = self.evidence(episode_runner=runner)
        self.assertEqual(len(evidence["pairs"]), 2)
        self.assertTrue(all(not row["valid"] for row in evidence["pairs"]))
        self.assertEqual(evidence["v"], "?")
        self.assertEqual(evidence["reason"], "replay_contains_error_or_invalid_repair")
        self.assertEqual(validate_evidence(evidence, trace=self.trace)["v"], "?")

    def test_changed_actor_and_uncertified_restoration_fail_closed(self):
        evidence = verify_candidate(self.candidate, self.trace,
            lambda: PreparationEnvironment(task_id="source-a"), lambda: ScriptedActor(error_rate=0.1),
            study=self.study, replay_spec=self.spec)
        self.assertEqual(evidence["v"], "?")
        self.assertEqual(len(evidence["pairs"]), self.spec.replay_pairs)
        self.assertEqual(evidence["reason"], "replay_contains_error_or_invalid_repair")
        trace = copy.deepcopy(self.trace)
        trace["episode"]["capabilities"]["environment"]["restore_certified"] = False
        trace["trace_hash"] = digest({k: v for k, v in trace.items() if k != "trace_hash"})
        with self.assertRaises(ValueError):
            verify_candidate(repair(trace), trace, lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
                             study=self.study, replay_spec=self.spec)

    def test_private_source_hash_and_memory_claim_rechecked_on_load(self):
        evidence = self.evidence()
        memory = build_memory(self.candidate, self.trace, evidence)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bank.json"
            bank = freeze_bank([memory], path, source_traces=[self.trace])
            bank["audit_sources"][self.trace["trace_id"]]["episode"]["steps"][0]["action"] = {"type": "finish"}
            bank["bank_hash"] = digest({k: v for k, v in bank.items() if k != "bank_hash"})
            write_json(path, bank)
            with self.assertRaises(ValueError):
                load_bank(path)

    def test_unverified_hypotheses_are_separate(self):
        memory = build_memory(self.candidate, self.trace, require_verified=False)
        self.assertEqual(validate_memory(memory, require_verified=False)["v"], "?")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                freeze_bank([memory], Path(temporary) / "bank.json", source_traces=[self.trace])

    def test_cost_bounds_are_registered_not_observed(self):
        with self.assertRaises(ValueError):
            score_bounds(StudySpec(inference_weights={"api_cost": 1}), ReplaySpec(replay_pairs=1))
        lower, upper = score_bounds(StudySpec(max_steps=4, max_model_calls=4), self.spec)
        self.assertEqual((lower, upper), (-0.04, 1.0))

    def test_actual_model_or_fingerprint_drift_rejects_replay(self):
        for field, value in (("model", "different-served-model"), ("fingerprint", "different-build")):
            with self.subTest(field=field):
                count = 0
                class DriftActor(ScriptedActor):
                    def propose(actor_self, *args, **kwargs):
                        return replace(super(DriftActor, actor_self).propose(*args, **kwargs), **{field: value})
                def factory():
                    nonlocal count
                    count += 1
                    return ScriptedActor() if count <= 2 else DriftActor()
                evidence = verify_candidate(self.candidate, self.trace,
                    lambda: PreparationEnvironment(task_id="source-a"), factory,
                    study=self.study, replay_spec=self.spec)
                self.assertEqual(evidence["v"], "?")
                self.assertEqual(len(evidence["pairs"]), self.spec.replay_pairs)

    def test_missing_fingerprint_is_unknown_not_certified(self):
        trace = copy.deepcopy(self.trace)
        for step in trace["episode"]["steps"]:
            if step.get("actor_response"):
                step["actor_response"]["fingerprint"] = None
        trace["episode"]["response_identity"] = {"model": "scripted-preparation.v3", "fingerprint": None,
                                                   "fingerprint_status": "unknown"}
        trace["trace_hash"] = digest({k: v for k, v in trace.items() if k != "trace_hash"})
        evidence = verify_candidate(repair(trace), trace,
            lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
            study=self.study, replay_spec=self.spec)
        self.assertEqual(evidence["v"], 1)
        self.assertEqual(evidence["response_identity_audit"]["fingerprint_status"], "unknown")

    def test_response_identity_is_recomputed_when_reading_evidence(self):
        evidence = self.evidence()
        row = evidence["pairs"][0]
        row["arms"]["corrected"]["steps"][1]["actor_response"]["fingerprint"] = "changed"
        row["record_hash"] = digest({k: v for k, v in row.items() if k != "record_hash"})
        with self.assertRaises(ValueError):
            evaluate_evidence(evidence["plan"], evidence["pairs"])


if __name__ == "__main__":
    unittest.main()
