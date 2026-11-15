import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cave_mem_v3.common.contracts import StudySpec, digest, read_json, write_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.memory_generation.bank import build_memory, freeze_bank, load_bank, validate_memory
from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.memory_generation.replay import ReplaySpec, evaluate_evidence, score_bounds, validate_evidence, verify_candidate
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

    def test_fixed_k_replay_and_bank_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            evidence = self.evidence(output_dir=Path(temporary) / "replays")
            self.assertEqual(evidence["v"], 1)
            self.assertEqual(evidence["mean_difference"], 0.98)
            self.assertNotIn("lower_bound", evidence)
            self.assertEqual(evidence["independent_source_tasks"], 1)
            self.assertEqual(len(evidence["pairs"]), self.spec.replay_pairs)
            self.assertEqual(evidence["n_replay_pairs"], self.spec.replay_pairs)
            memory = build_memory(self.candidate, self.trace, evidence)
            self.assertEqual(memory.quality["mean_difference"], 0.98)
            path = Path(temporary) / "bank.json"
            bank = freeze_bank([memory], path, source_traces=[self.trace])
            self.assertEqual(load_bank(path), bank)
            with self.assertRaises(FileExistsError):
                freeze_bank([memory], path, source_traces=[self.trace])
            with self.assertRaises(FileExistsError):
                self.evidence(output_dir=Path(temporary) / "replays")

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
