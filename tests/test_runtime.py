import copy
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace

from cave_mem_v3.common.contracts import StudySpec, digest
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.agents.runtime import run_episode


class RuntimeTests(unittest.TestCase):
    def test_source_repair_replays_the_same_checkpoint(self):
        env = PreparationEnvironment()
        spec = StudySpec(max_steps=4)
        failure = run_episode(env, ScriptedActor(), spec=spec, seed=7)
        self.assertFalse(failure["success"])
        checkpoint = failure["steps"][0]["checkpoint"]
        original = run_episode(PreparationEnvironment(), ScriptedActor(), spec=spec,
            seed=9, checkpoint=checkpoint, first_action={"type": "operate"})
        corrected = run_episode(PreparationEnvironment(), ScriptedActor(), spec=spec,
            seed=9, checkpoint=checkpoint, first_action={"type": "prepare"})
        self.assertEqual(original["initial_checkpoint_hash"], corrected["initial_checkpoint_hash"])
        self.assertFalse(original["success"])
        self.assertTrue(corrected["success"])
        self.assertEqual(corrected["costs"]["model_calls"], 2)
        self.assertEqual(corrected["costs"]["tool_calls"], 3)
        self.assertIsNone(corrected["steps"][0]["actor_response"])

    def test_checkpoint_captures_private_state_but_actor_only_gets_public_context(self):
        env = PreparationEnvironment(safe_intent=True)
        trace = run_episode(env, ScriptedActor(), spec=StudySpec(max_steps=4), seed=5)
        messages = str(trace["steps"][0]["actor_response"]["messages"])
        self.assertNotIn("preparation-checkpoint.v3", messages)
        self.assertTrue(trace["success"])

    def test_invalid_actor_action_is_failure_not_infrastructure_error(self):
        trace = run_episode(PreparationEnvironment(), ScriptedActor(error_rate=1),
                            spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "invalid_action")
        self.assertIs(trace["success"], False)
        self.assertEqual(trace["costs"]["tool_calls"], 0)
        self.assertEqual(trace["costs"]["model_calls"], 1)

    def test_reset_failure_keeps_identity_and_unknown_outcome(self):
        class Broken(PreparationEnvironment):
            def reset(self, seed=0):
                raise RuntimeError("backend unavailable")
        trace = run_episode(Broken(), ScriptedActor(), spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])
        self.assertIsNone(trace["utility"])
        self.assertIn("environment_config", trace)
        self.assertIn("actor_config", trace)

    def test_collect_history_records_reset_errors_and_resume_does_not_retry(self):
        from cave_mem_v3.memory_generation.collection import CollectionConfig, collect_history
        from cave_mem_v3.common.contracts import read_json
        from cave_mem_v3.benchmarks.tasks import build_manifest
        calls = []
        class Broken(PreparationEnvironment):
            def reset(self, seed=0):
                calls.append(seed)
                raise RuntimeError("backend reset unavailable")
        manifest = build_manifest([{"task_id": "broken", "group_id": "broken-group", "role": "history"}],
                                  benchmark="fixture", revision="v1")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            config = CollectionConfig(runs_per_task=2, max_runs=2)
            factory = lambda task: Broken(task_id=task["task_id"])
            first = collect_history(manifest, config, StudySpec(), factory, ScriptedActor, output)
            self.assertEqual(first["infrastructure_errors"], 2)
            self.assertEqual(first["failed_runs"], 0)
            self.assertEqual(first["completed_runs"], 0)
            traces = list((output / "traces").glob("*.json"))
            self.assertEqual(len(traces), 2)
            episode = read_json(traces[0])["episode"]
            self.assertEqual(episode["status"], "error")
            self.assertIn("environment_config", episode)
            second = collect_history(manifest, config, StudySpec(), factory, ScriptedActor, output)
            self.assertEqual(second, first)
            self.assertEqual(len(calls), 2)

    def test_restore_error_keeps_frozen_identity_and_unknown_outcome(self):
        class BrokenRestore(PreparationEnvironment):
            def restore(self, checkpoint):
                raise RuntimeError("snapshot image unavailable")
        source = PreparationEnvironment()
        source.reset()
        trace = run_episode(BrokenRestore(), ScriptedActor(), spec=StudySpec(), seed=0,
                            checkpoint=source.snapshot(), first_action={"type": "prepare"})
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])
        self.assertIsNone(trace["utility"])
        self.assertIn("environment_config", trace)
        self.assertEqual(trace["costs"]["model_calls"], 0)

    def test_model_budget_cannot_be_extended_by_forced_action(self):
        env = PreparationEnvironment(max_steps=5)
        env.reset()
        trace = run_episode(env, ScriptedActor(), spec=StudySpec(max_steps=5, max_model_calls=1),
            seed=0, checkpoint=env.snapshot(), first_action={"type": "prepare"})
        self.assertEqual(trace["status"], "budget_exhausted")
        self.assertEqual(trace["costs"]["model_calls"], 1)
        self.assertEqual(trace["costs"]["tool_calls"], 2)

    def test_restore_cannot_change_task(self):
        env = PreparationEnvironment(task_id="source")
        env.reset()
        trace = run_episode(PreparationEnvironment(task_id="other"), ScriptedActor(),
            spec=StudySpec(), seed=0, checkpoint=env.snapshot(), first_action={"type": "prepare"})
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])

    def test_wall_time_is_charged_before_utility(self):
        spec = StudySpec(lambda_inference=1, inference_weights={"wall_seconds": 1})
        trace = run_episode(PreparationEnvironment(), ScriptedActor(), spec=spec, seed=0)
        self.assertGreater(trace["costs"]["wall_seconds"], 0)
        self.assertEqual(trace["utility"], -trace["costs"]["wall_seconds"])
        expected = copy.deepcopy(trace)
        checksum = expected.pop("trace_hash")
        self.assertEqual(checksum, digest(expected))

    def test_configuration_mutation_invalidates_rollout(self):
        class Mutating(ScriptedActor):
            def propose(self, *args, **kwargs):
                response = super().propose(*args, **kwargs)
                self.error_rate = 0.5
                return response
        trace = run_episode(PreparationEnvironment(), Mutating(), spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])

    def test_actual_response_model_drift_with_unchanged_configuration_is_error(self):
        class DriftingActor(ScriptedActor):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def propose(self, *args, **kwargs):
                self.calls += 1
                response = super().propose(*args, **kwargs)
                return replace(response, model="first" if self.calls == 1 else "changed")
        trace = run_episode(PreparationEnvironment(safe_intent=True), DriftingActor(), spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "error")
        self.assertIn("model changed", trace["error"]["message"])
        self.assertEqual(trace["costs"]["model_calls"], 2)
        self.assertEqual(trace["costs"]["tool_calls"], 1)

    def test_no_actor_call_has_unknown_response_identity(self):
        env = PreparationEnvironment()
        env.reset()
        trace = run_episode(env, ScriptedActor(), spec=StudySpec(), seed=0,
                            checkpoint=env.snapshot(), first_action={"type": "operate"})
        self.assertEqual(trace["response_identity"],
                         {"model": None, "fingerprint": None, "fingerprint_status": "unknown"})


if __name__ == "__main__":
    unittest.main()
