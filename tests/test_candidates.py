import copy
import json
import unittest
from types import SimpleNamespace

from cave_mem_v3.memory_generation.candidates import CandidateGenerationError, candidate_templates, create_candidate, finalize_candidate, generate_candidates, validate_candidate
from cave_mem_v3.common.contracts import StudySpec, digest
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.agents.runtime import run_episode


def source_trace(*, safe=False):
    env = PreparationEnvironment(task_id="source-a", safe_intent=safe)
    episode = run_episode(env, ScriptedActor(), spec=StudySpec(max_steps=4, max_model_calls=4), seed=42)
    trace = {"schema_version": "history-trace.v3", "trace_id": "source-trace-success" if safe else "source-trace-failure",
             "task_id": "source-a", "role": "history", "group_id": "source-family-a",
             "manifest_hash": "manifest", "run_hash": "run", "run_index": 0, "replicate": 0,
             "seed": 42, "episode": episode}
    trace["trace_hash"] = digest(trace)
    return trace


def repair(trace):
    return create_candidate(trace, 0, mechanism="Operating before preparation loses the resource",
                            correction={"type": "prepare"},
                            positive_conditions=[{"field": "requires_preparation", "equals": True}],
                            exclusion_conditions=[{"field": "prepared", "equals": True}],
                            scope="The recorded preparation task and source checkpoint", reviewer="test-reviewer")


class CandidateTests(unittest.TestCase):
    def test_template_requires_actual_repair_and_retains_real_action(self):
        trace = source_trace()
        templates = candidate_templates([trace, source_trace(safe=True)])
        self.assertEqual(len(templates), 1)
        self.assertEqual(templates[0]["problem_action"], {"type": "operate"})
        with self.assertRaises(ValueError):
            finalize_candidate(templates[0], trace)
        validate_candidate(repair(trace), trace)

    def test_provenance_and_hypothesis_cannot_change_after_sealing(self):
        trace = source_trace()
        candidate = repair(trace)
        for field, value in (("problem_action", {"type": "finish"}), ("mechanism", "different mechanism"),
                             ("source_task_id", "target-test"), ("checkpoint_hash", "other")):
            edited = copy.deepcopy(candidate)
            edited[field] = value
            with self.assertRaises(ValueError):
                validate_candidate(edited, trace)

    def test_infrastructure_error_and_test_role_cannot_create_memory(self):
        trace = source_trace()
        for update in (("status", "error"), ("success", None)):
            changed = copy.deepcopy(trace)
            changed["episode"][update[0]] = update[1]
            changed["trace_hash"] = digest({k: v for k, v in changed.items() if k != "trace_hash"})
            with self.assertRaises(ValueError):
                repair(changed)
        trace["role"] = "test"
        trace["trace_hash"] = digest({k: v for k, v in trace.items() if k != "trace_hash"})
        with self.assertRaises(ValueError):
            repair(trace)

    def test_model_prompt_omits_private_snapshots_and_verifier(self):
        trace = source_trace()
        trace["episode"]["private_verifier"] = {"secret": "DO_NOT_SHOW_VERIFIER"}
        trace["episode"]["steps"][0]["checkpoint"]["private_secret"] = "DO_NOT_SHOW_SNAPSHOT"
        trace["trace_hash"] = digest({k: v for k, v in trace.items() if k != "trace_hash"})
        proposal = {"step_index": 0, "mechanism": "Preparation is missing", "correction": {"type": "prepare"},
                    "positive_conditions": [{"field": "requires_preparation", "equals": True}],
                    "exclusion_conditions": [], "scope": "recorded checkpoint"}
        class Provider:
            def configuration(self):
                return {"provider": "test"}
            def generate(self, messages, **kwargs):
                content = json.dumps(messages)
                self_test.assertNotIn("DO_NOT_SHOW_VERIFIER", content)
                self_test.assertNotIn("DO_NOT_SHOW_SNAPSHOT", content)
                return SimpleNamespace(text=json.dumps({"candidates": [proposal]}), model="hypothesis-test",
                                       usage={"model_calls": 1}, attempt={"status": "completed"})
        self_test = self
        result = generate_candidates(trace, Provider())
        self.assertEqual(result["candidates"][0]["status"], "hypothesis")
        self.assertNotIn("v", result["candidates"][0])
        validate_candidate(result["candidates"][0], trace)

    def test_conditions_must_be_executable_and_failure_signature_is_recorded(self):
        trace = source_trace()
        candidate = repair(trace)
        self.assertIn("observation_before", candidate["failure_signature"])
        self.assertIn("operate", candidate["failure_signature"])
        draft = copy.deepcopy(candidate)
        draft["failure_signature"] = ""
        with self.assertRaises(ValueError):
            finalize_candidate(draft, trace)
        draft["failure_signature"] = "observed action failure"
        draft["positive_conditions"] = [{"observation": "applies here"}]
        with self.assertRaises(ValueError):
            finalize_candidate(draft, trace)

    def test_malformed_billed_response_preserves_usage_without_retry(self):
        class Provider:
            calls = 0
            def configuration(self):
                return {"provider": "test"}
            def generate(self, messages, **kwargs):
                self.calls += 1
                return SimpleNamespace(text="not json", model="test-model", usage={"model_calls": 1, "input_tokens": 23},
                                       attempt={"status": "completed", "usage": {"input_tokens": 23}})
        provider = Provider()
        with self.assertRaises(CandidateGenerationError) as caught:
            generate_candidates(source_trace(), provider)
        error = caught.exception
        self.assertEqual(provider.calls, 1)
        self.assertEqual(error.generation["usage"]["input_tokens"], 23)
        self.assertEqual(error.generation["failure_stage"], "parse")
        self.assertEqual(error.generation["status"], "error")
        self.assertTrue(error.generation["response_hash"])
        self.assertEqual(error.candidates, [])

    def test_partial_candidates_and_configuration_errors_preserve_metering(self):
        good = {"step_index": 0, "mechanism": "missing preparation", "correction": {"type": "prepare"},
                "positive_conditions": [{"field": "prepared", "equals": False}], "scope": "source only"}
        class Provider:
            changed = False
            drift = False
            def configuration(self):
                return {"provider": "changed" if self.changed else "fixed"}
            def generate(self, messages, **kwargs):
                if self.drift:
                    self.changed = True
                return SimpleNamespace(text=json.dumps({"candidates": [good, {"step_index": 999}]}),
                    model="test-model", usage={"model_calls": 1, "output_tokens": 29}, attempt={"status": "completed"})
        with self.assertRaises(CandidateGenerationError) as caught:
            generate_candidates(source_trace(), Provider())
        self.assertEqual(len(caught.exception.candidates), 1)
        self.assertEqual(caught.exception.generation["failure_stage"], "candidate_validation")
        provider = Provider()
        provider.drift = True
        with self.assertRaises(CandidateGenerationError) as caught:
            generate_candidates(source_trace(), provider)
        self.assertEqual(caught.exception.generation["usage"]["output_tokens"], 29)
        self.assertEqual(caught.exception.generation["failure_stage"], "configuration")

    def test_provider_failure_retains_attempt_and_unknown_cost(self):
        class ProviderFailure(RuntimeError):
            attempt = {"status": "failed", "usage": {"model_calls": 1, "api_cost": None}}
        class Provider:
            def configuration(self):
                return {"provider": "test"}
            def generate(self, messages, **kwargs):
                raise ProviderFailure("SECRET_GATEWAY_RESPONSE")
        with self.assertRaises(CandidateGenerationError) as caught:
            generate_candidates(source_trace(), Provider())
        error = caught.exception
        self.assertEqual(error.attempt["status"], "failed")
        self.assertIsNone(error.generation["usage"]["api_cost"])
        self.assertNotIn("SECRET_GATEWAY_RESPONSE", str(error))


if __name__ == "__main__":
    unittest.main()
