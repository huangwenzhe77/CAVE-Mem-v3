import unittest

from cave_mem_v3.common.contracts import PublicContext
from cave_mem_v3.agents.providers import OpenAIConfig, OpenAICompatibleProvider, ProviderActor, ProviderError, render_messages
from cave_mem_v3.environments.base import EnvStep, PreparationEnvironment
from cave_mem_v3.agents.runtime import run_episode
from cave_mem_v3.common.contracts import StudySpec


def response(text='{"type":"prepare"}', usage=None, *, model="Qwen/Qwen2.5-7B-Instruct", fingerprint=None,
             finish_reason="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
            "model": model, "system_fingerprint": fingerprint,
            "usage": usage or {"prompt_tokens": 11, "completion_tokens": 3}}


class ProviderTests(unittest.TestCase):
    def test_bounded_history_keeps_recent_observation_and_records_cap(self):
        history = tuple({"action": {"type": "text", "command": f"action-{index}"},
                         "observation": f"seen-{index} " + "x" * 700}
                        for index in range(12))
        context = PublicContext(task_id="task", instruction="do task", observation="now",
                                history=history)
        bounded = render_messages(context, max_history_chars=1500)[-1]["content"]
        self.assertIn("seen-11", bounded)
        self.assertIn("omitted_early_actions", bounded)
        self.assertNotIn("seen-0", bounded)
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig()), max_history_chars=1500)
        self.assertEqual(actor.configuration()["max_history_chars"], 1500)
        self.assertNotIn("max_history_chars", ProviderActor(OpenAICompatibleProvider(OpenAIConfig())).configuration())

    def test_explicit_thinking_mode_is_frozen_in_request(self):
        requests = []
        provider = OpenAICompatibleProvider(OpenAIConfig(revision="fixed", thinking_mode="disabled"),
            transport=lambda payload: requests.append(payload) or response())
        provider.generate([{"role": "user", "content": "choose one action"}], stage="source", seed=3)
        self.assertEqual(requests[0]["thinking"], {"type": "disabled"})
        with self.assertRaisesRegex(ValueError, "thinking_mode"):
            OpenAIConfig(thinking_mode="unknown")
        low_requests = []
        low = OpenAICompatibleProvider(OpenAIConfig(revision="fixed", reasoning_effort="low"),
            transport=lambda payload: low_requests.append(payload) or response())
        low.generate([{"role": "user", "content": "choose one action"}], stage="source", seed=3)
        self.assertEqual(low_requests[0]["reasoning_effort"], "low")
        with self.assertRaisesRegex(ValueError, "only one"):
            OpenAIConfig(thinking_mode="disabled", reasoning_effort="low")

    def test_length_response_is_charged_and_graded_without_executing_action(self):
        class GradedEnvironment(PreparationEnvironment):
            def __init__(self):
                super().__init__()
                self.verifier_report = None
                self.finalize_calls = 0

            def configuration(self):
                return {**super().configuration(), "adapter": "terminal-bench-fs-boundary.v1"}

            def finalize(self):
                self.finalize_calls += 1
                self.verifier_report = {"success": False}
                return EnvStep(self.observe(), True, False)

        for text in ('{"type":"prepare"}', '', None):
            with self.subTest(text=text):
                env = GradedEnvironment()
                actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig(),
                    transport=lambda _: response(text, finish_reason="length")))
                trace = run_episode(env, actor, spec=StudySpec(), seed=0)
                self.assertEqual(trace["status"], "terminal")
                self.assertEqual(trace["termination_reason"], "model_output_truncated")
                self.assertEqual(trace["steps"][0]["actor_response"]["finish_reason"], "length")
                self.assertEqual(trace["costs"]["model_calls"], 1)
                self.assertEqual(trace["costs"]["tool_calls"], 0)
                self.assertEqual(env.finalize_calls, 1)
                self.assertFalse(trace["success"])

    def test_empty_configuration_never_downloads_or_connects(self):
        provider = OpenAICompatibleProvider(OpenAIConfig())
        self.assertEqual(provider.configuration()["model"], "Qwen/Qwen2.5-7B-Instruct")
        with self.assertRaisesRegex(ValueError, "not configured"):
            provider.generate([{"role": "user", "content": "test"}], stage="source", seed=0)

    def test_one_physical_call_preserves_usage_and_revision(self):
        calls = []
        provider = OpenAICompatibleProvider(OpenAIConfig(revision="fixed", send_seed=True),
            transport=lambda payload: calls.append(payload) or response())
        actor = ProviderActor(provider)
        result = actor.propose(PublicContext("task", "instruction", "observation"), stage="source", seed=17)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["seed"], 17)
        self.assertEqual(result.usage["input_tokens"], 11)
        self.assertIsNone(result.usage["api_cost"])
        self.assertEqual(actor.configuration()["provider"]["revision"], "fixed")
        self.assertFalse(actor.capabilities()["deterministic_sampling_certified"])

    def test_actor_json_mode_is_explicit_and_metered(self):
        calls = []
        provider = OpenAICompatibleProvider(OpenAIConfig(response_format="json_object"),
            transport=lambda payload: calls.append(payload) or response('{"type":"shell","command":"echo ok"}'))
        result = ProviderActor(provider).propose(PublicContext("task", "instruction", "observation"),
            stage="source", seed=1)
        self.assertEqual(calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(result.action["command"], "echo ok")
        self.assertEqual(result.usage["model_calls"], 1)
        self.assertEqual(provider.configuration()["response_format"], "json_object")

    def test_teacher_default_does_not_request_actor_json_mode(self):
        calls = []
        OpenAICompatibleProvider(OpenAIConfig(), transport=lambda payload: calls.append(payload) or response()).generate(
            [{"role": "user", "content": "test"}], stage="source_hypothesis", seed=1)
        self.assertNotIn("response_format", calls[0])
        with self.assertRaisesRegex(ValueError, "response_format"):
            OpenAIConfig(response_format="unsupported")

    def test_malformed_completed_response_remains_a_metered_failure(self):
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig(), transport=lambda _: response("not json")))
        trace = run_episode(PreparationEnvironment(), actor, spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "invalid_action")
        self.assertFalse(trace["success"])
        self.assertEqual(trace["costs"]["model_calls"], 1)
        self.assertEqual(trace["costs"]["input_tokens"], 11)

    def test_transport_failure_is_not_silently_retried(self):
        calls = []
        def failing(payload):
            calls.append(payload)
            raise TimeoutError("secret response body must not be in public errors")
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig(), transport=failing))
        trace = run_episode(PreparationEnvironment(), actor, spec=StudySpec(), seed=0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])
        self.assertEqual(trace["costs"]["physical_attempts"], 1)
        self.assertIsNone(trace["costs"]["input_tokens"])
        self.assertNotIn("secret response", str(trace["error"]))

    def test_phase_one_actor_rejects_reminder(self):
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig(), transport=lambda _: response()))
        with self.assertRaisesRegex(ValueError, "cannot receive reminders"):
            actor.propose(PublicContext("task", "instruction", "obs"), stage="source", seed=0, reminder="hint")

    def test_response_model_mismatch_is_charged_but_not_an_outcome(self):
        provider = OpenAICompatibleProvider(OpenAIConfig(), transport=lambda _: response(model="different-model"))
        trace = run_episode(PreparationEnvironment(), ProviderActor(provider), spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])
        self.assertEqual(trace["costs"]["model_calls"], 1)
        self.assertEqual(trace["costs"]["input_tokens"], 11)
        self.assertEqual(trace["costs"]["tool_calls"], 0)
        self.assertEqual(trace["failed_attempt"]["identity_error"], "response_model_mismatch")

    def test_response_model_alias_must_be_registered(self):
        provider = OpenAICompatibleProvider(OpenAIConfig(response_model="served-alias"),
            transport=lambda _: response(model="served-alias"))
        result = provider.generate([{"role": "user", "content": "go"}], stage="source", seed=0)
        self.assertEqual(result.model, "served-alias")
        self.assertEqual(provider.configuration()["response_model"], "served-alias")

    def test_model_field_cannot_be_silently_filled_from_request(self):
        raw = response()
        raw.pop("model")
        provider = OpenAICompatibleProvider(OpenAIConfig(), transport=lambda _: raw)
        with self.assertRaises(ProviderError) as caught:
            provider.generate([{"role": "user", "content": "go"}], stage="source", seed=0)
        self.assertEqual(caught.exception.attempt["identity_error"], "response_model_missing")
        self.assertEqual(caught.exception.attempt["usage"]["input_tokens"], 11)

    def test_registered_fingerprint_must_match_and_usage_is_retained(self):
        for fingerprint in (None, "changed"):
            with self.subTest(fingerprint=fingerprint):
                provider = OpenAICompatibleProvider(OpenAIConfig(expected_fingerprint="fixed"),
                    transport=lambda _: response(fingerprint=fingerprint))
                with self.assertRaises(ProviderError) as caught:
                    provider.generate([{"role": "user", "content": "go"}], stage="source", seed=0)
                self.assertEqual(caught.exception.attempt["identity_error"], "response_fingerprint_mismatch")
                self.assertEqual(caught.exception.attempt["usage"]["output_tokens"], 3)

    def test_missing_fingerprint_is_unknown_not_weight_attestation(self):
        provider = OpenAICompatibleProvider(OpenAIConfig(revision="operator-declared"),
            transport=lambda _: response('{"type":"operate"}', fingerprint=None))
        trace = run_episode(PreparationEnvironment(), ProviderActor(provider), spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "terminal")
        self.assertEqual(trace["response_identity"]["fingerprint_status"], "unknown")
        self.assertIsNone(trace["response_identity"]["fingerprint"])
        self.assertFalse(provider.capabilities()["weight_revision_certified"])

    def test_nonempty_fingerprint_change_invalidates_episode_after_charging(self):
        calls = []
        def transport(_):
            calls.append(1)
            return response('{"type":"prepare"}' if len(calls) == 1 else '{"type":"operate"}',
                            fingerprint="first" if len(calls) == 1 else "changed")
        provider = OpenAICompatibleProvider(OpenAIConfig(), transport=transport)
        trace = run_episode(PreparationEnvironment(safe_intent=True), ProviderActor(provider), spec=StudySpec(), seed=0)
        self.assertEqual(trace["status"], "error")
        self.assertIsNone(trace["success"])
        self.assertEqual(trace["costs"]["model_calls"], 2)
        self.assertEqual(trace["costs"]["tool_calls"], 1)
        self.assertEqual(trace["response_identity"]["fingerprint_status"], "mismatch")
        self.assertEqual(trace["steps"][1]["actor_response"]["fingerprint"], "changed")


if __name__ == "__main__":
    unittest.main()
