import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from cave_mem_v3.agents.providers import OpenAIConfig, OpenAICompatibleProvider, ProviderActor, ScriptedActor
from cave_mem_v3.common.contracts import StudySpec, canonical, digest
from cave_mem_v3.environments.base import EnvStep, PreparationEnvironment
from cave_mem_v3.interventions.execution import TargetPendingError, run_target_arm
from cave_mem_v3.interventions.reminder import render_reminder
from cave_mem_v3.memory_generation.bank import build_memory
from cave_mem_v3.memory_generation.replay import ReplaySpec, verify_candidate
from test_candidates import repair, source_trace


class GradedFixture(PreparationEnvironment):
    """A test-only finalizer makes its scoring rule explicit at forced stops."""
    def finalize(self):
        state = self._require()
        if not state["terminal"]:
            state["terminal"] = True
            state["success"] = state["phase"] == "operated"
        self.verifier_report = {"success": state["success"], "fixture_only": True}
        return EnvStep(self.observe(), True, state["success"])


def provider_actor(texts, *, fingerprint="fixed", usage=None):
    payloads = []
    iterator = iter(texts)
    def transport(payload):
        payloads.append(payload)
        return {"model": "Qwen/Qwen2.5-7B-Instruct", "system_fingerprint": fingerprint,
                "choices": [{"message": {"content": next(iterator)}, "finish_reason": "stop"}],
                "usage": usage or {"prompt_tokens": 7, "completion_tokens": 2}}
    return ProviderActor(OpenAICompatibleProvider(OpenAIConfig(revision="fixed"), transport=transport)), payloads


class TargetExecutionTests(unittest.TestCase):
    def setup_checkpoint(self, cls=PreparationEnvironment, **kwargs):
        env = cls(**kwargs)
        context = env.reset(7)
        return context, env.snapshot()

    def run_arm(self, arm, actor=None, *, cls=PreparationEnvironment, context=None,
                checkpoint=None, action=None, remaining_model_calls=3, remaining_steps=4,
                journal=None, **kwargs):
        if context is None:
            context, checkpoint = self.setup_checkpoint(cls)
        return run_target_arm(cls(), actor or ScriptedActor(), context=context, checkpoint=checkpoint,
            first_action=action or {"type": "operate"}, arm=arm,
            reminder="Remember to prepare the resource." if arm == "remind" else None,
            spec=StudySpec(max_steps=4, max_model_calls=4), seed=17,
            remaining_model_calls=remaining_model_calls, remaining_steps=remaining_steps,
            journal=journal, **kwargs)

    def test_silent_executes_cached_action_without_a_model_call(self):
        class NeverCall(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                raise AssertionError("Silent resampled its cached first action")
        result = self.run_arm("silent", NeverCall())
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["success"])
        self.assertEqual(result["costs"]["model_calls"], 0)
        self.assertEqual(result["steps"][0]["action"], {"type": "operate"})
        self.assertEqual(result["steps"][0]["action_origin"], "cached_a0")

    def test_remind_redrafts_once_and_does_not_persist_advice(self):
        actor, payloads = provider_actor(['{"type":"prepare"}', '{"type":"operate"}', '{"type":"finish"}'])
        before = copy.deepcopy(actor.configuration())
        result = self.run_arm("remind", actor)
        self.assertTrue(result["success"])
        self.assertEqual(result["costs"]["model_calls"], 3)
        self.assertEqual(result["cost_breakdown"]["intervention"]["model_calls"], 1)
        self.assertEqual(actor.configuration(), before)
        self.assertIn("Remember to prepare", str(payloads[0]))
        self.assertNotIn("Remember to prepare", str(payloads[1:]))
        self.assertEqual([s["action"]["type"] for s in result["steps"]], ["prepare", "operate", "finish"])

    def test_continuation_seed_stream_is_matched_after_redraft(self):
        events = {"silent": [], "remind": []}
        context, checkpoint = self.setup_checkpoint(GradedFixture)
        for arm in events:
            self.run_arm(arm, cls=GradedFixture, context=context, checkpoint=checkpoint,
                         action={"type": "prepare"}, journal=events[arm].append)
        seeds = {arm: [e["seed"] for e in rows if e["kind"] == "actor.started" and e["stage"] == "target_continuation"]
                 for arm, rows in events.items()}
        self.assertEqual(seeds["silent"], seeds["remind"])
        self.assertEqual(len(seeds["silent"]), 2)

    def test_same_reserved_budget_and_zero_continuation_are_supported(self):
        for arm in ("silent", "remind"):
            result = self.run_arm(arm, cls=GradedFixture, action={"type": "prepare"},
                                  remaining_model_calls=1, remaining_steps=1)
            self.assertEqual(result["budget"]["continuation_model_calls"], 0)
            self.assertEqual(result["costs"]["model_calls"], 0 if arm == "silent" else 1)
            self.assertEqual(result["costs"]["tool_calls"], 1)
            self.assertTrue(result["outcome_valid"])

    def test_invalid_actions_are_scored_symmetrically_not_silently_excluded(self):
        malformed, _ = provider_actor(["not a JSON action"])
        silent = self.run_arm("silent", cls=GradedFixture, action={"type": "not_allowed"})
        remind = self.run_arm("remind", malformed, cls=GradedFixture)
        for result in (silent, remind):
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["termination_reason"], "invalid_actor_output")
            self.assertTrue(result["outcome_valid"])
            self.assertFalse(result["success"])
            self.assertEqual(result["costs"]["tool_calls"], 0)
        self.assertEqual(remind["actor_behavior"], "malformed_action")
        self.assertEqual(remind["costs"]["input_tokens"], 7)

    def test_no_finalizer_does_not_fabricate_a_terminal_failure(self):
        result = self.run_arm("silent", action={"type": "not_allowed"})
        self.assertEqual(result["termination_reason"], "finalization_unsupported")
        self.assertIsNone(result["success"])
        self.assertFalse(result["outcome_valid"])

    def test_weighted_unknown_cost_keeps_outcome_without_utility(self):
        context, checkpoint = self.setup_checkpoint(GradedFixture)
        actor, _ = provider_actor(['{"type":"finish"}'])
        result = run_target_arm(GradedFixture(), actor, context=context, checkpoint=checkpoint,
            first_action={"type": "operate"}, arm="remind", reminder="hint",
            spec=StudySpec(max_steps=4, max_model_calls=4, inference_weights={"api_cost": 1.0}),
            seed=1, remaining_model_calls=3, remaining_steps=4)
        self.assertTrue(result["outcome_valid"])
        self.assertFalse(result["utility_valid"])
        self.assertIsNone(result["costs"]["api_cost"])

    def test_drift_is_rejected_before_executing_the_revised_action(self):
        actor, _ = provider_actor(['{"type":"prepare"}'], fingerprint="changed")
        result = self.run_arm("remind", actor, expected_response_identity={
            "model": "Qwen/Qwen2.5-7B-Instruct", "fingerprint": "original"})
        self.assertEqual(result["error"]["scope"], "pair_shared")
        self.assertIsNone(result["success"])
        self.assertEqual(result["costs"]["model_calls"], 1)
        self.assertEqual(result["costs"]["tool_calls"], 0)

    def test_transport_timeout_remains_pending_and_is_never_retried(self):
        calls, events = [], []
        def transport(payload):
            calls.append(payload)
            raise TimeoutError("potentially billed")
        actor = ProviderActor(OpenAICompatibleProvider(OpenAIConfig(), transport=transport))
        with self.assertRaises(TargetPendingError) as raised:
            self.run_arm("remind", actor, journal=events.append)
        self.assertEqual(len(calls), 1)
        self.assertEqual(raised.exception.partial_record["costs"]["model_calls"], 1)
        self.assertEqual(events[-1]["kind"], "actor.pending")
        self.assertFalse(any(e["kind"] == "step.started" for e in events))

    def test_timeout_partial_record_preserves_settled_timing_and_prior_actions(self):
        clock, events = [0.0], []
        class TimedTimeout(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                clock[0] += 7.0
                raise TimeoutError("unknown completion")
        class TimedEnvironment(GradedFixture):
            def restore(self, checkpoint):
                clock[0] += 3.0
                return super().restore(checkpoint)
            def step(self, action, seed=0):
                clock[0] += 2.0
                return super().step(action, seed=seed)
        with patch("cave_mem_v3.interventions.execution.time.perf_counter", side_effect=lambda: clock[0]):
            with self.assertRaises(TargetPendingError) as raised:
                self.run_arm("silent", TimedTimeout(), cls=TimedEnvironment,
                             action={"type": "prepare"}, journal=events.append)
        partial = raised.exception.partial_record
        self.assertEqual(partial["costs"]["wall_seconds"], 9.0)
        self.assertEqual(partial["cost_breakdown"]["continuation"]["wall_seconds"], 9.0)
        self.assertEqual(partial["operational_costs"]["wall_seconds"], 3.0)
        self.assertEqual(partial["costs"]["tool_calls"], 1)
        self.assertEqual(len(partial["steps"]), 1)
        self.assertEqual(partial["termination_reason"], "request_completion_unknown")
        self.assertIsNone(partial["success"])
        self.assertEqual(events[-1]["kind"], "actor.pending")

    def test_journal_failure_stops_before_external_request(self):
        actor, payloads = provider_actor(['{"type":"finish"}'])
        def journal(event):
            if event["kind"] == "actor.started":
                raise OSError("disk full")
        with self.assertRaisesRegex(RuntimeError, "journal write failed"):
            self.run_arm("remind", actor, journal=journal)
        self.assertEqual(payloads, [])

    def test_keyboard_interrupt_preserves_started_without_completed(self):
        events = []
        class Interrupted(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            self.run_arm("remind", Interrupted(), journal=events.append)
        self.assertEqual(events[-1]["kind"], "actor.started")

    def test_restore_mismatch_blocks_without_actor_or_tools(self):
        class WrongRestore(PreparationEnvironment):
            def restore(self, checkpoint):
                result = super().restore(checkpoint)
                self._state["prepared"] = True
                return result
        result = self.run_arm("remind", cls=WrongRestore)
        self.assertEqual(result["termination_reason"], "restoration_mismatch")
        self.assertEqual(result["costs"]["model_calls"], 0)

    def test_nested_and_explicit_finalizers_are_operational_costs(self):
        for termination in ("finish", "environment_limit", "continuation_limit"):
            with self.subTest(termination=termination):
                clock = [0.0]
                class TimedFixture(GradedFixture):
                    def restore(self, checkpoint):
                        clock[0] += 3.0
                        return super().restore(checkpoint)
                    def step(self, action, seed=0):
                        clock[0] += 2.0
                        state = self._require()
                        state["steps"] += 1
                        state["phase"] = "operated"
                        if action["type"] == "finish" or state["steps"] == self.max_steps:
                            self.finalize()
                        return EnvStep(self.observe(), state["terminal"], state["success"])
                    def finalize(self):
                        clock[0] += 10.0
                        return super().finalize()
                env = TimedFixture(max_steps=1 if termination == "environment_limit" else 4)
                context = env.reset()
                checkpoint = env.snapshot()
                def journal(event):
                    clock[0] += 1.0
                with patch("cave_mem_v3.interventions.execution.time.perf_counter", side_effect=lambda: clock[0]):
                    result = run_target_arm(env, ScriptedActor(), context=context, checkpoint=checkpoint,
                        first_action={"type": "finish" if termination == "finish" else "prepare"},
                        arm="silent", reminder=None, spec=StudySpec(max_steps=4, max_model_calls=4,
                            inference_weights={"wall_seconds": 1.0}), seed=1,
                        remaining_model_calls=1, remaining_steps=1, journal=journal)
                self.assertTrue(result["success"])
                self.assertEqual(result["costs"]["wall_seconds"], 2.0)
                self.assertEqual(result["cost_breakdown"]["continuation"]["wall_seconds"], 2.0)
                self.assertEqual(result["operational_costs"]["finalization_wall_seconds"], 10.0)
                self.assertEqual(result["operational_costs"]["finalization_calls"], 1)
                self.assertEqual(result["utility"], 0.98)
                self.assertNotIn("finalize", vars(env))


class TargetReminderTests(unittest.TestCase):
    def memory(self):
        trace = source_trace()
        candidate = repair(trace)
        evidence = verify_candidate(candidate, trace,
            lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
            study=StudySpec(max_steps=4, max_model_calls=4), replay_spec=ReplaySpec(replay_pairs=2))
        return build_memory(candidate, trace, evidence)

    def test_rendering_is_bounded_deterministic_and_omits_audit_evidence(self):
        memory = self.memory()
        value = render_reminder(memory, {"type": "operate"})
        self.assertEqual(value, render_reminder(memory, {"type": "operate"}))
        self.assertEqual(value["text_hash"], digest(value["text"]))
        self.assertIn(memory.memory_id, value["text"])
        self.assertNotIn('"pairs"', value["text"])
        self.assertNotIn("preparation-checkpoint", value["text"])
        self.assertNotIn("private_verifier", value["text"])
        with self.assertRaisesRegex(ValueError, "character budget"):
            render_reminder(memory, {"type": "operate"}, max_chars=10)

    def test_source_entry_still_rejects_reminders_but_target_entry_is_local(self):
        actor = ScriptedActor()
        context = PreparationEnvironment().reset()
        with self.assertRaisesRegex(ValueError, "cannot receive reminders"):
            actor.propose(context, stage="source", seed=1, reminder="hint")
        hinted = actor.propose_target(context, stage="target", seed=1, reminder="hint")
        ordinary = actor.propose_target(context, stage="target", seed=1)
        self.assertEqual(hinted.action, {"type": "prepare"})
        self.assertEqual(ordinary.action, {"type": "operate"})
        self.assertNotIn("One-time advisory", canonical(ordinary.messages))


if __name__ == "__main__":
    unittest.main()
