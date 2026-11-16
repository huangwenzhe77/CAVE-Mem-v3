"""Live single-arm execution on explicitly non-cloneable software fixtures."""
import copy
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ProviderError, ScriptedActor
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, plain, read_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.interventions.execution import TargetPendingError, run_live_target_arm
from cave_mem_v3.interventions.single import _check_live_protocol, _prepare_live, collect_single_target
from cave_mem_v3.interventions.single_outcomes import summarize_target_single
from cave_mem_v3.target_data.sampling import RiskHazard
from test_target_execution import GradedFixture
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture


class NoncloneableFixture(GradedFixture):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resets = 0
        self.closed = 0
        self.actions = []

    def capabilities(self):
        return {**super().capabilities(), "restore_certified": False,
                "isolated_sessions": False, "randomized_single_supported": True}

    def reset(self, seed=0):
        self.resets += 1
        return super().reset(seed)

    def snapshot(self):
        raise AssertionError("single-arm execution requested a snapshot")

    def restore(self, checkpoint):
        raise AssertionError("single-arm execution restored an environment")

    def state_fingerprint(self):
        raise AssertionError("single-arm execution requested a clone fingerprint")

    def step(self, action, seed=0):
        self.actions.append(copy.deepcopy(action))
        return super().step(action, seed=seed)

    def close(self):
        self.closed += 1


class SingleExecutionTests(unittest.TestCase):
    def run_live(self, arm, actor=None, *, env=None, action=None, events=None, calls=3):
        env = env or NoncloneableFixture()
        context = env.reset()
        result = run_live_target_arm(env, actor or ScriptedActor(), context=context,
            first_action=action or {"type": "operate"}, arm=arm,
            reminder="Prepare if required." if arm == "remind" else None,
            spec=StudySpec(max_steps=4, max_model_calls=4), seed=17,
            remaining_model_calls=calls, remaining_steps=4,
            journal=events.append if events is not None else None)
        return env, result

    def test_silent_uses_same_live_session_and_cached_draft_without_cloning(self):
        class NeverCall(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                raise AssertionError("Silent must not regenerate the cached first action")
        env, record = self.run_live("silent", NeverCall())
        self.assertEqual(record["schema_version"], "target-live-arm.v1")
        self.assertEqual(env.resets, 1)
        self.assertEqual(env.actions, [{"type": "operate"}])
        self.assertEqual(record["costs"]["model_calls"], 0)
        self.assertFalse(record["success"])
        self.assertIsNone(record["initial_checkpoint_hash"])
        self.assertIsNone(record["restore_fingerprint"])

    def test_remind_is_once_and_later_calls_get_no_advice(self):
        seen = []
        class ObservedActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                seen.append(kwargs.get("reminder"))
                return super().propose_target(*args, **kwargs)
        events = []
        env, record = self.run_live("remind", ObservedActor(), events=events)
        self.assertTrue(record["success"])
        self.assertEqual(env.resets, 1)
        self.assertEqual(seen, ["Prepare if required.", None, None])
        self.assertEqual(record["costs"]["model_calls"], 3)
        self.assertEqual(events[0]["kind"], "session.checked")
        self.assertFalse(any(event["kind"].startswith("restore") for event in events))

    def test_live_public_context_drift_stops_without_actor_or_action(self):
        env = NoncloneableFixture()
        context = env.reset()
        env._state["prepared"] = True
        result = run_live_target_arm(env, ScriptedActor(), context=context,
            first_action={"type": "operate"}, arm="silent", reminder=None,
            spec=StudySpec(max_steps=4, max_model_calls=4), seed=1,
            remaining_model_calls=3, remaining_steps=4)
        self.assertEqual(result["termination_reason"], "live_context_changed")
        self.assertIsNone(result["success"])
        self.assertEqual(env.actions, [])

    def prepare(self, env, actor, *, probability=1e-9, encoder=None):
        encoder = encoder or EncoderFixture()
        events = []
        run = {"record_hash": "frozen", "role": "target_train", "study": plain(StudySpec(max_steps=4, max_model_calls=4)),
            "actor_config": actor.configuration(), "environment_configs": {"fixture": env.configuration()},
            "encoder_identity": copy.deepcopy(encoder.identity), "exposure_rules": None,
            "risk_hazard": plain(RiskHazard(low_risk_probability=probability, high_risk_probability=probability))}
        return _prepare_live(run, {"task_id": "fixture"}, "trace", 17, env, actor, encoder, events.append), events

    def test_invalid_unselected_prefix_is_graded_without_fake_failure(self):
        class AlreadySolved(NoncloneableFixture):
            def reset(self, seed=0):
                super().reset(seed)
                self._state["phase"] = "operated"
                return self.observe()
        record, events = self.prepare(AlreadySolved(), ScriptedActor(error_rate=1))
        self.assertEqual(record["status"], "no_selection")
        self.assertEqual(record["termination_reason"], "invalid_actor_output")
        self.assertTrue(record["unselected_terminal_outcome"])
        self.assertEqual(record["costs"]["tool_calls"], 0)
        self.assertEqual(record["operational_costs"]["finalization_calls"], 1)
        self.assertTrue(any(event["kind"] == "finalize.completed" for event in events))

    def test_sampling_failure_stops_before_actor(self):
        class NeverCall(ScriptedActor):
            def propose(self, *args, **kwargs):
                raise AssertionError("incomplete sampling called actor")
        with patch("cave_mem_v3.interventions.single.OnlineCheckpointSampler.consider", return_value={
                "status": "sampling_incomplete", "selected": False}):
            record, events = self.prepare(NoncloneableFixture(), NeverCall())
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["costs"]["model_calls"], 0)
        self.assertFalse(any(event["kind"] == "actor.started" for event in events))

    def test_malformed_selected_draft_is_not_assigned_and_is_graded(self):
        class Malformed(ScriptedActor):
            def propose(self, *args, **kwargs):
                return replace(super().propose(*args, **kwargs), parse_error="invalid_json")
        record, events = self.prepare(NoncloneableFixture(), Malformed(), probability=.99)
        self.assertEqual(record["status"], "preparation_error")
        self.assertEqual(record["termination_reason"], "invalid_actor_output")
        self.assertEqual(record["costs"]["model_calls"], 1)
        self.assertEqual(record["costs"]["tool_calls"], 0)
        self.assertNotIn("pre_treatment_state", record["steps"][-1])
        self.assertFalse(record["unselected_terminal_outcome"])

    def test_live_actor_timeout_remains_pending(self):
        calls = []
        class TimeoutActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                raise TimeoutError("completion unknown")
        env = NoncloneableFixture()
        with self.assertRaises(TargetPendingError):
            self.run_live("remind", TimeoutActor(), env=env)
        self.assertEqual(len(calls), 1)
        self.assertEqual(env.actions, [])


class SingleCollectionTests(unittest.TestCase):
    def fixture(self, root):
        return make_prepared_fixture(root / "source")

    def collect(self, root, fixture, *, probability=.99, actor_factory=ScriptedActor,
                envs=None, select=True, bank=None):
        envs = [] if envs is None else envs
        def env_factory(task):
            env = NoncloneableFixture(task_id=task["task_id"])
            envs.append(env)
            return env
        return collect_single_target(fixture["manifest"], bank=bank or fixture["bank"],
            role="target_train", spec=StudySpec(max_steps=4, max_model_calls=4),
            env_factory=env_factory, actor_factory=actor_factory, output_dir=root / "single",
            encoder=EncoderFixture(), assignment_probability=probability,
            hazard=RiskHazard(low_risk_probability=.99 if select else 1e-9,
                              high_risk_probability=.99 if select else 1e-9))

    def test_one_live_session_freezes_before_assignment_and_executes_only_selected_arm(self):
        for probability in (.99, .01):
            with self.subTest(probability=probability), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self.fixture(root)
                envs = []
                self.collect(root, fixture, probability=probability, envs=envs)
                session = next((root / "single" / "sessions").iterdir())
                item = read_json(session / "input.json")
                preparation = read_json(session / "preparation.json")
                assignment = read_json(session / "assignment.json")
                result = read_json(session / "result.json")["record"]
                active = [env for env in envs if env.resets]
                self.assertEqual(len(active), 1)
                self.assertEqual(active[0].resets, 1)
                self.assertEqual(active[0].closed, 1)
                self.assertEqual(item["a0"], {"type": "operate"})
                self.assertIsNone(preparation["steps"][-1]["result"])
                self.assertEqual(preparation["costs"]["model_calls"], 1)
                self.assertEqual(preparation["costs"]["tool_calls"], 0)
                self.assertEqual(item["pre_treatment_state"]["randomization_status"], "not_assigned")
                self.assertEqual(assignment["input_hash"], item["record_hash"])
                self.assertEqual(assignment["propensity"], assignment["b"][assignment["arm"]])
                self.assertEqual(result["arm"], assignment["arm"])
                self.assertEqual(assignment["arm"], "remind" if probability == .99 else "silent")
                self.assertEqual(len(active[0].actions), 3 if assignment["arm"] == "remind" else 1)
                self.assertFalse((session / "silent").exists() or (session / "remind").exists())

    def test_preparation_timeout_preserves_pending_without_assignment_or_retry(self):
        calls = []
        class TimeoutActor(ScriptedActor):
            def propose(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                raise TimeoutError("prefix completion unknown")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            self.collect(root, fixture, actor_factory=TimeoutActor)
            session = next((root / "single" / "sessions").iterdir())
            self.assertEqual(read_json(session / "pending.json")["phase"], "preparation")
            self.assertFalse((session / "assignment.json").exists())
            self.assertFalse((session / "result.json").exists())
            def prohibited():
                raise AssertionError("pending resume constructed an actor")
            with self.assertRaisesRegex(ValueError, "incomplete single-session"):
                self.collect(root, fixture, actor_factory=prohibited)
            self.assertEqual(len(calls), 1)

    def test_completed_resume_is_readonly_before_factories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            expected = self.collect(root, fixture)
            saved = {path.relative_to(root): path.read_bytes() for path in (root / "single").rglob("*.json")}
            def prohibited():
                raise AssertionError("completed resume constructed actor")
            self.assertEqual(expected, self.collect(root, fixture, actor_factory=prohibited))
            self.assertEqual(saved, {path.relative_to(root): path.read_bytes() for path in (root / "single").rglob("*.json")})

    def test_unrelated_backend_cannot_claim_fixture_protocol(self):
        class Impostor:
            def configuration(self):
                return {"adapter": "preparation-fixture.v3", "research_role": "fixture_only"}
            def capabilities(self):
                return {"randomized_single_supported": True}
        with self.assertRaisesRegex(ValueError, "adapter review required"):
            _check_live_protocol(Impostor(), ScriptedActor())

    def test_reusing_actor_factory_is_rejected_before_any_rollout(self):
        shared = ScriptedActor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            envs = []
            with self.assertRaisesRegex(ValueError, "reused an actor"):
                self.collect(root, fixture, actor_factory=lambda: shared, envs=envs)
            self.assertFalse(any(env.resets for env in envs))

    def test_known_fingerprint_survives_missing_next_draft_and_stops_future_tasks(self):
        drafts = []
        class DriftingActor(ScriptedActor):
            def propose(self, context, **kwargs):
                drafts.append(context.task_id)
                response = super().propose(context, **kwargs)
                return replace(response, fingerprint="A" if len(drafts) == 1 else None)
            def propose_target(self, context, **kwargs):
                response = super().propose_target(context, **kwargs)
                return replace(response, fingerprint="A" if len(drafts) == 1 else "B")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = build_manifest([
                {"task_id": "source-a", "group_id": "source-family-a", "role": "history"},
                *[{"task_id": "target-" + str(i), "group_id": "target-family-" + str(i), "role": "target_train"}
                  for i in range(3)]], benchmark="preparation-fixture", revision="fixture-v3")
            fixture = make_prepared_fixture(root / "source", manifest=manifest)
            def execute(actor_factory=DriftingActor, env_factory=None):
                return collect_single_target(manifest, bank=fixture["bank"], spec=StudySpec(max_steps=4, max_model_calls=4),
                    actor_factory=actor_factory, env_factory=env_factory or (lambda task: NoncloneableFixture(
                        task_id=task["task_id"], safe_intent=True)), encoder=EncoderFixture(),
                    hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
                    assignment_probability=.99, output_dir=root / "single")
            execute()
            sessions = sorted((root / "single" / "sessions").iterdir())
            self.assertEqual(len(sessions), 2)
            second = read_json(sessions[1] / "result.json")["record"]
            self.assertEqual(second["termination_reason"], "response_fingerprint_drift")
            self.assertIsNone(second["success"])
            self.assertEqual(len(drafts), 2)
            def prohibited(*args):
                raise AssertionError("shared-failure resume constructed a factory")
            with self.assertRaisesRegex(ValueError, "shared protocol failure"):
                execute(actor_factory=prohibited, env_factory=prohibited)

    def test_frozen_configuration_or_encoder_drift_stops_run_before_later_tasks(self):
        for drift in ("actor", "environment", "encoder", "provider_identity"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                calls, resets = [], []
                manifest = build_manifest([
                    {"task_id": "source-a", "group_id": "source-family-a", "role": "history"},
                    *[{"task_id": f"target-{i}", "group_id": f"target-family-{i}", "role": "target_train"}
                      for i in range(3)]], benchmark="preparation-fixture", revision="fixture-v3")
                fixture = make_prepared_fixture(root / "source", manifest=manifest)
                class DriftingActor(ScriptedActor):
                    def propose(self, context, **kwargs):
                        calls.append(context.task_id)
                        response = super().propose(context, **kwargs)
                        if drift == "actor":
                            self.error_rate = .25
                        if drift == "provider_identity":
                            raise ProviderError("registered identity differs", {
                                "status": "failed", "error_class": "ValueError",
                                "identity_error": "response_model_mismatch", "usage": response.usage})
                        return response
                class DriftingEnvironment(NoncloneableFixture):
                    def reset(self, seed=0):
                        resets.append(self.task_id)
                        context = super().reset(seed)
                        if drift == "environment":
                            self.safe_intent = True
                        return context
                class DriftingEncoder(EncoderFixture):
                    def encode_query(self, text):
                        vector = super().encode_query(text)
                        if drift == "encoder":
                            self.identity = {**self.identity, "revision": "changed"}
                        return vector
                def execute(actor_factory=DriftingActor, env_factory=None, encoder=None):
                    return collect_single_target(manifest, bank=fixture["bank"],
                        spec=StudySpec(max_steps=4, max_model_calls=4), actor_factory=actor_factory,
                        env_factory=env_factory or (lambda task: DriftingEnvironment(task_id=task["task_id"])),
                        encoder=encoder or DriftingEncoder(),
                        hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
                        assignment_probability=.5, output_dir=root / "single")
                summary = execute()
                self.assertEqual(summary["blocked_rollouts"], 1)
                self.assertEqual(summary["unattempted_rollouts"], 2)
                self.assertEqual(summary["valid_outcomes"], 0)
                self.assertEqual(len(resets), 1)
                self.assertEqual(len(calls), 0 if drift == "environment" else 1)
                session = next((root / "single" / "sessions").iterdir())
                preparation = read_json(session / "preparation.json")
                self.assertEqual(preparation["error"]["scope"], "run_shared")
                self.assertIn("provider response identity" if drift == "provider_identity" else "frozen",
                              preparation["error"]["message"])
                self.assertFalse((session / "assignment.json").exists())
                self.assertEqual(summary, summarize_target_single(root / "single"))
                artifacts = {p.relative_to(root): p.read_bytes() for p in (root / "single").rglob("*.json")}
                def prohibited(*args):
                    raise AssertionError("shared-drift resume constructed an execution instance")
                with self.assertRaisesRegex(ValueError, "shared protocol failure"):
                    execute(actor_factory=prohibited, env_factory=prohibited, encoder=EncoderFixture())
                self.assertEqual(artifacts, {p.relative_to(root): p.read_bytes() for p in (root / "single").rglob("*.json")})

    def test_no_selection_keeps_denominator_without_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            self.collect(root, fixture, select=False)
            session = next((root / "single" / "sessions").iterdir())
            item = read_json(session / "input.json")
            self.assertEqual((item["status"], item["reason"]), ("excluded", "no_selection"))
            self.assertFalse((session / "assignment.json").exists())

    def test_invalid_frozen_risk_rubric_preserves_summarizable_blocked_session(self):
        class NeverCall(ScriptedActor):
            def propose(self, *args, **kwargs):
                raise AssertionError("sampling failure invoked actor")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self.fixture(root)
            summary = collect_single_target(fixture["manifest"], bank=fixture["bank"],
                spec=StudySpec(max_steps=4, max_model_calls=4), encoder=EncoderFixture(),
                env_factory=lambda task: NoncloneableFixture(task_id=task["task_id"]),
                actor_factory=NeverCall, output_dir=root / "single", assignment_probability=.5,
                hazard=RiskHazard(risk_rubric="unsupported-frozen-rubric"))
            self.assertEqual(summary["blocked_rollouts"], 1)
            self.assertEqual(summary["valid_outcomes"], 0)
            self.assertEqual(summary, summarize_target_single(root / "single"))
            session = next((root / "single" / "sessions").iterdir())
            self.assertFalse((session / "assignment.json").exists())
            preparation = read_json(session / "preparation.json")
            self.assertEqual(preparation["selection"]["status"], "sampling_incomplete")
            self.assertEqual(preparation["costs"]["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
