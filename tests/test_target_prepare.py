import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.cli import main
from cave_mem_v3.common.contracts import StudySpec, digest, read_json, write_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.target_data.collection import collect_target
from cave_mem_v3.target_data.sampling import RiskHazard

ROOT = Path(__file__).resolve().parents[1]


class EncoderFixture:
    identity = {"kind": "fixture-encoder", "revision": "test"}

    def encode_query(self, text):
        return [1.0, float(len(text) % 7 + 1)]

    def encode_documents(self, texts):
        return [self.encode_query(text) for text in texts]


class CountingEnvironment(PreparationEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_calls = 0
        self.step_calls = 0

    def validate_action(self, action):
        self.validate_calls += 1
        return super().validate_action(action)

    def step(self, action, seed=0):
        self.step_calls += 1
        return super().step(action, seed=seed)


class MalformedDraftActor(ScriptedActor):
    def propose(self, *args, **kwargs):
        response = super().propose(*args, **kwargs)
        return replace(response, action={"type": "__unparsed_action__"}, parse_error="invalid_json")


class PrepareTargetTests(unittest.TestCase):
    def test_prepare_target_cli_freezes_selected_checkpoint_and_stops(self):
        class LocalEncoder(EncoderFixture):
            def __init__(self, path, *, revision, device):
                self.path, self.revision, self.device = path, revision, device
                self.identity = {**self.identity, "path": path, "revision": revision,
                                 "device": device}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({
                "benchmark": "preparation-fixture", "revision": "fixture-v3",
                "tasks": [{"task_id": "target-training", "group_id": "target-training",
                           "role": "target_train"}],
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
                    "memory_count": 0, "memories": [], "audit_sources": {}}
            bank["bank_hash"] = digest(bank)
            bank_path = root / "bank.json"
            write_json(bank_path, bank)
            encoder_path = root / "encoder"
            encoder_path.mkdir()
            output = root / "prepared"
            stdout = io.StringIO()
            with patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", LocalEncoder), \
                    patch("cave_mem_v3.memory_generation.bank.load_bank", return_value=bank), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["manifest", "--catalog", str(catalog),
                                       "--output", str(manifest)]), 0)
                stdout.seek(0)
                stdout.truncate(0)
                self.assertEqual(main([
                    "prepare-target", "--config", str(ROOT / "configs/phase1/demo.toml"),
                    "--manifest", str(manifest), "--output", str(output),
                    "--role", "target_train", "--bank", str(bank_path),
                    "--encoder-path", str(encoder_path), "--encoder-revision", "pinned-test",
                    "--low-risk-probability", ".99", "--high-risk-probability", ".99",
                ]), 0)

            summary = json.loads(stdout.getvalue())
            run = read_json(output / "run.json")
            trace = read_json(next((output / "traces").glob("*.json")))
            episode = trace["episode"]
            state = episode["steps"][0]["pre_treatment_state"]

        self.assertEqual(summary["pre_treatment_states_frozen"], 1)
        self.assertEqual(run["pre_treatment"]["bank_hash"], bank["bank_hash"])
        self.assertEqual(run["pre_treatment"]["encoder_identity"]["revision"], "pinned-test")
        self.assertEqual(trace["trace_hash"], digest({k: v for k, v in trace.items() if k != "trace_hash"}))
        self.assertEqual(episode["status"], "pre_treatment_ready")
        self.assertIsNone(episode["success"])
        self.assertIsNone(episode["utility"])
        self.assertEqual(state["randomization_status"], "not_assigned")
        self.assertEqual(state["outcome_access"], "none")
        self.assertIsNone(episode["steps"][0]["result"])
        self.assertNotIn("private_verifier", episode)

    def test_invalid_exposure_rule_hash_is_rejected_before_actor_construction(self):
        manifest = build_manifest(
            [{"task_id": "task-a", "group_id": "group-a", "role": "target_train"}],
            benchmark="fixture", revision="v1")
        bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
                "memory_count": 0, "memories": [], "audit_sources": {}}
        bank["bank_hash"] = digest(bank)

        def should_not_construct_actor():
            raise AssertionError("invalid rules must fail before actor construction")

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "schema/hash"):
            collect_target(
                manifest, role="target_train", spec=StudySpec(),
                env_factory=lambda task: PreparationEnvironment(task_id=task["task_id"]),
                actor_factory=should_not_construct_actor,
                output_dir=Path(directory) / "run", prepare_before_action=True,
                bank=bank, encoder=EncoderFixture(),
                exposure_rules={"schema_version": "action-exposure-rules.v1",
                                "frozen": True, "rules": {}, "rules_hash": "invalid"})

    def test_malformed_exposure_predicate_is_rejected_before_actor_construction(self):
        manifest = build_manifest(
            [{"task_id": "task-a", "group_id": "group-a", "role": "target_train"}],
            benchmark="fixture", revision="v1")
        bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
                "memory_count": 0, "memories": [], "audit_sources": {}}
        bank["bank_hash"] = digest(bank)
        rules = {"schema_version": "action-exposure-rules.v1", "frozen": True,
                 "rules": {"memory-a": {
                     "schema_version": "action-exposure-rule.v1", "confidence": "high",
                     "exposed_when": {"action": [{"field": "_private", "equals": True}],
                                      "context": []},
                     "not_exposed_when": {"action": [{"field": "tool", "exists": False}],
                                          "context": []}}}}
        rules["rules_hash"] = digest(rules)

        def should_not_construct_actor():
            raise AssertionError("invalid predicates must fail before actor construction")

        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "safe field"):
            collect_target(
                manifest, role="target_train", spec=StudySpec(),
                env_factory=lambda task: PreparationEnvironment(task_id=task["task_id"]),
                actor_factory=should_not_construct_actor,
                output_dir=Path(directory) / "run", prepare_before_action=True,
                bank=bank, encoder=EncoderFixture(), exposure_rules=rules)

    def test_preparation_freezes_selected_state_without_executing_a0(self):
        manifest = build_manifest(
            [{"task_id": "task-a", "group_id": "group-a", "role": "target_train"}],
            benchmark="fixture", revision="v1")
        bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
                "memory_count": 0, "memories": [], "audit_sources": {}}
        bank["bank_hash"] = digest(bank)
        envs = []
        actors = []

        def env_factory(task):
            env = CountingEnvironment(task_id=task["task_id"], safe_intent=True)
            envs.append(env)
            return env

        def actor_factory():
            actor = ScriptedActor()
            actors.append(actor)
            return actor

        with tempfile.TemporaryDirectory() as directory:
            summary = collect_target(
                manifest, role="target_train", spec=StudySpec(max_steps=4),
                env_factory=env_factory, actor_factory=actor_factory,
                output_dir=Path(directory) / "run",
                hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
                prepare_before_action=True, bank=bank, encoder=EncoderFixture())

            run = read_json(Path(directory) / "run" / "run.json")
            trace = read_json(next((Path(directory) / "run" / "traces").glob("*.json")))
            episode = trace["episode"]
            selected_step = episode["steps"][0]
            state = selected_step["pre_treatment_state"]

        self.assertEqual(summary["pre_treatment_states_frozen"], 1)
        self.assertEqual(run["pre_treatment"]["bank_hash"], bank["bank_hash"])
        self.assertEqual(run["pre_treatment"]["encoder_identity"], EncoderFixture.identity)
        self.assertEqual(trace["trace_hash"], digest({k: v for k, v in trace.items() if k != "trace_hash"}))
        self.assertEqual(episode["status"], "pre_treatment_ready")
        self.assertIsNone(episode["success"])
        self.assertIsNone(episode["utility"])
        self.assertEqual(state["randomization_status"], "not_assigned")
        self.assertEqual(state["outcome_access"], "none")
        self.assertEqual(state["record_hash"], digest({k: v for k, v in state.items() if k != "record_hash"}))
        self.assertEqual(selected_step["action"], episode["checkpoint_sampling"]["selected_action"])
        self.assertIsNone(selected_step["result"])
        self.assertEqual(episode["costs"]["tool_calls"], 0)
        self.assertNotIn("private_verifier", episode)
        self.assertEqual(len(actors), 2)  # one for frozen configuration, one for the rollout
        self.assertEqual(envs[-1].validate_calls, 0)
        self.assertEqual(envs[-1].step_calls, 0)
        self.assertEqual(envs[-1].snapshot()["state"]["steps"], 0)

    def test_selected_unparsed_a0_is_preparation_error_not_a_ready_state(self):
        from cave_mem_v3.interventions.inputs import load_prepared_run
        manifest = build_manifest(
            [{"task_id": "task-a", "group_id": "group-a", "role": "target_train"}],
            benchmark="fixture", revision="v1")
        bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
                "memory_count": 0, "memories": [], "audit_sources": {}}
        bank["bank_hash"] = digest(bank)
        envs = []
        def factory(task):
            env = CountingEnvironment(task_id=task["task_id"])
            envs.append(env)
            return env
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            collect_target(manifest, role="target_train", spec=StudySpec(max_steps=4),
                env_factory=factory, actor_factory=MalformedDraftActor, output_dir=root,
                hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
                prepare_before_action=True, bank=bank, encoder=EncoderFixture())
            trace = read_json(next((root / "traces").glob("*.json")))
            _, inputs = load_prepared_run(root, bank)
        self.assertEqual(trace["episode"]["status"], "preparation_error")
        self.assertIsNone(trace["episode"]["success"])
        self.assertNotIn("pre_treatment_state", trace["episode"]["steps"][-1])
        self.assertEqual(inputs[0]["reason"], "preparation_error")
        self.assertEqual(envs[-1].step_calls, 0)


if __name__ == "__main__":
    unittest.main()
