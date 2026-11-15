"""End-to-end paired software checks; no external model or benchmark calls."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.cli import main
from cave_mem_v3.common.contracts import StudySpec, canonical, read_json, write_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.interventions.outcomes import summarize_target_pairs
from test_target_pair_inputs import make_prepared_fixture

ROOT = Path(__file__).resolve().parents[1]


class TargetPairTests(unittest.TestCase):
    def collect(self, fixture, root, *, actor_factory=None, env_factory=None):
        return collect_target_pairs(fixture["target_dir"], fixture["bank"],
            env_factory=env_factory or fixture["env_factory"],
            actor_factory=actor_factory or fixture["actor_factory"], output_dir=root)

    def test_paired_fixture_preserves_original_inputs_and_restores_each_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            frozen = fixture["trace_path"].read_bytes()
            summary = self.collect(fixture, root / "paired")
            pair = next((root / "paired" / "pairs").iterdir())
            plan = read_json(pair / "plan.json")
            silent = read_json(pair / "silent" / "result.json")["record"]
            remind = read_json(pair / "remind" / "result.json")["record"]
            self.assertEqual(fixture["trace_path"].read_bytes(), frozen)
            self.assertEqual(silent["initial_checkpoint_hash"], remind["initial_checkpoint_hash"])
            self.assertEqual(plan["checkpoint_hash"], silent["initial_checkpoint_hash"])
            self.assertFalse(silent["success"])
            self.assertTrue(remind["success"])
            self.assertEqual(silent["costs"]["model_calls"], 0)
            self.assertEqual(remind["costs"]["model_calls"], 3)
            self.assertEqual(plan["order_probability"], .5)
            self.assertEqual(set(plan["order"]), {"silent", "remind"})
            self.assertEqual((summary["planned_pairs"], summary["completed_pairs"], summary["valid_outcomes"]), (1, 1, 1))
            row = summary["pairs"][0]
            self.assertEqual((row["H"], row["rescue"]), (0, 1))
            self.assertAlmostEqual(row["D"], .97)
            self.assertEqual(row["actual_experiment_additive_costs"]["model_calls"], 4)
            self.assertEqual(row["deployment_costs"]["silent"]["model_calls"], 1)
            self.assertEqual(row["deployment_costs"]["remind"]["model_calls"], 4)
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))

    def test_completed_resume_never_reconstructs_actor_or_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            summary = self.collect(fixture, root / "paired")
            files = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
            def prohibited(*args):
                raise AssertionError("completed resume attempted execution")
            resumed = self.collect(fixture, root / "paired", actor_factory=prohibited, env_factory=prohibited)
            self.assertEqual(summary, resumed)
            self.assertEqual(files, {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")})

    def test_changed_frozen_factory_is_blocked_before_intervention(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            self.collect(fixture, root / "paired", actor_factory=lambda: ScriptedActor(error_rate=.2))
            pair = next((root / "paired" / "pairs").iterdir())
            self.assertEqual(read_json(pair / "closed.json")["status"], "blocked")
            self.assertEqual(read_json(pair / "audit.json")["status"], "blocked")
            self.assertFalse((pair / "plan.json").exists())
            self.assertFalse(list(pair.glob("*/result.json")))

    def test_timeout_is_pending_and_reentry_never_retries(self):
        calls = []
        class TimeoutActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                raise TimeoutError("completion is unknown")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            summary = self.collect(fixture, root / "paired", actor_factory=TimeoutActor)
            pair = next((root / "paired" / "pairs").iterdir())
            pending = list(pair.glob("*/pending.json"))
            self.assertEqual(len(pending), 1)
            self.assertEqual(summary["pending_pairs"], 1)
            self.assertEqual(summary["valid_outcomes"], 0)
            self.assertTrue(all(summary["pairs"][0][label] is None for label in ("H", "rescue", "D")))
            self.assertEqual(len(calls), 1)
            self.assertFalse((pair / "closed.json").exists())
            self.assertIsNone(read_json(pending[0])["partial_record"]["success"])
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))
            with self.assertRaisesRegex(ValueError, "incomplete paired attempt"):
                self.collect(fixture, root / "paired", actor_factory=TimeoutActor)
            self.assertEqual(len(calls), 1)

    def test_interrupted_request_leaves_started_event_and_is_not_replayed(self):
        calls = []
        class InterruptedActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            with self.assertRaises(KeyboardInterrupt):
                self.collect(fixture, root / "paired", actor_factory=InterruptedActor)
            pair = next((root / "paired" / "pairs").iterdir())
            events = [read_json(p)["event"] for p in pair.glob("*/events/*.json")]
            self.assertTrue(any(e["kind"] == "actor.started" for e in events))
            self.assertFalse(any(e["kind"] == "actor.completed" for e in events))
            self.assertEqual(len(calls), 1)
            with self.assertRaisesRegex(ValueError, "incomplete paired attempt"):
                self.collect(fixture, root / "paired", actor_factory=InterruptedActor)
            self.assertEqual(len(calls), 1)

    def test_corrupted_arm_result_and_event_are_rejected_by_pure_summary(self):
        for artifact in ("result", "event"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = make_prepared_fixture(root / "inputs")
                self.collect(fixture, root / "paired")
                pair = next((root / "paired" / "pairs").iterdir())
                path = (pair / "silent" / "result.json" if artifact == "result"
                        else next((pair / "silent" / "events").glob("*.json")))
                value = read_json(path)
                if artifact == "result":
                    value["record"]["success"] = True
                else:
                    value["event"]["kind"] = "forged.event"
                write_json(path, value)
                corrupted = path.read_bytes()
                with self.assertRaises(ValueError):
                    summarize_target_pairs(root / "paired")
                self.assertEqual(path.read_bytes(), corrupted)

    def test_excluded_inputs_create_no_pair_or_actor(self):
        for options in ({"empty_bank": True}, {"select": False}, {"target_max_calls": 1}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = make_prepared_fixture(root / "inputs", **options)
                def prohibited(*args):
                    raise AssertionError("excluded input started execution")
                summary = self.collect(fixture, root / "paired", actor_factory=prohibited, env_factory=prohibited)
                self.assertFalse((root / "paired" / "pairs").exists())
                self.assertEqual(summary, summarize_target_pairs(root / "paired"))

    def test_reminder_can_cause_a_real_fixture_regression(self):
        class HarmfulReminder(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                response = super().propose_target(*args, **kwargs)
                if kwargs.get("reminder") is not None:
                    action = {"type": "operate"}
                    return replace(response, action=action, text=canonical(action))
                return response
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs", target_safe=True)
            summary = self.collect(fixture, root / "paired", actor_factory=HarmfulReminder)
            pair = next((root / "paired" / "pairs").iterdir())
            silent = read_json(pair / "silent" / "result.json")["record"]
            remind = read_json(pair / "remind" / "result.json")["record"]
            self.assertTrue(silent["success"])
            self.assertFalse(remind["success"])
            self.assertTrue(silent["outcome_valid"] and remind["outcome_valid"])
            self.assertEqual((summary["pairs"][0]["H"], summary["pairs"][0]["rescue"]), (1, 0))
            self.assertAlmostEqual(summary["pairs"][0]["D"], -.99)
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))

    def test_unknown_weighted_cost_retains_outcomes_but_not_utility(self):
        class UnknownCostActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                response = super().propose_target(*args, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs", target_spec=StudySpec(
                max_steps=4, max_model_calls=4, inference_weights={"api_cost": 1.0}))
            summary = self.collect(fixture, root / "paired", actor_factory=UnknownCostActor)
            pair = next((root / "paired" / "pairs").iterdir())
            remind = read_json(pair / "remind" / "result.json")["record"]
            self.assertTrue(remind["success"])
            self.assertTrue(remind["outcome_valid"])
            self.assertFalse(remind["utility_valid"])
            self.assertIsNone(remind["costs"]["api_cost"])
            self.assertIsNone(remind["utility"])
            self.assertEqual((summary["valid_outcomes"], summary["valid_utilities"]), (1, 0))
            self.assertEqual((summary["pairs"][0]["H"], summary["pairs"][0]["rescue"]), (0, 1))
            self.assertIsNone(summary["pairs"][0]["D"])
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))

    def test_validation_infrastructure_failure_is_summarizable_and_not_retried(self):
        class BrokenValidator(PreparationEnvironment):
            def validate_action(self, action):
                raise RuntimeError("validator unavailable")
        instances = []
        def env_factory(task):
            cls = PreparationEnvironment if len(instances) < 2 else BrokenValidator
            env = cls(task_id=task["task_id"])
            instances.append(env)
            return env
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            summary = self.collect(fixture, root / "paired", env_factory=env_factory)
            self.assertEqual(summary["completed_pairs"], 1)
            self.assertEqual(summary["valid_outcomes"], 0)
            self.assertTrue(all(summary["pairs"][0][key] is None for key in ("D", "H", "rescue")))
            pair = next((root / "paired" / "pairs").iterdir())
            for arm in ("silent", "remind"):
                result = read_json(pair / arm / "result.json")["record"]
                self.assertEqual(result["termination_reason"], "validation_system_failure")
                self.assertEqual(result["costs"]["tool_calls"], 0)
                events = [read_json(path)["event"] for path in (pair / arm / "events").glob("*.json")]
                self.assertTrue(any(event["kind"] == "validation.failed" for event in events))
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))
            previous_instances = len(instances)
            self.assertEqual(summary, self.collect(fixture, root / "paired", env_factory=env_factory))
            self.assertEqual(len(instances), previous_instances)

    def test_unsupported_reminder_blocks_before_actor_started_and_is_not_retried(self):
        class NoReminder(ScriptedActor):
            propose_target = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            summary = self.collect(fixture, root / "paired", actor_factory=NoReminder)
            self.assertEqual(summary["blocked_pairs"], 1)
            self.assertTrue(all(summary["pairs"][0][key] is None for key in ("D", "H", "rescue")))
            pair = next((root / "paired" / "pairs").iterdir())
            result = read_json(pair / "remind" / "result.json")["record"]
            self.assertEqual(result["termination_reason"], "target_reminder_not_supported")
            self.assertEqual(result["costs"]["model_calls"], 0)
            events = [read_json(path)["event"] for path in (pair / "remind" / "events").glob("*.json")]
            self.assertFalse(any(event["kind"] == "actor.started" for event in events))
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))
            def prohibited():
                raise AssertionError("blocked result was retried")
            self.assertEqual(summary, self.collect(fixture, root / "paired", actor_factory=prohibited))

    def test_invalid_actor_response_preserves_unknown_cost_and_no_labels(self):
        calls = []
        class NoResponse(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                return None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = make_prepared_fixture(root / "inputs")
            summary = self.collect(fixture, root / "paired", actor_factory=NoResponse)
            self.assertEqual(len(calls), 1)
            self.assertEqual(summary["valid_outcomes"], 0)
            self.assertTrue(all(summary["pairs"][0][key] is None for key in ("D", "H", "rescue")))
            pair = next((root / "paired" / "pairs").iterdir())
            result = read_json(pair / "remind" / "result.json")["record"]
            self.assertEqual(result["termination_reason"], "invalid_actor_response")
            self.assertEqual(result["costs"]["model_calls"], 1)
            self.assertIsNone(result["costs"]["api_cost"])
            self.assertIsNone(result["costs"]["physical_attempts"])
            self.assertEqual(summary, summarize_target_pairs(root / "paired"))
            self.assertEqual(summary, self.collect(fixture, root / "paired", actor_factory=NoResponse))
            self.assertEqual(len(calls), 1)


class TargetPairCliTests(unittest.TestCase):
    def fixture_and_config(self, root):
        fixture = make_prepared_fixture(root / "inputs")
        config = root / "fixture.toml"
        config.write_text((ROOT / "configs" / "phase1" / "demo.toml").read_text(encoding="utf-8")
                          .replace('study_id = "phase1-fixture-demo"', 'study_id = "phase1-pilot"'), encoding="utf-8")
        return fixture, config

    def test_pair_cli_then_readonly_summary_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, config = self.fixture_and_config(root)
            argv = ["pair-target", "--config", str(config), "--target-run", str(fixture["target_dir"]),
                    "--bank", str(root / "inputs" / "bank.json"), "--output", str(root / "paired")]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(argv), 0)
            summary = json.loads(output.getvalue())
            self.assertEqual(summary["valid_outcomes"], 1)
            output = io.StringIO()
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("summary called actor")), \
                    patch("cave_mem_v3.cli._env_factory", side_effect=AssertionError("summary opened environment")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(main(["summarize-target-pairs", "--run", str(root / "paired")]), 0)
            self.assertEqual(json.loads(output.getvalue()), summary)
            output = io.StringIO()
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("resume called actor")), \
                    contextlib.redirect_stdout(output):
                self.assertEqual(main(argv), 0)
            self.assertEqual(json.loads(output.getvalue()), summary)

    def test_pending_cli_returns_nonzero_without_retry(self):
        calls = []
        class TimeoutActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                raise TimeoutError("unknown completion")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture, config = self.fixture_and_config(root)
            output = io.StringIO()
            with patch("cave_mem_v3.cli._actor", side_effect=lambda raw: TimeoutActor()), contextlib.redirect_stdout(output):
                code = main(["pair-target", "--config", str(config), "--target-run", str(fixture["target_dir"]),
                    "--bank", str(root / "inputs" / "bank.json"), "--output", str(root / "paired")])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(output.getvalue())["pending_pairs"], 1)
            self.assertEqual(len(calls), 1)

    def test_offline_demo_runs_and_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "demo"
            command = [sys.executable, str(ROOT / "examples" / "phase2_demo.py"), "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = read_json(output / "summary.json")
            self.assertEqual((summary["completed_pairs"], summary["valid_outcomes"]), (1, 1))
            self.assertIn("No external model or real benchmark was run", result.stdout)
            saved = (output / "summary.json").read_bytes()
            again = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=30)
            self.assertNotEqual(again.returncode, 0)
            self.assertIn("existing artifacts are preserved", again.stderr)
            self.assertEqual((output / "summary.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
