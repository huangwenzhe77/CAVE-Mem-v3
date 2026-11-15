"""Offline CLI and durable-observation checks for the randomized single-arm path."""
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
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.cli import main
from cave_mem_v3.common.contracts import digest, read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.interventions.single_outcomes import summarize_target_single
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture

ROOT = Path(__file__).resolve().parents[1]


class LocalEncoder(EncoderFixture):
    def __init__(self, path, *, revision, device):
        self.identity = {**self.identity, "path": path, "revision": revision, "device": device}


def _reject_paired_labels(test, value):
    if isinstance(value, dict):
        test.assertTrue({"D", "H", "rescue", "silent_success", "remind_success"}.isdisjoint(value))
        for child in value.values():
            _reject_paired_labels(test, child)
    elif isinstance(value, list):
        for child in value:
            _reject_paired_labels(test, child)


class TargetSingleCliTests(unittest.TestCase):
    def fixture(self, root, *, probability=.5, empty_bank=False, max_calls=4, select=True, unknown_weight=False,
                task_count=1):
        root = Path(root)
        manifest = None if task_count == 1 else build_manifest([
            {"task_id": "source-a", "group_id": "source-family-a", "role": "history"},
            *[{"task_id": f"target-{index}", "group_id": f"target-family-{index}", "role": "target_train"}
              for index in range(task_count)]], benchmark="preparation-fixture", revision="fixture-v3")
        fixture = make_prepared_fixture(root / "inputs", empty_bank=empty_bank, manifest=manifest)
        manifest = root / "manifest.json"
        write_json(manifest, fixture["manifest"])
        bank_path = root / "bank.json"
        write_json(bank_path, fixture["bank"])
        config = root / "fixture.toml"
        contents = (ROOT / "configs" / "phase1" / "demo.toml").read_text(encoding="utf-8")
        contents = contents.replace("max_model_calls = 4", f"max_model_calls = {max_calls}")
        if unknown_weight:
            contents += '\n[study.inference_weights]\napi_cost = 1.0\n'
        config.write_text(contents, encoding="utf-8")
        encoder = root / "encoder"
        encoder.mkdir()
        output = root / "single"
        hazard = ".99" if select else "0.000000000001"
        argv = ["collect-single-target", "--config", str(config), "--manifest", str(manifest),
                "--bank", str(bank_path), "--output", str(output), "--encoder-path", str(encoder),
                "--encoder-revision", "fixture-v1", "--assignment-probability", str(probability),
                "--low-risk-probability", hazard, "--high-risk-probability", hazard]
        return argv, output

    def invoke(self, argv, *, actor=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", LocalEncoder), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if actor is None:
                code = main(argv)
            else:
                with patch("cave_mem_v3.cli._actor", side_effect=lambda raw: actor()):
                    code = main(argv)
        return code, json.loads(stdout.getvalue()) if stdout.getvalue() else None, stderr.getvalue()

    def test_cli_collects_only_assigned_arm_with_known_probability(self):
        for probability, expected_arm in ((1e-12, "silent"), (1 - 1e-12, "remind")):
            with self.subTest(arm=expected_arm), tempfile.TemporaryDirectory() as temporary:
                argv, output = self.fixture(temporary, probability=probability)
                code, summary, error = self.invoke(argv)
                self.assertEqual(code, 0, error)
                self.assertEqual((summary["planned_rollouts"], summary["completed_rollouts"], summary["valid_outcomes"]), (1, 1, 1))
                row = summary["observations"][0]
                self.assertEqual(row["arm"], expected_arm)
                self.assertEqual(row["propensity"], probability if expected_arm == "remind" else 1 - probability)
                self.assertEqual(row["b"], {"silent": 1 - probability, "remind": probability})
                self.assertEqual(row["observed_success"], expected_arm == "remind")
                self.assertEqual(row["observed_costs"]["model_calls"], 4 if expected_arm == "remind" else 1)
                self.assertEqual(len(list((output / "sessions").glob("*/result.json"))), 1)
                _reject_paired_labels(self, summary)

    def test_readonly_summary_and_completed_reentry_make_no_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary)
            code, original, error = self.invoke(argv)
            self.assertEqual(code, 0, error)
            saved = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*.json")}
            stdout = io.StringIO()
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("constructed actor")), \
                    patch("cave_mem_v3.cli._env_factory", side_effect=AssertionError("constructed environment")), \
                    contextlib.redirect_stdout(stdout):
                self.assertEqual(main(["summarize-target-single", "--run", str(output)]), 0)
            self.assertEqual(original, json.loads(stdout.getvalue()))
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("resume actor")):
                code, resumed, error = self.invoke(argv)
            self.assertEqual(code, 0, error)
            self.assertEqual(original, resumed)
            self.assertEqual(saved, {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*.json")})

    def test_invalid_probability_and_test_role_fail_before_encoder_or_actor(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary)
            cases = []
            for probability in ("0", "1", "-0.1", "nan", "inf"):
                changed = list(argv)
                changed[changed.index("--assignment-probability") + 1] = probability
                cases.append(changed)
            cases.extend([argv + ["--role", role] for role in ("test", "validation")])
            for changed in cases:
                with self.subTest(argv=changed[-2:]), \
                        patch("cave_mem_v3.cli._actor", side_effect=AssertionError("actor should not be built")), \
                        patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", side_effect=AssertionError("encoder should not load")), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(changed), 2)
            self.assertFalse(output.exists())

    def test_assignment_probability_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary)
            position = argv.index("--assignment-probability")
            del argv[position:position + 2]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                main(argv)
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())

    def test_pending_returns_nonzero_and_reentry_does_not_retry(self):
        calls = []
        class TimeoutActor(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                calls.append(kwargs["call_id"])
                raise TimeoutError("external completion unknown")
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary, probability=1 - 1e-12)
            code, summary, error = self.invoke(argv, actor=TimeoutActor)
            self.assertEqual(code, 2, error)
            self.assertEqual((summary["pending_rollouts"], summary["valid_outcomes"]), (1, 0))
            self.assertIsNone(summary["observations"][0]["observed_success"])
            self.assertEqual(summary, summarize_target_single(output))
            self.assertEqual(len(calls), 1)
            code, resumed, error = self.invoke(argv, actor=TimeoutActor)
            self.assertEqual(code, 2)
            self.assertIsNone(resumed)
            self.assertIn("incomplete single-session", error)
            self.assertEqual(len(calls), 1)
            _reject_paired_labels(self, summary)

    def test_exclusions_keep_costs_and_unknown_outcomes_without_assignment(self):
        for options, reason in (({"empty_bank": True}, "no_memory"),
                                ({"max_calls": 1}, "insufficient_budget"),
                                ({"select": False}, "no_selection")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                argv, output = self.fixture(temporary, **options)
                code, summary, error = self.invoke(argv)
                self.assertEqual(code, 0, error)
                self.assertEqual((summary["excluded_rollouts"], summary["valid_outcomes"]), (1, 0))
                row = summary["observations"][0]
                self.assertEqual(row["reason"], reason)
                self.assertIsNone(row["arm"])
                self.assertIsNone(row["observed_success"])
                self.assertIsNone(row["utility"])
                self.assertGreaterEqual(row["observed_costs"]["model_calls"], 1)
                self.assertFalse(list(output.glob("sessions/*/assignment.json")))

    def test_unknown_weighted_fee_keeps_observed_outcome_without_utility(self):
        class UnknownFee(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                response = super().propose_target(*args, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary, probability=1 - 1e-12, unknown_weight=True)
            code, summary, error = self.invoke(argv, actor=UnknownFee)
            self.assertEqual(code, 0, error)
            self.assertEqual((summary["valid_outcomes"], summary["valid_utilities"]), (1, 0))
            row = summary["observations"][0]
            self.assertTrue(row["observed_success"])
            self.assertIsNone(row["observed_costs"]["api_cost"])
            self.assertIsNone(row["utility"])
            self.assertEqual(summary, summarize_target_single(output))

    def test_resealed_assignment_probability_arm_and_draw_tampering_rejected(self):
        for field, value in (("propensity", .7), ("arm", "forged-arm"), ("draw", .12345), ("z", 7)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                argv, output = self.fixture(temporary)
                self.assertEqual(self.invoke(argv)[0], 0)
                path = next(output.glob("sessions/*/assignment.json"))
                assignment = read_json(path)
                assignment[field] = value
                write_json(path, seal(assignment))
                with self.assertRaisesRegex(ValueError, "assignment"):
                    summarize_target_single(output)

    def test_resealed_cached_action_and_state_cannot_replace_original_preparation(self):
        for field in ("a0", "x"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                argv, output = self.fixture(temporary)
                self.assertEqual(self.invoke(argv)[0], 0)
                path = next(output.glob("sessions/*/input.json"))
                value = read_json(path)
                if field == "a0":
                    value["a0"] = {"type": "finish"}
                    value["a0_hash"] = digest(value["a0"])
                else:
                    state = value["pre_treatment_state"]
                    state["x"]["b"]["remaining"]["environment_steps"] -= 1
                    state["x_hash"] = digest(state["x"])
                    value["pre_treatment_state"] = seal(state)
                    value["x_hash"] = state["x_hash"]
                write_json(path, seal(value))
                with self.assertRaisesRegex(ValueError, "input differs"):
                    summarize_target_single(output)

    def test_resealed_result_costs_and_outcome_cannot_replace_raw_events(self):
        for field in ("costs", "success"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                argv, output = self.fixture(temporary, probability=1 - 1e-12)
                self.assertEqual(self.invoke(argv)[0], 0)
                path = next(output.glob("sessions/*/result.json"))
                wrapper = read_json(path)
                record = wrapper["record"]
                if field == "costs":
                    record["costs"]["input_tokens"] += 1
                else:
                    record["success"] = not record["success"]
                wrapper["record"] = seal(record)
                write_json(path, seal(wrapper))
                with self.assertRaises(ValueError):
                    summarize_target_single(output)

    def test_resealed_preparation_event_chain_cannot_forge_treatment_before_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary)
            self.assertEqual(self.invoke(argv)[0], 0)
            previous = None
            for path in sorted(output.glob("sessions/*/preparation_events/*.json")):
                entry = read_json(path)
                if entry["event"]["kind"] == "actor.started":
                    entry["event"]["stage"] = "target_remind"
                entry["previous_hash"] = previous
                entry = seal(entry)
                write_json(path, entry)
                previous = entry["record_hash"]
            with self.assertRaisesRegex(ValueError, "request identity/order"):
                summarize_target_single(output)

    def test_single_demo_runs_offline_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            command = [sys.executable, str(ROOT / "examples" / "phase2_single_demo.py"), "--output", str(output)]
            result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = read_json(output / "summary.json")
            self.assertEqual((summary["completed_rollouts"], summary["valid_outcomes"]), (1, 1))
            self.assertIn("No external model or real benchmark was run", result.stdout)
            _reject_paired_labels(self, summary)
            saved = (output / "summary.json").read_bytes()
            rerun = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True, timeout=30)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("existing artifacts are preserved", rerun.stderr)
            self.assertEqual(saved, (output / "summary.json").read_bytes())

    def test_locally_consistent_resealed_second_task_cannot_change_served_identity(self):
        from cave_mem_v3.interventions.single import _input, single_assignment
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary, probability=1 - 1e-12, task_count=2)
            self.assertEqual(self.invoke(argv)[0], 0)
            run = read_json(output / "run.json")
            directory = sorted((output / "sessions").iterdir())[1]
            preparation = read_json(directory / "preparation.json")
            preparation["response_identity"]["fingerprint"] = "different-task-backend"
            for step in preparation["steps"]:
                step["actor_response"]["fingerprint"] = "different-task-backend"
            preparation = seal(preparation)
            write_json(directory / "preparation.json", preparation)
            task = next(task for task in run["manifest"]["tasks"] if task["task_id"] == preparation["task_id"])
            item = _input(run, task, preparation)
            write_json(directory / "input.json", item)
            old_assignment = read_json(directory / "assignment.json")
            assignment = single_assignment(run, item, old_assignment["reminder"])
            write_json(directory / "assignment.json", assignment)
            for journal in ("preparation_events", "events"):
                previous = None
                for path in sorted((directory / journal).glob("*.json")):
                    entry = read_json(path)
                    if entry["event"]["kind"] == "actor.completed":
                        entry["event"]["response"]["fingerprint"] = "different-task-backend"
                    if journal == "events":
                        entry["owner_hash"] = assignment["record_hash"]
                    entry["previous_hash"] = previous
                    entry = seal(entry)
                    write_json(path, entry)
                    previous = entry["record_hash"]
            wrapper = read_json(directory / "result.json")
            record = wrapper["record"]
            record["response_identity"]["fingerprint"] = "different-task-backend"
            for step in record["steps"]:
                if step.get("actor_response"):
                    step["actor_response"]["fingerprint"] = "different-task-backend"
            wrapper["record"] = seal(record)
            wrapper["assignment_hash"] = assignment["record_hash"]
            write_json(directory / "result.json", seal(wrapper))
            closed = read_json(directory / "closed.json")
            closed.update(input_hash=item["record_hash"], assignment_hash=assignment["record_hash"])
            write_json(directory / "closed.json", seal(closed))
            with self.assertRaisesRegex(ValueError, "identity drift was not retained"):
                summarize_target_single(output)

    def test_readonly_summary_rejects_attempt_after_observed_shared_identity_failure(self):
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
            argv, output = self.fixture(temporary, probability=1 - 1e-12, task_count=3)
            code, summary, error = self.invoke(argv, actor=DriftingActor)
            self.assertEqual(code, 0, error)
            self.assertEqual(summary["valid_outcomes"], 1)
            self.assertEqual(summary["unattempted_rollouts"], 1)
            self.assertEqual(summary, summarize_target_single(output))
            run = read_json(output / "run.json")
            third_id = digest([run["record_hash"], run["task_order"][2]])[:24]
            (output / "sessions" / f"000002-{third_id}").mkdir()
            with self.assertRaisesRegex(ValueError, "continued after a shared failure"):
                summarize_target_single(output)

    def test_readonly_summary_rejects_attempt_after_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            argv, output = self.fixture(temporary, task_count=2)
            self.assertEqual(self.invoke(argv)[0], 0)
            first = sorted((output / "sessions").iterdir())[0]
            result = read_json(first / "result.json")
            result["cleanup_errors"] = ["OSError"]
            write_json(first / "result.json", seal(result))
            closed = read_json(first / "closed.json")
            closed.update(cleanup_errors=["OSError"], status="blocked", reason="cleanup_error")
            write_json(first / "closed.json", seal(closed))
            with self.assertRaisesRegex(ValueError, "continued after a shared failure"):
                summarize_target_single(output)


if __name__ == "__main__":
    unittest.main()
