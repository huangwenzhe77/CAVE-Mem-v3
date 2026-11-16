"""End-to-end command boundary checks, including frozen source provenance."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cave_mem_v3.cli import main
from cave_mem_v3.common.contracts import read_json, write_json
from cave_mem_v3.memory_generation.bank import load_bank

ROOT = Path(__file__).resolve().parents[1]


class CommandTests(unittest.TestCase):
    def invoke(self, *args, expected=0):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main([str(a) for a in args])
        self.assertEqual(result, expected, stderr.getvalue())
        return json.loads(stdout.getvalue()) if result == 0 else stderr.getvalue()

    def test_target_rollout_samples_once_then_continues_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            output = root / "target-run"
            config = ROOT / "configs" / "phase1" / "demo.toml"
            self.invoke("manifest", "--catalog", ROOT / "configs" / "benchmarks" / "demo.json",
                        "--output", manifest)
            result = self.invoke("collect-target", "--config", config, "--manifest", manifest,
                                 "--output", output, "--role", "target_train",
                                 "--low-risk-probability", ".99",
                                 "--high-risk-probability", ".99")
            self.assertEqual(result["attempted_rollouts"], 1)
            self.assertEqual(result["selected_checkpoints"], 1)
            trace_path = next((output / "traces").glob("*.json"))
            trace = read_json(trace_path)
            episode = trace["episode"]
            self.assertEqual(episode["status"], "terminal")
            self.assertEqual(episode["checkpoint_sampling"]["status"], "selected")
            decisions = episode["checkpoint_sampling"]["selection_decisions"]
            self.assertEqual(len([item for item in decisions if item["selected"]]), 1)
            selected = episode["checkpoint_sampling"]["selected_step_index"]
            for step in episode["steps"][selected + 1:]:
                self.assertEqual(step["checkpoint_sampling"]["status"], "already_selected")
            before = trace_path.read_bytes()
            self.invoke("collect-target", "--config", config, "--manifest", manifest,
                        "--output", output, "--role", "target_train",
                        "--low-risk-probability", ".99",
                        "--high-risk-probability", ".99")
            self.assertEqual(before, trace_path.read_bytes())

    def test_complete_phase_one_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, collection = root / "manifest.json", root / "collection"
            config = ROOT / "configs" / "phase1" / "demo.toml"
            self.invoke("manifest", "--catalog", ROOT / "configs" / "benchmarks" / "demo.json", "--output", manifest)
            summary = self.invoke("collect", "--config", config, "--manifest", manifest, "--output", collection)
            self.assertEqual(summary["attempted_runs"], 4)
            self.assertEqual(summary["failed_runs"], 2)
            self.assertEqual(summary["successful_runs"], 2)
            original = {p.name: p.read_bytes() for p in (collection / "traces").glob("*.json")}
            self.invoke("collect", "--config", config, "--manifest", manifest, "--output", collection)
            self.assertEqual(original, {p.name: p.read_bytes() for p in (collection / "traces").glob("*.json")})
            candidates = root / "candidates.json"
            self.invoke("candidates", "--collection", collection, "--output", candidates)
            raw = read_json(candidates)
            self.assertEqual(len(raw["candidates"]), 2)
            for draft in raw["candidates"]:
                draft.update(mechanism="operation requires prior preparation", correction={"type": "prepare"},
                             positive_conditions=[{"field": "requires_preparation", "equals": True}],
                             exclusion_conditions=[{"field": "prepared", "equals": True}],
                             scope="deterministic development fixture only", proposed_by="fixture-test")
            write_json(candidates, raw)
            unset = root / "unset-replay-k.toml"
            unset.write_text(config.read_text().replace("replay_pairs = 2\n", ""))
            error = self.invoke("verify", "--config", unset, "--manifest", manifest,
                                "--collection", collection, "--candidates", candidates,
                                "--output", root / "not-started", expected=2)
            self.assertIn("replay_pairs", error)
            self.assertFalse((root / "not-started").exists())
            verification = root / "verification"
            self.invoke("verify", "--config", config, "--manifest", manifest, "--collection", collection,
                        "--candidates", candidates, "--output", verification)
            records = read_json(verification / "verification.json")["records"]
            self.assertTrue(all(record["evidence"]["v"] == 1 for record in records))
            self.assertTrue(all(record["evidence"]["plan"]["replay_spec"]["replay_pairs"] == 2 for record in records))
            automatic_bank = load_bank(verification / "bank.json")
            self.assertEqual(automatic_bank["memory_count"], 2)
            bank_path = root / "bank.json"
            result = self.invoke("bank", "--verification", verification, "--output", bank_path)
            self.assertEqual(result["verified_memories"], 2)
            bank = load_bank(bank_path)
            self.assertEqual(bank["memory_count"], 2)
            self.assertTrue(all(m["quality"]["v"] == 1 for m in bank["memories"]))
            self.invoke("bank", "--verification", verification, "--output", bank_path, expected=2)

    def test_pilot_server_choices_are_reported_without_download_or_requests(self):
        result = self.invoke("check", "--config", ROOT / "configs" / "phase1" / "server.toml")
        self.assertTrue(result["ready"])
        self.assertEqual(result["pending"], [])
        self.assertEqual(result["model_downloaded"], Path("/home/ljj/models/Qwen2.5-7B-Instruct").is_dir())

    def test_missing_collection_budget_does_not_fall_back_to_library_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            self.invoke("manifest", "--catalog", ROOT / "configs" / "benchmarks" / "demo.json", "--output", manifest)
            config = root / "incomplete.toml"
            config.write_text((ROOT / "configs" / "phase1" / "demo.toml").read_text().replace("runs_per_task = 2\n", ""))
            error = self.invoke("collect", "--config", config, "--manifest", manifest,
                                "--output", root / "collection", expected=2)
            self.assertIn("runs_per_task", error)
            self.assertFalse((root / "collection").exists())

    def test_generation_journal_retains_paid_parse_errors_and_resumes_without_repeating(self):
        from cave_mem_v3.agents.providers import OpenAIConfig, OpenAICompatibleProvider
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, collection = root / "manifest.json", root / "history"
            config = ROOT / "configs" / "phase1" / "demo.toml"
            self.invoke("manifest", "--catalog", ROOT / "configs" / "benchmarks" / "demo.json", "--output", manifest)
            self.invoke("collect", "--config", config, "--manifest", manifest, "--output", collection)
            calls = []
            candidate = {"step_index": 0, "mechanism": "missing preparation", "correction": {"type": "prepare"},
                         "positive_conditions": [{"field": "requires_preparation", "equals": True}],
                         "exclusion_conditions": [], "scope": "fixture only"}
            def transport(payload):
                calls.append(payload)
                text = json.dumps({"candidates": [candidate]}) if len(calls) == 1 else "invalid JSON"
                return {"model": "Qwen/Qwen2.5-7B-Instruct", "choices": [{"message": {"content": text}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 30}}
            provider = OpenAICompatibleProvider(OpenAIConfig(), transport=transport)
            output = root / "generated.json"
            with patch("cave_mem_v3.cli._teacher", return_value=provider):
                result = self.invoke("candidates", "--collection", collection, "--output", output,
                                     "--generate", "--config", config)
                self.assertEqual(result["generation_errors"], 1)
                saved = read_json(output)
                self.assertEqual(len(saved["candidates"]), 1)
                self.assertEqual(sum(g["usage"]["input_tokens"] for g in saved["generations"]), 200)
                output.unlink()  # Simulate loss of final aggregate after durable per-call entries.
                self.invoke("candidates", "--collection", collection, "--output", output,
                            "--generate", "--config", config)
                self.assertEqual(len(calls), 2)
            retry_calls = []
            def repaired_transport(payload):
                retry_calls.append(payload)
                return {"model": "Qwen/Qwen2.5-7B-Instruct",
                        "choices": [{"message": {"content": json.dumps({"candidates": [candidate]})}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 30}}
            retry_provider = OpenAICompatibleProvider(OpenAIConfig(max_output_tokens=16384),
                                                      transport=repaired_transport)
            recovered = root / "recovered.json"
            with patch("cave_mem_v3.cli._teacher", return_value=retry_provider):
                result = self.invoke("candidates", "--collection", collection, "--output", recovered,
                                     "--generate", "--config", config,
                                     "--reuse-successes-from", output)
            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["generation_errors"], 0)
            self.assertEqual(len(retry_calls), 1)
            self.assertEqual(len(read_json(recovered)["reuse_source"]["reused_trace_ids"]), 1)

    def test_automatic_generation_excludes_unparsed_actor_failure(self):
        from cave_mem_v3.agents.providers import OpenAIConfig, OpenAICompatibleProvider
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidates.json"
            trace = {"trace_id": "bad-action", "task_id": "regex-log", "run_hash": "source-run",
                     "trace_hash": "source-trace", "episode": {"success": False, "status": "invalid_action"}}
            provider = OpenAICompatibleProvider(OpenAIConfig(),
                transport=lambda _: self.fail("unparsed task failure must not be sent to teacher"))
            with patch("cave_mem_v3.cli._collection_traces", return_value=[trace]), \
                 patch("cave_mem_v3.cli._teacher", return_value=provider):
                result = self.invoke("candidates", "--collection", directory, "--output", output,
                                     "--generate", "--config", ROOT / "configs" / "phase1" / "demo.toml")
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(result["excluded_source_traces"], 1)
            self.assertEqual(read_json(output)["excluded_source_traces"][0]["trace_id"], "bad-action")

    def test_automatic_generation_excludes_scored_failure_without_executable_step(self):
        from test_candidates import source_trace
        from cave_mem_v3.common.contracts import digest
        from cave_mem_v3.agents.providers import OpenAIConfig, OpenAICompatibleProvider
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidates.json"
            trace = source_trace()
            episode = trace["episode"]
            episode["steps"][0].update(action={"type": "__unparsed_action__"},
                                       validation_error="invalid JSON", result=None)
            trace["trace_hash"] = digest({k: v for k, v in trace.items() if k != "trace_hash"})
            provider = OpenAICompatibleProvider(OpenAIConfig(),
                transport=lambda _: self.fail("no executable source step must not reach teacher"))
            with patch("cave_mem_v3.cli._collection_traces", return_value=[trace]), \
                 patch("cave_mem_v3.cli._teacher", return_value=provider):
                result = self.invoke("candidates", "--collection", directory, "--output", output,
                                     "--generate", "--config", ROOT / "configs" / "phase1" / "demo.toml")
            self.assertEqual(result["candidate_count"], 0)
            self.assertEqual(read_json(output)["excluded_source_traces"][0]["reason"],
                             "no_executable_source_step")


if __name__ == "__main__":
    unittest.main()
