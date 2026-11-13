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
            unset = root / "unset-confirmation.toml"
            unset.write_text(config.read_text().replace("confirmation_pairs = 16\n", ""))
            error = self.invoke("verify", "--config", unset, "--manifest", manifest,
                                "--collection", collection, "--candidates", candidates,
                                "--output", root / "not-started", expected=2)
            self.assertIn("confirmation_pairs", error)
            self.assertFalse((root / "not-started").exists())
            verification = root / "verification"
            self.invoke("verify", "--config", config, "--manifest", manifest, "--collection", collection,
                        "--candidates", candidates, "--output", verification)
            records = read_json(verification / "verification.json")["records"]
            self.assertTrue(all(record["evidence"]["v"] == 1 for record in records))
            self.assertTrue(all(record["evidence"]["plan"]["replay_spec"]["family_size"] == 2 for record in records))
            bank_path = root / "bank.json"
            result = self.invoke("bank", "--verification", verification, "--output", bank_path)
            self.assertEqual(result["verified_memories"], 2)
            bank = load_bank(bank_path)
            self.assertEqual(bank["memory_count"], 2)
            self.assertTrue(all(m["quality"]["v"] == 1 for m in bank["memories"]))
            self.invoke("bank", "--verification", verification, "--output", bank_path, expected=2)

    def test_unconfigured_server_is_reported_without_download_or_requests(self):
        result = self.invoke("check", "--config", ROOT / "configs" / "phase1" / "server.toml")
        self.assertFalse(result["ready"])
        self.assertIn("actor.base_url", result["pending"])
        self.assertIn("verification.confirmation_pairs", result["pending"])

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
            with patch("cave_mem_v3.cli._actor", return_value=SimpleNamespace(provider=provider)):
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


if __name__ == "__main__":
    unittest.main()
