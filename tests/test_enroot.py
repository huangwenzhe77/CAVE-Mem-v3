import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cave_mem_v3.environments.enroot import EnrootBackend
from cave_mem_v3.environments.terminal import DEFAULT_PROTOCOL, TerminalInfrastructureError


class EnrootCommandTests(unittest.TestCase):
    def test_verifier_error_keeps_private_raw_diagnostics_without_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            backend = EnrootBackend(root / "snapshots")
            backend.base_image = "sha256:" + "a" * 64
            task = SimpleNamespace(task_id="fix-code-vulnerability", task_dir=root,
                                   asset_hash="pinned-assets", spec={"task_config": {"verifier": {}}},
                                   verify_assets=lambda: None)
            def fake_start(image, command, protocol, **kwargs):
                logs = kwargs["logs"]
                (logs / "reward.txt").write_text("0\n", encoding="utf-8")
                (logs / "ctrf.json").write_text(json.dumps({"results": {"tests": [],
                    "summary": {"tests": 0, "passed": 0, "failed": 0}}}), encoding="utf-8")
                (logs / "original-repo-ctrf.json").write_text("{}", encoding="utf-8")
                (kwargs["diagnostic_dir"] / "verifier.stdout").write_text("full collection error", encoding="utf-8")
                (kwargs["diagnostic_dir"] / "verifier.stderr").write_text("hook warning", encoding="utf-8")
                return {"exit_code": 0, "stdout": "truncated", "stderr": "hook warning", "truncated": True}
            backend._start = fake_start
            backend._run = lambda *args, **kwargs: {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False}
            with self.assertRaisesRegex(TerminalInfrastructureError, "private verifier diagnostic"):
                backend.verify("sha256:" + "b" * 64, task, DEFAULT_PROTOCOL)
            saved = list((root / "snapshots" / "verifier-diagnostics").iterdir())
            self.assertEqual(len(saved), 1)
            self.assertEqual((saved[0] / "logs/reward.txt").read_text().strip(), "0")
            self.assertEqual((saved[0] / "logs/original-repo-ctrf.json").read_text(), "{}")
            self.assertEqual((saved[0] / "verifier.stdout").read_text(), "full collection error")
            metadata = json.loads((saved[0] / "metadata.json").read_text())
            self.assertEqual(metadata["outcome"], "error")
            self.assertEqual(metadata["task_id"], task.task_id)
            self.assertEqual(metadata["asset_hash"], task.asset_hash)

    def test_preserved_verifier_stdout_is_not_truncated_with_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = EnrootBackend(directory, executable=sys.executable)
            saved = Path(directory) / "raw"
            saved.mkdir()
            report = backend._run(["-c", "import sys; print('x'*100); print('warning', file=sys.stderr)"],
                                  limit=10, preserve_output_dir=saved)
            self.assertTrue(report["truncated"])
            self.assertEqual(len(report["stdout"]), 10)
            self.assertIn("x" * 100, (saved / "verifier.stdout").read_text())
            self.assertIn("warning", (saved / "verifier.stderr").read_text())

    def test_timeout_keeps_partial_verifier_output(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = EnrootBackend(directory, executable=sys.executable)
            saved = Path(directory) / "raw"
            saved.mkdir()
            with self.assertRaisesRegex(TerminalInfrastructureError, "timed out"):
                backend._run(["-c", "import sys,time; print('before timeout', flush=True); time.sleep(2)"],
                             timeout=.1, preserve_output_dir=saved)
            self.assertIn("before timeout", (saved / "verifier.stdout").read_text())

    def test_diagnostic_copy_failure_does_not_hide_primary_verifier_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tests").mkdir()
            (root / "tests" / "test.sh").write_text("#!/bin/bash\n", encoding="utf-8")
            backend = EnrootBackend(root / "snapshots")
            task = SimpleNamespace(task_id="fix-code-vulnerability", task_dir=root,
                                   asset_hash="pinned-assets", spec={"task_config": {"verifier": {}}},
                                   verify_assets=lambda: None)
            def fake_start(image, command, protocol, **kwargs):
                (kwargs["logs"] / "reward.txt").write_text("0\n", encoding="utf-8")
                (kwargs["logs"] / "ctrf.json").write_text("{}", encoding="utf-8")
                return {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False}
            backend._start = fake_start
            backend._run = lambda *args, **kwargs: {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False}
            with patch("cave_mem_v3.environments.enroot.shutil.copyfile", side_effect=PermissionError("unreadable")):
                with self.assertRaisesRegex(TerminalInfrastructureError,
                                            "invalid/incomplete CTRF.*diagnostic preservation failed: PermissionError"):
                    backend.verify("sha256:" + "b" * 64, task, DEFAULT_PROTOCOL)

    def test_actor_has_no_verifier_mount_and_snapshots_tmp(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = EnrootBackend(directory)
            backend.base_image = "sha256:" + "a" * 64
            backend.base_path = Path(directory) / "base.sqsh"
            backend.base_path.touch()
            calls = []

            def fake_run(args, **kwargs):
                calls.append(args)
                return {"exit_code": 0, "stdout": "", "stderr": "", "truncated": False}

            backend._run = fake_run
            backend._start(backend.base_image, "printf '%s' hello > /tmp/result", DEFAULT_PROTOCOL,
                           name="cave-test")
            start = calls[-1]
            self.assertIn("--net", start)
            self.assertNotIn("--mount", start)
            self.assertIn("cp -a /tmp/. /app/.cave-tmp/", start[-1])
            self.assertIn("cd -- /app", start[-1])


if __name__ == "__main__":
    unittest.main()
