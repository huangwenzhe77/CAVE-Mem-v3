import tempfile
import unittest
from pathlib import Path

from cave_mem_v3.environments.enroot import EnrootBackend
from cave_mem_v3.environments.terminal import DEFAULT_PROTOCOL


class EnrootCommandTests(unittest.TestCase):
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
