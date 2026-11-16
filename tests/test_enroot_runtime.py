import tempfile
import unittest
from unittest.mock import patch

from cave_mem_v3.environments.enroot_runtime import EnrootRuntimeBackend


class EnrootRuntimeTimeoutTests(unittest.TestCase):
    def test_verifier_floor_extends_only_runtime_and_records_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = EnrootRuntimeBackend(directory)
            frozen = {"verifier_timeout": 600, "command_timeout": 60}
            seen = []

            def fake_verify(self, image, task, protocol):
                seen.append(protocol)
                return {"execution": {"exit_code": 0}}

            with patch("cave_mem_v3.environments.enroot.EnrootBackend.verify", fake_verify), \
                    patch.dict("os.environ", {"CAVE_VERIFIER_TIMEOUT_FLOOR_SECONDS": "1800"}):
                result = backend.verify("image", object(), frozen)
            self.assertEqual(frozen["verifier_timeout"], 600)
            self.assertEqual(seen[0]["verifier_timeout"], 1800)
            self.assertEqual(seen[0]["command_timeout"], 60)
            self.assertEqual(result["execution"]["effective_verifier_timeout_seconds"], 1800)
            self.assertEqual(result["execution"]["frozen_verifier_timeout_seconds"], 600)

    def test_invalid_floor_fails_before_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = EnrootRuntimeBackend(directory)
            with patch.dict("os.environ", {"CAVE_VERIFIER_TIMEOUT_FLOOR_SECONDS": "bad"}):
                with self.assertRaisesRegex(ValueError, "must be an integer"):
                    backend.verify("image", object(), {"verifier_timeout": 600})


if __name__ == "__main__":
    unittest.main()
