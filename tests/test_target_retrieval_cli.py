import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.cli import main


class TargetRetrievalCliTests(unittest.TestCase):
    def test_requires_existing_local_encoder_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "retrieval.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main(["retrieve-target", "--target-run", "target-run", "--bank", "bank.json",
                             "--output", str(output), "--encoder-path", str(root / "missing-model"),
                             "--encoder-revision", "pinned-revision"])
            self.assertEqual(code, 2)
            self.assertIn("existing local embedding model", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_passes_local_encoder_identity_inputs_to_retrieval(self):
        class Encoder:
            def __init__(self, path, *, revision, device):
                self.path, self.revision, self.device = path, revision, device

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            stdout = io.StringIO()
            expected = {"retrieved": 1}
            with patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", Encoder), \
                    patch("cave_mem_v3.target_data.retrieval_collection.retrieve_target_run",
                          return_value=expected) as retrieve, \
                    contextlib.redirect_stdout(stdout):
                code = main(["retrieve-target", "--target-run", "target-run", "--bank", "bank.json",
                             "--output", str(root / "retrieval.json"), "--encoder-path", str(model),
                             "--encoder-revision", "pinned-revision", "--device", "cpu"])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue()), expected)
            encoder = retrieve.call_args.kwargs["encoder"]
            self.assertEqual(encoder.path, str(model.resolve()))
            self.assertEqual(encoder.revision, "pinned-revision")
            self.assertEqual(encoder.device, "cpu")


if __name__ == "__main__":
    unittest.main()
