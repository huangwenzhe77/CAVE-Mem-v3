import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.common.contracts import digest, write_json
from cave_mem_v3.target_data.retrieval_collection import retrieve_target_run


class RetrievalCollectionTests(unittest.TestCase):
    def fixture(self, root, *, sampling_status="selected"):
        target = root / "target"
        (target / "traces").mkdir(parents=True)
        run = {"schema_version": "target-run.v1", "role": "target_train",
               "manifest_hash": "manifest-hash", "task_order": ["task-a"]}
        run["run_hash"] = digest(run)
        write_json(target / "run.json", run)
        checkpoint = {"state": "at-checkpoint"}
        action = {"type": "shell", "command": "git status"}
        context = {"task_id": "task-a", "instruction": "Inspect repo.",
                   "observation": "Repo ready.", "facts": {}}
        sample = {"status": sampling_status,
                  "selected_step_index": 0 if sampling_status == "selected" else None,
                  "selected_checkpoint_hash": digest(checkpoint) if sampling_status == "selected" else None,
                  "selected_checkpoint": checkpoint if sampling_status == "selected" else None,
                  "selected_public_context": context,
                  "selected_action": action if sampling_status == "selected" else None}
        episode = {"checkpoint_sampling": sample,
                   "steps": [{"index": 0, "checkpoint_hash": digest(checkpoint),
                              "context": context, "action": action}]}
        trace = {"schema_version": "target-trace.v1", "trace_id": "trace-a",
                 "task_id": "task-a", "role": "target_train", "run_hash": run["run_hash"],
                 "run_index": 0, "manifest_hash": run["manifest_hash"], "episode": episode}
        trace["trace_hash"] = digest(trace)
        write_json(target / "traces" / "000000-trace-a.json", trace)
        summary = {"schema_version": "target-summary.v1", "run_hash": run["run_hash"],
                   "manifest_hash": run["manifest_hash"], "role": run["role"],
                   "planned_rollouts": 1, "attempted_rollouts": 1,
                   "summary_hash": "placeholder", "trace_ids": ["trace-a"]}
        summary["summary_hash"] = digest({key: value for key, value in summary.items()
                                           if key != "summary_hash"})
        write_json(target / "summary.json", summary)
        bank_file = root / "bank.json"
        bank_file.write_text("unused")
        return target, bank_file

    def test_selected_checkpoint_is_retrieved_and_bound_to_trace_and_bank(self):
        class Encoder:
            identity = {"model": "fake-encoder", "revision": "test"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, bank_file = self.fixture(root)
            output = root / "retrieval.json"
            bank = {"bank_hash": "frozen-bank"}
            result = {"status": "retrieved", "bank_hash": "frozen-bank",
                      "encoder_identity": Encoder.identity, "candidates": [
                          {"rank": 1, "memory_id": "mem-a", "score": 0.82}],
                      "selected_memory_id": "mem-a", "similarity_score": 0.82}
            with patch("cave_mem_v3.target_data.retrieval_collection.load_bank", return_value=bank), \
                    patch("cave_mem_v3.target_data.retrieval_collection.retrieve_memory", return_value=result) as retrieve:
                summary = retrieve_target_run(target, bank_file, output, encoder=Encoder())
            self.assertEqual(summary["retrieved"], 1)
            self.assertEqual(retrieve.call_args.kwargs["action"], {"type": "shell", "command": "git status"})
            artifact = json.loads(output.read_text(encoding="utf-8"))
            record = artifact["records"][0]
            self.assertEqual(record["selected_memory_id"], "mem-a")
            self.assertEqual(artifact["bank_hash"], "frozen-bank")

    def test_no_selected_checkpoint_is_recorded_without_retrieval(self):
        class Encoder:
            identity = {"model": "fake-encoder", "revision": "test"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, bank_file = self.fixture(root, sampling_status="no_selection")
            with patch("cave_mem_v3.target_data.retrieval_collection.load_bank",
                       return_value={"bank_hash": "frozen-bank"}), \
                    patch("cave_mem_v3.target_data.retrieval_collection.retrieve_memory") as retrieve:
                summary = retrieve_target_run(target, bank_file, root / "retrieval.json", encoder=Encoder())
            self.assertEqual(summary["not_retrieved"], 1)
            retrieve.assert_not_called()

    def test_selected_action_mismatch_is_not_retrieved(self):
        class Encoder:
            identity = {"model": "fake-encoder", "revision": "test"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, bank_file = self.fixture(root)
            trace_path = target / "traces" / "000000-trace-a.json"
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            trace["episode"]["checkpoint_sampling"]["selected_action"] = {"type": "finish"}
            trace.pop("trace_hash")
            trace["trace_hash"] = digest(trace)
            write_json(trace_path, trace)
            with patch("cave_mem_v3.target_data.retrieval_collection.load_bank",
                       return_value={"bank_hash": "frozen-bank"}), \
                    patch("cave_mem_v3.target_data.retrieval_collection.retrieve_memory") as retrieve:
                summary = retrieve_target_run(target, bank_file, root / "retrieval.json", encoder=Encoder())
            self.assertEqual(summary["invalid_selection"], 1)
            retrieve.assert_not_called()

    def test_incomplete_target_run_is_rejected_before_writing_artifact(self):
        class Encoder:
            identity = {"model": "fake-encoder", "revision": "test"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, bank_file = self.fixture(root)
            summary_path = target / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["planned_rollouts"] = 2
            summary.pop("summary_hash")
            summary["summary_hash"] = digest(summary)
            write_json(summary_path, summary)
            output = root / "retrieval.json"
            with patch("cave_mem_v3.target_data.retrieval_collection.load_bank",
                       return_value={"bank_hash": "frozen-bank"}), \
                    self.assertRaisesRegex(ValueError, "run is incomplete"):
                retrieve_target_run(target, bank_file, output, encoder=Encoder())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
