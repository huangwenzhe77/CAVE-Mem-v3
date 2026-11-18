import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cave_mem_v3.common.contracts import PublicContext, digest
from cave_mem_v3.target_data.retrieval import retrieve_memory


class FakeEncoder:
    identity = {"kind": "test", "revision": "fixture-v1"}

    def __init__(self, vectors=None):
        self.vectors = vectors or {}
        self.query = None
        self.documents = None

    def encode_query(self, text):
        self.query = text
        return [1.0, 0.0]

    def encode_documents(self, texts):
        self.documents = list(texts)
        return [self.vectors.get(text, [0.0, 1.0]) for text in texts]


def frozen_bank(raw_memories=None):
    bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
            "memory_count": len(raw_memories or []), "memories": raw_memories or [],
            "audit_sources": {}}
    bank["bank_hash"] = digest(bank)
    return bank


def memory(memory_id, mechanism, correction_secret):
    return SimpleNamespace(
        memory_id=memory_id,
        mechanism=mechanism,
        problem_action={"type": "shell", "command": "git status"},
        positive_conditions=({"field": "phase", "equals": "ready"},),
        exclusion_conditions=(),
        correction={"command": correction_secret},
        evidence={"private_marker": correction_secret},
    )


class TargetMemoryRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.context = PublicContext(
            "target-1", "Repair the project safely.", "The build step is ready.",
            facts={"phase": "ready", "success": True, "snapshot": {"secret": "hidden"},
                   "metadata": {"private_marker": "hidden"}, "future_outcome": "hidden",
                   "source_trace_id": "private-source-id"},
            history=({"action": {"type": "shell", "command": "git status"},
                      "observation": "working tree clean", "private_verifier": "hidden"},),
            admissible_actions=("shell",), remaining_steps=5)
        self.candidates = [memory("mem-b", "mechanism-b", "do not expose b"),
                           memory("mem-a", "mechanism-a", "do not expose a")]
        self.bank = frozen_bank([{"memory_id": "placeholder-b"},
                                 {"memory_id": "placeholder-a"}])

    def test_ranks_by_cosine_and_logs_bank_encoder_and_query_identity(self):
        texts = [
            '{"exclusion_conditions":[],"mechanism":"mechanism-b",'
            '"positive_conditions":[{"equals":"ready","field":"phase"}],'
            '"problem_action":{"command":"git status","type":"shell"}}',
            '{"exclusion_conditions":[],"mechanism":"mechanism-a",'
            '"positive_conditions":[{"equals":"ready","field":"phase"}],'
            '"problem_action":{"command":"git status","type":"shell"}}',
        ]
        encoder = FakeEncoder({texts[0]: [0.0, 1.0], texts[1]: [0.8, 0.6]})
        with patch("cave_mem_v3.target_data.retrieval.bank_memories", return_value=self.candidates):
            result = retrieve_memory(self.bank, context=self.context,
                                     action={"type": "shell", "command": "git status"},
                                     encoder=encoder)
        self.assertEqual(result["status"], "retrieved")
        self.assertEqual(result["selected_memory_id"], "mem-a")
        self.assertAlmostEqual(result["similarity_score"], 0.8)
        self.assertEqual([row["memory_id"] for row in result["candidates"]], ["mem-a", "mem-b"])
        self.assertEqual(result["bank_hash"], self.bank["bank_hash"])
        self.assertEqual(result["encoder_identity"], encoder.identity)
        self.assertNotIn('"success"', encoder.query)
        self.assertNotIn("hidden", encoder.query)
        self.assertNotIn("private-source-id", encoder.query)
        self.assertIn('"a0"', encoder.query)
        for document in encoder.documents:
            self.assertNotIn("do not expose", document)

    def test_ties_are_broken_by_memory_id(self):
        encoder = FakeEncoder()
        with patch("cave_mem_v3.target_data.retrieval.bank_memories",
                   return_value=[self.candidates[0], self.candidates[1]]):
            result = retrieve_memory(self.bank, context=self.context, action={"type": "finish"},
                                     encoder=encoder)
        self.assertEqual([row["memory_id"] for row in result["candidates"]], ["mem-a", "mem-b"])

    def test_public_alfworld_family_beats_higher_cross_family_cosine(self):
        context = PublicContext("target-1", "Heat a mug.", "A fridge is closed.",
                                facts={"benchmark": "alfworld-text",
                                       "task_type": "pick_heat_then_place_in_recep"})
        cool = memory("mem-cool", "cooling", "secret")
        cool.positive_conditions = ({"field": "task_type", "equals": "pick_cool_then_place_in_recep"},)
        heat = memory("mem-heat", "heating", "secret")
        heat.positive_conditions = ({"field": "task_type", "equals": "pick_heat_then_place_in_recep"},)
        encoder = FakeEncoder()

        def documents(texts):
            return [[1.0, 0.0] if '"cooling"' in text else [0.8, 0.6] for text in texts]

        encoder.encode_documents = documents
        with patch("cave_mem_v3.target_data.retrieval.bank_memories", return_value=[cool, heat]):
            result = retrieve_memory(self.bank, context=context, action={"type": "text", "command": "look"},
                                     encoder=encoder)
        self.assertEqual(result["selected_memory_id"], "mem-heat")
        self.assertEqual([row["memory_id"] for row in result["candidates"]], ["mem-heat", "mem-cool"])
        self.assertAlmostEqual(result["similarity_score"], 0.8)

    def test_applicable_same_family_memory_beats_higher_cosine(self):
        context = PublicContext("target-1", "Heat a mug.", "The fridge is closed.",
                                facts={"benchmark": "alfworld-text",
                                       "task_type": "pick_heat_then_place_in_recep"})
        inapplicable = memory("mem-high", "high score", "secret")
        inapplicable.positive_conditions = (
            {"field": "task_type", "equals": "pick_heat_then_place_in_recep"},
            {"source": "observation", "regex": "microwave is open"})
        applicable = memory("mem-low", "lower score", "secret")
        applicable.positive_conditions = (
            {"field": "task_type", "equals": "pick_heat_then_place_in_recep"},
            {"source": "observation", "regex": "fridge is closed"})
        encoder = FakeEncoder()
        encoder.encode_documents = lambda texts: [
            [1.0, 0.0] if '"high score"' in text else [0.8, 0.6] for text in texts]
        with patch("cave_mem_v3.target_data.retrieval.bank_memories",
                   return_value=[inapplicable, applicable]):
            result = retrieve_memory(self.bank, context=context,
                                     action={"type": "text", "command": "look"}, encoder=encoder)
        self.assertEqual(result["selected_memory_id"], "mem-low")

    def test_empty_verified_bank_returns_no_match(self):
        empty = frozen_bank()
        result = retrieve_memory(empty, context=self.context, action={"type": "finish"},
                                 encoder=FakeEncoder())
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["reason"], "empty_verified_bank")
        self.assertIsNone(result["selected_memory_id"])

    def test_tampered_bank_hash_is_rejected_before_memory_access(self):
        changed = dict(self.bank)
        changed["metadata"] = {"edited": True}
        with patch("cave_mem_v3.target_data.retrieval.bank_memories") as validate:
            with self.assertRaisesRegex(ValueError, "checksum"):
                retrieve_memory(changed, context=self.context, action={"type": "finish"},
                                encoder=FakeEncoder())
        validate.assert_not_called()

    def test_invalid_embedding_shapes_and_values_fail_closed(self):
        for bad_query, bad_docs, error in (
                ([], [[1.0]], "cannot be empty"),
                ([1.0, 0.0], [[1.0]], "dimensions differ"),
                ([float("nan"), 0.0], [[1.0, 0.0]], "finite numbers")):
            class InvalidEncoder(FakeEncoder):
                def encode_query(self, text):
                    return bad_query

                def encode_documents(self, texts):
                    return bad_docs

            with self.subTest(error=error), patch(
                    "cave_mem_v3.target_data.retrieval.bank_memories", return_value=[self.candidates[0]]):
                with self.assertRaisesRegex(ValueError, error):
                    retrieve_memory(self.bank, context=self.context, action={"type": "finish"},
                                    encoder=InvalidEncoder())


if __name__ == "__main__":
    unittest.main()
