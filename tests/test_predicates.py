import unittest

from cave_mem_v3.common.contracts import PublicContext
from cave_mem_v3.memory_generation.predicates import evaluate_conditions, evaluate_predicate, validate_predicate


class PredicateTests(unittest.TestCase):
    def setUp(self):
        self.context = PublicContext("task", "Prepare before operating", "prepared=false; version=2",
            {"prepared": False, "version": 2, "package": {"name": "example"}},
            ({"action": {"type": "inspect"}, "observation": "old state", "private_verifier": "SECRET"},))

    def test_public_comparisons_and_typed_values(self):
        self.assertEqual(evaluate_predicate({"field": "prepared", "equals": False}, self.context), 1)
        self.assertEqual(evaluate_predicate({"field": "version", "in": [1, 2]}, self.context), 1)
        self.assertEqual(evaluate_predicate({"field": "package.name", "not_equals": "other"}, self.context), 1)
        self.assertEqual(evaluate_predicate({"field": "prepared", "equals": 0}, self.context), 0)

    def test_unknown_is_not_false_and_exists_means_observed(self):
        self.assertEqual(evaluate_predicate({"field": "installed", "equals": True}, self.context), "?")
        self.assertEqual(evaluate_predicate({"field": "installed", "not_equals": True}, self.context), "?")
        self.assertEqual(evaluate_predicate({"field": "installed", "exists": False}, self.context), 1)
        self.assertEqual(evaluate_predicate({"field": "version", "exists": True}, self.context), 1)
        self.assertEqual(evaluate_conditions([{"field": "version", "equals": 2}],
                                            [{"field": "installed", "equals": True}], self.context), "?")
        self.assertEqual(evaluate_conditions([{"field": "version", "equals": 1}], [], self.context), 0)

    def test_regex_reads_only_public_text(self):
        self.assertEqual(evaluate_predicate({"source": "observation", "regex": "version=2"}, self.context), 1)
        self.assertEqual(evaluate_predicate({"source": "instruction", "regex": "Prepare"}, self.context), 1)
        self.assertEqual(evaluate_predicate({"source": "history", "regex": "old state"}, self.context), 1)
        self.assertEqual(evaluate_predicate({"source": "history", "regex": "SECRET"}, self.context), 0)

    def test_code_narrative_and_private_fields_are_rejected(self):
        bad = [{"observation": "probably applicable"}, {"eval": "__import__('os')"},
               {"field": "score", "equals": 1}, {"field": "nested.private_verifier", "exists": True},
               {"source": "checkpoint", "regex": "anything"},
               {"field": "prepared", "equals": True, "not_equals": False},
               {"source": "observation", "regex": "["}]
        for predicate in bad:
            with self.subTest(predicate=predicate), self.assertRaises(ValueError):
                validate_predicate(predicate)
        with self.assertRaises(TypeError):
            evaluate_predicate({"field": "prepared", "exists": True}, {"facts": {"prepared": True}})


if __name__ == "__main__":
    unittest.main()
