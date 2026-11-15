import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cave_mem_v3.common.contracts import PublicContext, digest
from cave_mem_v3.target_data.pre_treatment_state import build_pre_treatment_state
from cave_mem_v3.target_data.retrieval import _query_text


class FakeEncoder:
    identity = {"kind": "fake", "revision": "test-v1"}

    def encode_query(self, text):
        return [1.0, 0.0] if '"phase":"ready"' in text else [0.0, 1.0]


def context(**facts):
    return PublicContext("task-a", "Fix the project", "Package is ready.",
                         facts=facts, remaining_steps=7)


def memory(*, quality_v=1):
    return SimpleNamespace(
        memory_id="mem-1", source_task_id="history-a", source_trace_id="trace-a",
        source_checkpoint_hash="checkpoint-a", mechanism="Do not build before setup.",
        problem_action={"type": "shell", "command": "build"},
        correction={"type": "shell", "command": "setup"},
        positive_conditions=({"field": "phase", "equals": "ready"},),
        exclusion_conditions=({"field": "locked", "equals": True},),
        quality={"v": quality_v, "evidence_hash": "evidence-a", "n_replay_pairs": 2},
    )


def frozen_bank(raw_memories, sources=None):
    value = {"schema_version": "verified-source-bank.v3", "frozen": True,
             "memory_count": len(raw_memories), "memories": raw_memories,
             "audit_sources": sources or {}, "metadata": {}}
    value["bank_hash"] = digest(value)
    return value


def retrieval_result(status, bank, public_context, action, *, memory_id=None, score=None):
    result = {"status": status, "bank_hash": bank["bank_hash"],
              "selected_memory_id": memory_id, "similarity_score": score,
              "query_hash": digest(_query_text(public_context, action))}
    if status == "retrieved":
        result["candidates"] = [{"rank": 1, "memory_id": memory_id, "score": score}]
    else:
        result["candidates"] = []
    return result


def exposure_rules(*, exposed_command="build", not_exposed_command="inspect"):
    rule = {
        "schema_version": "action-exposure-rule.v1",
        "confidence": "high",
        "exposed_when": {"action": [
            {"field": "type", "equals": "shell"},
            {"field": "command", "equals": exposed_command}],
            "context": [{"field": "phase", "equals": "ready"}]},
        "not_exposed_when": {"action": [
            {"field": "type", "equals": "shell"},
            {"field": "command", "equals": not_exposed_command}],
            "context": [{"field": "phase", "equals": "ready"}]},
    }
    value = {"schema_version": "action-exposure-rules.v1", "frozen": True,
             "rules": {"mem-1": rule}}
    value["rules_hash"] = digest(value)
    return value


class PreTreatmentStateTests(unittest.TestCase):
    def test_verified_memory_recomputes_rho_and_keeps_e_distinct_unknown(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = context(phase="ready", locked=False)
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.91)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            result = build_pre_treatment_state(
                context=public_context,
                action=action,
                retrieval=retrieval, bank=bank,
                remaining_computation={"model_calls": 5, "environment_steps": 7})
        x = result["x"]
        self.assertEqual(x["v"], 1)
        self.assertEqual(x["rho"], 1)
        self.assertEqual(x["e"]["value"], "?")
        self.assertEqual(x["d"]["value"], "?")
        self.assertNotIn("task_id", x["context"])
        self.assertNotIn("evidence_hash", x["E"])
        self.assertEqual(result["task_id"], "task-a")
        self.assertEqual(x["b"]["remaining"]["model_calls"], 5)
        self.assertEqual(result["randomization_status"], "not_assigned")
        self.assertEqual(result["outcome_access"], "none")
        self.assertEqual(result["x_hash"], digest(x))

    def test_exclusion_predicate_vetoes_rho(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = context(phase="ready", locked=True)
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.5)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            result = build_pre_treatment_state(
                context=public_context,
                action=action,
                retrieval=retrieval, bank=bank)
        self.assertEqual(result["x"]["rho"], 0)

    def test_missing_public_fact_keeps_rho_unknown(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = PublicContext("task-a", "Fix", "Ready.")
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.5)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            result = build_pre_treatment_state(
                context=public_context,
                action=action,
                retrieval=retrieval, bank=bank)
        self.assertEqual(result["x"]["rho"], "?")

    def test_no_match_and_unknown_budget_are_explicit(self):
        bank = frozen_bank([])
        public_context = context()
        action = {"type": "shell", "command": "inspect"}
        retrieval = retrieval_result("no_match", bank, public_context, action)
        result = build_pre_treatment_state(
            context=public_context, action=action,
            retrieval=retrieval, bank=bank, encoder=FakeEncoder())
        self.assertEqual(result["status"], "prepared_no_memory")
        self.assertEqual(result["x"]["v"], "?")
        self.assertEqual(result["x"]["rho"], "?")
        self.assertEqual(result["x"]["b"]["status"], "unknown")
        self.assertEqual(result["x"]["feature_map"]["text_embedding"], [0.0, 1.0])

    def test_mismatched_bank_or_unverified_memory_fails_closed(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = context()
        action = {"type": "shell"}
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            build_pre_treatment_state(context=public_context, action=action,
                retrieval=retrieval_result("no_match", {**bank, "bank_hash": "other"},
                                           public_context, action), bank=bank)
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.5)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(quality_v="?"),)):
            with self.assertRaisesRegex(ValueError, "v=1"):
                build_pre_treatment_state(context=context(), action={"type": "shell"},
                                          retrieval=retrieval, bank=bank)

    def test_outcome_and_private_context_fields_are_excluded(self):
        bank = frozen_bank([])
        public_context = PublicContext(
            "task-a", "Fix the project", "Package is ready.",
            facts={"success": True, "metadata": {"secret": "hidden"}, "phase": "ready"},
            admissible_actions=("inspect", {"name": "unsafe", "metadata": {"success": True}}),
            remaining_steps=7)
        action = {"type": "shell", "command": "inspect"}
        retrieval = retrieval_result("no_match", bank, public_context, action)
        result = build_pre_treatment_state(
            context=public_context, action=action, retrieval=retrieval,
            bank=bank)
        facts = result["x"]["context"]["facts"]
        self.assertEqual(facts, {"phase": "ready"})
        self.assertNotIn("success", facts)
        self.assertEqual(result["x"]["context"]["admissible_actions"], ["inspect", {"name": "unsafe"}])

    def test_private_or_outcome_fields_in_cached_action_are_rejected(self):
        bank = frozen_bank([])
        retrieval = retrieval_result("no_match", bank, context(), {"type": "shell", "command": "inspect"})
        with self.assertRaisesRegex(ValueError, "private/outcome"):
            build_pre_treatment_state(
                context=context(),
                action={"type": "shell", "command": "inspect", "metadata": {"success": True}},
                retrieval=retrieval, bank=bank)

    def test_frozen_encoder_adds_text_feature_and_source_target_divergence(self):
        source_context = context(phase="ready")
        source_step = {"checkpoint_hash": "checkpoint-a", "context": source_context.__dict__,
                       "action": {"type": "shell", "command": "build"}}
        source = {"trace-a": {"trace_id": "trace-a", "episode": {"steps": [source_step]}}}
        bank = frozen_bank([{"memory_id": "mem-1"}], sources=source)
        public_context = context(phase="ready")
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.9)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            result = build_pre_treatment_state(
                context=public_context, action=action, retrieval=retrieval, bank=bank,
                encoder=FakeEncoder())
        self.assertEqual(result["x"]["feature_map"]["status"],
                         "frozen_text_encoder_plus_structured.v1")
        self.assertEqual(result["x"]["feature_map"]["encoder_identity"], FakeEncoder.identity)
        self.assertEqual(result["x"]["feature_map"]["text_embedding"], [1.0, 0.0])
        self.assertAlmostEqual(result["x"]["d"]["value"], 0.0)
        self.assertEqual(result["x"]["d"]["metric"],
                         "source-target-public-prefix-action-cosine-distance.v1")

    def test_explicit_exposure_rule_can_return_one_or_zero(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = context(phase="ready")
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.99)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            exposed = build_pre_treatment_state(
                context=public_context, action=action, retrieval=retrieval, bank=bank,
                exposure_rules=exposure_rules())
            safe_action = {"type": "shell", "command": "inspect"}
            safe_retrieval = retrieval_result("retrieved", bank, public_context, safe_action,
                                              memory_id="mem-1", score=.99)
            not_exposed = build_pre_treatment_state(
                context=public_context, action=safe_action, retrieval=safe_retrieval,
                bank=bank, exposure_rules=exposure_rules())
        self.assertEqual(exposed["x"]["e"]["value"], 1)
        self.assertEqual(exposed["x"]["e"]["confidence"], "high")
        self.assertEqual(not_exposed["x"]["e"]["value"], 0)

    def test_conflicting_or_incomplete_exposure_evidence_stays_unknown(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = context(phase="ready")
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.99)
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            conflict = build_pre_treatment_state(
                context=public_context, action=action, retrieval=retrieval, bank=bank,
                exposure_rules=exposure_rules(not_exposed_command="build"))
            incomplete = build_pre_treatment_state(
                context=PublicContext("task-a", "Fix", "Ready."), action=action,
                retrieval=retrieval_result("retrieved", bank, PublicContext("task-a", "Fix", "Ready."),
                                           action, memory_id="mem-1", score=.99),
                bank=bank, exposure_rules=exposure_rules())
        self.assertEqual(conflict["x"]["e"]["value"], "?")
        self.assertEqual(conflict["x"]["e"]["reason"], "exposure_rules_conflict")
        self.assertEqual(incomplete["x"]["e"]["value"], "?")
        self.assertEqual(incomplete["x"]["e"]["reason"], "exposure_rule_not_decisive")

    def test_invalid_exposure_rule_hash_is_rejected(self):
        bank = frozen_bank([{"memory_id": "mem-1"}])
        public_context = context(phase="ready")
        action = {"type": "shell", "command": "build"}
        retrieval = retrieval_result("retrieved", bank, public_context, action,
                                     memory_id="mem-1", score=.9)
        rules = exposure_rules()
        rules["rules"]["mem-1"]["confidence"] = "low"
        with patch("cave_mem_v3.target_data.pre_treatment_state.bank_memories",
                   return_value=(memory(),)):
            with self.assertRaisesRegex(ValueError, "exposure rules schema/hash"):
                build_pre_treatment_state(context=public_context, action=action,
                                          retrieval=retrieval, bank=bank,
                                          exposure_rules=rules)


if __name__ == "__main__":
    unittest.main()
