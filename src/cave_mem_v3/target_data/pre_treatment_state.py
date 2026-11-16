"""Outcome-blind construction of the structured pre-treatment state."""
from __future__ import annotations

import copy
import math
import re

from cave_mem_v3.common.contracts import PublicContext, digest, plain
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.predicates import evaluate_conditions, evaluate_predicate
from cave_mem_v3.target_data.retrieval import _query_text

_PRIVATE_CONTEXT_KEYS = frozenset({
    "success", "terminal_success", "future_success", "future_outcome", "outcome",
    "reward", "score", "label", "target", "targets", "verifier",
    "verifier_answer", "private_verifier", "private", "hidden_state",
    "environment_state", "snapshot", "checkpoint", "checkpoint_id", "rng",
    "actor_state", "metadata", "terminal", "failure_label", "source_task_id",
    "source_trace_id",
})


def _public_value(value):
    if isinstance(value, dict):
        return {key: _public_value(item) for key, item in value.items()
                if isinstance(key, str) and key.lower() not in _PRIVATE_CONTEXT_KEYS
                and not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    return copy.deepcopy(value)


def _context(value: PublicContext | dict) -> PublicContext:
    if isinstance(value, PublicContext):
        value = plain(value)
    if not isinstance(value, dict):
        raise TypeError("context must be a public PublicContext or mapping")
    allowed = {key: copy.deepcopy(value[key]) for key in (
        "task_id", "instruction", "observation", "facts", "history",
        "admissible_actions", "remaining_steps") if key in value}
    if not isinstance(allowed.get("task_id"), str) or not allowed["task_id"]:
        raise ValueError("public context requires task_id")
    if not isinstance(allowed.get("instruction"), str) or not isinstance(allowed.get("observation"), str):
        raise ValueError("public context requires instruction and observation text")
    allowed["facts"] = _public_value(allowed.get("facts", {}))
    allowed["admissible_actions"] = _public_value(allowed.get("admissible_actions", ()))
    allowed["history"] = [
        {key: _public_value(event[key]) for key in ("action", "observation") if key in event}
        for event in allowed.get("history", []) if isinstance(event, dict)
    ]
    return PublicContext(**allowed)


def _public_action(action: dict) -> dict:
    if not isinstance(action, dict) or not action:
        raise ValueError("cached unassisted action a0 must be a nonempty mapping")
    def contains_private_key(value):
        if isinstance(value, dict):
            return any(
                not isinstance(key, str) or key.lower() in _PRIVATE_CONTEXT_KEYS
                or key.startswith("_") or contains_private_key(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_private_key(item) for item in value)
        return False

    if contains_private_key(action):
        raise ValueError("cached action contains private/outcome fields")
    return copy.deepcopy(action)


def _remaining_budget(value: dict | None) -> dict:
    if value is None:
        return {"status": "unknown", "reason": "remaining_computation_not_supplied"}
    if not isinstance(value, dict) or not value:
        raise ValueError("remaining_computation must be a nonempty mapping")
    normalized = {}
    for name, amount in value.items():
        if not isinstance(name, str) or not name:
            raise ValueError("remaining computation keys must be nonempty strings")
        if type(amount) is not int or amount < 0:
            raise ValueError("remaining computation values must be nonnegative integers")
        normalized[name] = amount
    return {"status": "known", "remaining": normalized}


def _unit_vector(value, name: str) -> tuple[list[float], float]:
    try:
        vector = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} embedding must be a numeric vector") from exc
    if not vector:
        raise ValueError(f"{name} embedding cannot be empty")
    normalized = []
    for component in vector:
        if isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(component):
            raise ValueError(f"{name} embedding must contain finite numeric values")
        normalized.append(float(component))
    norm = math.sqrt(sum(component * component for component in normalized))
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError(f"{name} embedding must have a finite nonzero norm")
    return normalized, norm


def _source_checkpoint(bank: dict, memory) -> tuple[PublicContext, dict] | None:
    sources = bank.get("audit_sources", {})
    if not isinstance(sources, dict):
        return None
    matches = []
    for trace in sources.values():
        episode = trace.get("episode", {}) if isinstance(trace, dict) else {}
        for step in episode.get("steps", []) if isinstance(episode, dict) else []:
            if (isinstance(step, dict)
                    and step.get("checkpoint_hash") == memory.source_checkpoint_hash
                    and isinstance(step.get("context"), dict)
                    and isinstance(step.get("action"), dict)):
                matches.append((_context(step["context"]), _public_action(step["action"])))
    if len(matches) != 1:
        return None
    return matches[0]


def _conditions_match(conditions, *, context: PublicContext, action: dict) -> int | str:
    if not isinstance(conditions, dict) or set(conditions) != {"action", "context"}:
        raise ValueError("exposure condition groups require action and context arrays")
    action_conditions = conditions["action"]
    context_conditions = conditions["context"]
    if (not isinstance(action_conditions, list) or not action_conditions
            or not isinstance(context_conditions, list)):
        raise ValueError("exposure rules need action predicates and a context predicate array")
    values = []
    for predicate in action_conditions:
        if not isinstance(predicate, dict) or set(predicate) - {"field", "equals", "not_equals", "in", "exists"}:
            raise ValueError("action exposure predicates support field equality, inequality, in, or exists")
        field = predicate.get("field")
        operators = set(predicate) - {"field"}
        if (not isinstance(field, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", field) is None
                or not operators or len(operators) != 1):
            raise ValueError("action exposure predicate requires a safe field and one operator")
        if any(part.lower() in _PRIVATE_CONTEXT_KEYS or part.startswith("_") for part in field.split(".")):
            raise ValueError("action exposure predicate cannot read private fields")
        found, value = True, action
        for part in field.split("."):
            if not isinstance(value, dict) or part not in value:
                found = False
                break
            value = value[part]
        operator = next(iter(operators))
        if operator == "exists":
            expected = predicate[operator]
            if type(expected) is not bool:
                raise ValueError("action exists predicate requires a boolean")
            values.append(int(found == expected))
        elif not found:
            values.append("?")
        elif operator == "equals":
            values.append(int(type(value) is type(predicate[operator]) and value == predicate[operator]))
        elif operator == "not_equals":
            values.append(int(not (type(value) is type(predicate[operator]) and value == predicate[operator])))
        elif operator == "in":
            expected = predicate[operator]
            if not isinstance(expected, list) or not expected:
                raise ValueError("action in predicate requires a nonempty list")
            values.append(int(any(type(value) is type(item) and value == item for item in expected)))
        else:
            raise ValueError("unsupported action exposure predicate operator")
    for predicate in context_conditions:
        values.append(evaluate_predicate(predicate, context))
    if 0 in values:
        return 0
    if "?" in values:
        return "?"
    return 1


def _exposure(*, memory_id: str | None, context: PublicContext, action: dict,
              rules: dict | None) -> dict:
    if memory_id is None:
        return {"value": "?", "confidence": "?", "reason": "no_retrieved_memory"}
    if rules is None:
        return {"value": "?", "confidence": "?", "reason": "no_frozen_action_mechanism_exposure_rubric"}
    if (not isinstance(rules, dict)
            or set(rules) != {"schema_version", "frozen", "rules", "rules_hash"}
            or rules.get("schema_version") != "action-exposure-rules.v1"
            or rules.get("frozen") is not True
            or rules.get("rules_hash") != digest({key: value for key, value in rules.items() if key != "rules_hash"})
            or not isinstance(rules.get("rules"), dict)):
        raise ValueError("frozen action exposure rules schema/hash is invalid")
    rule = rules["rules"].get(memory_id)
    if rule is None:
        return {"value": "?", "confidence": "?", "reason": "no_rule_for_memory",
                "rules_hash": rules["rules_hash"]}
    if (not isinstance(rule, dict)
            or set(rule) != {"schema_version", "confidence", "exposed_when", "not_exposed_when"}
            or rule.get("schema_version") != "action-exposure-rule.v1"
            or rule.get("confidence") not in ("low", "medium", "high")):
        raise ValueError("action exposure rule requires schema version and declared confidence")
    exposed = _conditions_match(rule.get("exposed_when"), context=context, action=action)
    not_exposed = _conditions_match(rule.get("not_exposed_when"), context=context, action=action)
    if exposed == 1 and not_exposed != 1:
        value, reason = 1, "exposed_rule_matched"
    elif not_exposed == 1 and exposed != 1:
        value, reason = 0, "not_exposed_rule_matched"
    elif exposed == 1 and not_exposed == 1:
        value, reason = "?", "exposure_rules_conflict"
    else:
        value, reason = "?", "exposure_rule_not_decisive"
    return {"value": value, "confidence": rule["confidence"], "reason": reason,
            "rule_hash": digest(rule), "rules_hash": rules["rules_hash"],
            "rule_schema_version": rule["schema_version"]}


def validate_exposure_rules(rules: dict | None) -> None:
    """Validate a frozen rule artifact before any model request is made."""
    if rules is None:
        return
    if (not isinstance(rules, dict)
            or set(rules) != {"schema_version", "frozen", "rules", "rules_hash"}
            or rules.get("schema_version") != "action-exposure-rules.v1"
            or rules.get("frozen") is not True
            or rules.get("rules_hash") != digest({key: value for key, value in rules.items() if key != "rules_hash"})
            or not isinstance(rules.get("rules"), dict)):
        raise ValueError("frozen action exposure rules schema/hash is invalid")
    for memory_id, rule in rules["rules"].items():
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("exposure rule keys must be nonempty memory IDs")
        if (not isinstance(rule, dict)
                or set(rule) != {"schema_version", "confidence", "exposed_when", "not_exposed_when"}
                or rule.get("schema_version") != "action-exposure-rule.v1"
                or rule.get("confidence") not in ("low", "medium", "high")):
            raise ValueError("action exposure rule requires schema version and declared confidence")
        _conditions_match(rule["exposed_when"], context=PublicContext("validation", "validate", ""), action={})
        _conditions_match(rule["not_exposed_when"], context=PublicContext("validation", "validate", ""), action={})


def _divergence(*, context: PublicContext, action: dict, memory,
                bank: dict, encoder) -> tuple[dict, list[float] | None]:
    if encoder is None:
        return {"value": "?", "reason": "frozen_text_encoder_not_configured"}, None
    if not isinstance(getattr(encoder, "identity", None), dict) or not encoder.identity:
        raise ValueError("frozen feature encoder requires a nonempty identity")
    if not callable(getattr(encoder, "encode_query", None)):
        raise TypeError("frozen feature encoder must expose encode_query")
    target_text = _query_text(context, action)
    target_vector, target_norm = _unit_vector(encoder.encode_query(target_text), "target")
    if memory is None:
        return {"value": "?", "reason": "no_retrieved_memory"}, target_vector
    source = _source_checkpoint(bank, memory)
    if source is None:
        return {"value": "?", "reason": "source_checkpoint_context_unavailable"}, target_vector
    source_context, source_action = source
    source_text = _query_text(source_context, source_action)
    source_vector, source_norm = _unit_vector(encoder.encode_query(source_text), "source")
    if len(source_vector) != len(target_vector):
        raise ValueError("source and target encoder dimensions differ")
    cosine = sum(left * right for left, right in zip(source_vector, target_vector)) / (source_norm * target_norm)
    cosine = max(-1.0, min(1.0, cosine))
    return {
        "value": 1.0 - cosine,
        "metric": "source-target-public-prefix-action-cosine-distance.v1",
        "source_query_hash": digest(source_text),
        "target_query_hash": digest(target_text),
        "interpretation": "continuous divergence feature; no threshold or applicability claim",
    }, target_vector


def build_pre_treatment_state(*, context: PublicContext | dict, action: dict,
                              retrieval: dict, bank: dict,
                              remaining_computation: dict | None = None,
                              encoder=None, exposure_rules: dict | None = None) -> dict:
    """Freeze structured Eq. (2) inputs without assigning Silent/Remind.

    The function has no episode/result argument. ``e`` is computed only from a
    frozen per-memory typed rule. ``d`` is a continuous source/target public
    prefix-action cosine distance when the source checkpoint is available; it
    carries no threshold or applicability interpretation.
    """
    if not isinstance(bank, dict):
        raise TypeError("bank must be a frozen source-memory bank")
    if not isinstance(retrieval, dict):
        raise TypeError("retrieval must be a retrieval result mapping")
    if encoder is not None and (not isinstance(getattr(encoder, "identity", None), dict)
                                or not encoder.identity
                                or not callable(getattr(encoder, "encode_query", None))):
        raise ValueError("frozen feature encoder requires identity and encode_query")
    context = _context(context)
    action = _public_action(action)
    bank_hash = bank.get("bank_hash")
    if not isinstance(bank_hash, str) or retrieval.get("bank_hash") != bank_hash:
        raise ValueError("retrieval result and memory bank hashes differ")
    if (bank.get("schema_version") != "verified-source-bank.v3"
            or bank.get("frozen") is not True
            or bank_hash != digest({key: value for key, value in bank.items() if key != "bank_hash"})):
        raise ValueError("frozen memory bank schema/hash is invalid")
    expected_query_hash = digest(_query_text(context, action))
    if retrieval.get("query_hash") != expected_query_hash:
        raise ValueError("retrieval query hash does not match this checkpoint and a0")

    retrieval_status = retrieval.get("status")
    if retrieval_status not in ("retrieved", "no_match"):
        raise ValueError("only completed retrieved/no_match results can build state")
    raw_memories = bank.get("memories")
    if not isinstance(raw_memories, list):
        raise ValueError("frozen memory bank memories must be a list")
    memories = bank_memories(bank) if raw_memories else ()
    by_id = {memory.memory_id: memory for memory in memories}
    memory_id = retrieval.get("selected_memory_id")
    memory = None
    if retrieval_status == "retrieved":
        candidates = retrieval.get("candidates")
        if (not isinstance(candidates, list) or not candidates
                or not isinstance(candidates[0], dict)
                or candidates[0].get("rank") != 1
                or candidates[0].get("memory_id") != memory_id
                or candidates[0].get("score") != retrieval.get("similarity_score")):
            raise ValueError("retrieval top-1 fields are inconsistent")
        if not isinstance(memory_id, str) or memory_id not in by_id:
            raise ValueError("retrieved memory is absent from the verified bank")
        memory = by_id[memory_id]
        v = memory.quality.get("v", "?")
        if v != 1:
            raise ValueError("primary retrieved memories must have verified v=1 evidence")
        rho = evaluate_conditions(memory.positive_conditions,
                                  memory.exclusion_conditions, context)
        evidence = copy.deepcopy(memory.quality)
        memory_record = {
            "mechanism": memory.mechanism,
            "problem_action": copy.deepcopy(memory.problem_action),
            "correction": copy.deepcopy(memory.correction),
            "positive_conditions": copy.deepcopy(memory.positive_conditions),
            "exclusion_conditions": copy.deepcopy(memory.exclusion_conditions),
        }
        retrieval_metadata = {"status": retrieval_status,
                              "similarity_score": retrieval.get("similarity_score")}
        record_status = "prepared"
    else:
        if memory_id is not None:
            raise ValueError("no_match result must not select a memory")
        if memories:
            raise ValueError("no_match is invalid when a verified memory bank is nonempty")
        memory = None
        v = "?"
        rho = "?"
        evidence = {"v": "?", "reason": "no_retrieved_memory"}
        memory_record = None
        retrieval_metadata = {"status": "no_match", "similarity_score": None}
        record_status = "prepared_no_memory"

    # Retrieval similarity is deliberately not used to infer action exposure.
    exposure = _exposure(memory_id=memory_id, context=context, action=action,
                         rules=exposure_rules)
    divergence, text_embedding = _divergence(context=context, action=action,
                                              memory=memory, bank=bank, encoder=encoder)
    budget = _remaining_budget(remaining_computation)
    evidence_feature = {key: copy.deepcopy(value) for key, value in evidence.items()
                        if key in ("v", "reason", "scope", "n_replay_pairs",
                                   "mean_difference", "acceptance_epsilon", "decision_rule",
                                   "claim_boundary")}
    policy_context = plain(context)
    policy_context.pop("task_id", None)
    x = {
        "schema_version": "pre-treatment-state.v1",
        "context": policy_context,
        "a0": action,
        "m": memory_record,
        "E": evidence_feature,
        "d": divergence,
        "v": v,
        "rho": rho,
        "b": budget,
        "e": exposure,
        "retrieval": retrieval_metadata,
        "feature_map": ({"status": "frozen_text_encoder_plus_structured.v1",
                         "encoder_identity": copy.deepcopy(encoder.identity),
                         "text_query_hash": expected_query_hash,
                         "text_embedding": text_embedding,
                         "structured_schema_version": "pre-treatment-structured.v1"}
                        if encoder is not None and isinstance(getattr(encoder, "identity", None), dict)
                        else {"status": "structured_only_incomplete",
                              "reason": "frozen_F_omega_encoder_not_configured"}),
    }
    result = {
        "schema_version": "pre-treatment-state-record.v1",
        "status": record_status,
        "bank_hash": bank_hash,
        "memory_id": memory_id,
        "task_id": context.task_id,
        "retrieval_hash": digest(retrieval),
        "context_hash": digest(plain(context)),
        "a0_hash": digest(action),
        "x": x,
        "x_hash": digest(x),
        "randomization_status": "not_assigned",
        "outcome_access": "none",
    }
    result["record_hash"] = digest(result)
    return result
