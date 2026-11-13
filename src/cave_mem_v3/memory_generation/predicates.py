"""Executable predicates over existing public observations only.

These functions never execute commands, acquire observations, or read snapshots.
Missing public facts remain unknown; ``exists`` tests the observation map itself.
"""
from __future__ import annotations

import copy
import re
from typing import Literal

from cave_mem_v3.common.contracts import PublicContext, canonical, finite

Truth = Literal[0, 1, "?"]
_PRIVATE_FIELDS = frozenset({
    "success", "terminal_success", "future_success", "future_outcome", "outcome",
    "reward", "score", "label", "target", "targets", "verifier", "verifier_answer",
    "private_verifier", "private", "hidden_state", "environment_state", "snapshot",
    "checkpoint", "rng", "actor_state", "metadata", "task_id", "checkpoint_id",
    "source_task_id", "terminal", "failure_label",
})


def _scalar(value) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        finite(value, "predicate value")
        return True
    return False


def validate_predicate(predicate: dict) -> dict:
    """Validate one field comparison or one regex on a public text source."""
    if not isinstance(predicate, dict) or not predicate:
        raise ValueError("a predicate must be a nonempty object")
    keys = set(predicate)
    if "field" in keys:
        operators = keys & {"equals", "not_equals", "in", "exists"}
        if len(operators) != 1 or keys != {"field", *operators}:
            raise ValueError("field predicates require exactly one equals/not_equals/in/exists operator")
        field = predicate["field"]
        if not isinstance(field, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*", field) is None:
            raise ValueError("predicate field must be a public fact path")
        if any(part.lower() in _PRIVATE_FIELDS or part.startswith("_") for part in field.split(".")):
            raise ValueError("private/outcome identifiers cannot be predicate inputs")
        operator = next(iter(operators))
        value = predicate[operator]
        if operator == "exists":
            if type(value) is not bool:
                raise ValueError("exists requires a boolean")
        elif operator == "in":
            if not isinstance(value, list) or not value or not all(_scalar(item) for item in value):
                raise ValueError("in requires a nonempty array of scalar values")
        elif not _scalar(value):
            raise ValueError("comparison values must be finite scalars")
    elif keys == {"source", "regex"}:
        if predicate["source"] not in ("observation", "instruction", "history"):
            raise ValueError("regex source must be observation, instruction, or public history")
        pattern = predicate["regex"]
        if not isinstance(pattern, str) or not pattern or len(pattern) > 2048:
            raise ValueError("regex must be a nonempty pattern of at most 2048 characters")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError("invalid public-observation regex") from exc
    else:
        raise ValueError("unsupported predicate schema; narrative conditions are not executable")
    return copy.deepcopy(predicate)


def _same(left, right) -> bool:
    return type(left) is type(right) and left == right


def evaluate_predicate(predicate: dict, context: PublicContext) -> Truth:
    predicate = validate_predicate(predicate)
    if not isinstance(context, PublicContext):
        raise TypeError("predicates accept PublicContext, never raw environment or audit state")
    if "field" in predicate:
        found, value = True, context.facts
        for part in predicate["field"].split("."):
            if not isinstance(value, dict) or part not in value:
                found = False
                break
            value = value[part]
        if "exists" in predicate:
            return int(found == predicate["exists"])
        if not found or not _scalar(value):
            return "?"
        if "equals" in predicate:
            return int(_same(value, predicate["equals"]))
        if "not_equals" in predicate:
            return int(not _same(value, predicate["not_equals"]))
        return int(any(_same(value, item) for item in predicate["in"]))
    source = predicate["source"]
    if source == "history":
        # The gate reads only executed actions and their visible observations.
        # Metadata or accidentally attached scorer fields cannot enter regexes.
        public_events = [{key: event[key] for key in ("action", "observation") if key in event}
                         for event in context.history if isinstance(event, dict)]
        public_events = [event for event in public_events if event]
        text = canonical(public_events) if public_events else ""
    else:
        text = getattr(context, source)
    if not isinstance(text, str) or not text:
        return "?"
    return int(re.search(predicate["regex"], text) is not None)


def evaluate_conditions(positive: list[dict] | tuple[dict, ...],
                        exclusion: list[dict] | tuple[dict, ...], context: PublicContext) -> Truth:
    """All positive conditions must hold; any exclusion vetoes applicability."""
    if not positive:
        raise ValueError("at least one positive predicate is required")
    p = [evaluate_predicate(item, context) for item in positive]
    n = [evaluate_predicate(item, context) for item in exclusion]
    if 0 in p or 1 in n:
        return 0
    if "?" in p or "?" in n:
        return "?"
    return 1
