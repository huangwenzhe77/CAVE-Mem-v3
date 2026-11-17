"""Transparent risk cues computed from public pre-action information only.

These binary cues are a pilot sampling heuristic, not calibrated error
probabilities. Freeze this rubric before collecting target outcomes.
"""
from __future__ import annotations

import json
import re
from typing import Any


_ERROR_CUES = re.compile(
    r"\b(?:error|failed|failure|exception|traceback|not found|cannot|unable|"
    r"invalid|retry|again|denied)\b|"
    r"\b(?:non[- ]zero exit(?: code)?|exit(?:ed)?(?: with)? (?:status|code) [1-9]\d*|"
    r"returned? (?:an? )?(?:non[- ]zero|[1-9]\d*) exit(?: code)?)\b",
    re.IGNORECASE,
)
_NONZERO_EXIT_FIELD = re.compile(
    r"\b(?:exit_code|exit code|return_code|return code|status)\s*[:=]\s*[\"']?([1-9]\d*)\b",
    re.IGNORECASE,
)


def _has_public_error_cue(observation: str) -> bool:
    if _ERROR_CUES.search(observation) or _NONZERO_EXIT_FIELD.search(observation):
        return True
    try:
        payload = json.loads(observation)
    except (json.JSONDecodeError, TypeError):
        return False
    return (isinstance(payload, dict)
            and type(payload.get("exit_code")) is int
            and payload["exit_code"] != 0)


def _action_key(action: Any) -> str | None:
    if not isinstance(action, dict) or not action:
        return None
    try:
        return json.dumps(action, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _prior_actions(context: dict, prior_steps: list[dict]) -> list[str]:
    # The sampler should pass the complete pre-checkpoint step history. Public
    # context history is a fallback; counting both can double-count the same
    # events when the caller derives prior_steps from context.history.
    events = prior_steps
    if not events:
        history = context.get("history", ())
        events = history if isinstance(history, (list, tuple)) else []
    actions = []
    for event in events:
        if isinstance(event, dict):
            key = _action_key(event.get("action"))
            if key is not None:
                actions.append(key)
    return actions


def _prior_tool_failure(prior_steps: list[dict]) -> bool:
    for step in prior_steps:
        if not isinstance(step, dict):
            continue
        if step.get("validation_error"):
            return True
        if step.get("valid") is False or step.get("tool_error") is True:
            return True
        result = step.get("result")
        if isinstance(result, dict):
            if result.get("valid") is False or result.get("tool_error") is True:
                return True
            exit_code = result.get("exit_code")
            if type(exit_code) is int and exit_code != 0:
                return True
    return False


def risk_features(context: dict, prior_steps: list[dict], *,
                  rubric: str = "public-pre-action-risk-cues.v1") -> dict:
    """Return frozen public cues and their preregistered integer score.

    ``context`` is the public context immediately before the candidate action.
    ``prior_steps`` must contain only steps strictly before that checkpoint.
    Candidate-step action/result, task outcome, memory contents, and private
    checkpoint state are deliberately not inspected.
    """
    if not isinstance(context, dict):
        raise TypeError("context must be a public-context mapping")
    if not isinstance(prior_steps, list) or any(not isinstance(step, dict) for step in prior_steps):
        raise TypeError("prior_steps must be a list of prior step mappings")

    observation = context.get("observation", "")
    if not isinstance(observation, str):
        raise TypeError("public observation must be text")

    actions = _prior_actions(context, prior_steps)
    repeated_prior_action = len(actions) != len(set(actions))
    cues = {
        "explicit_public_error_cue": _has_public_error_cue(observation),
        "prior_tool_failure": _prior_tool_failure(prior_steps),
        "repeated_prior_action": repeated_prior_action,
    }
    if rubric == "public-pre-action-risk-cues.v1":
        return {**cues, "risk_score": sum(cues.values()), "risk_score_max": 3,
                "risk_rubric": rubric}
    if rubric != "public-pre-action-risk-cues.v2":
        raise ValueError("unsupported public checkpoint risk rubric")
    actions = context.get("admissible_actions", ())
    if not isinstance(actions, (list, tuple)) or any(not isinstance(a, str) for a in actions):
        raise TypeError("public admissible actions must be a text sequence")
    interaction = (isinstance(context.get("facts"), dict)
                   and context["facts"].get("benchmark") == "alfworld-text"
                   and any(re.match(r"^(?:take|put|use|heat|cool|clean|slice|move)\s", a, re.I)
                           for a in actions))
    return {**cues, "object_interaction_available": bool(interaction),
            "risk_score": sum(cues.values()) + 3 * int(bool(interaction)),
            "risk_score_max": 6, "risk_rubric": rubric}
