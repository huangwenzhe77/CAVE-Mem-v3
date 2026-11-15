"""Online, outcome-blind sampling of one pre-action checkpoint per episode."""
from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from cave_mem_v3.common.contracts import digest
from cave_mem_v3.target_data.risk_signals import risk_features


@dataclass(frozen=True)
class RiskHazard:
    """Frozen per-decision selection hazards for the target-data pilot."""

    low_risk_probability: float = 0.05
    high_risk_probability: float = 0.75
    risk_rubric: str = "public-pre-action-risk-cues.v1"
    schema_version: str = "risk-hazard.v1"

    def __post_init__(self) -> None:
        low = self.low_risk_probability
        high = self.high_risk_probability
        if (isinstance(low, bool) or not isinstance(low, (int, float))
                or not math.isfinite(low) or not 0 < low < 1):
            raise ValueError("low_risk_probability must be between zero and one")
        if (isinstance(high, bool) or not isinstance(high, (int, float))
                or not math.isfinite(high) or not low <= high < 1):
            raise ValueError("high_risk_probability must be >= low risk and < one")
        if not self.risk_rubric or not self.schema_version:
            raise ValueError("hazard and risk rubric versions are required")

    def probability(self, features: dict) -> float:
        score = features.get("risk_score")
        maximum = features.get("risk_score_max")
        if (type(score) is not int or type(maximum) is not int
                or maximum < 1 or not 0 <= score <= maximum):
            raise ValueError("risk features must contain a bounded integer risk score")
        fraction = score / maximum
        return self.low_risk_probability + fraction * (
            self.high_risk_probability - self.low_risk_probability)


def _public_context(value) -> dict:
    """Whitelist public fields and avoid double-counting context.history."""
    if hasattr(value, "__dataclass_fields__"):
        value = {name: copy.deepcopy(getattr(value, name))
                 for name in value.__dataclass_fields__}
    if not isinstance(value, dict):
        raise TypeError("checkpoint context must be a PublicContext or mapping")
    required = ("task_id", "instruction", "observation")
    if any(not isinstance(value.get(name), str) or not value[name] for name in required):
        raise ValueError("checkpoint context needs task_id, instruction, and observation")
    # Prior actions/results are carried separately, exactly once, by prior_steps.
    return {name: copy.deepcopy(value[name]) for name in (
        "task_id", "instruction", "observation", "facts", "admissible_actions",
        "remaining_steps") if name in value}


def _incomplete_record(*, role: str, task_id: str, trace_id: str, seed: int,
                       hazard: RiskHazard, decisions: list[dict], reason: str,
                       error: Exception) -> dict:
    return {
        "schema_version": "target-checkpoint-sample.v1",
        "status": "sampling_incomplete",
        "reason": reason,
        "role": role,
        "task_id": task_id,
        "trace_id": trace_id,
        "seed": seed,
        "hazard": asdict(hazard),
        "selected_step_index": None,
        "selected_checkpoint_hash": None,
        "selected_checkpoint": None,
        "selected_public_context": None,
        "inclusion_probability": None,
        "selection_decisions": copy.deepcopy(decisions),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def select_first_checkpoint(steps: Iterable[dict], *, role: str, task_id: str,
                            trace_id: str, seed: int,
                            hazard: RiskHazard | None = None) -> dict:
    """Select the first pre-action checkpoint passing its risk-biased hazard.

    ``steps`` must be a live iterator consumed online; iteration stops at the
    first selection and the function never receives an episode outcome. Each
    call selects at most one checkpoint for its trace. The caller must invoke
    this once per task if enforcing the proposal's task-level one-point limit.
    Every step must expose its pre-action public ``context``, checkpoint hash,
    and restorable snapshot. Current action/result data are ignored; only
    completed earlier steps enter risk cues. A stream error returns an
    incomplete record with the decisions observed so far; callers must preserve
    that record as an infrastructure/incomplete outcome, not as a task failure.
    """
    if role not in ("target_train", "validation"):
        raise ValueError("checkpoint sampling is allowed only for target_train or validation")
    if not isinstance(task_id, str) or not task_id or not isinstance(trace_id, str) or not trace_id:
        raise ValueError("task_id and trace_id are required")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if hazard is None:
        hazard = RiskHazard()

    rng = random.Random(seed)
    previous_nonselection = 1.0
    prior_steps: list[dict] = []
    decisions = []
    iterator = iter(steps)
    ordinal = 0
    while True:
        try:
            step = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="rollout_stream_error", error=exc)
        if not isinstance(step, dict):
            error = TypeError("each observed step must be a mapping")
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=error)
        index = step.get("index", ordinal)
        if type(index) is not int or index < 0:
            error = ValueError("step index must be a nonnegative integer")
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=error)
        checkpoint_hash = step.get("checkpoint_hash")
        if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
            error = ValueError("each candidate needs a pre-action checkpoint_hash")
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=error)
        checkpoint = step.get("checkpoint")
        if not isinstance(checkpoint, dict):
            error = ValueError("each candidate needs a restorable pre-action checkpoint mapping")
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=error)
        try:
            if digest(checkpoint) != checkpoint_hash:
                raise ValueError("checkpoint_hash does not match the supplied checkpoint snapshot")
            context = _public_context(step.get("context"))
        except Exception as exc:
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=exc)
        if context["task_id"] != task_id:
            error = ValueError("checkpoint context task_id differs from the sampled task")
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=error)

        try:
            features = risk_features(context, prior_steps)
            if features.get("risk_rubric") != hazard.risk_rubric:
                raise ValueError("risk signal rubric differs from the frozen hazard configuration")
            probability = hazard.probability(features)
        except Exception as exc:
            return _incomplete_record(role=role, task_id=task_id, trace_id=trace_id,
                                      seed=seed, hazard=hazard, decisions=decisions,
                                      reason="invalid_checkpoint_event", error=exc)
        draw = rng.random()
        inclusion_probability = previous_nonselection * probability
        decision = {
            "step_index": index,
            "checkpoint_hash": checkpoint_hash,
            "risk_features": copy.deepcopy(features),
            "selection_hazard": probability,
            "uniform_draw": draw,
            "survival_probability_before": previous_nonselection,
            "conditional_inclusion_probability": inclusion_probability,
            "selected": draw < probability,
        }
        decisions.append(decision)
        if decision["selected"]:
            return {
                "schema_version": "target-checkpoint-sample.v1",
                "status": "selected",
                "role": role,
                "task_id": task_id,
                "trace_id": trace_id,
                "seed": seed,
                "hazard": asdict(hazard),
                "selected_step_index": index,
                "selected_checkpoint_hash": checkpoint_hash,
                "selected_checkpoint": copy.deepcopy(step.get("checkpoint")),
                "selected_public_context": context,
                "inclusion_probability": inclusion_probability,
                "selection_decisions": decisions,
            }

        previous_nonselection *= 1 - probability
        if previous_nonselection <= 0 or not math.isfinite(previous_nonselection):
            raise ArithmeticError("nonselection probability became numerically invalid")
        # Only after the current decision is rejected does this action/result
        # become prior history for the next pre-action checkpoint.
        prior_steps.append({key: copy.deepcopy(step[key]) for key in (
            "action", "result", "validation_error", "valid", "tool_error") if key in step})
        ordinal += 1

    return {
        "schema_version": "target-checkpoint-sample.v1",
        "status": "no_selection",
        "reason": "rollout_ended_without_selection",
        "role": role,
        "task_id": task_id,
        "trace_id": trace_id,
        "seed": seed,
        "hazard": asdict(hazard),
        "selected_step_index": None,
        "selected_checkpoint_hash": None,
        "selected_checkpoint": None,
        "selected_public_context": None,
        "inclusion_probability": None,
        "no_selection_probability": previous_nonselection,
        "selection_decisions": decisions,
    }
