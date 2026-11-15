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


class OnlineCheckpointSampler:
    """Make one synchronous selection decision at each pre-action node.

    Create one instance per rollout. ``consider`` is called before the actor
    proposes an action. After one node is selected, later calls return
    ``already_selected`` without evaluating risk, allowing the rollout itself
    to continue to its terminal outcome.
    """

    def __init__(self, *, role: str, task_id: str, trace_id: str, seed: int,
                 hazard: RiskHazard | None = None):
        if role not in ("target_train", "validation"):
            raise ValueError("checkpoint sampling is allowed only for target_train or validation")
        if not isinstance(task_id, str) or not task_id or not isinstance(trace_id, str) or not trace_id:
            raise ValueError("task_id and trace_id are required")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.role = role
        self.task_id = task_id
        self.trace_id = trace_id
        self.seed = seed
        self.hazard = hazard or RiskHazard()
        self._rng = random.Random(seed)
        self._survival = 1.0
        self._decisions: list[dict] = []
        self._selected: dict | None = None
        self._incomplete: dict | None = None

    @property
    def selected(self) -> bool:
        return self._selected is not None

    @property
    def complete(self) -> bool:
        return self.selected or self._incomplete is not None

    def _incomplete_result(self, reason: str, error: Exception) -> dict:
        self._incomplete = _incomplete_record(
            role=self.role, task_id=self.task_id, trace_id=self.trace_id,
            seed=self.seed, hazard=self.hazard, decisions=self._decisions,
            reason=reason, error=error)
        return {"status": "sampling_incomplete", "selected": False,
                "error": copy.deepcopy(self._incomplete["error"])}

    def consider(self, context, checkpoint: dict, index: int,
                 prior_steps: list[dict] | None = None,
                 checkpoint_hash: str | None = None) -> dict:
        """Evaluate one pre-action node using only its public prefix.

        ``prior_steps`` contains completed steps strictly before this node.
        The candidate node's action/result and episode outcome are not inputs.
        """
        if self.selected:
            return {"status": "already_selected", "selected": True, "decision": None}
        if self._incomplete is not None:
            return {"status": "sampling_incomplete", "selected": False,
                    "error": copy.deepcopy(self._incomplete["error"])}
        try:
            if type(index) is not int or index < 0:
                raise ValueError("step index must be a nonnegative integer")
            if not isinstance(checkpoint, dict):
                raise ValueError("each candidate needs a restorable pre-action checkpoint mapping")
            actual_checkpoint_hash = digest(checkpoint)
            if checkpoint_hash is not None and checkpoint_hash != actual_checkpoint_hash:
                raise ValueError("checkpoint_hash does not match the supplied checkpoint snapshot")
            public = _public_context(context)
            if public["task_id"] != self.task_id:
                raise ValueError("checkpoint context task_id differs from the sampled task")
            if prior_steps is None:
                prior_steps = []
            features = risk_features(public, prior_steps)
            if features.get("risk_rubric") != self.hazard.risk_rubric:
                raise ValueError("risk signal rubric differs from the frozen hazard configuration")
            probability = self.hazard.probability(features)
        except Exception as exc:
            return self._incomplete_result("invalid_checkpoint_event", exc)

        draw = self._rng.random()
        inclusion_probability = self._survival * probability
        selected = draw < probability
        decision = {
            "step_index": index,
            "checkpoint_hash": actual_checkpoint_hash,
            "risk_features": copy.deepcopy(features),
            "selection_hazard": probability,
            "uniform_draw": draw,
            "survival_probability_before": self._survival,
            "conditional_inclusion_probability": inclusion_probability,
            "selected": selected,
        }
        self._decisions.append(decision)
        if selected:
            self._selected = {
                "selected_step_index": index,
                "selected_checkpoint_hash": actual_checkpoint_hash,
                "selected_checkpoint": copy.deepcopy(checkpoint),
                "selected_public_context": public,
                "inclusion_probability": inclusion_probability,
            }
            return {"status": "selected", "selected": True,
                    "decision": copy.deepcopy(decision)}

        self._survival *= 1 - probability
        if self._survival <= 0 or not math.isfinite(self._survival):
            return self._incomplete_result(
                "invalid_sampling_probability",
                ArithmeticError("nonselection probability became numerically invalid"))
        return {"status": "not_selected", "selected": False,
                "decision": copy.deepcopy(decision)}

    def finalize(self) -> dict:
        """Return an auditable per-rollout record after the rollout ends."""
        if self._incomplete is not None:
            return copy.deepcopy(self._incomplete)
        common = {
            "schema_version": "target-checkpoint-sample.v1",
            "role": self.role,
            "task_id": self.task_id,
            "trace_id": self.trace_id,
            "seed": self.seed,
            "hazard": asdict(self.hazard),
            "selection_decisions": copy.deepcopy(self._decisions),
        }
        if self._selected is not None:
            return {**common, "status": "selected", **copy.deepcopy(self._selected)}
        return {
            **common,
            "status": "no_selection",
            "reason": "rollout_ended_without_selection",
            "selected_step_index": None,
            "selected_checkpoint_hash": None,
            "selected_checkpoint": None,
            "selected_public_context": None,
            "inclusion_probability": None,
            "no_selection_probability": self._survival,
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
    sampler = OnlineCheckpointSampler(role=role, task_id=task_id,
                                      trace_id=trace_id, seed=seed, hazard=hazard)
    prior_steps: list[dict] = []
    iterator = iter(steps)
    ordinal = 0
    while True:
        try:
            step = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            sampler._incomplete_result("rollout_stream_error", exc)
            return sampler.finalize()
        if not isinstance(step, dict):
            sampler._incomplete_result("invalid_checkpoint_event",
                                       TypeError("each observed step must be a mapping"))
            return sampler.finalize()
        index = step.get("index", ordinal)
        if type(index) is not int or index < 0:
            sampler._incomplete_result("invalid_checkpoint_event",
                                       ValueError("step index must be a nonnegative integer"))
            return sampler.finalize()
        checkpoint = step.get("checkpoint")
        expected_hash = step.get("checkpoint_hash")
        if not isinstance(expected_hash, str) or not expected_hash:
            sampler._incomplete_result(
                "invalid_checkpoint_event",
                ValueError("each candidate needs a pre-action checkpoint_hash"))
            return sampler.finalize()
        result = sampler.consider(step.get("context"), checkpoint, index, prior_steps,
                                  checkpoint_hash=expected_hash)
        if result["status"] in ("selected", "sampling_incomplete"):
            return sampler.finalize()
        # Only after the current decision is rejected does this action/result
        # become prior history for the next pre-action checkpoint.
        prior_steps.append({key: copy.deepcopy(step[key]) for key in (
            "action", "result", "validation_error", "valid", "tool_error") if key in step})
        ordinal += 1
    return sampler.finalize()
