"""Shared phase-one contracts. Private snapshots are never actor inputs."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def canonical(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return float(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def stage_seed(base_seed: int, trial_id: str, replicate: int, stage: str, step: int = 0) -> int:
    return int(digest([base_seed, trial_id, replicate, stage, step])[:15], 16) % (2**31 - 1)


@dataclass(frozen=True)
class PublicContext:
    task_id: str
    instruction: str
    observation: str
    facts: dict[str, Any] = field(default_factory=dict)
    history: tuple[dict[str, Any], ...] = ()
    admissible_actions: tuple[str, ...] = ()
    remaining_steps: int = 20

    def __post_init__(self) -> None:
        if not self.task_id or not self.instruction:
            raise ValueError("task_id and instruction are required")
        if type(self.remaining_steps) is not int or self.remaining_steps < 0:
            raise ValueError("remaining_steps must be a nonnegative integer")
        object.__setattr__(self, "facts", copy.deepcopy(self.facts))
        object.__setattr__(self, "history", tuple(copy.deepcopy(self.history)))
        object.__setattr__(self, "admissible_actions", tuple(self.admissible_actions))
        canonical(self)


@dataclass(frozen=True)
class StudySpec:
    study_id: str = "phase1-pilot"
    base_seed: int = 42
    max_steps: int = 30
    max_model_calls: int = 30
    lambda_inference: float = 0.01
    lambda_tool: float = 0.0
    inference_weights: dict[str, float] = field(default_factory=lambda: {"model_calls": 1.0})
    tool_weights: dict[str, float] = field(default_factory=lambda: {"tool_calls": 1.0})

    def __post_init__(self) -> None:
        if not self.study_id:
            raise ValueError("study_id is required")
        for name in ("base_seed", "max_steps", "max_model_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "base_seed" else 1):
                raise ValueError(f"invalid {name}")
        for name in ("lambda_inference", "lambda_tool"):
            finite(getattr(self, name), name, minimum=0)
        for name in ("inference_weights", "tool_weights"):
            weights = copy.deepcopy(getattr(self, name))
            if not weights:
                raise ValueError("cost weights must be explicit")
            for key, value in weights.items():
                if not isinstance(key, str) or not key:
                    raise ValueError("invalid cost name")
                finite(value, key, minimum=0)
            object.__setattr__(self, name, weights)


def weighted_cost(costs: dict, weights: dict) -> float | None:
    total = 0.0
    for key, weight in weights.items():
        if weight == 0:
            continue
        if costs.get(key) is None:
            return None
        total += weight * finite(costs[key], key, minimum=0)
    return total


def utility(success: bool, costs: dict, spec: StudySpec) -> float | None:
    if type(success) is not bool:
        raise ValueError("success must be bool")
    inference = weighted_cost(costs, spec.inference_weights) if spec.lambda_inference else 0.0
    tool = weighted_cost(costs, spec.tool_weights) if spec.lambda_tool else 0.0
    if inference is None or tool is None:
        return None
    return float(success) - spec.lambda_inference * inference - spec.lambda_tool * tool


@dataclass(frozen=True)
class SourceMemory:
    """Proposal m=(p-, a-, f, correction, P, N, E, q); E remains evidence."""
    memory_id: str
    source_task_id: str
    source_trace_id: str
    source_checkpoint_hash: str
    mechanism: str
    problem_action: dict[str, Any]
    correction: dict[str, Any]
    positive_conditions: tuple[dict[str, Any], ...] = ()
    exclusion_conditions: tuple[dict[str, Any], ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "source-memory.v3"

    def __post_init__(self) -> None:
        if not all((self.source_task_id, self.source_trace_id, self.source_checkpoint_hash, self.mechanism)):
            raise ValueError("source memory needs provenance and mechanism")
        for name in ("problem_action", "correction", "evidence", "quality"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        for name in ("positive_conditions", "exclusion_conditions"):
            object.__setattr__(self, name, tuple(copy.deepcopy(getattr(self, name))))
        canonical(self)

    def content_hash(self) -> str:
        value = plain(self)
        value.pop("memory_id")
        return digest(value)
