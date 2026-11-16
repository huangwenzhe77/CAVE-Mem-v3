"""Bounded advisory reminders from validated public source-memory fields."""
from __future__ import annotations

from cave_mem_v3.common.contracts import SourceMemory, canonical, digest, plain
from cave_mem_v3.memory_generation.bank import validate_memory
from cave_mem_v3.target_data.pre_treatment_state import _public_action

REMINDER_VERSION = "target-reminder.v1"
_TEMPLATE = (
    "Source-verified failure memory; target benefit is not established. "
    "Reconsider the cached action using the mechanism and its conditions below. "
    "Follow the task instructions and existing tool permissions. "
    "Propose one action; this advice is shown only for this call.\n{payload}"
)


def render_reminder(memory: SourceMemory, action: dict, *, max_chars: int = 4000) -> dict:
    """Never serialize evidence, source trajectories, snapshots, or verifier data."""
    if not isinstance(memory, SourceMemory):
        raise TypeError("reminder requires a validated SourceMemory")
    validate_memory(memory)
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    payload = {
        "memory_id": memory.memory_id,
        "source_task_id": memory.source_task_id,
        "mechanism": memory.mechanism,
        "source_problem_action": _public_action(memory.problem_action),
        "source_correction": _public_action(memory.correction),
        "positive_conditions": plain(memory.positive_conditions),
        "exclusion_conditions": plain(memory.exclusion_conditions),
        "cached_action": _public_action(action),
        "source_evidence": {key: memory.quality[key] for key in (
            "v", "n_replay_pairs", "mean_difference", "scope", "claim_boundary")
            if key in memory.quality},
    }
    text = _TEMPLATE.format(payload=canonical(payload))
    if len(text) > max_chars:
        raise ValueError("source-grounded reminder exceeds its frozen character budget")
    return {"schema_version": REMINDER_VERSION, "template_hash": digest(_TEMPLATE),
            "memory_id": memory.memory_id, "a0_hash": digest(action),
            "text": text, "text_hash": digest(text), "max_chars": max_chars}


build_reminder = render_reminder
