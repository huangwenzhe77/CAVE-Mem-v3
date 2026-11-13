"""Immutable verified source memory banks with separately stored audit sources."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from cave_mem_v3.memory_generation.candidates import checked_trace, validate_candidate
from cave_mem_v3.common.contracts import SourceMemory, canonical, digest, plain, read_json
from cave_mem_v3.memory_generation.replay import validate_evidence


def _quality(evidence: dict | None) -> dict:
    if not evidence:
        return {"v": "?", "reason": "unverified_hypothesis", "scope": None}
    spec = evidence["plan"]["replay_spec"]
    return {"v": evidence["v"], "reason": evidence["reason"], "scope": evidence["scope"],
            "endpoint": evidence["endpoint"], "confidence": spec["confidence"],
            "family_size": spec["family_size"], "n_confirmation": evidence["n_confirmation"],
            "lower_bound": evidence["lower_bound"], "upper_bound": evidence["upper_bound"],
            "evidence_hash": evidence["evidence_hash"], "claim_boundary": evidence["claim_boundary"]}


def build_memory(candidate: dict, trace: dict, evidence: dict | None = None, *,
                 require_verified: bool = True) -> SourceMemory:
    """Construct proposal's m without turning a hypothesis into valid evidence."""
    validate_candidate(candidate, trace)
    if evidence is not None:
        validate_evidence(evidence, candidate=candidate, trace=trace)
    if require_verified and (not evidence or evidence["v"] != 1):
        raise ValueError("a primary memory requires supported independent source confirmation")
    memory = SourceMemory("pending", candidate["source_task_id"], candidate["source_trace_id"],
                          candidate["checkpoint_hash"], candidate["mechanism"],
                          copy.deepcopy(candidate["problem_action"]), copy.deepcopy(candidate["correction"]),
                          tuple(candidate["positive_conditions"]), tuple(candidate["exclusion_conditions"]),
                          copy.deepcopy(evidence) if evidence else {}, _quality(evidence))
    return replace(memory, memory_id="mem-" + memory.content_hash())


def validate_memory(memory: SourceMemory, trace: dict | None = None, *, require_verified: bool = True) -> dict:
    if memory.schema_version != "source-memory.v3" or memory.memory_id != "mem-" + memory.content_hash():
        raise ValueError("memory identity/schema mismatch")
    if not memory.evidence:
        if require_verified or memory.quality != _quality(None):
            raise ValueError("unverified memory cannot enter the primary bank")
        return {"v": "?", "reason": "unverified_hypothesis"}
    check = validate_evidence(memory.evidence, trace=trace)
    candidate = memory.evidence["plan"]["candidate"]
    expected = {"source_task_id": candidate["source_task_id"], "source_trace_id": candidate["source_trace_id"],
                "source_checkpoint_hash": candidate["checkpoint_hash"], "mechanism": candidate["mechanism"],
                "problem_action": candidate["problem_action"], "correction": candidate["correction"],
                "positive_conditions": candidate["positive_conditions"],
                "exclusion_conditions": candidate["exclusion_conditions"], "quality": _quality(memory.evidence)}
    if any(canonical(getattr(memory, key)) != canonical(value) for key, value in expected.items()):
        raise ValueError("memory claim or quality differs from source-tested evidence")
    if require_verified and check["v"] != 1:
        raise ValueError("only independently supported source memories enter the primary bank")
    return check


def _validate_bank(bank: dict) -> list[SourceMemory]:
    if bank.get("schema_version") != "verified-source-bank.v3" or bank.get("frozen") is not True:
        raise ValueError("unsupported or mutable source bank")
    if bank.get("bank_hash") != digest({k: v for k, v in bank.items() if k != "bank_hash"}):
        raise ValueError("frozen bank checksum mismatch")
    sources = bank.get("audit_sources")
    if not isinstance(sources, dict):
        raise ValueError("bank needs separate audit source trajectories")
    for key, trace in sources.items():
        checked_trace(trace)
        if key != trace["trace_id"]:
            raise ValueError("bank source trace key mismatch")
    memories = []
    for raw in bank.get("memories", []):
        memory = SourceMemory(**raw)
        trace = sources.get(memory.source_trace_id)
        if trace is None:
            raise ValueError("bank memory has no original audit source")
        validate_memory(memory, trace)
        memories.append(memory)
    if not memories or len({m.memory_id for m in memories}) != len(memories):
        raise ValueError("bank requires nonempty unique verified memories")
    if bank.get("memory_count") != len(memories):
        raise ValueError("bank memory count mismatch")
    if set(sources) != {m.source_trace_id for m in memories}:
        raise ValueError("bank audit sources must exactly cover its memory provenance")
    return memories


def freeze_bank(memories: Iterable[SourceMemory], path: str | Path, *,
                source_traces: Iterable[dict], metadata: dict | None = None) -> dict:
    """Create once; raw audit sources are not eligible prompt/feature inputs."""
    memories = list(memories)
    all_sources = list(source_traces)
    if len({t.get("trace_id") for t in all_sources}) != len(all_sources):
        raise ValueError("duplicate source trace identity")
    needed = {m.source_trace_id for m in memories}
    sources = {t["trace_id"]: copy.deepcopy(t) for t in all_sources if t.get("trace_id") in needed}
    bank = {"schema_version": "verified-source-bank.v3", "frozen": True,
            "memory_count": len(memories), "memories": [plain(m) for m in memories],
            "audit_sources": sources, "metadata": copy.deepcopy(metadata or {}),
            "audit_source_access": "private;never pass raw bank or snapshots to actor or target feature encoder"}
    bank["bank_hash"] = digest(bank)
    _validate_bank(bank)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing a previously frozen experimental bank.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(bank, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return bank


def load_bank(path: str | Path) -> dict:
    """Recheck hashes, measured labels, source provenance and confidence claims."""
    bank = read_json(path)
    _validate_bank(bank)
    return bank


def bank_memories(bank: dict) -> tuple[SourceMemory, ...]:
    return tuple(_validate_bank(bank))
