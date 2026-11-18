"""Top-1 semantic retrieval of verified source memories for a target checkpoint."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Protocol, Sequence

from cave_mem_v3.common.contracts import PublicContext, canonical, digest, plain
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.predicates import evaluate_conditions


class EmbeddingEncoder(Protocol):
    """Minimal dense-embedding interface used by the ExpeL-style KNN retriever."""

    identity: dict

    def encode_query(self, text: str) -> Sequence[float]: ...
    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEncoder:
    """Local-only sentence-transformer adapter; never downloads model weights."""

    def __init__(self, model_path: str, *, revision: str, device: str = "cpu"):
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError("encoder model_path must point to an existing local model directory")
        if not isinstance(revision, str) or not revision:
            raise ValueError("encoder revision must be explicitly recorded")
        if not isinstance(device, str) or not device:
            raise ValueError("encoder device must be a nonempty string")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("install the optional 'encoder' dependencies to use SentenceTransformerEncoder") from exc
        self.identity = {"kind": "sentence_transformers", "model_path": str(path.resolve()),
                         "revision": revision, "device": device,
                         "normalization": "cosine_at_retrieval.v1",
                         "ranking_policy": "alfworld_public_family_applicability_then_cosine.v1",
                         "weight_identity_status": "declared_revision_not_content_attestation"}
        self._model = SentenceTransformer(str(path), device=device, local_files_only=True)

    def encode_query(self, text: str) -> Sequence[float]:
        return self._model.encode([text], convert_to_numpy=True, normalize_embeddings=False)[0].tolist()

    def encode_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = self._model.encode(list(texts), convert_to_numpy=True,
                                     normalize_embeddings=False, show_progress_bar=False)
        return vectors.tolist()


_PRIVATE_FACT_KEYS = frozenset({
    "success", "terminal_success", "future_success", "future_outcome", "outcome",
    "reward", "score", "label", "target", "targets", "verifier", "verifier_answer",
    "private_verifier", "private", "hidden_state", "environment_state", "snapshot",
    "checkpoint", "checkpoint_id", "rng", "actor_state", "metadata", "terminal",
    "failure_label", "source_task_id", "source_trace_id",
})


def _public_value(value):
    if isinstance(value, dict):
        return {key: _public_value(item) for key, item in value.items()
                if isinstance(key, str) and key.lower() not in _PRIVATE_FACT_KEYS
                and not key.startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_public_value(item) for item in value]
    return copy.deepcopy(value)


def _query_text(context: PublicContext | dict, action: dict) -> str:
    if isinstance(context, PublicContext):
        context = plain(context)
    if not isinstance(context, dict):
        raise TypeError("retrieval context must be a PublicContext or public mapping")
    for field in ("instruction", "observation"):
        if not isinstance(context.get(field), str):
            raise ValueError(f"public context requires text field {field}")
    if not isinstance(action, dict) or not action:
        raise ValueError("cached actor action a0 must be a nonempty mapping")
    history = context.get("history", ())
    if not isinstance(history, (list, tuple)):
        raise ValueError("public context history must be a list or tuple")
    public_history = []
    for event in history:
        if isinstance(event, dict):
            public_history.append({key: _public_value(event[key]) for key in ("action", "observation")
                                  if key in event})
    query = {
        "instruction": context["instruction"],
        "observation": context["observation"],
        "facts": _public_value(context.get("facts", {})),
        "history": public_history,
        "admissible_actions": _public_value(context.get("admissible_actions", [])),
        "remaining_steps": context.get("remaining_steps"),
        "a0": _public_value(action),
    }
    return canonical(query)


def _memory_text(memory) -> str:
    # Correction and evidence are intentionally excluded: retrieval matches the
    # failure pattern, while applicability and reminder construction are later steps.
    return canonical({
        "mechanism": memory.mechanism,
        "problem_action": memory.problem_action,
        "positive_conditions": memory.positive_conditions,
        "exclusion_conditions": memory.exclusion_conditions,
    })


def _task_family_rank(memory, facts: dict) -> int:
    """Prefer an explicitly matching public task family over an incompatible one."""
    if (not isinstance(facts, dict) or facts.get("benchmark") != "alfworld-text"
            or not isinstance(facts.get("task_type"), str)):
        return 1
    family_conditions = [condition for condition in memory.positive_conditions
                         if condition.get("field") == "task_type" and "equals" in condition]
    if not family_conditions:
        return 1
    return 2 if all(condition["equals"] == facts["task_type"] for condition in family_conditions) else 0


def _applicability_rank(memory, context: PublicContext) -> int:
    result = evaluate_conditions(memory.positive_conditions, memory.exclusion_conditions, context)
    return {1: 2, "?": 1, 0: 0}[result]


def _vector(value, name: str) -> list[float]:
    try:
        values = list(value)
    except TypeError as exc:
        raise ValueError(f"{name} embedding must be a numeric vector") from exc
    if not values:
        raise ValueError(f"{name} embedding cannot be empty")
    result = []
    for component in values:
        if isinstance(component, bool) or not isinstance(component, (int, float)) or not math.isfinite(component):
            raise ValueError(f"{name} embedding must contain only finite numbers")
        result.append(float(component))
    norm = math.sqrt(sum(component * component for component in result))
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"{name} embedding must have a finite nonzero norm")
    return result


def retrieve_memory(bank: dict, *, context: PublicContext | dict, action: dict,
                    encoder: EmbeddingEncoder) -> dict:
    """Rank verified memories by dense cosine similarity and return the top one.

    Similarity identifies a candidate only. It does not establish that the
    memory is applicable, that its mechanism is active, or that a reminder is
    warranted.
    """
    if not isinstance(bank, dict):
        raise TypeError("bank must be a frozen verified-source-bank mapping")
    bank_hash = bank.get("bank_hash")
    if not isinstance(bank_hash, str) or not bank_hash:
        raise ValueError("frozen memory bank requires bank_hash")
    if not isinstance(getattr(encoder, "identity", None), dict) or not encoder.identity:
        raise ValueError("embedding encoder requires a nonempty frozen identity")
    query = _query_text(context, action)

    # A correctly hashed zero-memory bank is reported as no_match. Nonempty banks
    # go through the existing provenance/evidence validator and must contain v=1.
    if bank.get("schema_version") != "verified-source-bank.v3" or bank.get("frozen") is not True:
        raise ValueError("unsupported or mutable memory bank")
    if bank_hash != digest({key: value for key, value in bank.items() if key != "bank_hash"}):
        raise ValueError("frozen memory bank checksum mismatch")
    raw_memories = bank.get("memories")
    if not isinstance(raw_memories, list):
        raise ValueError("frozen memory bank memories must be a list")
    if not raw_memories:
        if bank.get("memory_count") != 0 or bank.get("audit_sources") != {}:
            raise ValueError("empty bank metadata is inconsistent")
        return {"schema_version": "target-memory-retrieval.v1", "status": "no_match",
                "reason": "empty_verified_bank", "bank_hash": bank_hash,
                "encoder_identity": copy.deepcopy(encoder.identity),
                "query_hash": digest(query), "selected_memory_id": None,
                "similarity_score": None, "candidates": []}

    memories = bank_memories(bank)
    if not memories:
        return {"schema_version": "target-memory-retrieval.v1", "status": "no_match",
                "reason": "no_verified_candidates", "bank_hash": bank_hash,
                "encoder_identity": copy.deepcopy(encoder.identity),
                "query_hash": digest(query), "selected_memory_id": None,
                "similarity_score": None, "candidates": []}

    documents = [_memory_text(memory) for memory in memories]
    query_vector = _vector(encoder.encode_query(query), "query")
    document_vectors = encoder.encode_documents(documents)
    if len(document_vectors) != len(memories):
        raise ValueError("encoder returned a different number of document vectors")
    vectors = [_vector(value, f"document {index}") for index, value in enumerate(document_vectors)]
    if any(len(value) != len(query_vector) for value in vectors):
        raise ValueError("query and document embedding dimensions differ")

    query_norm = math.sqrt(sum(value * value for value in query_vector))
    facts = _public_value(plain(context).get("facts", {}) if isinstance(context, PublicContext)
                          else context.get("facts", {}))
    family_aware = isinstance(facts, dict) and facts.get("benchmark") == "alfworld-text"
    public_context = None
    if family_aware:
        values = plain(context) if isinstance(context, PublicContext) else context
        public_context = PublicContext(**{key: _public_value(values[key]) for key in (
            "task_id", "instruction", "observation", "facts", "history",
            "admissible_actions", "remaining_steps") if key in values})
    ranked = []
    for memory, vector in zip(memories, vectors):
        norm = math.sqrt(sum(value * value for value in vector))
        score = sum(left * right for left, right in zip(query_vector, vector)) / (query_norm * norm)
        ranked.append((memory.memory_id, max(-1.0, min(1.0, score)), _task_family_rank(memory, facts),
                       _applicability_rank(memory, public_context) if family_aware else 1))
    ranked.sort(key=lambda item: (-item[2], -item[3], -item[1], item[0]))
    candidates = [{"rank": rank, "memory_id": memory_id, "score": score}
                  for rank, (memory_id, score, _, _) in enumerate(ranked, start=1)]
    return {"schema_version": "target-memory-retrieval.v1", "status": "retrieved",
            "reason": None, "bank_hash": bank_hash,
            "encoder_identity": copy.deepcopy(encoder.identity),
            "query_hash": digest(query), "selected_memory_id": candidates[0]["memory_id"],
            "similarity_score": candidates[0]["score"], "candidates": candidates}
