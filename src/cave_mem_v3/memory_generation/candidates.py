"""Source-only repair hypotheses, bound to recorded failed trajectories.

A generated explanation is never evidence of repair efficacy. Private snapshots
and verifier materials are excluded from hypothesis-model prompts.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Iterable

from cave_mem_v3.common.contracts import canonical, digest, plain
from cave_mem_v3.memory_generation.predicates import validate_predicate


class CandidateGenerationError(ValueError):
    """A failed hypothesis request or parse, retaining any charged attempt."""

    def __init__(self, message: str, *, generation: dict, candidates: Iterable[dict] = ()):
        super().__init__(message)
        self.generation = copy.deepcopy(generation)
        self.candidates = copy.deepcopy(list(candidates))
        self.attempt = copy.deepcopy(generation.get("attempt"))


def checked_trace(trace: dict) -> dict:
    """Validate the collection envelope before using its source records."""
    if trace.get("schema_version") != "history-trace.v3" or trace.get("role") != "history":
        raise ValueError("candidates require a history-trace.v3 source")
    if trace.get("trace_hash") != digest({k: v for k, v in trace.items() if k != "trace_hash"}):
        raise ValueError("source trace checksum mismatch")
    if not all(isinstance(trace.get(k), str) and trace[k] for k in ("trace_id", "task_id", "manifest_hash", "run_hash")):
        raise ValueError("source trace is missing provenance")
    episode = trace.get("episode")
    if not isinstance(episode, dict) or not isinstance(episode.get("steps"), list):
        raise ValueError("source trace needs an episode and recorded steps")
    return episode


def _failed_step(trace: dict, step_index: int) -> dict:
    episode = checked_trace(trace)
    if episode.get("success") is not False or episode.get("status") not in ("terminal", "budget_exhausted", "invalid_action"):
        raise ValueError("repair hypotheses require a completed unsuccessful episode, not an infrastructure error")
    if type(step_index) is not int or not 0 <= step_index < len(episode["steps"]):
        raise ValueError("candidate step is outside its source trajectory")
    step = episode["steps"][step_index]
    if step.get("index") != step_index or not isinstance(step.get("checkpoint"), dict):
        raise ValueError("source step has no aligned restorable checkpoint")
    if not isinstance(step.get("action"), dict) or not step["action"]:
        raise ValueError("source step has no executable original action")
    if step.get("context", {}).get("task_id") != trace["task_id"]:
        raise ValueError("source step task identity mismatch")
    return step


def _binding(trace: dict, step_index: int) -> dict:
    step = _failed_step(trace, step_index)
    return {"source_task_id": trace["task_id"], "source_trace_id": trace["trace_id"],
            "source_trace_hash": trace["trace_hash"], "manifest_hash": trace["manifest_hash"],
            "run_hash": trace["run_hash"], "step_index": step_index,
            "checkpoint_hash": digest(step["checkpoint"]),
            "problem_action": copy.deepcopy(step["action"])}


def source_failure_signature(trace: dict, step_index: int) -> str:
    """Describe the recorded local event; do not invent a causal explanation."""
    step = _failed_step(trace, step_index)
    result = step.get("result") or {}
    return canonical({"source_status": trace["episode"]["status"],
                      "action": step["action"],
                      "observation_before": step["context"].get("observation", ""),
                      "observation_after": (result.get("context") or {}).get("observation", ""),
                      "action_valid": result.get("valid"),
                      "validation_error": step.get("validation_error")})


def candidate_templates(traces: Iterable[dict], *, max_steps_per_trace: int = 3) -> list[dict]:
    """Return editable hypotheses; no guessed cause or invented correction."""
    if type(max_steps_per_trace) is not int or max_steps_per_trace < 1:
        raise ValueError("max_steps_per_trace must be positive")
    drafts = []
    for trace in traces:
        episode = checked_trace(trace)
        if episode.get("success") is not False or episode.get("status") == "error":
            continue
        for index in range(min(len(episode["steps"]), max_steps_per_trace)):
            try:
                binding = _binding(trace, index)
            except ValueError:
                continue
            drafts.append({"schema_version": "repair-candidate.v3", "status": "draft", **binding,
                           "mechanism": "", "correction": None, "positive_conditions": [],
                           "exclusion_conditions": [], "scope": "", "failure_signature": source_failure_signature(trace, index),
                           "proposed_by": "", "notes": "Fill a falsifiable hypothesis; source replay will decide validity."})
    return drafts


def _conditions(value: Any, name: str) -> list[dict]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be an array of observable conditions")
    return [validate_predicate(condition) for condition in value]


def finalize_candidate(draft: dict, trace: dict) -> dict:
    """Freeze editable fields while checking provenance against the real trace."""
    if draft.get("schema_version") != "repair-candidate.v3":
        raise ValueError("unsupported candidate schema")
    binding = _binding(trace, draft.get("step_index"))
    if any(draft.get(key) != value for key, value in binding.items()):
        raise ValueError("candidate provenance or original action differs from the recorded failure")
    for name in ("mechanism", "scope", "proposed_by", "failure_signature"):
        if not isinstance(draft.get(name), str) or not draft[name].strip():
            raise ValueError(f"candidate needs {name}")
    correction = draft.get("correction")
    if not isinstance(correction, dict) or not isinstance(correction.get("type"), str) or not correction["type"]:
        raise ValueError("candidate needs a typed executable correction")
    if correction == binding["problem_action"]:
        raise ValueError("correction is identical to the original action")
    positive = _conditions(draft.get("positive_conditions", []), "positive_conditions")
    if not positive:
        raise ValueError("at least one observable positive condition is required")
    result = {"schema_version": "repair-candidate.v3", "status": "hypothesis", **binding,
              "mechanism": draft["mechanism"].strip(), "correction": copy.deepcopy(correction),
              "positive_conditions": positive,
              "exclusion_conditions": _conditions(draft.get("exclusion_conditions", []), "exclusion_conditions"),
              "scope": draft["scope"].strip(), "failure_signature": draft["failure_signature"].strip(),
              "proposed_by": draft["proposed_by"].strip(), "notes": str(draft.get("notes", ""))}
    if "generation" in draft:
        result["generation"] = copy.deepcopy(draft["generation"])
    result["candidate_hash"] = digest(result)
    return result


def create_candidate(trace: dict, step_index: int, *, mechanism: str, correction: dict,
                     positive_conditions: Iterable[dict], exclusion_conditions: Iterable[dict] = (),
                     scope: str, reviewer: str = "manual", failure_signature: str = "", notes: str = "") -> dict:
    draft = {"schema_version": "repair-candidate.v3", **_binding(trace, step_index),
             "mechanism": mechanism, "correction": correction, "positive_conditions": list(positive_conditions),
             "exclusion_conditions": list(exclusion_conditions), "scope": scope,
             "proposed_by": reviewer,
             "failure_signature": failure_signature or source_failure_signature(trace, step_index), "notes": notes}
    return finalize_candidate(draft, trace)


def validate_candidate(candidate: dict, trace: dict) -> None:
    if canonical(candidate) != canonical(finalize_candidate(candidate, trace)):
        raise ValueError("candidate checksum or frozen hypothesis changed")


def public_source_episode(trace: dict) -> dict:
    """An allowlist view for a source hypothesis model, never a raw snapshot."""
    episode = checked_trace(trace)
    steps = []
    for index, step in enumerate(episode["steps"]):
        context = step.get("context", {})
        result = step.get("result") or {}
        next_context = result.get("context") or {}
        steps.append({"step_index": index,
                      "instruction": context.get("instruction", ""),
                      "observation": context.get("observation", ""),
                      "action": copy.deepcopy(step.get("action")),
                      "next_observation": next_context.get("observation", ""),
                      "action_valid": result.get("valid")})
    return {"task_id": trace["task_id"], "success": episode.get("success"),
            "status": episode.get("status"), "steps": steps}


def generate_candidates(trace: dict, provider: Any, *, successful_traces: Iterable[dict] = (),
                        max_candidates: int = 3, seed: int = 42) -> dict:
    """Propose source hypotheses with one metered structured model request.

    Reference successes must be historical runs of the same task. Generated
    fields are validated and sealed exactly like manually written candidates.
    No retry or best-of-N filtering is hidden in this function.
    """
    episode = checked_trace(trace)
    if episode.get("success") is not False or episode.get("status") == "error":
        raise ValueError("candidate generation needs a failed source episode")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    successful_traces = list(successful_traces)
    successes = []
    for reference in successful_traces:
        reference_episode = checked_trace(reference)
        if reference["task_id"] != trace["task_id"] or reference_episode.get("success") is not True:
            raise ValueError("successful references must be historical successes of the same task")
        successes.append(public_source_episode(reference))
    schema = {"candidates": [{"step_index": 0, "mechanism": "falsifiable failure hypothesis",
                              "correction": {"type": "shell", "command": "executable repair"},
                              "positive_conditions": [{"source": "observation", "regex": "observed pattern"}],
                              "exclusion_conditions": [{"field": "prepared", "equals": True}],
                              "scope": "source validity boundary", "failure_signature": "observed failure"}]}
    messages = [{"role": "system", "content": (
        "Propose source-side repair hypotheses from the supplied historical observations. "
        "Do not claim verified causality or target transfer. Use only observed facts; do not invent hidden state. "
        "Choose a recorded step and one executable action replacing that step. Return JSON only, with no markdown. "
        "Conditions must be executable: a public fact field with exactly one equals/not_equals/in/exists operator, "
        "or source observation/instruction/history with a regex. Do not use private scorer/outcome fields. "
        f"Return at most {max_candidates} distinct candidates using this schema: {canonical(schema)}")},
        {"role": "user", "content": canonical({"failed": public_source_episode(trace), "successes": successes})}]
    before = copy.deepcopy(provider.configuration())
    generation = {"status": "started", "error_type": None, "failure_stage": None,
                  "provider_config": before, "model": None, "prompt_hash": digest(messages),
                  "response_hash": None, "usage": None, "attempt": None, "seed": seed,
                  "reference_trace_hashes": [r["trace_hash"] for r in successful_traces]}
    candidates, failure_stage = [], "provider"
    try:
        response = provider.generate(messages, stage="source_hypothesis", seed=seed,
                                     call_id="hypothesis-" + trace["trace_hash"][:20])
        # Capture metering before checking model output or frozen configuration.
        generation.update(status="completed", model=response.model,
                          response_hash=digest(response.text), usage=plain(response.usage),
                          attempt=plain(response.attempt))
        failure_stage = "configuration"
        if provider.configuration() != before:
            raise ValueError("hypothesis provider configuration changed")
        failure_stage = "parse"
        decoded = json.loads(response.text)
        proposals = decoded.get("candidates") if isinstance(decoded, dict) else None
        if not isinstance(proposals, list) or not 0 <= len(proposals) <= max_candidates:
            raise ValueError("hypothesis model returned an invalid number of candidates")
        failure_stage = "candidate_validation"
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise ValueError("hypothesis candidates must be objects")
            draft = {**proposal, "schema_version": "repair-candidate.v3",
                     **_binding(trace, proposal.get("step_index")),
                     "proposed_by": "model:" + str(response.model), "generation": generation}
            if not draft.get("failure_signature"):
                draft["failure_signature"] = source_failure_signature(trace, draft["step_index"])
            candidate = finalize_candidate(draft, trace)
            if any(c["candidate_hash"] == candidate["candidate_hash"] for c in candidates):
                raise ValueError("hypothesis model returned duplicate candidates")
            candidates.append(candidate)
    except Exception as exc:
        if generation["attempt"] is None:
            attempt = getattr(exc, "attempt", None)
            generation["attempt"] = plain(attempt) if attempt is not None else {
                "status": "failed", "error_class": type(exc).__name__,
                "usage": {"model_calls": None, "physical_attempts": None,
                          "input_tokens": None, "output_tokens": None, "api_cost": None}}
            generation["usage"] = copy.deepcopy(generation["attempt"].get("usage"))
        generation.update(status="error", error_type=type(exc).__name__, failure_stage=failure_stage)
        raise CandidateGenerationError("source hypothesis generation failed at " + failure_stage + ": " + type(exc).__name__,
                                       generation=generation, candidates=candidates) from exc
    return {"candidates": candidates, "generation": generation}
