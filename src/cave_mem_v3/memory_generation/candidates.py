"""Source-only repair hypotheses, bound to recorded failed trajectories.

A generated explanation is never evidence of repair efficacy. Private snapshots
and verifier materials are excluded from hypothesis-model prompts.
"""
from __future__ import annotations

import copy
import json
from typing import Any, Iterable

from cave_mem_v3.common.contracts import PublicContext, canonical, digest, plain
from cave_mem_v3.memory_generation.collection import scored_source_outcome
from cave_mem_v3.memory_generation.predicates import evaluate_conditions, validate_predicate


def _decode_candidate_json(text: str) -> dict:
    """Decode a complete teacher JSON object despite harmless wrapper text."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("teacher returned empty candidate output")
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        starts = [index for index, char in enumerate(value) if char == "{"]
        decoded = None
        for start in starts:
            try:
                candidate, end = decoder.raw_decode(value[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and not value[start + end:].strip():
                decoded = candidate
                break
        if decoded is None:
            raise ValueError("teacher output is not one complete JSON object")
    if not isinstance(decoded, dict):
        raise ValueError("teacher output must be a JSON object")
    return decoded


def _teacher_source_view(trace: dict, eligible: list[int]) -> dict:
    """Expand representative eligible checkpoints and retain every public action."""
    episode = public_source_episode(trace)
    if len(eligible) <= 7:
        keep = set(eligible)
        selection_rule = "all_eligible.v1"
    else:
        # Long trajectories often fail in the middle. Keep the boundary steps
        # and three evenly spaced eligible checkpoints without a second model
        # request or a post hoc choice among multiple proposals.
        keep = set(eligible[:2] + eligible[-2:])
        keep.update(eligible[round((len(eligible) - 1) * fraction)]
                    for fraction in (0.25, 0.5, 0.75))
        selection_rule = "first_two_last_two_and_eligible_quartiles.v1"
    steps = []
    omitted_path = []
    for step in episode["steps"]:
        if step["step_index"] not in keep:
            omitted_path.append({"step_index": step["step_index"],
                                 "action": copy.deepcopy(step.get("action")),
                                 "observation": str(step.get("next_observation", ""))[:80],
                                 "action_valid": step.get("action_valid")})
            continue
        item = copy.deepcopy(step)
        for key in ("observation", "next_observation"):
            if isinstance(item.get(key), str) and len(item[key]) > 1000:
                item[key] = item[key][:1000] + "..."
        actions = item.get("admissible_actions")
        if isinstance(actions, list) and len(actions) > 30:
            item["admissible_actions"] = actions[:30]
            item["admissible_actions_truncated"] = True
        steps.append(item)
    return {"task_id": episode["task_id"], "success": episode["success"],
            "status": episode["status"], "termination_reason": episode["termination_reason"],
            "actor_behavior": episode["actor_behavior"], "steps": steps,
            "expanded_step_selection": selection_rule,
            "omitted_step_count": len(episode["steps"]) - len(steps),
            "omitted_action_path": omitted_path}


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
    if scored_source_outcome(episode) is not False:
        raise ValueError("repair hypotheses require a scored unsuccessful source episode")
    if type(step_index) is not int or not 0 <= step_index < len(episode["steps"]):
        raise ValueError("candidate step is outside its source trajectory")
    step = episode["steps"][step_index]
    if step.get("index") != step_index or not isinstance(step.get("checkpoint"), dict):
        raise ValueError("source step has no aligned restorable checkpoint")
    if not isinstance(step.get("result"), dict) and not step.get("validation_error"):
        raise ValueError("source step has neither an observed result nor a recorded action error")
    action = step.get("action")
    # A typed action with a command remains useful evidence even when the
    # environment rejected it (for example, an agent returned json_object
    # where ALFWorld requires text). The correction route can normalize it;
    # an unparsed response still has no recoverable action to repair.
    if (not isinstance(action, dict) or not action
            or action.get("type") == "__unparsed_action__"
            or action.get("type") in ("text", "shell")
            and (not isinstance(action.get("command"), str)
                 or not action["command"].strip())):
        raise ValueError("source step has no executable original action")
    if step.get("context", {}).get("task_id") != trace["task_id"]:
        raise ValueError("source step task identity mismatch")
    return step


def eligible_source_steps(trace: dict) -> list[int]:
    if scored_source_outcome(trace.get("episode") or {}) is not False:
        return []
    episode = checked_trace(trace)
    eligible = []
    for index in range(len(episode["steps"])):
        try:
            _failed_step(trace, index)
        except ValueError:
            continue
        eligible.append(index)
    return eligible


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


def candidate_templates(traces: Iterable[dict], *, max_steps_per_trace: int = 1) -> list[dict]:
    """Return editable hypotheses; no guessed cause or invented correction."""
    if type(max_steps_per_trace) is not int or max_steps_per_trace != 1:
        raise ValueError("each eligible source trace must have exactly one candidate template")
    drafts = []
    for trace in traces:
        episode = checked_trace(trace)
        if scored_source_outcome(episode) is not False:
            continue
        for index in eligible_source_steps(trace)[:1]:
            binding = _binding(trace, index)
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
                      "admissible_actions": context.get("admissible_actions", []),
                      "action": copy.deepcopy(step.get("action")),
                      "next_observation": next_context.get("observation", ""),
                      "action_valid": result.get("valid"),
                      "validation_error": step.get("validation_error"),
                      "finish_reason": (step.get("actor_response") or {}).get("finish_reason")})
    return {"task_id": trace["task_id"], "success": episode.get("success"),
            "status": episode.get("status"), "termination_reason": episode.get("termination_reason"),
            "actor_behavior": episode.get("actor_behavior"), "steps": steps}


def generate_candidates(trace: dict, provider: Any, *, successful_traces: Iterable[dict] = (),
                        max_candidates: int = 1, seed: int = 42) -> dict:
    """Propose source hypotheses with one metered structured model request.

    Reference successes must be historical runs of the same task. Generated
    fields are validated and sealed exactly like manually written candidates.
    No retry or best-of-N filtering is hidden in this function.
    """
    episode = checked_trace(trace)
    if not eligible_source_steps(trace):
        raise ValueError("candidate generation needs a scored source failure with an executable original action")
    if type(max_candidates) is not int or max_candidates != 1:
        raise ValueError("each eligible source trace must request exactly one candidate")
    successful_traces = list(successful_traces)
    successes = []
    for reference in successful_traces:
        reference_episode = checked_trace(reference)
        if reference["task_id"] != trace["task_id"] or scored_source_outcome(reference_episode) is not True:
            raise ValueError("successful references must be historical successes of the same task")
        successes.append(_teacher_source_view(reference, list(range(len(reference_episode["steps"])))))
    action_type = next((episode["steps"][i]["action"].get("type") for i in eligible_source_steps(trace)), "shell")
    source_view = _teacher_source_view(trace, eligible_source_steps(trace))
    visible_steps = [step["step_index"] for step in source_view["steps"]]
    correction_example = ({"type": "text", "command": "one exact admissible command"}
                          if action_type == "text" else {"type": "shell", "command": "executable repair"})
    schema = {"candidates": [{"step_index": 0, "mechanism": "falsifiable failure hypothesis",
                              "correction": correction_example,
                              "positive_conditions": [{"source": "observation", "regex": "observed pattern"}],
                              "exclusion_conditions": [{"field": "prepared", "equals": True}],
                              "scope": "source validity boundary", "failure_signature": "observed failure"}]}
    messages = [{"role": "system", "content": (
        "Propose source-side repair hypotheses from the supplied historical observations. "
        "Do not claim verified causality or target transfer. Use only observed facts; do not invent hidden state. "
        "Choose a step only from eligible_step_indices and one executable action replacing that step. "
        "The replacement must be one executable action of the same type as the recorded original. "
        "For text games, copy the replacement command exactly from that step's public admissible actions; "
        "for shell tasks, directly create or repair the artifact. Do not use a placeholder or multi-step plan. "
        "Prefer a checkpoint where one replacement action can complete the remaining task within the recorded budget. "
        "Keep the mechanism, scope, and failure signature concise. "
        "Write positive and exclusion conditions for the transferable failure mechanism, not for the exact source scene. "
        "Text-game object identifiers and coordinates (for example bar__plus_01_dot_28) must never appear in condition regexes; "
        "use stable object classes, task instruction wording, or observable action patterns instead. "
        "The source correction command must still be copied exactly from its source checkpoint's admissible actions. "
        "Do not make an exclusion condition that merely negates a source-scene identifier. "
        "Every positive condition must already be true in the chosen step's instruction, observation, public history, "
        "or public facts; every exclusion condition must already be false there. Do not refer to observations "
        "from later steps in any condition. "
        "Return JSON only, with no markdown. "
        "Conditions must be executable: a public fact field with exactly one equals/not_equals/in/exists operator, "
        "or source observation/instruction/history with a regex. Do not use private scorer/outcome fields. "
        f"Return exactly one candidate using this schema: {canonical(schema)}")},
        {"role": "user", "content": canonical({"failed": source_view,
                                                 "eligible_step_indices": visible_steps,
                                                 "successes": successes})}]
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
        decoded = _decode_candidate_json(response.text)
        proposals = decoded.get("candidates") if isinstance(decoded, dict) else None
        if not isinstance(proposals, list) or len(proposals) != 1:
            raise ValueError("hypothesis model must return exactly one candidate")
        failure_stage = "candidate_validation"
        for proposal in proposals:
            if not isinstance(proposal, dict):
                raise ValueError("hypothesis candidates must be objects")
            if proposal.get("step_index") not in visible_steps:
                raise ValueError("hypothesis step was not in the public eligible source view")
            draft = {**proposal, "schema_version": "repair-candidate.v3",
                     **_binding(trace, proposal.get("step_index")),
                     "proposed_by": "model:" + str(response.model), "generation": generation}
            if not draft.get("failure_signature"):
                draft["failure_signature"] = source_failure_signature(trace, draft["step_index"])
            candidate = finalize_candidate(draft, trace)
            source_context = PublicContext(**episode["steps"][candidate["step_index"]]["context"])
            if evaluate_conditions(candidate["positive_conditions"],
                                   candidate["exclusion_conditions"], source_context) != 1:
                raise ValueError("hypothesis conditions do not hold at the source checkpoint")
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
        generation.update(status="error", error_type=type(exc).__name__,
                          error_message=str(exc), failure_stage=failure_stage)
        raise CandidateGenerationError("source hypothesis generation failed at " + failure_stage + ": " + type(exc).__name__,
                                       generation=generation, candidates=candidates) from exc
    return {"candidates": candidates, "generation": generation}
