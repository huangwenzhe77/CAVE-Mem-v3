"""Fixed-K source action-repair confirmation for proposal equation (6).

The inference is conditional on one source checkpoint and its declared replay
randomness; repeated seeds do not create independent source tasks.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Callable

from cave_mem_v3.memory_generation.candidates import checked_trace, validate_candidate
from cave_mem_v3.common.contracts import StudySpec, canonical, digest, finite, plain, read_json, stage_seed, utility, write_json


SOURCE_PLAN_SCHEMA = "source-replay-plan.v6"
SOURCE_EVIDENCE_SCHEMA = "source-evidence.v6"
SOURCE_DECISION_RULE = "all_corrections_success_and_positive_mean.v1"
_LEGACY_DECISION_RULE = "complete_valid_pairs_and_mean_difference_gt_acceptance_epsilon"
_PLAN_VERSIONS = {
    "source-replay-plan.v5": ("source-replay-pair.v5", "source-evidence.v5", _LEGACY_DECISION_RULE),
    SOURCE_PLAN_SCHEMA: ("source-replay-pair.v6", SOURCE_EVIDENCE_SCHEMA, SOURCE_DECISION_RULE),
}


def _protocol(plan: dict) -> tuple[str, str, str]:
    try:
        version = _PLAN_VERSIONS[plan.get("schema_version")]
    except (KeyError, TypeError):
        raise ValueError("unsupported source replay plan") from None
    if plan["schema_version"] == SOURCE_PLAN_SCHEMA and plan.get("decision_rule") != SOURCE_DECISION_RULE:
        raise ValueError("source replay decision rule differs from its schema")
    return version


@dataclass(frozen=True)
class ReplaySpec:
    replay_pairs: int
    acceptance_epsilon: float = 0.0
    cost_upper_bounds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.replay_pairs) is not int or self.replay_pairs < 1:
            raise ValueError("replay_pairs must be a positive integer")
        finite(self.acceptance_epsilon, "acceptance_epsilon", minimum=0)
        object.__setattr__(self, "cost_upper_bounds", copy.deepcopy(self.cost_upper_bounds))
        for name, value in self.cost_upper_bounds.items():
            finite(value, name + " upper bound", minimum=0)


def score_bounds(study: StudySpec, spec: ReplaySpec) -> tuple[float, float]:
    """A predeclared finite range for utility, not an observed data range."""
    bounds = {"model_calls": float(study.max_model_calls),
              "physical_attempts": float(study.max_model_calls),
              "tool_calls": float(study.max_steps), **spec.cost_upper_bounds}
    penalty = 0.0
    for coefficient, weights in ((study.lambda_inference, study.inference_weights),
                                 (study.lambda_tool, study.tool_weights)):
        for name, weight in weights.items():
            if coefficient and weight:
                if name not in bounds:
                    raise ValueError("utility evidence requires a finite declared upper bound for " + name)
                penalty += coefficient * weight * bounds[name]
    return -penalty, 1.0


def _seed(study: StudySpec, candidate: dict, replicate: int) -> int:
    return stage_seed(study.base_seed, candidate["candidate_hash"], replicate, "source-replay")


def episode_response_identity(episode: dict) -> dict:
    """Recompute served identity from recorded responses, not configuration alone."""
    model, fingerprint, missing, count = None, None, False, 0
    for step in episode.get("steps", []):
        response = step.get("actor_response")
        if response is None:
            continue
        if not isinstance(response, dict):
            raise ValueError("recorded actor response must be an object")
        current_model, current_fingerprint = response.get("model"), response.get("fingerprint")
        if not isinstance(current_model, str) or not current_model:
            raise ValueError("recorded actor response is missing model identity")
        if model is not None and model != current_model:
            raise ValueError("recorded response model drifted within episode")
        model = current_model
        if current_fingerprint == "":
            current_fingerprint = None
        if current_fingerprint is not None and not isinstance(current_fingerprint, str):
            raise ValueError("recorded actor response fingerprint is invalid")
        if current_fingerprint is None:
            missing = True
        elif fingerprint is not None and fingerprint != current_fingerprint:
            raise ValueError("recorded actor response fingerprint drifted within episode")
        else:
            fingerprint = current_fingerprint
        count += 1
    identity = {"model": model, "fingerprint": fingerprint,
                "fingerprint_status": "unknown" if missing or not count else "reported"}
    if "response_identity" in episode and episode["response_identity"] != identity:
        raise ValueError("episode response identity summary differs from recorded responses")
    return identity


def _check_source_response_identity(episode: dict, expected: dict) -> dict:
    identity = episode_response_identity(episode)
    if identity["model"] is None:
        # An overridden action can terminate without any continuation LLM call.
        return identity
    if identity["model"] != expected["model"]:
        raise ValueError("replay actor returned a different model from the historical source")
    if (identity["fingerprint"] is not None and expected["fingerprint"] is not None
            and identity["fingerprint"] != expected["fingerprint"]):
        raise ValueError("replay actor fingerprint differs from the historical source")
    return identity


def make_plan(candidate: dict, trace: dict, study: StudySpec, spec: ReplaySpec) -> dict:
    """Freeze a new replay under the current all-corrections-success rule."""
    return _make_plan(candidate, trace, study, spec, SOURCE_PLAN_SCHEMA)


def _make_plan(candidate: dict, trace: dict, study: StudySpec, spec: ReplaySpec,
               schema_version: str) -> dict:
    # Legacy plans are reconstructed only for historical provenance validation.
    if schema_version not in _PLAN_VERSIONS:
        raise ValueError("unsupported source replay plan")
    validate_candidate(candidate, trace)
    episode = checked_trace(trace)
    if not isinstance(episode.get("actor_config"), dict) or not isinstance(episode.get("environment_config"), dict):
        raise ValueError("source trace must record actor and environment configuration")
    capabilities = episode.get("capabilities", {})
    if capabilities.get("environment", {}).get("restore_certified") is not True:
        raise ValueError("source checkpoint restoration has not been certified")
    if capabilities.get("actor", {}).get("stateless") is not True:
        raise ValueError("source actor must be frozen and stateless")
    source_response_identity = episode_response_identity(episode)
    if source_response_identity["model"] is None:
        raise ValueError("historical source must record the actor's actual response identity")
    schedule = [_seed(study, candidate, i) for i in range(spec.replay_pairs)]
    if len(set(schedule)) != len(schedule):
        raise ValueError("replay seed collision; choose another base seed before collection")
    plan = {"schema_version": schema_version, "candidate": copy.deepcopy(candidate),
            "source_trace_hash": trace["trace_hash"], "study": plain(study), "replay_spec": plain(spec),
            "actor_config": episode["actor_config"], "environment_config": episode["environment_config"],
            "capabilities": capabilities, "score_bounds": list(score_bounds(study, spec)),
            "source_response_identity": source_response_identity,
            "schedule": schedule, "endpoint": "terminal_utility",
            "scope": candidate["scope"], "coupling": "same_checkpoint_stage_step_seed;provider_noise_as_declared"}
    if schema_version == SOURCE_PLAN_SCHEMA:
        plan["decision_rule"] = SOURCE_DECISION_RULE
    plan["plan_hash"] = digest(plan)
    return plan


def _collect_pair(plan: dict, checkpoint: dict, env_factory: Callable, actor_factory: Callable,
                  replicate: int, episode_runner: Callable) -> dict:
    study = StudySpec(**plan["study"])
    candidate = plan["candidate"]
    seed = plan["schedule"][replicate]
    row = {"schema_version": _protocol(plan)[0], "plan_hash": plan["plan_hash"],
           "candidate_hash": candidate["candidate_hash"], "checkpoint_hash": candidate["checkpoint_hash"],
           "replicate": replicate, "seed": seed,
           "sample_id": digest([plan["plan_hash"], replicate, seed]), "arms": {}, "valid": False}
    order = ["original", "corrected"]
    random.Random(stage_seed(seed, plan["plan_hash"], replicate, "arm_order")).shuffle(order)
    row["arm_order"] = order
    sessions = []
    for name in order:
        action = candidate["problem_action"] if name == "original" else candidate["correction"]
        env = actor = None
        arm = {"status": "error", "success": None, "utility": None, "costs": {}, "steps": [],
               "first_action": copy.deepcopy(action)}
        try:
            env, actor = env_factory(), actor_factory()
            if any(env is pair[0] or actor is pair[1] for pair in sessions):
                raise ValueError("each source branch must use independent environment and actor sessions")
            sessions.append((env, actor))
            if env.capabilities().get("restore_certified") is not True or actor.capabilities().get("stateless") is not True:
                raise ValueError("source replay needs certified restoration and a stateless actor")
            if env.configuration() != plan["environment_config"] or actor.configuration() != plan["actor_config"]:
                raise ValueError("source replay changed the actor or environment configuration")
            env.restore(copy.deepcopy(checkpoint))
            if digest(env.snapshot()) != candidate["checkpoint_hash"]:
                raise ValueError("restored source checkpoint does not match the collected snapshot")
            restore_fingerprint = env.state_fingerprint()
            execution = episode_runner(env, actor, spec=study, seed=seed,
                                       checkpoint=copy.deepcopy(checkpoint), first_action=copy.deepcopy(action),
                                       capture_checkpoints=False)
            if not isinstance(execution, dict):
                raise ValueError("episode runner must return a trace object")
            arm.update(copy.deepcopy(execution))
            arm.update(first_action=copy.deepcopy(action), restore_fingerprint=restore_fingerprint)
            if env.configuration() != plan["environment_config"] or actor.configuration() != plan["actor_config"]:
                raise ValueError("source branch changed frozen configuration")
            if arm.get("actor_config") != plan["actor_config"] or arm.get("environment_config") != plan["environment_config"]:
                raise ValueError("episode configuration does not match replay plan")
            if arm.get("status") != "error":
                _check_source_response_identity(arm, plan["source_response_identity"])
            if arm.get("utility") is None and arm.get("status") != "error":
                arm.update(status="error", success=None, utility=None,
                           error={"type": "UnmeasuredUtility", "reason": "required_costs_unknown"})
        except Exception as exc:
            arm.update(status="error", success=None, utility=None, error={"type": type(exc).__name__, "reason": "source_branch_failed"})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    arm.update(status="error", success=None, utility=None, error={"type": "CleanupError", "reason": "source_cleanup_failed"})
        row["arms"][name] = arm
    original, corrected = row["arms"]["original"], row["arms"]["corrected"]
    row["valid"] = (original["status"] != "error" and corrected["status"] not in ("error", "invalid_action")
                    and corrected.get("termination_reason") not in ("invalid_actor_output", "model_output_truncated")
                    and all(arm.get("steps") and arm["steps"][0].get("action") == action
                            for arm, action in ((original, candidate["problem_action"]),
                                                (corrected, candidate["correction"])))
                    and original.get("restore_fingerprint") is not None
                    and original.get("restore_fingerprint") == corrected.get("restore_fingerprint"))
    row["record_hash"] = digest(row)
    return row


def _check_records(rows: list[dict], plan: dict) -> tuple[list[float], bool]:
    spec = ReplaySpec(**plan["replay_spec"])
    study = StudySpec(**plan["study"])
    candidate = plan["candidate"]
    lower, upper = score_bounds(study, spec)
    differences, usable = [], True
    if len(rows) > len(plan["schedule"]):
        raise ValueError("more replay records than preregistered samples")
    for index, row in enumerate(rows):
        if row.get("record_hash") != digest({k: v for k, v in row.items() if k != "record_hash"}):
            raise ValueError("source replay record checksum mismatch")
        seed = plan["schedule"][index]
        expected = {"schema_version": _protocol(plan)[0], "plan_hash": plan["plan_hash"],
                    "candidate_hash": candidate["candidate_hash"], "checkpoint_hash": candidate["checkpoint_hash"],
                    "replicate": index, "seed": seed,
                    "sample_id": digest([plan["plan_hash"], index, seed])}
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("source replay sample identity/order/protocol mismatch")
        if set(row.get("arms", {})) != {"original", "corrected"} or sorted(row.get("arm_order", [])) != ["corrected", "original"]:
            raise ValueError("source replay requires exactly two named arms")
        scores = []
        row_usable = True
        for name in ("original", "corrected"):
            arm = row["arms"][name]
            expected_action = candidate["problem_action"] if name == "original" else candidate["correction"]
            if arm.get("first_action") != expected_action:
                raise ValueError("source replay tested a different first action")
            if arm.get("status") == "error":
                row_usable = False
                continue
            if arm.get("status") not in ("terminal", "budget_exhausted", "invalid_action") or type(arm.get("success")) is not bool:
                raise ValueError("source arm has an invalid outcome status")
            if arm["status"] != "terminal" and arm["success"]:
                raise ValueError("nonterminal source arm cannot claim success")
            if name == "corrected" and (arm["status"] == "invalid_action"
                    or arm.get("termination_reason") in ("invalid_actor_output", "model_output_truncated")):
                row_usable = False
            if arm.get("actor_config") != plan["actor_config"] or arm.get("environment_config") != plan["environment_config"]:
                raise ValueError("source replay configuration differs from registered source")
            _check_source_response_identity(arm, plan["source_response_identity"])
            if (arm.get("initial_checkpoint_hash") != candidate["checkpoint_hash"]
                    or arm.get("initial_state_fingerprint") != arm.get("restore_fingerprint")):
                raise ValueError("source episode started from a different checkpoint")
            caps = arm.get("capabilities", {})
            if caps.get("actor", {}).get("stateless") is not True or caps.get("environment", {}).get("restore_certified") is not True:
                raise ValueError("source arm lacks restoration/stateless certification")
            steps = arm.get("steps", [])
            if not steps or steps[0].get("action") != expected_action:
                # The adapter may repair an invalid proposed action into a
                # different executable command. That branch is measured, but
                # it cannot confirm the candidate's declared replacement.
                row_usable = False
            costs = arm.get("costs", {})
            for key, bound in {"model_calls": study.max_model_calls, "tool_calls": study.max_steps, **spec.cost_upper_bounds}.items():
                if costs.get(key) is None or finite(costs[key], key, minimum=0) > bound:
                    raise ValueError("source branch exceeded/missed declared cost bound: " + key)
            score = finite(arm.get("utility"), "source utility")
            recomputed = utility(arm["success"], costs, study)
            if recomputed is None or not math.isclose(score, recomputed, rel_tol=0, abs_tol=1e-10) or not lower <= score <= upper:
                raise ValueError("source utility does not match measured costs and registered bounds")
            scores.append(score)
        original, corrected = row["arms"]["original"], row["arms"]["corrected"]
        matched = original.get("restore_fingerprint") is not None and original.get("restore_fingerprint") == corrected.get("restore_fingerprint")
        row_usable = row_usable and matched
        if row.get("valid") is not row_usable:
            raise ValueError("source pair validity marker does not match its arms")
        usable = usable and row_usable
        if row_usable:
            differences.append(scores[1] - scores[0])
    return differences, usable


def evaluate_evidence(plan: dict, pairs: list[dict]) -> dict:
    """Recompute fixed-K statistics from every planned pair, preserving errors."""
    _, evidence_schema, decision_rule = _protocol(plan)
    if plan.get("plan_hash") != digest({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("source replay plan checksum mismatch")
    study, spec = StudySpec(**plan["study"]), ReplaySpec(**plan["replay_spec"])
    expected_schedule = [_seed(study, plan["candidate"], i) for i in range(spec.replay_pairs)]
    if plan.get("schedule") != expected_schedule or plan.get("score_bounds") != list(score_bounds(study, spec)):
        raise ValueError("replay plan schedule or bounded utility changed")
    if len(expected_schedule) != len(set(expected_schedule)):
        raise ValueError("replay schedule contains duplicate seeds")
    differences, all_valid = _check_records(pairs, plan)
    identities = [plan["source_response_identity"]]
    identities.extend(episode_response_identity(arm)
                      for row in pairs for arm in row["arms"].values()
                      if arm.get("status") != "error")
    fingerprints = {identity["fingerprint"] for identity in identities if identity["fingerprint"] is not None}
    if len(fingerprints) > 1:
        raise ValueError("served actor fingerprint changed across source replay pairs")
    identity_audit = {"source": copy.deepcopy(plan["source_response_identity"]),
                      "observed_fingerprints": sorted(fingerprints),
                      "fingerprint_status": "unknown" if any(identity["model"] is not None and identity["fingerprint_status"] == "unknown"
                                                               for identity in identities) else "reported",
                      "limitation": "reported model/fingerprint checks do not certify remote weights or seed coupling"}
    report = {"schema_version": evidence_schema, "plan": copy.deepcopy(plan),
              "pairs": copy.deepcopy(pairs), "n_replay_pairs": len(pairs),
              "n_usable_pairs": len(differences), "v": "?", "reason": "incomplete_replay",
              "mean_difference": None, "decision_rule": decision_rule,
              "endpoint": "terminal_utility", "scope": plan["scope"],
              "response_identity_audit": identity_audit,
              "claim_boundary": "source_checkpoint_action_repair_only;mechanism_and_target_reminder_effect_unproven",
              "independent_source_tasks": 1}
    current_rule = evidence_schema == SOURCE_EVIDENCE_SCHEMA
    if current_rule:
        report["n_successful_corrections"] = sum(row["arms"]["corrected"].get("success") is True
                                                  for row in pairs)
        report["all_corrections_successful"] = None
    if len(pairs) == spec.replay_pairs and all_valid:
        average = mean(differences)
        corrections_succeeded = all(row["arms"]["corrected"]["success"] is True for row in pairs)
        passed = average > spec.acceptance_epsilon and (not current_rule or corrections_succeeded)
        report.update(mean_difference=average,
                      v=1 if passed else 0,
                      reason="supported_source_repair" if passed else "improvement_not_supported")
        if current_rule:
            report["all_corrections_successful"] = corrections_succeeded
            if not corrections_succeeded:
                report["reason"] = "correction_not_consistently_successful"
    elif len(pairs) == spec.replay_pairs:
        report["reason"] = "replay_contains_error_or_invalid_repair"
    report["evidence_hash"] = digest(report)
    return report


def validate_evidence(evidence: dict, *, candidate: dict | None = None, trace: dict | None = None) -> dict:
    if evidence.get("schema_version") not in ("source-evidence.v5", SOURCE_EVIDENCE_SCHEMA):
        raise ValueError("unsupported source evidence")
    plan = evidence["plan"]
    if evidence["schema_version"] != _protocol(plan)[1]:
        raise ValueError("source evidence and plan schema versions differ")
    if candidate is not None and canonical(candidate) != canonical(plan["candidate"]):
        raise ValueError("evidence belongs to a different repair hypothesis")
    if trace is not None:
        expected = _make_plan(plan["candidate"], trace, StudySpec(**plan["study"]),
                              ReplaySpec(**plan["replay_spec"]), plan["schema_version"])
        if canonical(plan) != canonical(expected):
            raise ValueError("source evidence plan differs from its provenance")
    recomputed = evaluate_evidence(plan, evidence["pairs"])
    if canonical(recomputed) != canonical(evidence):
        raise ValueError("source evidence statistics or checksum changed")
    return {"v": evidence["v"], "reason": evidence["reason"], "scope": evidence["scope"]}


def require_current_evidence(evidence: dict, *, replay_pairs: int | None = None) -> dict:
    """Require current successful evidence; optionally freeze a pilot's K.

    Legacy evidence remains readable through validate_evidence/load_bank, but
    cannot be silently promoted into a run requiring the new source rule.
    """
    result = validate_evidence(evidence)
    if (evidence["schema_version"] != SOURCE_EVIDENCE_SCHEMA
            or evidence["decision_rule"] != SOURCE_DECISION_RULE or result["v"] != 1):
        raise ValueError("target memory requires current all-corrections-success evidence")
    if replay_pairs is not None:
        if type(replay_pairs) is not int or replay_pairs < 1:
            raise ValueError("required replay_pairs must be a positive integer")
        if evidence["plan"]["replay_spec"]["replay_pairs"] != replay_pairs:
            raise ValueError("source replay count differs from the target protocol")
    return result


def verify_candidate(candidate: dict, trace: dict, env_factory: Callable, actor_factory: Callable, *,
                     study: StudySpec, replay_spec: ReplaySpec, output_dir: str | Path | None = None,
                     episode_runner: Callable | None = None) -> dict:
    """Run exactly K preregistered source replay pairs. Never retry errors."""
    if episode_runner is None:
        from cave_mem_v3.agents.runtime import run_episode
        episode_runner = run_episode
    plan = make_plan(candidate, trace, study, replay_spec)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    checkpoint = copy.deepcopy(trace["episode"]["steps"][candidate["step_index"]]["checkpoint"])
    pairs = []
    for index in range(replay_spec.replay_pairs):
        row = _collect_pair(plan, checkpoint, env_factory, actor_factory, index, episode_runner)
        pairs.append(row)
        if output is not None:
            write_json(output / "pairs" / f"{index:06d}.json", row)
    evidence = evaluate_evidence(plan, pairs)
    if output is not None:
        # The plan on disk is part of the audit trail and may not be rewritten.
        if canonical(read_json(output / "plan.json")) != canonical(plan):
            raise ValueError("frozen replay plan changed during collection")
        write_json(output / "evidence.json", evidence)
    return evidence
