"""Independent source action-repair confirmation for proposal equation (6).

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


@dataclass(frozen=True)
class ReplaySpec:
    screening_pairs: int = 4
    confirmation_pairs: int = 32
    confidence: float = 0.95
    acceptance_epsilon: float = 0.0
    family_size: int = 1
    cost_upper_bounds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("screening_pairs", "confirmation_pairs", "family_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < finite(self.confidence, "confidence") < 1:
            raise ValueError("confidence must be inside (0,1)")
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


def _seed(study: StudySpec, candidate: dict, cohort: str, replicate: int) -> int:
    return stage_seed(study.base_seed, candidate["candidate_hash"], replicate, "source:" + cohort)


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
    schedule = {cohort: [_seed(study, candidate, cohort, i) for i in range(count)]
                for cohort, count in (("screening", spec.screening_pairs), ("confirmation", spec.confirmation_pairs))}
    all_seeds = schedule["screening"] + schedule["confirmation"]
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("replay seed collision; choose another base seed before collection")
    plan = {"schema_version": "source-replay-plan.v3", "candidate": copy.deepcopy(candidate),
            "source_trace_hash": trace["trace_hash"], "study": plain(study), "replay_spec": plain(spec),
            "actor_config": episode["actor_config"], "environment_config": episode["environment_config"],
            "capabilities": capabilities, "score_bounds": list(score_bounds(study, spec)),
            "source_response_identity": source_response_identity,
            "schedule": schedule, "endpoint": "terminal_utility",
            "scope": candidate["scope"], "coupling": "same_checkpoint_stage_step_seed;provider_noise_as_declared"}
    plan["plan_hash"] = digest(plan)
    return plan


def _collect_pair(plan: dict, checkpoint: dict, env_factory: Callable, actor_factory: Callable,
                  cohort: str, replicate: int, episode_runner: Callable) -> dict:
    study = StudySpec(**plan["study"])
    candidate = plan["candidate"]
    seed = plan["schedule"][cohort][replicate]
    row = {"schema_version": "source-replay-pair.v3", "plan_hash": plan["plan_hash"],
           "candidate_hash": candidate["candidate_hash"], "checkpoint_hash": candidate["checkpoint_hash"],
           "cohort": cohort, "replicate": replicate, "seed": seed,
           "sample_id": digest([plan["plan_hash"], cohort, replicate, seed]), "arms": {}, "valid": False}
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
                    and original.get("restore_fingerprint") is not None
                    and original.get("restore_fingerprint") == corrected.get("restore_fingerprint"))
    row["record_hash"] = digest(row)
    return row


def _check_records(rows: list[dict], plan: dict, cohort: str) -> tuple[list[float], bool]:
    spec = ReplaySpec(**plan["replay_spec"])
    study = StudySpec(**plan["study"])
    candidate = plan["candidate"]
    lower, upper = score_bounds(study, spec)
    differences, usable = [], True
    if len(rows) > len(plan["schedule"][cohort]):
        raise ValueError("more replay records than preregistered samples")
    for index, row in enumerate(rows):
        if row.get("record_hash") != digest({k: v for k, v in row.items() if k != "record_hash"}):
            raise ValueError("source replay record checksum mismatch")
        seed = plan["schedule"][cohort][index]
        expected = {"schema_version": "source-replay-pair.v3", "plan_hash": plan["plan_hash"],
                    "candidate_hash": candidate["candidate_hash"], "checkpoint_hash": candidate["checkpoint_hash"],
                    "cohort": cohort, "replicate": index, "seed": seed,
                    "sample_id": digest([plan["plan_hash"], cohort, index, seed])}
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
            if name == "corrected" and arm["status"] == "invalid_action":
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
                raise ValueError("source first action was not executed/recorded")
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


def evaluate_evidence(plan: dict, screening: list[dict], confirmation: list[dict]) -> dict:
    """Recompute statistics from all preplanned attempts, without discarding errors."""
    if plan.get("plan_hash") != digest({k: v for k, v in plan.items() if k != "plan_hash"}):
        raise ValueError("source replay plan checksum mismatch")
    study, spec = StudySpec(**plan["study"]), ReplaySpec(**plan["replay_spec"])
    expected_schedule = {cohort: [_seed(study, plan["candidate"], cohort, i) for i in range(count)]
                         for cohort, count in (("screening", spec.screening_pairs), ("confirmation", spec.confirmation_pairs))}
    if plan.get("schedule") != expected_schedule or plan.get("score_bounds") != list(score_bounds(study, spec)):
        raise ValueError("replay plan schedule or bounded utility changed")
    seeds = expected_schedule["screening"] + expected_schedule["confirmation"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("screening and confirmation must use unique nonoverlapping seeds")
    screen_d, screen_valid = _check_records(screening, plan, "screening")
    confirm_d, confirm_valid = _check_records(confirmation, plan, "confirmation")
    identities = [plan["source_response_identity"]]
    identities.extend(episode_response_identity(arm)
                      for row in [*screening, *confirmation] for arm in row["arms"].values()
                      if arm.get("status") != "error")
    fingerprints = {identity["fingerprint"] for identity in identities if identity["fingerprint"] is not None}
    if len(fingerprints) > 1:
        raise ValueError("served actor fingerprint changed across source replay cohorts")
    identity_audit = {"source": copy.deepcopy(plan["source_response_identity"]),
                      "observed_fingerprints": sorted(fingerprints),
                      "fingerprint_status": "unknown" if any(identity["model"] is not None and identity["fingerprint_status"] == "unknown"
                                                               for identity in identities) else "reported",
                      "limitation": "reported model/fingerprint checks do not certify remote weights or seed coupling"}
    screening_passed = (len(screening) == spec.screening_pairs and screen_valid and mean(screen_d) > spec.acceptance_epsilon)
    if confirmation and not screening_passed:
        raise ValueError("confirmation occurred without the registered screening gate passing")
    report = {"schema_version": "source-evidence.v3", "plan": copy.deepcopy(plan),
              "screening": copy.deepcopy(screening), "confirmation": copy.deepcopy(confirmation),
              "screening_passed": screening_passed, "n_confirmation": len(confirmation),
              "n_usable_confirmation": len(confirm_d), "v": "?", "reason": "incomplete_confirmation",
              "mean_difference": None, "lower_bound": None, "upper_bound": None,
              "endpoint": "terminal_utility", "scope": plan["scope"],
              "response_identity_audit": identity_audit,
              "claim_boundary": "source_checkpoint_action_repair_only;mechanism_and_target_reminder_effect_unproven",
              "independent_source_tasks": 1,
              "interval_assumptions": "bounded utility;independent replicate randomness conditional on source checkpoint;declared provider coupling"}
    if not screening_passed:
        report["reason"] = "screening_rejected" if screen_valid and len(screening) == spec.screening_pairs else "screening_incomplete_or_error"
    elif not confirm_valid:
        report["reason"] = "confirmation_contains_error_or_invalid_repair"
    elif len(confirmation) == spec.confirmation_pairs:
        width = plan["score_bounds"][1] - plan["score_bounds"][0]
        alpha = (1.0 - spec.confidence) / spec.family_size
        radius = width * math.sqrt(2.0 * math.log(2.0 / alpha) / len(confirm_d))
        average = mean(confirm_d)
        lower, upper = max(-width, average - radius), min(width, average + radius)
        passed = lower > spec.acceptance_epsilon
        report.update(mean_difference=average, lower_bound=lower, upper_bound=upper,
                      v=1 if passed else 0,
                      reason="supported_source_repair" if passed else "improvement_not_supported")
    report["evidence_hash"] = digest(report)
    return report


def validate_evidence(evidence: dict, *, candidate: dict | None = None, trace: dict | None = None) -> dict:
    if evidence.get("schema_version") != "source-evidence.v3":
        raise ValueError("unsupported source evidence")
    plan = evidence["plan"]
    if candidate is not None and canonical(candidate) != canonical(plan["candidate"]):
        raise ValueError("evidence belongs to a different repair hypothesis")
    if trace is not None:
        expected = make_plan(plan["candidate"], trace, StudySpec(**plan["study"]), ReplaySpec(**plan["replay_spec"]))
        if canonical(plan) != canonical(expected):
            raise ValueError("source evidence plan differs from its provenance")
    recomputed = evaluate_evidence(plan, evidence["screening"], evidence["confirmation"])
    if canonical(recomputed) != canonical(evidence):
        raise ValueError("source evidence statistics or checksum changed")
    return {"v": evidence["v"], "reason": evidence["reason"], "scope": evidence["scope"]}


def verify_candidate(candidate: dict, trace: dict, env_factory: Callable, actor_factory: Callable, *,
                     study: StudySpec, replay_spec: ReplaySpec, output_dir: str | Path | None = None,
                     episode_runner: Callable | None = None) -> dict:
    """Run fixed screening then independent fixed confirmation. Never retry errors."""
    if episode_runner is None:
        from cave_mem_v3.agents.runtime import run_episode
        episode_runner = run_episode
    plan = make_plan(candidate, trace, study, replay_spec)
    output = Path(output_dir) if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "plan.json", plan)
    checkpoint = copy.deepcopy(trace["episode"]["steps"][candidate["step_index"]]["checkpoint"])
    screening, confirmation = [], []
    for cohort, count, rows in (("screening", replay_spec.screening_pairs, screening),
                                ("confirmation", replay_spec.confirmation_pairs, confirmation)):
        if cohort == "confirmation" and not evaluate_evidence(plan, screening, [])["screening_passed"]:
            break
        for index in range(count):
            row = _collect_pair(plan, checkpoint, env_factory, actor_factory, cohort, index, episode_runner)
            rows.append(row)
            if output is not None:
                write_json(output / cohort / f"{index:06d}.json", row)
    evidence = evaluate_evidence(plan, screening, confirmation)
    if output is not None:
        # The plan on disk is part of the audit trail and may not be rewritten.
        if canonical(read_json(output / "plan.json")) != canonical(plan):
            raise ValueError("frozen replay plan changed during collection")
        write_json(output / "evidence.json", evidence)
    return evidence
