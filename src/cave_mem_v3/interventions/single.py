"""Randomized single-arm collection in one live session, without cloning."""
from __future__ import annotations
from cave_mem_v3.target_data.partitions import target_partition, partition_fields, partition_tasks, check_partition_resume

import copy
import random
import time
from pathlib import Path

from cave_mem_v3.agents.providers import ActorResponse, ProviderError
from cave_mem_v3.agents.runtime import add_usage, empty_costs
from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import PublicContext, StudySpec, digest, finite, plain, read_json, stage_seed, write_json
from cave_mem_v3.interventions.execution import TargetPendingError, run_live_target_arm
from cave_mem_v3.interventions.inputs import _validate_source_partition, checked, seal
from cave_mem_v3.interventions.reminder import render_reminder
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.replay import require_current_evidence
from cave_mem_v3.target_data.collection import _close, _exclusive_json
from cave_mem_v3.target_data.pre_treatment_state import build_pre_treatment_state, validate_exposure_rules
from cave_mem_v3.target_data.retrieval import retrieve_memory
from cave_mem_v3.target_data.sampling import OnlineCheckpointSampler, OrdinaryActorSampler, RiskHazard


PROTOCOL = "live_session_single_randomized.v1"


class _JournalError(RuntimeError):
    pass


class _IdentityError(ValueError):
    pass


def single_assignment(run, item, reminder):
    """Freeze one treatment draw with known nonzero probabilities for both arms."""
    probability = run["assignment_probability"]
    assignment_seed = stage_seed(run["study"]["base_seed"], item["trace_id"], 0, "single-assignment")
    draw = random.Random(assignment_seed).random()
    arm = "remind" if draw < probability else "silent"
    probabilities = {"silent": 1.0 - probability, "remind": probability}
    return seal({"schema_version": "target-single-assignment.v1", "run_hash": run["record_hash"],
        "input_hash": item["record_hash"], "task_id": item["task_id"], "trace_id": item["trace_id"],
        "probability_remind": probability, "b": probabilities, "assignment_seed": assignment_seed,
        "draw": draw, "arm": arm, "z": int(arm == "remind"), "propensity": probabilities[arm],
        "execution_seed": stage_seed(run["study"]["base_seed"], item["trace_id"], 0, "single-execution"),
        "reminder": reminder, "budget": {"remaining_model_calls": item["remaining_model_calls"],
            "remaining_steps": item["remaining_steps"], "continuation_model_calls": item["remaining_model_calls"] - 1,
            "intervention_reserved_calls": 1}, "protocol": PROTOCOL})


def _journal(directory, owner_hash, phase):
    previous, count = None, 0
    def emit(event):
        nonlocal previous, count
        if event.get("event_index") != count:
            raise _JournalError("noncontiguous single-session event journal")
        value = seal({"schema_version": "target-single-event.v1", "owner_hash": owner_hash,
                      "phase": phase, "previous_hash": previous, "event": event})
        try:
            _exclusive_json(directory / f"{count:06d}.json", value)
        except Exception as exc:
            raise _JournalError("single-session journal failed; do not retry") from exc
        previous, count = value["record_hash"], count + 1
    return emit


def _check_live_protocol(env, actor):
    if actor.capabilities().get("stateless") is not True:
        raise ValueError("single-session collection requires a stateless frozen actor")
    config, caps = env.configuration(), env.capabilities()
    from cave_mem_v3.environments.base import PreparationEnvironment
    if isinstance(env, PreparationEnvironment) and config.get("research_role") == "fixture_only":
        return
    # A new live backend needs a concrete reviewed adapter; unsupported systems
    # are never made eligible by failing paired restoration.
    from cave_mem_v3.environments.terminal import TerminalEnvironment, PROTOCOL as TERMINAL_PROTOCOL, protocol_hash
    if not isinstance(env, TerminalEnvironment) or caps.get("randomized_single_supported") is not True:
        raise ValueError("live adapter outside reviewed Terminal filesystem protocol; adapter review required")
    audit = config.get("protocol", {}).get("audit", {})
    if (audit.get("asset_hash") != config.get("asset_hash")
            or audit.get("repository_commit") != config.get("repository_commit")
            or audit.get("backend") != config.get("backend")
            or audit.get("scope") != TERMINAL_PROTOCOL
            or audit.get("protocol_hash") != protocol_hash(config["protocol"])
            or not audit.get("reviewer")
            or not audit.get("review_notes") or audit.get("reviewed_constraints") != {
                "no_external_state": True, "no_persistent_services": True,
                "filesystem_boundary_semantics": True, "clock_and_rng_are_exogenous": True}):
        raise ValueError("Terminal live task scope has not been reviewed")


def _prepare_live(run, task, trace_id, seed, env, actor, encoder, emit, expected_identity=None):
    """Stop before validating/executing the selected draft; retain the live env."""
    started = time.perf_counter()
    costs = empty_costs()
    operational_seconds, event_index = 0.0, 0
    identity = {"model": None, "fingerprint": None, "fingerprint_status": "not_observed"}
    record = {"schema_version": "target-single-preparation.v1", "run_hash": run["record_hash"],
              "task_id": task["task_id"], "trace_id": trace_id, "seed": seed,
              "status": "error", "steps": [], "costs": costs, "response_identity": identity,
              "error": None, "selection": None}
    no_memory = run.get("bundle", {}).get("policy_config", {}).get("kind") == "no_memory"
    sampler_type = OrdinaryActorSampler if no_memory else OnlineCheckpointSampler
    sampler = sampler_type(role=run["role"], task_id=task["task_id"],
        trace_id=trace_id, seed=seed, hazard=RiskHazard(**run["risk_hazard"]),
        final_authorization=run.get("final_authorization"))
    spec = StudySpec(**run["study"])
    pending = None
    finalization_seconds = 0.0
    finalization_calls = 0
    original_finalize = getattr(env, "finalize", None)
    finalize_instance_value = vars(env).get("finalize")
    finalize_had_instance_value = "finalize" in vars(env)
    finalize_meter_installed = False
    def event(kind, **fields):
        nonlocal event_index
        emit({"event_index": event_index, "kind": kind, **plain(fields)})
        event_index += 1
    def frozen():
        if actor.configuration() != run["actor_config"] or env.configuration() != run["environment_configs"][task["task_id"]]:
            raise _IdentityError("frozen live actor/environment configuration changed")
        if not no_memory and encoder.identity != run["encoder_identity"]:
            raise _IdentityError("frozen encoder identity changed")
    def grade_stopped_prefix():
        terminal, success = env.terminal_status()
        if not terminal and callable(getattr(env, "finalize", None)):
            event("finalize.started", reason=record.get("termination_reason"))
            final = env.finalize()
            if not final.terminal or type(final.success) is not bool:
                raise ValueError("prefix verifier did not supply an explicit outcome")
            terminal, success = final.terminal, final.success
            event("finalize.completed", result=final, private_verifier=getattr(env, "verifier_report", None))
        record["unselected_terminal_outcome"] = success if terminal and type(success) is bool else None
        record["outcome_status"] = "observed" if type(record["unselected_terminal_outcome"]) is bool else "unknown"
        if getattr(env, "verifier_report", None) is not None:
            record["private_verifier"] = copy.deepcopy(env.verifier_report)
    try:
        _check_live_protocol(env, actor)
        frozen()
        event("reset.started", seed=seed)
        reset_started = time.perf_counter()
        context = env.reset(seed)
        operational_seconds += time.perf_counter() - reset_started
        if not isinstance(context, PublicContext) or context.task_id != task["task_id"]:
            raise ValueError("live reset returned an invalid public context")
        event("reset.completed", context=context)
        if callable(original_finalize):
            def measured_finalize(*args, **kwargs):
                nonlocal finalization_seconds, finalization_calls
                finalization_calls += 1
                begin = time.perf_counter()
                try:
                    return original_finalize(*args, **kwargs)
                finally:
                    finalization_seconds += time.perf_counter() - begin
            env.finalize = measured_finalize
            finalize_meter_installed = True
        prior = []
        for index in range(spec.max_steps):
            if context.remaining_steps < 1 or index >= spec.max_model_calls or env.terminal_status()[0]:
                break
            frozen()
            # The sampler accepts a descriptor for bookkeeping. It is explicitly
            # not an environment checkpoint and is never used for restoration.
            point = {"schema_version": "live-public-point.v1", "context_hash": digest(context), "index": index}
            decision = sampler.consider(context, point, index, prior, checkpoint_hash=digest(point))
            event("selection.considered", index=index, context=context, point=point, decision=decision)
            step = {"index": index, "context": plain(context), "point": point,
                    "selection": decision, "actor_response": None, "action": None, "result": None}
            record["steps"].append(step)
            if decision.get("status") == "sampling_incomplete":
                raise ValueError("checkpoint sampling incomplete; stop before actor request")
            call_seed = stage_seed(seed, context.task_id, 0, "single-prefix", index)
            call_id = f"single:{trace_id}:prefix:{index}"
            event("actor.started", call_id=call_id, stage="target_single_prefix", seed=call_seed, context=context)
            call_started = time.perf_counter()
            try:
                response = actor.propose(context, stage="target_single_prefix", seed=call_seed,
                                         reminder=None, call_id=call_id)
            except ProviderError as exc:
                add_usage(costs, exc.attempt.get("usage", {}))
                if exc.attempt.get("error_class") in ("TimeoutError", "URLError", "ConnectionError", "ConnectionResetError"):
                    event("actor.pending", call_id=call_id, attempt=exc.attempt)
                    raise TargetPendingError(record, exc.attempt) from exc
                event("actor.failed", call_id=call_id, attempt=exc.attempt)
                if exc.attempt.get("identity_error"):
                    raise _IdentityError("prefix provider response identity changed") from exc
                raise
            except (TimeoutError, ConnectionError) as exc:
                attempt = {"call_id": call_id, "status": "pending", "error_class": type(exc).__name__,
                           "usage": {"model_calls": 1.0}}
                add_usage(costs, attempt["usage"])
                event("actor.pending", call_id=call_id, attempt=attempt)
                raise TargetPendingError(record, attempt) from exc
            except Exception as exc:
                add_usage(costs, {"model_calls": 1.0})
                event("actor.failed", call_id=call_id, error_class=type(exc).__name__)
                raise
            finally:
                costs["wall_seconds"] += time.perf_counter() - call_started
            if not isinstance(response, ActorResponse):
                add_usage(costs, {"model_calls": 1.0})
                event("actor.failed", call_id=call_id, error_class="InvalidActorResponse")
                raise ValueError("prefix returned invalid ActorResponse")
            add_usage(costs, response.usage)
            step.update(actor_response=plain(response), action=copy.deepcopy(response.action))
            event("actor.completed", call_id=call_id, response=response)
            if response.usage.get("model_calls") != 1 or response.usage.get("physical_attempts") not in (0, 1):
                raise ValueError("prefix requires one metered non-retried actor call")
            if not response.model or (identity["model"] is not None and identity["model"] != response.model):
                raise _IdentityError("prefix actor model identity changed")
            if identity["fingerprint"] and response.fingerprint and identity["fingerprint"] != response.fingerprint:
                raise _IdentityError("prefix actor fingerprint changed")
            for key, value in (("model", response.model), ("fingerprint", response.fingerprint)):
                if value is not None and (expected_identity or {}).get(key) is not None and value != expected_identity[key]:
                    raise _IdentityError("prefix actor identity changed across tasks")
            missing = identity["fingerprint_status"] == "unknown" or not response.fingerprint
            identity.update(model=response.model, fingerprint=response.fingerprint or identity["fingerprint"],
                            fingerprint_status="unknown" if missing else "reported")
            frozen()
            if response.parse_error is not None:
                step["validation_error"] = "malformed_actor_action"
                record.update(status="preparation_error" if decision.get("selected") else "no_selection",
                              actor_behavior="malformed_action", termination_reason="invalid_actor_output")
                event("action.rejected", index=index, reason="malformed_actor_action")
                break
            if decision.get("selected"):
                retrieval_started = time.perf_counter()
                try:
                    retrieval = retrieve_memory(run["bank"], context=context, action=response.action, encoder=encoder)
                    state = build_pre_treatment_state(context=context, action=response.action, retrieval=retrieval,
                        bank=run["bank"], encoder=encoder, remaining_computation={
                            "model_calls": spec.max_model_calls - index - 1,
                            "environment_steps": min(spec.max_steps - index, context.remaining_steps)},
                        exposure_rules=run["exposure_rules"])
                finally:
                    costs["wall_seconds"] += time.perf_counter() - retrieval_started
                frozen()
                step["pre_treatment_state"] = state
                record["status"] = "prepared"
                break
            valid, reason = env.validate_action(response.action)
            if not valid:
                step["validation_error"] = reason
                event("action.rejected", index=index, reason=reason)
                record.update(status="no_selection", actor_behavior="invalid_action",
                              termination_reason="invalid_actor_output")
                break
            tool_seed = stage_seed(seed, task["task_id"], 0, "single-prefix-environment", index)
            event("step.started", index=index, action=response.action, seed=tool_seed)
            costs["tool_calls"] += 1
            step_started = time.perf_counter()
            finalization_before_step = finalization_seconds
            try:
                result = env.step(copy.deepcopy(response.action), seed=tool_seed)
            except Exception as exc:
                costs["tool_cost"] = None
                event("step.failed", index=index, error_class=type(exc).__name__)
                raise
            finally:
                costs["wall_seconds"] += max(0.0, time.perf_counter() - step_started
                                              - (finalization_seconds - finalization_before_step))
            costs["tool_cost"] = None if result.tool_cost is None or costs["tool_cost"] is None else costs["tool_cost"] + finite(result.tool_cost, "tool_cost", minimum=0)
            step["result"] = plain(result)
            event("step.completed", index=index, result=result)
            if not isinstance(result.context, PublicContext) or result.context.task_id != task["task_id"]:
                raise ValueError("invalid live prefix observation")
            prior.append({"action": response.action, "result": plain(result), "valid": result.valid})
            if not result.valid:
                record.update(status="no_selection", actor_behavior="invalid_action",
                              termination_reason="invalid_actor_output")
                break
            if result.terminal:
                record["unselected_terminal_outcome"] = result.success
                record["termination_reason"] = "task_terminal"
                break
            context = result.context
        if record["status"] != "prepared":
            if record["status"] != "preparation_error":
                record["status"] = "no_selection"
            record.setdefault("termination_reason", "budget_exhausted")
            grade_stopped_prefix()
        frozen()
    except TargetPendingError as exc:
        pending = exc
        raise
    except _JournalError:
        raise
    except Exception as exc:
        record.update(status="error", error={"type": type(exc).__name__, "message": str(exc),
            "scope": "run_shared" if isinstance(exc, _IdentityError) else "preparation"})
        event("preparation.failed", error=record["error"])
    finally:
        if finalize_meter_installed:
            if finalize_had_instance_value:
                env.finalize = finalize_instance_value
            else:
                del env.finalize
        record["selection"] = sampler.finalize()
        record["operational_costs"] = {"wall_seconds": max(0, time.perf_counter() - started - costs["wall_seconds"]),
                                       "reset_wall_seconds": operational_seconds,
                                       "finalization_wall_seconds": finalization_seconds,
                                       "finalization_calls": finalization_calls}
        if pending is not None:
            pending.partial_record = copy.deepcopy(record)
    return seal(record)


def _input(run, task, preparation):
    result = {"schema_version": "target-single-input.v1", "run_hash": run["record_hash"],
        "task_id": task["task_id"], "trace_id": preparation["trace_id"],
        "preparation_hash": preparation["record_hash"], "status": "blocked", "reason": "preparation_error"}
    if preparation["status"] == "no_selection":
        return seal({**result, "status": "excluded", "reason": "no_selection"})
    if preparation["status"] != "prepared":
        return seal(result)
    step = preparation["steps"][-1]
    state = step["pre_treatment_state"]
    remaining = state["x"]["b"]["remaining"]
    result.update(status="ready", reason=None, task=task, context=step["context"],
        a0=step["action"], a0_hash=state["a0_hash"], x_hash=state["x_hash"],
        pre_treatment_state=state, public_point_hash=digest(step["point"]),
        remaining_model_calls=remaining["model_calls"], remaining_steps=remaining["environment_steps"],
        shared_costs=preparation["costs"], response_identity=preparation["response_identity"],
        selection=preparation["selection"], actor_config=run["actor_config"],
        environment_config=run["environment_configs"][task["task_id"]])
    if state["memory_id"] is None:
        result.update(status="excluded", reason="no_memory")
    elif remaining["model_calls"] < 1 or remaining["environment_steps"] < 1:
        result.update(status="excluded", reason="insufficient_budget")
    return seal(result)


def collect_single_target(manifest, *, bank, role="target_train", spec, env_factory,
                          actor_factory, output_dir, encoder, hazard=None, exposure_rules=None,
                          assignment_probability, validation_role=None):
    """Freeze and randomize once, continue that same session, never retry attempts."""
    from cave_mem_v3.interventions.single_outcomes import summarize_target_single
    check_partition_resume(output_dir, role, validation_role)
    probability = finite(assignment_probability, "assignment_probability", minimum=0)
    if isinstance(assignment_probability, bool) or not 0 < probability < 1:
        raise ValueError("both single-arm logging probabilities must be strictly positive")
    validate_manifest(manifest, verify_assets=True)
    checked(bank, "bank_hash", "verified-source-bank.v3")
    if bank.get("frozen") is not True:
        raise ValueError("single-arm collection requires a frozen bank")
    _validate_source_partition({"manifest_hash": manifest["manifest_hash"], "manifest": manifest}, bank)
    memories = {m.memory_id: m for m in bank_memories(bank)} if bank["memories"] else {}
    for memory in memories.values():
        require_current_evidence(memory.evidence, replay_pairs=2)
    if not memories and (bank.get("memory_count") != 0 or bank.get("audit_sources") != {}):
        raise ValueError("invalid empty bank")
    validate_exposure_rules(exposure_rules)
    if not isinstance(getattr(encoder, "identity", None), dict) or not encoder.identity:
        raise ValueError("single-arm collection requires a frozen encoder identity")
    tasks = sorted(copy.deepcopy(partition_tasks(manifest, role, validation_role)), key=lambda t: t["task_id"])
    if not tasks:
        raise ValueError("manifest has no target_train tasks")
    random.Random(spec.base_seed).shuffle(tasks)
    hazard = hazard or RiskHazard()
    root = Path(output_dir)
    if (root / "run.json").exists():
        stored = checked(read_json(root / "run.json"), schema="target-single-run.v1")
        fixed = {"manifest": manifest, "bank": bank, "study": plain(spec), "role": role, **partition_fields(role, validation_role),
                 "task_order": [task["task_id"] for task in tasks], "risk_hazard": plain(hazard),
                 "encoder_identity": encoder.identity, "exposure_rules": exposure_rules,
                 "assignment_probability": probability, "protocol": PROTOCOL}
        if any(stored.get(key) != value for key, value in fixed.items()):
            raise ValueError("single-arm run inputs changed; choose a new output directory")
        existing = summarize_target_single(root)
        # Validate durable evidence before any factory, reset, actor call, or
        # environment step. Even a partial session must never be reconstructed.
        for index, task in enumerate(tasks):
            trace_id = digest([stored["record_hash"], task["task_id"]])[:24]
            directory = root / "sessions" / f"{index:06d}-{trace_id}"
            if (directory / "attempt.json").exists() and not (directory / "closed.json").exists():
                raise ValueError("incomplete single-session attempt exists; preserve and reconcile before retry")
        if all((root / "sessions" / f"{index:06d}-{digest([stored['record_hash'], task['task_id']])[:24]}" / "closed.json").exists()
               for index, task in enumerate(tasks)):
            return existing
        for index, task in enumerate(tasks):
            trace_id = digest([stored["record_hash"], task["task_id"]])[:24]
            directory = root / "sessions" / f"{index:06d}-{trace_id}"
            if not (directory / "closed.json").exists():
                continue
            closed = read_json(directory / "closed.json")
            preparation = read_json(directory / "preparation.json")
            result = read_json(directory / "result.json")["record"] if (directory / "result.json").exists() else {}
            if (closed.get("cleanup_errors") or (preparation.get("error") or {}).get("scope") == "run_shared"
                    or (result.get("error") or {}).get("scope") in ("pair_shared", "session_shared")):
                raise ValueError("single run stopped on a shared protocol failure; do not continue or retry")
    used = []
    actor = actor_factory()
    used.append(actor)
    configs, capabilities = {}, {}
    try:
        actor_config, actor_capabilities = actor.configuration(), actor.capabilities()
        for task in tasks:
            env = env_factory(copy.deepcopy(task))
            if any(env is old for old in used):
                raise ValueError("factory reused a live environment instance")
            used.append(env)
            try:
                _check_live_protocol(env, actor)
                configs[task["task_id"]], capabilities[task["task_id"]] = env.configuration(), env.capabilities()
            finally:
                _close(env)
    finally:
        _close(actor)
    run = seal({"schema_version": "target-single-run.v1", "manifest": manifest,
        "manifest_hash": manifest["manifest_hash"], "bank": bank, "bank_hash": bank["bank_hash"],
        "study": plain(spec), "role": role, **partition_fields(role, validation_role), "task_order": [t["task_id"] for t in tasks],
        "actor_config": actor_config, "actor_capabilities": actor_capabilities,
        "environment_configs": configs, "environment_capabilities": capabilities,
        "risk_hazard": plain(hazard), "encoder_identity": encoder.identity,
        "exposure_rules": exposure_rules, "exposure_rules_hash": digest(exposure_rules) if exposure_rules else None,
        "assignment_probability": probability, "protocol": PROTOCOL})
    if (root / "run.json").exists():
        if read_json(root / "run.json") != run:
            raise ValueError("single-arm run inputs changed; choose a new output directory")
        existing = summarize_target_single(root)
        if existing.get("pending_sessions"):
            raise ValueError("incomplete single-session attempt exists; preserve and reconcile before retry")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("new single-arm output directory must be empty")
        _exclusive_json(root / "run.json", run)
    reference = {"model": None, "fingerprint": None}
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        seed = stage_seed(spec.base_seed, task["task_id"], 0, "single-prefix-rollout")
        directory = root / "sessions" / f"{index:06d}-{trace_id}"
        if (directory / "closed.json").exists():
            for filename in ("preparation.json", "result.json"):
                if (directory / filename).exists():
                    saved = read_json(directory / filename)
                    identity = saved.get("record", saved).get("response_identity", {})
                    for key in reference:
                        if identity.get(key) is not None:
                            if reference[key] is not None and reference[key] != identity[key]:
                                raise ValueError("completed single sessions have inconsistent actor identities")
                            reference[key] = identity[key]
            continue
        if (directory / "attempt.json").exists():
            raise ValueError("incomplete single-session attempt exists; preserve and reconcile before retry")
        _exclusive_json(directory / "attempt.json", seal({"schema_version": "target-single-attempt.v1",
            "run_hash": run["record_hash"], "trace_id": trace_id, "task_id": task["task_id"],
            "seed": seed, "status": "started"}))
        env = actor = None
        item = assignment = result = pending = None
        cleanup_errors = []
        try:
            actor = actor_factory()
            if any(actor is old for old in used):
                raise ValueError("factory reused an actor across live sessions")
            used.append(actor)
            env = env_factory(copy.deepcopy(task))
            if any(env is old for old in used):
                raise ValueError("factory reused an environment across live sessions")
            used.append(env)
            preparation = _prepare_live(run, task, trace_id, seed, env, actor, encoder,
                _journal(directory / "preparation_events", run["record_hash"], "preparation"), reference)
            _exclusive_json(directory / "preparation.json", preparation)
            item = _input(run, task, preparation)
            _exclusive_json(directory / "input.json", item)
            if item["status"] == "ready":
                reminder = render_reminder(memories[item["pre_treatment_state"]["memory_id"]], item["a0"])
                assignment = single_assignment(run, item, reminder)
                _exclusive_json(directory / "assignment.json", assignment)
                expected_identity = copy.deepcopy(reference)
                for key in ("model", "fingerprint"):
                    if item["response_identity"].get(key) is not None:
                        expected_identity[key] = item["response_identity"][key]
                result = run_live_target_arm(env, actor, context=PublicContext(**item["context"]),
                    first_action=item["a0"], arm=assignment["arm"],
                    reminder=reminder["text"] if assignment["arm"] == "remind" else None,
                    spec=spec, seed=assignment["execution_seed"],
                    remaining_model_calls=item["remaining_model_calls"], remaining_steps=item["remaining_steps"],
                    expected_response_identity=expected_identity,
                    journal=_journal(directory / "events", assignment["record_hash"], "execution"))
        except TargetPendingError as exc:
            pending = {"schema_version": "target-single-pending.v1", "run_hash": run["record_hash"],
                "trace_id": trace_id, "assignment_hash": assignment["record_hash"] if assignment else None,
                "phase": "execution" if assignment else "preparation", "partial_record": exc.partial_record,
                "attempt": exc.attempt, "status": "pending"}
        finally:
            for instance in (env, actor):
                if instance is not None:
                    try:
                        _close(instance)
                    except Exception as exc:
                        cleanup_errors.append(type(exc).__name__)
        if pending is not None:
            _exclusive_json(directory / "pending.json", seal({**pending, "cleanup_errors": cleanup_errors}))
            break
        if result is not None:
            _exclusive_json(directory / "result.json", seal({"schema_version": "target-single-result.v1",
                "assignment_hash": assignment["record_hash"], "record": result, "cleanup_errors": cleanup_errors}))
        _exclusive_json(directory / "closed.json", seal({"schema_version": "target-single-closed.v1",
            "run_hash": run["record_hash"], "trace_id": trace_id, "input_hash": item["record_hash"],
            "assignment_hash": assignment["record_hash"] if assignment else None,
            "status": "blocked" if cleanup_errors else "completed" if result is not None else item["status"],
            "reason": "cleanup_error" if cleanup_errors else item["reason"], "cleanup_errors": cleanup_errors}))
        if (cleanup_errors or (preparation.get("error") or {}).get("scope") == "run_shared"
                or result is not None and (result.get("error") or {}).get("scope") in ("pair_shared", "session_shared")):
            break
        for observed in (preparation.get("response_identity", {}), (result or {}).get("response_identity", {})):
            for key in reference:
                if observed.get(key) is not None:
                    reference[key] = observed[key]
    summary = summarize_target_single(root)
    write_json(root / "summary.json", summary)
    return summary
