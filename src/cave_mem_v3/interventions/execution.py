"""Execute a single restored target arm with one-shot advisory intervention."""
from __future__ import annotations

import copy
import time
from typing import Callable

from cave_mem_v3.agents.providers import ActorResponse, ProviderError
from cave_mem_v3.agents.runtime import add_usage, empty_costs
from cave_mem_v3.common.contracts import PublicContext, StudySpec, canonical, digest, finite, plain, stage_seed, utility


class TargetPendingError(RuntimeError):
    """An external request may have completed; inspection is required, not retry."""

    def __init__(self, record: dict, attempt: dict):
        super().__init__("target request completion is unknown; do not retry automatically")
        self.partial_record = copy.deepcopy(record)
        self.attempt = copy.deepcopy(attempt)


class _JournalFailure(RuntimeError):
    pass


class _ArmFailure(RuntimeError):
    def __init__(self, code: str, *, scope: str = "arm_local"):
        super().__init__(code)
        self.code, self.scope = code, scope


def run_target_arm(env, actor, *, context: PublicContext, checkpoint: dict,
                   first_action: dict, arm: str, reminder: str | None,
                   spec: StudySpec, seed: int, remaining_model_calls: int,
                   remaining_steps: int, journal: Callable[[dict], None] | None = None,
                   expected_response_identity: dict | None = None) -> dict:
    return _run_target_arm(env, actor, context=context, checkpoint=checkpoint,
        first_action=first_action, arm=arm, reminder=reminder, spec=spec, seed=seed,
        remaining_model_calls=remaining_model_calls, remaining_steps=remaining_steps,
        journal=journal, expected_response_identity=expected_response_identity, live=False)


def run_live_target_arm(env, actor, *, context: PublicContext, first_action: dict,
                        arm: str, reminder: str | None, spec: StudySpec, seed: int,
                        remaining_model_calls: int, remaining_steps: int,
                        journal: Callable[[dict], None] | None = None,
                        expected_response_identity: dict | None = None) -> dict:
    """Continue the already active session, without snapshot, clone, or restore."""
    return _run_target_arm(env, actor, context=context, checkpoint=None,
        first_action=first_action, arm=arm, reminder=reminder, spec=spec, seed=seed,
        remaining_model_calls=remaining_model_calls, remaining_steps=remaining_steps,
        journal=journal, expected_response_identity=expected_response_identity, live=True)


def _run_target_arm(env, actor, *, context, checkpoint, first_action, arm, reminder,
                    spec, seed, remaining_model_calls, remaining_steps, journal,
                    expected_response_identity, live):
    """The caller owns fresh env/actor instances and closes them after this call.

    The cached draft is already charged by preparation. Reserve one remaining
    call for Remind and give both arms R-1 ordinary continuation calls. No arm
    borrows the unused Silent slot. Journal failures and BaseException abort
    immediately; a transport timeout raises TargetPendingError and cannot be
    interpreted as a terminal task failure.
    """
    if arm not in ("silent", "remind"):
        raise ValueError("arm must be silent or remind")
    if not isinstance(context, PublicContext) or (not live and not isinstance(checkpoint, dict)):
        raise TypeError("target execution requires public context and checkpoint")
    if not isinstance(first_action, dict) or not first_action:
        raise ValueError("cached action must be a nonempty mapping")
    for name, value in (("remaining_model_calls", remaining_model_calls), ("remaining_steps", remaining_steps)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if remaining_model_calls > spec.max_model_calls or remaining_steps > min(spec.max_steps, context.remaining_steps):
        raise ValueError("remaining budgets exceed frozen study/environment limits")
    if arm == "silent" and reminder is not None:
        raise ValueError("Silent must not receive a reminder")
    if arm == "remind" and (not isinstance(reminder, str) or not reminder.strip()):
        raise ValueError("Remind needs one nonempty frozen reminder")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    started = time.perf_counter()
    finalization_seconds = 0.0
    finalization_calls = 0
    original_finalize = None
    finalize_instance_value = None
    finalize_had_instance_value = False
    finalize_meter_installed = False
    pending_failure = None
    costs = empty_costs()
    intervention_costs, continuation_costs = empty_costs(), empty_costs()
    record = {
        "schema_version": "target-live-arm.v1" if live else "target-arm.v1", "arm": arm, "seed": seed,
        "status": "error", "success": None, "utility": None,
        "outcome_valid": False, "utility_valid": False,
        "termination_reason": None, "actor_behavior": "not_observed", "error": None,
        "a0_hash": digest(first_action), "initial_checkpoint_hash": None if live else digest(checkpoint),
        "restore_fingerprint": None, "steps": [], "costs": costs,
        "budget": {"remaining_model_calls": remaining_model_calls, "remaining_steps": remaining_steps,
                   "continuation_model_calls": remaining_model_calls - 1,
                   "intervention_reserved_calls": 1},
        "actor_config": copy.deepcopy(actor.configuration()),
        "environment_config": copy.deepcopy(env.configuration()),
        "response_identity": {"model": None, "fingerprint": None, "fingerprint_status": "not_observed"},
        "reminder_hash": digest(reminder) if reminder is not None else None,
    }
    if live:
        record["initial_public_context_hash"] = digest(context)
        record["execution_mode"] = "same_live_session_no_restore.v1"
    identity_reference = copy.deepcopy(expected_response_identity or {})
    event_index = 0

    def emit(kind, **fields):
        nonlocal event_index
        event = {"event_index": event_index, "arm": arm, "kind": kind, **plain(fields)}
        event_index += 1
        if journal is not None:
            try:
                journal(copy.deepcopy(event))
            except Exception as exc:
                raise _JournalFailure("target journal write failed; stop without retry") from exc

    def check_configuration():
        if actor.configuration() != record["actor_config"] or env.configuration() != record["environment_config"]:
            raise _ArmFailure("configuration_drift", scope="pair_shared")

    def check_identity(response):
        model, fingerprint = response.model, response.fingerprint or None
        identity = record["response_identity"]
        if not isinstance(model, str) or not model or fingerprint is not None and not isinstance(fingerprint, str):
            raise _ArmFailure("invalid_response_identity", scope="pair_shared")
        for ref in (identity_reference, identity):
            if ref.get("model") is not None and ref["model"] != model:
                raise _ArmFailure("response_model_drift", scope="pair_shared")
            if ref.get("fingerprint") is not None and fingerprint is not None and ref["fingerprint"] != fingerprint:
                raise _ArmFailure("response_fingerprint_drift", scope="pair_shared")
        previous_unknown = identity["fingerprint_status"] == "unknown"
        identity.update(model=model, fingerprint=fingerprint or identity.get("fingerprint"),
                        fingerprint_status="unknown" if previous_unknown or fingerprint is None else "reported")

    def actor_call(current, *, index, is_reminder=False):
        stage = "target_remind" if is_reminder else "target_continuation"
        stream = "remind-draft" if is_reminder else "live-continuation" if live else "paired-continuation"
        call_seed = stage_seed(seed, context.task_id, 0, stream, index)
        call_id = f"target:{context.task_id}:{seed}:{arm}:{stage}:{index}"
        hint = reminder if is_reminder else None
        usage_target = intervention_costs if is_reminder else continuation_costs
        method = getattr(actor, "propose_target", None)
        if not callable(method):
            if hint is not None:
                raise _ArmFailure("target_reminder_not_supported", scope="pair_shared")
            method = actor.propose
        emit("actor.started", call_id=call_id, stage=stage, seed=call_seed,
             context=plain(current), reminder_hash=digest(hint) if hint else None)
        call_started = time.perf_counter()
        try:
            response = method(current, stage=stage, seed=call_seed, reminder=hint, call_id=call_id)
        except ProviderError as exc:
            add_usage(costs, exc.attempt.get("usage", {}))
            add_usage(usage_target, exc.attempt.get("usage", {}))
            record["failed_attempt"] = copy.deepcopy(exc.attempt)
            if exc.attempt.get("error_class") in ("TimeoutError", "URLError", "ConnectionError", "ConnectionResetError"):
                record["termination_reason"] = "request_completion_unknown"
                emit("actor.pending", call_id=call_id, attempt=exc.attempt)
                raise TargetPendingError(record, exc.attempt) from exc
            emit("actor.failed", call_id=call_id, attempt=exc.attempt)
            if exc.attempt.get("identity_error"):
                raise _ArmFailure("response_identity_drift", scope="pair_shared") from exc
            raise _ArmFailure("provider_failure") from exc
        except (TimeoutError, ConnectionError) as exc:
            attempt = {"call_id": call_id, "error_class": type(exc).__name__, "status": "pending",
                       "usage": {"model_calls": 1.0, "physical_attempts": None}}
            add_usage(costs, attempt["usage"])
            add_usage(usage_target, attempt["usage"])
            record["termination_reason"] = "request_completion_unknown"
            emit("actor.pending", call_id=call_id, attempt=attempt)
            raise TargetPendingError(record, attempt) from exc
        except Exception as exc:
            if isinstance(exc, _JournalFailure):
                raise
            emit("actor.failed", call_id=call_id, error_class=type(exc).__name__)
            if isinstance(exc, _ArmFailure):
                raise
            # The request has been invoked but its usage is unavailable.
            add_usage(costs, {"model_calls": 1.0})
            add_usage(usage_target, {"model_calls": 1.0})
            raise _ArmFailure("actor_failure") from exc
        finally:
            usage_target["wall_seconds"] += time.perf_counter() - call_started
        if not isinstance(response, ActorResponse):
            add_usage(costs, {"model_calls": 1.0})
            add_usage(usage_target, {"model_calls": 1.0})
            emit("actor.failed", call_id=call_id, error_class="InvalidActorResponse")
            raise _ArmFailure("invalid_actor_response")
        add_usage(costs, response.usage)
        add_usage(usage_target, response.usage)
        emit("actor.completed", call_id=call_id, response=plain(response))
        check_identity(response)
        check_configuration()
        return response

    try:
        if not actor.capabilities().get("stateless"):
            raise _ArmFailure("stateful_actor_unsupported", scope="pair_shared")
        capabilities = env.capabilities()
        record["environment_capabilities"] = copy.deepcopy(capabilities)
        if live:
            restored_context = env.observe()
            if canonical(restored_context) != canonical(context):
                raise _ArmFailure("live_context_changed", scope="session_shared")
            emit("session.checked", public_context_hash=digest(context), mode=record["execution_mode"])
        else:
            if not capabilities.get("restore_certified") or not capabilities.get("isolated_sessions"):
                raise _ArmFailure("restoration_uncertified", scope="pair_shared")
            emit("restore.started", checkpoint_hash=record["initial_checkpoint_hash"])
            try:
                restored_context = env.restore(copy.deepcopy(checkpoint))
            except Exception as exc:
                raise _ArmFailure("restoration_failed", scope="pair_shared") from exc
            if digest(env.snapshot()) != digest(checkpoint) or canonical(restored_context) != canonical(context):
                raise _ArmFailure("restoration_mismatch", scope="pair_shared")
            record["restore_fingerprint"] = env.state_fingerprint()
            emit("restore.completed", checkpoint_hash=digest(env.snapshot()),
                 state_fingerprint=record["restore_fingerprint"])
        check_configuration()
        terminal, _ = env.terminal_status()
        if terminal:
            raise _ArmFailure("terminal_checkpoint", scope="pair_shared")
        original_finalize = getattr(env, "finalize", None)
        if callable(original_finalize):
            # Also meter grading initiated inside env.step (finish or the
            # adapter's step limit). The caller provides a fresh environment;
            # the original method is restored even on interruption.
            finalize_had_instance_value = "finalize" in vars(env)
            finalize_instance_value = vars(env).get("finalize")
            def measured_finalize(*args, **kwargs):
                nonlocal finalization_seconds, finalization_calls
                finalization_calls += 1
                finalization_started = time.perf_counter()
                try:
                    return original_finalize(*args, **kwargs)
                finally:
                    finalization_seconds += time.perf_counter() - finalization_started
            env.finalize = measured_finalize
            finalize_meter_installed = True
        current = restored_context
        action = copy.deepcopy(first_action)
        response = None
        if arm == "remind":
            response = actor_call(current, index=0, is_reminder=True)
            action = copy.deepcopy(response.action)
        continuation_calls = 0
        for step_index in range(remaining_steps):
            check_configuration()
            if step_index:
                if continuation_calls >= remaining_model_calls - 1:
                    record["termination_reason"] = "budget_exhausted"
                    break
                response = actor_call(current, index=continuation_calls)
                continuation_calls += 1
                action = copy.deepcopy(response.action)
            step = {"index": step_index, "context": plain(current), "action": copy.deepcopy(action),
                    "action_hash": digest(action), "action_origin": "cached_a0" if arm == "silent" and step_index == 0 else "actor",
                    "actor_response": plain(response) if response is not None else None,
                    "result": None}
            record["steps"].append(step)
            if response is not None and response.parse_error is not None:
                record.update(actor_behavior="malformed_action", termination_reason="invalid_actor_output")
                step["validation_error"] = "malformed_actor_action"
                emit("action.rejected", step_index=step_index, reason="malformed_actor_action")
                break
            try:
                valid, reason = env.validate_action(action)
            except Exception as exc:
                step["validation_system_error"] = type(exc).__name__
                emit("validation.failed", step_index=step_index, error_class=type(exc).__name__)
                raise _ArmFailure("validation_system_failure") from exc
            if not valid:
                record.update(actor_behavior="invalid_action", termination_reason="invalid_actor_output")
                step["validation_error"] = reason
                emit("action.rejected", step_index=step_index, reason=reason)
                break
            record["actor_behavior"] = "valid"
            tool_seed = stage_seed(seed, context.task_id, 0, "live-environment" if live else "paired-environment", step_index)
            emit("step.started", step_index=step_index, action=action, seed=tool_seed)
            costs["tool_calls"] += 1.0
            continuation_costs["tool_calls"] += 1.0
            step_started = time.perf_counter()
            finalization_before_step = finalization_seconds
            try:
                result = env.step(copy.deepcopy(action), seed=tool_seed)
            except Exception as exc:
                costs["tool_cost"] = continuation_costs["tool_cost"] = None
                emit("step.failed", step_index=step_index, error_class=type(exc).__name__)
                raise _ArmFailure("tool_backend_failure") from exc
            finally:
                continuation_costs["wall_seconds"] += max(0.0,
                    time.perf_counter() - step_started - (finalization_seconds - finalization_before_step))
            step["result"] = plain(result)
            if result.tool_cost is None:
                costs["tool_cost"] = continuation_costs["tool_cost"] = None
            elif costs["tool_cost"] is not None:
                value = finite(result.tool_cost, "tool_cost", minimum=0)
                costs["tool_cost"] += value
                continuation_costs["tool_cost"] += value
            emit("step.completed", step_index=step_index, result=plain(result))
            check_configuration()
            if not isinstance(result.context, PublicContext) or result.context.task_id != context.task_id:
                raise _ArmFailure("invalid_environment_observation")
            if not result.valid:
                record.update(actor_behavior="invalid_action", termination_reason="invalid_actor_output")
                break
            if result.terminal:
                record["termination_reason"] = "task_terminal"
                break
            current = result.context
        else:
            record["termination_reason"] = "budget_exhausted"

        terminal, success = env.terminal_status()
        if not terminal:
            finalize = getattr(env, "finalize", None)
            if not callable(finalize):
                raise _ArmFailure("finalization_unsupported")
            emit("finalize.started", reason=record["termination_reason"])
            try:
                final = finalize()
            except Exception as exc:
                emit("finalize.failed", error_class=type(exc).__name__)
                raise _ArmFailure("verifier_failure") from exc
            if not final.terminal or type(final.success) is not bool:
                raise _ArmFailure("verifier_missing_outcome")
            terminal, success = final.terminal, final.success
            emit("finalize.completed", result=plain(final), private_verifier=getattr(env, "verifier_report", None))
        if not terminal or type(success) is not bool:
            raise _ArmFailure("terminal_outcome_missing")
        check_configuration()
        record.update(status="completed", success=success, outcome_valid=True)
        verifier = getattr(env, "verifier_report", None)
        if verifier is not None:
            record["private_verifier"] = copy.deepcopy(verifier)
        record["outcome_source"] = "private_verifier" if verifier is not None else "environment_terminal"
    except TargetPendingError as exc:
        pending_failure = exc
        raise
    except _JournalFailure:
        raise
    except Exception as exc:
        code = exc.code if isinstance(exc, _ArmFailure) else "target_execution_failure"
        scope = exc.scope if isinstance(exc, _ArmFailure) else "arm_local"
        record.update(status="error", success=None, utility=None, outcome_valid=False, utility_valid=False,
                      termination_reason=code, error={"code": code, "scope": scope, "type": type(exc).__name__})
        emit("arm.failed", code=code, scope=scope, error_class=type(exc).__name__)
    finally:
        if finalize_meter_installed:
            if finalize_had_instance_value:
                env.finalize = finalize_instance_value
            else:
                del env.finalize
        costs["wall_seconds"] = intervention_costs["wall_seconds"] + continuation_costs["wall_seconds"]
        record["cost_breakdown"] = {"intervention": intervention_costs, "continuation": continuation_costs}
        record["operational_costs"] = {
            "wall_seconds": max(0.0, time.perf_counter() - started - costs["wall_seconds"]),
            "finalization_wall_seconds": finalization_seconds,
            "finalization_calls": finalization_calls,
        }
        if pending_failure is not None:
            # Include time and costs settled by both finally blocks without
            # converting this incomplete attempt into a terminal arm result.
            pending_failure.partial_record = copy.deepcopy(record)
    if record["outcome_valid"]:
        record["utility"] = utility(record["success"], costs, spec)
        record["utility_valid"] = record["utility"] is not None
    record["record_hash"] = digest(record)
    return record
