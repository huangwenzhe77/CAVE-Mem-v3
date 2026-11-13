"""One historical rollout or one source-replay continuation, without reminders."""
from __future__ import annotations

import copy
import time

from cave_mem_v3.common.contracts import PublicContext, StudySpec, digest, finite, plain, stage_seed, utility


def empty_costs():
    return {"model_calls": 0.0, "physical_attempts": 0.0, "input_tokens": 0.0,
            "output_tokens": 0.0, "cached_input_tokens": 0.0, "api_cost": 0.0,
            "tool_calls": 0.0, "tool_cost": 0.0, "wall_seconds": 0.0}


def add_usage(costs, usage):
    for name in ("model_calls", "physical_attempts", "input_tokens", "output_tokens", "cached_input_tokens", "api_cost"):
        value = usage.get(name)
        costs[name] = None if value is None or costs[name] is None else costs[name] + finite(value, name, minimum=0)


def run_episode(env, actor, *, spec: StudySpec, seed: int, checkpoint=None,
                first_action=None, capture_checkpoints=True) -> dict:
    """Run within both remaining environment steps and the supplied continuation cap.

    first_action executes once without an actor call. Both source replay arms
    therefore pay for the same continuation protocol. Tool/verifier errors retain
    unknown outcomes, whereas malformed actor actions and exhausted task budgets
    are task failures. The caller owns env.close() to preserve audit access.
    """
    started = time.perf_counter()
    costs = empty_costs()
    record = {"schema_version": "source-trajectory.v3", "status": "error",
              "success": None, "utility": None, "seed": seed,
              "spec": plain(spec), "steps": [], "costs": costs, "error": None,
              "response_identity": {"model": None, "fingerprint": None, "fingerprint_status": "unknown"}}
    pending_response = None
    missing_fingerprint = False
    try:
        record.update(actor_config=copy.deepcopy(actor.configuration()),
                      environment_config=copy.deepcopy(env.configuration()),
                      capabilities={"actor": copy.deepcopy(actor.capabilities()),
                                    "environment": copy.deepcopy(env.capabilities())})
        context = env.reset(seed) if checkpoint is None else env.restore(copy.deepcopy(checkpoint))
        if not isinstance(context, PublicContext):
            raise TypeError("environment must expose a PublicContext")
        if checkpoint is not None and digest(env.snapshot()) != digest(checkpoint):
            raise ValueError("restored checkpoint differs from the requested state")
        initial_snapshot = env.snapshot()
        record.update(task_id=context.task_id,
                      initial_checkpoint_hash=digest(initial_snapshot),
                      initial_state_fingerprint=env.state_fingerprint())
        if capture_checkpoints:
            record["initial_checkpoint"] = initial_snapshot
        if not actor.capabilities().get("stateless"):
            raise ValueError("source actor must be stateless")
        terminal, success = env.terminal_status()
        if terminal:
            raise ValueError("cannot start a rollout at a terminal checkpoint")
        calls = 0
        for index in range(min(spec.max_steps, context.remaining_steps)):
            if actor.configuration() != record["actor_config"] or env.configuration() != record["environment_config"]:
                raise ValueError("frozen actor/environment configuration changed")
            step = {"index": index, "context": plain(context), "checkpoint_hash": digest(env.snapshot()),
                    "state_fingerprint": env.state_fingerprint(), "actor_response": None,
                    "action": None, "result": None}
            if capture_checkpoints:
                step["checkpoint"] = copy.deepcopy(env.snapshot())
            record["steps"].append(step)
            if index == 0 and first_action is not None:
                action = copy.deepcopy(first_action)
                step["action_origin"] = "source_replay_override"
            else:
                if calls >= spec.max_model_calls:
                    record["steps"].pop()
                    record.update(status="budget_exhausted", success=False)
                    break
                calls += 1
                pending_response = actor.propose(context, stage="source_continuation" if first_action is not None else "source_collection",
                    seed=stage_seed(seed, context.task_id, 0, "actor", index), reminder=None,
                    call_id=f"{context.task_id}:{seed}:actor:{index}")
                add_usage(costs, pending_response.usage)
                step["actor_response"] = plain(pending_response)
                identity = record["response_identity"]
                response_model, fingerprint = pending_response.model, pending_response.fingerprint
                if not isinstance(response_model, str) or not response_model:
                    identity["fingerprint_status"] = "mismatch"
                    raise ValueError("actor response has no model identity")
                if identity["model"] is not None and identity["model"] != response_model:
                    identity["fingerprint_status"] = "mismatch"
                    raise ValueError("actor response model changed within the episode")
                identity["model"] = response_model
                if fingerprint == "":
                    fingerprint = None
                if fingerprint is not None and not isinstance(fingerprint, str):
                    identity["fingerprint_status"] = "mismatch"
                    raise ValueError("actor response fingerprint must be a string or unknown")
                if fingerprint is not None:
                    if identity["fingerprint"] is not None and identity["fingerprint"] != fingerprint:
                        identity["fingerprint_status"] = "mismatch"
                        raise ValueError("actor response fingerprint changed within the episode")
                    identity["fingerprint"] = fingerprint
                else:
                    missing_fingerprint = True
                identity["fingerprint_status"] = "unknown" if missing_fingerprint else "reported"
                action = copy.deepcopy(pending_response.action)
                step["action_origin"] = "actor"
                pending_response = None
            step["action"] = action
            valid, reason = env.validate_action(action)
            if not valid:
                step["validation_error"] = reason
                record.update(status="invalid_action", success=False)
                break
            costs["tool_calls"] += 1.0
            result = env.step(action, seed=stage_seed(seed, context.task_id, 0, "environment", index))
            step["result"] = plain(result)
            if result.tool_cost is None or costs["tool_cost"] is None:
                costs["tool_cost"] = None
            else:
                costs["tool_cost"] += finite(result.tool_cost, "tool_cost", minimum=0)
            if not result.valid:
                record.update(status="invalid_action", success=False)
                break
            if result.terminal:
                if type(result.success) is not bool:
                    raise ValueError("terminal result lacks boolean success")
                record.update(status="terminal", success=result.success)
                break
            if not isinstance(result.context, PublicContext):
                raise TypeError("step did not return PublicContext")
            context = result.context
        else:
            record.update(status="budget_exhausted", success=False)
        if record["status"] == "budget_exhausted" and callable(getattr(env, "finalize", None)):
            # Benchmarks with an external terminal grader evaluate the files at
            # the exact budget boundary without granting another actor action.
            final = env.finalize()
            if not final.terminal or type(final.success) is not bool:
                raise ValueError("finalization must provide an explicit terminal outcome")
            record.update(status="terminal", success=final.success,
                          termination_reason="budget_exhausted", finalization=plain(final))
        if actor.configuration() != record["actor_config"] or env.configuration() != record["environment_config"]:
            raise ValueError("frozen actor/environment configuration changed")
    except Exception as exc:
        record.update(status="error", success=None, utility=None,
                      error={"type": type(exc).__name__, "message": str(exc)})
        attempt = getattr(exc, "attempt", None)
        if attempt is not None:
            record["failed_attempt"] = copy.deepcopy(attempt)
            add_usage(costs, attempt.get("usage", {}))
            if attempt.get("identity_error"):
                record["response_identity"]["fingerprint_status"] = "mismatch"
        if record["steps"] and record["steps"][-1].get("action") is not None and record["steps"][-1].get("result") is None:
            costs["tool_cost"] = None
    finally:
        costs["wall_seconds"] = time.perf_counter() - started
        if record["status"] != "error" and type(record["success"]) is bool:
            try:
                record["utility"] = utility(record["success"], costs, spec)
            except Exception as exc:
                record.update(status="error", success=None, utility=None,
                              error={"type": type(exc).__name__, "message": str(exc)})
        verifier = getattr(env, "verifier_report", None)
        if verifier is not None:
            record["private_verifier"] = copy.deepcopy(verifier)
        record["trace_hash"] = digest(record)
    return record
