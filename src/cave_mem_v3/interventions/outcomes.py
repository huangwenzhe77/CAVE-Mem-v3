"""Pure artifact verification and recomputation of paired labels and costs."""
from __future__ import annotations
from cave_mem_v3.target_data.partitions import target_partition, partition_tasks, validation_binding

from collections import Counter
from pathlib import Path

from cave_mem_v3.agents.runtime import add_usage, empty_costs
from cave_mem_v3.common.contracts import StudySpec, digest, finite, read_json, stage_seed, utility
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.interventions.reminder import render_reminder
from cave_mem_v3.memory_generation.bank import bank_memories


def combine_costs(*records):
    """Unknown additive components remain unknown, even when other arms know them."""
    keys = set().union(*(record.keys() for record in records))
    return {key: None if any(record.get(key) is None for record in records)
            else sum(finite(record[key], key, minimum=0) for record in records) for key in keys}


def _events(directory, plan, arm):
    records, previous = [], None
    for index, path in enumerate(sorted((directory / "events").glob("*.json"))):
        entry = checked(read_json(path), schema="target-arm-event.v1")
        event = entry["event"]
        if (path.name != f"{index:06d}.json" or entry["plan_hash"] != plan["record_hash"]
                or entry["previous_hash"] != previous or event["event_index"] != index or event["arm"] != arm):
            raise ValueError("target event sequence/hash chain mismatch")
        records.append(event)
        previous = entry["record_hash"]
    return records


def _arm(directory, plan, item, arm):
    attempt = checked(read_json(directory / "attempt.json"))
    if attempt != seal({"plan_hash": plan["record_hash"], "arm": arm, "status": "started"}):
        raise ValueError("arm attempt identity mismatch")
    wrapper = checked(read_json(directory / "result.json"), schema="target-arm-result.v1")
    if wrapper["plan_hash"] != plan["record_hash"] or wrapper["arm"] != arm:
        raise ValueError("arm result identity mismatch")
    record = wrapper["record"]
    if record["status"] == "setup_error":
        return {"success": None, "costs": None, "identity": {}, "valid": False, "operational_costs": {}}
    checked(record, schema="target-arm.v1")
    budget = {"remaining_model_calls": item["remaining_model_calls"], "remaining_steps": item["remaining_steps"],
              "continuation_model_calls": item["remaining_model_calls"] - 1, "intervention_reserved_calls": 1}
    if (record["arm"] != arm or record["seed"] != plan["seed"]
            or record["initial_checkpoint_hash"] != item["checkpoint_hash"] or record["a0_hash"] != item["a0_hash"]
            or record["actor_config"] != item["actor_config"] or record["environment_config"] != item["environment_config"]
            or record["budget"] != budget
            or record["reminder_hash"] != (plan["reminder"]["text_hash"] if arm == "remind" else None)):
        raise ValueError("arm frozen identity/budget/reminder mismatch")
    events = _events(directory, plan, arm)
    costs = empty_costs()
    calls, tools, final_outcomes, identities = {}, {}, [], []
    continuation_index, reminder_count, restored = 0, 0, False
    for event in events:
        kind = event["kind"]
        if kind == "restore.completed":
            if event["checkpoint_hash"] != item["checkpoint_hash"] or event["state_fingerprint"] != record["restore_fingerprint"]:
                raise ValueError("restore event differs from frozen checkpoint")
            restored = True
        elif kind == "actor.started":
            call_id = event["call_id"]
            if call_id in calls or not restored:
                raise ValueError("actor invoked twice or before restoration")
            is_reminder = event["stage"] == "target_remind"
            if is_reminder:
                reminder_count += 1
                index, stream = 0, "remind-draft"
                if arm != "remind" or calls or tools or event["reminder_hash"] != plan["reminder"]["text_hash"]:
                    raise ValueError("reminder is not a single pre-action intervention")
            else:
                if event["stage"] != "target_continuation" or event["reminder_hash"] is not None:
                    raise ValueError("ordinary continuation received a reminder")
                index, stream = continuation_index, "paired-continuation"
                continuation_index += 1
            if event["seed"] != stage_seed(plan["seed"], item["task_id"], 0, stream, index):
                raise ValueError("actor seed differs from paired stream")
            calls[call_id] = {"start": event, "end": None}
        elif kind in ("actor.completed", "actor.failed", "actor.pending"):
            call = calls.get(event["call_id"])
            if call is None or call["end"] is not None:
                raise ValueError("actor response missing or duplicate attempt")
            call["end"] = event
            if kind == "actor.completed":
                response = event["response"]
                add_usage(costs, response["usage"])
                identities.append({"model": response["model"], "fingerprint": response.get("fingerprint")})
            else:
                add_usage(costs, event.get("attempt", {}).get("usage", {"model_calls": 1.0}))
        elif kind == "step.started":
            index = event["step_index"]
            if index in tools or index != len(tools) or not restored:
                raise ValueError("tool sequence is not contiguous")
            if event["seed"] != stage_seed(plan["seed"], item["task_id"], 0, "paired-environment", index):
                raise ValueError("tool seed differs from paired stream")
            tools[index] = {"start": event, "end": None}
            costs["tool_calls"] += 1
        elif kind in ("step.completed", "step.failed"):
            step = tools.get(event["step_index"])
            if step is None or step["end"] is not None:
                raise ValueError("tool result missing or duplicate attempt")
            step["end"] = event
            result = event.get("result", {})
            fee = result.get("tool_cost")
            costs["tool_cost"] = None if fee is None or costs["tool_cost"] is None else costs["tool_cost"] + finite(fee, "tool_cost", minimum=0)
            if result.get("terminal"):
                final_outcomes.append(result["success"])
        elif kind == "finalize.completed":
            if not event["result"].get("terminal"):
                raise ValueError("finalizer did not produce a terminal outcome")
            final_outcomes.append(event["result"]["success"])
    if reminder_count > (1 if arm == "remind" else 0) or continuation_index > budget["continuation_model_calls"] or len(tools) > budget["remaining_steps"]:
        raise ValueError("arm exceeded frozen budget")
    responses = [value["end"]["response"] for value in calls.values()
                 if value["end"] is not None and value["end"]["kind"] == "actor.completed"]
    response_index = 0
    for index, step in enumerate(record["steps"]):
        if step["index"] != index or step["action_hash"] != digest(step["action"]):
            raise ValueError("arm step action identity mismatch")
        if index == 0 and arm == "silent":
            if step["action"] != item["a0"] or step["actor_response"] is not None or step["action_origin"] != "cached_a0":
                raise ValueError("Silent must execute cached a0 without resampling")
        else:
            if response_index >= len(responses) or step["actor_response"] != responses[response_index] or step["action"] != responses[response_index]["action"]:
                raise ValueError("executed action differs from recorded actor response")
            response_index += 1
        if index in tools:
            tool = tools[index]
            result = tool["end"].get("result") if tool["end"] else None
            if tool["start"]["action"] != step["action"] or result != step["result"]:
                raise ValueError("arm tool journal differs from result")
        elif step.get("result") is not None or ("validation_error" not in step
                and not any(event["kind"] == "validation.failed" and event["step_index"] == index
                            and event["error_class"] == step.get("validation_system_error") for event in events)):
            raise ValueError("step has no tool attempt, explicit rejection, or validation failure")
    # Durations are observations, not replayable computations. Check both timed
    # components and their sum; never treat elapsed duration as additive dollars.
    costs["wall_seconds"] = sum(finite(record["cost_breakdown"][key]["wall_seconds"], "wall_seconds", minimum=0)
                                for key in ("intervention", "continuation"))
    if any(record["costs"].get(key) != value for key, value in costs.items()):
        raise ValueError("arm aggregate costs differ from raw attempts and timed components")
    components = combine_costs(record["cost_breakdown"]["intervention"], record["cost_breakdown"]["continuation"])
    if any(components.get(key) != value for key, value in costs.items()):
        raise ValueError("arm cost breakdown differs from metered total")
    for key, value in record["operational_costs"].items():
        finite(value, key, minimum=0)
    valid = record["status"] == "completed" and not wrapper["cleanup_errors"]
    if valid:
        if (not restored or not final_outcomes or any(type(y) is not bool for y in final_outcomes)
                or len(set(final_outcomes)) != 1 or record["success"] != final_outcomes[-1]
                or any(call["end"] is None for call in calls.values())
                or any(tool["end"] is None for tool in tools.values())
                or reminder_count != (1 if arm == "remind" else 0)):
            raise ValueError("completed arm lacks independent terminal evidence")
    else:
        if record["status"] == "error" and record["success"] is not None:
            raise ValueError("infrastructure error cannot supply a task failure label")
    expected_utility = utility(record["success"], costs, StudySpec(**plan["study"])) if record["status"] == "completed" else None
    if record["utility"] != expected_utility or record["outcome_valid"] != (record["status"] == "completed") or record["utility_valid"] != (expected_utility is not None):
        raise ValueError("arm cached validity/utility does not match raw evidence")
    return {"success": record["success"] if valid else None, "costs": costs, "valid": valid,
            "identities": identities, "operational_costs": record["operational_costs"]}


def _pair(root, run, item, memories):
    from cave_mem_v3.interventions.collection import pair_plan
    result = {"input_hash": item["record_hash"], "task_id": item["task_id"],
        "status": item["status"], "reason": item["reason"], "outcome_valid": False,
        "utility_valid": False, "silent_success": None, "remind_success": None,
        "D": None, "H": None, "rescue": None}
    if item["status"] != "ready":
        return seal(result)
    directory = root / "pairs" / item["record_hash"]
    if not (directory / "audit.attempt.json").exists():
        result["status"] = "unattempted"
        return seal(result)
    if checked(read_json(directory / "audit.attempt.json")) != seal({"input_hash": item["record_hash"], "status": "started"}):
        raise ValueError("restoration attempt identity mismatch")
    result["status"] = "pending"
    if not (directory / "audit.json").exists():
        return seal(result)
    audit = checked(read_json(directory / "audit.json"), schema="target-restore-audit.v1")
    if audit["input_hash"] != item["record_hash"] or audit["checkpoint_hash"] != item["checkpoint_hash"]:
        raise ValueError("restoration audit identity mismatch")
    closed = checked(read_json(directory / "closed.json")) if (directory / "closed.json").exists() else None
    if closed is not None and closed["input_hash"] != item["record_hash"]:
        raise ValueError("pair close record identity mismatch")
    if audit["status"] != "passed":
        if closed is not None:
            if closed["status"] != "blocked" or closed["audit_hash"] != audit["record_hash"]:
                raise ValueError("blocked audit close record mismatch")
            result.update(status="blocked", reason="restoration_audit_failed")
        return seal(result)
    restores = audit.get("restores", [])
    if (len(restores) != 2 or restores[0] != restores[1]
            or restores[0]["snapshot_hash"] != item["checkpoint_hash"]
            or not restores[0].get("fingerprint")):
        raise ValueError("passed audit lacks two matching actual checkpoint restores")
    probe = audit.get("probe", {})
    if item["environment_config"].get("adapter") == "preparation-fixture.v3":
        if probe != {"kind": "fixture_mutation", "branches_isolated": True}:
            raise ValueError("passed fixture audit lacks mutation isolation evidence")
    elif not all(probe.get("filesystem_probe", {}).get(key) is True for key in (
            "persisted_after_restore", "branches_isolated", "base_branch_isolated")):
        raise ValueError("passed Terminal audit lacks filesystem isolation evidence")
    if not (directory / "plan.json").exists():
        return seal(result)
    plan = checked(read_json(directory / "plan.json"), schema="target-pair-plan.v1")
    expected = pair_plan(run, item, audit, render_reminder(memories[item["memory_id"]], item["a0"]))
    if plan != expected:
        raise ValueError("paired plan differs from frozen input/randomization/reminder")
    result["order"] = plan["order"]
    arms, pending_attempts = {}, {}
    for arm in plan["order"]:
        path = directory / arm
        if arm == plan["order"][1] and (path / "attempt.json").exists() and plan["order"][0] not in arms:
            raise ValueError("second arm started before the first arm had a known result")
        if (path / "result.json").exists():
            arms[arm] = _arm(path, plan, item, arm)
            wrapper = read_json(path / "result.json")
            record = wrapper["record"]
            if record.get("outcome_valid") and record.get("restore_fingerprint") != restores[0]["fingerprint"]:
                raise ValueError("arm restored a fingerprint outside its audited checkpoint")
            if (wrapper["cleanup_errors"] or record["status"] == "setup_error"
                    or (record.get("error") or {}).get("scope") == "pair_shared"):
                other = "remind" if arm == "silent" else "silent"
                if arm == plan["order"][0] and (directory / other / "attempt.json").exists():
                    raise ValueError("second arm started after a shared protocol error")
        elif (path / "pending.json").exists():
            pending = checked(read_json(path / "pending.json"))
            if pending["plan_hash"] != plan["record_hash"] or pending["arm"] != arm:
                raise ValueError("pending request identity mismatch")
            _events(path, plan, arm)
            pending_attempts[arm] = {"costs": pending["partial_record"]["costs"],
                                    "operational_costs": pending["partial_record"]["operational_costs"],
                                    "attempt": pending["attempt"]}
    result["pending_attempts"] = pending_attempts
    result["shared_preparation_costs"] = item["shared_costs"]
    result["arm_costs"] = {name: arm["costs"] for name, arm in arms.items()}
    if closed is None:
        return seal(result)
    if pending_attempts:
        raise ValueError("pending external request cannot have a closed pair")
    if closed.get("plan_hash") != plan["record_hash"] or closed["status"] not in ("blocked", "completed"):
        raise ValueError("pair close record differs from frozen plan")
    result.update(status=closed["status"], reason=None if closed["status"] == "completed" else "shared_protocol_error")
    if closed["status"] == "completed" and set(arms) != {"silent", "remind"}:
        raise ValueError("completed pair is missing an arm")
    identities = [item["response_identity"]]
    for arm in arms.values():
        identities.extend(arm.get("identities", []))
    identity_valid = all(len({identity.get(key) for identity in identities if identity.get(key) is not None}) <= 1
                         for key in ("model", "fingerprint"))
    valid = closed["status"] == "completed" and all(a["valid"] for a in arms.values()) and identity_valid
    result["response_identity_status"] = "consistent_reported" if identity_valid and all(i.get("fingerprint") for i in identities) else "unknown" if identity_valid else "mismatch"
    result["outcome_valid"] = valid
    if not valid and result["status"] == "completed":
        result["reason"] = "arm_error" if identity_valid else "response_identity_mismatch"
    shared = item["shared_costs"]
    result["shared_preparation_costs"] = shared
    result["arm_costs"] = {name: arm["costs"] for name, arm in arms.items()}
    result["operational_costs"] = {"audit_wall_seconds": audit["operational_wall_seconds"],
                                   "arms": {name: arm["operational_costs"] for name, arm in arms.items()}}
    if len(arms) == 2 and all(arm["costs"] is not None for arm in arms.values()):
        actual = combine_costs(shared, arms["silent"]["costs"], arms["remind"]["costs"])
        result["measured_duration_sum_seconds"] = actual.pop("wall_seconds", None)
        result["actual_experiment_additive_costs"] = actual
        result["deployment_costs"] = {name: combine_costs(shared, arm["costs"]) for name, arm in arms.items()}
    if valid:
        silent, remind = arms["silent"]["success"], arms["remind"]["success"]
        spec = StudySpec(**run["study"])
        utilities = {name: utility(arm["success"], result["deployment_costs"][name], spec) for name, arm in arms.items()}
        result.update(silent_success=silent, remind_success=remind, H=int(silent and not remind),
                      rescue=int(not silent and remind), utilities=utilities,
                      utility_valid=all(value is not None for value in utilities.values()))
        if result["utility_valid"]:
            result["D"] = utilities["remind"] - utilities["silent"]
    return seal(result)


def summarize_target_pairs(output_dir):
    """Read-only recomputation: no environment, model, or test outcome is consulted."""
    root = Path(output_dir)
    run = checked(read_json(root / "run.json"), schema="target-pair-run.v1")
    checked(run["prepared_run"], "run_hash", "target-run.v1")
    prepared = run["prepared_run"]
    tasks = partition_tasks(prepared["manifest"], prepared["role"], prepared.get("validation_role"))
    if (len(prepared["task_order"]) != len(tasks)
            or set(prepared["task_order"]) != {task["task_id"] for task in tasks}):
        raise ValueError("paired task schedule differs from frozen partition")
    checked(run["bank"], "bank_hash", "verified-source-bank.v3")
    if (run["source_run_hash"] != run["prepared_run"]["run_hash"] or run["bank_hash"] != run["bank"]["bank_hash"]
            or run["study"] != run["prepared_run"]["study"]):
        raise ValueError("paired run identity mismatch")
    if prepared["role"] == "validation" and any(run.get(k) != v for k, v in validation_binding(prepared).items()):
        raise ValueError("paired validation metadata differs from preparation")
    memories = {m.memory_id: m for m in bank_memories(run["bank"])} if run["bank"]["memories"] else {}
    inputs = run["inputs"]
    if [item["task_id"] for item in inputs] != run["prepared_run"]["task_order"]:
        raise ValueError("paired inventory differs from frozen preparation schedule")
    for item in inputs:
        checked(item, schema="target-pair-input.v1")
        if item["run_hash"] != run["source_run_hash"]:
            raise ValueError("paired input belongs to another preparation run")
    allowed = {item["record_hash"] for item in inputs if item["status"] == "ready"}
    if any(path.name not in allowed for path in (root / "pairs").glob("*")):
        raise ValueError("unregistered pair artifacts")
    pairs = [_pair(root, run, item, memories) for item in inputs]
    counts = Counter(pair["status"] for pair in pairs)
    reasons = Counter(pair["reason"] for pair in pairs if pair["reason"] is not None)
    return seal({"schema_version": "target-pair-summary.v1", "run_hash": run["record_hash"],
        **validation_binding(prepared),
        "planned_rollouts": len(inputs), "planned_pairs": len(allowed),
        "completed_pairs": counts["completed"], "pending_pairs": counts["pending"],
        "blocked_pairs": counts["blocked"], "excluded_rollouts": counts["excluded"],
        "unattempted_pairs": counts["unattempted"], "reasons": dict(reasons),
        "error_pairs": sum(pair["reason"] in ("arm_error", "shared_protocol_error", "response_identity_mismatch") for pair in pairs),
        "valid_outcomes": sum(pair["outcome_valid"] for pair in pairs),
        "valid_utilities": sum(pair["utility_valid"] for pair in pairs), "pairs": pairs,
        "claim_boundary": "Software paired protocol; fixture outcomes are not real benchmark results."})
