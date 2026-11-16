"""Read-only validation of observed single-arm outcomes; no counterfactual labels."""
from __future__ import annotations
from cave_mem_v3.target_data.partitions import target_partition, partition_tasks, validation_binding

from collections import Counter
import math
import random
from pathlib import Path

from cave_mem_v3.agents.runtime import add_usage, empty_costs
from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import StudySpec, digest, finite, plain, read_json, stage_seed, utility
from cave_mem_v3.interventions.inputs import _validate_source_partition, checked, seal
from cave_mem_v3.interventions.outcomes import combine_costs
from cave_mem_v3.interventions.reminder import render_reminder
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.predicates import evaluate_conditions
from cave_mem_v3.memory_generation.replay import require_current_evidence
from cave_mem_v3.target_data.pre_treatment_state import _context, _exposure, _public_action, validate_exposure_rules
from cave_mem_v3.target_data.retrieval import _query_text
from cave_mem_v3.target_data.sampling import OnlineCheckpointSampler, OrdinaryActorSampler, RiskHazard


def _cost_record(value):
    if not isinstance(value, dict) or set(value) != set(empty_costs()):
        raise ValueError("single-arm cost schema mismatch")
    for key, amount in value.items():
        if amount is not None:
            finite(amount, key, minimum=0)
    return value


def _events(directory, owner_hash, phase):
    previous, result = None, []
    for index, path in enumerate(sorted(directory.glob("*.json"))):
        entry = checked(read_json(path), schema="target-single-event.v1")
        if (path.name != f"{index:06d}.json" or entry.get("owner_hash") != owner_hash
                or entry.get("phase") != phase or entry.get("previous_hash") != previous
                or entry.get("event", {}).get("event_index") != index):
            raise ValueError("single-session journal ownership or hash chain mismatch")
        result.append(entry["event"])
        previous = entry["record_hash"]
    return result


def _prepared_state(run, preparation, memories):
    step = preparation["steps"][-1]
    state = checked(step["pre_treatment_state"], schema="pre-treatment-state-record.v1")
    context, action = _context(step["context"]), _public_action(step["action"])
    policy_context = plain(context)
    policy_context.pop("task_id")
    x = state["x"]
    remaining = {"model_calls": run["study"]["max_model_calls"] - len(preparation["steps"]),
                 "environment_steps": min(run["study"]["max_steps"] - step["index"], context.remaining_steps)}
    if (state.get("bank_hash") != run["bank_hash"] or state.get("task_id") != preparation["task_id"]
            or state.get("context_hash") != digest(plain(context)) or state.get("a0_hash") != digest(action)
            or state.get("x_hash") != digest(x) or state.get("randomization_status") != "not_assigned"
            or state.get("outcome_access") != "none" or x.get("schema_version") != "pre-treatment-state.v1"
            or x.get("a0") != action or x.get("context") != policy_context
            or x.get("b") != {"status": "known", "remaining": remaining}):
        raise ValueError("single-session frozen state/action/context/budget mismatch")
    feature = x.get("feature_map", {})
    vector = feature.get("text_embedding")
    if (feature.get("status") != "frozen_text_encoder_plus_structured.v1"
            or feature.get("structured_schema_version") != "pre-treatment-structured.v1"
            or feature.get("encoder_identity") != run["encoder_identity"]
            or feature.get("text_query_hash") != digest(_query_text(context, action))
            or not isinstance(vector, list) or not vector
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector)
            or sum(value * value for value in vector) <= 0):
        raise ValueError("single-session feature map identity is invalid")
    memory_id = state.get("memory_id")
    if memory_id is None:
        if (memories or state.get("status") != "prepared_no_memory" or x.get("m") is not None
                or x.get("v") != "?" or x.get("rho") != "?" or x.get("retrieval", {}).get("status") != "no_match"):
            raise ValueError("invalid no-memory single-session state")
    else:
        memory = memories.get(memory_id)
        if memory is None or state.get("status") != "prepared" or x.get("v") != 1:
            raise ValueError("single-session state memory is absent from the source bank")
        expected_memory = {name: plain(getattr(memory, name)) for name in (
            "mechanism", "problem_action", "correction", "positive_conditions", "exclusion_conditions")}
        evidence = {name: value for name, value in memory.quality.items() if name in (
            "v", "reason", "scope", "n_replay_pairs", "mean_difference", "acceptance_epsilon", "decision_rule", "claim_boundary")}
        if (x.get("m") != expected_memory or x.get("E") != evidence
                or x.get("rho") != evaluate_conditions(memory.positive_conditions, memory.exclusion_conditions, context)
                or x.get("retrieval", {}).get("status") != "retrieved"):
            raise ValueError("single-session state memory/evidence differs from verified source")
    if x.get("e") != _exposure(memory_id=memory_id, context=context, action=action, rules=run["exposure_rules"]):
        raise ValueError("single-session action exposure differs from its frozen rubric")


def _preparation(events, record, run, task, trace_id, seed, memories, *, partial=False):
    if partial:
        if record.get("schema_version") != "target-single-preparation.v1":
            raise ValueError("invalid pending single-session preparation schema")
    else:
        checked(record, schema="target-single-preparation.v1")
    if (record["run_hash"] != run["record_hash"] or record["task_id"] != task["task_id"]
            or record["trace_id"] != trace_id or record["seed"] != seed
            or record["status"] not in ("prepared", "no_selection", "preparation_error", "error")):
        raise ValueError("single-session preparation identity/status mismatch")
    no_memory = run.get("bundle", {}).get("policy_config", {}).get("kind") == "no_memory"
    sampler_type = OrdinaryActorSampler if no_memory else OnlineCheckpointSampler
    sampler = sampler_type(role=run["role"], task_id=task["task_id"], trace_id=trace_id,
        seed=seed, hazard=RiskHazard(**run["risk_hazard"]),
        final_authorization=run.get("final_authorization"))
    costs, prior, calls, tools, seen_steps = empty_costs(), [], {}, {}, []
    current_context = None
    identities, terminal_outcomes = [], []
    finalizing = False
    known = {"reset.started", "reset.completed", "selection.considered", "actor.started", "actor.completed",
             "actor.failed", "actor.pending", "step.started", "step.completed", "step.failed",
             "action.rejected", "preparation.failed", "finalize.started", "finalize.completed"}
    for event in events:
        kind = event.get("kind")
        if kind not in known:
            raise ValueError("unknown single-session preparation event")
        if kind == "reset.started":
            if event["event_index"] != 0 or event["seed"] != seed:
                raise ValueError("preparation reset seed/order mismatch")
        elif kind == "reset.completed":
            if event["event_index"] != 1 or event["context"].get("task_id") != task["task_id"]:
                raise ValueError("preparation reset context/order mismatch")
            current_context = event["context"]
        elif kind == "selection.considered":
            index = len(seen_steps)
            if event["index"] != index or event["context"] != current_context or sampler.selected:
                raise ValueError("single-session sampling was reordered or continued after selection")
            if index and (seen_steps[-1].get("result") is None or seen_steps[-1]["result"].get("valid") is not True
                          or seen_steps[-1]["result"].get("terminal") is not False):
                raise ValueError("sampling continued without a completed valid nonterminal prefix")
            if index >= len(record["steps"]):
                raise ValueError("single-session preparation omits a sampled step")
            step = record["steps"][index]
            point = {"schema_version": "live-public-point.v1", "context_hash": digest(current_context), "index": index}
            decision = sampler.consider(current_context, point, index, prior, checkpoint_hash=digest(point))
            if (event["point"] != point or event["decision"] != decision or step.get("index") != index
                    or step.get("context") != current_context or step.get("point") != point or step.get("selection") != decision):
                raise ValueError("single-session selected point differs from its frozen public prefix")
            seen_steps.append(step)
        elif kind == "actor.started":
            index = len(seen_steps) - 1
            expected_id = f"single:{trace_id}:prefix:{index}"
            if (index < 0 or event["call_id"] != expected_id or expected_id in calls
                    or event["stage"] != "target_single_prefix" or event["context"] != current_context
                    or event.get("reminder") is not None or event.get("reminder_hash") is not None
                    or event["seed"] != stage_seed(seed, task["task_id"], 0, "single-prefix", index)):
                raise ValueError("preparation actor request identity/order mismatch")
            calls[expected_id] = {"index": index, "end": None}
        elif kind in ("actor.completed", "actor.failed", "actor.pending"):
            call = calls.get(event.get("call_id"))
            if call is None or call["end"] is not None:
                raise ValueError("preparation actor result lacks a unique started request")
            call["end"] = event
            if kind == "actor.completed":
                response = event["response"]
                step = seen_steps[call["index"]]
                if step["actor_response"] != response or step["action"] != response["action"]:
                    raise ValueError("prepared cached action differs from original actor response")
                usage = response["usage"]
                identities.append({"model": response["model"], "fingerprint": response.get("fingerprint") or None})
            else:
                usage = event.get("attempt", {}).get("usage", {"model_calls": 1.0})
            add_usage(costs, usage)
        elif kind == "step.started":
            index = event["index"]
            if (index != len(tools) or index >= len(seen_steps) or seen_steps[index]["selection"].get("selected")
                    or seen_steps[index]["action"] != event["action"]
                    or seen_steps[index].get("actor_response") is None
                    or seen_steps[index]["actor_response"].get("parse_error") is not None
                    or any(value["end"] is None for value in calls.values())
                    or event["seed"] != stage_seed(seed, task["task_id"], 0, "single-prefix-environment", index)):
                raise ValueError("selected draft was executed before assignment or prefix tool identity differs")
            tools[index] = {"start": event, "end": None}
            costs["tool_calls"] += 1
        elif kind in ("step.completed", "step.failed"):
            index = event["index"]
            tool = tools.get(index)
            if tool is None or tool["end"] is not None:
                raise ValueError("preparation tool result lacks a unique started request")
            tool["end"] = event
            result = event.get("result")
            if seen_steps[index]["result"] != result:
                raise ValueError("preparation tool journal differs from saved steps")
            fee = result.get("tool_cost") if result is not None else None
            costs["tool_cost"] = None if fee is None or costs["tool_cost"] is None else costs["tool_cost"] + finite(fee, "tool_cost", minimum=0)
            if result is not None:
                prior.append({"action": seen_steps[index]["action"], "result": result, "valid": result["valid"]})
                current_context = result["context"]
                if result.get("terminal") is True:
                    terminal_outcomes.append(result.get("success"))
        elif kind == "finalize.started":
            if record["status"] == "prepared" or finalizing:
                raise ValueError("selected input was graded before assignment or prefix was graded twice")
            finalizing = True
        elif kind == "finalize.completed":
            if not finalizing or event["result"].get("terminal") is not True:
                raise ValueError("unselected prefix lacks started terminal grading")
            terminal_outcomes.append(event["result"].get("success"))
    if seen_steps != record["steps"] or sampler.finalize() != record["selection"]:
        raise ValueError("preparation sampling record differs from its original events")
    costs["wall_seconds"] = finite(record["costs"]["wall_seconds"], "wall_seconds", minimum=0)
    if _cost_record(record["costs"]) != costs:
        raise ValueError("single-session preparation costs differ from original attempts")
    for name, value in record["operational_costs"].items():
        finite(value, name, minimum=0)
    if record["status"] == "prepared":
        if (not seen_steps or not sampler.selected or seen_steps[-1].get("result") is not None
                or "validation_error" in seen_steps[-1] or seen_steps[-1]["actor_response"].get("parse_error") is not None
                or any(value["end"] is None for value in calls.values())
                or any(value["end"] is None for value in tools.values())
                or costs["model_calls"] != len(seen_steps) or costs["tool_calls"] != len(seen_steps) - 1):
            raise ValueError("prepared single-session input has incomplete/executed selected action")
        _prepared_state(run, record, memories)
    elif record["status"] == "no_selection" and sampler.selected:
        raise ValueError("selected single-session point cannot be excluded as no_selection")
    if record["status"] in ("no_selection", "preparation_error"):
        outcome = record.get("unselected_terminal_outcome")
        if outcome is not None and (type(outcome) is not bool or not terminal_outcomes
                or any(value is not outcome for value in terminal_outcomes)):
            raise ValueError("unselected prefix outcome has no matching observed terminal event")
        if record.get("outcome_status") != ("observed" if type(outcome) is bool else "unknown"):
            raise ValueError("unselected prefix outcome validity mismatch")
    if record["status"] != "error":
        expected_identity = {"model": identities[0]["model"] if identities else None,
            "fingerprint": next((value["fingerprint"] for value in identities if value["fingerprint"]), None),
            "fingerprint_status": "not_observed" if not identities else
                "reported" if all(value["fingerprint"] for value in identities) else "unknown"}
        if (any(len({value[key] for value in identities if value[key] is not None}) > 1 for key in ("model", "fingerprint"))
                or record["response_identity"] != expected_identity):
            raise ValueError("single-session preparation actor identity drift")
    return costs


def _live_arm(events, record, item, assignment, spec, *, partial=False, policy_execution=False):
    """Recompute one observed continuation from its original execution events."""
    schema = "target-policy-live-arm.v1" if policy_execution else "target-live-arm.v1"
    if partial:
        if record.get("schema_version") != schema:
            raise ValueError("unsupported pending live-arm schema")
    else:
        checked(record, schema=schema)
    arm, seed = assignment["arm"], assignment["execution_seed"]
    if policy_execution and (item["remaining_model_calls"] < 0 or item["remaining_steps"] < 1
            or arm == "remind" and item["remaining_model_calls"] < 1):
        raise ValueError("policy continuation violates frozen budget")
    reminder_hash = assignment["reminder"]["text_hash"] if arm == "remind" else None
    budget = {"remaining_model_calls": item["remaining_model_calls"],
              "remaining_steps": item["remaining_steps"],
              "continuation_model_calls": max(0, item["remaining_model_calls"] - 1) if policy_execution else item["remaining_model_calls"] - 1,
              "intervention_reserved_calls": min(1, item["remaining_model_calls"]) if policy_execution else 1}
    if (record["arm"] != arm or record["seed"] != seed or record["a0_hash"] != item["a0_hash"]
            or record.get("initial_checkpoint_hash") is not None or record.get("restore_fingerprint") is not None
            or record.get("execution_mode") != "same_live_session_no_restore.v1"
            or record.get("initial_public_context_hash") != digest(item["context"])
            or record["actor_config"] != item["actor_config"]
            or record["environment_config"] != item["environment_config"]
            or record["budget"] != budget or record["reminder_hash"] != reminder_hash):
        raise ValueError("observed arm differs from frozen live input, assignment, or budget")
    costs = empty_costs()
    components = {"intervention": empty_costs(), "continuation": empty_costs()}
    calls, tools, identities, final_outcomes = {}, {}, [], []
    continuation_count = reminder_count = 0
    session_ready = False
    finalizing = False
    current_context = item["context"]
    stopped = False
    known_kinds = {"session.checked", "actor.started", "actor.completed", "actor.failed", "actor.pending",
                   "step.started", "step.completed", "step.failed", "action.rejected", "validation.failed",
                   "finalize.started", "finalize.completed", "finalize.failed", "arm.failed"}
    for index, event in enumerate(events):
        kind = event.get("kind")
        if kind not in known_kinds or event.get("event_index") != index or event.get("arm") != arm:
            raise ValueError("unknown, reordered, or foreign single-arm execution event")
        if stopped:
            raise ValueError("single-arm events continued after a final error or pending request")
        if kind == "session.checked":
            if index != 0 or session_ready or event.get("public_context_hash") != digest(item["context"]):
                raise ValueError("single-arm session context differs from frozen public input")
            if event.get("mode") != "same_live_session_no_restore.v1":
                raise ValueError("single-arm execution may not restore a checkpoint")
            session_ready = True
        elif kind == "actor.started":
            call_id = event["call_id"]
            if (not session_ready or call_id in calls or finalizing or final_outcomes
                    or event.get("context") != current_context
                    or any(value["end"] is None or value["end"]["kind"] != "actor.completed" for value in calls.values())
                    or any(value["end"] is None or value["end"]["kind"] != "step.completed" for value in tools.values())):
                raise ValueError("actor started outside the single live execution sequence")
            is_reminder = event.get("stage") == "target_remind"
            if is_reminder:
                reminder_count += 1
                stream, call_index = "remind-draft", 0
                if arm != "remind" or calls or tools or event.get("reminder_hash") != reminder_hash:
                    raise ValueError("Remind must be the sole intervention before the first action")
            else:
                if (event.get("stage") != "target_continuation" or event.get("reminder_hash") is not None
                        or len(tools) != continuation_count + 1
                        or list(tools.values())[-1]["end"]["result"].get("valid") is not True):
                    raise ValueError("ordinary continuation precedes the first action or receives a reminder")
                stream, call_index = "live-continuation", continuation_count
                continuation_count += 1
            if event["seed"] != stage_seed(seed, item["task_id"], 0, stream, call_index):
                raise ValueError("single-arm actor seed differs from frozen assignment")
            expected_id = f"target:{item['task_id']}:{seed}:{arm}:{event['stage']}:{call_index}"
            if call_id != expected_id:
                raise ValueError("single-arm actor request identity mismatch")
            calls[call_id] = {"start": event, "end": None, "component": "intervention" if is_reminder else "continuation"}
        elif kind in ("actor.completed", "actor.failed", "actor.pending"):
            call = calls.get(event.get("call_id"))
            if call is None or call["end"] is not None:
                raise ValueError("actor completion has no unique prior start")
            call["end"] = event
            if kind == "actor.completed":
                response = event["response"]
                usage = response["usage"]
                identities.append({"model": response["model"], "fingerprint": response.get("fingerprint") or None})
            else:
                usage = event.get("attempt", {}).get("usage", {"model_calls": 1.0})
                if kind == "actor.pending":
                    stopped = True
            add_usage(costs, usage)
            add_usage(components[call["component"]], usage)
        elif kind == "step.started":
            step_index = event["step_index"]
            if (not session_ready or type(step_index) is not int or step_index != len(tools)
                    or finalizing or final_outcomes or event["seed"] != stage_seed(seed, item["task_id"], 0, "live-environment", step_index)):
                raise ValueError("single-arm tool sequence or seed differs from the frozen execution")
            if any(value["end"] is None for value in calls.values()) or any(value["end"] is None for value in tools.values()):
                raise ValueError("tool started while an earlier request is unresolved")
            if step_index == 0 and arm == "silent":
                if calls or event["action"] != item["a0"]:
                    raise ValueError("Silent resampled before its cached first action")
            else:
                expected_calls = step_index + (1 if arm == "remind" else 0)
                if len(calls) != expected_calls or not calls:
                    raise ValueError("single-arm action has no unique preceding actor request")
                response_event = list(calls.values())[-1]["end"]
                if (response_event["kind"] != "actor.completed" or response_event["response"].get("parse_error") is not None
                        or response_event["response"]["action"] != event["action"]):
                    raise ValueError("single-arm tool ran before a valid matching actor response")
            tools[step_index] = {"start": event, "end": None}
            costs["tool_calls"] += 1.0
            components["continuation"]["tool_calls"] += 1.0
        elif kind in ("step.completed", "step.failed"):
            step = tools.get(event.get("step_index"))
            if step is None or step["end"] is not None:
                raise ValueError("tool result has no unique prior start")
            step["end"] = event
            result = event.get("result", {})
            amount = result.get("tool_cost")
            for target in (costs, components["continuation"]):
                target["tool_cost"] = (None if amount is None or target["tool_cost"] is None
                                       else target["tool_cost"] + finite(amount, "tool_cost", minimum=0))
            if result.get("terminal") is True:
                final_outcomes.append(result.get("success"))
            elif result:
                current_context = result["context"]
        elif kind == "finalize.started":
            if not session_ready or finalizing:
                raise ValueError("invalid single-arm finalization order")
            finalizing = True
        elif kind == "finalize.completed":
            if not finalizing or event["result"].get("terminal") is not True:
                raise ValueError("finalization result lacks a started terminal evaluation")
            final_outcomes.append(event["result"].get("success"))
        elif kind == "arm.failed":
            stopped = True
    if (reminder_count > (1 if arm == "remind" else 0)
            or continuation_count > budget["continuation_model_calls"] or len(tools) > budget["remaining_steps"]):
        raise ValueError("observed single arm exceeds its frozen budget")
    responses = [value["end"]["response"] for value in calls.values()
                 if value["end"] is not None and value["end"]["kind"] == "actor.completed"]
    response_index = 0
    previous_context = item["context"]
    for index, step in enumerate(record["steps"]):
        if (step.get("index") != index or step.get("action_hash") != digest(step.get("action"))
                or step.get("context") != previous_context):
            raise ValueError("single-arm action or public context prefix is inconsistent")
        if index == 0 and arm == "silent":
            if step["action"] != item["a0"] or step["actor_response"] is not None or step["action_origin"] != "cached_a0":
                raise ValueError("Silent must execute the cached a0 without a new draft")
        else:
            if (response_index >= len(responses) or step["actor_response"] != responses[response_index]
                    or step["action"] != responses[response_index]["action"]):
                raise ValueError("single-arm action differs from its observed actor response")
            response_index += 1
        if index in tools:
            entry = tools[index]
            result = entry["end"].get("result") if entry["end"] else None
            if entry["start"]["action"] != step["action"] or result != step.get("result"):
                raise ValueError("single-arm tool journal differs from the saved steps")
            if isinstance(result, dict):
                previous_context = result["context"]
        elif step.get("result") is not None or not any(
            event.get("step_index") == index and (
                event["kind"] == "action.rejected" and event.get("reason") == step.get("validation_error")
                or event["kind"] == "validation.failed" and event.get("error_class") == step.get("validation_system_error"))
            for event in events):
            raise ValueError("unexecuted single-arm action lacks an observed rejection or failure")
    if len(record["steps"]) < len(tools):
        raise ValueError("single-arm result omits executed tools")
    for name in ("intervention", "continuation"):
        observed = _cost_record(record["cost_breakdown"][name])
        components[name]["wall_seconds"] = finite(observed["wall_seconds"], "wall_seconds", minimum=0)
        if observed != components[name]:
            raise ValueError("single-arm cost component differs from recorded usage")
    costs["wall_seconds"] = sum(value["wall_seconds"] for value in components.values())
    if _cost_record(record["costs"]) != costs:
        raise ValueError("single-arm aggregate costs differ from raw events")
    for name, value in record["operational_costs"].items():
        finite(value, name, minimum=0)
    completed = record["status"] == "completed" and not partial
    if completed:
        if (not session_ready or not final_outcomes or any(type(value) is not bool for value in final_outcomes)
                or len(set(final_outcomes)) != 1 or record["success"] != final_outcomes[-1]
                or reminder_count != (1 if arm == "remind" else 0)
                or any(value["end"] is None for value in calls.values())
                or any(value["end"] is None for value in tools.values())):
            raise ValueError("completed single arm lacks independently recorded terminal evidence")
    elif record["success"] is not None:
        raise ValueError("pending or erroneous single arm cannot have a task outcome")
    value = utility(record["success"], costs, spec) if completed else None
    if record.get("utility") != value or record.get("outcome_valid") != completed or record.get("utility_valid") != (value is not None):
        raise ValueError("single-arm cached utility or validity differs from raw observations")
    identities = [item["response_identity"], *identities]
    identity_valid = all(len({identity.get(key) for identity in identities if identity.get(key) is not None}) <= 1
                         for key in ("model", "fingerprint"))
    if completed and not identity_valid:
        raise ValueError("completed single arm contains actor identity drift")
    return {"outcome_valid": completed, "success": record["success"] if completed else None,
            "costs": costs, "operational_costs": record["operational_costs"],
            "response_identity_status": "mismatch" if not identity_valid else
                "consistent_reported" if all(value.get("fingerprint") for value in identities) else "unknown"}


def _session(directory, run, task, trace_id, seed, memories):
    from cave_mem_v3.interventions.single import _input, single_assignment
    row = {"schema_version": "target-single-observation.v1", "task_id": task["task_id"],
           "trace_id": trace_id, "status": "unattempted", "reason": None,
           "arm": None, "z": None, "propensity": None, "b": None, "inclusion_probability": None,
           "outcome_valid": False, "observed_success": None, "utility_valid": False, "utility": None,
           "shared_preparation_costs": None, "arm_costs": None, "observed_costs": None,
           "group_id": task.get("group_id"), "groups": task.get("groups", []), "template_id": task.get("template_id")}
    if not directory.exists():
        return seal(row)
    allowed = {"attempt.json", "preparation.json", "input.json", "assignment.json", "result.json",
                "closed.json", "pending.json", "preparation_events", "events"}
    if run.get("schema_version") == "target-auto-run.v1":
        allowed.add("route.json")
    if any(path.name not in allowed for path in directory.iterdir()):
        raise ValueError("unregistered single-session artifacts or extra treatment arm")
    attempt = checked(read_json(directory / "attempt.json"), schema="target-single-attempt.v1")
    expected = seal({"schema_version": "target-single-attempt.v1", "run_hash": run["record_hash"],
                     "trace_id": trace_id, "task_id": task["task_id"], "seed": seed, "status": "started"})
    if attempt != expected:
        raise ValueError("single-session attempt identity differs from fixed schedule")
    row.update(status="pending", reason="live_session_incomplete")
    prefix_events = _events(directory / "preparation_events", run["record_hash"], "preparation")
    pending = checked(read_json(directory / "pending.json"), schema="target-single-pending.v1") if (directory / "pending.json").exists() else None
    if pending is not None and (pending["run_hash"] != run["record_hash"] or pending["trace_id"] != trace_id
            or pending["status"] != "pending" or (directory / "closed.json").exists() or (directory / "result.json").exists()):
        raise ValueError("pending single-session record conflicts with a completed result")
    if not (directory / "preparation.json").exists():
        if any((directory / name).exists() for name in ("input.json", "assignment.json", "result.json", "closed.json", "events")):
            raise ValueError("single-session execution has no frozen preparation")
        if pending is not None:
            if pending["phase"] != "preparation" or pending["assignment_hash"] is not None:
                raise ValueError("pending preparation has an assignment or wrong phase")
            row["shared_preparation_costs"] = _preparation(prefix_events, pending["partial_record"], run, task,
                trace_id, seed, memories, partial=True)
            row["observed_costs"] = row["shared_preparation_costs"]
            row["pending_attempt"] = pending["attempt"]
        return seal(row)
    preparation = read_json(directory / "preparation.json")
    shared = _preparation(prefix_events, preparation, run, task, trace_id, seed, memories)
    row["shared_preparation_costs"] = row["observed_costs"] = shared
    row["inclusion_probability"] = preparation["selection"].get("inclusion_probability")
    row["operational_costs"] = {"preparation": preparation["operational_costs"]}
    if not (directory / "input.json").exists():
        if any((directory / name).exists() for name in ("assignment.json", "result.json", "closed.json")):
            raise ValueError("single-session execution has no frozen input")
        return seal(row)
    item = checked(read_json(directory / "input.json"), schema="target-single-input.v1")
    if item != _input(run, task, preparation):
        raise ValueError("single-session input differs from original pre-assignment preparation")
    assignment = None
    if (directory / "assignment.json").exists():
        if item["status"] != "ready":
            raise ValueError("ineligible single-session input was randomized")
        assignment = checked(read_json(directory / "assignment.json"), schema="target-single-assignment.v1")
        memory = memories[item["pre_treatment_state"]["memory_id"]]
        expected = single_assignment(run, item, render_reminder(memory, item["a0"]))
        if assignment != expected:
            raise ValueError("single-arm assignment, propensity, reminder, or frozen input was changed")
        row.update(arm=assignment["arm"], z=assignment["z"], propensity=assignment["propensity"],
                   b=assignment["b"], input_hash=item["record_hash"], assignment_hash=assignment["record_hash"],
                   x_hash=item["x_hash"], a0_hash=item["a0_hash"], memory_id=memory.memory_id,
                   source_task_id=memory.source_task_id, source_trace_id=memory.source_trace_id)
    elif any((directory / name).exists() for name in ("events", "result.json")):
        raise ValueError("single-arm execution started before persisted random assignment")
    if pending is not None:
        if assignment is None or pending["phase"] != "execution" or pending["assignment_hash"] != assignment["record_hash"]:
            raise ValueError("pending execution assignment identity mismatch")
        events = _events(directory / "events", assignment["record_hash"], "execution")
        arm = _live_arm(events, pending["partial_record"], item, assignment, StudySpec(**run["study"]), partial=True)
        row.update(arm_costs=arm["costs"], observed_costs=combine_costs(shared, arm["costs"]), pending_attempt=pending["attempt"])
        return seal(row)
    arm = result = None
    if (directory / "result.json").exists():
        result = checked(read_json(directory / "result.json"), schema="target-single-result.v1")
        if result["assignment_hash"] != assignment["record_hash"]:
            raise ValueError("single-arm result belongs to a different assignment")
        arm = _live_arm(_events(directory / "events", assignment["record_hash"], "execution"),
                        result["record"], item, assignment, StudySpec(**run["study"]))
        row.update(arm_costs=arm["costs"], observed_costs=combine_costs(shared, arm["costs"]),
                   response_identity_status=arm["response_identity_status"])
        row["operational_costs"]["execution"] = arm["operational_costs"]
    elif assignment is not None:
        _events(directory / "events", assignment["record_hash"], "execution")
    if not (directory / "closed.json").exists():
        return seal(row)
    closed = checked(read_json(directory / "closed.json"), schema="target-single-closed.v1")
    cleanup = closed["cleanup_errors"]
    if not isinstance(cleanup, list) or any(not isinstance(value, str) for value in cleanup):
        raise ValueError("single-session cleanup errors must be explicit strings")
    expected_status = "blocked" if cleanup else "completed" if result is not None else item["status"]
    expected_reason = "cleanup_error" if cleanup else item["reason"]
    expected = seal({"schema_version": "target-single-closed.v1", "run_hash": run["record_hash"],
        "trace_id": trace_id, "input_hash": item["record_hash"],
        "assignment_hash": assignment["record_hash"] if assignment else None,
        "status": expected_status, "reason": expected_reason, "cleanup_errors": cleanup})
    if closed != expected or (assignment is not None and result is None):
        raise ValueError("single-session close record has no matching observed execution or exclusion")
    if result is not None and result["cleanup_errors"] != cleanup:
        raise ValueError("single-session result and cleanup records disagree")
    row.update(status=closed["status"], reason=closed["reason"])
    if arm is not None and arm["outcome_valid"] and not cleanup:
        value = utility(arm["success"], row["observed_costs"], StudySpec(**run["study"]))
        row.update(outcome_valid=True, observed_success=arm["success"], utility=value, utility_valid=value is not None)
    elif arm is not None and not cleanup:
        row["reason"] = "execution_error"
    return seal(row)


def _identity_stop(directory, row, reference):
    """Validate served identities across tasks while retaining genuine drift errors."""
    shared_failure = False
    pending = read_json(directory / "pending.json") if (directory / "pending.json").exists() else None
    for phase, filename, journal in (("preparation", "preparation.json", "preparation_events"),
                                     ("execution", "result.json", "events")):
        saved = read_json(directory / filename) if (directory / filename).exists() else None
        record = saved.get("record", saved) if saved is not None else (
            pending["partial_record"] if pending is not None and pending["phase"] == phase else None)
        if record is None:
            continue
        error = record.get("error") or {}
        shared = record.get("status") == "error" and error.get("scope") in ("run_shared", "pair_shared", "session_shared")
        drift = False
        for path in sorted((directory / journal).glob("*.json")):
            event = read_json(path)["event"]
            if event["kind"] != "actor.completed":
                continue
            response = event["response"]
            for key in ("model", "fingerprint"):
                value = response.get(key) or None
                if value is not None and reference.get(key) is not None and value != reference[key]:
                    drift = True
                elif value is not None and reference.get(key) is None:
                    reference[key] = value
        if drift and not shared:
            raise ValueError("single-session actor identity drift was not retained as a shared failure")
        if drift and row["outcome_valid"]:
            raise ValueError("actor identity drift cannot provide an observed outcome")
        shared_failure = shared_failure or shared
    return shared_failure or row["reason"] == "cleanup_error" or row["status"] in ("pending", "unattempted")


def summarize_target_single(output_dir):
    """Recheck raw single-arm logs without models, environments, writes, or estimators."""
    from cave_mem_v3.interventions.single import PROTOCOL
    root = Path(output_dir)
    run = checked(read_json(root / "run.json"), schema="target-single-run.v1")
    target_partition(run.get("role"), run.get("validation_role"))
    validate_manifest(run["manifest"])
    spec = StudySpec(**run["study"])
    probability = run["assignment_probability"]
    if (run["protocol"] != PROTOCOL
            or run["manifest_hash"] != run["manifest"]["manifest_hash"]
            or isinstance(probability, bool) or not 0 < finite(probability, "assignment_probability") < 1
            or run["actor_capabilities"].get("stateless") is not True):
        raise ValueError("single-session run role/protocol/randomization identity mismatch")
    checked(run["bank"], "bank_hash", "verified-source-bank.v3")
    if run["bank_hash"] != run["bank"]["bank_hash"] or run["bank"].get("frozen") is not True:
        raise ValueError("single-session frozen source bank identity mismatch")
    _validate_source_partition(run, run["bank"])
    memories = {memory.memory_id: memory for memory in bank_memories(run["bank"])} if run["bank"]["memories"] else {}
    if not memories and (run["bank"].get("memory_count") != 0 or run["bank"].get("audit_sources") != {}):
        raise ValueError("single-session empty source bank is inconsistent")
    for memory in memories.values():
        require_current_evidence(memory.evidence, replay_pairs=2)
    validate_exposure_rules(run["exposure_rules"])
    if run["exposure_rules_hash"] != (digest(run["exposure_rules"]) if run["exposure_rules"] is not None else None):
        raise ValueError("single-session frozen exposure rules changed")
    RiskHazard(**run["risk_hazard"])
    tasks = sorted(partition_tasks(run["manifest"], run["role"], run.get("validation_role")), key=lambda task: task["task_id"])
    random.Random(spec.base_seed).shuffle(tasks)
    if (not tasks or run["task_order"] != [task["task_id"] for task in tasks]
            or set(run["environment_configs"]) != set(run["task_order"])
            or set(run["environment_capabilities"]) != set(run["task_order"])):
        raise ValueError("single-session task schedule differs from its frozen manifest")
    observations, expected_directories = [], set()
    reference, stopped = {"model": None, "fingerprint": None}, False
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        name = f"{index:06d}-{trace_id}"
        expected_directories.add(name)
        seed = stage_seed(spec.base_seed, task["task_id"], 0, "single-prefix-rollout")
        directory = root / "sessions" / name
        if stopped and directory.exists():
            raise ValueError("single-session attempts continued after a shared failure, pending request, or schedule gap")
        row = _session(directory, run, task, trace_id, seed, memories)
        stopped = _identity_stop(directory, row, reference) or stopped
        observations.append(row)
    if any(path.name not in expected_directories for path in (root / "sessions").glob("*")):
        raise ValueError("single-session directory contains unregistered tasks")
    counts = Counter(row["status"] for row in observations)
    return seal({"schema_version": "target-single-summary.v1", "run_hash": run["record_hash"],
        **validation_binding(run),
        "planned_rollouts": len(tasks), "completed_rollouts": counts["completed"],
        "pending_rollouts": counts["pending"], "pending_sessions": counts["pending"],
        "unattempted_rollouts": counts["unattempted"], "excluded_rollouts": counts["excluded"],
        "blocked_rollouts": counts["blocked"],
        "error_rollouts": sum(row["reason"] in ("execution_error", "preparation_error", "cleanup_error") for row in observations),
        "assigned_arms": dict(Counter(row["arm"] for row in observations if row["arm"] is not None)),
        "valid_outcomes": sum(row["outcome_valid"] for row in observations),
        "valid_utilities": sum(row["utility_valid"] for row in observations),
        "reasons": dict(Counter(row["reason"] for row in observations if row["reason"] is not None)),
        "observations": observations,
        "claim_boundary": "Only assigned-arm outcomes are observed. No individual counterfactual or paired label is identified. Fixture outcomes are software observations only."})
