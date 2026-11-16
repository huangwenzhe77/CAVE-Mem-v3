"""Read frozen preparation artifacts without sampling actions or observing outcomes."""
from __future__ import annotations
from cave_mem_v3.target_data.partitions import target_partition, partition_fields, partition_tasks, check_partition_resume

import copy
import math
from pathlib import Path

from cave_mem_v3.agents.runtime import add_usage, empty_costs
from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import StudySpec, digest, finite, plain, read_json, stage_seed
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.predicates import evaluate_conditions
from cave_mem_v3.memory_generation.replay import episode_response_identity, require_current_evidence
from cave_mem_v3.target_data.pre_treatment_state import _context, _exposure, _public_action, validate_exposure_rules
from cave_mem_v3.target_data.retrieval import _query_text
from cave_mem_v3.target_data.sampling import OnlineCheckpointSampler, RiskHazard


def seal(value: dict, field: str = "record_hash") -> dict:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = digest(result)
    return result


def checked(value: dict, field: str = "record_hash", schema: str | None = None) -> dict:
    if (not isinstance(value, dict) or value.get(field) != digest({k: v for k, v in value.items() if k != field})
            or schema is not None and value.get("schema_version") != schema):
        raise ValueError(f"invalid {schema or 'record'} schema/checksum")
    return value


def _validate_source_partition(run: dict, bank: dict) -> None:
    """Bind every private source to the same complete, frozen task partition."""
    manifest_hash = run["manifest_hash"]
    tasks = {task["task_id"]: task for task in run["manifest"]["tasks"]}
    metadata = bank.get("metadata", {})
    if (not isinstance(metadata, dict) or "manifest_hash" in metadata
            and metadata["manifest_hash"] != manifest_hash):
        raise ValueError("bank metadata manifest differs from the preparation manifest")
    sources = bank.get("audit_sources")
    if not isinstance(sources, dict):
        raise ValueError("bank needs audit source trajectories")
    for trace in sources.values():
        if not isinstance(trace, dict) or trace.get("manifest_hash") != manifest_hash:
            raise ValueError("bank source manifest differs from the preparation manifest")
        task_id = trace.get("task_id")
        task = tasks.get(task_id)
        if task is None or task.get("role") != "history" or trace.get("role") != "history":
            raise ValueError("bank source task is not in the preparation manifest history partition")
        episode = trace.get("episode", {})
        if (trace.get("group_id") != task.get("group_id")
                or episode.get("task_id") != task_id
                or episode.get("environment_config", {}).get("task_id") != task_id
                or any(step.get("context", {}).get("task_id") != task_id
                       for step in episode.get("steps", []))):
            raise ValueError("bank source task identity differs from its manifest entry")


def _validate_no_selection(run: dict, trace: dict) -> None:
    """Reconstruct the denominator decision without copying terminal outcomes."""
    episode = checked(trace["episode"], "trace_hash", "source-trajectory.v3")
    sampler = OnlineCheckpointSampler(role=trace["role"], task_id=trace["task_id"],
        trace_id=trace["trace_id"], seed=trace["seed"], hazard=RiskHazard(**run["risk_hazard"]))
    prior = []
    for number, step in enumerate(episode["steps"]):
        if step.get("index") != number:
            raise ValueError("target steps are not a contiguous prefix")
        decision = sampler.consider(step["context"], step["checkpoint"], number, prior,
                                    checkpoint_hash=step["checkpoint_hash"])
        if decision != step.get("checkpoint_sampling") or decision.get("selected"):
            raise ValueError("no-selection trace differs from frozen sampling")
        prior.append({key: copy.deepcopy(step[key]) for key in (
            "action", "result", "validation_error") if key in step})
        if isinstance(step.get("result"), dict):
            prior[-1]["valid"] = step["result"].get("valid", True)
        elif "validation_error" in step:
            prior[-1]["valid"] = False
    if sampler.finalize() != episode.get("checkpoint_sampling"):
        raise ValueError("no-selection metadata differs from frozen sampling")


def _validate_shared_costs(episode: dict) -> None:
    """Recompute shared preparation charges from recorded responses and tools."""
    expected = empty_costs()
    actual = episode.get("costs")
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError("preparation cost fields differ from the runtime cost schema")
    for step in episode["steps"]:
        usage = step.get("actor_response", {}).get("usage")
        if not isinstance(usage, dict):
            raise ValueError("prepared action has no recorded response usage")
        add_usage(expected, usage)
        result = step.get("result")
        if result is not None:
            if not isinstance(result, dict):
                raise ValueError("executed preparation step has invalid tool results")
            expected["tool_calls"] += 1.0
            cost = result.get("tool_cost")
            expected["tool_cost"] = (None if cost is None or expected["tool_cost"] is None
                                     else expected["tool_cost"] + finite(cost, "tool_cost", minimum=0))
    for name, value in actual.items():
        if value is not None:
            finite(value, name, minimum=0)
        if name == "wall_seconds":
            # Wall time is measured rather than derivable from model/tool usage.
            finite(value, name, minimum=0)
        elif value != expected[name]:
            raise ValueError(f"preparation shared {name} differs from recorded response/tool usage")


def _validate_selection(run: dict, trace: dict) -> tuple[dict, dict]:
    episode = trace["episode"]
    sample = episode.get("checkpoint_sampling") or {}
    index = sample.get("selected_step_index")
    steps = episode.get("steps")
    if type(index) is not int or index < 0 or not isinstance(steps, list) or len(steps) != index + 1:
        raise ValueError("prepared trace must end exactly at its selected step")
    sampler = OnlineCheckpointSampler(role=trace["role"], task_id=trace["task_id"],
        trace_id=trace["trace_id"], seed=trace["seed"], hazard=RiskHazard(**run["risk_hazard"]))
    prior = []
    for number, step in enumerate(steps):
        if step.get("index") != number:
            raise ValueError("target steps are not a contiguous prefix")
        response = step.get("actor_response")
        if (not isinstance(response, dict) or response.get("action") != step.get("action")
                or response.get("parse_error") is not None):
            raise ValueError("preparation lacks a successfully parsed cached actor action")
        decision = sampler.consider(step["context"], step["checkpoint"], number, prior,
                                    checkpoint_hash=step["checkpoint_hash"])
        if decision != step.get("checkpoint_sampling"):
            raise ValueError("checkpoint sampling decision differs from frozen prefix and seed")
        if number < index:
            if (decision.get("selected") or not isinstance(step.get("result"), dict)
                    or step["result"].get("valid") is not True
                    or step["result"].get("terminal") is not False
                    or step["result"].get("context") != steps[number + 1].get("context")):
                raise ValueError("invalid executed preparation prefix")
            prior.append({"action": step["action"], "result": step["result"],
                          "valid": step["result"].get("valid", True)})
    expected_sample = sampler.finalize()
    expected_sample["selected_action"] = steps[index]["action"]
    if expected_sample != sample:
        raise ValueError("selection metadata/checkpoint differs from frozen sampling")
    step = steps[index]
    if (step.get("result") is not None or "validation_error" in step
            or episode.get("status") != "pre_treatment_ready"
            or episode.get("success") is not None or episode.get("utility") is not None
            or "private_verifier" in episode):
        raise ValueError("selected action was executed, validated, or has outcome access")
    return step, sample


def _prepared(run: dict, trace: dict, task: dict, bank: dict, memories: dict) -> dict:
    episode = checked(trace["episode"], "trace_hash", "source-trajectory.v3")
    if (episode.get("task_id") != task["task_id"] or episode.get("seed") != trace["seed"]
            or episode.get("spec") != run["study"]
            or episode.get("actor_config") != run["actor_config"]
            or episode.get("environment_config") != run["environment_configs"][task["task_id"]]
            or episode.get("capabilities", {}).get("actor", {}).get("stateless") is not True):
        raise ValueError("preparation actor/environment identity or statelessness mismatch")
    identity = episode_response_identity(episode)
    step, sample = _validate_selection(run, trace)
    _validate_shared_costs(episode)
    state = checked(step["pre_treatment_state"], schema="pre-treatment-state-record.v1")
    context = _context(step["context"])
    action = _public_action(step["action"])
    policy_context = plain(context)
    policy_context.pop("task_id")
    x = state["x"]
    if (state.get("task_id") != task["task_id"] or context.task_id != task["task_id"]
            or state.get("bank_hash") != bank["bank_hash"]
            or state.get("context_hash") != digest(plain(context))
            or state.get("a0_hash") != digest(action) or state.get("x_hash") != digest(x)
            or state.get("randomization_status") != "not_assigned" or state.get("outcome_access") != "none"
            or x.get("schema_version") != "pre-treatment-state.v1"
            or x.get("a0") != action or x.get("context") != policy_context):
        raise ValueError("prepared state/context/action/bank identity mismatch")
    feature = x.get("feature_map", {})
    vector = feature.get("text_embedding")
    if (feature.get("status") != "frozen_text_encoder_plus_structured.v1"
            or feature.get("structured_schema_version") != "pre-treatment-structured.v1"
            or feature.get("encoder_identity") != run["pre_treatment"]["encoder_identity"]
            or feature.get("text_query_hash") != digest(_query_text(context, action))
            or not isinstance(vector, list) or not vector
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector)
            or sum(v * v for v in vector) <= 0):
        raise ValueError("frozen feature map is incomplete or inconsistent")
    spec = StudySpec(**run["study"])
    if step["index"] >= spec.max_steps:
        raise ValueError("selected checkpoint exceeds the frozen step budget")
    calls = episode["costs"].get("model_calls")
    if (isinstance(calls, bool) or not isinstance(calls, (int, float))
            or calls != len(episode["steps"]) or calls > spec.max_model_calls
            or episode["costs"].get("tool_calls") != step["index"]):
        raise ValueError("preparation budget disagrees with cached responses and executed prefix")
    remaining_calls = spec.max_model_calls - int(calls)
    remaining_steps = max(0, min(spec.max_steps - step["index"], context.remaining_steps))
    if x.get("b") != {"status": "known", "remaining": {
            "model_calls": remaining_calls, "environment_steps": remaining_steps}}:
        raise ValueError("frozen remaining budget is inconsistent; never rewrite existing x")
    memory_id = state.get("memory_id")
    if memory_id is None:
        if (state.get("status") != "prepared_no_memory" or memories or x.get("m") is not None
                or x.get("v") != "?" or x.get("rho") != "?"
                or x.get("retrieval", {}).get("status") != "no_match"):
            raise ValueError("invalid no-memory preparation")
        return {"status": "excluded", "reason": "no_memory"}
    memory = memories.get(memory_id)
    if memory is None or state.get("status") != "prepared" or x.get("v") != 1:
        raise ValueError("prepared memory is not in the verified source bank")
    require_current_evidence(memory.evidence, replay_pairs=2)
    expected_memory = {name: plain(getattr(memory, name)) for name in (
        "mechanism", "problem_action", "correction", "positive_conditions", "exclusion_conditions")}
    expected_evidence = {key: copy.deepcopy(value) for key, value in memory.quality.items()
        if key in ("v", "reason", "scope", "n_replay_pairs", "mean_difference", "acceptance_epsilon",
                   "decision_rule", "claim_boundary")}
    rules = run["pre_treatment"].get("exposure_rules")
    if run["pre_treatment"].get("exposure_rules_hash") != (digest(rules) if rules is not None else None):
        raise ValueError("frozen exposure rubric missing or changed; prepare a new run")
    validate_exposure_rules(rules)
    if (x.get("m") != expected_memory or x.get("E") != expected_evidence
            or x.get("rho") != evaluate_conditions(memory.positive_conditions, memory.exclusion_conditions, context)
            or x.get("e") != _exposure(memory_id=memory_id, context=context, action=action, rules=rules)):
        raise ValueError("memory evidence or public applicability/exposure fields differ")
    if x.get("retrieval", {}).get("status") != "retrieved":
        raise ValueError("prepared state lacks completed retrieval")
    if remaining_calls < 1 or remaining_steps < 1:
        return {"status": "excluded", "reason": "insufficient_budget"}
    return {"status": "ready", "reason": None, "task": copy.deepcopy(task),
        "checkpoint": copy.deepcopy(step["checkpoint"]), "checkpoint_hash": step["checkpoint_hash"],
        "context": plain(context), "a0": action, "a0_hash": state["a0_hash"], "x": copy.deepcopy(x),
        "x_hash": state["x_hash"], "state_hash": state["record_hash"], "memory_id": memory_id,
        "shared_costs": copy.deepcopy(episode["costs"]), "response_identity": identity,
        "remaining_model_calls": remaining_calls, "remaining_steps": remaining_steps,
        "inclusion_probability": sample["inclusion_probability"],
        "actor_config": copy.deepcopy(run["actor_config"]),
        "environment_config": copy.deepcopy(run["environment_configs"][task["task_id"]])}


def load_prepared_run(target_dir: str | Path, bank: dict) -> tuple[dict, list[dict]]:
    """Return a frozen input inventory, including every exclusion and error denominator."""
    root = Path(target_dir)
    run = checked(read_json(root / "run.json"), "run_hash", "target-run.v1")
    target_partition(run.get("role"), run.get("validation_role"))
    if (run.get("pre_treatment", {}).get("mode")
            != "freeze_before_selected_action_execution.v1"):
        raise ValueError("paired target input requires frozen prepare-target artifacts")
    validate_manifest(run["manifest"], verify_assets=True)
    if run.get("manifest_hash") != run["manifest"]["manifest_hash"]:
        raise ValueError("prepared manifest hash mismatch")
    checked(bank, "bank_hash", "verified-source-bank.v3")
    if bank.get("frozen") is not True or run["pre_treatment"].get("bank_hash") != bank["bank_hash"]:
        raise ValueError("paired bank differs from the frozen preparation bank")
    _validate_source_partition(run, bank)
    if bank.get("memories"):
        memories = {m.memory_id: m for m in bank_memories(bank)}
    elif bank.get("memories") == [] and bank.get("memory_count") == 0 and bank.get("audit_sources") == {}:
        memories = {}
    else:
        raise ValueError("invalid empty bank")
    tasks = {task["task_id"]: task for task in partition_tasks(run["manifest"], run["role"], run.get("validation_role"))}
    if (len(run["task_order"]) != len(set(run["task_order"])) or set(tasks) != set(run["task_order"])):
        raise ValueError("prepared task schedule differs from manifest")
    records, expected_names = [], set()
    for index, task_id in enumerate(run["task_order"]):
        trace_id = digest([run["run_hash"], task_id])[:24]
        filename = f"{index:06d}-{trace_id}.json"
        expected_names.add(filename)
        path = root / "traces" / filename
        base = {"schema_version": "target-pair-input.v1", "task_id": task_id, "trace_id": trace_id,
                "run_hash": run["run_hash"], "trace_hash": None, "raw_file_hash": None}
        if not path.exists():
            base.update(status="blocked", reason="preparation_missing")
        else:
            import hashlib
            base["raw_file_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                trace = checked(read_json(path), "trace_hash", "target-trace.v1")
                base["trace_hash"] = trace["trace_hash"]
                if (trace.get("run_hash") != run["run_hash"] or trace.get("task_id") != task_id
                        or trace.get("role") != run["role"] or trace.get("trace_id") != trace_id
                        or trace.get("group_id") != tasks[task_id].get("group_id")
                        or trace.get("validation_role") != run.get("validation_role")
                        or trace.get("manifest_hash") != run["manifest_hash"] or trace.get("run_index") != index
                        or trace.get("seed") != stage_seed(StudySpec(**run["study"]).base_seed, task_id, 0, "target_rollout")):
                    raise ValueError("prepared trace identity mismatch")
                episode = trace["episode"]
                sample = episode.get("checkpoint_sampling") or {}
                if episode.get("status") == "error" or trace.get("cleanup_errors"):
                    base.update(status="blocked", reason="preparation_error")
                elif sample.get("status") == "no_selection":
                    # The original no-selection trace may have ended naturally. No outcome is copied.
                    _validate_no_selection(run, trace)
                    base.update(status="excluded", reason="no_selection")
                elif sample.get("status") != "selected":
                    base.update(status="blocked", reason="sampling_incomplete")
                else:
                    base.update(_prepared(run, trace, tasks[task_id], bank, memories))
            except (ValueError, KeyError, TypeError, OSError) as exc:
                base.update(status="blocked", reason="invalid_prepared_input", detail=str(exc))
        records.append(seal(base))
    if any(path.name not in expected_names for path in (root / "traces").glob("*.json")):
        raise ValueError("prepared directory contains unregistered traces")
    return copy.deepcopy(run), records
