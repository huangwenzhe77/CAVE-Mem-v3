"""Complete frozen-policy validation rollouts, with read-only journal reconstruction."""
from __future__ import annotations
import copy
import random
import time
from collections import Counter
from pathlib import Path

from cave_mem_v3.common.contracts import PublicContext, StudySpec, digest, finite, plain, read_json, stage_seed, utility, write_json
from cave_mem_v3.agents.runtime import empty_costs
from cave_mem_v3.interventions.execution import TargetPendingError, run_frozen_policy_arm
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.interventions.outcomes import combine_costs
from cave_mem_v3.interventions.reminder import render_reminder
from cave_mem_v3.interventions.single import _check_live_protocol, _input, _journal, _prepare_live
from cave_mem_v3.interventions.single_outcomes import _events, _preparation, _live_arm
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.policy.frozen import load_frozen_policy, validate_bundle, policy_decision
from cave_mem_v3.target_data.collection import _close, _exclusive_json
from cave_mem_v3.target_data.partitions import partition_tasks
from cave_mem_v3.benchmarks.tasks import validate_manifest


def _role(role, validation_role):
    if (role, validation_role) != ("validation", "selection"):
        raise ValueError("policy rollout only permits validation/selection; test is locked")


def _tasks(bundle):
    tasks = sorted(partition_tasks(bundle["manifest"], "validation", "selection"), key=lambda t: t["task_id"])
    random.Random(bundle["study"]["base_seed"]).shuffle(tasks)
    return tasks


def _assignment(bundle, item, trace_id, memories):
    decision = policy_decision(bundle, item, trace_id)
    memory = memories.get(item["pre_treatment_state"].get("memory_id"))
    reminder = render_reminder(memory, item["a0"]) if memory is not None else None
    return seal({"schema_version": "policy-rollout-assignment.v1", "decision": decision,
        "arm": decision["arm"], "execution_seed": decision["execution_seed"], "reminder": reminder,
        "input_hash": item["record_hash"], "bundle_hash": bundle["record_hash"]})


def _run_fields(bundle):
    return {key: copy.deepcopy(bundle[key]) for key in ("manifest", "manifest_hash", "bank", "bank_hash",
        "actor_config", "encoder_identity", "study", "risk_hazard", "exposure_rules", "exposure_rules_hash")}


def existing_validation_summary(root, bundle, config_identity=None):
    root = Path(root)
    if not (root / "run.json").exists():
        return None
    stored = checked(read_json(root / "run.json"), schema="policy-validation-run.v1")
    _role(stored.get("role"), stored.get("validation_role"))
    if stored["bundle"] != bundle or stored.get("config_identity") != config_identity:
        raise ValueError("validation package changed or execution configuration changed; use a new output directory")
    report = summarize_policy_validation(root)
    if report["attempted_tasks"] == len(_tasks(bundle)) and not report["pending_tasks"]:
        return report
    raise ValueError("incomplete validation run preserved; no automatic retry/resume")


def collect_policy_validation(bundle_path, *, env_factory, actor_factory, encoder, output_dir,
                              role="validation", validation_role="selection", config_identity=None):
    _role(role, validation_role)
    bundle = load_frozen_policy(bundle_path)
    validate_manifest(bundle["manifest"], verify_assets=True)
    tasks = _tasks(bundle)
    if bundle["policy_config"]["kind"] != "no_memory" and encoder.identity != bundle["encoder_identity"]:
        raise ValueError("execution encoder differs from frozen package")
    root = Path(output_dir)
    existing = existing_validation_summary(root, bundle, config_identity)
    if existing is not None:
        return existing
    if root.exists() and any(root.iterdir()):
        raise ValueError("new validation directory must be empty")
    used, env_configs, env_caps = [], {}, {}
    actor = actor_factory()
    used.append(actor)
    try:
        if actor.configuration() != bundle["actor_config"]:
            raise ValueError("actor differs from frozen package")
        actor_caps = actor.capabilities()
        for task in tasks:
            env = env_factory(copy.deepcopy(task))
            if any(env is instance for instance in used):
                raise ValueError("factory reused live instance")
            used.append(env)
            try:
                _check_live_protocol(env, actor)
                env_configs[task["task_id"]], env_caps[task["task_id"]] = env.configuration(), env.capabilities()
            finally:
                _close(env)
    finally:
        _close(actor)
    run = seal({"schema_version": "policy-validation-run.v1", **_run_fields(bundle),
        "bundle": bundle, "bundle_hash": bundle["record_hash"], "role": role, "validation_role": validation_role,
        "task_order": [t["task_id"] for t in tasks], "environment_configs": env_configs,
        "environment_capabilities": env_caps, "actor_capabilities": actor_caps,
        "protocol": "full_task_once_selected_policy_live.v1", "config_identity": config_identity})
    _exclusive_json(root / "run.json", run)
    memories = {m.memory_id: m for m in bank_memories(bundle["bank"])} if bundle["bank"]["memories"] else {}
    reference = {"model": None, "fingerprint": None}
    artifacts = bundle["artifacts"]
    if "model" in artifacts:
        reference = {key: artifacts["model"]["training"]["served_identity"].get(key) for key in reference}
    spec = StudySpec(**run["study"])
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        seed = stage_seed(spec.base_seed, task["task_id"], 0, "policy-validation-prefix")
        directory = root / "sessions" / f"{index:06d}-{trace_id}"
        _exclusive_json(directory / "attempt.json", seal({"schema_version": "policy-validation-attempt.v1",
            "run_hash": run["record_hash"], "task_id": task["task_id"], "trace_id": trace_id, "seed": seed}))
        env = actor = None
        preparation = item = assignment = result = pending = None
        cleanup = []
        try:
            actor = actor_factory()
            env = env_factory(copy.deepcopy(task))
            if any(instance is old for instance in (actor, env) for old in used):
                raise ValueError("factory reused an execution instance")
            used.extend((actor, env))
            preparation = _prepare_live(run, task, trace_id, seed, env, actor, encoder,
                _journal(directory / "preparation_events", run["record_hash"], "preparation"), reference)
            _exclusive_json(directory / "preparation.json", preparation)
            item = _input(run, task, preparation)
            _exclusive_json(directory / "input.json", item)
            if preparation["status"] == "prepared":
                decision_started = time.perf_counter()
                assignment = _assignment(bundle, item, trace_id, memories)
                decision_seconds = time.perf_counter() - decision_started
                _exclusive_json(directory / "assignment.json", assignment)
                _exclusive_json(directory / "decision_timing.json", seal({"schema_version": "policy-decision-timing.v1",
                    "assignment_hash": assignment["record_hash"], "input_hash": item["record_hash"],
                    "wall_seconds": decision_seconds, "scope": "local_gate_uncertainty_guard_and_reminder_render_once"}))
                expected = {**reference, **{k: v for k, v in item["response_identity"].items() if k in reference and v is not None}}
                result = run_frozen_policy_arm(env, actor, context=PublicContext(**item["context"]),
                    first_action=item["a0"], arm=assignment["arm"],
                    reminder=assignment["reminder"]["text"] if assignment["arm"] == "remind" else None,
                    spec=spec, seed=assignment["execution_seed"], remaining_model_calls=item["remaining_model_calls"],
                    remaining_steps=item["remaining_steps"], expected_response_identity=expected,
                    journal=_journal(directory / "events", assignment["record_hash"], "execution"))
        except TargetPendingError as exc:
            pending = {"schema_version": "policy-validation-pending.v1", "run_hash": run["record_hash"],
                "trace_id": trace_id, "phase": "execution" if assignment else "preparation",
                "assignment_hash": assignment["record_hash"] if assignment else None,
                "partial_record": exc.partial_record, "attempt": exc.attempt}
        finally:
            for instance in (env, actor):
                if instance is not None:
                    try:
                        _close(instance)
                    except Exception as exc:
                        cleanup.append(type(exc).__name__)
        if pending:
            _exclusive_json(directory / "pending.json", seal({**pending, "cleanup_errors": cleanup}))
            break
        if result:
            _exclusive_json(directory / "result.json", seal({"schema_version": "policy-validation-result.v1",
                "assignment_hash": assignment["record_hash"], "record": result}))
        _exclusive_json(directory / "closed.json", seal({"schema_version": "policy-validation-closed.v1",
            "run_hash": run["record_hash"], "trace_id": trace_id, "input_hash": item["record_hash"],
            "assignment_hash": assignment["record_hash"] if assignment else None,
            "result_hash": result["record_hash"] if result else None, "cleanup_errors": cleanup}))
        shared_error = (preparation.get("error") or {}).get("scope") == "run_shared" or (result and (result.get("error") or {}).get("scope") in ("pair_shared", "session_shared"))
        if cleanup or shared_error:
            break
        for record in (preparation, result or {}):
            for key in reference:
                if record.get("response_identity", {}).get(key) is not None:
                    reference[key] = record["response_identity"][key]
    summary = summarize_policy_validation(root)
    write_json(root / "summary.json", summary)
    return summary


def _session(directory, run, task, trace_id, seed, memories, *, schema_prefix="policy-validation"):
    if schema_prefix not in ("policy-validation", "policy-test"):
        raise ValueError("unsupported policy session schema")
    row = {"task_id": task["task_id"], "trace_id": trace_id, "status": "unattempted", "reason": None,
        "success": None, "utility": None, "outcome_valid": False, "utility_valid": False, "costs": None,
        "preparation_costs": None, "decision_costs": None, "continuation_costs": None, "operational_costs": {},
        "arm": None, "decision": None, "termination_reason": None,
        "group_id": task.get("group_id"), "groups": task.get("groups", []), "template_id": task.get("template_id")}
    if not directory.exists():
        return row, []
    allowed = {"attempt.json", "preparation.json", "input.json", "assignment.json", "result.json", "closed.json", "pending.json", "decision_timing.json", "preparation_events", "events"}
    if any(path.name not in allowed for path in directory.iterdir()):
        raise ValueError("unregistered validation session artifact")
    expected = seal({"schema_version": schema_prefix + "-attempt.v1", "run_hash": run["record_hash"],
        "task_id": task["task_id"], "trace_id": trace_id, "seed": seed})
    if read_json(directory / "attempt.json") != expected:
        raise ValueError("validation attempt identity mismatch")
    row.update(status="pending", reason="live_session_incomplete")
    pending = checked(read_json(directory / "pending.json"), schema=schema_prefix + "-pending.v1") if (directory / "pending.json").exists() else None
    if pending and (pending["run_hash"] != run["record_hash"] or pending["trace_id"] != trace_id or (directory / "closed.json").exists() or (directory / "result.json").exists()):
        raise ValueError("validation pending/completed conflict")
    events = _events(directory / "preparation_events", run["record_hash"], "preparation")
    identities = []
    if not (directory / "preparation.json").exists():
        if any((directory / name).exists() for name in ("input.json", "assignment.json", "result.json", "closed.json", "events")):
            raise ValueError("validation execution lacks preparation")
        if pending:
            if pending["phase"] != "preparation" or pending["assignment_hash"] is not None:
                raise ValueError("pending prefix cannot have treatment assignment")
            row["costs"] = row["preparation_costs"] = _preparation(events, pending["partial_record"], run, task, trace_id, seed, memories, partial=True)
        return row, identities
    preparation = read_json(directory / "preparation.json")
    shared = _preparation(events, preparation, run, task, trace_id, seed, memories)
    identities.append(preparation)
    row.update(costs=shared, preparation_costs=shared, termination_reason=preparation.get("termination_reason"))
    row["operational_costs"]["preparation"] = preparation["operational_costs"]
    if not (directory / "input.json").exists():
        if any((directory / name).exists() for name in ("assignment.json", "result.json", "closed.json", "events")):
            raise ValueError("validation continuation lacks frozen input")
        return row, identities
    item = checked(read_json(directory / "input.json"), schema="target-single-input.v1")
    if item != _input(run, task, preparation):
        raise ValueError("validation input differs from original preparation")
    assignment = None
    if (directory / "assignment.json").exists():
        if preparation["status"] != "prepared":
            raise ValueError("unprepared validation session was assigned")
        assignment = read_json(directory / "assignment.json")
        if assignment != _assignment(run["bundle"], item, trace_id, memories):
            raise ValueError("frozen policy decision was changed after preparation")
        row.update(arm=assignment["arm"], decision=assignment["decision"])
    elif any((directory / name).exists() for name in ("result.json", "events")):
        raise ValueError("policy execution began before persisted decision")
    decision_costs = empty_costs()
    if (directory / "decision_timing.json").exists():
        timing = checked(read_json(directory / "decision_timing.json"), schema="policy-decision-timing.v1")
        if assignment is None or timing["assignment_hash"] != assignment["record_hash"] or timing["input_hash"] != item["record_hash"] or timing["scope"] != "local_gate_uncertainty_guard_and_reminder_render_once":
            raise ValueError("decision timing identity/scope mismatch")
        decision_costs["wall_seconds"] = finite(timing["wall_seconds"], "decision wall_seconds", minimum=0)
        row["decision_costs"] = decision_costs
        row["costs"] = combine_costs(shared, decision_costs)
    elif any((directory / name).exists() for name in ("result.json", "events")):
        raise ValueError("policy execution lacks persisted decision timing")
    arm = result = None
    if pending:
        if assignment is None or pending["phase"] != "execution" or pending["assignment_hash"] != assignment["record_hash"]:
            raise ValueError("pending policy execution assignment mismatch")
        arm = _live_arm(_events(directory / "events", assignment["record_hash"], "execution"), pending["partial_record"], item, assignment, StudySpec(**run["study"]), partial=True, policy_execution=True)
    elif (directory / "result.json").exists():
        result = checked(read_json(directory / "result.json"), schema=schema_prefix + "-result.v1")
        if assignment is None or result["assignment_hash"] != assignment["record_hash"]:
            raise ValueError("policy result assignment mismatch")
        arm = _live_arm(_events(directory / "events", assignment["record_hash"], "execution"), result["record"], item, assignment, StudySpec(**run["study"]), policy_execution=True)
        if result["record"].get("environment_capabilities") != run["environment_capabilities"][task["task_id"]]:
            raise ValueError("validation execution environment capabilities changed")
        identities.append(result["record"])
        row["termination_reason"] = result["record"]["termination_reason"]
    if arm:
        row.update(continuation_costs=arm["costs"], costs=combine_costs(shared, decision_costs, arm["costs"]))
        row["operational_costs"]["continuation"] = arm["operational_costs"]
    if not (directory / "closed.json").exists():
        return row, identities
    closed = checked(read_json(directory / "closed.json"), schema=schema_prefix + "-closed.v1")
    cleanup = closed["cleanup_errors"]
    expected = seal({"schema_version": schema_prefix + "-closed.v1", "run_hash": run["record_hash"],
        "trace_id": trace_id, "input_hash": item["record_hash"], "assignment_hash": assignment["record_hash"] if assignment else None,
        "result_hash": result["record"]["record_hash"] if result else None, "cleanup_errors": cleanup})
    if closed != expected or assignment is not None and result is None or not isinstance(cleanup, list) or any(not isinstance(v, str) for v in cleanup):
        raise ValueError("validation close identity/completion mismatch")
    row.update(status="completed", reason=None)
    if cleanup:
        row.update(status="error", reason="cleanup_error")
    elif arm and arm["outcome_valid"]:
        row.update(success=arm["success"], outcome_valid=True)
    elif arm:
        row.update(status="error", reason="execution_error")
    elif preparation["status"] in ("no_selection", "preparation_error") and type(preparation.get("unselected_terminal_outcome")) is bool:
        row.update(success=preparation["unselected_terminal_outcome"], outcome_valid=True,
            reason=("ordinary_actor_no_memory" if run["bundle"]["policy_config"]["kind"] == "no_memory" else "no_checkpoint") if preparation["status"] == "no_selection" else "invalid_prefix_action")
    else:
        row.update(status="error", reason="preparation_error")
    if row["outcome_valid"]:
        row["utility"] = utility(row["success"], row["costs"], StudySpec(**run["study"]))
        row["utility_valid"] = row["utility"] is not None
    return row, identities


def summarize_policy_validation(output_dir):
    root = Path(output_dir)
    run = checked(read_json(root / "run.json"), schema="policy-validation-run.v1")
    _role(run.get("role"), run.get("validation_role"))
    if run.get("protocol") != "full_task_once_selected_policy_live.v1" or run.get("actor_capabilities", {}).get("stateless") is not True:
        raise ValueError("validation protocol/stateless actor capability mismatch")
    bundle = validate_bundle(run["bundle"])
    if run["bundle_hash"] != bundle["record_hash"] or any(run[k] != v for k, v in _run_fields(bundle).items()):
        raise ValueError("validation run differs from frozen policy bundle")
    tasks = _tasks(bundle)
    if (run["task_order"] != [t["task_id"] for t in tasks]
            or set(run["environment_configs"]) != set(run["task_order"])
            or set(run["environment_capabilities"]) != set(run["task_order"])
            or any(not isinstance(caps, dict) or not caps for caps in run["environment_capabilities"].values())):
        raise ValueError("validation task schedule differs from frozen manifest")
    memories = {m.memory_id: m for m in bank_memories(bundle["bank"])} if bundle["bank"]["memories"] else {}
    rows, expected, stopped = [], set(), False
    reference = {"model": None, "fingerprint": None}
    if "model" in bundle["artifacts"]:
        reference = {k: bundle["artifacts"]["model"]["training"]["served_identity"].get(k) for k in reference}
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        name = f"{index:06d}-{trace_id}"
        expected.add(name)
        directory = root / "sessions" / name
        if stopped and directory.exists():
            raise ValueError("validation continued after incomplete/shared-failure session")
        seed = stage_seed(run["study"]["base_seed"], task["task_id"], 0, "policy-validation-prefix")
        row, records = _session(directory, run, task, trace_id, seed, memories)
        shared_error = False
        for record in records:
            is_shared = (record.get("error") or {}).get("scope") in ("run_shared", "pair_shared", "session_shared")
            shared_error |= is_shared
            for key in reference:
                observed = record.get("response_identity", {}).get(key)
                if observed is not None and reference[key] is not None and observed != reference[key] and not is_shared:
                    raise ValueError("validation served model identity drift")
                if observed is not None:
                    reference[key] = observed
        stopped = shared_error or row["status"] in ("pending", "unattempted") or row["reason"] == "cleanup_error"
        rows.append(seal(row))
    if any(path.name not in expected for path in (root / "sessions").glob("*")):
        raise ValueError("unregistered validation task session")
    counts = Counter(r["status"] for r in rows)
    return seal({"schema_version": "policy-validation-summary.v1", "run_hash": run["record_hash"],
        "bundle_hash": bundle["record_hash"], "role": "validation", "validation_role": "selection",
        "planned_tasks": len(tasks), "attempted_tasks": len(tasks) - counts["unattempted"],
        "pending_tasks": counts["pending"], "error_tasks": counts["error"],
        "observed_outcomes": sum(r["outcome_valid"] for r in rows),
        "observed_successes": sum(r["success"] is True for r in rows),
        "unknown_outcomes": sum(not r["outcome_valid"] for r in rows),
        "valid_utilities": sum(r["utility_valid"] for r in rows), "tasks": rows,
        "claim_boundary": "Complete validation task denominator. One live policy arm, no paired H/rescue. Unknown failures are not task failures; fixture observations are software checks only."})
