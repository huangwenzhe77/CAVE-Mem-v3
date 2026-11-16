"""Final-test live executor. Separate raw schemas; no validation/training source reads."""
from __future__ import annotations
import copy
import time
from collections import Counter
from pathlib import Path
from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import PublicContext, StudySpec, digest, read_json, stage_seed, write_json
from cave_mem_v3.interventions.execution import TargetPendingError, run_frozen_policy_arm
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.interventions.single import _check_live_protocol, _input, _journal, _prepare_live
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.policy.dataset import _response_identities, _served_identity
from cave_mem_v3.policy.rollout import _assignment, _run_fields, _session
from cave_mem_v3.policy.final_evaluation import validate_final_evaluation_plan, final_policy_tasks, _policy_directory
from cave_mem_v3.target_data.collection import _close, _exclusive_json


def collect_final_policy(plan, policy_id, *, env_factory, actor_factory, encoder):
    validate_final_evaluation_plan(plan)
    entry = next(e for e in plan['policies'] if e['policy_id'] == policy_id)
    bundle = entry['bundle']
    validate_manifest(bundle['manifest'], verify_assets=True)
    tasks = final_policy_tasks(plan, policy_id)
    root = _policy_directory(plan, policy_id)
    if root.exists():
        raise ValueError('final policy output already exists; no retries')
    if bundle['policy_config']['kind'] != 'no_memory' and encoder.identity != bundle['encoder_identity']:
        raise ValueError('final encoder differs from frozen plan')
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
    run = seal({"schema_version": "policy-test-run.v1", **_run_fields(bundle),
        "bundle": bundle, "bundle_hash": bundle["record_hash"], "role": "test", "validation_role": None,
        "task_order": [t["task_id"] for t in tasks], "environment_configs": env_configs,
        "environment_capabilities": env_caps, "actor_capabilities": actor_caps,
        "protocol": "frozen_final_test_policy_live.v1", "final_authorization": {"plan": plan, "policy_id": policy_id},
        "final_plan_hash": plan["record_hash"], "policy_id": policy_id})
    if actor_caps != plan['actor_capabilities'] or any(env_configs[t['task_id']] != plan['environment_configs'][t['task_id']]
            or env_caps[t['task_id']] != plan['environment_capabilities'][t['task_id']] for t in tasks):
        raise ValueError('final environment or actor capabilities changed after preregistration')
    _exclusive_json(root / "run.json", run)
    memories = {m.memory_id: m for m in bank_memories(bundle["bank"])} if bundle["bank"]["memories"] else {}
    reference = {key: plan['served_identity'][key] for key in ('model', 'fingerprint')}
    spec = StudySpec(**run["study"])
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        seed = stage_seed(spec.base_seed, task["task_id"], 0, "policy-test-prefix")
        directory = root / "sessions" / f"{index:06d}-{trace_id}"
        _exclusive_json(directory / "attempt.json", seal({"schema_version": "policy-test-attempt.v1",
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
            pending = {"schema_version": "policy-test-pending.v1", "run_hash": run["record_hash"],
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
            _exclusive_json(directory / "result.json", seal({"schema_version": "policy-test-result.v1",
                "assignment_hash": assignment["record_hash"], "record": result}))
        _exclusive_json(directory / "closed.json", seal({"schema_version": "policy-test-closed.v1",
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
    summary = summarize_final_policy(root, plan, policy_id)
    write_json(root / "summary.json", summary)
    return summary


def summarize_final_policy(root, plan, policy_id):
    root = Path(root)
    validate_final_evaluation_plan(plan)
    entry = next(e for e in plan['policies'] if e['policy_id'] == policy_id)
    bundle = entry['bundle']
    run = checked(read_json(root / 'run.json'), schema='policy-test-run.v1')
    if (run.get('role') != 'test' or run.get('validation_role') is not None
            or run.get('protocol') != 'frozen_final_test_policy_live.v1' or run.get('policy_id') != policy_id
            or run.get('final_plan_hash') != plan['record_hash']
            or run.get('final_authorization') != {'plan': plan, 'policy_id': policy_id}
            or run.get('bundle') != bundle or run.get('bundle_hash') != bundle['record_hash']
            or any(run.get(k) != v for k, v in _run_fields(bundle).items())):
        raise ValueError('test run differs from frozen final plan/policy/role')
    tasks = final_policy_tasks(plan, policy_id)
    if run['task_order'] != [t['task_id'] for t in tasks]:
        raise ValueError('test policy task schedule differs from group assignment')
    if (run['environment_configs'] != {t['task_id']: plan['environment_configs'][t['task_id']] for t in tasks}
            or run['environment_capabilities'] != {t['task_id']: plan['environment_capabilities'][t['task_id']] for t in tasks}
            or run['actor_capabilities'] != plan['actor_capabilities']):
        raise ValueError('test environment/actor differs from frozen final plan')
    memories = {m.memory_id: m for m in bank_memories(bundle["bank"])} if bundle["bank"]["memories"] else {}
    rows, expected, stopped = [], set(), False
    reference = {key: plan['served_identity'][key] for key in ('model', 'fingerprint')}
    identities = []
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        name = f"{index:06d}-{trace_id}"
        expected.add(name)
        directory = root / "sessions" / name
        if stopped and directory.exists():
            raise ValueError("final test continued after incomplete/shared-failure session")
        seed = stage_seed(run["study"]["base_seed"], task["task_id"], 0, "policy-test-prefix")
        row, records = _session(directory, run, task, trace_id, seed, memories, schema_prefix="policy-test")
        identities.extend(_response_identities(records))
        shared_error = False
        for record in records:
            is_shared = (record.get("error") or {}).get("scope") in ("run_shared", "pair_shared", "session_shared")
            shared_error |= is_shared
            for key in reference:
                observed = record.get("response_identity", {}).get(key)
                if observed is not None and reference[key] is not None and observed != reference[key] and not is_shared:
                    raise ValueError("final test served model identity drift")
                if observed is not None:
                    reference[key] = observed
        stopped = shared_error or row["status"] in ("pending", "unattempted") or row["reason"] == "cleanup_error"
        rows.append(seal(row))
    if any(path.name not in expected for path in (root / "sessions").glob("*")):
        raise ValueError("unregistered final test task session")
    counts = Counter(r["status"] for r in rows)
    served = _served_identity(identities)
    if rows and all(r['status'] == 'completed' and r.get('costs', {}).get('model_calls') == 0 for r in rows):
        served['status'] = 'not_required_no_actor_calls'
    return seal({"schema_version": "policy-test-summary.v1", "run_hash": run["record_hash"],
        "bundle_hash": bundle["record_hash"], "role": "test", "validation_role": None, "policy_id": policy_id, "final_plan_hash": plan["record_hash"],
        "served_identity": served,
        "planned_tasks": len(tasks), "attempted_tasks": len(tasks) - counts["unattempted"],
        "pending_tasks": counts["pending"], "error_tasks": counts["error"],
        "observed_outcomes": sum(r["outcome_valid"] for r in rows),
        "observed_successes": sum(r["success"] is True for r in rows),
        "unknown_outcomes": sum(not r["outcome_valid"] for r in rows),
        "valid_utilities": sum(r["utility_valid"] for r in rows), "tasks": rows,
        "claim_boundary": "Complete assigned test task denominator. One live policy arm, no paired H/rescue. Unknown failures are not task failures; fixture observations are software checks only."})
