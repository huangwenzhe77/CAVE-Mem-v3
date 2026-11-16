"""Select paired or randomized single-arm collection while the target is live.

The restoration audit runs before the original selected session is closed. A
failed audit routes to a single arm in that unchanged session. Once a paired
arm starts, failures remain paired failures and never become single-arm data.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

from cave_mem_v3.common.contracts import PublicContext, StudySpec, digest, finite, plain, read_json, stage_seed, write_json
from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.interventions.collection import _audit, execute_audited_pair
from cave_mem_v3.interventions.execution import TargetPendingError, run_live_target_arm
from cave_mem_v3.interventions.inputs import checked, seal, _validate_source_partition
from cave_mem_v3.interventions.reminder import render_reminder
from cave_mem_v3.interventions.single import (_check_live_protocol, _input, _journal,
    _prepare_live, single_assignment)
from cave_mem_v3.interventions.single_outcomes import _events, _preparation, _session
from cave_mem_v3.interventions.outcomes import _pair
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.replay import require_current_evidence
from cave_mem_v3.target_data.collection import _close, _exclusive_json
from cave_mem_v3.target_data.partitions import check_partition_resume, partition_fields, partition_tasks
from cave_mem_v3.target_data.pre_treatment_state import validate_exposure_rules
from cave_mem_v3.target_data.sampling import RiskHazard


PROTOCOL = "live_select_audited_pair_else_single.v1"


def _pair_item(item, checkpoint):
    """Add the actual selected snapshot to the already frozen live input."""
    value = {key: copy.deepcopy(data) for key, data in item.items() if key != "record_hash"}
    value.update(checkpoint=copy.deepcopy(checkpoint), checkpoint_hash=digest(checkpoint),
                 memory_id=item["pre_treatment_state"]["memory_id"])
    return seal(value)


def _route(run, item, mode, reason, *, pair_item=None, audit=None):
    return seal({"schema_version": "target-auto-route.v1", "run_hash": run["record_hash"],
        "trace_id": item["trace_id"], "input_hash": item["record_hash"],
        "mode": mode, "reason": reason,
        "pair_input_hash": pair_item["record_hash"] if pair_item else None,
        "audit_hash": audit["record_hash"] if audit else None})


def summarize_target_auto(output_dir):
    """Recheck the original live prefix and the selected route without calls."""
    root = Path(output_dir)
    run = checked(read_json(root / "run.json"), schema="target-auto-run.v1")
    memories = {m.memory_id: m for m in bank_memories(run["bank"])} if run["bank"]["memories"] else {}
    tasks = {t["task_id"]: t for t in partition_tasks(run["manifest"], run["role"], run.get("validation_role"))}
    rows = []
    expected_pair_dirs = set()
    for index, task_id in enumerate(run["task_order"]):
        task = tasks[task_id]
        trace_id = digest([run["record_hash"], task_id])[:24]
        seed = stage_seed(run["study"]["base_seed"], task_id, 0, "single-prefix-rollout")
        directory = root / "sessions" / f"{index:06d}-{trace_id}"
        route_path = directory / "route.json"
        if not route_path.exists():
            if directory.exists():
                row = _session(directory, run, task, trace_id, seed, memories)
                if row["status"] != "pending":
                    raise ValueError("auto session closed without a persisted route")
                rows.append(seal({"kind": None, "observation": row, "input_hash": None}))
            else:
                rows.append(seal({"kind": None, "observation": {"task_id": task_id,
                    "trace_id": trace_id, "status": "unattempted"}, "input_hash": None}))
            continue
        route = checked(read_json(route_path), schema="target-auto-route.v1")
        if not (directory / "preparation.json").exists() or not (directory / "input.json").exists():
            raise ValueError("auto route has no completed preparation and input")
        preparation = checked(read_json(directory / "preparation.json"), schema="target-single-preparation.v1")
        _preparation(_events(directory / "preparation_events", run["record_hash"], "preparation"),
            preparation, run, task, trace_id, seed, memories)
        item = checked(read_json(directory / "input.json"), schema="target-single-input.v1")
        if item != _input(run, task, preparation):
            raise ValueError("auto input differs from its recorded live prefix")
        if (route["run_hash"] != run["record_hash"] or route["trace_id"] != trace_id
                or route["input_hash"] != item["record_hash"]):
            raise ValueError("auto route identity mismatch")
        if route["mode"] == "paired":
            if item["status"] != "ready" or route["reason"] != "restoration_audit_passed":
                raise ValueError("ineligible or unexplained paired route")
            pair_dir = root / "pairs" / route["pair_input_hash"]
            expected_pair_dirs.add(pair_dir.name)
            pair_item = checked(read_json(pair_dir / "input.json"))
            audit = checked(read_json(pair_dir / "audit.json"), schema="target-restore-audit.v1")
            if (pair_item["record_hash"] != route["pair_input_hash"]
                    or audit["record_hash"] != route["audit_hash"] or audit["status"] != "passed"
                    or audit["input_hash"] != pair_item["record_hash"]
                    or pair_item["memory_id"] != item["pre_treatment_state"]["memory_id"]
                    or pair_item["checkpoint_hash"] != digest(pair_item["checkpoint"])
                    or {key: pair_item[key] for key in item if key != "record_hash"}
                        != {key: item[key] for key in item if key != "record_hash"}):
                raise ValueError("paired route/input/audit mismatch")
            result = _pair(root, run, pair_item, memories)
            session_close = directory / "closed.json"
            if session_close.exists():
                closed = checked(read_json(session_close), schema="target-auto-paired-session-closed.v1")
                if (closed["run_hash"] != run["record_hash"] or closed["trace_id"] != trace_id
                        or closed["cleanup_errors"]):
                    raise ValueError("paired live prefix cleanup or identity failed")
            else:
                result = {**result, "status": "pending", "reason": "live_prefix_not_closed",
                          "outcome_valid": False, "utility_valid": False, "D": None, "H": None,
                          "rescue": None, "silent_success": None, "remind_success": None}
            rows.append(seal({"kind": "paired", "observation": result,
                              "input_hash": item["record_hash"]}))
        elif route["mode"] in ("single", "none"):
            if route["mode"] == "none" and item["status"] == "ready":
                raise ValueError("eligible input silently dropped by auto route")
            if route["mode"] == "single" and item["status"] != "ready":
                raise ValueError("ineligible input randomized by auto route")
            if route["mode"] == "none" and (route["reason"] != item["reason"]
                    or route["pair_input_hash"] is not None):
                raise ValueError("excluded auto route differs from frozen input")
            if route["pair_input_hash"] is not None:
                pair_dir = root / "pairs" / route["pair_input_hash"]
                expected_pair_dirs.add(pair_dir.name)
                if {path.name for path in pair_dir.iterdir()} != {
                        "input.json", "audit.attempt.json", "audit.json"}:
                    raise ValueError("single fallback has paired treatment artifacts")
                pair_item = checked(read_json(pair_dir / "input.json"))
                audit = checked(read_json(pair_dir / "audit.json"), schema="target-restore-audit.v1")
                if (route["mode"] != "single" or route["reason"] != "restoration_audit_failed"
                        or audit["status"] == "passed" or audit["record_hash"] != route["audit_hash"]
                        or audit["input_hash"] != pair_item["record_hash"]):
                    raise ValueError("single fallback requires a failed recorded audit")
            elif route["audit_hash"] is not None:
                raise ValueError("auto route has audit hash without its input")
            elif route["mode"] == "single" and not (route["reason"] == "restoration_uncertified"
                    or route["reason"].startswith("snapshot_unavailable:")):
                raise ValueError("single route has no supported pre-treatment reason")
            result = _session(directory, run, task, trace_id, seed, memories)
            rows.append(seal({"kind": route["mode"], "observation": result,
                              "input_hash": item["record_hash"]}))
        else:
            raise ValueError("unknown auto collection route")
    pair_root = root / "pairs"
    if pair_root.exists() and {path.name for path in pair_root.iterdir()} != expected_pair_dirs:
        raise ValueError("auto run contains unregistered paired attempts")
    return seal({"schema_version": "target-auto-summary.v1", "run_hash": run["record_hash"],
        "observations": rows, "counts": dict(Counter(row["kind"] for row in rows)),
        "pending": sum(row["observation"]["status"] in ("pending", "unattempted") for row in rows)})


def collect_target_auto(manifest, *, bank, role="target_train", spec: StudySpec,
                        env_factory, actor_factory, output_dir, encoder,
                        assignment_probability, hazard=None, exposure_rules=None,
                        validation_role=None):
    """Audit a selected checkpoint while live, then pair or randomize once."""
    check_partition_resume(output_dir, role, validation_role)
    probability = finite(assignment_probability, "assignment_probability", minimum=0)
    if isinstance(assignment_probability, bool) or not 0 < probability < 1:
        raise ValueError("both single-arm logging probabilities must be strictly positive")
    validate_manifest(manifest, verify_assets=True)
    checked(bank, "bank_hash", "verified-source-bank.v3")
    if bank.get("frozen") is not True:
        raise ValueError("auto collection requires a frozen bank")
    _validate_source_partition({"manifest_hash": manifest["manifest_hash"], "manifest": manifest}, bank)
    memories = {m.memory_id: m for m in bank_memories(bank)} if bank["memories"] else {}
    for memory in memories.values():
        require_current_evidence(memory.evidence, replay_pairs=2)
    if not memories and (bank.get("memory_count") != 0 or bank.get("audit_sources") != {}):
        raise ValueError("invalid empty bank")
    validate_exposure_rules(exposure_rules)
    if not isinstance(getattr(encoder, "identity", None), dict) or not encoder.identity:
        raise ValueError("auto collection requires a frozen encoder")
    tasks = sorted(copy.deepcopy(partition_tasks(manifest, role, validation_role)), key=lambda t: t["task_id"])
    if not tasks:
        raise ValueError("manifest has no tasks for this target partition")
    random.Random(spec.base_seed).shuffle(tasks)
    hazard = hazard or RiskHazard()
    used, configs, capabilities = [], {}, {}
    actor = actor_factory()
    used.append(actor)
    try:
        actor_config, actor_capabilities = actor.configuration(), actor.capabilities()
        for task in tasks:
            env = env_factory(copy.deepcopy(task))
            if any(env is old for old in used):
                raise ValueError("factory reused an environment in auto preflight")
            used.append(env)
            try:
                _check_live_protocol(env, actor)
                configs[task["task_id"]], capabilities[task["task_id"]] = env.configuration(), env.capabilities()
            finally:
                _close(env)
    finally:
        _close(actor)
    run = seal({"schema_version": "target-auto-run.v1", "manifest": manifest,
        "manifest_hash": manifest["manifest_hash"], "bank": bank, "bank_hash": bank["bank_hash"],
        "study": plain(spec), "role": role, **partition_fields(role, validation_role),
        "task_order": [t["task_id"] for t in tasks], "actor_config": actor_config,
        "actor_capabilities": actor_capabilities, "environment_configs": configs,
        "environment_capabilities": capabilities, "risk_hazard": plain(hazard),
        "encoder_identity": encoder.identity, "exposure_rules": exposure_rules,
        "assignment_probability": probability, "protocol": PROTOCOL})
    root = Path(output_dir)
    if (root / "run.json").exists():
        if read_json(root / "run.json") != run:
            raise ValueError("auto run inputs changed; choose a new directory")
        summary = summarize_target_auto(root)
        if summary["pending"] == 0:
            return summary
        # Completed-prefix resume is allowed; any started, unfinished session is not.
        for index, task in enumerate(tasks):
            trace_id = digest([run["record_hash"], task["task_id"]])[:24]
            directory = root / "sessions" / f"{index:06d}-{trace_id}"
            if (directory / "attempt.json").exists() and not (directory / "closed.json").exists() \
                    and not (directory / "route.json").exists():
                raise ValueError("incomplete live session; preserve and reconcile before retry")
            if (directory / "route.json").exists():
                route = read_json(directory / "route.json")
                if route["mode"] == "paired":
                    pair_dir = root / "pairs" / route["pair_input_hash"]
                    if not (pair_dir / "closed.json").exists():
                        raise ValueError("incomplete paired attempt; preserve and reconcile before retry")
                elif not (directory / "closed.json").exists():
                    raise ValueError("incomplete live single session; preserve and reconcile before retry")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("new auto output directory must be empty")
        _exclusive_json(root / "run.json", run)
    reference = {"model": None, "fingerprint": None}
    for index, task in enumerate(tasks):
        trace_id = digest([run["record_hash"], task["task_id"]])[:24]
        seed = stage_seed(spec.base_seed, task["task_id"], 0, "single-prefix-rollout")
        directory = root / "sessions" / f"{index:06d}-{trace_id}"
        if (directory / "route.json").exists():
            continue
        if (directory / "attempt.json").exists():
            raise ValueError("incomplete auto session; preserve and reconcile before retry")
        _exclusive_json(directory / "attempt.json", seal({"schema_version": "target-single-attempt.v1",
            "run_hash": run["record_hash"], "trace_id": trace_id, "task_id": task["task_id"],
            "seed": seed, "status": "started"}))
        env = actor = None
        item = assignment = result = pending = None
        cleanup_errors = []
        route_mode = None
        try:
            actor = actor_factory()
            if any(actor is old for old in used):
                raise ValueError("factory reused an actor in auto collection")
            used.append(actor)
            env = env_factory(copy.deepcopy(task))
            if any(env is old for old in used):
                raise ValueError("factory reused an environment in auto collection")
            used.append(env)
            preparation = _prepare_live(run, task, trace_id, seed, env, actor, encoder,
                _journal(directory / "preparation_events", run["record_hash"], "preparation"), reference)
            _exclusive_json(directory / "preparation.json", preparation)
            item = _input(run, task, preparation)
            _exclusive_json(directory / "input.json", item)
            if item["status"] != "ready":
                route_mode = "none"
                _exclusive_json(directory / "route.json", _route(run, item, "none", item["reason"]))
            else:
                pair_item = audit = None
                caps = env.capabilities()
                route_reason = "restoration_uncertified"
                if caps.get("restore_certified") is True and caps.get("isolated_sessions") is True:
                    try:
                        checkpoint = env.snapshot()
                        if plain(env.observe()) != item["context"]:
                            raise ValueError("live selected context changed before route")
                        original_fingerprint = env.state_fingerprint()
                        pair_item = _pair_item(item, checkpoint)
                        pair_dir = root / "pairs" / pair_item["record_hash"]
                        _exclusive_json(pair_dir / "input.json", pair_item)
                        _exclusive_json(pair_dir / "audit.attempt.json", seal({"input_hash": pair_item["record_hash"], "status": "started"}))
                        audit = _audit(pair_item, env_factory, actor_factory, used)
                        _exclusive_json(pair_dir / "audit.json", audit)
                        if (digest(env.snapshot()) != pair_item["checkpoint_hash"]
                                or env.state_fingerprint() != original_fingerprint):
                            raise ValueError("restoration audit changed the original live session")
                        route_reason = "restoration_audit_failed"
                    except Exception as exc:
                        if pair_item is not None:
                            raise
                        route_reason = "snapshot_unavailable:" + type(exc).__name__
                if plain(env.observe()) != item["context"] or actor.configuration() != run["actor_config"]:
                    raise ValueError("live selected session changed during restoration audit")
                if audit is not None and audit["status"] == "passed":
                    route_mode = "paired"
                    _exclusive_json(directory / "route.json", _route(run, item, "paired",
                        "restoration_audit_passed", pair_item=pair_item, audit=audit))
                    if not execute_audited_pair(run, pair_item, audit, memories[item["pre_treatment_state"]["memory_id"]],
                            directory=pair_dir, env_factory=env_factory, actor_factory=actor_factory, used=used):
                        pending = "paired_external_request"
                else:
                    route_mode = "single"
                    _exclusive_json(directory / "route.json", _route(run, item, "single", route_reason,
                        pair_item=pair_item, audit=audit))
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
            if isinstance(pending, dict):
                _exclusive_json(directory / "pending.json", seal({**pending, "cleanup_errors": cleanup_errors}))
            break
        if result is not None:
            _exclusive_json(directory / "result.json", seal({"schema_version": "target-single-result.v1",
                "assignment_hash": assignment["record_hash"], "record": result,
                "cleanup_errors": cleanup_errors}))
        if route_mode != "paired":
            _exclusive_json(directory / "closed.json", seal({"schema_version": "target-single-closed.v1",
                "run_hash": run["record_hash"], "trace_id": trace_id, "input_hash": item["record_hash"],
                "assignment_hash": assignment["record_hash"] if assignment else None,
                "status": "blocked" if cleanup_errors else "completed" if result is not None else item["status"],
                "reason": "cleanup_error" if cleanup_errors else item["reason"],
                "cleanup_errors": cleanup_errors}))
        else:
            _exclusive_json(directory / "closed.json", seal({"schema_version": "target-auto-paired-session-closed.v1",
                "run_hash": run["record_hash"], "trace_id": trace_id, "cleanup_errors": cleanup_errors}))
        if cleanup_errors or (preparation.get("error") or {}).get("scope") == "run_shared" \
                or result is not None and (result.get("error") or {}).get("scope") in ("pair_shared", "session_shared"):
            break
        for observed in (preparation.get("response_identity", {}), (result or {}).get("response_identity", {})):
            for key in reference:
                if observed.get(key) is not None:
                    reference[key] = observed[key]
    summary = summarize_target_auto(root)
    write_json(root / "summary.json", summary)
    return summary
