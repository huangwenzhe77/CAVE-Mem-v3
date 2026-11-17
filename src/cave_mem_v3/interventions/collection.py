"""Durable paired target collection. Every external attempt is write-ahead logged."""
from __future__ import annotations
from cave_mem_v3.target_data.partitions import validation_binding

import copy
import random
import time
from pathlib import Path

from cave_mem_v3.common.contracts import StudySpec, digest, plain, read_json, stage_seed, write_json
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.target_data.collection import _close, _exclusive_json
from cave_mem_v3.target_data.pre_treatment_state import _context
from cave_mem_v3.interventions.execution import TargetPendingError, run_target_arm
from cave_mem_v3.interventions.inputs import checked, load_prepared_run, seal
from cave_mem_v3.interventions.reminder import render_reminder


def pair_plan(run, item, audit, reminder):
    """Randomize execution order only; both potential outcomes are attempted."""
    seed = stage_seed(run["study"]["base_seed"], item["record_hash"], 0, "target-pair")
    order_seed = stage_seed(seed, item["record_hash"], 0, "arm-order")
    draw = random.Random(order_seed).random()
    return seal({"schema_version": "target-pair-plan.v1", "run_hash": run["record_hash"],
        "input_hash": item["record_hash"], "task_id": item["task_id"],
        "a0_hash": item["a0_hash"], "x_hash": item["x_hash"],
        "checkpoint_hash": item["checkpoint_hash"], "bank_hash": run["bank_hash"],
        "audit_hash": audit["record_hash"], "reminder": reminder,
        "seed": seed, "order_seed": order_seed, "order_draw": draw,
        "order_probability": 0.5,
        "order": ["silent", "remind"] if draw < .5 else ["remind", "silent"],
        "remaining_model_calls": item["remaining_model_calls"],
        "remaining_steps": item["remaining_steps"], "study": run["study"],
        "policy": "paired_restore_one_reminder_equal_continuation_budget.v1"})


def _check_instance(item, env, actor):
    if env.configuration() != item["environment_config"] or actor.configuration() != item["actor_config"]:
        raise ValueError("frozen actor/environment configuration mismatch")
    caps = env.capabilities()
    if not actor.capabilities().get("stateless") or not caps.get("restore_certified") or not caps.get("isolated_sessions"):
        raise ValueError("uncertified restoration or stateful actor")


def _audit(item, env_factory, actor_factory, used):
    started = time.perf_counter()
    instances, environments, checks, errors = [], [], [], []
    report = {"schema_version": "target-restore-audit.v1", "input_hash": item["record_hash"],
              "checkpoint_hash": item["checkpoint_hash"], "status": "blocked"}
    try:
        actor = actor_factory()
        instances.append(actor)
        if any(actor is previous for previous in used):
            raise ValueError("factory reused an actor instance")
        used.append(actor)
        for _ in range(2):
            env = env_factory(copy.deepcopy(item["task"]))
            instances.append(env)
            if any(env is previous for previous in used):
                raise ValueError("factory reused an environment instance")
            used.append(env)
            environments.append(env)
            _check_instance(item, env, actor)
            restored = env.restore(copy.deepcopy(item["checkpoint"]))
            if digest(env.snapshot()) != item["checkpoint_hash"] or plain(restored) != item["context"] or env.terminal_status()[0]:
                raise ValueError("actual selected checkpoint failed restoration")
            checks.append({"snapshot_hash": digest(env.snapshot()), "fingerprint": env.state_fingerprint()})
        first, second = environments
        if checks[0] != checks[1]:
            raise ValueError("independent restore fingerprints differ")
        adapter = item["environment_config"].get("adapter", "")
        if adapter == "preparation-fixture.v3":
            first.step({"type": "prepare"}, seed=0)
            if digest(first.snapshot()) == item["checkpoint_hash"] or digest(second.snapshot()) != item["checkpoint_hash"]:
                raise ValueError("fixture mutation probe did not isolate branches")
            first.restore(copy.deepcopy(item["checkpoint"]))
            if digest(first.snapshot()) != item["checkpoint_hash"]:
                raise ValueError("fixture mutation probe did not restore")
            report["probe"] = {"kind": "fixture_mutation", "branches_isolated": True}
        elif adapter == "alfworld-text-prefix-replay.v1":
            from cave_mem_v3.environments.alfworld import ALFWorldEnvironment
            if not isinstance(first, ALFWorldEnvironment) or not first.capabilities().get("paired_primary_eligible"):
                raise ValueError("ALFWorld adapter lacks reviewed paired restoration")
            initial_second = digest(second.snapshot())
            actions = first.observe().admissible_actions
            if not actions:
                raise ValueError("ALFWorld checkpoint has no public action for isolation probe")
            probe_action = {"type": "text", "command": actions[0]}
            result = first.step(probe_action, seed=0)
            if (not result.valid or digest(first.snapshot()) == item["checkpoint_hash"]
                    or digest(second.snapshot()) != initial_second):
                raise ValueError("ALFWorld mutation probe did not isolate branches")
            first.restore(copy.deepcopy(item["checkpoint"]))
            if (digest(first.snapshot()) != item["checkpoint_hash"]
                    or digest(second.snapshot()) != item["checkpoint_hash"]):
                raise ValueError("ALFWorld mutation probe did not restore the checkpoint")
            report["probe"] = {"kind": "alfworld_action_prefix_mutation",
                               "command_hash": digest(probe_action),
                               "branches_isolated": True, "restored": True}
        else:
            from cave_mem_v3.environments.terminal import TerminalEnvironment
            if not isinstance(first, TerminalEnvironment) or not first.capabilities().get("paired_primary_eligible"):
                raise ValueError("adapter outside reviewed Terminal filesystem protocol; adapter review required")
            diagnostic = first.diagnose_restoration()
            probe = diagnostic.get("filesystem_probe", {})
            if not all(probe.get(k) is True for k in ("persisted_after_restore", "branches_isolated", "base_branch_isolated")):
                raise ValueError("actual checkpoint filesystem probe failed")
            if digest(first.snapshot()) != item["checkpoint_hash"] or digest(second.snapshot()) != item["checkpoint_hash"]:
                raise ValueError("diagnostic mutated a base branch")
            report["probe"] = diagnostic  # Diagnostics never supply task-scope approval.
        for env in environments:
            _check_instance(item, env, actor)
        report.update(status="passed", restores=checks)
    except Exception as exc:
        report.update(error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        for instance in reversed(instances):
            try:
                _close(instance)
            except Exception as exc:
                errors.append(type(exc).__name__)
        if errors:
            report.update(status="blocked", cleanup_errors=errors)
        report["operational_wall_seconds"] = time.perf_counter() - started
    return seal(report)


def _journal(directory, plan, arm):
    previous = None
    count = 0
    def emit(event):
        nonlocal previous, count
        if event.get("event_index") != count or event.get("arm") != arm:
            raise ValueError("noncontiguous target journal")
        value = seal({"schema_version": "target-arm-event.v1", "plan_hash": plan["record_hash"],
                      "previous_hash": previous, "event": event})
        _exclusive_json(directory / f"{count:06d}.json", value)
        previous, count = value["record_hash"], count + 1
    return emit


def execute_audited_pair(run, item, audit, memory, *, directory, env_factory,
                         actor_factory, used):
    """Execute both arms only after a persisted, passing restoration audit.

    Return False when an external request is pending; the caller must stop the
    whole collection without starting another arm or retrying that request.
    """
    if audit["status"] != "passed":
        raise ValueError("paired execution requires a passed restoration audit")
    reminder = render_reminder(memory, item["a0"])
    plan = pair_plan(run, item, audit, reminder)
    _exclusive_json(directory / "plan.json", plan)
    reference = copy.deepcopy(item["response_identity"])
    pair_status = "completed"
    for arm in plan["order"]:
        arm_dir = directory / arm
        _exclusive_json(arm_dir / "attempt.json", seal({"plan_hash": plan["record_hash"], "arm": arm, "status": "started"}))
        actor = env = None
        result = pending = None
        cleanup_errors = []
        try:
            actor = actor_factory()
            if any(actor is old for old in used):
                raise ValueError("factory reused an actor across arms")
            used.append(actor)
            env = env_factory(copy.deepcopy(item["task"]))
            if any(env is old for old in used):
                raise ValueError("factory reused an environment across arms")
            used.append(env)
            _check_instance(item, env, actor)
        except Exception as exc:
            result = {"status": "setup_error", "error": {"type": type(exc).__name__, "message": str(exc)}}
        try:
            if result is None:
                result = run_target_arm(env, actor, context=_context(item["context"]),
                    checkpoint=item["checkpoint"], first_action=item["a0"], arm=arm,
                    reminder=reminder["text"] if arm == "remind" else None,
                    spec=StudySpec(**run["study"]), seed=plan["seed"],
                    remaining_model_calls=item["remaining_model_calls"], remaining_steps=item["remaining_steps"],
                    expected_response_identity=reference, journal=_journal(arm_dir / "events", plan, arm))
        except TargetPendingError as exc:
            pending = seal({"plan_hash": plan["record_hash"], "arm": arm, "status": "pending",
                            "partial_record": exc.partial_record, "attempt": exc.attempt})
        finally:
            for instance in (env, actor):
                if instance is not None:
                    try:
                        _close(instance)
                    except Exception as exc:
                        cleanup_errors.append(type(exc).__name__)
        if pending is not None:
            _exclusive_json(arm_dir / "pending.json", seal({**pending, "cleanup_errors": cleanup_errors}))
            return False
        _exclusive_json(arm_dir / "result.json", seal({"schema_version": "target-arm-result.v1",
            "plan_hash": plan["record_hash"], "arm": arm, "record": result, "cleanup_errors": cleanup_errors}))
        if (cleanup_errors or result["status"] == "setup_error"
                or (result.get("error") or {}).get("scope") == "pair_shared"):
            pair_status = "blocked"
            break
        identity = result.get("response_identity", {})
        for key in ("model", "fingerprint"):
            if identity.get(key) is not None:
                reference[key] = identity[key]
    _exclusive_json(directory / "closed.json", seal({"input_hash": item["record_hash"],
        "plan_hash": plan["record_hash"], "status": pair_status}))
    return True


def collect_target_pairs(target_dir, bank, *, env_factory, actor_factory, output_dir):
    """Run a frozen paired pilot; completed records resume without model calls.

    Any write-ahead attempt without a result is an interruption. Re-entry fails
    closed, preserving evidence for manual reconciliation rather than retrying.
    """
    from cave_mem_v3.interventions.outcomes import summarize_target_pairs
    prepared, inputs = load_prepared_run(target_dir, bank)
    run = seal({"schema_version": "target-pair-run.v1", "source_run_hash": prepared["run_hash"],
        "bank_hash": bank["bank_hash"], "study": prepared["study"],
        "inputs": inputs, "prepared_run": prepared, "bank": bank,
        **validation_binding(prepared),
        "policy": "target_train_paired_only.v1" if prepared["role"] == "target_train" else "validation_paired_only.v1"})
    root = Path(output_dir)
    run_path = root / "run.json"
    if run_path.exists():
        if read_json(run_path) != run:
            raise ValueError("paired run inputs/configuration changed; choose a new directory")
        # Validate already persisted evidence before making any new external call.
        summarize_target_pairs(root)
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("new paired output directory must be empty")
        _exclusive_json(run_path, run)
    memories = {m.memory_id: m for m in bank_memories(bank)} if bank["memories"] else {}
    for item in inputs:
        if item["status"] != "ready":
            continue
        directory = root / "pairs" / item["record_hash"]
        if (directory / "closed.json").exists():
            continue  # The pure summarizer validates every completed artifact below.
        if (directory / "audit.attempt.json").exists():
            raise ValueError("incomplete paired attempt exists; preserve and reconcile it before any retry")
        _exclusive_json(directory / "audit.attempt.json", seal({"input_hash": item["record_hash"], "status": "started"}))
        used = []  # Keep live references so factories cannot reuse an instance across arms.
        audit = _audit(item, env_factory, actor_factory, used)
        _exclusive_json(directory / "audit.json", audit)
        if audit["status"] != "passed":
            _exclusive_json(directory / "closed.json", seal({"input_hash": item["record_hash"],
                "status": "blocked", "reason": "restoration_audit_failed", "audit_hash": audit["record_hash"]}))
            continue
        if not execute_audited_pair(run, item, audit, memories[item["memory_id"]],
                directory=directory, env_factory=env_factory, actor_factory=actor_factory, used=used):
            summary = summarize_target_pairs(root)
            write_json(root / "summary.json", summary)
            return summary
    summary = summarize_target_pairs(root)
    write_json(root / "summary.json", summary)
    return summary
