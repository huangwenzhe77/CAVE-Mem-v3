"""Bounded historical rollout collection with immutable traces and resume checks."""
from __future__ import annotations

import copy
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cave_mem_v3.common.contracts import StudySpec, digest, plain, read_json, stage_seed, write_json
from cave_mem_v3.benchmarks.tasks import validate_manifest


class _IdentityError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionConfig:
    """Both modes enforce per-task attempts and a global attempted-run cap."""
    mode: str = "fixed_runs"
    runs_per_task: int = 1
    max_runs: int = 100
    failure_target: int | None = None

    def __post_init__(self):
        if self.mode not in ("fixed_runs", "failure_target"):
            raise ValueError("mode must be fixed_runs or failure_target")
        for name in ("runs_per_task", "max_runs"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mode == "fixed_runs" and self.failure_target is not None:
            raise ValueError("fixed_runs cannot stop based on observed failures")
        if self.mode == "failure_target":
            if type(self.failure_target) is not int or not 1 <= self.failure_target <= self.max_runs:
                raise ValueError("failure_target must be between 1 and max_runs")


def _exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _close(obj):
    close = getattr(obj, "close", None)
    if close:
        close()


def _identities(tasks, env_factory, actor_factory):
    actor = actor_factory()
    try:
        actor_config = copy.deepcopy(actor.configuration())
        actor_capabilities = copy.deepcopy(actor.capabilities())
    finally:
        _close(actor)
    environments, capabilities = {}, {}
    for task in tasks:
        env = env_factory(copy.deepcopy(task))
        try:
            environments[task["task_id"]] = copy.deepcopy(env.configuration())
            capabilities[task["task_id"]] = copy.deepcopy(env.capabilities())
        finally:
            _close(env)
    return actor_config, actor_capabilities, environments, capabilities


def _episode_check(episode: dict, actor_config: dict, environment_config: dict, task_id: str) -> None:
    if not isinstance(episode, dict) or episode.get("status") not in ("terminal", "budget_exhausted", "invalid_action", "error"):
        raise ValueError("episode requires an explicit terminal/budget/invalid/error status")
    if episode["status"] == "error":
        if episode.get("success") is not None or episode.get("utility") is not None:
            raise ValueError("infrastructure errors cannot become success/failure outcomes")
    elif type(episode.get("success")) is not bool:
        raise ValueError("completed episode needs boolean success")
    if episode["status"] in ("budget_exhausted", "invalid_action") and episode["success"]:
        raise ValueError("budget exhaustion or invalid actions cannot be scored as success")
    if episode.get("actor_config") != actor_config or episode.get("environment_config") != environment_config:
        raise ValueError("episode actor/environment configuration changed")
    if episode.get("task_id") is not None and episode["task_id"] != task_id:
        raise ValueError("episode belongs to a different task")
    if episode["status"] != "error" and episode.get("task_id") != task_id:
        raise ValueError("completed episode lacks the scheduled task identity")
    if not isinstance(episode.get("costs"), dict) or not isinstance(episode.get("steps"), list):
        raise ValueError("episode must retain costs and steps")
    if "trace_hash" in episode and episode["trace_hash"] != digest({key: value for key, value in episode.items() if key != "trace_hash"}):
        raise ValueError("episode checksum mismatch")


def _response_identity(episode: dict) -> dict:
    """Legacy injected fixture episodes without identity remain unknown."""
    identity = episode.get("response_identity")
    if identity is None:
        return {"model": None, "fingerprint": None, "fingerprint_status": "unknown"}
    if not isinstance(identity, dict) or identity.get("fingerprint_status") not in ("reported", "unknown", "mismatch"):
        raise ValueError("invalid episode response identity")
    for field in ("model", "fingerprint"):
        if identity.get(field) is not None and (not isinstance(identity[field], str) or not identity[field]):
            raise ValueError("reported response model/fingerprint must be nonempty strings or unknown")
    return identity


def _identity_conflicts(episode: dict, reference: dict) -> list[str]:
    if episode["status"] == "error":
        return []
    identity = _response_identity(episode)
    conflicts = [field for field in ("model", "fingerprint")
                 if identity.get(field) is not None and reference.get(field) is not None
                 and identity[field] != reference[field]]
    if identity["fingerprint_status"] == "mismatch":
        conflicts.append("within_episode_identity")
    return conflicts


def _remember_identity(episode: dict, reference: dict) -> None:
    if episode["status"] != "error":
        identity = _response_identity(episode)
        for field in ("model", "fingerprint"):
            if identity.get(field) is not None:
                reference[field] = identity[field]


def _record_identity_error(episode: dict, reference: dict, conflicts: list[str]) -> dict:
    """Preserve the observed result, execution trace and cost of a drifting run."""
    episode = copy.deepcopy(episode)
    episode["observed_outcome_before_identity_check"] = {
        "status": episode["status"], "success": episode["success"], "utility": episode.get("utility"),
        "error": copy.deepcopy(episode.get("error")),
        "response_identity": copy.deepcopy(episode.get("response_identity")),
        "trace_hash": episode.get("trace_hash")}
    episode.update(status="error", success=None, utility=None,
                   error={"type": "ResponseIdentityError", "scope": "cross_episode_identity",
                          "conflicting_fields": conflicts, "expected": copy.deepcopy(reference),
                          "observed": copy.deepcopy(_response_identity(episode))})
    episode.pop("trace_hash", None)
    episode["trace_hash"] = digest(episode)
    return episode


def _summary(run, traces, config, task_ids, planned_runs, nominal_runs):
    successes = sum(trace["episode"]["success"] is True for trace in traces)
    failures = sum(trace["episode"]["success"] is False and trace["episode"]["status"] != "error" for trace in traces)
    errors = sum(trace["episode"]["status"] == "error" for trace in traces)
    identity_errors = sum((trace["episode"].get("error") or {}).get("scope") == "cross_episode_identity" for trace in traces)
    if config.mode == "failure_target" and failures >= config.failure_target:
        reason = "failure_target_reached"
    elif len(traces) >= planned_runs:
        if planned_runs == nominal_runs:
            reason = "fixed_schedule_complete" if config.mode == "fixed_runs" else "per_task_limit_reached"
        else:
            reason = "max_runs_reached"
    else:
        reason = "incomplete"
    limits_reached = []
    if len(traces) >= nominal_runs:
        limits_reached.append("runs_per_task")
    if len(traces) >= config.max_runs:
        limits_reached.append("max_runs")
    attempts = Counter(trace["task_id"] for trace in traces)
    measured = {trace["task_id"] for trace in traces if trace["episode"]["status"] != "error"}
    failure_tasks = {trace["task_id"] for trace in traces if trace["episode"]["success"] is False and trace["episode"]["status"] != "error"}
    result = {"schema_version": "history-summary.v3", "run_hash": run["run_hash"],
              "manifest_hash": run["manifest_hash"], "mode": config.mode,
              "stop_reason": reason, "limits_reached": limits_reached,
              "nominal_runs": nominal_runs, "planned_runs": planned_runs,
              "attempted_runs": len(traces), "completed_runs": successes + failures,
              "successful_runs": successes, "failed_runs": failures, "infrastructure_errors": errors,
              "response_identity_errors": identity_errors,
              "failure_rate_observed": failures / (successes + failures) if successes + failures else None,
              "history_task_count": len(task_ids), "tasks_attempted": len(attempts),
              "tasks_with_valid_outcome": len(measured), "tasks_with_failure": len(failure_tasks),
              "attempt_coverage": len(attempts) / len(task_ids),
              "valid_outcome_coverage": len(measured) / len(task_ids),
              "per_task_attempts": {task_id: attempts[task_id] for task_id in task_ids},
              "trace_ids": [trace["trace_id"] for trace in traces],
              "selection_note": ("Outcome-dependent stopping; exploratory corpus, not an unbiased failure-rate cohort."
                                 if config.mode == "failure_target" else "Fixed outcome-blind task/replicate schedule; infrastructure failures retained."),
              "sampling_unit": "task; repeat runs are not independent tasks"}
    result["summary_hash"] = digest(result)
    return result


def collect_history(manifest: dict, config: CollectionConfig, spec: StudySpec,
                    env_factory: Callable, actor_factory: Callable,
                    output_dir: str | Path, *, episode_runner: Callable | None = None) -> dict:
    """Collect history only, counting all attempted episodes against hard limits.

    Completed traces are immutable and checked on resume. A started attempt
    without a complete trace is not rerun automatically: its external cost and
    outcome are unknown and require inspection before starting a new run.
    """
    validate_manifest(manifest, verify_assets=True)
    tasks = [copy.deepcopy(task) for task in manifest["tasks"] if task["role"] == "history"]
    if not tasks:
        raise ValueError("phase one requires history tasks; target/validation/test cannot be collected")
    tasks.sort(key=lambda task: task["task_id"])
    random.Random(spec.base_seed).shuffle(tasks)
    actor_config, actor_capabilities, env_configs, env_capabilities = _identities(tasks, env_factory, actor_factory)
    run = {"schema_version": "history-run.v3", "manifest_hash": manifest["manifest_hash"],
           "schedule_policy": "round_robin_bounded_per_task.v1",
           "manifest": copy.deepcopy(manifest), "collection": plain(config), "study": plain(spec),
           "actor_config": actor_config, "actor_capabilities": actor_capabilities,
           "environment_configs": env_configs, "environment_capabilities": env_capabilities,
           "task_order": [task["task_id"] for task in tasks]}
    run["run_hash"] = digest(run)
    root = Path(output_dir)
    run_file = root / "run.json"
    if run_file.exists():
        existing = read_json(run_file)
        if existing != run:
            raise ValueError("run identity/configuration/manifest changed or run.json is corrupt; use a new output directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("new collection output directory must be empty")
        _exclusive_json(run_file, run)
    nominal_runs = len(tasks) * config.runs_per_task
    planned_runs = min(nominal_runs, config.max_runs)
    planned = []
    for index in range(planned_runs):
        task = tasks[index % len(tasks)]
        replicate = index // len(tasks)
        trace_id = digest([run["run_hash"], task["task_id"], replicate])[:24]
        planned.append((task, replicate, trace_id, stage_seed(spec.base_seed, task["task_id"], replicate, "history")))
    if episode_runner is None:
        from cave_mem_v3.agents.runtime import run_episode
        episode_runner = run_episode
    trace_dir = root / "traces"
    expected_names = {f"{i:06d}-{entry[2]}.json" for i, entry in enumerate(planned)}
    if trace_dir.exists() and any(p.name not in expected_names for p in trace_dir.iterdir()):
        raise ValueError("collection contains an unregistered trace file")
    traces, missing_seen = [], False
    response_reference = {"model": None, "fingerprint": None}
    for index, (task, replicate, trace_id, seed) in enumerate(planned):
        path = trace_dir / f"{index:06d}-{trace_id}.json"
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError("collection trace sequence is not a contiguous prefix")
        trace = read_json(path)
        expected = {"schema_version": "history-trace.v3", "trace_id": trace_id,
                    "task_id": task["task_id"], "role": "history", "group_id": task.get("group_id"),
                    "manifest_hash": manifest["manifest_hash"], "run_hash": run["run_hash"],
                    "run_index": index, "replicate": replicate, "seed": seed}
        if any(trace.get(key) != value for key, value in expected.items()):
            raise ValueError("trace identity differs from the fixed collection schedule")
        if trace.get("trace_hash") != digest({key: value for key, value in trace.items() if key != "trace_hash"}):
            raise ValueError("trace checksum mismatch")
        _episode_check(trace.get("episode"), actor_config, env_configs[task["task_id"]], task["task_id"])
        if _identity_conflicts(trace["episode"], response_reference):
            raise ValueError("stored completed episodes have inconsistent response identity; existing traces were not modified")
        _remember_identity(trace["episode"], response_reference)
        if config.mode == "failure_target" and sum(t["episode"]["success"] is False for t in traces) >= config.failure_target:
            raise ValueError("trace sequence continues past its registered stopping rule")
        traces.append(trace)
    for index in range(len(traces), planned_runs):
        if config.mode == "failure_target" and sum(t["episode"]["success"] is False for t in traces) >= config.failure_target:
            break
        task, replicate, trace_id, seed = planned[index]
        started_file = root / "attempts" / f"{index:06d}.json"
        if started_file.exists():
            raise ValueError("incomplete attempted episode: inspect its external outcome/cost; automatic replay is disabled")
        _exclusive_json(started_file, {"run_hash": run["run_hash"], "trace_id": trace_id,
                                       "task_id": task["task_id"], "seed": seed, "status": "started"})
        actor = env = None
        try:
            actor = actor_factory()
            env = env_factory(copy.deepcopy(task))
            if actor.configuration() != actor_config or env.configuration() != env_configs[task["task_id"]]:
                raise _IdentityError("factory configuration changed during collection")
            episode = episode_runner(env, actor, spec=spec, seed=seed, capture_checkpoints=True)
        except _IdentityError:
            raise
        except Exception as exc:
            # Provider failure details belong in runtime's metered attempt log.
            episode = {"status": "error", "success": None, "utility": None, "costs": {}, "steps": [],
                       "error": {"type": type(exc).__name__, "scope": "episode_factory_or_runner"},
                       "actor_config": actor_config, "environment_config": env_configs[task["task_id"]]}
        finally:
            cleanup_errors = []
            for item in (env, actor):
                if item is not None:
                    try:
                        _close(item)
                    except Exception as exc:
                        cleanup_errors.append(type(exc).__name__)
        _episode_check(episode, actor_config, env_configs[task["task_id"]], task["task_id"])
        identity_conflicts = _identity_conflicts(episode, response_reference)
        if identity_conflicts:
            episode = _record_identity_error(episode, response_reference, identity_conflicts)
        _remember_identity(episode, response_reference)
        trace = {"schema_version": "history-trace.v3", "trace_id": trace_id,
                 "task_id": task["task_id"], "role": "history", "group_id": task.get("group_id"),
                 "manifest_hash": manifest["manifest_hash"], "run_hash": run["run_hash"],
                 "run_index": index, "replicate": replicate, "seed": seed, "episode": episode}
        if cleanup_errors:
            trace["cleanup_errors"] = cleanup_errors
        trace["trace_hash"] = digest(trace)
        _exclusive_json(trace_dir / f"{index:06d}-{trace_id}.json", trace)
        traces.append(trace)
        write_json(root / "summary.json", _summary(run, traces, config, run["task_order"], planned_runs, nominal_runs))
    report = _summary(run, traces, config, run["task_order"], planned_runs, nominal_runs)
    write_json(root / "summary.json", report)
    return report
