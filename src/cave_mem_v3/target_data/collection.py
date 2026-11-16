"""Outcome-blind target rollouts with one risk-biased checkpoint per task."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Callable
from cave_mem_v3.target_data.partitions import target_partition, partition_fields, partition_tasks, check_partition_resume, validate_validation_bank

from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import StudySpec, digest, plain, read_json, stage_seed, write_json
from cave_mem_v3.memory_generation.collection import (
    _identity_conflicts, _record_identity_error, _remember_identity,
)
from cave_mem_v3.target_data.pre_treatment_state import validate_exposure_rules
from cave_mem_v3.target_data.sampling import OnlineCheckpointSampler, RiskHazard


def _exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _close(item) -> None:
    close = getattr(item, "close", None)
    if close:
        close()


def collect_target(manifest: dict, *, role: str, spec: StudySpec,
                   env_factory: Callable, actor_factory: Callable,
                   output_dir: str | Path, hazard: RiskHazard | None = None,
                   episode_runner: Callable | None = None,
                   prepare_before_action: bool = False, bank: dict | None = None,
                   encoder=None, exposure_rules: dict | None = None, validation_role: str | None = None,
                   config_identity: str | None = None) -> dict:
    """Run one fixed rollout per registered target task and retain each sample.

    A fresh sampler is created for each task rollout. In preparation mode, the
    selected action is cached and retrieval/state features are frozen before
    that action is validated or executed. The episode then stops without a
    target outcome. The default collection mode retains the original behavior.
    """
    check_partition_resume(output_dir, role, validation_role)
    if config_identity is not None and (not isinstance(config_identity, str) or not config_identity):
        raise ValueError("target collection config_identity must be a nonempty immutable identity")
    if type(prepare_before_action) is not bool:
        raise ValueError("prepare_before_action must be boolean")
    if prepare_before_action:
        if not isinstance(bank, dict) or not isinstance(bank.get("bank_hash"), str):
            raise ValueError("pre-treatment preparation requires a loaded frozen memory bank")
        if not isinstance(getattr(encoder, "identity", None), dict) or not encoder.identity:
            raise ValueError("pre-treatment preparation requires a frozen embedding encoder")
        validate_exposure_rules(exposure_rules)
    validate_manifest(manifest, verify_assets=True)
    tasks = copy.deepcopy(partition_tasks(manifest, role, validation_role))
    if role == "validation" and prepare_before_action:
        validate_validation_bank(manifest, bank)
    if not tasks:
        raise ValueError(f"manifest has no {role} tasks")
    tasks.sort(key=lambda task: task["task_id"])
    random.Random(spec.base_seed).shuffle(tasks)
    if hazard is None:
        hazard = RiskHazard()

    actor = actor_factory()
    try:
        actor_config = copy.deepcopy(actor.configuration())
        actor_capabilities = copy.deepcopy(actor.capabilities())
    finally:
        _close(actor)
    env_configs, env_capabilities = {}, {}
    for task in tasks:
        env = env_factory(copy.deepcopy(task))
        try:
            env_configs[task["task_id"]] = copy.deepcopy(env.configuration())
            env_capabilities[task["task_id"]] = copy.deepcopy(env.capabilities())
        finally:
            _close(env)

    run = {"schema_version": "target-run.v1", "manifest_hash": manifest["manifest_hash"],
           "manifest": copy.deepcopy(manifest), "role": role, **partition_fields(role, validation_role),
           "schedule_policy": "one_rollout_per_task.v1", "rollouts_per_task": 1,
           "study": plain(spec), "risk_hazard": plain(hazard),
           "actor_config": actor_config, "actor_capabilities": actor_capabilities,
           "environment_configs": env_configs, "environment_capabilities": env_capabilities,
           "task_order": [task["task_id"] for task in tasks]}
    if config_identity is not None:
        run["config_identity"] = config_identity
    if prepare_before_action:
        run["pre_treatment"] = {
            "mode": "freeze_before_selected_action_execution.v1",
            "bank_hash": bank["bank_hash"],
            "encoder_identity": copy.deepcopy(encoder.identity),
            "exposure_rules_hash": digest(exposure_rules) if exposure_rules is not None else None,
            "exposure_rules": copy.deepcopy(exposure_rules),
        }
    run["run_hash"] = digest(run)
    root = Path(output_dir)
    run_file = root / "run.json"
    if run_file.exists():
        if read_json(run_file) != run:
            raise ValueError("target run identity/configuration differs; choose a new output directory")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("new target output directory must be empty")
        _exclusive_json(run_file, run)

    if episode_runner is None:
        from cave_mem_v3.agents.runtime import run_episode
        episode_runner = run_episode
    trace_dir = root / "traces"
    expected_names = {f"{index:06d}-{digest([run['run_hash'], task['task_id']])[:24]}.json"
                      for index, task in enumerate(tasks)}
    if trace_dir.exists() and any(path.name not in expected_names for path in trace_dir.iterdir()):
        raise ValueError("target run contains an unregistered trace")

    traces = []
    missing_seen = False
    response_reference = {"model": None, "fingerprint": None}
    for index, task in enumerate(tasks):
        trace_id = digest([run["run_hash"], task["task_id"]])[:24]
        seed = stage_seed(spec.base_seed, task["task_id"], 0, "target_rollout")
        path = trace_dir / f"{index:06d}-{trace_id}.json"
        if path.exists():
            if missing_seen:
                raise ValueError("target trace sequence is not a contiguous prefix")
            trace = read_json(path)
            if (trace.get("trace_hash") != digest({k: v for k, v in trace.items() if k != "trace_hash"})
                    or trace.get("run_hash") != run["run_hash"]
                    or trace.get("trace_id") != trace_id or trace.get("task_id") != task["task_id"]
                    or trace.get("role") != role or trace.get("validation_role") != validation_role
                    or trace.get("seed") != seed):
                raise ValueError("target trace identity or checksum mismatch")
            if _identity_conflicts(trace["episode"], response_reference):
                raise ValueError("stored target rollouts have inconsistent response identity")
            _remember_identity(trace["episode"], response_reference)
            traces.append(trace)
            continue
        missing_seen = True
        attempt_path = root / "attempts" / f"{index:06d}.json"
        if attempt_path.exists():
            raise ValueError("incomplete target rollout exists; reconcile its outcome before retrying")
        _exclusive_json(attempt_path, {"run_hash": run["run_hash"], "trace_id": trace_id,
                                       "task_id": task["task_id"], "seed": seed, "status": "started"})
        env = actor = None
        cleanup_errors = []
        try:
            actor = actor_factory()
            env = env_factory(copy.deepcopy(task))
            if actor.configuration() != actor_config or env.configuration() != env_configs[task["task_id"]]:
                raise ValueError("actor/environment configuration changed during target collection")
            sampler = OnlineCheckpointSampler(role=role, task_id=task["task_id"],
                                               trace_id=trace_id, seed=seed, hazard=hazard)
            runner_args = {"spec": spec, "seed": seed, "capture_checkpoints": True,
                           "checkpoint_sampler": sampler}
            if prepare_before_action:
                from cave_mem_v3.target_data.pre_treatment_state import build_pre_treatment_state
                from cave_mem_v3.target_data.retrieval import retrieve_memory

                def prepare_selected(*, context, checkpoint, action, step_index,
                                     remaining_computation):
                    retrieval = retrieve_memory(bank, context=context, action=action,
                                                encoder=encoder)
                    return build_pre_treatment_state(
                        context=context, action=action, retrieval=retrieval, bank=bank,
                        encoder=encoder, remaining_computation=remaining_computation,
                        exposure_rules=exposure_rules)

                runner_args["prepare_selected"] = prepare_selected
            episode = episode_runner(env, actor, **runner_args)
        except Exception as exc:
            episode = {"status": "error", "success": None, "utility": None,
                       "costs": {}, "steps": [], "error": {"type": type(exc).__name__,
                                                                "scope": "target_rollout"},
                       "actor_config": actor_config,
                       "environment_config": env_configs[task["task_id"]]}
        finally:
            for item in (env, actor):
                if item is not None:
                    try:
                        _close(item)
                    except Exception as exc:
                        cleanup_errors.append(type(exc).__name__)
        identity_conflicts = _identity_conflicts(episode, response_reference)
        if identity_conflicts:
            episode = _record_identity_error(episode, response_reference, identity_conflicts)
        _remember_identity(episode, response_reference)
        trace = {"schema_version": "target-trace.v1", "trace_id": trace_id,
                 "task_id": task["task_id"], "role": role,
                 "validation_role": task.get("validation_role"),
                 "group_id": task.get("group_id"), "manifest_hash": manifest["manifest_hash"],
                 "run_hash": run["run_hash"], "run_index": index, "seed": seed,
                 "episode": episode}
        if cleanup_errors:
            trace["cleanup_errors"] = cleanup_errors
        trace["trace_hash"] = digest(trace)
        _exclusive_json(path, trace)
        traces.append(trace)
        write_json(root / "summary.json", _summary(run, traces))

    report = _summary(run, traces)
    write_json(root / "summary.json", report)
    return report


def _summary(run: dict, traces: list[dict]) -> dict:
    episodes = [trace["episode"] for trace in traces]
    selected = sum((episode.get("checkpoint_sampling") or {}).get("status") == "selected"
                   for episode in episodes)
    prepared = sum(episode.get("status") == "pre_treatment_ready" for episode in episodes)
    no_selection = sum((episode.get("checkpoint_sampling") or {}).get("status") == "no_selection"
                       for episode in episodes)
    sampling_errors = sum((episode.get("checkpoint_sampling") or {}).get("status") == "sampling_incomplete"
                          for episode in episodes)
    result = {"schema_version": "target-summary.v1", "run_hash": run["run_hash"],
              "manifest_hash": run["manifest_hash"], "role": run["role"],
              **partition_fields(run["role"], run.get("validation_role")),
              "planned_rollouts": len(run["task_order"]), "attempted_rollouts": len(traces),
              "successful_rollouts": sum(e.get("success") is True for e in episodes),
              "failed_rollouts": sum(e.get("success") is False and e.get("status") != "error" for e in episodes),
              "infrastructure_errors": sum(e.get("status") == "error" for e in episodes),
              "selected_checkpoints": selected, "rollouts_without_selection": no_selection,
              "pre_treatment_states_frozen": prepared,
              "sampling_incomplete": sampling_errors,
              "trace_ids": [trace["trace_id"] for trace in traces],
              "selection_policy": ("one online risk-biased checkpoint per task rollout; state frozen before selected action execution"
                                   if run.get("pre_treatment") else
                                   "one online risk-biased checkpoint per task rollout; rollout continues to terminal outcome")}
    result["summary_hash"] = digest(result)
    return result
