"""Auditable memory construction, target collection, policy learning and frozen evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path

from cave_mem_v3.common.config import load_config, readiness
from cave_mem_v3.common.contracts import StudySpec, digest, plain, read_json, write_json


def _new_file(path):
    if Path(path).exists():
        raise ValueError(f"output already exists: {path}; choose a new artifact path")


def _new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_policy_dataset(args):
    from cave_mem_v3.policy.dataset import write_policy_dataset
    sources = [{"kind": "paired", "run": run, "prepared": prepared}
               for run, prepared in args.paired]
    sources.extend({"kind": "single", "run": run} for run in args.single)
    sources.extend({"kind": "auto", "run": run} for run in args.auto)
    for run, exposure in args.exposure:
        matches = [source for source in sources if source["kind"] in ("paired", "auto")
                   and Path(source["run"]).resolve() == Path(run).resolve()]
        if len(matches) != 1 or "exposure" in matches[0]:
            raise ValueError("each --exposure RUN must match exactly one paired/auto source")
        matches[0]["exposure"] = exposure
    dataset = write_policy_dataset(sources, args.output, role=args.role, validation_role=args.validation_role)
    return {"output": str(Path(args.output).resolve()), "dataset_hash": dataset["record_hash"],
            "fixture_only": dataset["fixture_only"], **dataset["summary"]}


def _plan_final_evaluation(args):
    from cave_mem_v3.policy.final_evaluation import FinalEvaluationConfig, create_final_evaluation_plan
    lock = read_json(args.selection_lock)
    if not lock.get('winner'):
        raise ValueError('final plan requires a qualified selected lock')
    bundle = lock['winner']['bundle']
    raw = load_config(args.config)
    if plain(StudySpec(**raw['study'])) != bundle['study']:
        raise ValueError('final StudySpec differs from selected policy')
    result = create_final_evaluation_plan(args.selection_lock, read_json(args.baselines),
        config=FinalEvaluationConfig(**read_json(args.evaluation_config)), output=args.output,
        run_dir=args.run_dir, report_path=args.report, env_factory=_env_factory(raw,bundle['manifest']),
        actor_factory=lambda: _actor(raw))
    return {'plan_hash': result['record_hash'], 'output': result['plan_path'], 'groups': result['group_assignments']}


def _collect_final_evaluation(args):
    from cave_mem_v3.policy.final_evaluation import load_final_evaluation_plan, collect_final_evaluation
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    plan = load_final_evaluation_plan(args.plan)
    if Path(plan['run_dir']).exists():
        return collect_final_evaluation(args.plan,env_factory=None,actor_factory=None,encoder=None)
    raw = load_config(args.config)
    if plain(StudySpec(**raw['study'])) != plan['conditions']['study']:
        raise ValueError('final StudySpec changed')
    path = Path(args.encoder_path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError('encoder must be an existing local directory')
    encoder = SentenceTransformerEncoder(str(path),revision=args.encoder_revision,device=args.device)
    return collect_final_evaluation(args.plan,env_factory=_env_factory(raw,plan['conditions']['manifest']),
        actor_factory=lambda: _actor(raw),encoder=encoder)


def _report_final_evaluation(args):
    from cave_mem_v3.policy.final_evaluation import export_final_evaluation_report
    result = export_final_evaluation_report(args.plan)
    return {'report_hash':result['record_hash'], 'evaluation_valid':result['evaluation_valid'],
        'planned_tasks':result['planned_tasks'], 'observed_outcomes':result['observed_outcomes'],
        'unknown_outcomes':result['unknown_outcomes'], 'statistics':result['statistics']}


def _plan_conservative_selection(args):
    from cave_mem_v3.policy.conservative_selection import SelectionConfig, create_conservative_selection_plan
    result = create_conservative_selection_plan(read_json(args.candidates), lock_path=args.lock,
        config=SelectionConfig(**read_json(args.config)), output=args.output)
    return {"plan_hash": result["record_hash"], "candidates": result["candidate_order"], "output": result["plan_path"]}


def _collect_conservative_selection(args):
    from cave_mem_v3.policy.conservative_selection import load_conservative_selection_plan, collect_conservative_selection_plan
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    plan = load_conservative_selection_plan(args.plan)
    raw = load_config(args.config)
    if plain(StudySpec(**raw["study"])) != plan["conditions"]["study"]:
        raise ValueError("selection study configuration differs from prospective plan")
    if any(Path(path).exists() for path in (plan["lock_path"], *(e["rollout_dir"] for e in plan["candidates"]))):
        raise ValueError("selection output exists; no retries or encoder/provider construction")
    encoder = SentenceTransformerEncoder(args.encoder_path, revision=args.encoder_revision, device=args.device)
    return collect_conservative_selection_plan(args.plan, env_factory=_env_factory(raw, plan["conditions"]["manifest"]),
        actor_factory=lambda: _actor(raw), encoder=encoder)


def _select_conservative_policy(args):
    from cave_mem_v3.policy.conservative_selection import select_conservative_policy
    result = select_conservative_policy(args.plan)
    return {"status": result["status"], "lock_hash": result["record_hash"],
        "winner": result["winner"]["candidate_id"] if result["winner"] else None, "candidates": result["candidates"]}


def _plan_policy_selection(args):
    from cave_mem_v3.policy.selection import SelectionConfig, create_selection_plan
    result = create_selection_plan(read_json(args.candidates), audit_dir=args.audit_dir, lock_path=args.lock,
        config=SelectionConfig(**read_json(args.config)), output=args.output)
    return {"plan_hash": result["record_hash"], "candidates": result["candidate_order"], "output": result["plan_path"]}


def _collect_policy_selection(args):
    from cave_mem_v3.policy.selection import load_selection_plan, collect_selection_plan
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    plan = load_selection_plan(args.plan)
    raw = load_config(args.config)
    if plain(StudySpec(**raw["study"])) != plan["conditions"]["study"]:
        raise ValueError("selection study configuration differs from prospective plan")
    if any(Path(path).exists() for path in (plan["audit_dir"], plan["lock_path"], *(e["rollout_dir"] for e in plan["candidates"]))):
        raise ValueError("selection output exists; no retries or encoder/provider construction")
    encoder = SentenceTransformerEncoder(args.encoder_path, revision=args.encoder_revision, device=args.device)
    return collect_selection_plan(args.plan, env_factory=_env_factory(raw, plan["conditions"]["manifest"]),
        actor_factory=lambda: _actor(raw), encoder=encoder)


def _select_policy(args):
    from cave_mem_v3.policy.selection import select_policy
    result = select_policy(args.plan)
    return {"status": result["status"], "lock_hash": result["record_hash"],
        "winner": result["winner"]["candidate_id"] if result["winner"] else None, "candidates": result["candidates"]}


def _freeze_policy(args):
    from cave_mem_v3.benchmarks.tasks import load_manifest
    from cave_mem_v3.policy.frozen import freeze_policy
    from cave_mem_v3.target_data.sampling import RiskHazard
    raw = load_config(args.config)
    actor = _actor(raw)
    try:
        actor_config = actor.configuration()
    finally:
        close = getattr(actor, "close", None)
        if close:
            close()
    bundle = freeze_policy(load_manifest(args.manifest), read_json(args.bank), actor_config=actor_config,
        encoder_identity=read_json(args.encoder_identity), study=StudySpec(**raw["study"]),
        hazard=RiskHazard(**read_json(args.hazard)), policy_config=read_json(args.policy_config),
        exposure_rules=read_json(args.exposure_rules) if args.exposure_rules else None,
        model_path=args.model, calibration_path=args.calibration, warm_gate_path=args.warm_gate,
        bandit_path=args.bandit, output=args.output)
    return {"bundle_hash": bundle["record_hash"], "output": str(Path(args.output).resolve()), "validation_required": True}


def _collect_policy_validation(args):
    from cave_mem_v3.policy.rollout import collect_policy_validation, existing_validation_summary, _role
    from cave_mem_v3.policy.frozen import load_frozen_policy
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    _role(args.role, args.validation_role)
    bundle = load_frozen_policy(args.bundle)
    raw = load_config(args.config)
    if plain(StudySpec(**raw["study"])) != bundle["study"]:
        raise ValueError("validation StudySpec differs from frozen bundle")
    config_identity = digest({"execution_config": raw, "encoder_path": str(Path(args.encoder_path).expanduser().resolve()),
        "encoder_revision": args.encoder_revision, "device": args.device})
    existing = existing_validation_summary(args.output, bundle, config_identity)
    if existing is not None:
        return existing
    encoder = None
    if bundle['policy_config']['kind'] != 'no_memory':
        encoder_path = Path(args.encoder_path)
        if not encoder_path.is_dir():
            raise ValueError("encoder must be an existing local directory")
        encoder = SentenceTransformerEncoder(str(encoder_path.resolve()), revision=args.encoder_revision, device=args.device)
    return collect_policy_validation(args.bundle, env_factory=_env_factory(raw, bundle["manifest"]),
        actor_factory=lambda: _actor(raw), encoder=encoder, output_dir=args.output,
        role=args.role, validation_role=args.validation_role, config_identity=config_identity)


def _summarize_policy_validation(args):
    from cave_mem_v3.policy.rollout import summarize_policy_validation
    return summarize_policy_validation(args.run)


def _train_bandit(args):
    from cave_mem_v3.policy.bandit import BanditConfig, train_bandit
    result = train_bandit(args.dataset, args.model, args.calibration, args.warm_gate,
        config=BanditConfig(**read_json(args.config)), dr_path=args.dr, output=args.output)
    return {"output": str(Path(args.output).resolve()), "artifact_hash": result["record_hash"],
        "fixture_only": result["fixture_only"], "status": result["status"], "reason": result["reason"],
        "denominators": result["denominators"]}


def _estimate_single_dr(args):
    from cave_mem_v3.policy.dr import DRConfig, estimate_single_dr
    result = estimate_single_dr(args.dataset, config=DRConfig(**read_json(args.config)), output=args.output)
    return {"output": str(Path(args.output).resolve()), "artifact_hash": result["record_hash"],
            "fixture_only": result["fixture_only"], **result["summary"]}


def _train_policy(args):
    from cave_mem_v3.policy.supervised import SupervisedConfig, fit_supervised
    config = SupervisedConfig(**read_json(args.config))
    model = fit_supervised(args.dataset, config=config, output=args.output)
    return {"output": str(Path(args.output).resolve()), "model_hash": model["record_hash"],
            "fixture_only": model["fixture_only"], "heads": {k: v["status"] for k, v in model["heads"].items()}}


def _calibrate_policy(args):
    from cave_mem_v3.policy.calibration import CalibrationConfig, calibrate_supervised
    calibration = calibrate_supervised(args.model, args.dataset,
        config=CalibrationConfig(**read_json(args.config)), output=args.output)
    return {"output": str(Path(args.output).resolve()), "calibration_hash": calibration["record_hash"],
            "fixture_only": calibration["fixture_only"], "heads": {k: v["status"] for k, v in calibration["heads"].items()}}


def _train_initial_gate(args):
    from cave_mem_v3.policy.gate import GateConfig, fit_initial_gate
    gate = fit_initial_gate(args.model, args.calibration, args.dataset,
        config=GateConfig(**read_json(args.config)), output=args.output)
    return {"output": str(Path(args.output).resolve()), "gate_hash": gate["record_hash"],
            "fixture_only": gate["fixture_only"], "status": gate["head"]["status"]}


def _predict_policy(args):
    from cave_mem_v3.policy.supervised import load_supervised, predict_supervised, _save
    from cave_mem_v3.policy.calibration import predict_calibrated
    from cave_mem_v3.policy.gate import predict_initial_gate
    model = load_supervised(args.model)
    x = read_json(args.features)
    sources = list(model.get("raw_sources", []))
    if args.gate and not args.calibration:
        raise ValueError("initial gate prediction requires its calibration")
    if args.calibration:
        calibration = read_json(args.calibration)
        sources.extend(calibration.get("raw_sources", []))
        result = predict_initial_gate(read_json(args.gate), model, calibration, x) if args.gate else predict_calibrated(model, x, calibration)
    else:
        result = predict_supervised(model, x)
    if args.output:
        _save(args.output, result, sources)
    return result


def _provider(raw, section):
    from cave_mem_v3.agents.providers import OpenAICompatibleProvider, OpenAIConfig
    config = copy.deepcopy(raw[section])
    kind = config.pop("kind")
    if kind != "openai_compatible":
        raise ValueError(f"{section} must use an openai_compatible provider for real requests")
    for key in ("model", "base_url", "revision"):
        if not config.get(key):
            raise ValueError(f"configure {section}.{key} before real model requests")
    return OpenAICompatibleProvider(OpenAIConfig(**config))


def _actor(raw):
    from cave_mem_v3.agents.providers import ProviderActor, ScriptedActor
    config = copy.deepcopy(raw["actor"])
    kind = config.pop("kind")
    if kind == "scripted":
        if raw["environment"]["kind"] != "fixture":
            raise ValueError("scripted actor is restricted to software fixtures")
        return ScriptedActor(**config)
    prompt = config.pop("system_prompt", None)
    raw_with_actor = copy.deepcopy(raw)
    raw_with_actor["actor"] = {"kind": kind, **config}
    provider = _provider(raw_with_actor, "actor")
    if prompt is None:
        prompt = ('Solve the task using the visible observations and executed history. '
                  'Return exactly one JSON action: {"type":"shell","command":"..."} '
                  'or {"type":"finish"} when complete. Each shell command starts a fresh '
                  'process; filesystem changes persist. No markdown or explanation.')
    return ProviderActor(provider, system_prompt=prompt)


def _teacher(raw):
    if "teacher" not in raw:
        raise ValueError("automatic hypotheses require a separate [teacher] provider configuration")
    return _provider(raw, "teacher")


def _env_factory(raw, manifest):
    from cave_mem_v3.environments.base import PreparationEnvironment
    from cave_mem_v3.environments.terminal import TerminalEnvironment, task_from_directory
    kind = raw["environment"]["kind"]
    if kind == "fixture":
        if manifest["benchmark"] != "preparation-fixture":
            raise ValueError("fixture environment requires the explicit fixture benchmark")
        return lambda task: PreparationEnvironment(task_id=task["task_id"],
            max_steps=raw["study"].get("max_steps", 30), **task.get("environment", {}))
    if manifest["benchmark"] != "terminal-bench@2.0":
        raise ValueError("terminal environment requires terminal-bench@2.0")
    revision = manifest["revision"]
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("pin the full 40-character Terminal-Bench commit before execution")
    audit_map_path = raw["environment"].get("audit_map_path")
    audit_map = read_json(audit_map_path) if audit_map_path else {}
    if not isinstance(audit_map, dict) or not set(audit_map).issubset(
            {task["task_id"] for task in manifest["tasks"]}):
        raise ValueError("environment.audit_map_path must map registered task ids to audits")
    if any(not isinstance(audit, dict) for audit in audit_map.values()):
        raise ValueError("environment audit map entries must be records")

    def factory(task):
        protocol = {"max_steps": raw["study"].get("max_steps", 30),
                    **raw["environment"].get("protocol", {}), **task.get("protocol", {})}
        if task["task_id"] in audit_map:
            if "audit" in protocol or "audit_path" in protocol:
                raise ValueError("task audit is duplicated between manifest and external map")
            protocol["audit"] = copy.deepcopy(audit_map[task["task_id"]])
        terminal = task_from_directory(task["path"], task["task_id"],
                                       repository_commit=revision, protocol=protocol)
        expected = task.get("terminal_asset_hash")
        if expected is None or terminal.asset_hash != expected:
            raise ValueError("terminal task assets differ from the prepared manifest")
        backend_name = raw["environment"].get("backend", "docker")
        if backend_name == "enroot":
            from cave_mem_v3.environments.enroot_runtime import EnrootRuntimeBackend
            snapshots = raw["environment"].get("snapshot_dir")
            if not snapshots:
                raise ValueError("configure environment.snapshot_dir for Enroot")
            return TerminalEnvironment(terminal, backend=EnrootRuntimeBackend(snapshots))
        if backend_name != "docker":
            raise ValueError("environment.backend must be docker or enroot")
        return TerminalEnvironment(terminal)
    return factory


def _collection_traces(path):
    from cave_mem_v3.memory_generation.candidates import checked_trace
    path = Path(path)
    if not (path / "run.json").is_file():
        raise ValueError("expected a history collection directory with run.json")
    traces = [read_json(p) for p in sorted((path / "traces").glob("*.json"))]
    if not traces:
        raise ValueError("history collection has no traces")
    for trace in traces:
        checked_trace(trace)
    if len({t["run_hash"] for t in traces}) != 1 or len({t["manifest_hash"] for t in traces}) != 1:
        raise ValueError("source traces belong to different collections")
    return traces


def _manifest(args):
    from cave_mem_v3.benchmarks.tasks import build_manifest
    from cave_mem_v3.environments.terminal import task_asset_hash
    _new_file(args.output)
    catalog = read_json(args.catalog)
    tasks = copy.deepcopy(catalog["tasks"])
    for task in tasks:
        if "path" in task:
            path = Path(task["path"])
            if not path.is_absolute():
                path = Path(args.catalog).resolve().parent / path
            task["path"] = str(path.resolve())
            if catalog["benchmark"] == "terminal-bench@2.0":
                task["terminal_asset_hash"] = task_asset_hash(path)
        protocol = task.get("protocol", {})
        if protocol.get("audit_path"):
            path = Path(protocol.pop("audit_path"))
            if not path.is_absolute():
                path = Path(args.catalog).resolve().parent / path
            protocol["audit"] = read_json(path)
    manifest = build_manifest(tasks, benchmark=catalog["benchmark"], revision=catalog["revision"],
                              split_ratios=catalog.get("split_ratios"), seed=catalog.get("seed", 42),
                              validation_ratios=catalog.get("validation_ratios"))
    write_json(args.output, manifest)
    return {"manifest": str(Path(args.output).resolve()), "task_count": len(tasks)}


def _collect(args):
    from cave_mem_v3.memory_generation.collection import CollectionConfig, collect_history
    from cave_mem_v3.benchmarks.tasks import load_manifest
    raw = load_config(args.config)
    manifest = load_manifest(args.manifest)
    spec = StudySpec(**raw["study"])
    for key in ("runs_per_task", "max_runs"):
        if key not in raw["collection"]:
            raise ValueError(f"set collection.{key} explicitly before collecting historical trajectories")
    config = CollectionConfig(**raw["collection"])
    # Validate deployment choices before creating a run or charging a request.
    _actor(raw)
    return collect_history(manifest, config, spec, _env_factory(raw, manifest),
                           lambda: _actor(raw), args.output)


def _collect_target(args):
    from cave_mem_v3.benchmarks.tasks import load_manifest
    from cave_mem_v3.target_data.collection import collect_target
    from cave_mem_v3.target_data.sampling import RiskHazard
    from cave_mem_v3.target_data.partitions import check_partition_resume, partition_tasks
    check_partition_resume(args.output, args.role, args.validation_role)
    raw = load_config(args.config)
    manifest = load_manifest(args.manifest)
    partition_tasks(manifest, args.role, args.validation_role)
    spec = StudySpec(**raw["study"])
    _actor(raw)
    hazard = RiskHazard(low_risk_probability=args.low_risk_probability,
                        high_risk_probability=args.high_risk_probability)
    return collect_target(manifest, role=args.role, validation_role=args.validation_role, spec=spec,
                          env_factory=_env_factory(raw, manifest),
                          actor_factory=lambda: _actor(raw),
                          output_dir=args.output, hazard=hazard)


def _prepare_target(args):
    from cave_mem_v3.benchmarks.tasks import load_manifest
    from cave_mem_v3.memory_generation.bank import load_bank
    from cave_mem_v3.target_data.collection import collect_target
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    from cave_mem_v3.target_data.sampling import RiskHazard
    from cave_mem_v3.target_data.pre_treatment_state import validate_exposure_rules
    from cave_mem_v3.target_data.partitions import check_partition_resume, partition_tasks
    check_partition_resume(args.output, args.role, args.validation_role)
    raw = load_config(args.config)
    manifest = load_manifest(args.manifest)
    partition_tasks(manifest, args.role, args.validation_role)
    spec = StudySpec(**raw["study"])
    encoder_path = Path(args.encoder_path)
    if not encoder_path.is_dir():
        raise ValueError("--encoder-path must point to an existing local embedding model directory; downloads are disabled")
    bank = load_bank(args.bank)
    if args.role == "validation":
        from cave_mem_v3.target_data.partitions import validate_validation_bank
        validate_validation_bank(manifest, bank)
    try:
        encoder = SentenceTransformerEncoder(str(encoder_path.resolve()),
                                             revision=args.encoder_revision,
                                             device=args.device)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    exposure_rules = read_json(args.exposure_rules) if args.exposure_rules else None
    validate_exposure_rules(exposure_rules)
    _actor(raw)
    hazard = RiskHazard(low_risk_probability=args.low_risk_probability,
                        high_risk_probability=args.high_risk_probability)
    return collect_target(
        manifest, role=args.role, validation_role=args.validation_role, spec=spec,
        env_factory=_env_factory(raw, manifest), actor_factory=lambda: _actor(raw),
        output_dir=args.output, hazard=hazard, prepare_before_action=True,
        bank=bank, encoder=encoder, exposure_rules=exposure_rules)


def _retrieve_target(args):
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    from cave_mem_v3.target_data.retrieval_collection import retrieve_target_run
    encoder_path = Path(args.encoder_path)
    if not encoder_path.is_dir():
        raise ValueError("--encoder-path must point to an existing local embedding model directory; downloads are disabled")
    try:
        encoder = SentenceTransformerEncoder(str(encoder_path.resolve()), revision=args.encoder_revision,
                                             device=args.device)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return retrieve_target_run(args.target_run, args.bank, args.output, encoder=encoder)


def _pair_target(args):
    from cave_mem_v3.interventions.collection import collect_target_pairs
    from cave_mem_v3.interventions.inputs import checked
    raw = load_config(args.config)
    prepared = checked(read_json(Path(args.target_run) / "run.json"), "run_hash", "target-run.v1")
    if plain(StudySpec(**raw["study"])) != prepared["study"]:
        raise ValueError("pair-target must use the exact preparation StudySpec")
    return collect_target_pairs(args.target_run, read_json(args.bank),
        env_factory=_env_factory(raw, prepared["manifest"]), actor_factory=lambda: _actor(raw),
        output_dir=args.output)


def _summarize_target_pairs(args):
    from cave_mem_v3.interventions.outcomes import summarize_target_pairs
    return summarize_target_pairs(args.run)


def _collect_single_target(args):
    from cave_mem_v3.benchmarks.tasks import load_manifest
    if args.command == "collect-auto-target":
        from cave_mem_v3.interventions.auto import collect_target_auto as collector
    else:
        from cave_mem_v3.interventions.single import collect_single_target as collector
    from cave_mem_v3.target_data.pre_treatment_state import validate_exposure_rules
    from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
    from cave_mem_v3.target_data.sampling import RiskHazard
    from cave_mem_v3.target_data.partitions import check_partition_resume, partition_tasks
    check_partition_resume(args.output, args.role, args.validation_role)
    if not math.isfinite(args.assignment_probability) or not 0 < args.assignment_probability < 1:
        raise ValueError("assignment probability for Remind must be strictly between zero and one")
    raw = load_config(args.config)
    manifest = load_manifest(args.manifest, verify_assets=True)
    partition_tasks(manifest, args.role, args.validation_role)
    spec = StudySpec(**raw["study"])
    hazard = RiskHazard(low_risk_probability=args.low_risk_probability,
                        high_risk_probability=args.high_risk_probability)
    bank = read_json(args.bank)
    exposure_rules = read_json(args.exposure_rules) if args.exposure_rules else None
    validate_exposure_rules(exposure_rules)
    encoder_path = Path(args.encoder_path)
    if not encoder_path.is_dir():
        raise ValueError("--encoder-path must point to an existing local embedding model directory; downloads are disabled")
    try:
        encoder = SentenceTransformerEncoder(str(encoder_path.resolve()),
                                             revision=args.encoder_revision, device=args.device)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    return collector(manifest, bank=bank, role=args.role, validation_role=args.validation_role, spec=spec,
        env_factory=_env_factory(raw, manifest), actor_factory=lambda: _actor(raw),
        output_dir=args.output, encoder=encoder, hazard=hazard, exposure_rules=exposure_rules,
        assignment_probability=args.assignment_probability)


def _collect_auto_target(args):
    return _collect_single_target(args)


def _summarize_target_auto(args):
    from cave_mem_v3.interventions.auto import summarize_target_auto
    return summarize_target_auto(args.run)


def _judge_paired_exposure(args):
    from cave_mem_v3.interventions.exposure import judge_paired_exposure
    teacher = _teacher(load_config(args.config))
    return judge_paired_exposure(args.run, provider=teacher, output_dir=args.output)


def _summarize_paired_exposure(args):
    from cave_mem_v3.interventions.exposure import summarize_paired_exposure
    return summarize_paired_exposure(args.run)


def _summarize_target_single(args):
    from cave_mem_v3.interventions.single_outcomes import summarize_target_single
    return summarize_target_single(args.run)


def _candidates(args):
    from cave_mem_v3.memory_generation.candidates import candidate_templates, eligible_source_steps, generate_candidates, validate_candidate
    from cave_mem_v3.memory_generation.collection import scored_source_outcome
    if args.max_candidates != 1:
        raise ValueError("each eligible source trace must have exactly one candidate")
    _new_file(args.output)
    traces = _collection_traces(args.collection)
    if not args.generate:
        if args.reuse_successes_from:
            raise ValueError("--reuse-successes-from requires --generate")
        candidates = candidate_templates(traces, max_steps_per_trace=args.max_candidates)
        generations = []
    else:
        if not args.config:
            raise ValueError("--generate requires --config with a configured hypothesis model")
        provider = _teacher(load_config(args.config))
        failures = [t for t in traces if eligible_source_steps(t)]
        excluded = [{"trace_id": t["trace_id"], "task_id": t["task_id"],
                     "status": t["episode"].get("status"),
                     "reason": ("no_executable_source_step" if scored_source_outcome(t["episode"]) is False
                                else "not_a_scored_task_failure")}
                    for t in traces if t["episode"].get("success") is False and t not in failures]
        reused = {}
        prior_hash = None
        if args.reuse_successes_from:
            prior = read_json(args.reuse_successes_from)
            if (prior.get("schema_version") != "source-candidates.v3"
                    or prior.get("source_run_hash") != traces[0]["run_hash"]
                    or len(prior.get("generations", [])) != len(failures)):
                raise ValueError("prior candidate batch differs from this historical collection")
            prior_hash = digest(prior)
            prior_candidates = prior.get("candidates", [])
            if not isinstance(prior_candidates, list):
                raise ValueError("prior candidate batch is invalid")
            for trace, generation in zip(failures, prior["generations"]):
                matches = [candidate for candidate in prior_candidates
                           if candidate.get("source_trace_id") == trace["trace_id"]]
                if generation.get("status") == "completed":
                    if len(matches) != 1 or matches[0].get("generation") != generation:
                        raise ValueError("prior completed generation has no unique bound candidate")
                    validate_candidate(matches[0], trace)
                    reused[trace["trace_id"]] = {"candidates": matches, "generation": generation}
                elif generation.get("status") != "error" or matches:
                    raise ValueError("prior failed generation contains an unexpected candidate")
            if len(prior_candidates) != len(reused):
                raise ValueError("prior candidate batch has unregistered candidates")
        plan = {"source_run_hash": traces[0]["run_hash"], "source_trace_hashes": [t["trace_hash"] for t in traces],
                "provider_config": provider.configuration(), "max_candidates": args.max_candidates,
                "schedule": [t["trace_id"] for t in failures]}
        if prior_hash is not None:
            plan["reuse_source_hash"] = prior_hash
        plan["plan_hash"] = digest(plan)
        journal = Path(str(args.output) + ".generation")
        if journal.exists():
            if not (journal / "plan.json").is_file() or read_json(journal / "plan.json") != plan:
                raise ValueError("candidate generation journal differs from this frozen plan")
        else:
            journal.mkdir(parents=True)
            write_json(journal / "plan.json", plan)
        candidates, generations = [], []
        for index, trace in enumerate(failures):
            completed = journal / f"{index:06d}.json"
            pending = journal / f"{index:06d}.pending.json"
            if completed.exists():
                entry = read_json(completed)
                if (entry.get("plan_hash") != plan["plan_hash"] or entry.get("trace_hash") != trace["trace_hash"]
                    or entry.get("entry_hash") != digest({k: v for k, v in entry.items() if k != "entry_hash"})):
                    raise ValueError("candidate generation journal entry changed")
                result = entry["result"]
            else:
                if pending.exists():
                    raise ValueError("candidate request was interrupted; preserve its journal and reconcile its outcome before retrying")
                if trace["trace_id"] in reused:
                    result = reused[trace["trace_id"]]
                else:
                    with pending.open("x", encoding="utf-8") as stream:
                        json.dump({"plan_hash": plan["plan_hash"], "trace_hash": trace["trace_hash"],
                                   "status": "request_started"}, stream)
                    successes = [t for t in traces if t["task_id"] == trace["task_id"]
                                 and scored_source_outcome(t["episode"]) is True]
                    try:
                        result = generate_candidates(trace, provider, successful_traces=successes,
                                                     max_candidates=args.max_candidates, seed=trace["seed"])
                    except Exception as exc:
                        generation = getattr(exc, "generation", None)
                        if generation is None:
                            attempt = getattr(exc, "attempt", None)
                            generation = {"status": "error", "error_type": type(exc).__name__,
                                          "attempt": attempt, "usage": attempt.get("usage") if attempt else None}
                        result = {"candidates": getattr(exc, "candidates", []), "generation": generation}
                entry = {"plan_hash": plan["plan_hash"], "trace_hash": trace["trace_hash"], "result": result}
                entry["entry_hash"] = digest(entry)
                write_json(completed, entry)
                if pending.exists():
                    pending.unlink()
            candidates.extend(result["candidates"])
            generations.append(result["generation"])
    output = {"schema_version": "source-candidates.v3", "candidates": candidates,
              "generations": generations, "source_run_hash": traces[0]["run_hash"],
              "note": "Unverified hypotheses. Editing requires replay of the finalized candidate."}
    if args.generate:
        output["excluded_source_traces"] = excluded
        if prior_hash is not None:
            output["reuse_source"] = {"path": str(Path(args.reuse_successes_from).resolve()),
                                      "content_hash": prior_hash,
                                      "reused_trace_ids": sorted(reused)}
    write_json(args.output, output)
    return {"output": str(Path(args.output).resolve()), "candidate_count": len(candidates),
            "generation_errors": sum(g.get("status") == "error" for g in generations),
            "excluded_source_traces": len(excluded) if args.generate else 0}


def _verify(args):
    from cave_mem_v3.memory_generation.candidates import finalize_candidate, validate_candidate
    from cave_mem_v3.memory_generation.replay import ReplaySpec, verify_candidate
    from cave_mem_v3.benchmarks.tasks import load_manifest
    raw = load_config(args.config)
    manifest = load_manifest(args.manifest)
    traces = _collection_traces(args.collection)
    by_trace = {t["trace_id"]: t for t in traces}
    source = read_json(args.candidates)
    if source.get("source_run_hash") != traces[0]["run_hash"]:
        raise ValueError("candidate file belongs to a different historical run")
    candidates = []
    for draft in source["candidates"]:
        trace = by_trace[draft["source_trace_id"]]
        if "candidate_hash" in draft:
            validate_candidate(draft, trace)
            candidates.append(draft)
        else:
            candidates.append(finalize_candidate(draft, trace))
    if not candidates or len({c["candidate_hash"] for c in candidates}) != len(candidates):
        raise ValueError("provide a nonempty list of distinct finalized candidates")
    if any(t["manifest_hash"] != manifest["manifest_hash"] for t in traces):
        raise ValueError("source collection and verification manifest differ")
    spec = StudySpec(**raw["study"])
    verification_config = copy.deepcopy(raw["verification"])
    for key in ("replay_pairs",):
        if key not in verification_config:
            raise ValueError(f"set verification.{key} explicitly before source replay")
    replay_spec = ReplaySpec(**verification_config)
    actor = _actor(raw)
    factory = _env_factory(raw, manifest)
    by_task = {t["task_id"]: t for t in manifest["tasks"]}
    # Check original study and actor before any confirmation requests.
    for candidate in candidates:
        episode = by_trace[candidate["source_trace_id"]]["episode"]
        if episode.get("spec") != plain(spec):
            raise ValueError("source collection and replay must use the same StudySpec")
        if episode.get("actor_config") != actor.configuration():
            raise ValueError("source collection and replay must use the same frozen actor")
    output = _new_directory(args.output)
    write_json(output / "source_traces.json", traces)
    write_json(output / "candidates.json", {"candidates": candidates})
    records = []
    for candidate in candidates:
        trace = by_trace[candidate["source_trace_id"]]
        task = by_task[candidate["source_task_id"]]
        evidence = verify_candidate(candidate, trace, lambda task=task: factory(task), lambda: _actor(raw),
            study=spec, replay_spec=replay_spec, output_dir=output / candidate["candidate_hash"][:20])
        records.append({"candidate": candidate, "evidence": evidence})
        write_json(output / "verification.json", {"schema_version": "source-verification-run.v3",
            "manifest_hash": manifest["manifest_hash"], "records": records,
            "planned_candidates": len(candidates), "complete": len(records) == len(candidates)})
    from cave_mem_v3.memory_generation.bank import build_memory, freeze_bank
    memories = [build_memory(record["candidate"], by_trace[record["candidate"]["source_trace_id"]],
                             record["evidence"])
                for record in records if record["evidence"].get("v") == 1]
    bank_path = output / "bank.json" if memories else None
    if bank_path is not None:
        verification = read_json(output / "verification.json")
        freeze_bank(memories, bank_path, source_traces=traces,
                    metadata={"manifest_hash": manifest["manifest_hash"],
                              "verification_hash": digest(verification),
                              "excluded_candidates": [record["candidate"]["candidate_hash"]
                                  for record in records if record["evidence"].get("v") != 1]})
    return {"output": str(output.resolve()), "candidate_count": len(records),
            "verified_memories": len(memories), "bank": str(bank_path.resolve()) if bank_path else None}


def _bank(args):
    from cave_mem_v3.memory_generation.bank import build_memory, freeze_bank
    _new_file(args.output)
    root = Path(args.verification)
    verification = read_json(root / "verification.json")
    if not verification.get("complete"):
        raise ValueError("verification run is incomplete")
    traces = read_json(root / "source_traces.json")
    by_trace = {t["trace_id"]: t for t in traces}
    memories, excluded = [], []
    for record in verification["records"]:
        candidate, evidence = record["candidate"], record["evidence"]
        if evidence.get("v") != 1:
            excluded.append(candidate["candidate_hash"])
            continue
        memories.append(build_memory(candidate, by_trace[candidate["source_trace_id"]], evidence))
    freeze_bank(memories, args.output, source_traces=traces,
                metadata={"manifest_hash": verification["manifest_hash"],
                          "verification_hash": digest(verification), "excluded_candidates": excluded})
    return {"bank": str(Path(args.output).resolve()), "verified_memories": len(memories),
            "excluded_candidates": len(excluded)}


def _audit(args):
    from cave_mem_v3.benchmarks.tasks import load_manifest
    _new_file(args.output)
    raw = load_config(args.config)
    manifest = load_manifest(args.manifest)
    task = next((t for t in manifest["tasks"] if t["task_id"] == args.task), None)
    if task is None or task["role"] not in ("history", "target_train", "validation"):
        raise ValueError("audit requires a registered history, target_train or validation task")
    env = _env_factory(raw, manifest)(task)
    try:
        env.reset(StudySpec(**raw["study"]).base_seed)
        if not hasattr(env, "diagnose_restoration"):
            raise ValueError("audit command is for the real terminal adapter")
        report = env.diagnose_restoration()
        write_json(args.output, report)
    finally:
        env.close()
    return {"audit": str(Path(args.output).resolve()), "approved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser('plan-final-evaluation', help='freeze selection lock, fixed baselines and randomized held-out test groups')
    for key in ('selection-lock','baselines','evaluation-config','config','output','run-dir','report'):
        p.add_argument('--'+key,required=True)
    p = sub.add_parser('collect-final-evaluation', help='execute only test tasks assigned by a frozen final plan')
    for key in ('plan','config','encoder-path','encoder-revision'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    p = sub.add_parser('report-final-evaluation', help='audit final test journals and export immutable marginal group report')
    p.add_argument('--plan',required=True)
    p = sub.add_parser("plan-conservative-selection", help="register frozen candidates and new validation slots before outcomes")
    for key in ("candidates", "lock", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("collect-conservative-selection", help="execute a prospective validation plan without retries")
    for key in ("plan", "config", "encoder-path", "encoder-revision"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p = sub.add_parser("select-conservative-policy", help="audit planned full-task and worst-case non-restorable validation; freeze selection lock")
    p.add_argument("--plan", required=True)
    p = sub.add_parser("plan-policy-selection", help="register frozen candidates and new validation slots before outcomes")
    for key in ("candidates", "audit-dir", "lock", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("collect-policy-selection", help="execute a prospective validation plan without retries")
    for key in ("plan", "config", "encoder-path", "encoder-revision"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p = sub.add_parser("select-policy", help="audit planned full-task and independent paired validation; freeze selection lock")
    p.add_argument("--plan", required=True)
    p = sub.add_parser("freeze-policy", help="freeze component or learned policy before validation outcomes")
    for key in ("config", "manifest", "bank", "encoder-identity", "hazard", "policy-config", "output"):
        p.add_argument("--" + key, required=True)
    for key in ("exposure-rules", "model", "calibration", "warm-gate", "bandit"):
        p.add_argument("--" + key)
    p = sub.add_parser("collect-policy-validation", help="complete independent selection-validation tasks under a frozen policy")
    for key in ("bundle", "config", "encoder-path", "encoder-revision", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--role", default="validation")
    p.add_argument("--validation-role", default="selection")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p = sub.add_parser("summarize-policy-validation", help="reconstruct full-task validation outcomes from original events")
    p.add_argument("--run", required=True)
    p = sub.add_parser("train-bandit", help="train separate paired or DR same-capacity constrained candidate and no-risk comparator")
    for key in ("dataset", "model", "calibration", "warm-gate", "config", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--dr", help="audited crossfit artifact required only for single route")
    p = sub.add_parser("estimate-single-dr", help="grouped crossfit and DR training diagnostics from audited randomized single logs")
    for key in ("dataset", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("train-policy", help="fit paired benefit, harm and incremental cost heads from audited training data")
    for key in ("dataset", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("calibrate-policy", help="calibrate fixed heads on independent validation/calibration groups")
    for key in ("model", "dataset", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("train-initial-gate", help="supervise a same-capacity gate using conservative calibrated labels on training x")
    for key in ("model", "calibration", "dataset", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("predict-policy", help="offline prediction from a frozen x; no actor or environment execution")
    for key in ("model", "features"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--calibration")
    p.add_argument("--gate")
    p.add_argument("--output")
    p = sub.add_parser("build-policy-dataset", help="audit one explicit target partition and export a read-only policy dataset view")
    p.add_argument("--role", choices=("target_train", "validation"), default="target_train")
    p.add_argument("--validation-role", choices=("calibration", "selection"))
    p.add_argument("--paired", nargs=2, action="append", default=[], metavar=("RUN", "PREPARED"),
                   help="paired run directory and its original prepared directory; repeat for additional runs")
    p.add_argument("--single", action="append", default=[], help="single-arm run directory; may be repeated")
    p.add_argument("--auto", action="append", default=[], help="automatic paired-or-single run directory; may be repeated")
    p.add_argument("--exposure", nargs=2, action="append", default=[], metavar=("RUN", "LABELS"),
                   help="post-pair teacher exposure labels for a paired/auto source; may be repeated")
    p.add_argument("--output", required=True)
    p = sub.add_parser("check", help="check pending configuration, without model/network access")
    p.add_argument("--config", required=True)
    p = sub.add_parser("manifest", help="freeze benchmark roles and task assets")
    p.add_argument("--catalog", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("collect", help="collect historical tasks only; resume an identical run")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("collect-target", help="run target rollouts and select one action checkpoint per task")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", choices=("target_train", "validation"), required=True)
    p.add_argument("--validation-role", choices=("calibration", "selection"))
    p.add_argument("--low-risk-probability", type=float, default=0.05)
    p.add_argument("--high-risk-probability", type=float, default=0.75)
    p = sub.add_parser("prepare-target", help="freeze pre-treatment state at the selected checkpoint before executing a0")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--role", choices=("target_train", "validation"), required=True)
    p.add_argument("--validation-role", choices=("calibration", "selection"))
    p.add_argument("--bank", required=True)
    p.add_argument("--encoder-path", required=True,
                   help="existing local sentence-transformers model directory; no download is attempted")
    p.add_argument("--encoder-revision", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--exposure-rules",
                   help="optional versioned per-memory action-exposure rules JSON; absent rules yield e=?")
    p.add_argument("--low-risk-probability", type=float, default=0.05)
    p.add_argument("--high-risk-probability", type=float, default=0.75)
    p = sub.add_parser("pair-target", help="audit prepared target checkpoints and run independent Silent/Remind arms")
    for key in ("config", "target-run", "bank", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("summarize-target-pairs", help="verify saved artifacts and recompute paired labels without execution")
    p.add_argument("--run", required=True)
    p = sub.add_parser("collect-single-target", help="freeze inputs and run one randomized arm in the same live target session")
    for key in ("config", "manifest", "bank", "output", "encoder-path", "encoder-revision"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--role", default="target_train", help="target_train or validation; validation also requires --validation-role")
    p.add_argument("--validation-role", choices=("calibration", "selection"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--exposure-rules", help="optional frozen action-exposure rules; absent rules preserve e=?")
    p.add_argument("--assignment-probability", type=float, required=True,
                   help="frozen probability of Remind; strictly between zero and one")
    p.add_argument("--low-risk-probability", type=float, default=.05)
    p.add_argument("--high-risk-probability", type=float, default=.75)
    p = sub.add_parser("summarize-target-single", help="verify observed single-arm results without execution or counterfactual labels")
    p.add_argument("--run", required=True)
    p = sub.add_parser("collect-auto-target", help="audit each live selected checkpoint; pair when restorable, otherwise randomize one live arm")
    for key in ("config", "manifest", "bank", "output", "encoder-path", "encoder-revision"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--role", default="target_train", help="target_train or validation; validation also requires --validation-role")
    p.add_argument("--validation-role", choices=("calibration", "selection"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--exposure-rules", help="optional frozen action-exposure rules; absent rules preserve e=?")
    p.add_argument("--assignment-probability", type=float, required=True)
    p.add_argument("--low-risk-probability", type=float, default=.05)
    p.add_argument("--high-risk-probability", type=float, default=.75)
    p = sub.add_parser("summarize-target-auto", help="recompute paired and randomized single-arm observations")
    p.add_argument("--run", required=True)
    p = sub.add_parser("judge-paired-exposure", help="assess original-action mechanism exposure after paired target execution")
    for key in ("run", "config", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("summarize-paired-exposure", help="verify saved post-pair teacher exposure assessments")
    p.add_argument("--run", required=True)
    p = sub.add_parser("retrieve-target", help="retrieve one verified memory for each selected target checkpoint")
    p.add_argument("--target-run", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--encoder-path", required=True,
                   help="existing local sentence-transformers model directory; no download is attempted")
    p.add_argument("--encoder-revision", required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p = sub.add_parser("candidates", help="write editable hypotheses or propose them with an LLM")
    p.add_argument("--collection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-candidates", type=int, default=1,
                   help="one candidate per eligible failed history trace (fixed protocol)")
    p.add_argument("--generate", action="store_true")
    p.add_argument("--reuse-successes-from",
                   help="reuse validated successful generations from a prior batch of the same history; retry only failures")
    p.add_argument("--config")
    p = sub.add_parser("verify", help="independent matched source replay")
    for key in ("config", "manifest", "collection", "candidates", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("bank", help="freeze accepted source memories")
    p.add_argument("--verification", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("audit", help="generate an unapproved terminal restoration audit")
    for key in ("config", "manifest", "task", "output"):
        p.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            result = readiness(load_config(args.config))
        else:
            result = globals()["_" + args.command.replace("-", "_")](args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command == "pair-target" and result.get("pending_pairs"):
            return 2
        if args.command == "collect-single-target" and result.get("pending_rollouts"):
            return 2
        if args.command == "collect-auto-target" and result.get("pending"):
            return 2
        if args.command == "collect-policy-validation" and result.get("pending_tasks"):
            return 2
        if args.command in ("collect-policy-selection", "collect-conservative-selection") and result.get("status") == "incomplete":
            return 2
        if args.command == "collect-final-evaluation" and result.get("evaluation_valid") is not True:
            return 2
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"cave3: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
