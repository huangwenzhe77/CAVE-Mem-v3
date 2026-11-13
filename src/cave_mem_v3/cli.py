"""Phase one commands: manifest -> collect -> candidates -> verify -> bank."""
from __future__ import annotations

import argparse
import copy
import json
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


def _actor(raw):
    from cave_mem_v3.agents.providers import OpenAICompatibleProvider, OpenAIConfig, ProviderActor, ScriptedActor
    config = copy.deepcopy(raw["actor"])
    kind = config.pop("kind")
    if kind == "scripted":
        if raw["environment"]["kind"] != "fixture":
            raise ValueError("scripted actor is restricted to software fixtures")
        return ScriptedActor(**config)
    for key in ("model", "base_url", "revision"):
        if not config.get(key):
            raise ValueError(f"configure actor.{key} before real model requests")
    prompt = config.pop("system_prompt", None)
    provider = OpenAICompatibleProvider(OpenAIConfig(**config))
    if prompt is None:
        prompt = ('Solve the task using the visible observations and executed history. '
                  'Return exactly one JSON action: {"type":"shell","command":"..."} '
                  'or {"type":"finish"} when complete. Each shell command starts a fresh '
                  'process; filesystem changes persist. No markdown or explanation.')
    return ProviderActor(provider, system_prompt=prompt)


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

    def factory(task):
        protocol = {"max_steps": raw["study"].get("max_steps", 30),
                    **raw["environment"].get("protocol", {}), **task.get("protocol", {})}
        terminal = task_from_directory(task["path"], task["task_id"],
                                       repository_commit=revision, protocol=protocol)
        expected = task.get("terminal_asset_hash")
        if expected is None or terminal.asset_hash != expected:
            raise ValueError("terminal task assets differ from the prepared manifest")
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


def _candidates(args):
    from cave_mem_v3.memory_generation.candidates import candidate_templates, generate_candidates
    _new_file(args.output)
    traces = _collection_traces(args.collection)
    if not args.generate:
        candidates = candidate_templates(traces, max_steps_per_trace=args.max_candidates)
        generations = []
    else:
        if not args.config:
            raise ValueError("--generate requires --config with a configured hypothesis model")
        actor = _actor(load_config(args.config))
        provider = getattr(actor, "provider", None)
        if provider is None:
            raise ValueError("automatic hypotheses require a real configured provider")
        failures = [t for t in traces if t["episode"].get("success") is False and t["episode"].get("status") != "error"]
        plan = {"source_run_hash": traces[0]["run_hash"], "source_trace_hashes": [t["trace_hash"] for t in traces],
                "provider_config": provider.configuration(), "max_candidates": args.max_candidates,
                "schedule": [t["trace_id"] for t in failures]}
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
                with pending.open("x", encoding="utf-8") as stream:
                    json.dump({"plan_hash": plan["plan_hash"], "trace_hash": trace["trace_hash"],
                               "status": "request_started"}, stream)
                successes = [t for t in traces if t["task_id"] == trace["task_id"] and t["episode"].get("success") is True]
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
                pending.unlink()
            candidates.extend(result["candidates"])
            generations.append(result["generation"])
    output = {"schema_version": "source-candidates.v3", "candidates": candidates,
              "generations": generations, "source_run_hash": traces[0]["run_hash"],
              "note": "Unverified hypotheses. Editing requires replay of the finalized candidate."}
    write_json(args.output, output)
    return {"output": str(Path(args.output).resolve()), "candidate_count": len(candidates),
            "generation_errors": sum(g.get("status") == "error" for g in generations)}


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
    for key in ("screening_pairs", "confirmation_pairs"):
        if key not in verification_config:
            raise ValueError(f"set verification.{key} explicitly before source replay")
    if verification_config.get("family_size", len(candidates)) < len(candidates):
        raise ValueError("verification.family_size cannot be smaller than the frozen candidate batch")
    verification_config.setdefault("family_size", len(candidates))
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
    return {"output": str(output.resolve()), "candidate_count": len(records)}


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
    if task is None or task["role"] != "history":
        raise ValueError("phase-one audit requires a registered historical task")
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
    p = sub.add_parser("check", help="check pending configuration, without model/network access")
    p.add_argument("--config", required=True)
    p = sub.add_parser("manifest", help="freeze benchmark roles and task assets")
    p.add_argument("--catalog", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("collect", help="collect historical tasks only; resume an identical run")
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("candidates", help="write editable hypotheses or propose them with an LLM")
    p.add_argument("--collection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-candidates", type=int, default=3)
    p.add_argument("--generate", action="store_true")
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
            result = globals()["_" + args.command](args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"cave3: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
