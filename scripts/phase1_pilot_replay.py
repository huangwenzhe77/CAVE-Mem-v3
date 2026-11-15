"""Exploratory K=2 source replay; deliberately not the certified v=1 bank."""
import copy
import json
import random
import sys
from pathlib import Path

from cave_mem_v3.agents.runtime import run_episode
from cave_mem_v3.benchmarks.tasks import load_manifest
from cave_mem_v3.cli import _actor, _env_factory
from cave_mem_v3.common.config import load_config
from cave_mem_v3.common.contracts import StudySpec, digest, read_json, stage_seed, write_json
from cave_mem_v3.memory_generation.candidates import validate_candidate

root = Path(__file__).resolve().parents[1]
run = root / "runs" / sys.argv[1]
manifest = load_manifest(run / "manifest-final.json")
raw = load_config(root / "configs/phase1/server.toml")
spec = StudySpec(**raw["study"])
factory = _env_factory(raw, manifest)
traces = [read_json(p) for p in sorted((run / "collection/traces").glob("*.json"))]
by_trace = {t["trace_id"]: t for t in traces}
by_task = {t["task_id"]: t for t in manifest["tasks"]}
generated = read_json(run / "candidates.json")
if generated["source_run_hash"] != traces[0]["run_hash"]:
    raise ValueError("candidate source run mismatch")

bank = {"schema_version": "pilot-provisional-bank.v1", "gate": "two_independent_source_pairs_each_corrected_success_and_positive_utility_delta",
        "formal_v": "not_certified", "restoration": "diagnostics_only;task_scope_unapproved",
        "manifest_hash": manifest["manifest_hash"], "source_run_hash": traces[0]["run_hash"],
        "memories": [], "excluded": []}
for candidate in generated["candidates"]:
    trace = by_trace[candidate["source_trace_id"]]
    validate_candidate(candidate, trace)
    if trace["episode"]["success"] is not False or trace["manifest_hash"] != manifest["manifest_hash"]:
        raise ValueError("pilot candidate needs a failed, same-manifest source")
    checkpoint = trace["episode"]["steps"][candidate["step_index"]]["checkpoint"]
    if digest(checkpoint) != candidate["checkpoint_hash"]:
        raise ValueError("source checkpoint mismatch")
    audit = checkpoint["configuration"]["protocol"].get("audit", {})
    if (audit.get("status") != "diagnostics_passed_but_task_scope_unreviewed"
        or not audit.get("restoration_checks") or not audit.get("filesystem_probe", {}).get("branches_isolated")
        or audit.get("base_image") != checkpoint["base_image"]):
        raise ValueError("pilot needs measured recovery diagnostics for this base image")
    task = by_task[candidate["source_task_id"]]
    directory = run / "pilot-replay" / candidate["candidate_hash"]
    directory.mkdir(parents=True, exist_ok=False)
    pairs = []
    for index in range(2):
        seed = stage_seed(spec.base_seed, candidate["candidate_hash"], index, "pilot-confirmation")
        order = ["original", "corrected"]
        random.Random(seed).shuffle(order)
        row = {"candidate_hash": candidate["candidate_hash"], "checkpoint_hash": candidate["checkpoint_hash"],
               "pair_index": index, "seed": seed, "order": order, "arms": {}, "valid": False}
        for arm in order:
            env = factory(task)
            try:
                actor = _actor(raw)
                if actor.configuration() != trace["episode"]["actor_config"] or env.configuration() != trace["episode"]["environment_config"]:
                    raise ValueError("frozen actor/environment drift")
                action = candidate["problem_action"] if arm == "original" else candidate["correction"]
                result = run_episode(env, actor, spec=spec, seed=seed, checkpoint=copy.deepcopy(checkpoint),
                                     first_action=copy.deepcopy(action), capture_checkpoints=False)
                if result.get("initial_checkpoint_hash") != candidate["checkpoint_hash"]:
                    raise ValueError("restoration mismatch")
                if result.get("response_identity", {}).get("model") not in (None, trace["episode"]["response_identity"]["model"]):
                    raise ValueError("served actor model drift")
                row["arms"][arm] = result
            finally:
                env.close()
        a, b = row["arms"]["original"], row["arms"]["corrected"]
        row["valid"] = (a["status"] != "error" and b["status"] == "terminal"
            and b["success"] is True and a["utility"] is not None and b["utility"] is not None
            and b["utility"] > a["utility"] and a["initial_state_fingerprint"] == b["initial_state_fingerprint"])
        row["pair_hash"] = digest(row)
        write_json(directory / f"pair-{index}.json", row)
        pairs.append(row)
    passed = len(pairs) == 2 and all(p["valid"] for p in pairs)
    summary = {"candidate_hash": candidate["candidate_hash"], "passed": passed,
               "pair_hashes": [p["pair_hash"] for p in pairs],
               "utility_differences": [p["arms"]["corrected"]["utility"] - p["arms"]["original"]["utility"]
                                      if p["valid"] else None for p in pairs]}
    write_json(directory / "summary.json", summary)
    if passed:
        bank["memories"].append({"candidate": candidate, "source_trace_hash": trace["trace_hash"],
                                 "pilot_evidence": summary, "quality": "provisional_k2_only"})
    else:
        bank["excluded"].append(summary)
bank["memory_count"] = len(bank["memories"])
bank["bank_hash"] = digest(bank)
write_json(run / "pilot-provisional-bank.json", bank)
print(json.dumps({"candidate_count": len(generated["candidates"]), "provisional_memories": bank["memory_count"],
                  "excluded": len(bank["excluded"])}, ensure_ascii=False))
