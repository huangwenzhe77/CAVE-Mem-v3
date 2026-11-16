"""Auditable top-1 retrieval of verified memories for selected target checkpoints."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from cave_mem_v3.common.contracts import canonical, digest, read_json
from cave_mem_v3.memory_generation.bank import load_bank
from cave_mem_v3.target_data.retrieval import retrieve_memory


def retrieve_target_run(target_dir: str | Path, bank_path: str | Path,
                        output_path: str | Path, *, encoder) -> dict:
    """Create a separate, immutable retrieval artifact for a completed target run.

    This step only ranks verified memory candidates. It does not decide memory
    applicability, activate a failure mechanism, or construct a reminder.
    """
    target_dir = Path(target_dir)
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}; choose a new artifact path")
    run_path = target_dir / "run.json"
    summary_path = target_dir / "summary.json"
    trace_dir = target_dir / "traces"
    if not run_path.is_file() or not summary_path.is_file() or not trace_dir.is_dir():
        raise ValueError("target run requires run.json, summary.json, and traces/")

    run = read_json(run_path)
    expected_run_hash = run.get("run_hash")
    if not isinstance(expected_run_hash, str) or expected_run_hash != digest(
            {key: value for key, value in run.items() if key != "run_hash"}):
        raise ValueError("target run identity hash mismatch")
    summary = read_json(summary_path)
    if (summary.get("run_hash") != expected_run_hash
            or summary.get("summary_hash") != digest(
                {key: value for key, value in summary.items() if key != "summary_hash"})):
        raise ValueError("target run summary identity/hash mismatch")
    if (run.get("schema_version") != "target-run.v1"
            or run.get("role") not in ("target_train", "validation")
            or summary.get("schema_version") != "target-summary.v1"
            or summary.get("role") != run.get("role")
            or summary.get("manifest_hash") != run.get("manifest_hash")):
        raise ValueError("target run and summary roles/schema do not match")
    if (type(summary.get("planned_rollouts")) is not int
            or type(summary.get("attempted_rollouts")) is not int
            or summary["attempted_rollouts"] != summary["planned_rollouts"]
            or summary["planned_rollouts"] != len(run.get("task_order", []))):
        raise ValueError("target run is incomplete; retrieval requires all planned rollouts")

    bank = load_bank(bank_path)
    bank_hash = bank["bank_hash"]
    traces = [read_json(path) for path in sorted(trace_dir.glob("*.json"))]
    if not traces:
        raise ValueError("target run has no completed traces")
    if ([trace.get("trace_id") for trace in traces] != summary.get("trace_ids")
            or [trace.get("task_id") for trace in traces] != run.get("task_order")):
        raise ValueError("target traces do not match the frozen run summary order")

    records = []
    for index, trace in enumerate(traces):
        trace_hash = trace.get("trace_hash")
        if (trace.get("run_hash") != expected_run_hash
                or trace.get("manifest_hash") != run.get("manifest_hash")
                or trace.get("role") != run.get("role")
                or trace.get("run_index") != index
                or not isinstance(trace_hash, str)
                or trace_hash != digest({key: value for key, value in trace.items() if key != "trace_hash"})):
            raise ValueError("target trace identity/hash mismatch")
        episode = trace.get("episode", {})
        sample = episode.get("checkpoint_sampling") or {}
        record = {"trace_id": trace["trace_id"], "task_id": trace["task_id"],
                  "trace_hash": trace_hash,
                  "selected_checkpoint_hash": sample.get("selected_checkpoint_hash"),
                  "status": "not_retrieved"}
        if sample.get("status") != "selected":
            record.update(status="not_retrieved", reason=sample.get("status", "missing_sampling_record"))
            records.append(record)
            continue

        step_index = sample.get("selected_step_index")
        steps = episode.get("steps", [])
        if type(step_index) is not int or not 0 <= step_index < len(steps):
            record.update(status="invalid_selection", reason="selected_step_missing")
            records.append(record)
            continue
        step = steps[step_index]
        action = sample.get("selected_action")
        sampled_context = sample.get("selected_public_context")
        step_context = step.get("context")
        context_matches = (isinstance(sampled_context, dict) and isinstance(step_context, dict)
                           and all(step_context.get(key) == value
                                   for key, value in sampled_context.items()))
        if (step.get("checkpoint_hash") != sample.get("selected_checkpoint_hash")
                or not context_matches
                or canonical(step.get("action")) != canonical(action)
                or not isinstance(action, dict)):
            record.update(status="invalid_selection", reason="selected_context_action_or_checkpoint_mismatch")
            records.append(record)
            continue
        checkpoint = sample.get("selected_checkpoint")
        if (not isinstance(checkpoint, dict)
                or digest(checkpoint) != sample.get("selected_checkpoint_hash")):
            record.update(status="invalid_selection", reason="selected_checkpoint_hash_mismatch")
            records.append(record)
            continue

        retrieved = retrieve_memory(
            bank, context=step_context, action=action, encoder=encoder)
        record.update(status=retrieved["status"],
                      selected_memory_id=retrieved.get("selected_memory_id"),
                      similarity_score=retrieved.get("similarity_score"),
                      candidates=copy.deepcopy(retrieved.get("candidates", [])),
                      retrieval=copy.deepcopy(retrieved))
        records.append(record)

    artifact = {"schema_version": "target-memory-retrieval.v1",
                "target_run_hash": expected_run_hash,
                "target_summary_hash": summary["summary_hash"],
                "bank_hash": bank_hash,
                "encoder_identity": copy.deepcopy(encoder.identity),
                "retrieval_policy": "top1-cosine-public-context-plus-original-action.v1",
                "records": records}
    artifact["artifact_hash"] = digest(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return {"output": str(output_path.resolve()), "artifact_hash": artifact["artifact_hash"],
            "target_run_hash": expected_run_hash, "bank_hash": bank_hash,
                      "retrieved": sum(record["status"] == "retrieved" for record in records),
            "no_match": sum(record["status"] == "no_match" for record in records),
            "not_retrieved": sum(record["status"] == "not_retrieved" for record in records),
            "invalid_selection": sum(record["status"] == "invalid_selection" for record in records)}
