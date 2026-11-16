"""Post-pair teacher assessment of the cached original action's observed effect.

This is a result-side annotation. The teacher never sees the Remind arm, task
verifier, terminal success, utility or paired effect, and no single-arm session
receives this annotation. A teacher assessment is auditable evidence, not a
verified physical fact about a natural-language mechanism.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from cave_mem_v3.common.contracts import canonical, digest, plain, read_json, stage_seed, write_json
from cave_mem_v3.interventions.auto import summarize_target_auto
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.interventions.outcomes import summarize_target_pairs
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.target_data.collection import _exclusive_json
from cave_mem_v3.target_data.pre_treatment_state import _context


def _inventory(root: Path):
    run = checked(read_json(root / "run.json"))
    if run["schema_version"] == "target-pair-run.v1":
        observations = summarize_target_pairs(root)["pairs"]
        items = run["inputs"]
        entries = [(item, observation, root / "pairs" / item["record_hash"])
                   for item, observation in zip(items, observations) if item["status"] == "ready"]
    elif run["schema_version"] == "target-auto-run.v1":
        observations = summarize_target_auto(root)["observations"]
        entries = []
        for entry in observations:
            if entry["kind"] != "paired":
                continue
            # The persisted pair input is located by its frozen input hash.
            directory = next((p for p in (root / "sessions").iterdir()
                              if (p / "route.json").exists()
                              and read_json(p / "route.json").get("input_hash") == entry["input_hash"]), None)
            if directory is None:
                raise ValueError("auto paired route is missing its live session")
            route = checked(read_json(directory / "route.json"), schema="target-auto-route.v1")
            pair_dir = root / "pairs" / route["pair_input_hash"]
            item = checked(read_json(pair_dir / "input.json"))
            entries.append((item, entry["observation"], pair_dir))
    else:
        raise ValueError("posthoc exposure supports paired or auto target runs")
    memories = {m.memory_id: m for m in bank_memories(run["bank"])}
    return run, entries, memories


def _source_step(bank, memory):
    matches = []
    for trace in bank.get("audit_sources", {}).values():
        for step in trace.get("episode", {}).get("steps", []):
            if step.get("checkpoint_hash") == memory.source_checkpoint_hash:
                matches.append(step)
    if len(matches) != 1:
        raise ValueError("source memory has no unique original checkpoint record")
    return matches[0]


def _pre_x(item):
    if "pre_treatment_state" in item:
        state = item["pre_treatment_state"]
        return state["x"], state["x_hash"]
    return item["x"], item["x_hash"]


def _public_effect(step):
    result = step.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("context"), dict):
        return None
    after = plain(_context(result["context"]))
    return {"action": copy.deepcopy(step["action"]), "action_valid": result.get("valid"),
            "action_error": result.get("error"),
            "observation_after": after.get("observation"),
            "facts_after": copy.deepcopy(after.get("facts", {}))}


def _prompt(bank, memory, item, silent):
    source = _source_step(bank, memory)
    target = silent["steps"][0] if silent.get("steps") else {}
    target_effect = _public_effect(target)
    if target_effect is None:
        return None
    source_effect = _public_effect(source)
    context = plain(_context(item["context"]))
    evidence = {"memory_id": memory.memory_id, "mechanism": memory.mechanism,
        "source_original": source_effect, "source_failure_signature":
            memory.evidence["plan"]["candidate"].get("failure_signature"),
        "target_before": {"instruction": context.get("instruction"),
                          "observation": context.get("observation"),
                          "facts": context.get("facts", {})},
        "target_original": target_effect}
    messages = [{"role": "system", "content": (
        "Assess whether the target original action's observed immediate effect instantiates "
        "the historical failure mechanism. You see only public before/after observations "
        "from the Silent arm. Do not infer from terminal success, reward, or a Remind arm. "
        "Return exactly one JSON object with keys e, confidence, reason. "
        "e must be 0, 1, or '?'; use '?' when the observed effect is insufficient "
        "to confirm or rule out the mechanism. Confidence is low, medium, high, or '?'. "
        "The reason must cite concrete supplied action/observation evidence, not imagined state.")},
        {"role": "user", "content": canonical(evidence)}]
    return messages


def _assessment(text):
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {"e", "confidence", "reason"}:
        raise ValueError("teacher exposure response must contain e, confidence, reason")
    e = value["e"]
    if type(e) is not int and e != "?" or type(e) is int and e not in (0, 1):
        raise ValueError("teacher exposure e must be 0, 1, or ?")
    if value["confidence"] not in ("low", "medium", "high", "?"):
        raise ValueError("teacher exposure confidence is invalid")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("teacher exposure reason must be nonempty")
    return {"value": e, "confidence": value["confidence"],
            "reason": value["reason"].strip(), "stage": "post_silent_action_teacher.v1"}


def _final_x(item, assessment):
    pre, pre_hash = _pre_x(item)
    if digest(pre) != pre_hash:
        raise ValueError("paired frozen x hash mismatch")
    final = copy.deepcopy(pre)
    final["schema_version"] = "post-pair-state.v1"
    final["e"] = copy.deepcopy(assessment)
    return final


def judge_paired_exposure(pair_run_dir, *, provider, output_dir):
    """One metered teacher attempt per valid paired original-action result."""
    root = Path(pair_run_dir).resolve(strict=True)
    source_run, entries, memories = _inventory(root)
    config = copy.deepcopy(provider.configuration())
    run = seal({"schema_version": "posthoc-exposure-run.v1",
        "source_run": str(root), "source_run_hash": source_run["record_hash"],
        "bank_hash": source_run["bank_hash"], "teacher_config": config,
        "pair_input_hashes": [item["record_hash"] for item, _, _ in entries],
        "protocol": "one_blinded_teacher_assessment_after_valid_pair.v1"})
    output = Path(output_dir)
    if (output / "run.json").exists():
        if read_json(output / "run.json") != run:
            raise ValueError("posthoc exposure inputs changed; choose a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("new posthoc exposure output directory must be empty")
        _exclusive_json(output / "run.json", run)
    for item, observation, directory in entries:
        path = output / "labels" / (item["record_hash"] + ".json")
        if path.exists():
            checked(read_json(path), schema="posthoc-exposure-label.v1")
            continue
        attempt_path = output / "attempts" / (item["record_hash"] + ".json")
        if attempt_path.exists():
            raise ValueError("incomplete teacher exposure attempt; preserve and reconcile before retry")
        memory = memories[item["memory_id"]]
        silent_path = directory / "silent" / "result.json"
        silent = read_json(silent_path)["record"] if silent_path.exists() else {}
        messages = _prompt(source_run["bank"], memory, item, silent) if observation["outcome_valid"] else None
        pre, pre_hash = _pre_x(item)
        base = {"schema_version": "posthoc-exposure-label.v1", "judge_run_hash": run["record_hash"],
                "pair_input_hash": item["record_hash"], "memory_id": memory.memory_id,
                "pre_x_hash": pre_hash, "silent_record_hash": silent.get("record_hash"),
                "prompt_hash": digest(messages) if messages else None,
                "status": "unavailable", "assessment": None, "x_final": None,
                "model": None, "response_hash": None, "response_text": None,
                "usage": None, "provider_attempt": None, "error": None}
        if messages is None:
            _exclusive_json(path, seal({**base, "error": "silent_first_action_result_or_pair_unavailable"}))
            continue
        _exclusive_json(attempt_path, seal({"judge_run_hash": run["record_hash"],
            "pair_input_hash": item["record_hash"], "prompt_hash": digest(messages), "status": "started"}))
        try:
            seed = stage_seed(source_run["study"]["base_seed"], item["record_hash"], 0, "posthoc-exposure")
            response = provider.generate(messages, stage="target_posthoc_exposure", seed=seed,
                                         call_id="exposure-" + item["record_hash"][:20])
            if provider.configuration() != config:
                raise ValueError("teacher configuration changed during posthoc assessment")
            result = {**base, "model": response.model, "response_hash": digest(response.text),
                      "response_text": response.text, "usage": plain(response.usage),
                      "provider_attempt": plain(response.attempt)}
            try:
                if response.finish_reason == "length":
                    raise ValueError("teacher exposure response was truncated")
                assessment = _assessment(response.text)
                result.update(status="assessed", assessment=assessment,
                              x_final=_final_x(item, assessment))
            except (ValueError, TypeError, KeyError) as exc:
                result.update(status="invalid_response", error=type(exc).__name__)
        except Exception as exc:
            result = {**base, "status": "provider_error", "error": type(exc).__name__,
                      "provider_attempt": copy.deepcopy(getattr(exc, "attempt", None))}
        _exclusive_json(path, seal(result))
    return summarize_paired_exposure(output)


def summarize_paired_exposure(output_dir):
    """Validate saved teacher labels and final x without making model requests."""
    root = Path(output_dir)
    run = checked(read_json(root / "run.json"), schema="posthoc-exposure-run.v1")
    source, entries, memories = _inventory(Path(run["source_run"]))
    if source["record_hash"] != run["source_run_hash"] or source["bank_hash"] != run["bank_hash"]:
        raise ValueError("posthoc exposure source run changed")
    if [item["record_hash"] for item, _, _ in entries] != run["pair_input_hashes"]:
        raise ValueError("posthoc exposure paired inventory changed")
    labels = []
    expected = set()
    for item, observation, directory in entries:
        name = item["record_hash"] + ".json"
        expected.add(name)
        path = root / "labels" / name
        if not path.exists():
            raise ValueError("posthoc exposure label missing")
        record = checked(read_json(path), schema="posthoc-exposure-label.v1")
        silent_path = directory / "silent" / "result.json"
        silent = read_json(silent_path)["record"] if silent_path.exists() else {}
        messages = (_prompt(source["bank"], memories[item["memory_id"]], item, silent)
                    if observation["outcome_valid"] else None)
        pre, pre_hash = _pre_x(item)
        if (record["judge_run_hash"] != run["record_hash"]
                or record["pair_input_hash"] != item["record_hash"]
                or record["memory_id"] != item["memory_id"]
                or record["pre_x_hash"] != pre_hash
                or record["silent_record_hash"] != silent.get("record_hash")
                or record["prompt_hash"] != (digest(messages) if messages else None)):
            raise ValueError("posthoc exposure label source identity mismatch")
        if record["status"] not in ("assessed", "invalid_response", "provider_error", "unavailable"):
            raise ValueError("invalid posthoc exposure status")
        if record["response_text"] is not None and record["response_hash"] != digest(record["response_text"]):
            raise ValueError("posthoc exposure response hash mismatch")
        attempt_path = root / "attempts" / name
        if messages is not None and not attempt_path.exists():
            raise ValueError("posthoc exposure teacher attempt missing")
        if messages is None and attempt_path.exists():
            raise ValueError("unavailable pair has an unexpected teacher attempt")
        if attempt_path.exists():
            attempt = checked(read_json(attempt_path))
            if (attempt["judge_run_hash"] != run["record_hash"]
                    or attempt["pair_input_hash"] != item["record_hash"]
                    or attempt["prompt_hash"] != record["prompt_hash"]):
                raise ValueError("posthoc exposure attempt source identity mismatch")
        if record["status"] == "assessed":
            if (record["assessment"] != _assessment(record["response_text"])
                    or record["x_final"] != _final_x(item, record["assessment"])):
                raise ValueError("posthoc exposure final x differs from teacher assessment")
        elif record["x_final"] is not None or record["assessment"] is not None:
            raise ValueError("unavailable exposure label cannot contain final x")
        labels.append(record)
    if {p.name for p in (root / "labels").glob("*.json")} != expected:
        raise ValueError("unregistered posthoc exposure label")
    return seal({"schema_version": "posthoc-exposure-summary.v1", "judge_run_hash": run["record_hash"],
        "labels": labels, "assessed": sum(row["status"] == "assessed" for row in labels),
        "unknown": sum(row["status"] == "assessed" and row["assessment"]["value"] == "?" for row in labels),
        "invalid_or_unavailable": sum(row["status"] != "assessed" for row in labels)})
