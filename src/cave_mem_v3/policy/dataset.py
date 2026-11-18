"""Read-only, outcome-separated dataset views of audited target experiments.

No actor, environment, encoder, estimator, or policy is constructed here. Paths
are provenance references, never features. Both routes retain their complete
planned denominator; only explicit label masks authorize later model fitting.
"""
from __future__ import annotations

import copy
from collections import Counter
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import _components, validate_manifest
from cave_mem_v3.common.contracts import digest, read_json
from cave_mem_v3.interventions.inputs import checked, load_prepared_run, seal
from cave_mem_v3.interventions.outcomes import summarize_target_pairs
from cave_mem_v3.interventions.single_outcomes import summarize_target_single
from cave_mem_v3.memory_generation.bank import bank_memories
from cave_mem_v3.memory_generation.replay import require_current_evidence
from cave_mem_v3.target_data.collection import _exclusive_json
from cave_mem_v3.target_data.partitions import target_partition


SCHEMA = "policy-dataset.v1"


def _check_feature_fields(feature):
    fields = {"feature_map": {"status", "encoder_identity", "text_query_hash", "text_embedding", "structured_schema_version"},
              "d": {"value", "reason", "metric", "source_query_hash", "target_query_hash", "interpretation"},
              "retrieval": {"status", "similarity_score"},
              "e": {"value", "confidence", "reason", "rule_hash", "rules_hash", "rule_schema_version"}}
    for name, allowed in fields.items():
        if not isinstance(feature.get(name), dict) or set(feature[name]) - allowed:
            raise ValueError("unexpected field in frozen pre-treatment " + name)
    def audit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"Y", "Y0", "Y1", "D", "H"} or key.lower() in {
                    "rescue", "labels", "terminal_success", "future_outcome", "success", "reward",
                    "outcome", "private_verifier", "verifier_answer", "hidden_state", "audit_sources"}:
                    raise ValueError("private/outcome field cannot enter policy features")
                audit(child)
        elif isinstance(value, list):
            for child in value:
                audit(child)
    audit(feature)


def _fixture(value):
    if isinstance(value, dict):
        return value.get("research_role") == "fixture_only" or any(_fixture(v) for v in value.values())
    if isinstance(value, list):
        return any(_fixture(v) for v in value)
    return False


def _response_identities(value):
    """Read identities from already-audited original responses, not labels."""
    result = []
    if isinstance(value, dict):
        response = value.get("actor_response")
        if isinstance(response, dict):
            result.append({"model": response.get("model"), "fingerprint": response.get("fingerprint")})
        if value.get("kind") == "actor.completed":
            response = value["response"]
            result.append({"model": response.get("model"), "fingerprint": response.get("fingerprint")})
        for child in value.values():
            result.extend(_response_identities(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_response_identities(child))
    return result


def _served_identity(identities):
    result = {}
    for key in ("model", "fingerprint"):
        observed = {item[key] for item in identities if item.get(key)}
        if len(observed) > 1:
            raise ValueError("incompatible observed served model/fingerprint across valid training observations")
        result[key] = next(iter(observed), None)
    result["status"] = "reported" if identities and all(i.get("model") and i.get("fingerprint") for i in identities) else "unknown"
    return result


def _descriptor(value):
    if not isinstance(value, dict) or value.get("kind") not in ("paired", "single", "auto"):
        raise ValueError("dataset source must declare paired, single, or auto kind")
    required = {"kind", "run"} | ({"prepared"} if value["kind"] == "paired" else set())
    if set(value) not in (required, required | {"exposure"}) or ("exposure" in value and value["kind"] == "single"):
        raise ValueError("paired sources require run and original prepared directories; single/auto sources require run")
    result = {"kind": value["kind"]}
    for name in set(value) - {"kind"}:
        path = Path(value[name]).resolve(strict=True)
        if not path.is_dir():
            raise ValueError("dataset source paths must be directories")
        result[name] = str(path)
    return result


def _posthoc_exposure(source, run):
    if "exposure" not in source:
        return {}
    from cave_mem_v3.interventions.exposure import summarize_paired_exposure
    root = Path(source["exposure"])
    judge_run = checked(read_json(root / "run.json"), schema="posthoc-exposure-run.v1")
    if (judge_run["source_run_hash"] != run["record_hash"]
            or Path(judge_run["source_run"]).resolve() != Path(source["run"]).resolve()):
        raise ValueError("posthoc exposure labels belong to another target run")
    summary = summarize_paired_exposure(root)
    return {label["pair_input_hash"]: label for label in summary["labels"]}


def _add_posthoc_x(row, label):
    if label is None:
        return row
    row["posthoc_exposure_status"] = label["status"]
    if label["status"] == "assessed":
        row["x_final"] = copy.deepcopy(label["x_final"])
        row["e_observed"] = copy.deepcopy(label["assessment"])
    return row


def _conditions(prepared, bank, kind):
    treatment = prepared.get("pre_treatment", prepared)
    # A different RNG replicate is admissible, but budgets, reward weights,
    # actor/encoder/manifest/bank/rules must stay fixed within this dataset.
    study = {key: value for key, value in prepared["study"].items() if key != "base_seed"}
    shared = {"manifest_hash": prepared["manifest_hash"], "bank_hash": bank["bank_hash"],
              "actor_config": prepared["actor_config"], "study": study,
              "encoder_identity": treatment["encoder_identity"],
              "exposure_rules": treatment.get("exposure_rules"),
              "risk_hazard": prepared["risk_hazard"]}
    protocol = {"kind": kind, "environment_configs": prepared["environment_configs"],
                "environment_capabilities": prepared.get("environment_capabilities"),
                 "protocol": prepared.get("protocol") if kind in ("single", "auto") else
                    "paired_restore_one_reminder_equal_continuation_budget.v1"}
    return shared, protocol


def _axes(task):
    return (["task:" + task["task_id"]] + ["group:" + value for value in task.get("groups", [])]
            + (["group:" + task["group_id"]] if task.get("group_id") else [])
            + (["template:" + task["template_id"]] if task.get("template_id") else []))


def _row(run, prepared, task, trace_id, *, kind, input_hash, state, memory, observation,
         inclusion, checkpoint, costs):
    source = {"memory_id": None, "source_task_id": None, "source_trace_id": None,
              "source_checkpoint_hash": None, "mechanism": None}
    if memory is not None:
        source.update(memory_id=memory.memory_id, source_task_id=memory.source_task_id,
                      source_trace_id=memory.source_trace_id, mechanism=memory.mechanism,
                      source_checkpoint_hash=memory.evidence["plan"]["candidate"]["checkpoint_hash"])
    feature = copy.deepcopy(state["x"]) if state is not None else None
    if feature is not None and (state["x_hash"] != digest(feature)
            or state["a0_hash"] != digest(feature["a0"])):
        raise ValueError("dataset pre-treatment feature identity mismatch")
    if feature is not None and set(feature) != {"schema_version", "context", "a0", "m", "E", "d", "v", "rho", "b", "e", "retrieval", "feature_map"}:
        raise ValueError("unexpected field in frozen pre-treatment feature schema")
    if feature is not None:
        _check_feature_fields(feature)
    valid_y, valid_r = observation["outcome_valid"], observation["utility_valid"]
    if kind == "paired":
        labels = {"Y0": observation["silent_success"], "Y1": observation["remind_success"],
                  "R0": observation.get("utilities", {}).get("silent"),
                  "R1": observation.get("utilities", {}).get("remind"),
                  "D": observation["D"], "H": observation["H"], "rescue": observation["rescue"]}
        masks = {"paired_outcomes": valid_y, "advantage": valid_r, "harm": valid_y,
                 "rescue": valid_y, "observed_reward": False}
        assignment = {"kind": "paired_both_arms", "arm_order": observation.get("order"),
                      "propensity": None}
    else:
        labels = {"Y": observation["observed_success"], "R": observation["utility"]}
        masks = {"paired_outcomes": False, "advantage": False, "harm": False,
                 "rescue": False, "observed_reward": valid_r}
        assignment = {"kind": "randomized_single", "z": observation["z"],
                      "arm": observation["arm"], "propensity": observation["propensity"],
                      "probabilities": observation["b"]}
    masks["features"] = feature is not None
    masks["observed_outcome"] = valid_y
    costs["delta_valid"] = {name: bool(kind == "paired" and valid_y and feature is not None and value is not None)
                            for name, value in costs.get("delta", {}).items()}
    masks["cost_delta"] = bool(costs["delta_valid"] and all(costs["delta_valid"].values()))
    for key in ("advantage", "harm", "rescue", "observed_reward", "paired_outcomes"):
        masks[key] = bool(masks[key] and feature is not None)
    identity = {"kind": kind, "run_hash": run["record_hash"], "trace_id": trace_id,
                "task_id": task["task_id"]}
    return {"sample_id": digest(identity), "kind": kind, "role": prepared["role"],
            "validation_role": prepared.get("validation_role"), "status": observation["status"], "reason": observation["reason"],
            "features": feature, "labels": labels, "label_masks": masks, "assignment": assignment,
            "inclusion_probability": inclusion, "costs": copy.deepcopy(costs),
            "provenance": {**identity, "prepared_run_hash": prepared.get("run_hash"),
                "manifest_hash": prepared["manifest_hash"], "bank_hash": run["bank_hash"],
                "input_hash": input_hash, "x_hash": state["x_hash"] if state else None,
                "a0_hash": state["a0_hash"] if state else None, "checkpoint_hash": checkpoint,
                "group_id": task.get("group_id"), "groups": task.get("groups", []),
                "template_id": task.get("template_id"), **source},
            "grouping": {"target_axes": sorted(set(_axes(task))), "source_axes": []}}


def _paired(source, run):
    prepared, inputs = load_prepared_run(source["prepared"], run["bank"])
    if prepared != run["prepared_run"] or inputs != run["inputs"]:
        raise ValueError("paired dataset source differs from its original prepared traces")
    if any(item.get("reason") == "invalid_prepared_input" for item in inputs):
        raise ValueError("invalid original preparation cannot enter a policy dataset")
    summary = summarize_target_pairs(source["run"])
    tasks = {task["task_id"]: task for task in prepared["manifest"]["tasks"]}
    exposure = _posthoc_exposure(source, run)
    memories = {m.memory_id: m for m in bank_memories(run["bank"])} if run["bank"]["memories"] else {}
    rows, identities = [], []
    for index, (item, observation) in enumerate(zip(inputs, summary["pairs"])):
        state, costs, checkpoint, inclusion = None, {
            "shared": None, "shared_status": "unavailable_not_validated"}, None, None
        # Excluded selected inputs still have a frozen state in the original
        # trace. Keep it without creating a treatment outcome for the exclusion.
        path = Path(source["prepared"]) / "traces" / f"{index:06d}-{item['trace_id']}.json"
        if path.exists() and item["status"] in ("ready", "excluded"):
            trace = read_json(path)
            episode = trace["episode"]
            selection = episode["checkpoint_sampling"]
            costs["shared"] = episode["costs"]
            costs["shared_status"] = "validated"
            if selection.get("status") == "selected":
                step = episode["steps"][-1]
                state = step["pre_treatment_state"]
                checkpoint = step["checkpoint_hash"]
                inclusion = selection["inclusion_probability"]
            if observation["outcome_valid"]:
                identities.extend(_response_identities(episode))
                for arm in ("silent", "remind"):
                    identities.extend(_response_identities(read_json(Path(source["run"]) / "pairs" /
                        item["record_hash"] / arm / "result.json")))
        costs.update(arms=observation.get("arm_costs", {}),
                     deployment=observation.get("deployment_costs", {}),
                     pending=observation.get("pending_attempts", {}),
                     operational=observation.get("operational_costs"))
        delta = {}
        arms = costs["arms"]
        if set(arms) == {"silent", "remind"}:
            for key in arms["silent"]:
                a, b = arms["silent"][key], arms["remind"][key]
                delta[key] = b - a if a is not None and b is not None else None
        costs["delta"] = delta
        row = _row(run, prepared, tasks[item["task_id"]], item["trace_id"], kind="paired",
            input_hash=item["record_hash"], state=state,
            memory=memories.get(state.get("memory_id")) if state else None,
            observation=observation, inclusion=inclusion, checkpoint=checkpoint, costs=costs)
        row["provenance"].update(preparation_trace_path=str(path), trace_hash=item.get("trace_hash"),
                                 raw_file_hash=item.get("raw_file_hash"))
        rows.append(_add_posthoc_x(row, exposure.get(item["record_hash"])))
    return prepared, summary, rows, identities


def _single(source, run):
    summary = summarize_target_single(source["run"])
    tasks = {task["task_id"]: task for task in run["manifest"]["tasks"]}
    memories = {m.memory_id: m for m in bank_memories(run["bank"])} if run["bank"]["memories"] else {}
    rows, identities = [], []
    for index, observation in enumerate(summary["observations"]):
        directory = Path(source["run"]) / "sessions" / f"{index:06d}-{observation['trace_id']}"
        item = read_json(directory / "input.json") if (directory / "input.json").exists() else {}
        state = item.get("pre_treatment_state")
        if observation["outcome_valid"]:
            for journal in ("preparation_events", "events"):
                for path in sorted((directory / journal).glob("*.json")):
                    identities.extend(_response_identities(read_json(path)))
        costs = {"shared": observation["shared_preparation_costs"], "arm": observation["arm_costs"],
                 "observed": observation["observed_costs"], "operational": observation.get("operational_costs")}
        rows.append(_row(run, run, tasks[observation["task_id"]], observation["trace_id"], kind="single",
            input_hash=item.get("record_hash"), state=state,
            memory=memories.get(state.get("memory_id")) if state else None,
            observation=observation, inclusion=observation["inclusion_probability"],
            checkpoint=item.get("public_point_hash"), costs=costs))
    return run, summary, rows, identities


def _auto(source, run):
    from cave_mem_v3.interventions.auto import summarize_target_auto
    summary = summarize_target_auto(source["run"])
    if summary["pending"]:
        raise ValueError("auto target collection is incomplete; preserve its denominator and finish/reconcile it first")
    root = Path(source["run"])
    tasks = {task["task_id"]: task for task in run["manifest"]["tasks"]}
    memories = {m.memory_id: m for m in bank_memories(run["bank"])} if run["bank"]["memories"] else {}
    exposure = _posthoc_exposure(source, run)
    rows, identities = [], []
    for index, entry in enumerate(summary["observations"]):
        task_id = run["task_order"][index]
        observation = entry["observation"]
        trace_id = digest([run["record_hash"], task_id])[:24]
        directory = root / "sessions" / f"{index:06d}-{trace_id}"
        item = read_json(directory / "input.json")
        state = item.get("pre_treatment_state")
        if (directory / "preparation.json").exists():
            identities.extend(_response_identities(read_json(directory / "preparation.json")))
        if entry["kind"] == "paired":
            pair_dir = root / "pairs" / read_json(directory / "route.json")["pair_input_hash"]
            pair_item = read_json(pair_dir / "input.json")
            costs = {"shared": observation.get("shared_preparation_costs"),
                     "shared_status": "validated", "arms": observation.get("arm_costs", {}),
                     "deployment": observation.get("deployment_costs", {}),
                     "pending": observation.get("pending_attempts", {}),
                     "operational": observation.get("operational_costs")}
            arms = costs["arms"]
            costs["delta"] = ({key: arms["remind"][key] - arms["silent"][key]
                if arms["remind"][key] is not None and arms["silent"][key] is not None else None
                for key in arms["silent"]} if set(arms) == {"silent", "remind"} else {})
            for arm in ("silent", "remind"):
                path = pair_dir / arm / "result.json"
                if path.exists():
                    identities.extend(_response_identities(read_json(path)))
            kind, checkpoint = "paired", pair_item["checkpoint_hash"]
        else:
            costs = {"shared": observation["shared_preparation_costs"],
                     "arm": observation["arm_costs"], "observed": observation["observed_costs"],
                     "operational": observation.get("operational_costs")}
            path = directory / "result.json"
            if path.exists():
                identities.extend(_response_identities(read_json(path)))
            kind, checkpoint = "single", item.get("public_point_hash")
        row = _row(run, run, tasks[task_id], trace_id, kind=kind,
            input_hash=item["record_hash"], state=state,
            memory=memories.get(state.get("memory_id")) if state else None,
            observation=observation, inclusion=observation.get("inclusion_probability")
                if kind == "single" else item["selection"].get("inclusion_probability"),
            checkpoint=checkpoint, costs=costs)
        # Keep the actual auto route separate from the legacy row shape. A
        # blocked/excluded `none` route has no randomized single-arm outcome.
        row["route_kind"] = entry["kind"]
        rows.append(_add_posthoc_x(row, exposure.get(pair_item["record_hash"]) if kind == "paired" else None))
    return run, summary, rows, identities


def _group_rows(rows, manifest):
    tasks = {task["task_id"]: task for task in manifest["tasks"]}
    target_components = _components(manifest["tasks"])
    target_ids = {task["task_id"]: digest(sorted(t["task_id"] for t in component))
                  for component in target_components for task in component}
    parents = {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    for row in rows:
        p, grouping = row["provenance"], row["grouping"]
        if p["source_task_id"] is not None:
            grouping["source_axes"] = sorted(set(_axes(tasks[p["source_task_id"]]) +
                ["source_trace:" + p["source_trace_id"]]))
        axes = grouping["target_axes"] + grouping["source_axes"]
        first = find(axes[0])
        for axis in axes[1:]:
            parents[find(axis)] = first
        grouping["target_group"] = target_ids[p["task_id"]]
        grouping["checkpoint_group"] = digest([p["task_id"], p["checkpoint_hash"], p["a0_hash"]])
    components = {}
    for axis in parents:
        components.setdefault(find(axis), []).append(axis)
    for row in rows:
        row["grouping"]["source_target_group"] = digest(sorted(components[find(row["grouping"]["target_axes"][0])]))


def build_policy_dataset(sources, *, role="target_train", validation_role=None):
    """Revalidate sources and return a deterministic dataset; never write inputs.

    Each export accepts exactly one explicit partition. The default remains
    target_train. Validation must name calibration or selection; runs are never
    relabeled and no outcome-based or random-row split is performed.
    """
    partition = target_partition(role, validation_role)
    if not isinstance(sources, (list, tuple)) or not sources:
        raise ValueError("provide at least one paired or single source")
    descriptors = sorted((_descriptor(value) for value in sources), key=lambda x: (x["kind"], x["run"]))
    rows, sources_meta, seen_runs, seen_points = [], [], set(), set()
    fixed = manifest = None
    all_identities, software_only, route_protocols = [], False, {}
    for source in descriptors:
        run = checked(read_json(Path(source["run"]) / "run.json"),
                      schema={"paired": "target-pair-run.v1", "single": "target-single-run.v1",
                              "auto": "target-auto-run.v1"}[source["kind"]])
        prepared = run["prepared_run"] if source["kind"] == "paired" else run
        # Reject the role before any outcome-event reads, including a malicious
        # but consistently rehashed run that claims to be test/validation data.
        observed_partition = target_partition(prepared.get("role"), prepared.get("validation_role"))
        if observed_partition != partition:
            raise ValueError("policy dataset partition mismatch: target_train, validation/calibration and validation/selection must stay separate")
        validate_manifest(prepared["manifest"])
        checked(run["bank"], "bank_hash", "verified-source-bank.v3")
        for memory in bank_memories(run["bank"]) if run["bank"]["memories"] else ():
            require_current_evidence(memory.evidence, replay_pairs=2)
        if run["record_hash"] in seen_runs:
            raise ValueError("duplicate source run would count the same observations twice")
        seen_runs.add(run["record_hash"])
        conditions, protocol = _conditions(prepared, run["bank"], source["kind"])
        if source["kind"] in route_protocols and protocol != route_protocols[source["kind"]]:
            raise ValueError("incompatible environment/protocol within a collection route")
        route_protocols[source["kind"]] = protocol
        if fixed is None:
            fixed, manifest = conditions, prepared["manifest"]
        elif conditions != fixed:
            raise ValueError("incompatible manifest, bank, actor, encoder, study, sampling or exposure conditions")
        prepared, summary, new_rows, identities = {
            "paired": _paired, "single": _single, "auto": _auto}[source["kind"]](source, run)
        all_identities.extend(identities)
        _served_identity(all_identities)
        software_only = software_only or prepared["manifest"]["benchmark"] == "preparation-fixture" or _fixture(conditions) or _fixture(protocol)
        cohort = digest(protocol)
        for row in new_rows:
            p = row["provenance"]
            # Re-execution of one prepared checkpoint is not a new independent
            # training observation. Explicit replicate protocols are future work.
            point = (row["kind"], p["prepared_run_hash"] or p["run_hash"], p["trace_id"])
            if point in seen_points:
                raise ValueError("duplicate prepared checkpoint/rollout across dataset sources")
            seen_points.add(point)
            row["cohort_id"] = cohort
        rows.extend(new_rows)
        sources_meta.append({**source, "run_hash": run["record_hash"], "summary_hash": summary["record_hash"],
                             "cohort_id": cohort, "protocol": protocol,
                             "served_identity": _served_identity(identities),
                             "base_seed": prepared["study"]["base_seed"]})
    _group_rows(rows, manifest)
    rows.sort(key=lambda row: row["sample_id"])
    rows = [seal(row) for row in rows]
    return seal({"schema_version": SCHEMA if role == "target_train" else "policy-dataset.v2",
        **({"partition": partition} if role == "validation" else {}),
        "sources": sources_meta, "conditions": fixed,
        "manifest": manifest, "available_roles": [partition],
        "unavailable_roles": [p for p in ("target_train", "validation/calibration", "validation/selection", "test") if p != partition],
        "fixture_only": software_only, "served_identity": _served_identity(all_identities), "rows": rows,
        "summary": {"planned": len(rows), "by_kind": dict(Counter(r["kind"] for r in rows)),
            "by_route": dict(Counter(r.get("route_kind", r["kind"]) for r in rows)),
            "by_status": dict(Counter(r["status"] for r in rows)),
            "by_reason": dict(Counter(r["reason"] for r in rows if r["reason"])),
            "label_counts": {name: sum(r["label_masks"][name] for r in rows) for name in rows[0]["label_masks"]},
            "target_groups": len({r["grouping"]["target_group"] for r in rows}),
            "source_target_groups": len({r["grouping"]["source_target_group"] for r in rows})},
        "claim_boundary": "Pre-treatment features are live decision inputs. Post-pair x_final/e_observed are result-side teacher annotations, never live decision inputs. Labels, assignment, costs and provenance are separate. Grouped data are not independent task replicates; fixture data are software checks only."})


def write_policy_dataset(sources, output, *, role="target_train", validation_role=None):
    """Export once; source validation completes before any output file is created."""
    path = Path(output).resolve()
    if path.exists():
        raise ValueError("policy dataset output exists; choose a new file")
    for source in sources:
        descriptor = _descriptor(source)
        for name in ("run", "prepared", "exposure"):
            if name in descriptor and path.is_relative_to(Path(descriptor[name])):
                raise ValueError("policy dataset output must be outside raw source directories")
    dataset = build_policy_dataset(sources, role=role, validation_role=validation_role)
    _exclusive_json(path, dataset)
    return dataset


def load_policy_dataset(path, *, role="target_train", validation_role=None):
    """Rebuild from raw sources; a checksum alone cannot authenticate labels."""
    partition = target_partition(role, validation_role)
    dataset = checked(read_json(path), schema=SCHEMA if role == "target_train" else "policy-dataset.v2")
    if dataset["available_roles"] != [partition] or (role == "validation" and dataset.get("partition") != partition):
        raise ValueError("saved policy dataset partition mismatch")
    sources = [{key: source[key] for key in ("kind", "run", "prepared", "exposure") if key in source}
               for source in dataset["sources"]]
    if build_policy_dataset(sources, role=role, validation_role=validation_role) != dataset:
        raise ValueError("saved policy dataset differs from original audited sources")
    return dataset
