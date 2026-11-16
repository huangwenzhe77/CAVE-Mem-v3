"""Prospective non-restorable validation selection with explicit worst-case constraints."""
from __future__ import annotations
import math
from dataclasses import asdict
from pathlib import Path
from cave_mem_v3.common.contracts import digest, read_json
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.dataset import _response_identities, _served_identity
from cave_mem_v3.policy.frozen import load_frozen_policy, validate_bundle
from cave_mem_v3.policy.rollout import collect_policy_validation, summarize_policy_validation, _tasks
from cave_mem_v3.policy.selection import (SelectionConfig, _common, _resolve, _paths_disjoint,
    _protect_training_inputs, _binding, _start_slot, _check_slot, _task_groups, _group_interval)
from cave_mem_v3.target_data.collection import _close, _exclusive_json


def create_conservative_selection_plan(candidates, *, lock_path, config, output):
    if not isinstance(config, SelectionConfig) or not isinstance(candidates, list) or not candidates:
        raise ValueError("explicit selection config and nonempty prospective candidate list required")
    entries, identifiers, hashes = [], set(), set()
    common = None
    output, lock_path = map(_resolve, (output, lock_path))
    for candidate in candidates:
        if set(candidate) != {"candidate_id", "bundle", "rollout_dir"}:
            raise ValueError("candidate requires id, frozen bundle and future rollout directory")
        identifier = candidate["candidate_id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("unique nonempty candidate IDs required")
        path, rollout = _resolve(candidate["bundle"]), _resolve(candidate["rollout_dir"])
        bundle = load_frozen_policy(path)
        if bundle["record_hash"] in hashes:
            raise ValueError("duplicate frozen candidate policy")
        if common is not None and common != _common(bundle):
            raise ValueError("selection candidates differ in manifest, actor, bank, encoder, sampling or budgets")
        common = _common(bundle)
        identifiers.add(identifier)
        hashes.add(bundle["record_hash"])
        entries.append({"candidate_id": identifier, "bundle_path": path, "bundle": bundle,
            "bundle_hash": bundle["record_hash"], "rollout_dir": rollout})
    entries.sort(key=lambda e: e["candidate_id"])
    protected = [output, lock_path, *(e["rollout_dir"] for e in entries), *(e["bundle_path"] for e in entries)]
    _paths_disjoint(protected)
    _protect_training_inputs([output, lock_path, *(e["rollout_dir"] for e in entries)], [e["bundle"] for e in entries])
    for path in (output, lock_path, *(e["rollout_dir"] for e in entries)):
        if Path(path).exists():
            raise ValueError("plan must be frozen before any reserved output path exists")
    tasks = _tasks(entries[0]["bundle"])
    plan = seal({"schema_version": "conservative-policy-selection-plan.v1", "candidates": entries,
        "lock_path": lock_path, "plan_path": output, "config": asdict(config),
        "conditions": common, "role": "validation", "validation_role": "selection",
        "task_order": [t["task_id"] for t in tasks], "candidate_order": [e["candidate_id"] for e in entries],
        "ranking": "highest equal-task complete rollout mean utility; ties lexicographic candidate_id",
        "audit_protocol": "live_selected_checkpoint_worst_case_audit.v1",
        "risk_cost_alpha_per_candidate_metric": config.alpha / (2 * len(entries)),
        "chronology": "all reserved output paths absent at plan creation; bindings written before collectors; local hashes are provenance checks, not trusted timestamp signatures",
        "claim_boundary": "Full-task return ranks policies; non-restorable live checkpoints audit worst-case effective pi risk and c0+pi*remaining actor calls, not observed individual harm or counterfactual costs. Bootstrap diagnostics have no simultaneous safety guarantee after adaptive selection; test remains locked."})
    _exclusive_json(Path(output), plan)
    return plan


def load_conservative_selection_plan(path):
    plan = checked(read_json(path), schema="conservative-policy-selection-plan.v1")
    config = SelectionConfig(**plan["config"])
    if plan["role"] != "validation" or plan["validation_role"] != "selection" or _resolve(path) != plan["plan_path"]:
        raise ValueError("selection plan role/path mismatch")
    entries = plan["candidates"]
    if not entries or plan["candidate_order"] != sorted(e["candidate_id"] for e in entries) or len(set(plan["candidate_order"])) != len(entries):
        raise ValueError("selection plan candidate schedule mismatch")
    if len({e["bundle_hash"] for e in entries}) != len(entries):
        raise ValueError("duplicate plan policy")
    if plan["risk_cost_alpha_per_candidate_metric"] != config.alpha / (2 * len(entries)) or plan["audit_protocol"] != "live_selected_checkpoint_worst_case_audit.v1":
        raise ValueError("selection audit/confidence rule mismatch")
    if plan["ranking"] != "highest equal-task complete rollout mean utility; ties lexicographic candidate_id":
        raise ValueError("unsupported selection ranking rule")
    paths = [plan["plan_path"], plan["lock_path"]]
    for entry in entries:
        bundle = validate_bundle(entry["bundle"])
        if entry["bundle_hash"] != bundle["record_hash"] or _common(bundle) != plan["conditions"]:
            raise ValueError("candidate differs from frozen plan conditions")
        if _resolve(entry["bundle_path"]) != entry["bundle_path"] or _resolve(entry["rollout_dir"]) != entry["rollout_dir"]:
            raise ValueError("selection path changed or aliased")
        if load_frozen_policy(entry["bundle_path"]) != bundle:
            raise ValueError("frozen bundle file changed after plan creation")
        paths.extend((entry["bundle_path"], entry["rollout_dir"]))
    _paths_disjoint(paths)
    _protect_training_inputs([plan["plan_path"], plan["lock_path"], *(e["rollout_dir"] for e in entries)], [e["bundle"] for e in entries])
    if any(_resolve(p) != p for p in paths):
        raise ValueError("selection plan contains unresolved/aliased paths")
    if plan["task_order"] != [t["task_id"] for t in _tasks(entries[0]["bundle"])]:
        raise ValueError("selection tasks changed")
    return plan


def collect_conservative_selection_plan(plan_path, *, env_factory, actor_factory, encoder):
    plan = load_conservative_selection_plan(plan_path)
    if Path(plan["lock_path"]).exists():
        raise ValueError("selection already locked; no further collection")
    # Refuse any existing slot before beginning another; partial plans are retained.
    for path in (e["rollout_dir"] for e in plan["candidates"]):
        if Path(path).exists():
            raise ValueError("selection output already exists; preserve partial plan and do not retry")
    bundle = plan["candidates"][0]["bundle"]
    if encoder.identity != bundle["encoder_identity"]:
        raise ValueError("selection encoder differs from plan")
    def actor_checked():
        actor = actor_factory()
        if actor.configuration() != bundle["actor_config"]:
            _close(actor)
            raise ValueError("selection actor differs from plan")
        return actor
    for entry in plan["candidates"]:
        binding = _binding(plan, "conservative_rollout", entry)
        _start_slot(entry["rollout_dir"], binding)
        report = collect_policy_validation(entry["bundle_path"], env_factory=env_factory, actor_factory=actor_checked,
            encoder=encoder, output_dir=Path(entry["rollout_dir"]) / "run", config_identity=binding["record_hash"])
        if report["pending_tasks"] or report["attempted_tasks"] != report["planned_tasks"] or report["error_tasks"]:
            return {"status": "incomplete", "plan_hash": plan["record_hash"], "candidate_id": entry["candidate_id"]}
    return {"status": "collected", "plan_hash": plan["record_hash"], "candidates": len(plan["candidates"])}

def evaluate_conservative_selection_plan(plan_path):
    plan = load_conservative_selection_plan(plan_path)
    config = SelectionConfig(**plan["config"])
    groups = _task_groups(plan["conditions"]["manifest"])
    candidates, references = [], []
    observed_identities = []
    environment_reference = None
    for entry in plan["candidates"]:
        report = {"candidate_id": entry["candidate_id"], "bundle_hash": entry["bundle_hash"], "feasible": False,
            "reasons": [], "full_task_mean_utility": None, "full_task_reward_interval": None,
            "risk": None, "actor_calls": None, "checkpoint_audit": []}
        binding = _binding(plan, "conservative_rollout", entry)
        root = Path(entry["rollout_dir"])
        if not _check_slot(root, binding, {"run"}) or not (root / "run/run.json").exists():
            report["reasons"].append("full_rollout_missing")
            candidates.append(report)
            continue
        run = read_json(root / "run/run.json")
        if run.get("role") != "validation" or run.get("validation_role") != "selection" or run.get("bundle") != entry["bundle"] or run.get("config_identity") != binding["record_hash"]:
            raise ValueError("full rollout plan/bundle/role binding mismatch")
        summary = summarize_policy_validation(root / "run")
        identities = []
        for filename in ("preparation.json", "result.json"):
            for path in sorted((root / "run/sessions").glob("*/" + filename)):
                identities.extend(_response_identities(read_json(path)))
        identity = _served_identity(identities)
        observed_identities.append(identity)
        _served_identity(observed_identities)
        report["served_identity"] = identity
        if identity["status"] != "reported":
            report["reasons"].append("actual_served_identity_unknown")
        references.append({"candidate_id": entry["candidate_id"], "run_hash": run["record_hash"], "summary_hash": summary["record_hash"]})
        if run["task_order"] != plan["task_order"]:
            raise ValueError("full rollout task denominator differs from plan")
        environment = {key: run[key] for key in ("environment_configs", "environment_capabilities", "actor_capabilities")}
        if environment_reference is not None and environment != environment_reference:
            raise ValueError("candidate validation environment/actor capabilities differ")
        environment_reference = environment
        report["full_task_denominator"] = {key: summary[key] for key in ("planned_tasks", "attempted_tasks", "observed_outcomes", "unknown_outcomes", "valid_utilities", "pending_tasks", "error_tasks")}
        if summary["valid_utilities"] != len(plan["task_order"]) or summary["observed_outcomes"] != len(plan["task_order"]) or summary["error_tasks"] or summary["pending_tasks"]:
            report["reasons"].append("full_task_outcome_or_utility_incomplete")
        else:
            utilities = [row["utility"] for row in summary["tasks"]]
            report["full_task_mean_utility"] = math.fsum(utilities) / len(utilities)
            interval = _group_interval(utilities, [groups[row["task_id"]] for row in summary["tasks"]], config, config.alpha / 2, equal_task=True)
            report["full_task_reward_interval"] = interval
            if interval["status"] != "available":
                report["reasons"].append("full_task_groups_insufficient")
        risks, calls, audit_groups = [], [], []
        for index, row in enumerate(summary["tasks"]):
            observation = {"task_id": row["task_id"], "status": row["status"], "reason": row["reason"],
                "effective_probability": None, "risk_contribution": None, "actor_call_contribution": None}
            report["checkpoint_audit"].append(observation)
            if entry["bundle"]["policy_config"]["kind"] == "no_memory":
                observation.update(effective_probability=0.0, risk_contribution=0.0, actor_call_contribution=0.0,
                    source="structural_no_memory_no_gate_overhead_not_full_task_cost")
                risks.append(0.0)
                calls.append(0.0)
                audit_groups.append(groups[row["task_id"]])
                continue
            if row["decision"] is None:
                if row["reason"] not in ("no_checkpoint", "invalid_prefix_action"):
                    report["reasons"].append("checkpoint_state_incomplete")
                continue
            directory = root / "run/sessions" / f"{index:06d}-{row['trace_id']}"
            item = read_json(directory / "input.json")
            preparation = read_json(directory / "preparation.json")
            step = preparation["steps"][-1]
            response = step.get("actor_response") or {}
            usage = response.get("usage", {})
            if (step.get("selection", {}).get("selected") is not True
                    or type(usage.get("model_calls")) not in (int, float) or usage["model_calls"] != 1
                    or response.get("action") != item["a0"]):
                report["reasons"].append("cached_call_cost_unverified")
                continue
            remaining = item["remaining_model_calls"]
            if type(remaining) is not int or remaining < 0:
                report["reasons"].append("remaining_call_budget_unknown")
                continue
            decision = row["decision"]  # Already recomputed from frozen x by full raw journal audit.
            probability = decision["effective_probability"]
            risk, cost = probability, 1.0 + probability * remaining
            observation.update(effective_probability=probability, raw_probability=decision["raw_probability"],
                decision_reason=decision["reason"], risk_contribution=risk, actor_call_contribution=cost,
                source="worst_case_h_equals_one_and_remaining_call_bound_not_observed_counterfactual",
                c0_actor_calls=1.0, remaining_model_calls=remaining, selected_response_hash=digest(response),
                input_hash=item["record_hash"], decision_hash=decision["record_hash"])
            risks.append(risk)
            calls.append(cost)
            audit_groups.append(groups[row["task_id"]])
        adjusted = plan["risk_cost_alpha_per_candidate_metric"]
        report["risk"] = _group_interval(risks, audit_groups, config, adjusted)
        report["actor_calls"] = _group_interval(calls, audit_groups, config, adjusted)
        if report["risk"]["status"] == "available":
            risk = report["risk"]
            risk["bootstrap_upper"] = risk["upper"]
            all_structural_silent = entry["bundle"]["policy_config"]["kind"] in ("silent", "no_memory")
            risk["structural_zero_reminder_probability"] = all_structural_silent
            risk["zero_scope"] = ("ordinary actor has no memory intervention or gate overhead" if entry["bundle"]["policy_config"]["kind"] == "no_memory" else "globally silent by frozen policy kind") if all_structural_silent else "sample zeros do not prove globally zero reminder risk"
            risk["finite_support_upper"] = 0.0 if all_structural_silent else min(1.0, risk["mean"] + math.sqrt(math.log(1 / adjusted) / (2 * risk["groups"])))
            risk["upper"] = max(risk["bootstrap_upper"], risk["finite_support_upper"])
            risk["support_correction"] = "bounded group-rate Hoeffding diagnostic under independent groups; structural pi=0 exactly zero; no unconditional safety guarantee"
        for name, limit in (("risk", config.risk_limit), ("actor_calls", config.actor_call_budget)):
            metric = report[name]
            if metric["status"] != "available":
                report["reasons"].append(name + "_support_insufficient")
            elif metric["upper"] > limit:
                report["reasons"].append(name + "_upper_exceeds_limit")
        report["audit_denominator"] = {"planned": len(summary["tasks"]), "selected_frozen": sum(r["decision"] is not None for r in summary["tasks"]), "usable": len(risks)}
        if entry["bundle"]["policy_config"]["kind"] == "no_memory":
            report["constraint_scope"] = "structural zero memory intervention risk and zero extra gate actor calls; ordinary full-task actor costs remain in full rollout"
        report["reasons"] = sorted(set(report["reasons"]))
        report["feasible"] = not report["reasons"]
        candidates.append(report)
    plan_complete = all(c.get("served_identity", {}).get("status") == "reported"
        and c.get("full_task_denominator", {}).get("valid_utilities") == len(plan["task_order"])
        and c.get("full_task_denominator", {}).get("observed_outcomes") == len(plan["task_order"])
        and c.get("full_task_denominator", {}).get("attempted_tasks") == len(plan["task_order"])
        and not c.get("full_task_denominator", {}).get("pending_tasks") and not c.get("full_task_denominator", {}).get("error_tasks") for c in candidates)
    if not plan_complete:
        for candidate in candidates:
            candidate["feasible"] = False
            candidate["reasons"] = sorted(set(candidate["reasons"] + ["registered_plan_incomplete"]))
    viable = sorted((c for c in candidates if c["feasible"]), key=lambda c: (-c["full_task_mean_utility"], c["candidate_id"]))
    chosen = next((e for e in plan["candidates"] if viable and e["candidate_id"] == viable[0]["candidate_id"]), None)
    return seal({"schema_version": "conservative-policy-selection-lock.v1", "plan_hash": plan["record_hash"], "plan_path": plan["plan_path"],
        "config": plan["config"], "registered_plan_complete": plan_complete, "status": "selected" if chosen else "no_selection", "candidates": candidates,
        "winner": {"candidate_id": chosen["candidate_id"], "bundle_hash": chosen["bundle_hash"], "bundle": chosen["bundle"]} if chosen else None,
        "validation_references": references, "audit_protocol": plan["audit_protocol"],
        "test_access": "locked_pending_final_evaluation_protocol", "raw_sources": [e["rollout_dir"] for e in plan["candidates"]],
        "claim_boundary": "Selected by complete-task validation return among empirically feasible candidates. Live frozen checkpoints use worst-case h=1 and remaining-call bounds; no individual H or counterfactual cost is observed. Constraint scope is selected checkpoint actor calls, not complete task costs. Group bootstrap plus alpha division is diagnostic, not a theorem about simultaneous validity after candidate selection. Missing or unknown outcomes are not failures. Test outcomes were not used."})


def select_conservative_policy(plan_path):
    plan = load_conservative_selection_plan(plan_path)
    path = Path(plan["lock_path"])
    if path.exists():
        raise ValueError("selection lock exists; never overwrite or reselect on the same plan")
    result = evaluate_conservative_selection_plan(plan_path)
    _exclusive_json(path, result)
    return result


def load_conservative_selection_lock(path):
    result = checked(read_json(path), schema="conservative-policy-selection-lock.v1")
    plan = load_conservative_selection_plan(result["plan_path"])
    if _resolve(path) != plan["lock_path"] or evaluate_conservative_selection_plan(result["plan_path"]) != result:
        raise ValueError("selection lock differs from original prospective plan and validation evidence")
    return result
