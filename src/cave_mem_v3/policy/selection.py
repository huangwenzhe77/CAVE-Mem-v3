"""Prospective validation plans, plan-bound collection and read-only policy selection."""
from __future__ import annotations
import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import _components
from cave_mem_v3.common.contracts import StudySpec, digest, read_json, write_json
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.bandit import _cached_call_audits
from cave_mem_v3.policy.dataset import build_policy_dataset, _response_identities, _served_identity
from cave_mem_v3.policy.frozen import load_frozen_policy, policy_decision, validate_bundle
from cave_mem_v3.policy.rollout import collect_policy_validation, summarize_policy_validation, _tasks
from cave_mem_v3.target_data.collection import collect_target, _close, _exclusive_json
from cave_mem_v3.target_data.sampling import RiskHazard


@dataclass(frozen=True)
class SelectionConfig:
    risk_limit: float
    actor_call_budget: float
    minimum_groups: int = 2
    bootstrap_samples: int = 400
    alpha: float = .1
    seed: int = 42
    reward_aggregation: str = "equal_task"
    group_axis: str = "target_group"
    cost_unit: str = "selected_gate_actor_calls"

    def __post_init__(self):
        for name, lower in (("minimum_groups", 2), ("bootstrap_samples", 2), ("seed", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < lower:
                raise ValueError("invalid selection " + name)
        for name in ("risk_limit", "actor_call_budget", "alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("selection controls must be finite")
        if not 0 <= self.risk_limit <= 1 or self.actor_call_budget < 0 or not 0 < self.alpha < 1:
            raise ValueError("selection thresholds out of range")
        if self.reward_aggregation != "equal_task" or self.group_axis != "target_group" or self.cost_unit != "selected_gate_actor_calls":
            raise ValueError("v1 selection requires equal-task return, target-group audit, and explicit actor-call scope")


def _common(bundle):
    return {k: bundle[k] for k in ("manifest", "manifest_hash", "bank_hash", "actor_config", "encoder_identity", "study", "risk_hazard", "exposure_rules", "exposure_rules_hash")}


def _resolve(path):
    return str(Path(path).expanduser().resolve())


def _paths_disjoint(paths):
    for index, value in enumerate(paths):
        first = Path(value)
        for other in paths[index + 1:]:
            second = Path(other)
            if first == second or first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError("selection paths must not duplicate, alias, or contain one another")


def _protected_roots(bundle):
    roots = set(bundle.get("protected_source_roots", []))
    def visit(value):
        if isinstance(value, dict):
            if value.get("kind") in ("paired", "single") and "run" in value:
                for key in ("run", "prepared"):
                    if key in value:
                        roots.add(_resolve(value[key]))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(bundle.get("artifacts", {}))
    return roots


def _protect_training_inputs(paths, bundles):
    for bundle in bundles:
        for source in _protected_roots(bundle):
            for output in paths:
                a, b = Path(output), Path(source)
                if a == b or a.is_relative_to(b) or b.is_relative_to(a):
                    raise ValueError("selection outputs must stay outside protected raw training/calibration input roots")


def create_selection_plan(candidates, *, audit_dir, lock_path, config, output):
    if not isinstance(config, SelectionConfig) or not isinstance(candidates, list) or not candidates:
        raise ValueError("explicit selection config and nonempty prospective candidate list required")
    entries, identifiers, hashes = [], set(), set()
    common = None
    output, lock_path, audit_dir = map(_resolve, (output, lock_path, audit_dir))
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
    protected = [output, lock_path, audit_dir, *(e["rollout_dir"] for e in entries), *(e["bundle_path"] for e in entries)]
    _paths_disjoint(protected)
    _protect_training_inputs([output, lock_path, audit_dir, *(e["rollout_dir"] for e in entries)], [e["bundle"] for e in entries])
    for path in (output, lock_path, audit_dir, *(e["rollout_dir"] for e in entries)):
        if Path(path).exists():
            raise ValueError("plan must be frozen before any reserved output path exists")
    tasks = _tasks(entries[0]["bundle"])
    plan = seal({"schema_version": "policy-selection-plan.v1", "candidates": entries,
        "audit_dir": audit_dir, "lock_path": lock_path, "plan_path": output, "config": asdict(config),
        "conditions": common, "role": "validation", "validation_role": "selection",
        "task_order": [t["task_id"] for t in tasks], "candidate_order": [e["candidate_id"] for e in entries],
        "ranking": "highest equal-task complete rollout mean utility; ties lexicographic candidate_id",
        "audit_protocol": "independent_same_selection_task_checkpoint_paired_audit.v1",
        "risk_cost_alpha_per_candidate_metric": config.alpha / (2 * len(entries)),
        "chronology": "all reserved output paths absent at plan creation; bindings written before collectors; local hashes are provenance checks, not trusted timestamp signatures",
        "claim_boundary": "Full-task return ranks policies; independent checkpoint pairing audits effective pi*H and c0+pi*delta actor calls. Bootstrap diagnostics have no simultaneous safety guarantee after adaptive selection; test remains locked."})
    _exclusive_json(Path(output), plan)
    return plan


def load_selection_plan(path):
    plan = checked(read_json(path), schema="policy-selection-plan.v1")
    config = SelectionConfig(**plan["config"])
    if plan["role"] != "validation" or plan["validation_role"] != "selection" or _resolve(path) != plan["plan_path"]:
        raise ValueError("selection plan role/path mismatch")
    entries = plan["candidates"]
    if not entries or plan["candidate_order"] != sorted(e["candidate_id"] for e in entries) or len(set(plan["candidate_order"])) != len(entries):
        raise ValueError("selection plan candidate schedule mismatch")
    if len({e["bundle_hash"] for e in entries}) != len(entries):
        raise ValueError("duplicate plan policy")
    if plan["risk_cost_alpha_per_candidate_metric"] != config.alpha / (2 * len(entries)) or plan["audit_protocol"] != "independent_same_selection_task_checkpoint_paired_audit.v1":
        raise ValueError("selection audit/confidence rule mismatch")
    if plan["ranking"] != "highest equal-task complete rollout mean utility; ties lexicographic candidate_id":
        raise ValueError("unsupported selection ranking rule")
    paths = [plan["plan_path"], plan["lock_path"], plan["audit_dir"]]
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
    _protect_training_inputs([plan["plan_path"], plan["lock_path"], plan["audit_dir"], *(e["rollout_dir"] for e in entries)], [e["bundle"] for e in entries])
    if any(_resolve(p) != p for p in paths):
        raise ValueError("selection plan contains unresolved/aliased paths")
    if plan["task_order"] != [t["task_id"] for t in _tasks(entries[0]["bundle"])]:
        raise ValueError("selection tasks changed")
    return plan


def _binding(plan, kind, candidate=None):
    return seal({"schema_version": "selection-slot-binding.v1", "plan_hash": plan["record_hash"],
        "slot": kind, "candidate_id": candidate["candidate_id"] if candidate else None,
        "bundle_hash": candidate["bundle_hash"] if candidate else None,
        "conditions_hash": digest(plan["conditions"]), "task_order": plan["task_order"],
        "role": "validation", "validation_role": "selection"})


def _start_slot(path, binding):
    root = Path(path)
    if root.exists():
        raise ValueError("selection slot already exists; inspect preserved evidence, no automatic retry")
    _exclusive_json(root / "binding.json", binding)


def _check_slot(path, binding, allowed):
    root = Path(path)
    if not root.exists():
        return False
    if not (root / "binding.json").exists() or read_json(root / "binding.json") != binding:
        raise ValueError("validation slot is not bound to this plan before execution")
    if any(p.name not in allowed | {"binding.json"} for p in root.iterdir()):
        raise ValueError("unregistered selection slot output")
    return True


def collect_selection_plan(plan_path, *, env_factory, actor_factory, encoder):
    plan = load_selection_plan(plan_path)
    if Path(plan["lock_path"]).exists():
        raise ValueError("selection already locked; no further collection")
    # Refuse any existing slot before beginning another; partial plans are retained.
    for path in (plan["audit_dir"], *(e["rollout_dir"] for e in plan["candidates"])):
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
        binding = _binding(plan, "rollout", entry)
        _start_slot(entry["rollout_dir"], binding)
        report = collect_policy_validation(entry["bundle_path"], env_factory=env_factory, actor_factory=actor_checked,
            encoder=encoder, output_dir=Path(entry["rollout_dir"]) / "run", config_identity=binding["record_hash"])
        if report["pending_tasks"] or report["attempted_tasks"] != report["planned_tasks"] or report["error_tasks"]:
            return {"status": "incomplete", "plan_hash": plan["record_hash"], "candidate_id": entry["candidate_id"]}
    _start_slot(plan["audit_dir"], _binding(plan, "paired_audit"))
    root = Path(plan["audit_dir"])
    collect_target(bundle["manifest"], role="validation", validation_role="selection", spec=StudySpec(**bundle["study"]),
        env_factory=env_factory, actor_factory=actor_checked, output_dir=root / "prepared", encoder=encoder,
        bank=bundle["bank"], hazard=RiskHazard(**bundle["risk_hazard"]), exposure_rules=bundle["exposure_rules"], prepare_before_action=True,
        config_identity=_binding(plan, "paired_audit")["record_hash"])
    report = collect_target_pairs(root / "prepared", bundle["bank"], env_factory=env_factory,
        actor_factory=actor_checked, output_dir=root / "paired")
    return {"status": "collected" if not report["pending_pairs"] else "incomplete", "plan_hash": plan["record_hash"],
        "paired_outcomes": report["valid_outcomes"], "candidates": len(plan["candidates"])}


def _group_interval(values, groups, config, alpha, *, equal_task=False):
    names = sorted(set(groups))
    if len(names) < config.minimum_groups:
        return {"status": "unavailable", "reason": "insufficient_groups", "groups": len(names), "rows": len(values), "mean": None, "lower": None, "upper": None}
    means = [math.fsum(v for v, g in zip(values, groups) if g == name) / groups.count(name) for name in names]
    counts = [groups.count(name) for name in names]
    rng = random.Random(config.seed)
    draws = []
    for _ in range(config.bootstrap_samples):
        indexes = [rng.randrange(len(names)) for _ in names]
        if equal_task:
            draws.append(math.fsum(means[i] * counts[i] for i in indexes) / sum(counts[i] for i in indexes))
        else:
            draws.append(math.fsum(means[i] for i in indexes) / len(indexes))
    draws.sort()
    mean = math.fsum(values) / len(values) if equal_task else math.fsum(means) / len(means)
    lo = draws[max(0, math.floor(alpha * len(draws)))]
    hi = draws[min(len(draws) - 1, math.ceil((1 - alpha) * len(draws)) - 1)]
    return {"status": "available", "reason": None, "groups": len(names), "rows": len(values), "group_ids": names,
        "mean": mean, "lower": min(mean, lo), "upper": max(mean, hi), "alpha": alpha,
        "method": "fixed_seed_group_percentile_bootstrap_diagnostic",
        "aggregation": "equal_task" if equal_task else "equal_group"}


def _task_groups(manifest):
    return {t["task_id"]: digest(sorted(r["task_id"] for r in group)) for group in _components(manifest["tasks"]) for t in group}


def _load_audit(plan):
    root = Path(plan["audit_dir"])
    if not _check_slot(root, _binding(plan, "paired_audit"), {"prepared", "paired"}):
        return None, "audit_missing"
    if not (root / "prepared/run.json").exists() or not (root / "paired/run.json").exists():
        return None, "audit_incomplete"
    prepared_header = checked(read_json(root / "prepared/run.json"), "run_hash", "target-run.v1")
    if prepared_header.get("config_identity") != _binding(plan, "paired_audit")["record_hash"]:
        raise ValueError("paired raw preparation is not bound to this prospective plan")
    dataset = build_policy_dataset([{"kind": "paired", "run": str(root / "paired"), "prepared": str(root / "prepared")}], role="validation", validation_role="selection")
    bundle = plan["candidates"][0]["bundle"]
    run = read_json(root / "prepared/run.json")
    expected = {"manifest": bundle["manifest"], "manifest_hash": bundle["manifest_hash"], "actor_config": bundle["actor_config"],
        "study": bundle["study"], "risk_hazard": bundle["risk_hazard"], "role": "validation", "validation_role": "selection",
        "task_order": plan["task_order"]}
    if any(run.get(k) != v for k, v in expected.items()) or run["pre_treatment"]["encoder_identity"] != bundle["encoder_identity"] or run["pre_treatment"]["bank_hash"] != bundle["bank_hash"] or run["pre_treatment"].get("exposure_rules") != bundle["exposure_rules"]:
        raise ValueError("paired audit task/budget/actor/bank/encoder identity differs from plan")
    incomplete = any(row["status"] not in ("completed", "excluded") or row["status"] == "completed" and not row["label_masks"]["observed_outcome"] for row in dataset["rows"])
    return dataset, "audit_incomplete" if incomplete else None


def evaluate_selection_plan(plan_path):
    plan = load_selection_plan(plan_path)
    config = SelectionConfig(**plan["config"])
    dataset, audit_error = _load_audit(plan)
    if dataset and dataset["served_identity"]["status"] != "reported":
        audit_error = audit_error or "actual_audit_served_identity_unknown"
    cost_audits = _cached_call_audits(dataset) if dataset else {}
    groups = _task_groups(plan["conditions"]["manifest"])
    candidates, references = [], []
    observed_identities = []
    if dataset:
        observed_identities.append(dataset["served_identity"])
    for entry in plan["candidates"]:
        report = {"candidate_id": entry["candidate_id"], "bundle_hash": entry["bundle_hash"], "feasible": False,
            "reasons": [], "full_task_mean_utility": None, "full_task_reward_interval": None,
            "risk": None, "actor_calls": None, "checkpoint_audit": []}
        binding = _binding(plan, "rollout", entry)
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
        if dataset:
            prepared = read_json(Path(plan["audit_dir"]) / "prepared/run.json")
            if run["environment_configs"] != prepared["environment_configs"] or run["environment_capabilities"] != prepared["environment_capabilities"]:
                raise ValueError("paired audit environment differs from full rollout")
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
        if audit_error:
            report["reasons"].append(audit_error)
        else:
            risks, calls, audit_groups = [], [], []
            for row in dataset["rows"]:
                observation = {"sample_id": row["sample_id"], "task_id": row["provenance"]["task_id"],
                    "status": row["status"], "reason": row["reason"], "effective_probability": None,
                    "risk_contribution": None, "actor_call_contribution": None}
                report["checkpoint_audit"].append(observation)
                if entry["bundle"]["policy_config"]["kind"] == "no_memory":
                    observation.update(effective_probability=0.0, risk_contribution=0.0, actor_call_contribution=0.0,
                        source="structural_no_memory_no_gate_overhead_not_full_task_cost")
                    risks.append(0.0)
                    calls.append(0.0)
                    audit_groups.append(row["grouping"]["target_group"])
                    continue
                if not row["label_masks"]["features"]:
                    if row["reason"] != "no_selection":
                        report["reasons"].append("checkpoint_state_incomplete")
                    continue
                item = seal({"pre_treatment_state": {"x": row["features"]}, "a0_hash": row["provenance"]["a0_hash"],
                    "task_id": row["provenance"]["task_id"]})
                decision = policy_decision(entry["bundle"], item, row["provenance"]["trace_id"])
                probability = decision["effective_probability"]
                observation.update(effective_probability=probability, decision_reason=decision["reason"], raw_probability=decision["raw_probability"])
                cost_audit = cost_audits.get(row["sample_id"], {})
                if cost_audit.get("status") != "verified":
                    report["reasons"].append("cached_call_cost_unverified")
                    continue
                if row["status"] not in ("completed", "excluded"):
                    report["reasons"].append("paired_audit_incomplete")
                    continue
                if probability > 0 and (not row["label_masks"]["harm"] or not row["costs"]["delta_valid"].get("model_calls")):
                    report["reasons"].append("paired_risk_or_call_delta_unknown")
                    continue
                risk = probability * row["labels"]["H"] if probability > 0 else 0.0
                cost = 1.0 + probability * row["costs"]["delta"]["model_calls"] if probability > 0 else 1.0
                observation.update(risk_contribution=risk, actor_call_contribution=cost,
                    source="paired_observation" if probability > 0 else "structural_zero_reminder_probability_not_imputed_H")
                risks.append(risk)
                calls.append(cost)
                audit_groups.append(row["grouping"]["target_group"])
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
            report["audit_denominator"] = {"planned": len(dataset["rows"]), "selected_frozen": sum(r["label_masks"]["features"] for r in dataset["rows"]), "usable": len(risks)}
        if entry["bundle"]["policy_config"]["kind"] == "no_memory":
            report["constraint_scope"] = "structural zero memory intervention risk and zero extra gate actor calls; ordinary full-task actor costs remain in full rollout"
        report["reasons"] = sorted(set(report["reasons"]))
        report["feasible"] = not report["reasons"]
        candidates.append(report)
    plan_complete = audit_error is None and all(c.get("served_identity", {}).get("status") == "reported"
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
    return seal({"schema_version": "policy-selection-lock.v1", "plan_hash": plan["record_hash"], "plan_path": plan["plan_path"],
        "config": plan["config"], "registered_plan_complete": plan_complete, "status": "selected" if chosen else "no_selection", "candidates": candidates,
        "winner": {"candidate_id": chosen["candidate_id"], "bundle_hash": chosen["bundle_hash"], "bundle": chosen["bundle"]} if chosen else None,
        "validation_references": references, "paired_audit_dataset_hash": dataset["record_hash"] if dataset else None,
        "test_access": "locked_pending_final_evaluation_protocol", "raw_sources": [e["rollout_dir"] for e in plan["candidates"]] + [plan["audit_dir"]],
        "claim_boundary": "Selected by complete-task validation return among empirically feasible candidates. Independent checkpoint paired risk/call audit is not live individual harm. Group bootstrap plus alpha division is diagnostic, not a theorem about simultaneous validity after candidate selection. Missing or unknown outcomes are not failures. Test outcomes were not used."})


def select_policy(plan_path):
    plan = load_selection_plan(plan_path)
    path = Path(plan["lock_path"])
    if path.exists():
        raise ValueError("selection lock exists; never overwrite or reselect on the same plan")
    result = evaluate_selection_plan(plan_path)
    _exclusive_json(path, result)
    return result


def load_selection_lock(path):
    result = checked(read_json(path), schema="policy-selection-lock.v1")
    plan = load_selection_plan(result["plan_path"])
    if _resolve(path) != plan["lock_path"] or evaluate_selection_plan(result["plan_path"]) != result:
        raise ValueError("selection lock differs from original prospective plan and validation evidence")
    return result
