"""Grouped cross-fitting and doubly robust diagnostics from randomized single logs.

OOF arm pseudo-utilities are estimates, never paired observed outcomes. Fixed
candidate probabilities are configuration choices; no policy selection occurs.
"""
from __future__ import annotations

import copy
import math
import random
import platform
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.common.contracts import read_json
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.dataset import load_policy_dataset
from cave_mem_v3.policy.features import feature_contract, feature_vector
from cave_mem_v3.policy.supervised import SupervisedConfig, _fit_head, _numpy, _save, _weights


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be finite numeric")
    return float(value)


def _probability(value, name, *, strict=False):
    value = _finite(value, name)
    if not (0 < value < 1 if strict else 0 <= value <= 1):
        raise ValueError(name + " outside probability bounds")
    return value


@dataclass(frozen=True)
class DRConfig:
    candidate_remind_probability: float
    folds: int = 2
    group_axis: str = "target_group"
    seed: int = 42
    minimum_arm_groups: int = 2
    l2: float = .01
    inverse_weight_caps: tuple = (10.0, 100.0)

    def __post_init__(self):
        _probability(self.candidate_remind_probability, "candidate_remind_probability")
        for name, minimum in (("folds", 2), ("minimum_arm_groups", 2), ("seed", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(name + " invalid")
        if self.group_axis not in ("target_group", "source_target_group"):
            raise ValueError("invalid crossfit group axis")
        if _finite(self.l2, "l2") <= 0:
            raise ValueError("l2 must be positive")
        if not isinstance(self.inverse_weight_caps, (tuple, list)) or len(set(self.inverse_weight_caps)) != len(self.inverse_weight_caps):
            raise ValueError("weight caps must be an explicit unique sequence")
        for cap in self.inverse_weight_caps:
            if _finite(cap, "inverse_weight_cap") < 1:
                raise ValueError("inverse logging weight cap must be at least one")
        object.__setattr__(self, "inverse_weight_caps", tuple(self.inverse_weight_caps))


def _config_dict(config):
    result = asdict(config)
    result["inverse_weight_caps"] = list(config.inverse_weight_caps)
    return result


def dr_scores(reward, z, probabilities, mu, candidate_probability, *, inverse_weight_cap=None):
    """Eq.8, including the separate Silent reference and estimated advantage."""
    reward = _finite(reward, "observed reward")
    if type(z) is not int or z not in (0, 1):
        raise ValueError("observed assignment must be 0 or 1")
    if set(probabilities) != {"silent", "remind"} or set(mu) != {"silent", "remind"}:
        raise ValueError("both arm probabilities and OOF means are required")
    b = {key: _probability(value, key + " logging probability", strict=True) for key, value in probabilities.items()}
    if not math.isclose(sum(b.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("logging probabilities must sum to one")
    means = {key: _finite(value, key + " OOF mean") for key, value in mu.items()}
    p = _probability(candidate_probability, "candidate probability")
    arm = "remind" if z else "silent"
    inverse = 1.0 / b[arm]
    if not math.isfinite(inverse):
        raise ValueError("logging probability causes nonfinite inverse weight")
    weight = inverse
    if inverse_weight_cap is not None:
        cap = _finite(inverse_weight_cap, "inverse weight cap")
        if cap < 1:
            raise ValueError("inverse weight cap must be at least one")
        weight = min(weight, cap)
    pseudo = dict(means)
    pseudo[arm] += weight * (reward - means[arm])
    advantage = pseudo["remind"] - pseudo["silent"]
    value = (1 - p) * pseudo["silent"] + p * pseudo["remind"]
    result = {"estimated_arm_utilities": pseudo, "estimated_dr_advantage": advantage,
        "policy_value": value, "silent_value": pseudo["silent"], "improvement_over_silent": p * advantage,
        "inverse_logging_weight": inverse, "used_inverse_weight": weight,
        "policy_correction_weight": (p if z else 1 - p) * weight,
        "silent_correction_weight": weight if z == 0 else 0.0,
        "clipped": weight != inverse}
    if any(not math.isfinite(v) for v in (value, advantage, *pseudo.values(), result["improvement_over_silent"], result["policy_correction_weight"])):
        raise ValueError("DR arithmetic overflow; preserve input and investigate extreme weights")
    return result


def _eligibility(row):
    if row["kind"] != "single":
        return "paired_route_excluded"
    if not row["label_masks"]["features"]:
        return "frozen_features_unavailable"
    assignment = row["assignment"]
    z = assignment.get("z")
    if z is None:
        return "assignment_unavailable"
    if type(z) is not int or z not in (0, 1):
        raise ValueError("invalid recorded single assignment")
    arm = "remind" if z else "silent"
    probabilities = assignment.get("probabilities")
    if assignment.get("kind") != "randomized_single" or assignment.get("arm") != arm or not isinstance(probabilities, dict) or set(probabilities) != {"silent", "remind"}:
        raise ValueError("single assignment identity mismatch")
    for value in probabilities.values():
        _probability(value, "logging probability", strict=True)
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0, abs_tol=1e-12) or assignment.get("propensity") != probabilities[arm]:
        raise ValueError("logged propensity differs from assigned-arm probability")
    if not row["label_masks"]["observed_reward"]:
        return "observed_reward_unavailable"
    _finite(row["labels"].get("R"), "observed reward")
    return None


def _fold_inventory(rows, config):
    groups = sorted({row["grouping"][config.group_axis] for row in rows})
    # Verify actual shared axes agree with the selected component identity.
    axis_groups = {}
    for row in rows:
        axes = row["grouping"]["target_axes"] + (row["grouping"]["source_axes"] if config.group_axis == "source_target_group" else [])
        for axis in axes:
            group = row["grouping"][config.group_axis]
            if axis in axis_groups and axis_groups[axis] != group:
                raise ValueError("dependent task/template/source axes span different crossfit groups")
            axis_groups[axis] = group
    shuffled = list(groups)
    random.Random(config.seed).shuffle(shuffled)
    return {group: index % config.folds for index, group in enumerate(shuffled)}


def _fit_fold(rows, held_out, inventory, contract, config):
    np = _numpy()
    training = [r for r in rows if inventory[r["grouping"][config.group_axis]] != held_out and r["label_masks"]["features"]]
    norm = None
    standardized = []
    if training:
        matrix = np.asarray([feature_vector(r["features"], contract) for r in training])
        groups = [r["grouping"][config.group_axis] for r in training]
        weights = np.asarray(_weights(groups))
        mean = np.sum(weights[:, None] * matrix, axis=0)
        scale = np.sqrt(np.sum(weights[:, None] * (matrix - mean) ** 2, axis=0))
        scale[scale < 1e-12] = 1.0
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("nonfinite fold normalization")
        standardized = np.column_stack((np.ones(len(training)), (matrix - mean) / scale))
        norm = {"mean": mean.tolist(), "scale": scale.tolist(), "sample_ids": [r["sample_id"] for r in training],
            "group_ids": sorted(set(groups)), "weighting": "equal_group_then_equal_row"}
    heads = {}
    for z, name in enumerate(("silent", "remind")):
        indexes = [i for i, r in enumerate(training) if _eligibility(r) is None and r["assignment"]["z"] == z]
        head = _fit_head([standardized[i] for i in indexes], [training[i]["labels"]["R"] for i in indexes],
            [training[i]["grouping"][config.group_axis] for i in indexes], kind="ridge",
            config=SupervisedConfig(l2=config.l2, minimum_groups=config.minimum_arm_groups, group_axis=config.group_axis, seed=config.seed))
        head["sample_ids"] = [training[i]["sample_id"] for i in indexes]
        heads[name] = head
    return {"fold": held_out, "held_out_groups": sorted(g for g, fold in inventory.items() if fold == held_out),
        "normalization": norm, "heads": heads}


def _oof_means(row, fitted, contract):
    np = _numpy()
    if fitted["normalization"] is None:
        return None, "fold_training_features_unavailable"
    if any(head["status"] != "fitted" for head in fitted["heads"].values()):
        return None, "fold_arm_support_insufficient"
    norm = fitted["normalization"]
    vector = np.asarray(feature_vector(row["features"], contract))
    x = np.concatenate(([1.0], (vector - norm["mean"]) / norm["scale"]))
    means = {arm: float(x @ np.asarray(head["coefficients"])) for arm, head in fitted["heads"].items()}
    if not np.isfinite(x).all() or not all(math.isfinite(v) for v in means.values()):
        raise ValueError("nonfinite out-of-fold prediction")
    return means, None


def _mean(values):
    if not values:
        return None
    maximum = max(abs(value) for value in values)
    if not maximum:
        return 0.0
    return math.fsum(value / maximum / len(values) for value in values) * maximum


def _weight_summary(weights):
    if not weights:
        return {"count": 0, "sum": 0.0, "sum_status": "finite", "max": None, "median": None, "p95": None, "ess": 0.0}
    ordered = sorted(weights)
    maximum = max(weights)
    # Scale first so diagnostics remain stable for very large finite weights.
    scaled = [w / maximum for w in weights] if maximum else [0.0 for w in weights]
    squares = math.fsum(w * w for w in scaled)
    total = math.fsum(scaled) * maximum
    return {"count": len(weights), "sum": total if math.isfinite(total) else None,
        "sum_status": "finite" if math.isfinite(total) else "overflow_unrepresentable", "max": maximum,
        "median": ordered[(len(ordered) - 1) // 2], "p95": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
        "ess": math.fsum(scaled) ** 2 / squares if squares else 0.0}


def _aggregate(rows, key, group_axis):
    valid = [r for r in rows if r["status"] == "available"]
    grouped = {}
    for row in valid:
        grouped.setdefault(row["grouping"][group_axis], []).append(row[key])
    metrics = ("policy_value", "silent_value", "improvement_over_silent", "estimated_dr_advantage")
    means = {name: _mean([_mean([v[name] for v in values]) for values in grouped.values()]) for name in metrics}
    row_means = {name: _mean([r[key][name] for r in valid]) for name in metrics}
    weight_stats = {}
    for name in ("inverse_logging_weight", "used_inverse_weight", "policy_correction_weight", "silent_correction_weight"):
        weight_stats[name] = _weight_summary([r[key][name] for r in valid])
        group_weights = [_mean([v[name] for v in values]) for values in grouped.values()]
        weight_stats[name]["group_ess"] = _weight_summary(group_weights)["ess"]
    return {"status": "available_subset" if valid else "unavailable", "valid_rows": len(valid), "valid_groups": len(grouped),
        "equal_group_mean": means, "row_mean": row_means, "weights": weight_stats,
        "estimand": "available selected checkpoints: equal group then equal row; row mean separately shown; no checkpoint-inclusion weighting",
        "scope": "training diagnostic only, not independent policy selection or a full planned-population estimate",
        "missingness_warning": "No missing-at-random assumption is established. Conditioning on observed rewards/costs or available support can break randomization/positivity; original logging probabilities alone do not identify even the complete-case subset value. Missingness analysis is required before causal/population interpretation."}


def _estimate_dataset(dataset, config):
    if dataset["available_roles"] != ["target_train"] or any(r["role"] != "target_train" or r.get("validation_role") is not None for r in dataset["rows"]):
        raise ValueError("crossfit accepts target_train only")
    rows = [r for r in dataset["rows"] if r["kind"] == "single"]
    if not rows:
        raise ValueError("no single-arm rows; paired estimates must stay separate")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate single sample identity")
    reasons = {r["sample_id"]: _eligibility(r) for r in rows}
    feature_rows = [r for r in rows if r["label_masks"]["features"]]
    contract = feature_contract(feature_rows[0]["features"]) if feature_rows else None
    for row in feature_rows:
        feature_vector(row["features"], contract)
    inventory = _fold_inventory(rows, config)
    sufficient_folds = len(inventory) >= config.folds
    folds = [_fit_fold(rows, number, inventory, contract, config) for number in range(config.folds)] if sufficient_folds and contract else []
    results = []
    for row in rows:
        group, sample = row["grouping"][config.group_axis], row["sample_id"]
        reason = reasons[sample]
        result = {"sample_id": sample, "source_status": row["status"], "source_reason": row["reason"],
            "grouping": copy.deepcopy(row["grouping"]), "fold": inventory[group],
            "features": copy.deepcopy(row["features"]), "input_label_masks": copy.deepcopy(row["label_masks"]),
            "assignment": copy.deepcopy(row["assignment"]), "inclusion_probability": row["inclusion_probability"],
            "observed_reward": row["labels"].get("R"), "oof_means": None, "raw": None, "clipped": {},
            "status": "unavailable", "reason": reason, "dr_training_mask": False}
        if reason is None:
            if not sufficient_folds:
                reason = "insufficient_independent_fold_groups"
            else:
                means, reason = _oof_means(row, folds[inventory[group]], contract)
                result["oof_means"] = means
        if reason is None:
            args = (row["labels"]["R"], row["assignment"]["z"], row["assignment"]["probabilities"], result["oof_means"], config.candidate_remind_probability)
            result["raw"] = dr_scores(*args)
            result["clipped"] = {str(float(cap)): dr_scores(*args, inverse_weight_cap=cap) for cap in config.inverse_weight_caps}
            result.update(status="available", dr_training_mask=True)
        result["reason"] = reason
        results.append(result)
    summary = _aggregate(results, "raw", config.group_axis)
    sensitivity = {}
    for cap in config.inverse_weight_caps:
        key = str(float(cap))
        alternate = [{**row, "sensitivity": row["clipped"].get(key)} for row in results]
        report = _aggregate(alternate, "sensitivity", config.group_axis)
        shifts = {name: report["equal_group_mean"][name] - value if value is not None else None
                  for name, value in summary["equal_group_mean"].items()}
        if any(value is not None and not math.isfinite(value) for value in shifts.values()):
            raise ValueError("unrepresentable clipping sensitivity difference")
        sensitivity[key] = {**report, "signed_shift_from_raw": shifts,
            "absolute_shift_from_raw": {name: abs(value) if value is not None else None for name, value in shifts.items()},
            "clipped_rows": sum(r["clipped"].get(key, {}).get("clipped", False) for r in results),
            "warning": "Truncation changes the estimator and may introduce bias. Observed shifts are sensitivity diagnostics, not identified bias bounds."}
    return seal({"schema_version": "single-dr-crossfit.v1", "config": _config_dict(config),
        "numerical_runtime": {"numpy": _numpy().__version__, "python": platform.python_version()},
        "dataset_hash": dataset["record_hash"], "conditions": copy.deepcopy(dataset["conditions"]),
        "served_identity": copy.deepcopy(dataset["served_identity"]), "fixture_only": dataset["fixture_only"],
        "feature_contract": contract, "fold_assignment": inventory, "folds": folds, "rows": results,
        "summary": {**summary, "planned_single_rows": len(rows), "planned_single_groups": len(inventory),
            "excluded_paired_rows": len(dataset["rows"]) - len(rows),
            "by_reason": dict(Counter(r["reason"] for r in results if r["reason"])),
            "complete_planned_population": len(results) == summary["valid_rows"]},
        "weight_sensitivity": sensitivity,
        "candidate_policy": {"kind": "fixed_constant", "remind_probability": config.candidate_remind_probability,
            "frozen_by": "explicit_configuration_before_dataset_loading",
            "prospective_choice_verified": False,
            "selection_warning": "Software ordering does not prove the researcher chose this candidate independently of outcomes"},
        "claim_boundary": "Only assigned rewards are observed. OOF means, arm pseudo-utilities and estimated_dr_advantage are estimates, not paired D/H/rescue. Missing outcomes or costs remain missing. Fixed-group crossfit does not make an adaptively fitted policy's training value an independent evaluation. Shared source bank is not source holdout."})


def estimate_single_dr(dataset_path, *, config, output=None):
    # Configuration is validated before loading any target outcome.
    if not isinstance(config, DRConfig):
        raise ValueError("explicit DRConfig required before reading outcomes")
    dataset = load_policy_dataset(dataset_path)
    result = _estimate_dataset(dataset, config)
    result["dataset_path"] = str(Path(dataset_path).resolve())
    result["raw_sources"] = copy.deepcopy(dataset["sources"])
    result = seal(result)
    if output is not None:
        _save(output, result, dataset["sources"])
    return result


def load_single_dr(path):
    result = checked(read_json(path), schema="single-dr-crossfit.v1")
    rebuilt = estimate_single_dr(result["dataset_path"], config=DRConfig(**result["config"]))
    if rebuilt != result:
        raise ValueError("DR artifact differs from original audited data and frozen crossfit configuration")
    return result
