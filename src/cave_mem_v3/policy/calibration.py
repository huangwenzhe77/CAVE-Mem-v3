"""Independent grouped validation calibration; bounds are not conditional safety certificates."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.common.contracts import read_json
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.dataset import load_policy_dataset
from cave_mem_v3.policy.supervised import _label, _save, load_supervised, predict_supervised, validate_model


@dataclass(frozen=True)
class CalibrationConfig:
    alpha: float = .1
    minimum_groups: int = 10
    risk_bins: int = 4

    def __post_init__(self):
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("calibration alpha must lie strictly between 0 and 1")
        if type(self.minimum_groups) is not int or self.minimum_groups < 2:
            raise ValueError("calibration requires at least two independent groups")
        if type(self.risk_bins) is not int or not 1 <= self.risk_bins <= 100:
            raise ValueError("risk_bins must be a positive fixed count at most 100")


def _calibrate_dataset(model, dataset, config):
    validate_model(model)
    if dataset["available_roles"] != ["validation/calibration"] or any(r["role"] != "validation" or r.get("validation_role") != "calibration" for r in dataset["rows"]):
        raise ValueError("only validation/calibration may calibrate a fitted model")
    if dataset["conditions"] != model["training"]["conditions"]:
        raise ValueError("calibration manifest/bank/model/budget/encoder conditions mismatch")
    for key in ("model", "fingerprint"):
        training_identity = model["training"]["served_identity"].get(key)
        calibration_identity = dataset["served_identity"].get(key)
        if training_identity and calibration_identity and training_identity != calibration_identity:
            raise ValueError("calibration served model identity mismatch")
    axis = model["config"]["group_axis"]
    rows = [r for r in dataset["rows"] if r["kind"] == "paired" and r["label_masks"]["features"]]
    groups = {r["grouping"][axis] for r in rows}
    axes = {key for r in rows for key in (r["grouping"]["target_axes"] +
            (r["grouping"]["source_axes"] if axis == "source_target_group" else []))}
    if groups.intersection(model["training"]["group_ids"]) or axes.intersection(model["training"]["split_axes"]):
        raise ValueError("training/calibration task/template/dependency groups overlap")
    predictions = [predict_supervised(model, r["features"])["heads"] for r in rows]
    heads = {}
    for name, fitted in model["heads"].items():
        if name == "gate":
            continue
        values = [_label(row, name) for row in rows]
        if fitted["status"] != "fitted":
            heads[name] = {"status": "unavailable", "reason": "base_head_unavailable", "groups": 0}
            continue
        indexes = [i for i, value in enumerate(values) if value is not None]
        if name == "harm":
            bins = []
            for number in range(config.risk_bins):
                by_group = {}
                for i in indexes:
                    score = predictions[i][name]["value"]
                    if min(config.risk_bins - 1, int(score * config.risk_bins)) == number:
                        group = rows[i]["grouping"][axis]
                        by_group[group] = max(by_group.get(group, 0), values[i])
                n = len(by_group)
                available = n >= config.minimum_groups
                rate = sum(by_group.values()) / n if n else None
                upper = min(1.0, rate + math.sqrt(math.log(config.risk_bins / config.alpha) / (2 * n))) if available else 1.0
                bins.append({"index": number, "lower_score": number / config.risk_bins,
                    "upper_score": (number + 1) / config.risk_bins, "groups": n,
                    "group_ids": sorted(by_group), "group_values": by_group, "status": "available" if available else "unavailable",
                    "reason": None if available else "insufficient_calibration_groups",
                    "group_any_harm_rate": rate, "upper": upper})
            heads[name] = {"status": "binned", "method": "fixed_score_bin_group_max_binary_hoeffding_bonferroni",
                "bins": bins, "interpretation": "upper bound on average group-any-harm rate within a score bin under independent groups; not pointwise h(x)"}
        else:
            residuals = {}
            for i in indexes:
                group = rows[i]["grouping"][axis]
                error = abs(values[i] - predictions[i][name]["value"])
                residuals[group] = max(residuals.get(group, 0.0), error)
            n = len(residuals)
            rank = math.ceil((n + 1) * (1 - config.alpha))
            available = n >= config.minimum_groups and rank <= n
            heads[name] = {"status": "available" if available else "unavailable",
                "reason": None if available else "insufficient_calibration_groups_or_quantile_support",
                "method": "split_conformal_group_max_absolute_residual", "groups": n,
                "group_ids": sorted(residuals), "group_values": residuals, "rank": rank,
                "radius": sorted(residuals.values())[rank - 1] if available else None}
        heads[name]["sample_ids"] = [rows[i]["sample_id"] for i in indexes]
    return seal({"schema_version": "policy-calibration.v1", "model_hash": model["record_hash"],
        "config": asdict(config), "heads": heads, "group_axis": axis,
        "calibration": {"sample_groups": {r["sample_id"]: r["grouping"][axis] for r in rows},
            "dataset_hash": dataset["record_hash"], "served_identity": copy.deepcopy(dataset["served_identity"]), "role": "validation/calibration",
            "group_ids": sorted(groups), "split_axes": sorted(axes), "conditions": copy.deepcopy(dataset["conditions"]),
            "planned_rows": len(dataset["rows"]), "paired_feature_rows": len(rows)},
        "fixture_only": model["fixture_only"] or dataset["fixture_only"],
        "claim_boundary": "Held-out empirical group calibration. Coverage needs exchangeable/independent groups and applies separately to each head, not arbitrary shifts or pointwise safety. Shared bank is not a held-out source experiment. Insufficient support remains unavailable."})


def calibrate_supervised(model_path, dataset_path, *, config=None, output=None):
    model = load_supervised(model_path)
    dataset = load_policy_dataset(dataset_path, role="validation", validation_role="calibration")
    result = _calibrate_dataset(model, dataset, config or CalibrationConfig())
    result["calibration"]["dataset_path"] = str(Path(dataset_path).resolve())
    result["raw_sources"] = copy.deepcopy(dataset["sources"]) + model.get("raw_sources", [])
    result = seal(result)
    if output is not None:
        _save(output, result, dataset["sources"] + model.get("raw_sources", []))
    return result


def validate_calibration(calibration, model):
    checked(calibration, schema="policy-calibration.v1")
    config = CalibrationConfig(**calibration["config"])
    if (calibration["model_hash"] != model["record_hash"]
            or calibration["calibration"]["role"] != "validation/calibration"
            or calibration["calibration"]["conditions"] != model["training"]["conditions"]
            or calibration["group_axis"] != model["config"]["group_axis"]
            or set(calibration["heads"]) != set(model["heads"]) - {"gate"}):
        raise ValueError("calibration model/role/conditions mismatch")
    if (set(calibration["calibration"]["group_ids"]).intersection(model["training"]["group_ids"])
            or set(calibration["calibration"]["split_axes"]).intersection(model["training"]["split_axes"])):
        raise ValueError("calibration group overlap")
    for key in ("model", "fingerprint"):
        observed, trained = calibration["calibration"]["served_identity"].get(key), model["training"]["served_identity"].get(key)
        if observed and trained and observed != trained:
            raise ValueError("calibration served identity mismatch")
    for name, head in calibration["heads"].items():
        if not set(head.get("sample_ids", [])).issubset(calibration["calibration"]["sample_groups"]):
            raise ValueError("calibration head sample inventory mismatch")
        if not set(head.get("group_values", {})).issubset(calibration["calibration"]["group_ids"]):
            raise ValueError("calibration head invents independent groups")
        if name == "harm" and head["status"] == "binned":
            if len(head["bins"]) != config.risk_bins:
                raise ValueError("calibration risk bin count mismatch")
            for index, item in enumerate(head["bins"]):
                if item["index"] != index or item["lower_score"] != index / config.risk_bins or item["upper_score"] != (index + 1) / config.risk_bins:
                    raise ValueError("calibration risk bin identity mismatch")
                if not isinstance(item["upper"], (int, float)) or not math.isfinite(item["upper"]) or not 0 <= item["upper"] <= 1:
                    raise ValueError("invalid risk upper bound")
                values = item["group_values"]
                if not set(values).issubset(calibration["calibration"]["group_ids"]):
                    raise ValueError("calibration bin invents independent groups")
                n = len(values)
                if (any(type(v) is not int or v not in (0, 1) for v in values.values())
                        or item["groups"] != n or item["group_ids"] != sorted(values)):
                    raise ValueError("invalid grouped risk calibration support")
                rate = sum(values.values()) / n if n else None
                status = "available" if n >= config.minimum_groups else "unavailable"
                upper = min(1.0, rate + math.sqrt(math.log(config.risk_bins / config.alpha) / (2 * n))) if status == "available" else 1.0
                if item["status"] != status or item["group_any_harm_rate"] != rate or item["upper"] != upper:
                    raise ValueError("risk calibration differs from grouped evidence")
        elif head["status"] == "available":
            radius = head["radius"]
            if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius < 0:
                raise ValueError("invalid calibration radius")
            values = head["group_values"]
            n = len(values)
            rank = math.ceil((n + 1) * (1 - config.alpha))
            if (n < config.minimum_groups or rank > n or head["groups"] != n or head["rank"] != rank
                    or head["group_ids"] != sorted(values)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values.values())
                    or radius != sorted(values.values())[rank - 1]):
                raise ValueError("calibration radius differs from grouped support")
        elif head["status"] != "unavailable":
            raise ValueError("unknown calibration head status")
        if head["status"] == "unavailable" and head.get("radius") is not None:
            raise ValueError("unavailable calibration cannot carry finite radius")
        if model["heads"][name]["status"] != "fitted" and head["status"] != "unavailable":
            raise ValueError("unfitted model head cannot have available calibration")
    return calibration


def predict_calibrated(model, x, calibration):
    result = predict_supervised(model, x)
    validate_calibration(calibration, model)
    bounds = {}
    for name, head in calibration["heads"].items():
        value = result["heads"][name]["value"]
        if name == "harm":
            if head["status"] == "binned" and value is not None:
                item = head["bins"][min(len(head["bins"]) - 1, int(value * len(head["bins"])))]
                bounds[name] = {"status": item["status"], "lower": 0.0, "upper": item["upper"],
                    "group_any_harm_rate": item["group_any_harm_rate"], "groups": item["groups"], "reason": item["reason"]}
            else:
                bounds[name] = {"status": "unavailable", "lower": 0.0, "upper": 1.0, "reason": head.get("reason")}
        elif head["status"] == "available" and value is not None:
            bounds[name] = {"status": "available", "lower": value - head["radius"], "upper": value + head["radius"]}
        else:
            bounds[name] = {"status": "unavailable", "lower": None, "upper": None, "reason": head.get("reason")}
    result.update(calibration_hash=calibration["record_hash"], calibrated_bounds=bounds,
        bound_scope={"tau_and_cost": "groupwise predictive intervals for observed D and cost differences; not conditional-mean confidence intervals",
                     "harm": "group-any-harm score-bin upper rate; not pointwise h(x) safety",
                     "group_axis": calibration["group_axis"], "coverage": "separate per head, under group exchangeability/independence"})
    return seal(result)
