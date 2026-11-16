"""Audited paired supervision and fixed-capacity warm-start gate; no deployment."""
from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.common.contracts import digest, read_json
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.dataset import load_policy_dataset
from cave_mem_v3.policy.features import feature_contract, feature_vector
from cave_mem_v3.target_data.collection import _exclusive_json


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("policy models require the policy extra: pip install -e .[policy]") from exc
    return np


@dataclass(frozen=True)
class SupervisedConfig:
    # Kept for legacy cost heads and historical artifact reconstruction.
    l2: float = .01
    gamma: float | None = None
    max_iterations: int = 4000
    tolerance: float = 1e-6
    minimum_groups: int = 2
    group_axis: str = "target_group"
    seed: int = 42
    cost_components: tuple = ("model_calls", "tool_calls", "api_cost", "tool_cost", "wall_seconds")

    def __post_init__(self):
        for name in ("l2", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        if self.gamma is not None and (isinstance(self.gamma, bool) or not isinstance(self.gamma, (int, float)) or not math.isfinite(self.gamma) or self.gamma <= 0):
            raise ValueError("gamma must be explicitly finite and positive")
        for name, minimum in (("max_iterations", 1), ("minimum_groups", 2), ("seed", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError(name + " is invalid")
        if self.group_axis not in ("target_group", "source_target_group"):
            raise ValueError("group_axis must identify target or source-target dependency groups")
        allowed = {"model_calls", "tool_calls", "input_tokens", "output_tokens", "cached_input_tokens", "api_cost", "tool_cost", "wall_seconds"}
        if not isinstance(self.cost_components, (list, tuple)) or not self.cost_components or len(set(self.cost_components)) != len(self.cost_components) or set(self.cost_components) - allowed:
            raise ValueError("explicit unique observed cost components required")


def _config_dict(config):
    result = asdict(config)
    if config.gamma is None:
        result.pop("gamma")
    result["cost_components"] = list(config.cost_components)
    return result


def _weights(groups):
    counts = Counter(groups)
    return [1.0 / (len(counts) * counts[group]) for group in groups]


def _sigmoid(value):
    np = _numpy()
    return np.exp(-np.logaddexp(0, -value))


def _fit_head(matrix, values, groups, *, kind, config):
    np = _numpy()
    base = {"kind": kind, "rows": len(values), "groups": len(set(groups)),
            "group_ids": sorted(set(groups)), "coefficients": None, "status": "unavailable",
            "class_counts": {str(int(v)): values.count(v) for v in set(values)} if kind == "logistic" else None}
    if len(set(groups)) < config.minimum_groups:
        return {**base, "reason": "insufficient_training_groups"}
    if kind == "logistic" and len(set(values)) < 2:
        return {**base, "reason": "single_class_training_labels"}
    x, y, w = np.asarray(matrix, dtype=float), np.asarray(values, dtype=float), np.asarray(_weights(groups))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("head inputs must be finite")
    if kind == "ridge":
        wx, wy = x * np.sqrt(w)[:, None], y * np.sqrt(w)
        if x.shape[0] < x.shape[1]:
            coefficients = wx.T @ np.linalg.solve(wx @ wx.T + config.l2 * np.eye(len(y)), wy)
        else:
            coefficients = np.linalg.solve(wx.T @ wx + config.l2 * np.eye(x.shape[1]), wx.T @ wy)
        iterations, converged = 1, True
        loss = float(.5 * np.sum(w * (x @ coefficients - y) ** 2) + .5 * config.l2 * coefficients @ coefficients)
    else:
        coefficients = np.zeros(x.shape[1])
        # Frobenius bound on curvature is conservative and deterministic.
        step = 1.0 / (.25 * float(np.sum(w[:, None] * x * x)) + config.l2)
        converged = False
        for iterations in range(1, config.max_iterations + 1):
            gradient = x.T @ (w * (_sigmoid(x @ coefficients) - y)) + config.l2 * coefficients
            if float(np.max(np.abs(gradient))) <= config.tolerance:
                converged = True
                break
            coefficients -= step * gradient
        scores = x @ coefficients
        loss = float(np.sum(w * (np.logaddexp(0, scores) - y * scores)) + .5 * config.l2 * coefficients @ coefficients)
    if not np.isfinite(coefficients).all() or not math.isfinite(loss):
        raise ValueError("nonfinite model fit")
    diagnostics = {"iterations": iterations, "converged": converged, "objective": loss}
    if not converged:
        return {**base, "reason": "solver_not_converged", "diagnostics": diagnostics}
    return {**base, "status": "fitted", "reason": None, "coefficients": coefficients.tolist(), "diagnostics": diagnostics}


def _fit_eq7_head(matrix, values, groups, *, kind, gamma, config):
    """Fit one component of Eq. 7 on the same complete paired-row support."""
    np = _numpy()
    values = [float(v) for v in values]
    base = {"kind": kind, "rows": len(values), "groups": len(set(groups)),
            "group_ids": sorted(set(groups)), "coefficients": None, "status": "unavailable",
            "class_counts": {str(int(v)): values.count(v) for v in set(values)} if kind == "logistic" else None}
    if len(set(groups)) < config.minimum_groups:
        return {**base, "reason": "insufficient_joint_paired_groups"}
    if kind == "logistic" and set(values) != {0.0, 1.0}:
        return {**base, "reason": "joint_paired_harm_requires_both_classes"}
    x, y = np.asarray(matrix, dtype=float), np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Eq. 7 inputs must be finite")
    if kind == "ridge":
        coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = x @ coefficients - y
        loss = float(np.mean(residual ** 2))
        diagnostics = {"iterations": 1, "converged": True, "objective": loss,
            "solver": "ordinary_least_squares_minimum_norm_no_penalty"}
    else:
        coefficients = np.zeros(x.shape[1])
        scale = 1.0 / len(y)
        lipschitz = .25 * scale * float(np.linalg.norm(x, ord=2) ** 2)
        if not math.isfinite(lipschitz) or lipschitz <= 0:
            return {**base, "reason": "singular_logistic_design"}
        step = 1.0 / lipschitz
        converged = False
        raw_bce = None
        for iterations in range(1, config.max_iterations + 1):
            scores = x @ coefficients
            probabilities = _sigmoid(scores)
            gradient = scale * (x.T @ (probabilities - y))
            if float(np.max(np.abs(gradient))) <= config.tolerance:
                converged = True
                break
            coefficients -= step * gradient
        scores = x @ coefficients
        raw_bce = float(np.mean(np.logaddexp(0, scores) - y * scores))
        loss = gamma * raw_bce
        diagnostics = {"iterations": iterations, "converged": converged, "objective": loss,
            "mean_bce": raw_bce, "gamma": gamma,
            "solver": "unregularized_batch_gradient_descent"}
        if not converged:
            return {**base, "reason": "unregularized_logistic_not_converged", "diagnostics": diagnostics}
    if not np.isfinite(coefficients).all() or not math.isfinite(loss):
        return {**base, "reason": "unregularized_fit_nonfinite"}
    return {**base, "status": "fitted", "reason": None,
        "coefficients": coefficients.tolist(), "diagnostics": diagnostics}


def _label(row, head):
    labels, masks = row["labels"], row["label_masks"]
    if row["kind"] != "paired":
        return None
    if head == "tau":
        value = labels.get("D") if masks.get("advantage") else None
    elif head == "harm":
        value = labels.get("H") if masks.get("harm") else None
    elif head == "gate":
        value = int(labels["D"] > 0 and labels["H"] == 0) if masks.get("advantage") and masks.get("harm") else None
    else:
        component = head.split(":", 1)[1]
        value = row["costs"].get("delta", {}).get(component) if row["costs"].get("delta_valid", {}).get(component) else None
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("invalid masked training label")
        if head in ("harm", "gate") and value not in (0, 1):
            raise ValueError("binary training label required")
    return value


def _fit_dataset_legacy(dataset, config):
    """Reproduce historical v1 equal-group/L2 models without relabeling them."""
    np = _numpy()
    if dataset["available_roles"] != ["target_train"] or any(row["role"] != "target_train" or row.get("validation_role") is not None for row in dataset["rows"]):
        raise ValueError("base fitting accepts target_train only")
    rows = [row for row in dataset["rows"] if row["kind"] == "paired" and row["label_masks"]["features"]]
    if not rows:
        raise ValueError("no paired frozen features for fitting")
    contract = feature_contract(rows[0]["features"])
    matrix = np.asarray([feature_vector(row["features"], contract) for row in rows])
    groups = [row["grouping"][config.group_axis] for row in rows]
    weights = np.asarray(_weights(groups))
    mean = np.sum(weights[:, None] * matrix, axis=0)
    scale = np.sqrt(np.sum(weights[:, None] * (matrix - mean) ** 2, axis=0))
    scale[scale < 1e-12] = 1.0
    standardized = np.column_stack((np.ones(len(rows)), (matrix - mean) / scale)).tolist()
    heads = {}
    for name in ("tau", "harm", *("cost:" + key for key in config.cost_components)):
        values = [_label(row, name) for row in rows]
        indexes = [i for i, value in enumerate(values) if value is not None]
        heads[name] = _fit_head([standardized[i] for i in indexes], [values[i] for i in indexes],
            [groups[i] for i in indexes], kind="logistic" if name == "harm" else "ridge", config=config)
        heads[name]["sample_ids"] = [rows[i]["sample_id"] for i in indexes]
    return seal({"schema_version": "supervised-policy.v1", "config": _config_dict(config),
        "feature_contract": contract, "normalization": {"mean": mean.tolist(), "scale": scale.tolist(),
            "fit_role": "target_train", "weighting": "equal_group_then_equal_row", "sample_ids": [r["sample_id"] for r in rows]},
        "heads": heads, "capacity": {"family": "linear_on_frozen_features", "warm_gate_capacity": "binary_logistic_same_input_dimension",
            "input_dimension": matrix.shape[1], "parameters_per_head": matrix.shape[1] + 1,
            "intercept_penalized": True, "objective": "group_weighted_mean_loss_plus_l2_over_two"},
        "training": {"sample_groups": {r["sample_id"]: r["grouping"][config.group_axis] for r in rows},
            "dataset_hash": dataset["record_hash"], "conditions": copy.deepcopy(dataset["conditions"]),
            "manifest_hash": dataset["manifest"]["manifest_hash"], "group_axis": config.group_axis,
            "served_identity": copy.deepcopy(dataset["served_identity"]),
            "split_axes": sorted({axis for r in rows for axis in (r["grouping"]["target_axes"] +
                (r["grouping"]["source_axes"] if config.group_axis == "source_target_group" else []))}),
            "group_ids": sorted(set(groups)), "target_axes": sorted({axis for r in rows for axis in r["grouping"]["target_axes"]}),
            "planned_rows": len(dataset["rows"]), "paired_feature_rows": len(rows),
            "single_rows_excluded": sum(r["kind"] == "single" for r in dataset["rows"])},
        "fixture_only": dataset["fixture_only"],
        "cost_scope": "audited paired continuation cost differences; no complete base c0 is inferred",
        "claim_boundary": "Historical v1 equal-group/L2 predictive model. Not the proposal Eq. 7 objective."})


def _fit_dataset(dataset, config):
    """Internal pure fitter; public entry first rebuilds the audited dataset."""
    np = _numpy()
    if dataset["available_roles"] != ["target_train"] or any(row["role"] != "target_train" or row.get("validation_role") is not None for row in dataset["rows"]):
        raise ValueError("base fitting accepts target_train only")
    rows = [row for row in dataset["rows"] if row["kind"] == "paired" and row["label_masks"]["features"]]
    if not rows:
        raise ValueError("no paired frozen features for fitting")
    if config.gamma is None:
        raise ValueError("Eq. 7 training requires an explicit gamma; no proposal value is specified")
    contract = feature_contract(rows[0]["features"])
    matrix = np.asarray([feature_vector(row["features"], contract) for row in rows])
    groups = [row["grouping"][config.group_axis] for row in rows]
    mean = np.mean(matrix, axis=0)
    scale = np.sqrt(np.mean((matrix - mean) ** 2, axis=0))
    scale[scale < 1e-12] = 1.0
    standardized = np.column_stack((np.ones(len(rows)), (matrix - mean) / scale)).tolist()
    heads = {}
    paired_labels = {name: [_label(row, name) for row in rows] for name in ("tau", "harm")}
    joint_indexes = [i for i in range(len(rows)) if all(paired_labels[name][i] is not None for name in ("tau", "harm"))]
    joint_groups = [groups[i] for i in joint_indexes]
    for name in ("tau", "harm"):
        values = [paired_labels[name][i] for i in joint_indexes]
        head = _fit_eq7_head([standardized[i] for i in joint_indexes], values, joint_groups,
            kind="logistic" if name == "harm" else "ridge", gamma=config.gamma, config=config)
        head["sample_ids"] = [rows[i]["sample_id"] for i in joint_indexes]
        head["joint_pair_support"] = len(joint_indexes)
        heads[name] = head
    for name in ("cost:" + key for key in config.cost_components):
        values = [_label(row, name) for row in rows]
        indexes = [i for i, value in enumerate(values) if value is not None]
        heads[name] = _fit_head([standardized[i] for i in indexes], [values[i] for i in indexes],
            [groups[i] for i in indexes], kind="logistic" if name == "harm" else "ridge", config=config)
        heads[name]["sample_ids"] = [rows[i]["sample_id"] for i in indexes]
    return seal({"schema_version": "supervised-policy.v2", "config": _config_dict(config),
        "objective_protocol": "proposal_eq7_shared_complete_pair_support.v1",
        "paired_objective": {"gamma": config.gamma, "gamma_selection": "explicit_config",
            "valid_tau_rows": sum(v is not None for v in paired_labels["tau"]),
            "valid_harm_rows": sum(v is not None for v in paired_labels["harm"]),
            "joint_rows": len(joint_indexes), "joint_group_ids": sorted(set(joint_groups)),
            "tau_sample_ids": [rows[i]["sample_id"] for i in joint_indexes],
            "harm_sample_ids": [rows[i]["sample_id"] for i in joint_indexes],
            "tau_denominator": len(joint_indexes), "harm_denominator": len(joint_indexes),
            "loss": "mean_joint((tau-D)^2) + gamma*mean_joint(binary_cross_entropy(h,H)); no L2 penalty"},
        "feature_contract": contract, "normalization": {"mean": mean.tolist(), "scale": scale.tolist(),
            "fit_role": "target_train", "weighting": "ordinary_sample_mean_variance_for_feature_scaling_only", "sample_ids": [r["sample_id"] for r in rows]},
        "heads": heads, "capacity": {"family": "linear_on_frozen_features", "warm_gate_capacity": "binary_logistic_same_input_dimension",
            "input_dimension": matrix.shape[1], "parameters_per_head": matrix.shape[1] + 1,
            "head_objectives": {"tau": {"loss": "mean_joint_squared_error", "penalty": "none"},
                "harm": {"loss": "gamma_mean_joint_binary_cross_entropy", "penalty": "none"},
                **{"cost:" + key: {"loss": "group_weighted_mean_squared_error", "penalty": "l2_over_two_all_coefficients_including_intercept"}
                    for key in config.cost_components}}},
        "training": {"sample_groups": {r["sample_id"]: r["grouping"][config.group_axis] for r in rows},
            "dataset_hash": dataset["record_hash"], "conditions": copy.deepcopy(dataset["conditions"]),
            "manifest_hash": dataset["manifest"]["manifest_hash"], "group_axis": config.group_axis,
            "served_identity": copy.deepcopy(dataset["served_identity"]),
            "split_axes": sorted({axis for r in rows for axis in (r["grouping"]["target_axes"] +
                (r["grouping"]["source_axes"] if config.group_axis == "source_target_group" else []))}),
            "group_ids": sorted(set(groups)), "target_axes": sorted({axis for r in rows for axis in r["grouping"]["target_axes"]}),
            "planned_rows": len(dataset["rows"]), "paired_feature_rows": len(rows),
            "single_rows_excluded": sum(r["kind"] == "single" for r in dataset["rows"])},
        "fixture_only": dataset["fixture_only"],
        "cost_scope": "audited paired continuation cost differences; no complete base c0 is inferred",
        "claim_boundary": "Predictive software baseline. Shared-bank target grouping is not source holdout. No calibration, deployment or safety guarantee is implied by fitting."})


def _save(path, artifact, sources=()):
    path = Path(path).resolve()
    if path.exists():
        raise ValueError("policy output exists; choose a new artifact")
    for source in sources:
        for key in ("run", "prepared"):
            if key in source and path.is_relative_to(Path(source[key]).resolve()):
                raise ValueError("policy output must be outside raw experiment sources")
    _exclusive_json(path, artifact)


def fit_supervised(dataset_path, *, config=None, output=None):
    if config is None:
        raise ValueError("fit_supervised requires an explicit SupervisedConfig with gamma; the proposal does not specify gamma")
    if not isinstance(config, SupervisedConfig) or config.gamma is None:
        raise ValueError("fit_supervised requires an explicit positive gamma in SupervisedConfig")
    dataset = load_policy_dataset(dataset_path)
    model = _fit_dataset(dataset, config)
    model["training"]["dataset_path"] = str(Path(dataset_path).resolve())
    model["raw_sources"] = copy.deepcopy(dataset["sources"])
    model = seal(model)
    validate_model(model)
    if output is not None:
        _save(output, model, dataset["sources"])
    return model


def reproduce_supervised(model_path, *, dataset_path=None):
    """Rebuild either an Eq. 7 v2 model or a historical equal-group/L2 v1 model."""
    stored = load_supervised(model_path)
    source = dataset_path or stored["training"].get("dataset_path")
    if not source:
        raise ValueError("model reproduction requires its original audited dataset path")
    dataset = load_policy_dataset(source)
    if stored["schema_version"] == "supervised-policy.v2":
        rebuilt = _fit_dataset(dataset, SupervisedConfig(**stored["config"]))
    else:
        if "gamma" in stored["config"]:
            raise ValueError("legacy v1 model cannot declare the Eq. 7 gamma config")
        rebuilt = _fit_dataset_legacy(dataset, SupervisedConfig(**stored["config"]))
    rebuilt["training"]["dataset_path"] = str(Path(source).resolve())
    rebuilt["raw_sources"] = copy.deepcopy(dataset["sources"])
    return seal(rebuilt)


def validate_model(model):
    np = _numpy()
    if not isinstance(model, dict) or model.get("schema_version") not in ("supervised-policy.v1", "supervised-policy.v2"):
        raise ValueError("unsupported supervised model schema")
    checked(model, schema=model["schema_version"])
    config = SupervisedConfig(**model["config"])
    eq7 = model["schema_version"] == "supervised-policy.v2"
    if eq7:
        support = model.get("paired_objective", {})
        if (model.get("objective_protocol") != "proposal_eq7_shared_complete_pair_support.v1"
                or config.gamma is None or support.get("gamma") != config.gamma
                or support.get("gamma_selection") != "explicit_config"
                or support.get("tau_denominator") != support.get("harm_denominator")
                or support.get("tau_denominator") != support.get("joint_rows")
                or support.get("tau_sample_ids") != support.get("harm_sample_ids")
                or support.get("loss") != "mean_joint((tau-D)^2) + gamma*mean_joint(binary_cross_entropy(h,H)); no L2 penalty"):
            raise ValueError("invalid proposal Eq. 7 objective or shared paired support")
        expected_head_objectives = {"tau": {"loss": "mean_joint_squared_error", "penalty": "none"},
            "harm": {"loss": "gamma_mean_joint_binary_cross_entropy", "penalty": "none"},
            **{"cost:" + key: {"loss": "group_weighted_mean_squared_error", "penalty": "l2_over_two_all_coefficients_including_intercept"}
                for key in config.cost_components}}
        capacity = model.get("capacity", {})
        if "intercept_penalized" in capacity or capacity.get("head_objectives") != expected_head_objectives:
            raise ValueError("invalid per-head objective or penalty metadata")
    elif "paired_objective" in model or "objective_protocol" in model:
        raise ValueError("legacy v1 model cannot carry new-objective metadata")
    eq7 = model["schema_version"] == "supervised-policy.v2"
    contract = model["feature_contract"]
    dimension = len(contract["columns"])
    canonical_x = {"schema_version": "pre-treatment-state.v1", "feature_map": {"status": "frozen_text_encoder_plus_structured.v1",
        "encoder_identity": contract["encoder_identity"], "text_embedding": [1.0] * contract["embedding_dimension"],
        "structured_schema_version": "pre-treatment-structured.v1"},
        **{key: {} for key in ("context", "a0", "m", "E", "d", "v", "rho", "b", "e", "retrieval")}}
    if feature_contract(canonical_x) != contract:
        raise ValueError("noncanonical model feature contract")
    if contract.get("schema_version") != "policy-feature-contract.v1" or model["capacity"]["input_dimension"] != dimension or model["capacity"]["parameters_per_head"] != dimension + 1:
        raise ValueError("invalid model feature dimensions")
    norm = model["normalization"]
    expected_weighting = "ordinary_sample_mean_variance_for_feature_scaling_only" if eq7 else "equal_group_then_equal_row"
    if norm.get("weighting") != expected_weighting:
        raise ValueError("normalization weighting does not match supervised model protocol")
    if len(set(norm["sample_ids"])) != len(norm["sample_ids"]):
        raise ValueError("duplicate model training samples")
    if (model["training"]["group_axis"] != config.group_axis
            or set(model["training"]["sample_groups"]) != set(norm["sample_ids"])
            or sorted(set(model["training"]["sample_groups"].values())) != model["training"]["group_ids"]):
        raise ValueError("model training group metadata mismatch")
    if norm.get("fit_role") != "target_train":
        raise ValueError("normalization must be trained only on target_train")
    for name in ("mean", "scale"):
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in norm[name]):
            raise ValueError("normalization requires numeric scalars")
        value = np.asarray(norm[name], dtype=float)
        if value.shape != (dimension,) or not np.isfinite(value).all() or name == "scale" and np.any(value <= 0):
            raise ValueError("invalid frozen normalization")
    expected = {"tau", "harm", *("cost:" + key for key in config.cost_components)}
    if set(model["heads"]) != expected:
        raise ValueError("model head contract mismatch")
    for name, head in model["heads"].items():
        ids = head["sample_ids"]
        if (len(ids) != len(set(ids)) or head["rows"] != len(ids)
                or not set(ids).issubset(norm["sample_ids"])
                or head["group_ids"] != sorted({model["training"]["sample_groups"][sid] for sid in ids})
                or head["groups"] != len(head["group_ids"])):
            raise ValueError("model head support inventory mismatch")
        if head["kind"] != ("logistic" if name == "harm" else "ridge"):
            raise ValueError("model head family mismatch")
        if eq7 and name in ("tau", "harm") and (
                head.get("sample_ids") != model["paired_objective"]["tau_sample_ids"]
                or head.get("joint_pair_support") != model["paired_objective"]["joint_rows"]):
            raise ValueError("Eq. 7 heads must use the same complete paired rows")
        if head["status"] == "fitted":
            if head["groups"] < config.minimum_groups:
                raise ValueError("fitted head lacks independent training groups")
            if head["kind"] == "logistic" and (set(head["class_counts"]) != {"0", "1"}
                    or any(type(v) is not int or v <= 0 for v in head["class_counts"].values())
                    or sum(head["class_counts"].values()) != head["rows"]):
                raise ValueError("fitted binary head lacks both training classes")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in head["coefficients"]):
                raise ValueError("coefficients require numeric scalars")
            coefficients = np.asarray(head["coefficients"], dtype=float)
            if coefficients.shape != (dimension + 1,) or not np.isfinite(coefficients).all():
                raise ValueError("model coefficient dimension/nonfinite mismatch")
        elif head["status"] != "unavailable" or head["coefficients"] is not None:
            raise ValueError("invalid model availability status")
    return model


def load_supervised(path):
    return validate_model(read_json(path))


def predict_supervised(model, x):
    """Read only one frozen pre-treatment x; never consult outcome or task files."""
    np = _numpy()
    validate_model(model)
    vector = np.asarray(feature_vector(x, model["feature_contract"]))
    norm = model["normalization"]
    values = np.concatenate(([1.0], (vector - norm["mean"]) / norm["scale"]))
    if not np.isfinite(values).all():
        raise ValueError("nonfinite standardized features")
    outputs = {}
    for name, head in model["heads"].items():
        if head["status"] == "fitted":
            score = float(values @ np.asarray(head["coefficients"]))
            if not math.isfinite(score):
                raise ValueError("nonfinite linear prediction")
            value = float(_sigmoid(score)) if head["kind"] == "logistic" else score
            if not math.isfinite(value):
                raise ValueError("nonfinite model prediction")
            outputs[name] = {"status": "available", "value": value}
        else:
            outputs[name] = {"status": "unavailable", "value": None, "reason": head["reason"]}
    return seal({"schema_version": "supervised-prediction.v1", "model_hash": model["record_hash"],
        "x_hash": digest(x), "heads": outputs})
