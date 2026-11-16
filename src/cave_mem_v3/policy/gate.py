"""Supervised warm start from conservative calibrated labels on training x only."""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.common.contracts import digest, read_json
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.calibration import predict_calibrated, validate_calibration
from cave_mem_v3.policy.dataset import load_policy_dataset
from cave_mem_v3.policy.features import feature_vector
from cave_mem_v3.policy.supervised import SupervisedConfig, _fit_head, _numpy, _save, _sigmoid, load_supervised, validate_model


@dataclass(frozen=True)
class GateConfig:
    risk_threshold: float
    minimum_advantage: float = 0.0
    probability_threshold: float = .5

    def __post_init__(self):
        for name in ("risk_threshold", "minimum_advantage", "probability_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("gate thresholds must be explicit finite numbers")
        if not 0 <= self.risk_threshold <= 1 or self.minimum_advantage < 0 or not 0 < self.probability_threshold < 1:
            raise ValueError("gate thresholds outside valid ranges")


def _conservative_label(prediction, x, config):
    bounds = prediction["calibrated_bounds"]
    if x.get("v") != 1 or x.get("m") is None:
        return 0, "no_verified_memory"
    if any(bounds[name]["status"] != "available" for name in ("tau", "harm")):
        return 0, "calibration_unavailable"
    if bounds["tau"]["lower"] <= config.minimum_advantage:
        return 0, "insufficient_lower_advantage"
    if bounds["harm"]["upper"] > config.risk_threshold:
        return 0, "risk_upper_exceeds_threshold"
    return 1, "calibrated_support"


def _fit_gate_dataset(model, calibration, dataset, config):
    validate_model(model)
    validate_calibration(calibration, model)
    if dataset["available_roles"] != ["target_train"] or dataset["record_hash"] != model["training"]["dataset_hash"]:
        raise ValueError("warm gate must use the exact audited training dataset")
    rows = [r for r in dataset["rows"] if r["kind"] == "paired" and r["label_masks"]["features"]]
    if any(r["role"] != "target_train" or r.get("validation_role") is not None for r in rows):
        raise ValueError("warm gate training role mismatch")
    np = _numpy()
    matrix = np.asarray([feature_vector(r["features"], model["feature_contract"]) for r in rows])
    norm = model["normalization"]
    standardized = np.column_stack((np.ones(len(rows)), (matrix - norm["mean"]) / norm["scale"]))
    labels = [_conservative_label(predict_calibrated(model, r["features"], calibration), r["features"], config) for r in rows]
    head = _fit_head(standardized, [v for v, reason in labels],
        [r["grouping"][model["config"]["group_axis"]] for r in rows], kind="logistic", config=SupervisedConfig(**model["config"]))
    return seal({"schema_version": "supervised-initial-gate.v1", "model_hash": model["record_hash"],
        "calibration_hash": calibration["record_hash"], "training_dataset_hash": dataset["record_hash"],
        "config": asdict(config), "head": head, "feature_contract": copy.deepcopy(model["feature_contract"]),
        "normalization_hash": digest(model["normalization"]), "capacity": copy.deepcopy(model["capacity"]),
        "labels": [{"sample_id": r["sample_id"], "value": value, "reason": reason} for r, (value, reason) in zip(rows, labels)],
        "label_rule": "target_train x only: calibrated tau lower > minimum_advantage and harm upper <= risk_threshold; unavailable -> Silent",
        "fixture_only": model["fixture_only"] or calibration["fixture_only"],
        "claim_boundary": "Eq.9-style supervised warm start. Fixed-bin grouped risk bounds are not pointwise safety guarantees; thresholds are caller choices, not selected research settings. D already includes costs; costs are not subtracted again."})


def fit_initial_gate(model_path, calibration_path, dataset_path, *, config, output=None):
    model = load_supervised(model_path)
    calibration = read_json(calibration_path)
    dataset = load_policy_dataset(dataset_path)
    gate = _fit_gate_dataset(model, calibration, dataset, config)
    if output is not None:
        _save(output, gate, dataset["sources"] + calibration.get("raw_sources", []))
    return gate


def validate_gate(gate, model, calibration):
    checked(gate, schema="supervised-initial-gate.v1")
    validate_model(model)
    validate_calibration(calibration, model)
    GateConfig(**gate["config"])
    if (gate["model_hash"] != model["record_hash"] or gate["calibration_hash"] != calibration["record_hash"]
            or gate["training_dataset_hash"] != model["training"]["dataset_hash"]
            or gate["feature_contract"] != model["feature_contract"] or gate["capacity"] != model["capacity"]
            or gate["normalization_hash"] != digest(model["normalization"])):
        raise ValueError("warm gate model/calibration/training identity mismatch")
    head = gate["head"]
    if head["kind"] != "logistic":
        raise ValueError("warm gate must retain same-capacity logistic family")
    if head["status"] == "fitted":
        np = _numpy()
        if (head["groups"] < model["config"]["minimum_groups"] or head["groups"] != len(head["group_ids"])
                or not set(head["group_ids"]).issubset(model["training"]["group_ids"])
                or set(head["class_counts"]) != {"0", "1"}
                or any(type(v) is not int or v <= 0 for v in head["class_counts"].values())
                or sum(head["class_counts"].values()) != head["rows"]
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in head["coefficients"])):
            raise ValueError("warm gate lacks valid training support")
        coefficients = np.asarray(head["coefficients"], dtype=float)
        if coefficients.shape != (model["capacity"]["parameters_per_head"],) or not np.isfinite(coefficients).all():
            raise ValueError("warm gate coefficient dimension/nonfinite mismatch")
    elif head["status"] != "unavailable" or head["coefficients"] is not None:
        raise ValueError("invalid warm gate availability")
    return gate


def predict_initial_gate(gate, model, calibration, x):
    validate_gate(gate, model, calibration)
    prediction = predict_calibrated(model, x, calibration)
    config = GateConfig(**gate["config"])
    supported, reason = _conservative_label(prediction, x, config)
    probability = None
    if gate["head"]["status"] == "fitted":
        np = _numpy()
        vector = np.asarray(feature_vector(x, model["feature_contract"]))
        norm = model["normalization"]
        z = np.concatenate(([1.0], (vector - norm["mean"]) / norm["scale"]))
        score = float(z @ np.asarray(gate["head"]["coefficients"]))
        if not np.isfinite(z).all() or not math.isfinite(score):
            raise ValueError("nonfinite warm gate prediction")
        probability = float(_sigmoid(score))
    if probability is None:
        reason = "warm_gate_unavailable"
    action = int(bool(supported) and probability is not None and probability >= config.probability_threshold)
    if supported and probability is not None and not action:
        reason = "below_gate_probability_threshold"
    return seal({"schema_version": "initial-gate-prediction.v1", "gate_hash": gate["record_hash"],
        "model_hash": model["record_hash"], "calibration_hash": calibration["record_hash"],
        "x_hash": digest(x), "probability": probability, "initial_gate": action,
        "action": "Remind" if action else "Silent", "reason": reason,
        "prediction": prediction, "scope": "offline supervised initial gate; no actor or environment execution"})
