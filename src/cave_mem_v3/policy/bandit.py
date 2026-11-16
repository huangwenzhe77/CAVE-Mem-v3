"""Same-capacity constrained contextual bandit training; candidates are not deployed."""
from __future__ import annotations

import copy
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from cave_mem_v3.common.contracts import digest, read_json
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.calibration import CalibrationConfig, calibrate_supervised, predict_calibrated, validate_calibration
from cave_mem_v3.policy.dataset import load_policy_dataset
from cave_mem_v3.policy.dr import load_single_dr
from cave_mem_v3.policy.features import feature_contract, feature_vector
from cave_mem_v3.policy.gate import GateConfig, fit_initial_gate, validate_gate
from cave_mem_v3.policy.supervised import reproduce_supervised, _numpy, _save, _sigmoid, load_supervised


@dataclass(frozen=True)
class BanditConfig:
    route: str
    risk_limit: float
    actor_call_budget: float
    iterations: int = 300
    learning_rate: float = .05
    dual_learning_rate: float = .05
    kl_weight: float = .1
    bootstrap_samples: int = 200
    bootstrap_alpha: float = .1
    minimum_groups: int = 2
    seed: int = 42
    feasibility_tolerance: float = 1e-8
    cost_unit: str = "selected_gate_actor_calls"

    def __post_init__(self):
        if self.route not in ("paired", "single") or self.cost_unit != "selected_gate_actor_calls":
            raise ValueError("explicit paired/single route and supported actor-call budget unit required")
        for name, minimum in (("iterations", 1), ("bootstrap_samples", 2), ("minimum_groups", 2), ("seed", 0)):
            if type(getattr(self, name)) is not int or getattr(self, name) < minimum:
                raise ValueError("invalid " + name)
        for name in ("risk_limit", "actor_call_budget", "learning_rate", "dual_learning_rate", "kl_weight", "bootstrap_alpha", "feasibility_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + " must be finite numeric")
        if not 0 <= self.risk_limit <= 1 or self.actor_call_budget < 0 or self.learning_rate <= 0 or self.dual_learning_rate <= 0 or self.kl_weight < 0 or not 0 < self.bootstrap_alpha < 1 or self.feasibility_tolerance < 0:
            raise ValueError("bandit control outside valid bounds")


def eligibility(x):
    """Pre-treatment deployment mask shared by all training metrics."""
    if x is None:
        return False, "frozen_features_unavailable"
    if x.get("v") != 1 or x.get("m") is None:
        return False, "no_verified_memory"
    budget = x.get("b", {})
    if budget.get("status") != "known":
        return False, "budget_unknown"
    for key in ("model_calls", "environment_steps"):
        value = budget.get("remaining", {}).get(key)
        if type(value) is not int or value < 1:
            return False, "insufficient_budget"
    return True, None


def bootstrap_draws(groups, config):
    rng = random.Random(config.seed)
    return [[rng.randrange(len(groups)) for _ in groups] for _ in range(config.bootstrap_samples)]


def _aggregation(groups, draws):
    np = _numpy()
    names = sorted(set(groups))
    indexes = np.asarray([names.index(group) for group in groups])
    counts = np.bincount(indexes, minlength=len(names))
    bootstrap = np.asarray([np.bincount(draw, minlength=len(names)) / len(names) for draw in draws])
    return {"indexes": indexes, "counts": counts, "bootstrap": bootstrap}


def _group_means(values, groups, aggregation=None):
    np = _numpy()
    if aggregation is None:
        aggregation = _aggregation(groups, [])
    counts = aggregation["counts"]
    result = np.zeros((len(counts), *values.shape[1:]))
    np.add.at(result, aggregation["indexes"], values)
    return result / counts.reshape((len(counts),) + (1,) * (values.ndim - 1))


def empirical_upper(values, derivatives, groups, draws, alpha, aggregation=None):
    """Fixed group resampling quantile and its piecewise subgradient."""
    np = _numpy()
    aggregation = aggregation or _aggregation(groups, draws)
    grouped = _group_means(np.asarray(values), groups, aggregation)
    gradients = _group_means(np.asarray(derivatives), groups, aggregation)
    mean = float(np.mean(grouped))
    samples = aggregation["bootstrap"] @ grouped
    order = np.argsort(samples, kind="stable")
    position = min(len(samples) - 1, math.ceil((1 - alpha) * len(samples)) - 1)
    selected = int(order[position])
    if mean >= float(samples[selected]):
        upper, gradient, source = mean, np.mean(gradients, axis=0), "empirical_mean"
    else:
        upper, gradient, source = float(samples[selected]), aggregation["bootstrap"][selected] @ gradients, "bootstrap:" + str(selected)
    return {"mean": mean, "upper": upper, "gradient": gradient, "active_source": source}


def objective_and_gradient(theta, reference, matrix, advantage, harm, delta_calls, groups, draws,
                           config, risk_dual=0.0, cost_dual=0.0, *, risk_constraint=True, eligible=None, aggregation=None):
    """Full group-weighted Lagrangian; advantage already includes utility costs."""
    np = _numpy()
    theta, reference, matrix = np.asarray(theta), np.asarray(reference), np.asarray(matrix)
    advantage, harm, delta_calls = map(np.asarray, (advantage, harm, delta_calls))
    aggregation = aggregation or _aggregation(groups, draws)
    mask = np.ones(len(matrix)) if eligible is None else np.asarray(eligible, dtype=float)
    logits, reference_logits = matrix @ theta, matrix @ reference
    raw_p = _sigmoid(logits)
    p = raw_p * mask
    derivative = raw_p * (1 - raw_p) * mask
    pi_gradient = derivative[:, None] * matrix
    objective = float(np.mean(_group_means(p * advantage, groups, aggregation)))
    objective_gradient = np.mean(_group_means(pi_gradient * advantage[:, None], groups, aggregation), axis=0)
    # Bernoulli KL from logits is stable even when probabilities round to 0/1.
    kl_rows = mask * (raw_p * (logits - reference_logits) - np.logaddexp(0, logits) + np.logaddexp(0, reference_logits))
    kl = float(np.mean(_group_means(kl_rows, groups, aggregation)))
    kl_gradient = np.mean(_group_means(pi_gradient * (logits - reference_logits)[:, None], groups, aggregation), axis=0)
    risk = empirical_upper(p * harm, pi_gradient * harm[:, None], groups, draws, config.bootstrap_alpha, aggregation)
    cost = empirical_upper(1.0 + p * delta_calls, pi_gradient * delta_calls[:, None], groups, draws, config.bootstrap_alpha, aggregation)
    gradient = objective_gradient - config.kl_weight * kl_gradient - cost_dual * cost["gradient"]
    if risk_constraint:
        gradient -= risk_dual * risk["gradient"]
    lagrangian = objective - config.kl_weight * kl - cost_dual * (cost["upper"] - config.actor_call_budget)
    if risk_constraint:
        lagrangian -= risk_dual * (risk["upper"] - config.risk_limit)
    scalars = [objective, kl, lagrangian, risk["mean"], risk["upper"], cost["mean"], cost["upper"]]
    if not all(math.isfinite(value) for value in scalars) or not np.isfinite(gradient).all() or not np.isfinite(p).all():
        raise ValueError("bandit arithmetic nonfinite; retain input and revise numerical controls")
    return {"objective": objective, "kl": max(0.0, kl), "lagrangian": lagrangian,
        "risk_mean": risk["mean"], "risk_upper": risk["upper"], "risk_upper_source": risk["active_source"],
        "actor_calls_mean": cost["mean"], "actor_calls_upper": cost["upper"], "cost_upper_source": cost["active_source"],
        "mean_remind_probability": float(np.mean(_group_means(p, groups, aggregation))), "gradient": gradient}


def optimize_gate(reference, matrix, advantage, harm, delta_calls, groups, draws, config, *, risk_constraint=True, eligible=None):
    np = _numpy()
    theta = np.asarray(reference, dtype=float).copy()
    risk_dual = cost_dual = 0.0
    history = []
    aggregation = _aggregation(groups, draws)
    for step in range(config.iterations + 1):
        measured = objective_and_gradient(theta, reference, matrix, advantage, harm, delta_calls, groups, draws,
            config, risk_dual, cost_dual, risk_constraint=risk_constraint, eligible=eligible, aggregation=aggregation)
        gradient = measured.pop("gradient")
        history.append({"iteration": step, **measured, "risk_dual": risk_dual, "cost_dual": cost_dual,
            "gradient_norm": float(np.linalg.norm(gradient))})
        if step == config.iterations:
            break
        theta += config.learning_rate * gradient
        risk_dual = max(0.0, risk_dual + config.dual_learning_rate * (measured["risk_upper"] - config.risk_limit)) if risk_constraint else 0.0
        cost_dual = max(0.0, cost_dual + config.dual_learning_rate * (measured["actor_calls_upper"] - config.actor_call_budget))
        if not np.isfinite(theta).all() or not math.isfinite(risk_dual) or not math.isfinite(cost_dual):
            raise ValueError("bandit updates overflow; no candidate produced")
    final = history[-1]
    feasible = ((not risk_constraint or final["risk_upper"] <= config.risk_limit + config.feasibility_tolerance)
                and final["actor_calls_upper"] <= config.actor_call_budget + config.feasibility_tolerance)
    return {"mode": "constrained" if risk_constraint else "no_risk_constraint", "history": history,
        "final_coefficients": theta.tolist(), "status": "candidate" if feasible else "empirically_infeasible",
        "empirical_constraints_pass": feasible, "selected_iteration": config.iterations,
        "selection_rule": "last_iterate_only_no_validation_selection"}


def _cached_call_audits(dataset):
    """Inspect selected original responses only after the dataset's full raw audit."""
    sources = {source["run_hash"]: source for source in dataset["sources"]}
    audits = {}
    for row in dataset["rows"]:
        if not row["label_masks"]["features"]:
            continue
        source = sources[row["provenance"]["run_hash"]]
        if row["kind"] == "paired":
            path = Path(row["provenance"]["preparation_trace_path"])
            trace = read_json(path)
            step = trace["episode"]["steps"][-1]
            selected = step.get("checkpoint_sampling", {}).get("selected")
        else:
            run = read_json(Path(source["run"]) / "run.json")
            index = run["task_order"].index(row["provenance"]["task_id"])
            path = Path(source["run"]) / "sessions" / f"{index:06d}-{row['provenance']['trace_id']}" / "preparation.json"
            preparation = read_json(path)
            step = preparation["steps"][-1]
            selected = step.get("selection", {}).get("selected")
        response = step.get("actor_response") or {}
        usage = response.get("usage", {})
        if (selected is not True or usage.get("model_calls") != 1
                or type(usage.get("model_calls")) not in (int, float)
                or response.get("action") != row["features"]["a0"]):
            audits[row["sample_id"]] = {"status": "unavailable", "reason": "selected_actor_call_not_verified"}
        else:
            audits[row["sample_id"]] = {"status": "verified", "c0_actor_calls": 1.0,
                "selected_response_hash": digest(response), "path": str(path.resolve()),
                "source": "audited_original_selected_actor_response_usage"}
    return audits


def _train_dataset(dataset, model, calibration, warm, config, audits, dr=None):
    validate_gate(warm, model, calibration)
    if dataset["available_roles"] != ["target_train"] or any(r["role"] != "target_train" or r.get("validation_role") is not None for r in dataset["rows"]):
        raise ValueError("bandit updates accept target_train only")
    if dataset["conditions"] != model["training"]["conditions"]:
        raise ValueError("bandit actor/manifest/bank/budget/encoder changed")
    for key in ("model", "fingerprint"):
        a, b = dataset["served_identity"].get(key), model["training"]["served_identity"].get(key)
        if a and b and a != b:
            raise ValueError("bandit actual served identity mismatch")
    if config.route == "paired" and dataset["record_hash"] != model["training"]["dataset_hash"]:
        raise ValueError("paired bandit must use exact supervised training data")
    if config.route == "single":
        if dr is None or dr["dataset_hash"] != dataset["record_hash"] or dr["config"]["group_axis"] != model["config"]["group_axis"]:
            raise ValueError("single bandit requires matching audited out-of-fold DR targets")
        dr_rows = {r["sample_id"]: r for r in dr["rows"]}
    elif dr is not None:
        raise ValueError("paired objective cannot consume single DR targets")
    rows = [r for r in dataset["rows"] if r["kind"] == config.route]
    records, active = [], []
    for row in rows:
        allowed, reason = eligibility(row["features"])
        record = {"sample_id": row["sample_id"], "pre_treatment_eligible": allowed, "reason": reason,
            "group": row["grouping"][model["config"]["group_axis"]], "status": "excluded", "advantage": None,
            "risk_bound": None, "delta_actor_calls": None, "cost_audit": audits.get(row["sample_id"])}
        if not row["label_masks"]["features"] and row["reason"] != "no_selection":
            record["status"] = "blocked"
            reason = "planned_frozen_state_unavailable"
        if row["label_masks"]["features"]:
            record["status"] = "blocked"
            audit = audits.get(row["sample_id"], {})
            if audit.get("status") != "verified" or audit.get("c0_actor_calls") != 1.0:
                reason = "cached_actor_call_cost_unavailable"
            elif not allowed:
                record["status"] = "masked_silent"
                active.append(row)
            elif config.route == "paired":
                prediction = predict_calibrated(model, row["features"], calibration)
                risk = prediction["calibrated_bounds"]["harm"]
                if not row["label_masks"].get("advantage"):
                    reason = "paired_advantage_unavailable"
                elif risk["status"] != "available":
                    reason = "paired_risk_calibration_unavailable"
                elif row["costs"].get("delta_valid", {}).get("model_calls") is not True:
                    reason = "paired_actor_call_delta_unavailable"
                else:
                    record.update(advantage=row["labels"]["D"], risk_bound=risk["upper"],
                        delta_actor_calls=row["costs"]["delta"]["model_calls"])
            else:
                pseudo = dr_rows.get(row["sample_id"])
                if pseudo is None or not pseudo["dr_training_mask"] or pseudo["status"] != "available":
                    reason = "single_oof_dr_support_unavailable"
                else:
                    record.update(advantage=pseudo["raw"]["estimated_dr_advantage"], risk_bound=1.0,
                        delta_actor_calls=float(row["features"]["b"]["remaining"]["model_calls"]))
            if reason is None:
                for key in ("advantage", "risk_bound", "delta_actor_calls"):
                    value = record[key]
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError("invalid bandit target or cost")
                record["status"] = "ready"
                active.append(row)
        record["reason"] = reason
        records.append(record)
    blocked = [r for r in records if r["status"] == "blocked"]
    ready = [r for r in records if r["status"] in ("ready", "masked_silent")]
    groups = [r["group"] for r in ready]
    reason = ("warm_gate_unavailable" if warm["head"]["status"] != "fitted" else
              "incomplete_eligible_support" if blocked else "insufficient_training_groups" if len(set(groups)) < config.minimum_groups else None)
    result = {"schema_version": "constrained-bandit.v1", "config": asdict(config),
        "dataset_hash": dataset["record_hash"], "model_hash": model["record_hash"], "calibration_hash": calibration["record_hash"],
        "warm_gate_hash": warm["record_hash"], "dr_hash": dr["record_hash"] if dr else None,
        "capacity": copy.deepcopy(model["capacity"]), "feature_contract": copy.deepcopy(model["feature_contract"]),
        "normalization": copy.deepcopy(model["normalization"]), "conditions": copy.deepcopy(dataset["conditions"]),
        "served_identity": copy.deepcopy(dataset["served_identity"]), "fixture_only": dataset["fixture_only"] or model["fixture_only"],
        "rows": records, "group_axis": model["config"]["group_axis"], "bootstrap_group_order": sorted(set(groups)),
        "bootstrap_draws": [], "status": "unavailable", "reason": reason, "training": None, "no_risk_comparator": None,
        "denominators": {"planned_route_rows": len(rows), "other_route_rows_excluded": len(dataset["rows"]) - len(rows),
            "pre_treatment_eligible": sum(r["pre_treatment_eligible"] for r in records), "ready": len(ready),
            "frozen_feature_rows": sum(r["label_masks"]["features"] for r in rows),
            "policy_forced_silent_rows": sum(r["status"] == "masked_silent" for r in records),
            "by_reason": dict(Counter(r["reason"] for r in records if r["reason"]))},
        "evaluation_cohort": "all frozen-feature sampled checkpoints; ineligible effective pi=0 remains in group means and c0; no-selection/no-frozen-state rows reported separately, not total rollout or population estimate",
        "cost_contract": {"unit": config.cost_unit, "c0": "one original selected cached draft actor call verified in raw response; local encoder and gate add zero actor requests",
            "delta": "audited signed paired Remind minus Silent model_calls" if config.route == "paired" else "worst-case remaining actor calls; not measured counterfactual cost",
            "scope": "selected-gate overhead plus incremental continuation actor calls; excludes prior prefix and baseline Silent continuation; not total task, latency, tokens or money"},
        "risk_contract": "paired calibrated score-bin group-any-harm upper diagnostic" if config.route == "paired" else "explicit worst-case h=1, no individual H inferred",
        "eligibility": "verified memory and known positive remaining actor-call/environment-step budget; rho/e remain soft; ineligible pi=0 and gradient=0 in every metric but audited c0 remains counted",
        "claim_boundary": "Training candidates only. Fixed group bootstrap quantiles are empirical diagnostics, not a safety or population guarantee. Utility costs are already in D/DR. No validation selection, test access, actor change, or deployment authorization."}
    if reason is None:
        np = _numpy()
        vectors = np.asarray([feature_vector(r["features"], model["feature_contract"]) for r in active])
        norm = model["normalization"]
        matrix = np.column_stack((np.ones(len(active)), (vectors - norm["mean"]) / norm["scale"]))
        draws = bootstrap_draws(sorted(set(groups)), config)
        args = (warm["head"]["coefficients"], matrix, [r["advantage"] if r["pre_treatment_eligible"] else 0.0 for r in ready], [r["risk_bound"] if r["pre_treatment_eligible"] else 0.0 for r in ready],
                [r["delta_actor_calls"] if r["pre_treatment_eligible"] else 0.0 for r in ready], groups, draws, config)
        mask = [r["pre_treatment_eligible"] for r in ready]
        trained = optimize_gate(*args, eligible=mask)
        comparator = optimize_gate(*args, risk_constraint=False, eligible=mask)
        result.update(status="candidate" if trained["empirical_constraints_pass"] else "infeasible",
            reason=None if trained["empirical_constraints_pass"] else "final_empirical_constraints_failed",
            bootstrap_draws=draws, training=trained, no_risk_comparator=comparator)
    return seal(result)


def train_bandit(dataset_path, model_path, calibration_path, warm_gate_path, *, config, dr_path=None, output=None):
    if not isinstance(config, BanditConfig):
        raise ValueError("explicit BanditConfig required")
    dataset = load_policy_dataset(dataset_path)
    model, calibration, warm = load_supervised(model_path), read_json(calibration_path), read_json(warm_gate_path)
    if reproduce_supervised(model_path) != model:
        raise ValueError("supervised model differs from original audited training data")
    if calibrate_supervised(model_path, calibration["calibration"]["dataset_path"], config=CalibrationConfig(**calibration["config"])) != calibration:
        raise ValueError("calibration differs from original held-out data")
    if fit_initial_gate(model_path, calibration_path, model["training"]["dataset_path"], config=GateConfig(**warm["config"])) != warm:
        raise ValueError("warm gate differs from original calibrated training inputs")
    dr = load_single_dr(dr_path) if dr_path else None
    result = _train_dataset(dataset, model, calibration, warm, config, _cached_call_audits(dataset), dr)
    result["inputs"] = {"dataset": str(Path(dataset_path).resolve()), "model": str(Path(model_path).resolve()),
        "calibration": str(Path(calibration_path).resolve()), "warm_gate": str(Path(warm_gate_path).resolve()),
        "dr": str(Path(dr_path).resolve()) if dr_path else None}
    result = seal(result)
    validate_bandit_candidate(result)
    if output is not None:
        _save(output, result, dataset["sources"] + model.get("raw_sources", []) + calibration.get("raw_sources", []))
    return result


def load_bandit(path):
    artifact = checked(read_json(path), schema="constrained-bandit.v1")
    inputs = artifact["inputs"]
    rebuilt = train_bandit(inputs["dataset"], inputs["model"], inputs["calibration"], inputs["warm_gate"],
        config=BanditConfig(**artifact["config"]), dr_path=inputs["dr"])
    if rebuilt != artifact:
        raise ValueError("bandit artifact differs from audited training inputs")
    return artifact


def validate_bandit_candidate(artifact):
    checked(artifact, schema="constrained-bandit.v1")
    config = BanditConfig(**artifact["config"])
    if artifact["status"] not in ("candidate", "infeasible", "unavailable"):
        raise ValueError("invalid bandit status")
    if artifact["status"] == "candidate":
        np = _numpy()
        trained = artifact["training"]
        contract = artifact["feature_contract"]
        dimension = len(contract["columns"])
        canonical_x = {"schema_version": "pre-treatment-state.v1", "feature_map": {"status": "frozen_text_encoder_plus_structured.v1",
            "encoder_identity": contract["encoder_identity"], "text_embedding": [1.0] * contract["embedding_dimension"],
            "structured_schema_version": "pre-treatment-structured.v1"},
            **{key: {} for key in ("context", "a0", "m", "E", "d", "v", "rho", "b", "e", "retrieval")}}
        if feature_contract(canonical_x) != contract or artifact["normalization"].get("fit_role") != "target_train":
            raise ValueError("bandit feature contract/normalization provenance mismatch")
        if (trained["status"] != "candidate" or trained["empirical_constraints_pass"] is not True
                or len(trained["history"]) != config.iterations + 1
                or trained["selected_iteration"] != config.iterations
                or artifact["capacity"]["input_dimension"] != dimension
                or artifact["capacity"]["parameters_per_head"] != dimension + 1):
            raise ValueError("bandit candidate identity/capacity/history mismatch")
        for name in ("mean", "scale"):
            values = artifact["normalization"][name]
            if (len(values) != dimension or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                    or name == "scale" and any(v <= 0 for v in values)):
                raise ValueError("bandit normalization invalid")
        coefficients = trained["final_coefficients"]
        if len(coefficients) != dimension + 1 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in coefficients):
            raise ValueError("bandit coefficients invalid")
        for index, row in enumerate(trained["history"]):
            if row["iteration"] != index:
                raise ValueError("bandit iteration sequence mismatch")
            for key in ("risk_upper", "actor_calls_upper", "risk_dual", "cost_dual", "objective", "kl"):
                if isinstance(row[key], bool) or not isinstance(row[key], (int, float)) or not math.isfinite(row[key]):
                    raise ValueError("bandit metrics nonfinite")
            if row["risk_dual"] < 0 or row["cost_dual"] < 0 or row["kl"] < 0:
                raise ValueError("bandit negative dual/KL")
        final = trained["history"][-1]
        if (final["risk_upper"] > config.risk_limit + config.feasibility_tolerance
                or final["actor_calls_upper"] > config.actor_call_budget + config.feasibility_tolerance):
            raise ValueError("candidate fails frozen empirical constraints")
    return artifact


def predict_bandit(artifact, x):
    """Inspect a training candidate's stochastic gate; never execute it."""
    validate_bandit_candidate(artifact)
    if artifact["status"] != "candidate":
        return {"status": "unavailable", "raw_probability": None, "effective_probability": 0.0,
            "eligible": False, "reason": artifact["reason"], "deployment_authorized": False}
    np = _numpy()
    vector = np.asarray(feature_vector(x, artifact["feature_contract"]))
    norm = artifact["normalization"]
    matrix = np.concatenate(([1.0], (vector - norm["mean"]) / norm["scale"]))
    score = float(matrix @ np.asarray(artifact["training"]["final_coefficients"]))
    if not np.isfinite(matrix).all() or not math.isfinite(score):
        raise ValueError("bandit prediction nonfinite")
    raw = float(_sigmoid(score))
    allowed, reason = eligibility(x)
    return {"status": "candidate_only", "raw_probability": raw, "effective_probability": raw if allowed else 0.0,
        "eligible": allowed, "reason": reason, "deployment_authorized": False}
