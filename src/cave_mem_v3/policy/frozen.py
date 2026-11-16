"""Self-contained frozen policy packages and auditable once-per-task decisions."""
from __future__ import annotations
import copy
import math
import random
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import validate_manifest
from cave_mem_v3.common.contracts import StudySpec, digest, plain, read_json, stage_seed
from cave_mem_v3.interventions.inputs import checked, seal
from cave_mem_v3.policy.bandit import eligibility, load_bandit, predict_bandit, validate_bandit_candidate
from cave_mem_v3.policy.calibration import CalibrationConfig, calibrate_supervised, predict_calibrated, validate_calibration
from cave_mem_v3.policy.gate import GateConfig, fit_initial_gate, predict_initial_gate, validate_gate
from cave_mem_v3.policy.supervised import reproduce_supervised, load_supervised, _save
from cave_mem_v3.target_data.partitions import validate_validation_bank
from cave_mem_v3.target_data.pre_treatment_state import validate_exposure_rules
from cave_mem_v3.target_data.sampling import RiskHazard

BASELINES = {"no_memory", "silent", "always_remind", "similarity", "source", "applicability", "exposure", "hard_tristate"}
LEARNED = {"supervised", "bandit", "no_risk_bandit"}


def _policy_config(config):
    if not isinstance(config, dict) or config.get("kind") not in BASELINES | LEARNED:
        raise ValueError("explicit supported frozen policy kind required")
    expected = {"kind"}
    if config["kind"] == "similarity":
        expected.add("similarity_threshold")
        value = config.get("similarity_threshold")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("explicit finite cosine similarity threshold required")
    if config["kind"] in LEARNED:
        expected |= {"risk_threshold", "minimum_advantage"}
        GateConfig(risk_threshold=config.get("risk_threshold"), minimum_advantage=config.get("minimum_advantage"))
    if set(config) != expected:
        raise ValueError("unexpected or missing policy configuration fields")


def freeze_policy(manifest, bank, *, actor_config, encoder_identity, study, hazard, policy_config,
                  exposure_rules=None, model_path=None, calibration_path=None, warm_gate_path=None,
                  bandit_path=None, output=None):
    _policy_config(policy_config)
    validate_manifest(manifest, verify_assets=True)
    validate_validation_bank(manifest, bank)
    validate_exposure_rules(exposure_rules)
    StudySpec(**plain(study))
    RiskHazard(**plain(hazard))
    if not isinstance(actor_config, dict) or not actor_config or not isinstance(encoder_identity, dict) or not encoder_identity:
        raise ValueError("freeze explicit actor and encoder identities")
    artifacts = {}
    sources = []
    kind = policy_config["kind"]
    if kind == "supervised" and bandit_path is not None:
        raise ValueError("supervised bundle does not consume a bandit artifact")
    if kind in LEARNED:
        if not all((model_path, calibration_path, warm_gate_path)):
            raise ValueError("learned bundle needs model, calibration and initial gate")
        model = load_supervised(model_path)
        calibration, gate = read_json(calibration_path), read_json(warm_gate_path)
        if reproduce_supervised(model_path) != model:
            raise ValueError("model not reproduced from audited training data")
        if calibrate_supervised(model_path, calibration["calibration"]["dataset_path"], config=CalibrationConfig(**calibration["config"])) != calibration:
            raise ValueError("calibration not reproduced from original calibration data")
        if fit_initial_gate(model_path, calibration_path, model["training"]["dataset_path"], config=GateConfig(**gate["config"])) != gate:
            raise ValueError("initial gate not reproduced from audited inputs")
        artifacts.update(model=model, calibration=calibration, warm_gate=gate)
        sources.extend(model.get("raw_sources", []) + calibration.get("raw_sources", []))
        if kind in ("bandit", "no_risk_bandit"):
            if not bandit_path:
                raise ValueError("bandit package needs a fully audited candidate artifact")
            bandit = load_bandit(bandit_path)
            if (bandit["model_hash"] != model["record_hash"] or bandit["calibration_hash"] != calibration["record_hash"]
                    or bandit["warm_gate_hash"] != gate["record_hash"]):
                raise ValueError("bandit package artifact identity mismatch")
            artifacts["bandit"] = bandit
            # The bandit loader has just re-audited this dataset; retain only
            # source directory identities for future output-containment guards.
            sources.extend(read_json(bandit["inputs"]["dataset"])["sources"])
    elif any((model_path, calibration_path, warm_gate_path, bandit_path)):
        raise ValueError("component baselines cannot silently incorporate learned artifacts")
    bundle = seal({"schema_version": "frozen-policy-bundle.v1", "manifest": copy.deepcopy(manifest),
        "manifest_hash": manifest["manifest_hash"], "bank": copy.deepcopy(bank), "bank_hash": bank["bank_hash"],
        "actor_config": copy.deepcopy(actor_config), "encoder_identity": copy.deepcopy(encoder_identity),
        "study": plain(study), "risk_hazard": plain(hazard), "exposure_rules": copy.deepcopy(exposure_rules),
        "exposure_rules_hash": digest(exposure_rules) if exposure_rules is not None else None,
        "policy_config": copy.deepcopy(policy_config), "artifacts": artifacts,
        "protected_source_roots": sorted({str(Path(source[key]).resolve()) for source in sources for key in ("run", "prepared") if key in source}),
        "decision_protocol": "ordinary_actor_no_retrieval_no_gate_full_budget.v1" if kind == "no_memory" else "one_selected_draft_then_one_live_arm.v1",
        "validation_required": True, "inherited_training_feasibility": False,
        "component_baseline_scope": "individual declared mechanisms only, not reproductions of external paper systems",
        "guard_scope": "same frozen uncertainty postguard for all learned policies, including no-risk comparator; wrapper changes training policy and may change cost feasibility",
        "scope": "selection-validation execution package; test requires a later independent evaluation lock"})
    validate_bundle(bundle)
    if output is not None:
        _save(output, bundle, sources)
    return bundle


def _candidate_for_mode(artifact, kind):
    if kind != "no_risk_bandit":
        return artifact
    bandit = copy.deepcopy(artifact)
    comparator = bandit.get("no_risk_comparator")
    if comparator and comparator["status"] == "candidate":
        bandit.update(status="candidate", reason=None, training=comparator)
        bandit["config"]["risk_limit"] = 1.0
    else:
        bandit.update(status="unavailable", reason="no_risk_comparator_unavailable")
    return seal(bandit)


def validate_bundle(bundle):
    checked(bundle, schema="frozen-policy-bundle.v1")
    _policy_config(bundle["policy_config"])
    expected_protocol = "ordinary_actor_no_retrieval_no_gate_full_budget.v1" if bundle["policy_config"]["kind"] == "no_memory" else "one_selected_draft_then_one_live_arm.v1"
    if bundle.get("decision_protocol") != expected_protocol:
        raise ValueError("unsupported frozen decision protocol")
    validate_manifest(bundle["manifest"])
    if bundle["manifest_hash"] != bundle["manifest"]["manifest_hash"] or bundle["bank_hash"] != bundle["bank"]["bank_hash"]:
        raise ValueError("bundle manifest/bank identity mismatch")
    validate_validation_bank(bundle["manifest"], bundle["bank"])
    StudySpec(**bundle["study"])
    RiskHazard(**bundle["risk_hazard"])
    validate_exposure_rules(bundle["exposure_rules"])
    if bundle["exposure_rules_hash"] != (digest(bundle["exposure_rules"]) if bundle["exposure_rules"] is not None else None):
        raise ValueError("bundle exposure rules changed")
    if bundle["validation_required"] is not True or bundle["inherited_training_feasibility"] is not False:
        raise ValueError("wrapped policy must require fresh validation")
    kind = bundle["policy_config"]["kind"]
    artifacts = bundle["artifacts"]
    expected = {"model", "calibration", "warm_gate"} | ({"bandit"} if "bandit" in kind else set()) if kind in LEARNED else set()
    if set(artifacts) != expected:
        raise ValueError("bundle learned artifact inventory mismatch")
    if kind in LEARNED:
        model, cal, gate = (artifacts[name] for name in ("model", "calibration", "warm_gate"))
        validate_gate(gate, model, cal)
        conditions = model["training"]["conditions"]
        expected_conditions = {"manifest_hash": bundle["manifest_hash"], "bank_hash": bundle["bank_hash"],
            "actor_config": bundle["actor_config"], "encoder_identity": bundle["encoder_identity"],
            "risk_hazard": bundle["risk_hazard"], "exposure_rules": bundle["exposure_rules"],
            "study": {k: v for k, v in bundle["study"].items() if k != "base_seed"}}
        if conditions != expected_conditions:
            raise ValueError("bundle execution conditions differ from fitted model")
        if "bandit" in artifacts:
            bandit = artifacts["bandit"]
            validate_bandit_candidate(bandit)
            validate_bandit_candidate(_candidate_for_mode(bandit, kind))
            if bandit["model_hash"] != model["record_hash"] or bandit["calibration_hash"] != cal["record_hash"] or bandit["warm_gate_hash"] != gate["record_hash"]:
                raise ValueError("bundle bandit training identity mismatch")
    return bundle


def load_frozen_policy(path):
    return validate_bundle(read_json(path))


def policy_decision(bundle, item, trace_id):
    """Deterministic recomputable random draw from frozen input, not any outcome."""
    config = bundle["policy_config"]
    x = item["pre_treatment_state"]["x"]
    allowed, reason = eligibility(x)
    raw, effective, prediction = 0.0, 0.0, None
    kind = config["kind"]
    if allowed:
        if kind in BASELINES:
            raw = float({"no_memory": False, "silent": False, "always_remind": True, "source": x["v"] == 1,
                "applicability": x["rho"] == 1, "exposure": x["e"]["value"] == 1,
                "hard_tristate": x["v"] == x["rho"] == x["e"]["value"] == 1}.get(kind, False))
            if kind == "similarity":
                score = x["retrieval"].get("similarity_score")
                raw = float(score is not None and score >= config["similarity_threshold"])
            effective, reason = raw, "rule_remind" if raw else "rule_silent"
        else:
            artifacts = bundle["artifacts"]
            model, calibration = artifacts["model"], artifacts["calibration"]
            prediction = predict_calibrated(model, x, calibration)
            bounds = prediction["calibrated_bounds"]
            if kind == "supervised":
                initialized = predict_initial_gate(artifacts["warm_gate"], model, calibration, x)
                raw = initialized["probability"] or 0.0
            else:
                bandit = _candidate_for_mode(artifacts["bandit"], kind)
                raw = predict_bandit(bandit, x)["effective_probability"]
            if bounds["tau"]["status"] != "available" or bounds["harm"]["status"] != "available":
                reason = "uncertainty_unavailable"
            elif bounds["tau"]["lower"] <= config["minimum_advantage"]:
                reason = "benefit_not_supported"
            elif bounds["harm"]["upper"] > config["risk_threshold"]:
                reason = "risk_postguard"
            else:
                effective, reason = raw, "learned_with_postguard"
    seed = stage_seed(bundle["study"]["base_seed"], item["task_id"], 0, "frozen-policy-decision")
    draw = random.Random(seed).random()
    arm = "remind" if draw < effective else "silent"
    return seal({"schema_version": "frozen-policy-decision.v1", "bundle_hash": bundle["record_hash"],
        "input_hash": item["record_hash"], "x_hash": digest(x), "a0_hash": item["a0_hash"],
        "raw_probability": raw, "effective_probability": effective, "eligible": allowed,
        "reason": reason, "arm": arm, "decision_seed": seed, "draw": draw,
        "execution_seed": stage_seed(bundle["study"]["base_seed"], item["task_id"], 0, "frozen-policy-execution"),
        "prediction_hash": digest(prediction) if prediction is not None else None,
        "wrapper_requires_validation": True})
