"""Audited paired data -> heads -> calibration -> initial gate, entirely offline.

The tiny fixture deliberately lacks calibration/training support. This example
shows explicit unavailable heads and a Silent gate, not a measured research effect.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase2_demo import main as paired_demo, FixtureEncoder
from cave_mem_v3.cli import _actor, _env_factory
from cave_mem_v3.common.config import load_config
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.memory_generation.bank import load_bank
from cave_mem_v3.policy.dataset import write_policy_dataset
from cave_mem_v3.policy.supervised import SupervisedConfig, fit_supervised
from cave_mem_v3.policy.calibration import CalibrationConfig, calibrate_supervised
from cave_mem_v3.policy.gate import GateConfig, fit_initial_gate, predict_initial_gate
from cave_mem_v3.target_data.collection import collect_target
from cave_mem_v3.target_data.sampling import RiskHazard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "phase3-supervised-demo")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("use a new or empty output directory; failed evidence is preserved")
    paired_demo(["--output", str(output / "phase2")])
    phase2 = output / "phase2"
    manifest = read_json(phase2 / "source" / "manifest.json")
    bank = load_bank(phase2 / "source" / "bank.json")
    raw = load_config(ROOT / "configs/phase1/demo.toml")
    env_factory, actor_factory = _env_factory(raw, manifest), lambda: _actor(raw)
    prepared, paired = output / "calibration-prepared", output / "calibration-paired"
    collect_target(manifest, role="validation", validation_role="calibration", spec=StudySpec(**raw["study"]),
        env_factory=env_factory, actor_factory=actor_factory, output_dir=prepared,
        hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
        prepare_before_action=True, bank=bank, encoder=FixtureEncoder())
    collect_target_pairs(prepared, bank, env_factory=env_factory, actor_factory=actor_factory, output_dir=paired)
    training = write_policy_dataset([{"kind": "paired", "run": str(phase2 / "paired"), "prepared": str(phase2 / "prepared")}], output / "training.json")
    write_policy_dataset([{"kind": "paired", "run": str(paired), "prepared": str(prepared)}],
        output / "calibration.json", role="validation", validation_role="calibration")
    model = fit_supervised(output / "training.json", config=SupervisedConfig(gamma=1.0), output=output / "model.json")
    calibration = calibrate_supervised(output / "model.json", output / "calibration.json",
        config=CalibrationConfig(), output=output / "bounds.json")
    gate = fit_initial_gate(output / "model.json", output / "bounds.json", output / "training.json",
        config=GateConfig(risk_threshold=.1), output=output / "gate.json")
    prediction = predict_initial_gate(gate, model, calibration, training["rows"][0]["features"])
    write_json(output / "prediction.json", prediction)
    report = {"fixture_only": True, "external_model_requests": 0,
        "heads": {key: head["status"] for key, head in model["heads"].items()},
        "gate_status": gate["head"]["status"], "action": prediction["action"], "reason": prediction["reason"],
        "research_settings_selected": False, "output": str(output)}
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
