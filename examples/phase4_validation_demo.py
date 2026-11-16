"""Freeze Silent/Always-remind component baselines and complete offline validation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase1_demo import main as source_demo
from phase2_demo import FixtureEncoder
from cave_mem_v3.cli import _actor, _env_factory
from cave_mem_v3.common.config import load_config
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.policy.frozen import freeze_policy
from cave_mem_v3.policy.rollout import collect_policy_validation
from cave_mem_v3.target_data.sampling import RiskHazard


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/phase4-validation-demo")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("use new or empty directory; prior artifacts are preserved")
    source_demo(["--output", str(output / "source")])
    manifest, bank = (read_json(output / "source" / (name + ".json")) for name in ("manifest", "bank"))
    raw = load_config(ROOT / "configs/phase1/demo.toml")
    actor_factory = lambda: _actor(raw)
    paths = []
    # Freeze both before either validation outcome is observed.
    for kind in ("silent", "always_remind"):
        path = output / (kind + ".bundle.json")
        freeze_policy(manifest, bank, actor_config=actor_factory().configuration(), encoder_identity=FixtureEncoder.identity,
            study=StudySpec(**raw["study"]), hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
            policy_config={"kind": kind}, output=path)
        paths.append(path)
    reports = []
    for path in paths:
        report = collect_policy_validation(path, env_factory=_env_factory(raw, manifest), actor_factory=actor_factory,
            encoder=FixtureEncoder(), output_dir=output / path.stem)
        reports.append({"policy": read_json(path)["policy_config"]["kind"], "planned": report["planned_tasks"],
            "outcomes": report["observed_outcomes"], "successes": report["observed_successes"], "bundle_hash": report["bundle_hash"]})
    summary = {"fixture_only": True, "external_model_requests": 0, "policies": reports,
        "claim_boundary": "Software fixture full-task validation only; no winner selected and no test opened."}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
