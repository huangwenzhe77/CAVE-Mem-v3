"""Run source verification, target preparation, and paired software fixtures offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1_demo import main as source_demo
from cave_mem_v3.cli import _actor, _env_factory
from cave_mem_v3.common.config import load_config
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.memory_generation.bank import load_bank
from cave_mem_v3.target_data.collection import collect_target
from cave_mem_v3.target_data.sampling import RiskHazard


class FixtureEncoder:
    """Deterministic demonstration vectors; not a semantic embedding model."""
    identity = {"kind": "offline-fixture-encoder", "revision": "phase2-demo.v1",
                "research_role": "fixture_only"}

    def encode_query(self, text):
        return [1.0, float(text.count("prepare") + 1), float(text.count("operate") + 1)]

    def encode_documents(self, texts):
        return [self.encode_query(text) for text in texts]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "phase2-demo")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("demo output must be a new or empty directory; existing artifacts are preserved")
    source = output / "source"
    source_demo(["--output", str(source)])
    manifest = read_json(source / "manifest.json")
    bank = load_bank(source / "bank.json")
    raw = load_config(ROOT / "configs" / "phase1" / "demo.toml")
    env_factory = _env_factory(raw, manifest)
    actor_factory = lambda: _actor(raw)
    prepared = output / "prepared"
    collect_target(manifest, role="target_train", spec=StudySpec(**raw["study"]),
        env_factory=env_factory, actor_factory=actor_factory, output_dir=prepared,
        hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
        prepare_before_action=True, bank=bank, encoder=FixtureEncoder())
    summary = collect_target_pairs(prepared, bank, env_factory=env_factory,
        actor_factory=actor_factory, output_dir=output / "paired")
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Software fixture paired demo complete: {output}\n"
          "No external model or real benchmark was run. These are software observations only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
