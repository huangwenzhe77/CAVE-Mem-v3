"""Run one randomized target arm in a live, non-restorable software fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase1_demo import main as source_demo
from phase2_demo import FixtureEncoder
from cave_mem_v3.cli import _actor
from cave_mem_v3.common.config import load_config
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.interventions.single import collect_single_target
from cave_mem_v3.memory_generation.bank import load_bank
from cave_mem_v3.target_data.sampling import RiskHazard


class LiveOnlyFixture(PreparationEnvironment):
    """Deliberately exposes no snapshot, restore, or state_fingerprint method."""

    def __init__(self, *, task_id, max_steps=4, **kwargs):
        self._environment = PreparationEnvironment(task_id=task_id, max_steps=max_steps, **kwargs)

    def __getattribute__(self, name):
        if name in ("snapshot", "restore", "state_fingerprint"):
            raise AttributeError(f"live fixture has no {name} capability")
        return super().__getattribute__(name)

    def configuration(self):
        return {**self._environment.configuration(), "live_session_only": True}

    def capabilities(self):
        return {"restore_certified": False, "isolated_sessions": False,
                "future_randomness": "deterministic", "step_seed_applied": True,
                "research_role": "fixture_only"}

    def reset(self, seed=0):
        return self._environment.reset(seed)

    def observe(self):
        return self._environment.observe()

    def terminal_status(self):
        return self._environment.terminal_status()

    def validate_action(self, action):
        return self._environment.validate_action(action)

    def step(self, action, seed=0):
        return self._environment.step(action, seed)

    def close(self):
        self._environment.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "phase2-single-demo")
    parser.add_argument("--assignment-probability", type=float, default=.5,
                        help="software example probability of Remind; production CLI requires an explicit value")
    args = parser.parse_args(argv)
    if not 0 < args.assignment_probability < 1:
        parser.error("assignment probability must be strictly between zero and one")
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("demo output must be a new or empty directory; existing artifacts are preserved")
    source = output / "source"
    source_demo(["--output", str(source)])
    manifest = read_json(source / "manifest.json")
    bank = load_bank(source / "bank.json")
    raw = load_config(ROOT / "configs" / "phase1" / "demo.toml")
    summary = collect_single_target(manifest, bank=bank, role="target_train",
        spec=StudySpec(**raw["study"]),
        env_factory=lambda task: LiveOnlyFixture(task_id=task["task_id"],
            max_steps=raw["study"]["max_steps"], **task.get("environment", {})),
        actor_factory=lambda: _actor(raw), output_dir=output / "single", encoder=FixtureEncoder(),
        hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99),
        assignment_probability=args.assignment_probability)
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Software fixture single-arm demo complete: {output}\n"
          "No external model or real benchmark was run. Only the assigned arm was observed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
