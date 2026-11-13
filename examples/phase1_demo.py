"""Run the source-memory pipeline on explicit software fixtures; no network."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cave_mem_v3.cli import main as cave3
from cave_mem_v3.common.contracts import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "phase1-demo")
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("demo output must be a new or empty directory; existing artifacts are preserved")
    config = ROOT / "configs" / "phase1" / "demo.toml"
    manifest, history = output / "manifest.json", output / "history"
    candidates, verification, bank = output / "candidates.json", output / "verification", output / "bank.json"

    def run(*parts):
        code = cave3([str(part) for part in parts])
        if code:
            raise RuntimeError(f"demo stopped at {parts[0]}; completed artifacts are retained in {output}")

    run("manifest", "--catalog", ROOT / "configs" / "benchmarks" / "demo.json", "--output", manifest)
    run("collect", "--config", config, "--manifest", manifest, "--output", history)
    run("candidates", "--collection", history, "--output", candidates)
    drafts = read_json(candidates)
    for candidate in drafts["candidates"]:
        if candidate["source_task_id"] != "history-risk" or candidate["problem_action"] != {"type": "operate"}:
            raise ValueError("demo repair only applies to the registered preparation fixture")
        candidate.update(
            mechanism="Operation before preparation irreversibly fails this fixture.",
            correction={"type": "prepare"},
            positive_conditions=[{"field": "requires_preparation", "equals": True}],
            exclusion_conditions=[{"field": "prepared", "equals": True}],
            scope="Development fixture at the recorded checkpoint only.",
            proposed_by="explicit-fixture-demo",
        )
    write_json(candidates, drafts)
    run("verify", "--config", config, "--manifest", manifest, "--collection", history,
        "--candidates", candidates, "--output", verification)
    run("bank", "--verification", verification, "--output", bank)
    print(f"Software fixture demo complete: {bank}\nThis is not a real-model benchmark result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
