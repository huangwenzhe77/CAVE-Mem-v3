"""Freeze a scene-disjoint, family-balanced ALFWorld engineering manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from cave_mem_v3.benchmarks.tasks import build_manifest, load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-root", type=Path, required=True,
                        help="official ALFWorld json_2.1.1/train directory")
    parser.add_argument("--reference-manifest", type=Path, required=True,
                        help="existing manifest supplying the benchmark and frozen revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-per-family", type=int, default=4)
    parser.add_argument("--target-per-family", type=int, default=8)
    parser.add_argument("--validation-per-family", type=int, default=2)
    parser.add_argument("--test-per-family", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("manifest output already exists")
    quotas = {"history": args.history_per_family, "target_train": args.target_per_family,
              "validation": args.validation_per_family, "test": args.test_per_family}
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise ValueError("all per-family role counts must be positive")
    reference = load_manifest(args.reference_manifest)
    root = args.train_root.resolve(strict=True)
    scenes_by_family: dict[str, list[Path]] = {}
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        family = scene.name.split("-", 1)[0]
        if any((trial / "game.tw-pddl").is_file() for trial in scene.iterdir() if trial.is_dir()):
            scenes_by_family.setdefault(family, []).append(scene)
    if not scenes_by_family:
        raise ValueError("no ALFWorld game scenes found")
    tasks = []
    for family, scenes in sorted(scenes_by_family.items()):
        rng = random.Random(f"{args.seed}:{family}")
        rng.shuffle(scenes)
        needed = sum(quotas.values())
        if len(scenes) < needed:
            raise ValueError(f"{family} has {len(scenes)} scenes, needs {needed}")
        index = 0
        for role, count in quotas.items():
            for scene in scenes[index:index + count]:
                trials = sorted(trial for trial in scene.iterdir()
                                if trial.is_dir() and (trial / "game.tw-pddl").is_file())
                trial = trials[rng.randrange(len(trials))]
                relative = trial.relative_to(root).as_posix()
                task_id = "alfworld-" + hashlib.sha256(relative.encode()).hexdigest()[:20]
                task = {"task_id": task_id, "group_id": "alfworld-scene:" + scene.name,
                        "template_id": "alfworld-scene:" + scene.name,
                        "task_type": family, "path": str(trial), "role": role}
                if role == "validation":
                    task["validation_role"] = ("calibration" if len(
                        [item for item in tasks if item["role"] == "validation"
                         and item["task_type"] == family]) < (count + 1) // 2 else "selection")
                tasks.append(task)
            index += count
    manifest = build_manifest(tasks, benchmark=reference["benchmark"],
                              revision=reference["revision"], seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"output": str(args.output.resolve()), "manifest_hash": manifest["manifest_hash"],
                      "role_counts": manifest["role_counts"],
                      "families": sorted(scenes_by_family)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
