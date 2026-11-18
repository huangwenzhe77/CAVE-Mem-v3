"""Check a real ALFWorld game, restoration, encoder, and temporary storage."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import site
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("history", "target_train"), required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--encoder-revision")
    parser.add_argument("--textworld-site-packages", type=Path,
                        help="append an existing TextWorld installation after the active runtime packages")
    args = parser.parse_args()
    if args.textworld_site_packages:
        site.addsitedir(str(args.textworld_site_packages.resolve(strict=True)))

    from cave_mem_v3.benchmarks.tasks import load_manifest
    from cave_mem_v3.environments.alfworld import ALFWorldEnvironment

    tmp_dir = args.tmp_dir.resolve(strict=True)
    if not tmp_dir.is_dir() or tempfile.gettempdir() != str(tmp_dir):
        raise ValueError("set TMPDIR to the existing --tmp-dir before starting Python")
    free_gb = shutil.disk_usage(tmp_dir).free / (1024 ** 3)
    if args.min_free_gb < 0 or free_gb < args.min_free_gb:
        raise ValueError(f"temporary storage has only {free_gb:.2f} GiB free")
    if bool(args.encoder_path) != bool(args.encoder_revision):
        raise ValueError("encoder path and revision must be supplied together")

    manifest = load_manifest(args.manifest, verify_assets=True)
    tasks = [task for task in manifest["tasks"] if task["role"] == args.role]
    if not tasks:
        raise ValueError("manifest has no task in the requested role")
    task = sorted(tasks, key=lambda item: item["task_id"])[0]
    original = ALFWorldEnvironment(task, max_steps=args.max_steps)
    restored = ALFWorldEnvironment(task, max_steps=args.max_steps)
    try:
        original.reset()
        checkpoint = original.snapshot()
        restored.restore(checkpoint)
        if restored.state_fingerprint() != original.state_fingerprint():
            raise ValueError("two independent ALFWorld sessions differ after restoration")
    finally:
        original.close()
        restored.close()

    dimension = None
    if args.encoder_path:
        from cave_mem_v3.target_data.retrieval import SentenceTransformerEncoder
        encoder = SentenceTransformerEncoder(str(args.encoder_path.resolve(strict=True)),
                                             revision=args.encoder_revision)
        dimension = len(encoder.encode_query("ALFWorld runtime preflight"))
    print(json.dumps({"status": "passed", "manifest_hash": manifest["manifest_hash"],
                      "task_id": task["task_id"], "tmp_dir": str(tmp_dir),
                      "tmp_free_gb": round(free_gb, 2), "encoder_dimension": dimension,
                      "python_tmpdir": os.environ.get("TMPDIR")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
