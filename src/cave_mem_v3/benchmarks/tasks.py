"""Outcome-blind benchmark manifests and connected-group task partitions."""
from __future__ import annotations

import copy
import hashlib
import random
from collections import Counter
from pathlib import Path

from cave_mem_v3.common.contracts import canonical, digest, finite, read_json

ROLES = ("history", "target_train", "validation", "test")
VALIDATION_ROLES = ("calibration", "selection")
SCHEMA = "benchmark-manifest.v3"
OUTCOME_FIELDS = frozenset({"success", "terminal_success", "reward", "outcome", "score", "failure_count"})


def task_asset_hash(path: str | Path) -> str:
    """Hash all task files, including scoring assets; reject symbolic links."""
    if Path(path).is_symlink():
        raise ValueError("task assets cannot be symbolic links")
    root = Path(path).resolve(strict=True)
    files = [root] if root.is_file() else sorted(root.rglob("*"), key=lambda p: p.as_posix())
    records = []
    for item in files:
        if item.is_symlink():
            raise ValueError("task assets cannot contain symbolic links")
        if item.is_file():
            relative = item.name if root.is_file() else item.relative_to(root).as_posix()
            records.append([relative, hashlib.sha256(item.read_bytes()).hexdigest()])
    if not records:
        raise ValueError("task asset path contains no files")
    return digest(records)


def _without_outcomes(value) -> None:
    if isinstance(value, dict):
        forbidden = OUTCOME_FIELDS.intersection(value)
        if forbidden:
            raise ValueError(f"outcomes cannot enter task partition metadata: {sorted(forbidden)}")
        for child in value.values():
            _without_outcomes(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _without_outcomes(child)


def _components(tasks: list[dict]) -> list[list[dict]]:
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    for task in tasks:
        groups = task.get("groups", [])
        if not isinstance(groups, list) or any(not isinstance(g, str) or not g for g in groups):
            raise ValueError("groups must be a list of nonempty dependency names")
        group_id = task.get("group_id")
        if group_id is not None and (not isinstance(group_id, str) or not group_id):
            raise ValueError("group_id must be a nonempty string")
        if not groups and not group_id:
            raise ValueError("each task needs group_id or explicit groups")
        keys = ["task:" + task["task_id"]] + ["group:" + g for g in groups]
        if group_id:
            keys.append("group:" + group_id)
        if task.get("template_id"):
            if not isinstance(task["template_id"], str):
                raise ValueError("template_id must be a string")
            keys.append("template:" + task["template_id"])
        anchor = find(keys[0])
        for key in keys[1:]:
            parents[find(key)] = anchor
    components = {}
    for task in tasks:
        components.setdefault(find("task:" + task["task_id"]), []).append(task)
    return sorted(components.values(), key=lambda group: min(t["task_id"] for t in group))


def _ratios(value: dict, names: tuple[str, ...]) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("split ratios must explicitly name " + ", ".join(names))
    ratios = {name: finite(value[name], name + " ratio", minimum=0) for name in names}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    return ratios


def _allocate(groups: list[list[dict]], ratios: dict, seed: int, *, minimum=None) -> dict:
    active = [name for name, ratio in ratios.items() if ratio > 0]
    counts = {name: (minimum or {}).get(name, 1) if name in active else 0 for name in ratios}
    if len(groups) < sum(counts.values()):
        raise ValueError("insufficient independent dependency groups for the requested partitions")
    for _ in range(len(groups) - sum(counts.values())):
        name = max(active, key=lambda n: (ratios[n] * len(groups) - counts[n], -list(ratios).index(n)))
        counts[name] += 1
    ordered = list(groups)
    random.Random(seed).shuffle(ordered)
    result, start = {}, 0
    for name, count in counts.items():
        result[name] = ordered[start:start + count]
        start += count
    return result


def build_manifest(tasks: list[dict], *, benchmark: str, revision: str,
                   split_ratios: dict | None = None, seed: int = 42,
                   validation_ratios: dict | None = None) -> dict:
    """Freeze explicit roles or split dependency groups before observing outcomes.

    Ratios refer to independent groups, not task counts. Family/mechanism labels
    are descriptive unless explicitly included in ``groups`` as holdout axes.
    """
    if not isinstance(benchmark, str) or not benchmark.strip() or not isinstance(revision, str) or not revision.strip():
        raise ValueError("benchmark and immutable revision/asset version must be declared")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task manifest cannot be empty")
    tasks = copy.deepcopy(tasks)
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("task_id"), str) or not task["task_id"]:
            raise ValueError("each task requires task_id")
        if task["task_id"] in seen:
            raise ValueError("duplicate task_id")
        seen.add(task["task_id"])
        _without_outcomes(task)
        if task.get("path"):
            task["path"] = str(Path(task["path"]).resolve(strict=True))
            actual = task_asset_hash(task["path"])
            if task.get("asset_hash") and task["asset_hash"] != actual:
                raise ValueError("task assets disagree with the declared asset_hash")
            task["asset_hash"] = actual
            task["asset_state"] = "verified"
        elif task.get("asset_hash"):
            task["asset_state"] = "declared"
        else:
            task["asset_state"] = "unavailable"
    tasks.sort(key=lambda task: task["task_id"])
    groups = _components(tasks)
    if split_ratios is not None:
        ratios = _ratios(split_ratios, ROLES)
        if any("role" in task or "validation_role" in task for task in tasks):
            raise ValueError("automatic splitting requires unassigned tasks")
        allocations = _allocate(groups, ratios, seed, minimum={"validation": 2})
        for role, assigned_groups in allocations.items():
            for group in assigned_groups:
                for task in group:
                    task["role"] = role
        if allocations["validation"]:
            validation_ratios = _ratios(validation_ratios or {"calibration": 0.5, "selection": 0.5}, VALIDATION_ROLES)
            if any(value <= 0 for value in validation_ratios.values()):
                raise ValueError("both calibration and selection partitions need positive support")
            subsets = _allocate(allocations["validation"], validation_ratios, seed + 1)
            for subrole, assigned_groups in subsets.items():
                for group in assigned_groups:
                    for task in group:
                        task["validation_role"] = subrole
    else:
        ratios = None
        if validation_ratios is not None:
            raise ValueError("validation_ratios requires automatic group splitting")
    if any(task.get("role") not in ROLES for task in tasks):
        raise ValueError("explicit manifest tasks require a registered role")
    manifest = {"schema_version": SCHEMA, "benchmark": benchmark, "revision": revision,
                "split_method": "connected_group_random" if ratios is not None else "explicit",
                "split_seed": seed, "requested_group_ratios": ratios,
                "requested_validation_ratios": validation_ratios,
                "frozen_before_outcomes": True, "tasks": tasks,
                "independent_groups": len(groups),
                "role_counts": dict(Counter(task["role"] for task in tasks))}
    manifest["manifest_hash"] = digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict, *, verify_assets: bool = False) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA:
        raise ValueError("unsupported benchmark manifest")
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if manifest.get("manifest_hash") != digest(body):
        raise ValueError("manifest checksum mismatch")
    if manifest.get("frozen_before_outcomes") is not True:
        raise ValueError("manifest must be frozen before observing outcomes")
    if not manifest.get("benchmark") or not manifest.get("revision"):
        raise ValueError("benchmark/version missing")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest requires tasks")
    seen = set()
    for task in tasks:
        _without_outcomes(task)
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise ValueError("missing or duplicate task_id")
        seen.add(task_id)
        if task.get("role") not in ROLES:
            raise ValueError("unknown task role")
        if task["role"] == "validation":
            if task.get("validation_role") not in VALIDATION_ROLES:
                raise ValueError("validation tasks require calibration or selection subrole")
        elif "validation_role" in task:
            raise ValueError("validation_role belongs only to validation tasks")
        if task.get("asset_state") not in ("verified", "declared", "unavailable"):
            raise ValueError("task asset state must be explicit")
        if task["asset_state"] != "unavailable" and not task.get("asset_hash"):
            raise ValueError("declared or verified assets require asset_hash")
        if task["asset_state"] == "verified" and not task.get("path"):
            raise ValueError("verified assets require a local path")
        if verify_assets and task.get("path") and task_asset_hash(task["path"]) != task.get("asset_hash"):
            raise ValueError("task assets changed since manifest freezing")
    groups = _components(tasks)
    for group in groups:
        roles = {(task["role"], task.get("validation_role")) for task in group}
        if len(roles) != 1:
            raise ValueError("connected task/template dependency group crosses partitions")
    if manifest.get("independent_groups") != len(groups):
        raise ValueError("manifest dependency-group count mismatch")
    if manifest.get("role_counts") != dict(Counter(task["role"] for task in tasks)):
        raise ValueError("manifest role counts mismatch")


def load_manifest(path: str | Path, *, verify_assets: bool = False) -> dict:
    manifest = read_json(path)
    validate_manifest(manifest, verify_assets=verify_assets)
    return manifest
