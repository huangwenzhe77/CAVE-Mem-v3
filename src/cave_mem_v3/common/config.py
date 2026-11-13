"""Explicit phase-one configuration; reading it never loads model weights."""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path

from cave_mem_v3.common.contracts import StudySpec, digest


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    allowed = {"study", "actor", "environment", "collection", "verification"}
    if set(raw) - allowed:
        raise ValueError(f"unknown configuration sections: {sorted(set(raw) - allowed)}")
    for section in allowed:
        if section not in raw or not isinstance(raw[section], dict):
            raise ValueError(f"missing configuration section: {section}")
    StudySpec(**raw["study"])
    if raw["actor"].get("kind") not in ("scripted", "openai_compatible"):
        raise ValueError("actor.kind must be scripted or openai_compatible")
    if raw["environment"].get("kind") not in ("fixture", "terminal"):
        raise ValueError("environment.kind must be fixture or terminal")
    return copy.deepcopy(raw)


def readiness(raw: dict) -> dict:
    """List missing deployment choices without guessing sample sizes."""
    pending = []
    if raw["actor"]["kind"] == "openai_compatible":
        for key in ("model", "base_url", "revision"):
            if not raw["actor"].get(key):
                pending.append("actor." + key)
    collection = raw["collection"]
    mode = collection.get("mode", "fixed_runs")
    required = ["max_runs", "runs_per_task"]
    if mode == "failure_target":
        required.append("failure_target")
    for key in required:
        if type(collection.get(key)) is not int or collection[key] <= 0:
            pending.append("collection." + key)
    for key in ("screening_pairs", "confirmation_pairs"):
        if type(raw["verification"].get(key)) is not int or raw["verification"][key] <= 0:
            pending.append("verification." + key)
    return {"configuration_hash": digest(raw), "ready": not pending,
            "pending": pending, "model_downloaded": False,
            "note": "Configuration validation only; API, task assets and Docker are checked when used."}
