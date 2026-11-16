"""Operational Enroot verifier timeout floor for slow official dependency installs.

The task protocol, command, scorer and actor environment stay frozen. Only the
host wait for the private verifier subprocess may be extended, and the actual
limit is recorded with the private result.
"""
from __future__ import annotations

import os

from cave_mem_v3.environments.enroot import EnrootBackend


class EnrootRuntimeBackend(EnrootBackend):
    def verify(self, image, task, protocol):
        configured = protocol["verifier_timeout"]
        raw_floor = os.environ.get("CAVE_VERIFIER_TIMEOUT_FLOOR_SECONDS", "")
        if not raw_floor:
            return super().verify(image, task, protocol)
        try:
            floor = int(raw_floor)
        except ValueError as exc:
            raise ValueError("CAVE_VERIFIER_TIMEOUT_FLOOR_SECONDS must be an integer") from exc
        if floor < configured or floor > 3600:
            raise ValueError("verifier timeout floor must be between the frozen value and 3600 seconds")
        effective_protocol = {**protocol, "verifier_timeout": floor}
        result = super().verify(image, task, effective_protocol)
        result["execution"]["effective_verifier_timeout_seconds"] = floor
        result["execution"]["frozen_verifier_timeout_seconds"] = configured
        return result
