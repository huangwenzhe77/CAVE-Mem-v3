"""Pinned Terminal-Bench 2 task assets and a restricted Docker execution adapter.

This is an explicit filesystem-at-action-boundaries protocol, not Harbor's
interactive shell/session semantics. Tests and solutions never enter actor
containers. The official task test.sh is run in a separate scoring container.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cave_mem_v3.common.contracts import PublicContext, canonical, digest, read_json
from cave_mem_v3.environments.base import EnvStep


PROTOCOL = "stateless-offline-filesystem-v1"
DEFAULT_PROTOCOL = {"network": "none", "verifier_network": "bridge", "workdir": "/app",
    "max_steps": 30, "command_timeout": 60.0, "verifier_timeout": 600.0,
    "build_timeout": 1200.0, "cpus": 2.0, "memory_mb": 2048, "output_limit": 12000,
    "success_threshold": 1.0, "require_ctrf": True, "paired_scope": None}


class TerminalInfrastructureError(RuntimeError):
    """Unknown task outcome: backend, verifier, asset, or recovery failure."""


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("task path escapes the declared root")
    return path


def task_asset_hash(path: str | Path) -> str:
    root = Path(path).resolve()
    entries = []
    for item in sorted(root.rglob("*")):
        if item.is_symlink() or not item.resolve().is_relative_to(root) or (hasattr(item, "is_junction") and item.is_junction()):
            raise ValueError("task assets must not contain symbolic links/junctions or escape their root")
        if item.is_file() and ".git" not in item.relative_to(root).parts:
            entries.append((item.relative_to(root).as_posix(), hashlib.sha256(item.read_bytes()).hexdigest()))
    if not entries:
        raise ValueError("empty task directory")
    return digest(entries)


def _protocol(value: dict | None) -> dict:
    values = {**DEFAULT_PROTOCOL, **copy.deepcopy(value or {})}
    allowed = set(DEFAULT_PROTOCOL) | {"audit_path", "audit", "environment", "image"}
    if set(values) - allowed:
        raise ValueError("unknown terminal protocol fields: " + str(sorted(set(values) - allowed)))
    if values["network"] != "none" or values["verifier_network"] not in ("none", "bridge"):
        raise ValueError("actor protocol requires network=none; verifier network must be explicit")
    if not isinstance(values["workdir"], str) or not values["workdir"].startswith("/"):
        raise ValueError("terminal workdir must be an absolute container path")
    for key in ("max_steps", "output_limit", "memory_mb"):
        if type(values[key]) is not int or values[key] < 1:
            raise ValueError("invalid terminal " + key)
    for key in ("command_timeout", "verifier_timeout", "build_timeout", "cpus"):
        if isinstance(values[key], bool) or not math.isfinite(float(values[key])) or values[key] <= 0:
            raise ValueError("invalid terminal " + key)
    if type(values["require_ctrf"]) is not bool:
        raise ValueError("require_ctrf must be boolean")
    threshold = values["success_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not math.isfinite(threshold):
        raise ValueError("invalid success threshold")
    if values["paired_scope"] not in (None, PROTOCOL):
        raise ValueError("unsupported paired scope")
    env = values.get("environment", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
                                      or not isinstance(v, str) for k, v in env.items()):
        raise ValueError("environment must be explicit string variables")
    return values


def protocol_hash(protocol: dict) -> str:
    """Audit-independent identity of every execution/scoring setting."""
    return digest({k: v for k, v in _protocol(protocol).items() if k not in ("audit", "audit_path")})


@dataclass(frozen=True)
class TerminalTask:
    task_id: str
    task_dir: Path
    instruction: str
    asset_hash: str
    spec: dict[str, Any]

    def public_manifest(self) -> dict:
        return {"task_id": self.task_id, "asset_hash": self.asset_hash,
                "repository_commit": self.spec["repository_commit"],
                "benchmark": self.spec.get("benchmark", "terminal-bench@2.0"),
                "protocol": copy.deepcopy(self.spec["protocol"])}

    def verify_assets(self) -> None:
        if task_asset_hash(self.task_dir) != self.asset_hash:
            raise ValueError("task assets changed after manifest loading")


def task_from_directory(path: str | Path, task_id: str | None = None, *,
                        repository_commit: str = "", protocol: dict | None = None) -> TerminalTask:
    path = Path(path).resolve()
    for relative in ("instruction.md", "task.toml", "environment/Dockerfile", "tests/test.sh"):
        if not _safe_path(path, relative).is_file():
            raise ValueError("unsupported Terminal-Bench task: missing " + relative)
    for compose in ("docker-compose.yaml", "docker-compose.yml"):
        if (path / "environment" / compose).exists():
            raise ValueError("multi-container task needs a separate adapter")
    config = tomllib.loads((path / "task.toml").read_text(encoding="utf-8"))
    if config.get("steps") or config.get("environment", {}).get("os", "linux") != "linux":
        raise ValueError("only single-stage Linux tasks are supported")
    if config.get("verifier", {}).get("environment"):
        raise ValueError("separate verifier-image tasks need a separate adapter")
    instruction = (path / "instruction.md").read_text(encoding="utf-8")
    lines = instruction.splitlines()
    while lines and (not lines[0].strip() or re.match(r"^(<!--.*canary.*-->|#.*canary.*)$", lines[0], re.I)):
        lines.pop(0)
    instruction = "\n".join(lines).strip()
    if not instruction:
        raise ValueError("empty task instruction")
    return TerminalTask(task_id or path.name, path, instruction, task_asset_hash(path),
        {"repository_commit": repository_commit, "benchmark": "terminal-bench@2.0",
         "protocol": _protocol(protocol), "task_config": config})


def load_terminal_tasks(manifest_path: str | Path, root: str | Path | None = None) -> list[TerminalTask]:
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "terminal-task-manifest.v1":
        raise ValueError("unsupported terminal task manifest")
    commit = manifest.get("repository_commit", "")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("pin the complete benchmark repository commit")
    if manifest.get("benchmark") != "terminal-bench@2.0":
        raise ValueError("declare terminal-bench@2.0 explicitly")
    task_root = Path(root) if root is not None else Path(manifest.get("root", manifest_path.parent))
    if not task_root.is_absolute():
        task_root = manifest_path.parent / task_root
    tasks = []
    for row in manifest.get("tasks", []):
        task = task_from_directory(_safe_path(task_root.resolve(), row["path"]), row["task_id"],
                                   repository_commit=commit, protocol=row.get("protocol"))
        if row.get("asset_hash") != task.asset_hash:
            raise ValueError("task asset hash differs from frozen manifest: " + task.task_id)
        audit_path = task.spec["protocol"].get("audit_path")
        if audit_path:
            task.spec["protocol"]["audit"] = read_json(_safe_path(manifest_path.parent, audit_path))
        tasks.append(task)
    if not tasks or len({t.task_id for t in tasks}) != len(tasks):
        raise ValueError("declare a nonempty list of unique task IDs")
    return tasks


def parse_verifier_result(directory: str | Path, *, success_threshold: float = 1.0,
                          require_ctrf: bool = True) -> dict:
    directory = Path(directory)
    if (directory / "reward.json").is_file():
        rewards = json.loads((directory / "reward.json").read_text(encoding="utf-8"))
        if not isinstance(rewards, dict) or "reward" not in rewards:
            raise TerminalInfrastructureError("reward.json must contain the declared reward field")
    elif (directory / "reward.txt").is_file():
        try:
            rewards = {"reward": float((directory / "reward.txt").read_text(encoding="utf-8").strip())}
        except ValueError as exc:
            raise TerminalInfrastructureError("invalid verifier reward") from exc
    else:
        raise TerminalInfrastructureError("official verifier did not produce a reward file")
    if not rewards or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                          for v in rewards.values()):
        raise TerminalInfrastructureError("verifier rewards must be finite numbers")
    report = {"rewards": rewards, "success": rewards["reward"] >= success_threshold,
              "success_threshold": success_threshold}
    if require_ctrf:
        path = directory / "ctrf.json"
        if not path.is_file():
            raise TerminalInfrastructureError("reward exists without CTRF test execution evidence")
        try:
            ctrf = json.loads(path.read_text(encoding="utf-8"))
            results = ctrf["results"]
            tests, summary = results["tests"], results["summary"]
            if not isinstance(tests, list) or not tests or summary["tests"] != len(tests):
                raise ValueError("no complete test execution")
            if summary.get("errors", 0) or summary.get("other", 0):
                raise ValueError("collection/setup/infrastructure errors")
            if any(t.get("status") not in ("passed", "failed", "skipped", "pending") for t in tests):
                raise ValueError("unknown test execution status")
            if not any(t["status"] in ("passed", "failed") for t in tests):
                raise ValueError("all tests skipped")
            for status in ("passed", "failed", "skipped", "pending"):
                if summary.get(status, 0) != sum(t["status"] == status for t in tests):
                    raise ValueError("summary and actual test statuses differ")
            infrastructure = re.compile(r"\b(setup|teardown|collection)\b.*\b(error|failed|exception)\b|\b(failed|error)\b.*\b(setup|teardown|collection)\b|\b(importerror|modulenotfounderror)\b", re.I)
            for test in tests:
                metadata = " ".join(str(test.get(k, "")) for k in ("rawStatus", "raw_status", "message", "trace"))
                if str(test.get("rawStatus", test.get("raw_status", ""))).lower() == "error" or infrastructure.search(metadata):
                    raise ValueError("test setup/collection/teardown failed")
            if report["success"] and any(t["status"] == "failed" for t in tests):
                raise ValueError("success reward conflicts with failed tests")
            if not report["success"] and all(t["status"] == "passed" for t in tests):
                raise ValueError("failure reward conflicts with all tests passing")
            report["test_summary"] = summary
        except (KeyError, TypeError, ValueError) as exc:
            raise TerminalInfrastructureError("invalid/incomplete CTRF test execution evidence: " + str(exc)) from exc
    return report


class DockerBackend:
    """Docker CLI only; model commands are never executed by the host shell."""
    identity = "docker-cli-fs-boundary.v1"
    filesystem_boundaries = True

    def __init__(self, executable: str = "docker"):
        self.executable = executable

    def _run(self, arguments: list[str], *, timeout: float = 60, allowed=(), limit=12000):
        try:
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                completed = subprocess.run([self.executable, *arguments], stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, timeout=timeout, check=False)
                stdout.seek(0); stderr.seek(0)
                out, err = stdout.read(limit + 1), stderr.read(limit + 1)
            result = {"exit_code": completed.returncode,
                      "stdout": out[:limit].decode("utf-8", errors="replace"),
                      "stderr": err[:limit].decode("utf-8", errors="replace"),
                      "truncated": len(out) > limit or len(err) > limit}
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TerminalInfrastructureError("Docker execution unavailable/timed out: " + str(exc)) from exc
        if completed.returncode != 0 and completed.returncode not in allowed:
            raise TerminalInfrastructureError("Docker command failed: " + result["stderr"])
        return result

    def doctor(self):
        return json.loads(self._run(["version", "--format", "{{json .}}"])["stdout"])

    def image_available(self, image: str) -> bool:
        try:
            self._run(["image", "inspect", image])
            return True
        except TerminalInfrastructureError:
            return False

    def ensure_image(self, task: TerminalTask, protocol: dict) -> str:
        task.verify_assets()
        image = protocol.get("image")
        if image is None:
            tag = "cave-terminal-" + task.asset_hash[:20]
            if not self.image_available(tag):
                self._run(["build", "--tag", tag, "--file", str(task.task_dir / "environment" / "Dockerfile"),
                           str(task.task_dir / "environment")], timeout=protocol["build_timeout"])
            image = tag
        info = json.loads(self._run(["image", "inspect", image])["stdout"])[0]
        if info.get("Config", {}).get("Volumes"):
            raise TerminalInfrastructureError("image declares volumes outside filesystem checkpoint scope")
        if info.get("Os") != "linux":
            raise TerminalInfrastructureError("Linux Docker image required")
        # This does not prove arbitrary Dockerfile safety. It catches known
        # grader/oracle mount locations; manifest review still owns the scope.
        probe = self._run(["run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", info["Id"],
                           "-c", "test ! -e /tests/test.sh && test ! -e /solution && test ! -e /logs/verifier/reward.txt"], allowed=(1,))
        if probe["exit_code"]:
            raise TerminalInfrastructureError("actor image includes verifier/oracle artifacts")
        return info["Id"]

    def execute(self, image: str, command: str, protocol: dict, *, seed: int = 0) -> tuple[str, dict]:
        name = "cave-terminal-" + uuid.uuid4().hex
        argv = ["create", "--name", name, "--hostname", "cave-terminal", "--network", protocol["network"],
                "--cpus", str(protocol["cpus"]), "--memory", str(protocol["memory_mb"]) + "m",
                "--workdir", protocol["workdir"], "--entrypoint", "/bin/bash", "--security-opt", "no-new-privileges"]
        for key, value in sorted(protocol.get("environment", {}).items()):
            argv.extend(["--env", key + "=" + value])
        argv.extend([image, "-c", command])
        self._run(argv)
        try:
            try:
                result = self._run(["start", "--attach", name], timeout=protocol["command_timeout"],
                                   allowed=tuple(range(1, 125)) + tuple(range(126, 256)), limit=protocol["output_limit"])
            except TerminalInfrastructureError as exc:
                if not isinstance(exc.__cause__, subprocess.TimeoutExpired):
                    raise
                # A declared action time limit is a metered behavior outcome.
                # Kill this isolated command and preserve its partial files; a
                # daemon/kill failure still propagates as infrastructure error.
                self._run(["kill", name])
                result = self._run(["logs", name], limit=protocol["output_limit"])
                result["command_timed_out"] = True
            state = json.loads(self._run(["inspect", "--format", "{{json .State}}", name])["stdout"])
            if state.get("Running") or state.get("OOMKilled") or state.get("Error"):
                raise TerminalInfrastructureError("container did not reach a clean filesystem boundary")
            result["exit_code"] = 124 if result.get("command_timed_out") else state["ExitCode"]
            next_image = self._run(["commit", name])["stdout"].strip()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", next_image):
                raise TerminalInfrastructureError("invalid Docker snapshot image identity")
            return next_image, result
        finally:
            self._run(["rm", "--force", name], allowed=(1,))

    def verify(self, image: str, task: TerminalTask, protocol: dict) -> dict:
        task.verify_assets()
        name = "cave-verifier-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="cave-verifier-") as tmp:
            logs = Path(tmp)
            argv = ["create", "--name", name, "--hostname", "cave-terminal", "--network", protocol["verifier_network"],
                    "--cpus", str(protocol["cpus"]), "--memory", str(protocol["memory_mb"]) + "m",
                    "--workdir", protocol["workdir"], "--entrypoint", "/bin/bash",
                    "--mount", "type=bind,src=" + str(task.task_dir / "tests") + ",dst=/tests,readonly",
                    "--mount", "type=bind,src=" + str(logs) + ",dst=/logs/verifier"]
            env = {**protocol.get("environment", {}), **task.spec["task_config"].get("verifier", {}).get("env", {})}
            for key, value in sorted(env.items()):
                if not isinstance(value, str) or "${" in value:
                    raise ValueError("verifier environment requires resolved non-secret literal values")
                argv.extend(["--env", key + "=" + value])
            argv.extend([image, "-c", "bash /tests/test.sh"])
            self._run(argv)
            try:
                execution = self._run(["start", "--attach", name], timeout=protocol["verifier_timeout"],
                                     allowed=tuple(range(1, 125)) + tuple(range(126, 256)), limit=protocol["output_limit"])
                state = json.loads(self._run(["inspect", "--format", "{{json .State}}", name])["stdout"])
                if state.get("Running") or state.get("OOMKilled") or state.get("Error"):
                    raise TerminalInfrastructureError("verifier infrastructure did not finish cleanly")
                if state.get("ExitCode") not in (0, 1):
                    raise TerminalInfrastructureError("verifier exited outside declared test outcomes")
                report = parse_verifier_result(logs, success_threshold=protocol["success_threshold"],
                                              require_ctrf=protocol["require_ctrf"])
                report["execution"] = execution
                report["network"] = protocol["verifier_network"]
                return report
            finally:
                self._run(["rm", "--force", name], allowed=(1,))


class TerminalEnvironment:
    def __init__(self, task: TerminalTask, config: dict | None = None, *, backend=None):
        self.task = task
        self.protocol = _protocol({**task.spec["protocol"], **(config or {})})
        self.backend = backend or DockerBackend()
        self.max_steps = self.protocol["max_steps"]
        self._state = None
        self._base_image = None
        self.verifier_report = None

    def configuration(self):
        return {"adapter": "terminal-bench-fs-boundary.v1", "task_id": self.task.task_id,
                "asset_hash": self.task.asset_hash, "repository_commit": self.task.spec["repository_commit"],
                "protocol": copy.deepcopy(self.protocol), "backend": self.backend.identity,
                "semantics": PROTOCOL, "official_harbor_session_equivalence": False}

    def capabilities(self):
        audit = self.protocol.get("audit", {})
        declared = self.protocol["paired_scope"] == PROTOCOL
        records = audit.get("restoration_checks", [])
        certified = bool(declared and getattr(self.backend, "filesystem_boundaries", False)
            and audit.get("asset_hash") == self.task.asset_hash and audit.get("scope") == PROTOCOL
            and audit.get("protocol_hash") == protocol_hash(self.protocol)
            and audit.get("repository_commit") == self.task.spec["repository_commit"]
            and audit.get("backend") == self.backend.identity
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(audit.get("base_image", "")))
            and audit.get("reviewer") and audit.get("review_notes")
            and audit.get("reviewed_constraints") == {"no_external_state": True, "no_persistent_services": True,
                "filesystem_boundary_semantics": True, "clock_and_rng_are_exogenous": True}
            and len(records) >= 2 and len({r.get("check_id") for r in records}) == len(records)
            and all(r.get("check_id") and r.get("snapshot_hash") and r.get("snapshot_hash") == r.get("restored_hash") for r in records)
            and audit.get("filesystem_probe", {}).get("persisted_after_restore") is True
            and audit.get("filesystem_probe", {}).get("branches_isolated") is True
            and audit.get("filesystem_probe", {}).get("base_branch_isolated") is True)
        return {"restore": "immutable_image_and_explicit_context", "restore_certified": certified,
                "paired_primary_eligible": certified, "randomized_single_supported": True,
                "state_fingerprint_scope": PROTOCOL, "future_randomness": "exogenous_os_clock_rng_unmatched",
                "step_seed_applied": False, "isolated_sessions": True,
                "restoration_note": "certification is limited to reviewed stateless task protocol, not arbitrary Terminal-Bench tasks"}

    def reset(self, seed: int = 0):
        image = self.backend.ensure_image(self.task, self.protocol)
        if self.capabilities()["restore_certified"] and image != self.protocol["audit"]["base_image"]:
            raise TerminalInfrastructureError("built image differs from reviewed paired base image")
        self._base_image = image
        self._state = {"image": image, "steps": 0, "history": [], "terminal": False, "success": None,
                       "observation": "Terminal ready. Each shell action uses a fresh process; files persist. "
                           "Use an explicit shell cd within a command when needed. Submit finish to run terminal evaluation."}
        self.verifier_report = None
        return self.observe()

    def _require(self):
        if self._state is None:
            raise RuntimeError("reset or restore the terminal environment first")
        return self._state

    def observe(self):
        state = self._require()
        context = PublicContext(self.task.task_id, self.task.instruction, state["observation"],
            {"phase": "terminal" if state["terminal"] else "acting"}, tuple(copy.deepcopy(state["history"])),
            (), max(0, self.max_steps - state["steps"]))
        return context

    def snapshot(self):
        return {"schema_version": "terminal-checkpoint.v3", "configuration": self.configuration(),
                "base_image": self._base_image,
                "state": copy.deepcopy(self._require())}

    def restore(self, checkpoint):
        self.task.verify_assets()
        if checkpoint.get("schema_version") != "terminal-checkpoint.v3":
            raise ValueError("terminal checkpoint schema mismatch")
        restored_config = checkpoint.get("configuration", {})
        self._base_image = self.backend.ensure_image(self.task, self.protocol)
        if self._base_image != checkpoint.get("base_image"):
            raise ValueError("checkpoint base image differs from the current task image")
        if restored_config != self.configuration():
            raise ValueError("terminal checkpoint task/protocol mismatch")
        state = copy.deepcopy(checkpoint["state"])
        if not self.backend.image_available(state["image"]):
            raise TerminalInfrastructureError("checkpoint Docker image is unavailable")
        if type(state["steps"]) is not int or not 0 <= state["steps"] <= self.max_steps:
            raise ValueError("invalid terminal checkpoint steps")
        self._state = state
        self.verifier_report = None
        return self.observe()

    def state_fingerprint(self):
        return digest(self.snapshot())

    def diagnose_restoration(self, repetitions: int = 2):
        """Measured recovery checks plus an unapproved review template.

        Matching this adapter's state representation alone never approves paired
        use. A domain reviewer must establish the four task-scope assumptions.
        """
        if type(repetitions) is not int or repetitions < 2:
            raise ValueError("at least two independent restore checks required")
        checkpoint = self.snapshot()
        checks = []
        for index in range(repetitions):
            other = TerminalEnvironment(self.task, self.protocol, backend=self.backend)
            other.restore(copy.deepcopy(checkpoint))
            checks.append({"check_id": f"restore-{index}", "snapshot_hash": digest(checkpoint),
                           "restored_hash": other.state_fingerprint()})
            other.close()
        # Exercise real filesystem snapshots on disposable branches. Merely
        # comparing serialized image IDs cannot demonstrate that files restore.
        marker = "/tmp/cave-v3-restore-" + uuid.uuid4().hex
        token = uuid.uuid4().hex
        original_image = checkpoint["state"]["image"]
        seed_image, written = self.backend.execute(original_image,
            f"printf '%s' '{token}' > '{marker}'", self.protocol)
        if written["exit_code"] != 0:
            raise TerminalInfrastructureError("restore diagnostic could not write marker")
        branch_images = []
        observations = []
        for index in range(2):
            restored = copy.deepcopy(checkpoint)
            restored["state"]["image"] = seed_image
            other = TerminalEnvironment(self.task, self.protocol, backend=self.backend)
            try:
                other.restore(restored)
                _, observed = self.backend.execute(other._require()["image"], f"cat '{marker}'", self.protocol)
                observations.append({"branch": index, "kind": "restored", **observed})
                if observed["exit_code"] != 0 or observed["stdout"] != token:
                    raise TerminalInfrastructureError("restored snapshot lost written marker")
                image, changed = self.backend.execute(other._require()["image"],
                    f"printf '%s' '{token}-{index}' > '{marker}'", self.protocol)
                if changed["exit_code"] != 0:
                    raise TerminalInfrastructureError("restore diagnostic branch write failed")
                branch_images.append(image)
            finally:
                other.close()
        for index, image in enumerate(branch_images):
            _, observed = self.backend.execute(image, f"cat '{marker}'", self.protocol)
            observations.append({"branch": index, "kind": "after_other_branch_write", **observed})
            if observed["exit_code"] != 0 or observed["stdout"] != f"{token}-{index}":
                raise TerminalInfrastructureError("filesystem changes leaked between replay branches")
        _, unchanged = self.backend.execute(seed_image, f"cat '{marker}'", self.protocol)
        _, absent = self.backend.execute(original_image, f"test ! -e '{marker}'", self.protocol)
        if unchanged["exit_code"] != 0 or unchanged["stdout"] != token or absent["exit_code"] != 0:
            raise TerminalInfrastructureError("filesystem changes leaked into original checkpoint")
        base_image = self.backend.ensure_image(self.task, self.protocol)
        return {"schema_version": "terminal-restore-audit.v1", "scope": PROTOCOL,
                "asset_hash": self.task.asset_hash, "repository_commit": self.task.spec["repository_commit"],
                "protocol_hash": protocol_hash({**self.protocol, "paired_scope": PROTOCOL}),
                "backend": self.backend.identity, "base_image": base_image,
                "reviewer": "", "review_notes": "",
                "reviewed_constraints": {"no_external_state": False, "no_persistent_services": False,
                    "filesystem_boundary_semantics": False, "clock_and_rng_are_exogenous": False},
                "restoration_checks": checks,
                "filesystem_probe": {"persisted_after_restore": True, "branches_isolated": True,
                    "base_branch_isolated": True, "original_image": original_image,
                    "marked_image": seed_image, "branch_images": branch_images,
                    "observations": observations},
                "status": "diagnostics_passed_but_task_scope_unreviewed",
                "limitation": "matching serialized state is diagnostic; review complete task dependencies before approving scope"}

    def terminal_status(self):
        state = self._require()
        return state["terminal"], state["success"] if state["terminal"] else None

    def validate_action(self, action):
        if self._require()["terminal"]:
            return False, "terminal environment already finished"
        if not isinstance(action, dict):
            return False, "terminal action must be a typed mapping"
        if action == {"type": "finish"}:
            return True, None
        valid = (set(action) == {"type", "command"} and action.get("type") == "shell"
                 and isinstance(action.get("command"), str) and bool(action["command"].strip())
                 and "\x00" not in action["command"])
        return valid, None if valid else "expected shell(command) or finish"

    def step(self, action, seed=0):
        valid, reason = self.validate_action(action)
        if not valid:
            return EnvStep(self.observe(), False, None, False, None, reason)
        state = self._require()
        state["steps"] += 1
        if action["type"] == "shell":
            next_image, result = self.backend.execute(state["image"], action["command"], self.protocol, seed=seed)
            state["image"] = next_image
            state["observation"] = canonical(result)
            state["history"].append({"action": copy.deepcopy(action), "observation": state["observation"]})
        if action["type"] == "finish" or state["steps"] >= self.max_steps:
            # Official grader observes an isolated image branch. Its output is
            # private evaluation material, never actor history or gate features.
            self.finalize()
        return EnvStep(self.observe(), state["terminal"], state["success"] if state["terminal"] else None,
                       True, None)

    def finalize(self):
        """Grade current artifacts at finish or a budget boundary; no actor action."""
        state = self._require()
        if not state["terminal"]:
            self.verifier_report = self.backend.verify(state["image"], self.task, self.protocol)
            if type(self.verifier_report.get("success")) is not bool:
                raise TerminalInfrastructureError("verifier did not return explicit success")
            state.update(terminal=True, success=self.verifier_report["success"])
            state["observation"] = "Task execution ended; terminal evaluation recorded separately."
        return EnvStep(self.observe(), True, state["success"], True, None)

    def close(self):
        # Persist content-addressed images for replay. Do not globally prune Docker.
        pass
