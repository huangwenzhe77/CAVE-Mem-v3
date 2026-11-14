"""Enroot image checkpoints for a declared Terminal-Bench filesystem protocol."""
from __future__ import annotations

import hashlib
import re
import shlex
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from cave_mem_v3.environments.terminal import TerminalInfrastructureError, parse_verifier_result


class EnrootBackend:
    identity = "enroot-squashfs-fs-boundary.v1"
    filesystem_boundaries = True

    def __init__(self, snapshot_dir: str | Path, executable: str = "enroot"):
        self.executable = executable
        self.snapshot_dir = Path(snapshot_dir).resolve()
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.base_path = None
        self.base_image = None

    def _run(self, args, *, timeout=60, allowed=(0,), limit=12000):
        try:
            with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
                result = subprocess.run([self.executable, *args], stdin=subprocess.DEVNULL,
                                        stdout=out, stderr=err, timeout=timeout, check=False)
                out.seek(0)
                err.seek(0)
                stdout, stderr = out.read(limit + 1), err.read(limit + 1)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TerminalInfrastructureError("Enroot unavailable/timed out: " + str(exc)) from exc
        report = {"exit_code": result.returncode,
                  "stdout": stdout[:limit].decode("utf-8", errors="replace"),
                  "stderr": stderr[:limit].decode("utf-8", errors="replace"),
                  "truncated": len(stdout) > limit or len(stderr) > limit}
        if result.returncode not in allowed:
            raise TerminalInfrastructureError("Enroot command failed: " + report["stderr"])
        return report

    @staticmethod
    def _identity(path):
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha.update(block)
        return "sha256:" + sha.hexdigest()

    def _path(self, image):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise TerminalInfrastructureError("invalid Enroot image identity")
        if image == self.base_image:
            return self.base_path
        return self.snapshot_dir / (image[7:] + ".sqsh")

    def image_available(self, image):
        return self._path(image).is_file()

    def ensure_image(self, task, protocol):
        task.verify_assets()
        image = protocol.get("image")
        if not isinstance(image, str) or not Path(image).is_file() or Path(image).suffix != ".sqsh":
            raise TerminalInfrastructureError("pin an existing Enroot .sqsh in protocol.image")
        self.base_path = Path(image).resolve()
        self.base_image = self._identity(self.base_path)
        probe = self._run(["start", "--net", "--pid", "--ipc", str(self.base_path),
            "/bin/bash", "-c", "test ! -e /tests/test.sh && test ! -e /solution && "
            "test ! -e /logs/verifier/reward.txt && test ! -e /home/ljj/.config/cave-mem/teacher.env"],
            allowed=(0, 1))
        if probe["exit_code"]:
            raise TerminalInfrastructureError("Enroot actor image exposes verifier, solution, or credentials")
        return self.base_image

    def _start(self, image, command, protocol, *, name, verifier=False, task=None, logs=None):
        self._run(["create", "-n", name, str(self._path(image))], timeout=protocol["build_timeout"])
        if verifier:
            self._run(["start", "--rw", "--root", "--net", "--pid", "--ipc", name,
                       "/bin/mkdir", "-p", "/tests", "/logs/verifier"])
        mounts = [] if not verifier else ["--mount", str(logs.parent / "tests") + ":/tests",
                                         "--mount", str(logs) + ":/logs/verifier"]
        env = {**protocol.get("environment", {}),
               **(task.spec["task_config"].get("verifier", {}).get("env", {}) if verifier else {})}
        variables = []
        for key, value in sorted(env.items()):
            if not isinstance(value, str) or "${" in value:
                raise ValueError("Enroot requires resolved non-secret environment values")
            variables.extend(["--env", key + "=" + value])
        if verifier:
            script = "cd -- " + shlex.quote(protocol["workdir"]) + " && " + command
        else:
            script = ("mkdir -p /app/.cave-tmp && cp -a /app/.cave-tmp/. /tmp/ || exit 125; "
                      "cd -- " + shlex.quote(protocol["workdir"]) + " || exit 125; "
                      "bash -c " + shlex.quote(command) + "; status=$?; "
                      "rm -rf /app/.cave-tmp && mkdir /app/.cave-tmp && "
                      "cp -a /tmp/. /app/.cave-tmp/ || exit 125; exit $status")
        return self._run(["start", "--rw", "--root", "--pid", "--ipc",
                          *([] if verifier and protocol["verifier_network"] == "bridge" else ["--net"]),
                          *mounts, *variables, name, "/bin/bash", "-c", script],
                         timeout=protocol["verifier_timeout" if verifier else "command_timeout"],
                         allowed=tuple(range(256)), limit=protocol["output_limit"])

    def execute(self, image, command, protocol, *, seed=0):
        name = "cave-" + uuid.uuid4().hex
        created = False
        try:
            result = self._start(image, command, protocol, name=name)
            created = True
            if result["exit_code"] == 125:
                raise TerminalInfrastructureError("Enroot filesystem boundary synchronization failed")
            with tempfile.NamedTemporaryFile(dir=self.snapshot_dir, suffix=".sqsh", delete=False) as stream:
                temporary = Path(stream.name)
            try:
                temporary.unlink()
                self._run(["export", "-o", str(temporary), name], timeout=protocol["build_timeout"])
                identity = self._identity(temporary)
                destination = self._path(identity)
                if destination.exists():
                    temporary.unlink()
                else:
                    temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            return identity, result
        finally:
            self._run(["remove", "-f", name], allowed=(0, 1))

    def verify(self, image, task, protocol):
        task.verify_assets()
        name = "cave-verifier-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="cave-verifier-") as tmp:
            shutil.copytree(task.task_dir / "tests", Path(tmp) / "tests")
            logs = Path(tmp) / "logs"
            logs.mkdir()
            try:
                execution = self._start(image, "bash /tests/test.sh", protocol, name=name,
                                        verifier=True, task=task, logs=logs)
                if execution["exit_code"] not in (0, 1):
                    raise TerminalInfrastructureError("verifier exited outside declared test outcomes")
                try:
                    report = parse_verifier_result(logs, success_threshold=protocol["success_threshold"],
                                                   require_ctrf=protocol["require_ctrf"])
                except TerminalInfrastructureError as exc:
                    raise TerminalInfrastructureError(str(exc) + "; verifier stderr: " + execution["stderr"]) from exc
                report.update(execution=execution, network=protocol["verifier_network"])
                task.verify_assets()
                return report
            finally:
                self._run(["remove", "-f", name], allowed=(0, 1))
