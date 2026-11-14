"""One-shot H20 actor service and bounded history collection."""
import os
import signal
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
run = root / "runs" / sys.argv[1]
python = sys.executable
env = {**os.environ, "PYTHONPATH": str(root / "src")}
env.pop("OPENAI_API_KEY", None)
stage = sys.argv[2] if len(sys.argv) > 2 else "collect"
if stage not in ("collect", "pilot"):
    raise ValueError("stage must be collect or pilot")
with (root / "configs/phase1/server.toml").open("rb") as stream:
    actor = tomllib.load(stream)["actor"]
command = [python, "-m", "vllm.entrypoints.openai.api_server",
           "--model", actor["local_model_path"],
           "--served-model-name", actor["model"],
           "--host", "127.0.0.1", "--port", "8000",
           "--gpu-memory-utilization", "0.4", "--max-model-len", "8192",
           "--guided-decoding-backend", "lm-format-enforcer"]
with (run / ("actor-vllm-" + stage + ".log")).open("x", encoding="utf-8") as log:
    server = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)
    try:
        for _ in range(180):
            if server.poll() is not None:
                raise RuntimeError("vLLM exited before becoming ready; inspect actor-vllm.log")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=2) as reply:
                    if reply.status == 200:
                        break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("vLLM readiness timed out")
        command = ([python, "-m", "cave_mem_v3.cli", "collect",
            "--config", str(root / "configs/phase1/server.toml"),
            "--manifest", str(run / "manifest-final.json"),
            "--output", str(run / "collection")] if stage == "collect" else
            [python, str(root / "scripts/phase1_pilot_replay.py"), sys.argv[1]])
        result = subprocess.run(command, cwd=root, env=env, timeout=3300,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (run / (stage + "-console.log")).write_text(result.stdout, encoding="utf-8")
        print(result.stdout)
        sys.exit(result.returncode)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
