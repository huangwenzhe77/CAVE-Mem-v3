import copy
import json
import re
import tempfile
import unittest
from pathlib import Path

from cave_mem_v3.common.contracts import StudySpec, digest
from cave_mem_v3.agents.providers import OpenAIConfig, OpenAICompatibleProvider, ProviderActor
from cave_mem_v3.agents.runtime import run_episode
from cave_mem_v3.environments.terminal import (PROTOCOL, DockerBackend, TerminalEnvironment,
    TerminalInfrastructureError, parse_verifier_result, task_from_directory)


class MemoryBackend:
    """Injected immutable image store; no Docker or network access."""
    identity = "test-immutable-filesystem"
    filesystem_boundaries = True

    def __init__(self):
        self.base = "sha256:" + "0" * 64
        self.images = {self.base: {}}
        self.verifications = 0

    def image_available(self, image):
        return image in self.images

    def ensure_image(self, task, protocol):
        task.verify_assets()
        return self.base

    def execute(self, image, command, protocol, *, seed=0):
        files = copy.deepcopy(self.images[image])
        code, out = 0, ""
        match = re.fullmatch(r"printf '%s' '([^']+)' > '([^']+)'", command)
        if match:
            files[match[2]] = match[1]
        elif command.startswith("cat '"):
            path = command[5:-1]
            out = files.get(path, "")
            code = 0 if path in files else 1
        elif command.startswith("test ! -e '"):
            code = int(command[11:-1] in files)
        elif command == "write_result":
            files["/app/result"] = "correct"
        else:
            code = 1
        next_image = "sha256:" + digest(files)
        self.images[next_image] = files
        return next_image, {"exit_code": code, "stdout": out, "stderr": "", "truncated": False}

    def verify(self, image, task, protocol):
        self.verifications += 1
        return {"success": self.images[image].get("/app/result") == "correct",
                "private_test_name": "oracle-secret-that-actor-must-not-see"}


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "environment").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "instruction.md").write_text("Write the correct result.", encoding="utf-8")
        (self.root / "task.toml").write_text('version = "1.0"\n', encoding="utf-8")
        (self.root / "environment" / "Dockerfile").write_text("FROM example:fixed\n", encoding="utf-8")
        (self.root / "tests" / "test.sh").write_text("# private grader\n", encoding="utf-8")
        self.task = task_from_directory(self.root, "task", repository_commit="a" * 40)

    def tearDown(self):
        self.temp.cleanup()

    def test_no_automatic_restore_certification(self):
        env = TerminalEnvironment(self.task, backend=MemoryBackend())
        self.assertFalse(env.capabilities()["restore_certified"])
        before = env.configuration()
        env.reset()
        self.assertEqual(before, env.configuration())

    def test_diagnostics_measure_files_and_leave_review_unapproved(self):
        env = TerminalEnvironment(self.task, {"paired_scope": PROTOCOL}, backend=MemoryBackend())
        env.reset()
        before = env.snapshot()
        audit = env.diagnose_restoration()
        self.assertEqual(env.snapshot(), before)
        self.assertTrue(audit["filesystem_probe"]["persisted_after_restore"])
        self.assertTrue(audit["filesystem_probe"]["branches_isolated"])
        self.assertEqual(audit["reviewer"], "")
        self.assertFalse(env.capabilities()["restore_certified"])
        audit.update(reviewer="test reviewer", review_notes="fixture scope reviewed")
        audit["reviewed_constraints"] = {key: True for key in audit["reviewed_constraints"]}
        approved = TerminalEnvironment(self.task, {"paired_scope": PROTOCOL, "audit": audit}, backend=env.backend)
        self.assertTrue(approved.capabilities()["restore_certified"])
        damaged = copy.deepcopy(audit)
        damaged["filesystem_probe"]["branches_isolated"] = False
        rejected = TerminalEnvironment(self.task, {"paired_scope": PROTOCOL, "audit": damaged}, backend=env.backend)
        self.assertFalse(rejected.capabilities()["restore_certified"])

    def test_restore_preserves_file_branch_without_contaminating_base(self):
        env = TerminalEnvironment(self.task, backend=MemoryBackend())
        env.reset()
        base = env.snapshot()
        env.step({"type": "shell", "command": "write_result"})
        updated = env.snapshot()
        other = TerminalEnvironment(self.task, backend=env.backend)
        other.restore(updated)
        self.assertEqual(other.snapshot(), updated)
        self.assertTrue(other.finalize().success)
        env.restore(base)
        self.assertFalse(env.finalize().success)

    def test_budget_grades_current_files_without_an_extra_actor_action(self):
        backend = MemoryBackend()
        env = TerminalEnvironment(self.task, backend=backend)
        provider = OpenAICompatibleProvider(OpenAIConfig(), transport=lambda _: {
            "choices": [{"message": {"content": '{"type":"shell","command":"write_result"}'}}],
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10}})
        trace = run_episode(env, ProviderActor(provider), spec=StudySpec(max_model_calls=1), seed=2)
        self.assertEqual(trace["status"], "terminal")
        self.assertEqual(trace["termination_reason"], "budget_exhausted")
        self.assertTrue(trace["success"])
        self.assertEqual(trace["costs"]["model_calls"], 1)
        self.assertEqual(trace["costs"]["tool_calls"], 1)
        self.assertEqual(backend.verifications, 1)
        self.assertNotIn("oracle-secret", str(trace["steps"][0]["actor_response"]["messages"]))
        self.assertIn("oracle-secret", str(trace["private_verifier"]))

    def test_asset_mutation_prevents_restoring(self):
        env = TerminalEnvironment(self.task, backend=MemoryBackend())
        env.reset()
        snapshot = env.snapshot()
        (self.root / "tests" / "test.sh").write_text("# changed grader\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "assets changed"):
            env.restore(snapshot)

    def test_missing_grader_output_is_infrastructure_error(self):
        with self.assertRaises(TerminalInfrastructureError):
            parse_verifier_result(self.root)

    def test_reward_without_executed_tests_is_not_a_verified_failure(self):
        (self.root / "reward.txt").write_text("0", encoding="utf-8")
        with self.assertRaisesRegex(TerminalInfrastructureError, "CTRF"):
            parse_verifier_result(self.root)

    def test_failed_assertion_and_grader_setup_error_are_distinct(self):
        (self.root / "reward.txt").write_text("0", encoding="utf-8")
        result = {"results": {"summary": {"tests": 1, "failed": 1},
                               "tests": [{"name": "result", "status": "failed"}]}}
        (self.root / "ctrf.json").write_text(json.dumps(result), encoding="utf-8")
        self.assertFalse(parse_verifier_result(self.root)["success"])
        result["results"]["tests"][0]["message"] = "setup failed with ImportError"
        (self.root / "ctrf.json").write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaises(TerminalInfrastructureError):
            parse_verifier_result(self.root)

    def test_actor_container_has_no_tests_mount(self):
        class Commands(DockerBackend):
            def __init__(self):
                self.commands = []
            def _run(self, arguments, **kwargs):
                self.commands.append(arguments)
                output = ""
                if arguments[0] == "inspect":
                    output = json.dumps({"Running": False, "OOMKilled": False, "Error": "", "ExitCode": 0})
                if arguments[0] == "commit":
                    output = "sha256:" + "b" * 64
                return {"exit_code": 0, "stdout": output, "stderr": "", "truncated": False}
        backend = Commands()
        backend.execute("sha256:" + "a" * 64, "echo hello", self.task.spec["protocol"])
        create = backend.commands[0]
        self.assertIn("--network", create)
        self.assertEqual(create[create.index("--network") + 1], "none")
        self.assertNotIn("--mount", create)
        self.assertNotIn("/tests", str(create))


if __name__ == "__main__":
    unittest.main()
