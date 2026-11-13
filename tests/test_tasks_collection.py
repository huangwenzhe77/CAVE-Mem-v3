"""Leakage, stopping, immutable-artifact and restart checks for phase one."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from cave_mem_v3.memory_generation.collection import CollectionConfig, collect_history
from cave_mem_v3.common.contracts import StudySpec, digest, read_json, write_json
from cave_mem_v3.benchmarks.tasks import build_manifest, load_manifest, validate_manifest


class DummyActor:
    def __init__(self, version="actor-1"):
        self.version = version

    def configuration(self):
        return {"model": self.version}

    def capabilities(self):
        return {"stateless": True}


class DummyEnv:
    def __init__(self, task):
        self.task_id = task["task_id"]

    def configuration(self):
        return {"task_id": self.task_id, "protocol": "test"}

    def capabilities(self):
        return {"restore_certified": True}

    def close(self):
        pass


def runner_for(outcomes, calls):
    def runner(env, actor, **kwargs):
        calls.append((env.task_id, kwargs["seed"]))
        outcome = outcomes[(len(calls) - 1) % len(outcomes)]
        return {"status": "error" if outcome is None else "terminal", "success": outcome, "task_id": env.task_id,
                "utility": None if outcome is None else float(outcome), "costs": {"model_calls": 1}, "steps": [],
                "actor_config": actor.configuration(), "environment_config": env.configuration()}
    return runner


def identity_runner(identities, calls, outcomes=(True,)):
    base = runner_for(outcomes, calls)

    def runner(env, actor, **kwargs):
        episode = base(env, actor, **kwargs)
        episode["response_identity"] = copy.deepcopy(identities[(len(calls) - 1) % len(identities)])
        episode["trace_hash"] = digest(episode)
        return episode
    return runner


def task(task_id, role="history", **extra):
    return {"task_id": task_id, "group_id": "group-" + task_id, "role": role, **extra}


class ManifestTests(unittest.TestCase):
    def test_shared_template_and_transitive_groups_cannot_cross_partitions(self):
        examples = [
            [task("a", template_id="same"), task("b", "test", template_id="same")],
            [task("a", groups=["linked-1"]), task("b", groups=["linked-1", "linked-2"]),
             task("c", "target_train", groups=["linked-2"])],
            [task("a", "validation", validation_role="calibration", groups=["shared"]),
             task("b", "validation", validation_role="selection", groups=["shared"])],
        ]
        for tasks in examples:
            with self.subTest(tasks=tasks), self.assertRaisesRegex(ValueError, "crosses partitions"):
                build_manifest(tasks, benchmark="unit", revision="v1")

    def test_outcomes_duplicates_and_missing_validation_subrole_rejected(self):
        for tasks in ([task("a", reward=1)], [task("a"), task("a")], [task("a", "validation")]):
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                build_manifest(tasks, benchmark="unit", revision="v1")

    def test_group_split_deterministic_and_never_splits_connected_tasks(self):
        tasks = [{"task_id": str(i), "groups": ["g" + str(i // 2)]} for i in range(24)]
        ratios = {"history": .25, "target_train": .25, "validation": .25, "test": .25}
        first = build_manifest(tasks, benchmark="unit", revision="v1", split_ratios=ratios)
        second = build_manifest(list(reversed(tasks)), benchmark="unit", revision="v1", split_ratios=ratios)
        self.assertEqual(first, second)
        self.assertEqual(set(first["role_counts"]), set(ratios))
        self.assertEqual({t["validation_role"] for t in first["tasks"] if t["role"] == "validation"}, {"calibration", "selection"})
        by_group = {}
        for t in first["tasks"]:
            by_group.setdefault(t["groups"][0], set()).add((t["role"], t.get("validation_role")))
        self.assertTrue(all(len(roles) == 1 for roles in by_group.values()))

    def test_same_mechanism_alone_is_not_a_leakage_link(self):
        manifest = build_manifest([task("a", mechanism="backup"), task("b", "test", mechanism="backup")],
                                  benchmark="unit", revision="v1")
        self.assertEqual(manifest["independent_groups"], 2)

    def test_asset_changes_and_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "task"
            asset.mkdir()
            (asset / "instruction.txt").write_text("v1", encoding="utf-8")
            manifest = build_manifest([task("a", path=str(asset))], benchmark="unit", revision="v1")
            self.assertEqual(manifest["tasks"][0]["asset_state"], "verified")
            write_json(root / "manifest.json", manifest)
            self.assertEqual(load_manifest(root / "manifest.json"), manifest)
            (asset / "instruction.txt").write_text("v2", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "assets changed"):
                validate_manifest(manifest, verify_assets=True)
            manifest["tasks"][0]["role"] = "test"
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_manifest(manifest)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "run"
        self.manifest = build_manifest([task("a"), task("b"), task("reserved", "test")], benchmark="unit", revision="v1")
        self.spec = StudySpec(max_steps=4)

    def collect(self, config, outcomes=(False,), calls=None, **extra):
        calls = [] if calls is None else calls
        return collect_history(self.manifest, config, self.spec, DummyEnv, DummyActor, self.output,
                               episode_runner=runner_for(outcomes, calls), **extra)

    def test_fixed_runs_keeps_success_failure_error_and_never_touches_test(self):
        calls = []
        report = self.collect(CollectionConfig(runs_per_task=3, max_runs=10), (True, False, None), calls)
        self.assertEqual((report["successful_runs"], report["failed_runs"], report["infrastructure_errors"]), (2, 2, 2))
        self.assertEqual(report["stop_reason"], "fixed_schedule_complete")
        self.assertEqual(report["failure_rate_observed"], .5)
        self.assertEqual(set(t for t, _ in calls), {"a", "b"})
        self.assertNotEqual(calls[0][0], calls[1][0])
        self.assertEqual(len(list((self.output / "traces").glob("*.json"))), 6)

    def test_failure_target_ignores_infrastructure_errors_and_honors_cap(self):
        report = self.collect(CollectionConfig(mode="failure_target", runs_per_task=3, failure_target=2, max_runs=5), (None, False, True))
        self.assertEqual(report["attempted_runs"], 5)
        self.assertEqual(report["failed_runs"], 2)
        self.assertEqual(report["stop_reason"], "failure_target_reached")
        other = self.output.parent / "cap"
        capped = collect_history(self.manifest, CollectionConfig(mode="failure_target", runs_per_task=2, failure_target=2, max_runs=3),
                                 self.spec, DummyEnv, DummyActor, other, episode_runner=runner_for((None, True), []))
        self.assertEqual(capped["failed_runs"], 0)
        self.assertEqual(capped["stop_reason"], "max_runs_reached")
        self.assertEqual(capped["limits_reached"], ["max_runs"])

    def test_failure_target_cannot_exceed_per_task_attempt_limit(self):
        calls = []
        config = CollectionConfig(mode="failure_target", runs_per_task=1, failure_target=50, max_runs=100)
        report = self.collect(config, (False, None), calls)
        self.assertEqual(report["nominal_runs"], 2)
        self.assertEqual(report["planned_runs"], 2)
        self.assertEqual(report["attempted_runs"], 2)
        self.assertEqual(report["failed_runs"], 1)
        self.assertEqual(report["infrastructure_errors"], 1)
        self.assertEqual(report["per_task_attempts"], {"a": 1, "b": 1})
        self.assertEqual(report["stop_reason"], "per_task_limit_reached")
        self.assertEqual(report["limits_reached"], ["runs_per_task"])
        self.assertEqual(self.collect(config, (False, None), calls), report)
        self.assertEqual(len(calls), 2)

    def test_failure_target_stops_early_and_reports_simultaneous_hard_limits(self):
        report = self.collect(CollectionConfig(mode="failure_target", runs_per_task=4, failure_target=2, max_runs=10))
        self.assertEqual(report["attempted_runs"], 2)
        self.assertEqual(report["stop_reason"], "failure_target_reached")
        self.assertEqual(report["limits_reached"], [])
        other = self.output.parent / "both-limits"
        capped = collect_history(self.manifest, CollectionConfig(mode="failure_target", runs_per_task=1, failure_target=2, max_runs=2),
                                 self.spec, DummyEnv, DummyActor, other, episode_runner=runner_for((None,), []))
        self.assertEqual(capped["attempted_runs"], 2)
        self.assertEqual(capped["failed_runs"], 0)
        self.assertEqual(capped["infrastructure_errors"], 2)
        self.assertEqual(capped["stop_reason"], "per_task_limit_reached")
        self.assertEqual(capped["limits_reached"], ["runs_per_task", "max_runs"])

    def test_fixed_schedule_cap_does_not_satisfy_full_per_task_plan(self):
        report = self.collect(CollectionConfig(runs_per_task=4, max_runs=3))
        self.assertEqual(report["nominal_runs"], 8)
        self.assertEqual(report["planned_runs"], 3)
        self.assertEqual(report["stop_reason"], "max_runs_reached")
        self.assertEqual(sorted(report["per_task_attempts"].values()), [1, 2])

    def test_resume_reads_completed_traces_without_new_sampling(self):
        config = CollectionConfig(runs_per_task=2, max_runs=5)
        calls = []
        first = self.collect(config, (True, False), calls)
        second = self.collect(config, (True, False), calls)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 4)
        with self.assertRaisesRegex(ValueError, "identity"):
            collect_history(self.manifest, config, self.spec, DummyEnv, lambda: DummyActor("different"), self.output,
                            episode_runner=runner_for((False,), []))

    def test_corrupt_trace_and_identity_changes_are_not_overwritten(self):
        config = CollectionConfig(runs_per_task=1)
        self.collect(config)
        path = next((self.output / "traces").glob("*.json"))
        record = read_json(path)
        record["task_id"] = "reserved"
        record["trace_hash"] = digest({key: value for key, value in record.items() if key != "trace_hash"})
        write_json(path, record)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "identity"):
            self.collect(config)
        self.assertEqual(path.read_bytes(), original)

    def test_interrupted_paid_attempt_is_not_silently_repeated(self):
        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            collect_history(self.manifest, CollectionConfig(), self.spec, DummyEnv, DummyActor,
                            self.output, episode_runner=interrupt)
        with self.assertRaisesRegex(ValueError, "incomplete attempted episode"):
            self.collect(CollectionConfig())

    def test_drift_inside_collection_is_rejected(self):
        actor_calls = []
        def actor_factory():
            actor_calls.append(1)
            return DummyActor("first" if len(actor_calls) == 1 else "changed")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            collect_history(self.manifest, CollectionConfig(), self.spec, DummyEnv, actor_factory,
                            self.output, episode_runner=runner_for((True,), []))

    def test_cross_episode_fingerprint_drift_preserves_observation_and_cost(self):
        identities = [{"model": "served-model", "fingerprint": fingerprint, "fingerprint_status": "reported"}
                      for fingerprint in ("A", "B")]
        calls = []
        config = CollectionConfig(runs_per_task=1)
        runner = identity_runner(identities, calls, outcomes=(True, False))
        report = collect_history(self.manifest, config, self.spec, DummyEnv, DummyActor,
                                 self.output, episode_runner=runner)
        self.assertEqual((report["successful_runs"], report["failed_runs"], report["infrastructure_errors"]), (1, 0, 1))
        self.assertEqual(report["response_identity_errors"], 1)
        drifting = read_json(sorted((self.output / "traces").glob("*.json"))[1])["episode"]
        self.assertEqual(drifting["status"], "error")
        self.assertIsNone(drifting["success"])
        self.assertIsNone(drifting["utility"])
        self.assertEqual(drifting["response_identity"]["fingerprint"], "B")
        self.assertEqual(drifting["observed_outcome_before_identity_check"]["success"], False)
        self.assertEqual(drifting["costs"]["model_calls"], 1)
        self.assertEqual(drifting["trace_hash"], digest({key: value for key, value in drifting.items() if key != "trace_hash"}))
        resumed = collect_history(self.manifest, config, self.spec, DummyEnv, DummyActor,
                                  self.output, episode_runner=runner)
        self.assertEqual(report, resumed)
        self.assertEqual(len(calls), 2)

    def test_resume_detects_identity_drift_even_with_valid_trace_checksums(self):
        identity = {"model": "served-model", "fingerprint": "A", "fingerprint_status": "reported"}
        config = CollectionConfig(runs_per_task=1)
        collect_history(self.manifest, config, self.spec, DummyEnv, DummyActor, self.output,
                        episode_runner=identity_runner([identity], []))
        path = sorted((self.output / "traces").glob("*.json"))[1]
        trace = read_json(path)
        trace["episode"]["response_identity"]["fingerprint"] = "B"
        trace["episode"].pop("trace_hash")
        trace["episode"]["trace_hash"] = digest(trace["episode"])
        trace.pop("trace_hash")
        trace["trace_hash"] = digest(trace)
        write_json(path, trace)
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "inconsistent response identity"):
            self.collect(config)
        self.assertEqual(path.read_bytes(), original)

    def test_unknown_fingerprint_is_not_invented_but_known_models_must_match(self):
        identities = [
            {"model": "served-model", "fingerprint": None, "fingerprint_status": "unknown"},
            {"model": "served-model", "fingerprint": "A", "fingerprint_status": "reported"},
            {"model": "served-model", "fingerprint": None, "fingerprint_status": "unknown"},
            {"model": "different-model", "fingerprint": None, "fingerprint_status": "unknown"},
        ]
        report = collect_history(self.manifest, CollectionConfig(runs_per_task=2), self.spec, DummyEnv,
                                 DummyActor, self.output, episode_runner=identity_runner(identities, []))
        self.assertEqual(report["successful_runs"], 3)
        self.assertEqual(report["response_identity_errors"], 1)
        traces = [read_json(path) for path in sorted((self.output / "traces").glob("*.json"))]
        for index in (0, 2):
            self.assertEqual(traces[index]["episode"]["response_identity"], identities[index])
        self.assertEqual(traces[3]["episode"]["error"]["conflicting_fields"], ["model"])

    def test_no_history_cannot_collect_target_set(self):
        manifest = build_manifest([task("a", "test")], benchmark="unit", revision="v1")
        with self.assertRaisesRegex(ValueError, "history tasks"):
            collect_history(manifest, CollectionConfig(), self.spec, DummyEnv, DummyActor, self.output)


if __name__ == "__main__":
    unittest.main()
