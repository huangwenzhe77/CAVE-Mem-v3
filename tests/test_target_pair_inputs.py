"""Software fixtures for frozen paired-target admission; never benchmark evidence."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, digest, read_json, stage_seed, write_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.interventions.inputs import load_prepared_run, seal
from cave_mem_v3.memory_generation.bank import build_memory, freeze_bank
from cave_mem_v3.memory_generation.replay import ReplaySpec, verify_candidate
from cave_mem_v3.target_data.collection import collect_target
from cave_mem_v3.target_data.sampling import RiskHazard
from test_candidates import repair, source_trace
from test_target_prepare import EncoderFixture


def _legacy_evidence(evidence):
    old = copy.deepcopy(evidence)
    plan = old["plan"]
    plan["schema_version"] = "source-replay-plan.v5"
    plan.pop("decision_rule")
    old["plan"] = seal(plan, "plan_hash")
    for row in old["pairs"]:
        row["schema_version"] = "source-replay-pair.v5"
        row["plan_hash"] = old["plan"]["plan_hash"]
        row["sample_id"] = digest([row["plan_hash"], row["replicate"], row["seed"]])
        row["record_hash"] = digest({k: v for k, v in row.items() if k != "record_hash"})
    old["schema_version"] = "source-evidence.v5"
    old["decision_rule"] = "complete_valid_pairs_and_mean_difference_gt_acceptance_epsilon"
    old.pop("n_successful_corrections")
    old.pop("all_corrections_successful")
    return seal(old, "evidence_hash")


def make_prepared_fixture(root, *, target_safe=False, target_max_calls=4,
                          replay_pairs=2, legacy=False, empty_bank=False, select=True,
                          manifest=None, source_manifest_hash=None, source_group_id=None,
                          metadata=None, hazard=None, base_seed=42, target_spec=None):
    """Collect source and target artifacts from one frozen fixture manifest."""
    root = Path(root)
    manifest = manifest or build_manifest([
        {"task_id": "source-a", "group_id": "source-family-a", "role": "history"},
        {"task_id": "target-a", "group_id": "target-family-a", "role": "target_train"},
    ], benchmark="preparation-fixture", revision="fixture-v3")
    trace = source_trace()
    trace["manifest_hash"] = source_manifest_hash or manifest["manifest_hash"]
    if source_group_id is not None:
        trace["group_id"] = source_group_id
    trace = seal(trace, "trace_hash")
    if empty_bank:
        bank = seal({"schema_version": "verified-source-bank.v3", "frozen": True,
                     "memory_count": 0, "memories": [], "audit_sources": {}}, "bank_hash")
    else:
        candidate = repair(trace)
        evidence = verify_candidate(candidate, trace,
            lambda: PreparationEnvironment(task_id="source-a"), ScriptedActor,
            study=StudySpec(max_steps=4, max_model_calls=4),
            replay_spec=ReplaySpec(replay_pairs=replay_pairs))
        if legacy:
            evidence = _legacy_evidence(evidence)
        bank = freeze_bank([build_memory(candidate, trace, evidence)], root / "bank.json",
                           source_traces=[trace], metadata=metadata)
    target_dir = root / "prepared"

    def env_factory(task):
        return PreparationEnvironment(task_id=task["task_id"], safe_intent=target_safe)

    if hazard is None:
        probability = .99 if select else 1e-9
        hazard = RiskHazard(low_risk_probability=probability, high_risk_probability=probability)
    collect_target(manifest, role="target_train",
        spec=target_spec or StudySpec(max_steps=4, max_model_calls=target_max_calls, base_seed=base_seed),
        env_factory=env_factory, actor_factory=ScriptedActor, output_dir=target_dir,
        hazard=hazard, prepare_before_action=True, bank=bank, encoder=EncoderFixture())
    return {"bank": bank, "target_dir": target_dir, "manifest": manifest,
            "env_factory": env_factory, "actor_factory": ScriptedActor,
            "trace_path": next((target_dir / "traces").glob("*.json")),
            "source_trace": trace}


def rewrite_trace(fixture, mutate):
    """Reseal a deliberately changed fixture to exercise semantic validation."""
    trace = read_json(fixture["trace_path"])
    mutate(trace)
    for step in trace["episode"].get("steps", []):
        if "pre_treatment_state" in step:
            state = step["pre_treatment_state"]
            state["x_hash"] = digest(state["x"])
            step["pre_treatment_state"] = seal(state)
    trace["episode"] = seal(trace["episode"], "trace_hash")
    write_json(fixture["trace_path"], seal(trace, "trace_hash"))


class TargetPairInputTests(unittest.TestCase):
    def test_same_manifest_source_history_and_target_are_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary)
            run, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "ready")
        self.assertEqual(row["remaining_model_calls"], 3)
        self.assertEqual(row["remaining_steps"], 4)
        self.assertEqual(row["a0"], {"type": "operate"})
        self.assertEqual(row["checkpoint_hash"], digest(row["checkpoint"]))
        self.assertEqual(row["x_hash"], digest(row["x"]))
        self.assertEqual(fixture["source_trace"]["manifest_hash"], run["manifest_hash"])
        self.assertNotIn("success", row)
        self.assertNotIn("utility", row)

    def test_cross_manifest_bank_rejected_even_when_bank_is_internally_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary, source_manifest_hash="different-frozen-manifest")
            with self.assertRaisesRegex(ValueError, "source manifest differs"):
                load_prepared_run(fixture["target_dir"], fixture["bank"])

    def test_source_must_be_registered_history_and_match_group_identity(self):
        cases = [
            ([{"task_id": "other-source", "group_id": "history", "role": "history"}],
             {}, "history partition"),
            ([{"task_id": "source-a", "group_id": "source-family-a", "role": "test"}],
             {}, "history partition"),
            ([{"task_id": "source-a", "group_id": "source-family-a", "role": "history"}],
             {"source_group_id": "different-group"}, "task identity"),
        ]
        for sources, options, message in cases:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                manifest = build_manifest(sources + [{"task_id": "target-a",
                    "group_id": "target-family-a", "role": "target_train"}],
                    benchmark="preparation-fixture", revision="fixture-v3")
                fixture = make_prepared_fixture(temporary, manifest=manifest, **options)
                with self.assertRaisesRegex(ValueError, message):
                    load_prepared_run(fixture["target_dir"], fixture["bank"])

    def test_optional_bank_manifest_metadata_must_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary, metadata={"manifest_hash": "other"})
            with self.assertRaisesRegex(ValueError, "metadata manifest differs"):
                load_prepared_run(fixture["target_dir"], fixture["bank"])

    def test_empty_bank_no_selection_and_exhausted_calls_remain_in_denominator(self):
        for options, reason in (({"empty_bank": True}, "no_memory"),
                                ({"select": False}, "no_selection"),
                                ({"target_max_calls": 1}, "insufficient_budget")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                fixture = make_prepared_fixture(temporary, **options)
                _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
                self.assertEqual(len(rows), 1)
                self.assertEqual((rows[0]["status"], rows[0]["reason"]), ("excluded", reason))
                self.assertNotIn("success", rows[0])
                self.assertNotIn("utility", rows[0])

    def test_missing_and_corrupt_preparation_are_blocked_not_excluded(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt), tempfile.TemporaryDirectory() as temporary:
                fixture = make_prepared_fixture(temporary)
                if corrupt:
                    trace = read_json(fixture["trace_path"])
                    trace["episode"]["seed"] += 1
                    write_json(fixture["trace_path"], trace)
                else:
                    fixture["trace_path"].unlink()
                _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
                self.assertEqual(rows[0]["status"], "blocked")
                self.assertEqual(rows[0]["reason"], "invalid_prepared_input" if corrupt else "preparation_missing")

    def test_resealed_budget_action_and_outcome_tampering_are_blocked(self):
        def change_budget(trace):
            trace["episode"]["steps"][0]["pre_treatment_state"]["x"]["b"]["remaining"]["environment_steps"] -= 1

        def execute_selected(trace):
            trace["episode"]["steps"][0]["result"] = {"terminal": True, "success": False}

        def change_action(trace):
            trace["episode"]["steps"][0]["actor_response"]["action"] = {"type": "finish"}

        for mutate in (change_budget, execute_selected, change_action):
            with self.subTest(mutate=mutate.__name__), tempfile.TemporaryDirectory() as temporary:
                fixture = make_prepared_fixture(temporary)
                rewrite_trace(fixture, mutate)
                _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
                self.assertEqual((rows[0]["status"], rows[0]["reason"]), ("blocked", "invalid_prepared_input"))

    def test_no_memory_budget_tampering_is_not_hidden_by_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary, empty_bank=True)
            rewrite_trace(fixture, lambda trace: trace["episode"]["steps"][0]["pre_treatment_state"]["x"]["b"].update(status="unknown"))
            _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
            self.assertEqual(rows[0]["status"], "blocked")

    def test_old_rule_and_nonpilot_k_are_blocked_without_rewriting_bank(self):
        for options in ({"legacy": True}, {"replay_pairs": 3}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as temporary:
                fixture = make_prepared_fixture(temporary, **options)
                before = copy.deepcopy(fixture["bank"])
                _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
                self.assertEqual((rows[0]["status"], rows[0]["reason"]), ("blocked", "invalid_prepared_input"))
                self.assertEqual(fixture["bank"], before)

    def test_later_selected_state_keeps_correct_remaining_step_budget(self):
        base_seed = next(seed for seed in range(100)
            if (lambda rng: rng.random() >= .5 and rng.random() < .5)(
                random.Random(stage_seed(seed, "target-a", 0, "target_rollout"))))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary, target_safe=True, base_seed=base_seed,
                hazard=RiskHazard(low_risk_probability=.5, high_risk_probability=.5))
            _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
            self.assertEqual(rows[0]["status"], "ready")
            self.assertEqual(rows[0]["remaining_model_calls"], 2)
            self.assertEqual(rows[0]["remaining_steps"], 3)
            self.assertEqual(rows[0]["shared_costs"]["tool_calls"], 1)

    def test_resealed_false_no_selection_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary)
            rewrite_trace(fixture, lambda trace: trace["episode"]["checkpoint_sampling"].update(status="no_selection"))
            _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
            self.assertEqual(rows[0]["status"], "blocked")

    def test_impossible_selected_index_is_blocked_even_with_zero_remaining_budget(self):
        from cave_mem_v3.interventions.inputs import _prepared
        base_seed = next(seed for seed in range(100)
            if (lambda rng: rng.random() >= .5 and rng.random() < .5)(
                random.Random(stage_seed(seed, "target-a", 0, "target_rollout"))))
        with tempfile.TemporaryDirectory() as temporary:
            fixture = make_prepared_fixture(temporary, target_safe=True, base_seed=base_seed,
                hazard=RiskHazard(low_risk_probability=.5, high_risk_probability=.5))
            run = read_json(fixture["target_dir"] / "run.json")
            trace = read_json(fixture["trace_path"])
            run["study"]["max_steps"] = 1
            trace["episode"]["spec"] = copy.deepcopy(run["study"])
            state = trace["episode"]["steps"][1]["pre_treatment_state"]
            state["x"]["b"]["remaining"]["environment_steps"] = 0
            state["x_hash"] = digest(state["x"])
            trace["episode"]["steps"][1]["pre_treatment_state"] = seal(state)
            trace["episode"] = seal(trace["episode"], "trace_hash")
            task = next(task for task in run["manifest"]["tasks"] if task["task_id"] == "target-a")
            with self.assertRaisesRegex(ValueError, "exceeds the frozen step budget"):
                _prepared(run, trace, task, fixture["bank"], {})

    def test_resealed_shared_cost_changes_are_rejected(self):
        for field in ("model_calls", "physical_attempts", "input_tokens", "output_tokens",
                      "cached_input_tokens", "api_cost", "tool_calls", "tool_cost", "wall_seconds"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                fixture = make_prepared_fixture(temporary)
                def change_cost(trace):
                    costs = trace["episode"]["costs"]
                    costs[field] = -1 if field == "wall_seconds" else costs[field] + 1
                rewrite_trace(fixture, change_cost)
                _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
                self.assertEqual(rows[0]["status"], "blocked")
                self.assertIn("shared " if field != "wall_seconds" else "must be >= 0", rows[0]["detail"])

    def test_shared_usage_is_recomputed_with_unknown_charges_preserved(self):
        for unknown_fee in (False, True):
            with self.subTest(unknown_fee=unknown_fee), tempfile.TemporaryDirectory() as temporary:
                fixture = make_prepared_fixture(temporary)
                def record_charges(trace):
                    usage = trace["episode"]["steps"][0]["actor_response"]["usage"]
                    usage.update(physical_attempts=2, input_tokens=17, output_tokens=7,
                                 cached_input_tokens=5, api_cost=None if unknown_fee else .25)
                    trace["episode"]["costs"].update(usage)
                    trace["episode"]["costs"]["wall_seconds"] = .125
                rewrite_trace(fixture, record_charges)
                _, rows = load_prepared_run(fixture["target_dir"], fixture["bank"])
                self.assertEqual(rows[0]["status"], "ready")
                self.assertEqual(rows[0]["shared_costs"]["input_tokens"], 17)
                self.assertEqual(rows[0]["shared_costs"]["physical_attempts"], 2)
                self.assertEqual(rows[0]["shared_costs"]["api_cost"], None if unknown_fee else .25)


if __name__ == "__main__":
    unittest.main()
