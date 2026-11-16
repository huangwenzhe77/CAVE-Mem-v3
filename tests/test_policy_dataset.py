"""Dataset software checks; synthetic fixture labels are not research results."""
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.interventions.single import collect_single_target
from cave_mem_v3.policy.dataset import build_policy_dataset, load_policy_dataset, write_policy_dataset
from cave_mem_v3.cli import main
from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.target_data.sampling import RiskHazard
from test_target_pair_inputs import make_prepared_fixture, rewrite_trace
from test_target_prepare import EncoderFixture


def paired(root, **kwargs):
    fixture = make_prepared_fixture(root / "inputs", **kwargs)
    collect_target_pairs(fixture["target_dir"], fixture["bank"],
        env_factory=fixture["env_factory"], actor_factory=fixture["actor_factory"], output_dir=root / "paired")
    return fixture, {"kind": "paired", "run": str(root / "paired"), "prepared": str(fixture["target_dir"])}


def single(root, fixture, *, spec=None, actor=ScriptedActor, hazard=.99):
    collect_single_target(fixture["manifest"], bank=fixture["bank"],
        spec=spec or StudySpec(max_steps=4, max_model_calls=4),
        env_factory=fixture["env_factory"], actor_factory=actor, output_dir=root / "single",
        encoder=EncoderFixture(), assignment_probability=.5,
        hazard=RiskHazard(low_risk_probability=hazard, high_risk_probability=hazard))
    return {"kind": "single", "run": str(root / "single")}


class PolicyDatasetTests(unittest.TestCase):
    def test_paired_and_single_are_reproducible_readonly_and_label_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, pair = paired(root)
            one = single(root, fixture)
            before = {str(p): p.read_bytes() for p in root.rglob("*.json")}
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("no actor")), \
                    patch("cave_mem_v3.cli._env_factory", side_effect=AssertionError("no environment")):
                data = build_policy_dataset([pair, one])
                self.assertEqual(data, build_policy_dataset([one, pair]))
            self.assertEqual(before, {str(p): p.read_bytes() for p in root.rglob("*.json")})
            self.assertEqual(data["summary"]["planned"], 2)
            self.assertEqual(data["available_roles"], ["target_train"])
            pair_row = next(row for row in data["rows"] if row["kind"] == "paired")
            single_row = next(row for row in data["rows"] if row["kind"] == "single")
            self.assertTrue(pair_row["label_masks"]["advantage"])
            self.assertEqual(pair_row["labels"]["H"], 0)
            self.assertNotIn("H", single_row["labels"])
            self.assertNotIn("D", single_row["labels"])
            self.assertNotIn("labels", pair_row["features"])
            self.assertNotIn("task_id", pair_row["features"]["context"])
            self.assertEqual(pair_row["grouping"]["target_group"], single_row["grouping"]["target_group"])
            self.assertEqual(pair_row["grouping"]["source_target_group"], single_row["grouping"]["source_target_group"])
            self.assertIsNone(pair_row["assignment"]["propensity"])
            self.assertEqual(single_row["assignment"]["propensity"], .5)

    def test_export_reload_and_cli_never_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = paired(root)
            output = root / "dataset.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main(["build-policy-dataset", "--paired", source["run"], source["prepared"],
                             "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["planned"], 1)
            self.assertEqual(load_policy_dataset(output), read_json(output))
            saved = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "exists"):
                write_policy_dataset([source], output)
            self.assertEqual(output.read_bytes(), saved)

    def test_resealed_dataset_labels_do_not_replace_raw_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = paired(root)
            path = root / "dataset.json"
            data = write_policy_dataset([source], path)
            data["rows"][0]["labels"]["H"] = 1
            data["rows"][0] = seal(data["rows"][0])
            write_json(path, seal(data))
            with self.assertRaisesRegex(ValueError, "original audited sources"):
                load_policy_dataset(path)

    def test_raw_pair_input_forgery_is_rejected_before_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = paired(root)
            path = root / "paired" / "run.json"
            run = read_json(path)
            run["inputs"][0]["a0"] = {"type": "finish"}
            run["inputs"][0] = seal(run["inputs"][0])
            write_json(path, seal(run))
            with patch("cave_mem_v3.policy.dataset.summarize_target_pairs", side_effect=AssertionError("no outcomes")):
                with self.assertRaisesRegex(ValueError, "original prepared traces"):
                    build_policy_dataset([source])

    def test_test_role_refused_before_reading_outcomes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, _ = paired(root)
            source = single(root, fixture)
            path = root / "single" / "run.json"
            run = read_json(path)
            run["role"] = "test"
            write_json(path, seal(run))
            with patch("cave_mem_v3.policy.dataset.summarize_target_single", side_effect=AssertionError("no outcomes")):
                with self.assertRaisesRegex(ValueError, "outcomes are forbidden"):
                    build_policy_dataset([source])

    def test_duplicate_copied_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = paired(root)
            copy_dir = root / "copied"
            shutil.copytree(source["run"], copy_dir)
            duplicate = {**source, "run": str(copy_dir)}
            with self.assertRaisesRegex(ValueError, "duplicate source run"):
                build_policy_dataset([source, duplicate])

    def test_output_cannot_pollute_raw_input_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, source = paired(root)
            for key in ("run", "prepared"):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "outside raw"):
                    write_policy_dataset([source], Path(source[key]) / "dataset.json")

    def test_unknown_fee_keeps_pair_harm_without_advantage(self):
        class UnknownFee(ScriptedActor):
            def propose_target(self, context, **kwargs):
                response = super().propose_target(context, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = StudySpec(max_steps=4, max_model_calls=4, inference_weights={"api_cost": 1.0})
            fixture = make_prepared_fixture(root / "inputs", target_spec=spec)
            collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                actor_factory=UnknownFee, output_dir=root / "paired")
            source = {"kind": "paired", "run": str(root / "paired"), "prepared": str(fixture["target_dir"])}
            row = build_policy_dataset([source])["rows"][0]
            self.assertTrue(row["label_masks"]["harm"])
            self.assertFalse(row["label_masks"]["advantage"])
            self.assertIsNone(row["labels"]["D"])
            self.assertIsNone(row["costs"]["delta"]["api_cost"])

    def test_no_memory_no_selection_and_budget_exclusions_keep_denominator(self):
        for kwargs, reason in (({"empty_bank": True}, "no_memory"), ({"select": False}, "no_selection"),
                               ({"target_max_calls": 1}, "insufficient_budget")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                _, source = paired(Path(tmp), **kwargs)
                dataset = build_policy_dataset([source])
                self.assertEqual(dataset["summary"]["planned"], 1)
                self.assertEqual(dataset["rows"][0]["reason"], reason)
                self.assertFalse(dataset["rows"][0]["label_masks"]["harm"])
                self.assertIsNone(dataset["rows"][0]["labels"]["H"])

    def test_pending_single_retains_unknown_and_actual_costs(self):
        class Timeout(ScriptedActor):
            def propose(self, *args, **kwargs):
                raise TimeoutError("unknown completion")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "inputs")
            source = single(root, fixture, actor=Timeout)
            data = build_policy_dataset([source])
            row = data["rows"][0]
            self.assertEqual(row["status"], "pending")
            self.assertIsNone(row["labels"]["Y"])
            self.assertFalse(row["label_masks"]["observed_reward"])
            self.assertEqual(row["costs"]["observed"]["model_calls"], 1)

    def test_incompatible_settings_are_refused_without_silent_pooling(self):
        for field in ("actor_config", "study", "encoder_identity"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture, pair = paired(root)
                one = single(root, fixture)
                path = root / "single" / "run.json"
                run = read_json(path)
                if field == "study":
                    run[field]["lambda_inference"] += .1
                else:
                    run[field]["test_drift"] = True
                write_json(path, seal(run))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    build_policy_dataset([pair, one])

    def test_missing_original_prepared_artifacts_are_not_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, source = paired(root)
            fixture["trace_path"].unlink()
            with self.assertRaisesRegex(ValueError, "original prepared traces"):
                build_policy_dataset([source])

    def test_nested_outcome_cannot_enter_frozen_feature_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "inputs")
            def inject(trace):
                trace["episode"]["steps"][-1]["pre_treatment_state"]["x"]["feature_map"]["terminal_success"] = True
            rewrite_trace(fixture, inject)
            collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                                 actor_factory=ScriptedActor, output_dir=root / "paired")
            source = {"kind": "paired", "run": str(root / "paired"), "prepared": str(fixture["target_dir"])}
            with self.assertRaisesRegex(ValueError, "unexpected field"):
                build_policy_dataset([source])

    def test_actual_served_identity_conflict_across_runs_is_rejected(self):
        class BackendB(ScriptedActor):
            def propose(self, *args, **kwargs):
                response = super().propose(*args, **kwargs)
                return replace(response, fingerprint="backend-B")
            def propose_target(self, *args, **kwargs):
                response = super().propose_target(*args, **kwargs)
                return replace(response, fingerprint="backend-B")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture, pair = paired(root)
            one = single(root, fixture, actor=BackendB)
            self.assertEqual(build_policy_dataset([one])["served_identity"]["fingerprint"], "backend-B")
            with self.assertRaisesRegex(ValueError, "observed served"):
                build_policy_dataset([pair, one])

    def test_custom_benchmark_name_does_not_hide_fixture_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_manifest([
                {"task_id": "source-a", "group_id": "source-family-a", "role": "history"},
                {"task_id": "target-a", "group_id": "target-family-a", "role": "target_train"},
            ], benchmark="custom-label", revision="fixture-v3")
            _, source = paired(root, manifest=manifest)
            self.assertTrue(build_policy_dataset([source])["fixture_only"])

    def test_preparation_error_keeps_unknown_cost_status_and_raw_trace_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "inputs")
            def fail(trace):
                trace["episode"]["status"] = "error"
            rewrite_trace(fixture, fail)
            collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                                 actor_factory=ScriptedActor, output_dir=root / "paired")
            source = {"kind": "paired", "run": str(root / "paired"), "prepared": str(fixture["target_dir"])}
            row = build_policy_dataset([source])["rows"][0]
            self.assertEqual(row["reason"], "preparation_error")
            self.assertIsNone(row["costs"]["shared"])
            self.assertEqual(row["costs"]["shared_status"], "unavailable_not_validated")
            self.assertEqual(Path(row["provenance"]["preparation_trace_path"]).resolve(),
                             fixture["trace_path"].resolve())
            self.assertTrue(row["provenance"]["raw_file_hash"])
