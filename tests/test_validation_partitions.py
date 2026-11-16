"""Offline partition contracts; fixture outcomes do not establish research effects."""
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import replace

from cave_mem_v3.benchmarks.tasks import build_manifest
from cave_mem_v3.common.contracts import StudySpec, read_json, write_json
from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.cli import main
from cave_mem_v3.interventions.collection import collect_target_pairs
from cave_mem_v3.interventions.inputs import load_prepared_run, seal
from cave_mem_v3.interventions.outcomes import summarize_target_pairs
from cave_mem_v3.interventions.single import collect_single_target
from cave_mem_v3.interventions.single_outcomes import summarize_target_single
from cave_mem_v3.policy.dataset import build_policy_dataset, write_policy_dataset, load_policy_dataset
from cave_mem_v3.target_data.collection import collect_target
from cave_mem_v3.target_data.sampling import RiskHazard
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture


def manifest():
    return build_manifest([
        {"task_id": "source-a", "group_id": "source-family-a", "role": "history"},
        {"task_id": "target-a", "group_id": "target-family-a", "role": "target_train"},
        {"task_id": "calibration-a", "group_id": "calibration-family", "template_id": "calibration-template",
         "role": "validation", "validation_role": "calibration"},
        {"task_id": "selection-a", "group_id": "selection-family", "template_id": "selection-template",
         "role": "validation", "validation_role": "selection"},
        {"task_id": "test-a", "group_id": "test-family", "role": "test"},
    ], benchmark="preparation-fixture", revision="fixture-v3")


def collect(root, fixture, subrole, route, *, actor=ScriptedActor, hazard=.99, spec=None):
    options = dict(role="validation", validation_role=subrole,
        spec=spec or StudySpec(max_steps=4, max_model_calls=4), env_factory=fixture["env_factory"],
        actor_factory=actor, encoder=EncoderFixture(), bank=fixture["bank"],
        hazard=RiskHazard(low_risk_probability=hazard, high_risk_probability=hazard))
    if route == "paired":
        prepared = root / "prepared"
        collect_target(fixture["manifest"], **options, output_dir=prepared, prepare_before_action=True)
        collect_target_pairs(prepared, fixture["bank"], env_factory=fixture["env_factory"],
            actor_factory=actor, output_dir=root / "paired")
        return {"kind": "paired", "run": str(root / "paired"), "prepared": str(prepared)}
    collect_single_target(fixture["manifest"], **options, output_dir=root / "single", assignment_probability=.5)
    return {"kind": "single", "run": str(root / "single")}


class ValidationPartitionTests(unittest.TestCase):
    def test_both_routes_and_subroles_roundtrip_and_resume_without_model_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            for subrole in ("calibration", "selection"):
                sources = [collect(root / subrole / route, fixture, subrole, route) for route in ("paired", "single")]
                output = root / (subrole + ".json")
                data = write_policy_dataset(sources, output, role="validation", validation_role=subrole)
                self.assertEqual(load_policy_dataset(output, role="validation", validation_role=subrole), data)
                self.assertEqual(data["available_roles"], ["validation/" + subrole])
                self.assertEqual(data["summary"]["planned"], 2)
                for row in data["rows"]:
                    self.assertEqual((row["role"], row["validation_role"]), ("validation", subrole))
                    self.assertEqual(row["provenance"]["task_id"], subrole + "-a")
                    self.assertTrue(row["label_masks"]["features"])
                    self.assertTrue(row["label_masks"]["observed_outcome"])
                    self.assertIn("template:" + subrole + "-template", row["grouping"]["target_axes"])
                with self.assertRaises(ValueError):
                    load_policy_dataset(output)
                pair, single = sources
                never = Mock(side_effect=AssertionError("no model/environment construction"))
                collect_target_pairs(pair["prepared"], fixture["bank"], env_factory=never, actor_factory=never,
                    output_dir=pair["run"])
                collect_single_target(fixture["manifest"], bank=fixture["bank"], role="validation", validation_role=subrole,
                    spec=StudySpec(max_steps=4, max_model_calls=4), env_factory=never, actor_factory=never,
                    output_dir=single["run"], encoder=EncoderFixture(), assignment_probability=.5,
                    hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99))
                never.assert_not_called()

    def test_training_export_refuses_validation_before_outcome_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            for route in ("paired", "single"):
                source = collect(root / route, fixture, "calibration", route)
                with patch("cave_mem_v3.policy.dataset._paired", side_effect=AssertionError("outcomes")), \
                     patch("cave_mem_v3.policy.dataset._single", side_effect=AssertionError("outcomes")):
                    with self.assertRaisesRegex(ValueError, "partition mismatch"):
                        build_policy_dataset([source])
                    with self.assertRaisesRegex(ValueError, "partition mismatch"):
                        build_policy_dataset([source], role="validation", validation_role="selection")

    def test_bad_role_and_subrole_rejected_before_factories(self):
        never = Mock(side_effect=AssertionError("factory called"))
        with tempfile.TemporaryDirectory() as tmp:
            for role, subrole in (("test", None), ("validation", None), ("target_train", "calibration"), ("validation", "wrong")):
                options = dict(role=role, validation_role=subrole, spec=StudySpec(), env_factory=never,
                    actor_factory=never, output_dir=tmp)
                with self.assertRaises(ValueError):
                    collect_target({}, **options)
                with self.assertRaises(ValueError):
                    collect_single_target({}, **options, bank={}, encoder=None, assignment_probability=.5)
                with self.assertRaises(ValueError):
                    build_policy_dataset([], role=role, validation_role=subrole)
        never.assert_not_called()

    def test_validation_reentry_cannot_switch_partition_before_factories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            pair = collect(root / "pair", fixture, "calibration", "paired")
            one = collect(root / "one", fixture, "calibration", "single")
            never = Mock(side_effect=AssertionError("factory called"))
            options = dict(role="validation", validation_role="selection", spec=StudySpec(),
                env_factory=never, actor_factory=never)
            with self.assertRaisesRegex(ValueError, "partition changed"):
                collect_target(fixture["manifest"], **options, output_dir=pair["prepared"])
            with self.assertRaisesRegex(ValueError, "partition changed"):
                collect_single_target(fixture["manifest"], **options, output_dir=one["run"],
                    bank=fixture["bank"], encoder=EncoderFixture(), assignment_probability=.5)
            never.assert_not_called()

    def test_no_selection_and_no_memory_keep_validation_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for empty in (False, True):
                fixture = make_prepared_fixture(root / str(empty) / "history", manifest=manifest(), empty_bank=empty)
                for subrole in ("calibration", "selection"):
                    sources = [collect(root / str(empty) / subrole / route, fixture, subrole, route,
                        hazard=.99 if empty else 1e-9) for route in ("paired", "single")]
                    data = build_policy_dataset(sources, role="validation", validation_role=subrole)
                    self.assertEqual(data["summary"]["planned"], 2)
                    self.assertEqual(data["summary"]["by_reason"], {"no_memory" if empty else "no_selection": 2})
                    self.assertEqual(data["summary"]["label_counts"]["advantage"], 0)

    def test_unknown_cost_retains_valid_outcome_without_reward(self):
        class UnknownFee(ScriptedActor):
            def propose(self, context, **kwargs):
                response = super().propose(context, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
            def propose_target(self, context, **kwargs):
                response = super().propose_target(context, **kwargs)
                return replace(response, usage={**response.usage, "api_cost": None})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            for subrole in ("calibration", "selection"):
                sources = [collect(root / subrole / route, fixture, subrole, route, actor=UnknownFee,
                    spec=StudySpec(max_steps=4, max_model_calls=4, inference_weights={"api_cost": 1.0})) for route in ("paired", "single")]
                data = build_policy_dataset(sources, role="validation", validation_role=subrole)
                for row in data["rows"]:
                    self.assertTrue(row["label_masks"]["observed_outcome"])
                    self.assertFalse(row["label_masks"]["advantage"])
                    self.assertFalse(row["label_masks"]["observed_reward"])

    def test_readers_reject_forged_test_role_before_raw_outcome_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            pair = collect(root / "paired", fixture, "calibration", "paired")
            one = collect(root / "single", fixture, "calibration", "single")
            path = Path(pair["run"]) / "run.json"
            run = read_json(path)
            run["prepared_run"]["role"] = "test"
            run["prepared_run"] = seal(run["prepared_run"], "run_hash")
            run["source_run_hash"] = run["prepared_run"]["run_hash"]
            write_json(path, seal(run))
            with patch("cave_mem_v3.interventions.outcomes._pair", side_effect=AssertionError("outcomes")):
                with self.assertRaisesRegex(ValueError, "test outcomes are forbidden"):
                    summarize_target_pairs(pair["run"])
            path = Path(one["run"]) / "run.json"
            run = read_json(path)
            run["role"] = "test"
            write_json(path, seal(run))
            with patch("cave_mem_v3.interventions.single_outcomes._session", side_effect=AssertionError("outcomes")):
                with self.assertRaisesRegex(ValueError, "test outcomes are forbidden"):
                    summarize_target_single(one["run"])

    def test_manifest_rejects_cross_partition_template_and_dependency_groups(self):
        for field, value in (("template_id", "shared-template"), ("groups", ["shared-dependency"])):
            m = manifest()
            for task in m["tasks"]:
                if task["role"] in ("target_train", "validation"):
                    task[field] = value
            with self.assertRaises(ValueError):
                build_manifest(m["tasks"], benchmark=m["benchmark"], revision=m["revision"])

    def test_cli_requires_subrole_before_loading_encoder_or_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = ["--config", "missing.toml", "--manifest", "missing.json", "--output", tmp,
                "--bank", "missing-bank.json", "--encoder-path", "missing-model", "--encoder-revision", "test",
                "--role", "validation"]
            with patch("cave_mem_v3.cli.load_config", side_effect=AssertionError("config loaded")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["prepare-target", *args]), 2)
                self.assertEqual(main(["collect-single-target", *args, "--assignment-probability", ".5"]), 2)

    def test_validation_bank_mismatch_and_legacy_evidence_fail_before_factories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            current = manifest()
            other = make_prepared_fixture(root / "other")
            legacy = make_prepared_fixture(root / "legacy", manifest=current, legacy=True)
            never = Mock(side_effect=AssertionError("factory called"))
            for bank in (other["bank"], legacy["bank"]):
                with self.assertRaises(ValueError):
                    collect_target(current, role="validation", validation_role="calibration",
                        spec=StudySpec(), env_factory=never, actor_factory=never,
                        output_dir=root / "invalid", prepare_before_action=True,
                        bank=bank, encoder=EncoderFixture())
            never.assert_not_called()
            self.assertFalse((root / "invalid").exists())

    def test_legacy_ambiguous_validation_and_trace_subrole_mismatch_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            source = collect(root / "calibration", fixture, "calibration", "paired")
            trace_path = next((Path(source["prepared"]) / "traces").glob("*.json"))
            trace = read_json(trace_path)
            trace["validation_role"] = "selection"
            write_json(trace_path, seal(trace, "trace_hash"))
            _, inputs = load_prepared_run(source["prepared"], fixture["bank"])
            self.assertEqual(inputs[0]["reason"], "invalid_prepared_input")
            path = Path(source["prepared"]) / "run.json"
            run = read_json(path)
            run.pop("validation_role")
            write_json(path, seal(run, "run_hash"))
            with self.assertRaisesRegex(ValueError, "explicit validation_role"):
                load_prepared_run(source["prepared"], fixture["bank"])

    def test_cli_validation_paths_and_dataset_selection(self):
        class LocalEncoder(EncoderFixture):
            def __init__(self, *args, **kwargs):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            write_json(root / "manifest.json", fixture["manifest"])
            write_json(root / "bank.json", fixture["bank"])
            (root / "encoder").mkdir()
            config = Path(__file__).resolve().parents[1] / "configs/phase1/demo.toml"
            common = ["--config", str(config), "--manifest", str(root / "manifest.json"),
                "--bank", str(root / "bank.json"), "--encoder-path", str(root / "encoder"),
                "--encoder-revision", "fixture", "--role", "validation",
                "--low-risk-probability", ".99", "--high-risk-probability", ".99"]
            with patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", LocalEncoder), \
                    contextlib.redirect_stdout(io.StringIO()):
                for subrole in ("calibration", "selection"):
                    prepared, pair, one = (root / (subrole + suffix) for suffix in ("-prepared", "-pair", "-single"))
                    self.assertEqual(main(["prepare-target", *common, "--validation-role", subrole,
                        "--output", str(prepared)]), 0)
                    self.assertEqual(main(["pair-target", "--config", str(config), "--target-run", str(prepared),
                        "--bank", str(root / "bank.json"), "--output", str(pair)]), 0)
                    self.assertEqual(main(["collect-single-target", *common, "--validation-role", subrole,
                        "--assignment-probability", ".5", "--output", str(one)]), 0)
                    output = root / (subrole + "-dataset.json")
                    self.assertEqual(main(["build-policy-dataset", "--role", "validation", "--validation-role", subrole,
                        "--paired", str(pair), str(prepared), "--single", str(one), "--output", str(output)]), 0)
                    self.assertEqual(load_policy_dataset(output, role="validation", validation_role=subrole)["summary"]["planned"], 2)

    def test_cli_missing_validation_subset_fails_before_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history")
            write_json(root / "manifest.json", fixture["manifest"])
            args = ["--config", str(Path(__file__).resolve().parents[1] / "configs/phase1/demo.toml"),
                "--manifest", str(root / "manifest.json"), "--output", str(root / "new"),
                "--bank", "missing.json", "--encoder-path", "missing-model", "--encoder-revision", "fixture",
                "--role", "validation", "--validation-role", "calibration"]
            with patch("cave_mem_v3.target_data.retrieval.SentenceTransformerEncoder", side_effect=AssertionError("encoder")), \
                 patch("cave_mem_v3.cli._actor", side_effect=AssertionError("actor")), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["prepare-target", *args]), 2)
                self.assertEqual(main(["collect-single-target", *args, "--assignment-probability", ".5"]), 2)
