"""Software-only numerical relationships and audited-artifact integration."""
import contextlib
import copy
import io
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.common.contracts import digest, read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.dataset import build_policy_dataset, write_policy_dataset
from cave_mem_v3.policy.features import feature_contract, feature_vector
from cave_mem_v3.policy.supervised import SupervisedConfig, _fit_dataset, _fit_dataset_legacy, _fit_eq7_head, fit_supervised, load_supervised, predict_supervised, reproduce_supervised, validate_model
from cave_mem_v3.policy.calibration import CalibrationConfig, _calibrate_dataset, calibrate_supervised, predict_calibrated, validate_calibration
from cave_mem_v3.policy.gate import GateConfig, _fit_gate_dataset, fit_initial_gate, predict_initial_gate
from cave_mem_v3.cli import main
from test_policy_dataset import paired
from test_validation_partitions import collect, manifest
from test_target_pair_inputs import make_prepared_fixture


class SupervisedPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        _, source = paired(cls.root / "base")
        cls.base = build_policy_dataset([source])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def synthetic(self, role="target_train", count=40, *, single_class=False, separable=False):
        # Internal numerical unit data; never accepted by the public raw-artifact loader.
        dataset = copy.deepcopy(self.base)
        dataset["rows"] = []
        dataset["available_roles"] = [role]
        for index in range(count):
            row = copy.deepcopy(self.base["rows"][0])
            value = -1 + 2 * index / (count - 1)
            row["features"]["feature_map"]["text_embedding"] = [value, 1.0]
            row["features"]["rho"] = "?"
            row["features"]["e"]["value"] = "?"
            harm = 0 if single_class else int(value < -.5)
            if not single_class and not separable and index in (8,):
                harm = 0
            if not single_class and not separable and index in (12,):
                harm = 1
            row["labels"].update(D=2 * value - .3, H=harm)
            row["costs"]["delta"]["model_calls"] = 1.5 * value + 2
            row["costs"]["delta_valid"]["model_calls"] = True
            row["sample_id"] = role + ":" + str(index)
            row["role"], row["validation_role"] = ("target_train", None) if role == "target_train" else ("validation", role.split("/")[1])
            row["grouping"].update(target_group=role + str(index), source_target_group=role + "-joined",
                target_axes=["task:" + role + str(index)], source_axes=["task:shared-source", "source_trace:shared"])
            dataset["rows"].append(row)
        return seal(dataset)

    def fit(self, dataset=None):
        return _fit_dataset(dataset or self.synthetic(), SupervisedConfig(l2=.001, gamma=2.5, tolerance=1e-6, cost_components=("model_calls", "api_cost")))

    def calibration(self, model, dataset=None):
        return _calibrate_dataset(model, dataset or self.synthetic("validation/calibration"), CalibrationConfig(alpha=.2, minimum_groups=5, risk_bins=1))

    def test_known_linear_benefit_cost_and_logistic_risk_are_learned(self):
        dataset = self.synthetic()
        model = self.fit(dataset)
        self.assertEqual(model, self.fit(dataset))
        for row in dataset["rows"][::7]:
            heads = predict_supervised(model, row["features"])["heads"]
            self.assertAlmostEqual(heads["tau"]["value"], row["labels"]["D"], delta=.01)
            self.assertAlmostEqual(heads["cost:model_calls"]["value"], row["costs"]["delta"]["model_calls"], delta=.01)
        low, high = (predict_supervised(model, dataset["rows"][i]["features"])["heads"]["harm"]["value"] for i in (0, -1))
        self.assertGreater(low, .9)
        self.assertLess(high, .1)
        self.assertEqual(model["capacity"]["family"], "linear_on_frozen_features")

    def test_eq7_exact_unregularized_objectives_gamma_and_shared_denominator(self):
        matrix = [[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]]
        groups = ["a", "b", "c"]
        tau = _fit_eq7_head(matrix, [0.0, 1.0, 2.0], groups, kind="ridge", gamma=3.0, config=SupervisedConfig(gamma=3.0))
        self.assertAlmostEqual(tau["coefficients"][0], 1.0)
        self.assertAlmostEqual(tau["coefficients"][1], 1.0)
        self.assertAlmostEqual(tau["diagnostics"]["objective"], 0.0, places=30)
        harm = _fit_eq7_head([[1.0], [1.0]], [0, 1], ["a", "b"], kind="logistic", gamma=2.0,
            config=SupervisedConfig(gamma=2.0, tolerance=1e-3, max_iterations=20000))
        beta = harm["coefficients"][0]
        expected_bce = (max(beta, 0) - beta * 0 + __import__('math').log1p(__import__('math').exp(-abs(beta)))
            + max(beta, 0) - beta * 1 + __import__('math').log1p(__import__('math').exp(-abs(beta)))) / 2
        self.assertAlmostEqual(harm["diagnostics"]["objective"], 2.0 * expected_bce, places=12)
        dataset = self.synthetic()
        dataset["rows"][0]["label_masks"]["advantage"] = False
        dataset["rows"][1]["label_masks"]["harm"] = False
        model = self.fit(seal(dataset))
        support = model["paired_objective"]
        self.assertEqual(support["valid_tau_rows"], 39)
        self.assertEqual(support["valid_harm_rows"], 39)
        self.assertEqual(support["joint_rows"], 38)
        self.assertEqual(support["tau_denominator"], support["harm_denominator"])
        self.assertEqual(model["heads"]["tau"]["sample_ids"], model["heads"]["harm"]["sample_ids"])
        self.assertEqual(model["schema_version"], "supervised-policy.v2")
        self.assertNotIn("intercept_penalized", model["capacity"])
        self.assertEqual(model["capacity"]["head_objectives"]["tau"]["penalty"], "none")
        self.assertEqual(model["capacity"]["head_objectives"]["harm"]["penalty"], "none")
        self.assertEqual(model["capacity"]["head_objectives"]["cost:model_calls"]["penalty"],
            "l2_over_two_all_coefficients_including_intercept")
        bad_objective = copy.deepcopy(model)
        bad_objective["capacity"]["head_objectives"]["tau"]["penalty"] = "l2"
        with self.assertRaisesRegex(ValueError, "per-head objective"):
            validate_model(seal(bad_objective))
        bad = copy.deepcopy(model)
        bad.pop("objective_protocol")
        bad["record_hash"] = digest({k: v for k, v in bad.items() if k != "record_hash"})
        with self.assertRaises(ValueError):
            validate_model(bad)

    def test_old_v1_objective_is_read_as_legacy_and_reproduced_separately(self):
        legacy = _fit_dataset_legacy(self.synthetic(), SupervisedConfig(l2=.001, tolerance=1e-6,
            cost_components=("model_calls", "api_cost")))
        self.assertEqual(legacy["schema_version"], "supervised-policy.v1")
        self.assertNotIn("objective_protocol", legacy)
        self.assertEqual(legacy["capacity"]["objective"], "group_weighted_mean_loss_plus_l2_over_two")
        self.assertTrue(legacy["capacity"]["intercept_penalized"])
        self.assertEqual(validate_model(legacy), legacy)

    def test_calibration_and_initial_gate_use_bounds_without_refitting_normalization(self):
        dataset = self.synthetic()
        model = self.fit(dataset)
        before = copy.deepcopy(model)
        calibration = self.calibration(model)
        self.assertEqual(model, before)
        gate = _fit_gate_dataset(model, calibration, dataset, GateConfig(risk_threshold=.9))
        self.assertEqual(gate["head"]["status"], "fitted")
        self.assertIn(0, {v["value"] for v in gate["labels"]})
        self.assertIn(1, {v["value"] for v in gate["labels"]})
        negative = predict_initial_gate(gate, model, calibration, dataset["rows"][0]["features"])
        positive = predict_initial_gate(gate, model, calibration, dataset["rows"][-1]["features"])
        self.assertEqual(negative["action"], "Silent")
        self.assertEqual(positive["action"], "Remind")
        self.assertEqual(gate["capacity"], model["capacity"])

    def test_unknown_features_have_distinct_masks_and_known_encoder_dimension(self):
        x = copy.deepcopy(self.base["rows"][0]["features"])
        x["rho"] = "?"
        contract = feature_contract(x)
        missing = feature_vector(x, contract)
        x["rho"] = 0
        zero = feature_vector(x, contract)
        self.assertNotEqual(missing, zero)
        x["feature_map"]["text_embedding"].append(1.0)
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            feature_vector(x, contract)
        x = copy.deepcopy(self.base["rows"][0]["features"])
        x["context"]["future_outcome"] = True
        with self.assertRaisesRegex(ValueError, "outcome"):
            feature_vector(x, contract)

    def test_masks_are_head_specific_and_single_arm_never_trains_harm(self):
        dataset = self.synthetic()
        for row in dataset["rows"][:10]:
            row["labels"]["D"] = None
            row["label_masks"]["advantage"] = False
        for row in dataset["rows"][10:20]:
            row["costs"]["delta"]["api_cost"] = None
            row["costs"]["delta_valid"]["api_cost"] = False
        single = copy.deepcopy(dataset["rows"][0])
        single["kind"] = "single"
        single["labels"] = {"Y": True, "R": 1000000}
        dataset["rows"].append(single)
        model = self.fit(seal(dataset))
        self.assertEqual(model["heads"]["tau"]["rows"], 30)
        self.assertEqual(model["heads"]["harm"]["rows"], 30)
        self.assertEqual(model["heads"]["cost:api_cost"]["rows"], 30)
        self.assertEqual(model["training"]["single_rows_excluded"], 1)

    def test_single_class_or_insufficient_support_never_yields_safe_zero_risk(self):
        dataset = self.synthetic(single_class=True)
        model = self.fit(dataset)
        self.assertEqual(model["heads"]["harm"]["reason"], "joint_paired_harm_requires_both_classes")
        calibration = self.calibration(model)
        prediction = predict_calibrated(model, dataset["rows"][-1]["features"], calibration)
        self.assertEqual(prediction["calibrated_bounds"]["harm"]["upper"], 1)
        self.assertEqual(prediction["calibrated_bounds"]["harm"]["status"], "unavailable")
        gate = _fit_gate_dataset(model, calibration, dataset, GateConfig(risk_threshold=1))
        self.assertEqual(predict_initial_gate(gate, model, calibration, dataset["rows"][-1]["features"])["action"], "Silent")
        model = self.fit()
        small = self.calibration(model, self.synthetic("validation/calibration", count=3))
        self.assertEqual(small["heads"]["tau"]["status"], "unavailable")
        self.assertEqual(small["heads"]["harm"]["bins"][0]["upper"], 1)

    def test_unregularized_logistic_reports_unavailable_for_separable_labels(self):
        model = self.fit(self.synthetic(separable=True))
        self.assertEqual(model["heads"]["harm"]["status"], "unavailable")
        self.assertEqual(model["heads"]["harm"]["reason"], "unregularized_logistic_not_converged")

    def test_zero_calibration_harm_has_strictly_positive_upper_uncertainty(self):
        model = self.fit()
        calibration = self.calibration(model, self.synthetic("validation/calibration", single_class=True))
        item = calibration["heads"]["harm"]["bins"][0]
        self.assertEqual(item["group_any_harm_rate"], 0)
        self.assertGreater(item["upper"], 0)

    def test_eq7_sample_average_changes_when_duplicate_rows_change_empirical_distribution(self):
        dataset = self.synthetic()
        first = self.fit(dataset)
        for index in range(30):
            duplicate = copy.deepcopy(dataset["rows"][0])
            duplicate["sample_id"] = "duplicate:" + str(index)
            dataset["rows"].append(duplicate)
        second = self.fit(seal(dataset))
        self.assertNotEqual(first["heads"]["tau"]["coefficients"], second["heads"]["tau"]["coefficients"])
        self.assertEqual(second["heads"]["tau"]["groups"], 40)

    def test_eq7_harm_minimizer_is_invariant_to_positive_gamma_scale(self):
        matrix = [[1.0]] * 4
        values = [0, 0, 0, 1]
        groups = ["a", "b", "c", "d"]
        ordinary = _fit_eq7_head(matrix, values, groups, kind="logistic", gamma=1.0,
            config=SupervisedConfig(gamma=1.0, tolerance=1e-8))
        small_gamma = _fit_eq7_head(matrix, values, groups, kind="logistic", gamma=1e-8,
            config=SupervisedConfig(gamma=1e-8, tolerance=1e-8))
        self.assertEqual(ordinary["status"], "fitted")
        self.assertEqual(small_gamma["status"], "fitted")
        self.assertAlmostEqual(ordinary["coefficients"][0], small_gamma["coefficients"][0], places=6)
        self.assertAlmostEqual(small_gamma["coefficients"][0], -__import__("math").log(3), places=6)

    def test_roles_and_group_overlap_are_rejected(self):
        model = self.fit()
        for role in ("validation/calibration", "validation/selection"):
            with self.assertRaisesRegex(ValueError, "target_train"):
                self.fit(self.synthetic(role))
        with self.assertRaisesRegex(ValueError, "validation/calibration"):
            self.calibration(model, self.synthetic("validation/selection"))
        dataset = self.synthetic("validation/calibration")
        dataset["rows"][0]["grouping"]["target_axes"] = self.synthetic()["rows"][0]["grouping"]["target_axes"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.calibration(model, dataset)
        source_model = _fit_dataset(self.synthetic(), SupervisedConfig(gamma=2.5, group_axis="source_target_group"))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.calibration(source_model)

    def test_calibration_support_tampering_and_model_dimension_are_rejected(self):
        model = self.fit()
        calibration = self.calibration(model)
        bad = copy.deepcopy(calibration)
        bad["heads"]["harm"]["bins"][0]["upper"] = 0
        with self.assertRaises(ValueError):
            validate_calibration(seal(bad), model)
        bad = copy.deepcopy(calibration)
        bad["heads"]["tau"]["groups"] = 1
        with self.assertRaises(ValueError):
            validate_calibration(seal(bad), model)
        bad = copy.deepcopy(model)
        bad["heads"]["tau"]["coefficients"].append(1)
        with self.assertRaises(ValueError):
            validate_model(seal(bad))

    def test_audited_artifact_api_and_cli_pipeline_stays_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = make_prepared_fixture(root / "history", manifest=manifest())
            train_run = root / "train-pairs"
            from cave_mem_v3.interventions.collection import collect_target_pairs
            collect_target_pairs(fixture["target_dir"], fixture["bank"], env_factory=fixture["env_factory"],
                actor_factory=fixture["actor_factory"], output_dir=train_run)
            train_path = root / "train.json"
            write_policy_dataset([{"kind": "paired", "run": str(train_run), "prepared": str(fixture["target_dir"])}], train_path)
            cal_source = collect(root / "cal", fixture, "calibration", "paired")
            cal_path = root / "cal.json"
            write_policy_dataset([cal_source], cal_path, role="validation", validation_role="calibration")
            for name, config in (("model", asdict(SupervisedConfig(gamma=2.5))), ("cal", asdict(CalibrationConfig())), ("gate", asdict(GateConfig(.1)))):
                write_json(root / (name + "-config.json"), config)
            model_path, calibration_path, gate_path = (root / (name + ".model.json") for name in ("model", "calibration", "gate"))
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("no actor")), \
                    patch("cave_mem_v3.cli._env_factory", side_effect=AssertionError("no environment")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["train-policy", "--dataset", str(train_path), "--config", str(root / "model-config.json"), "--output", str(model_path)]), 0)
                self.assertEqual(main(["calibrate-policy", "--model", str(model_path), "--dataset", str(cal_path), "--config", str(root / "cal-config.json"), "--output", str(calibration_path)]), 0)
                self.assertEqual(main(["train-initial-gate", "--model", str(model_path), "--calibration", str(calibration_path), "--dataset", str(train_path), "--config", str(root / "gate-config.json"), "--output", str(gate_path)]), 0)
                x_path = root / "x.json"
                write_json(x_path, read_json(train_path)["rows"][0]["features"])
                prediction_path = root / "prediction.json"
                self.assertEqual(main(["predict-policy", "--model", str(model_path), "--calibration", str(calibration_path), "--gate", str(gate_path), "--features", str(x_path), "--output", str(prediction_path)]), 0)
            model = load_supervised(model_path)
            self.assertEqual(model, fit_supervised(train_path, config=SupervisedConfig(gamma=2.5)))
            self.assertEqual(read_json(prediction_path)["action"], "Silent")
            with self.assertRaises(ValueError):
                fit_supervised(cal_path)
            with self.assertRaisesRegex(ValueError, "explicit SupervisedConfig"):
                fit_supervised(train_path, output=root / "unspecified.model.json")
            with self.assertRaises(ValueError):
                fit_supervised(train_path, config=SupervisedConfig(gamma=2.5), output=model_path)
            with self.assertRaises(ValueError):
                calibrate_supervised(model_path, cal_path, output=train_run / "bad.json")

    def test_optional_numpy_missing_reports_install_instruction(self):
        import builtins
        from cave_mem_v3.policy.supervised import _numpy
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("blocked optional dependency")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=blocked):
            with self.assertRaisesRegex(ValueError, "policy extra"):
                _numpy()

    def test_resealed_risk_bounds_cannot_invent_groups_or_types(self):
        model = self.fit()
        calibration = self.calibration(model)
        bad = copy.deepcopy(calibration)
        bad["heads"]["harm"]["bins"][0]["group_values"]["unregistered"] = 0
        with self.assertRaises(ValueError):
            validate_calibration(seal(bad), model)
        bad = copy.deepcopy(model)
        bad["normalization"]["scale"][0] = True
        with self.assertRaises(ValueError):
            validate_model(seal(bad))
        bad = copy.deepcopy(model)
        bad["feature_contract"]["columns"][0] = "label.D"
        with self.assertRaises(ValueError):
            validate_model(seal(bad))

    def test_gate_labels_ignore_raw_pair_labels_after_bounds_are_frozen(self):
        dataset = self.synthetic()
        model = self.fit(dataset)
        calibration = self.calibration(model)
        first = _fit_gate_dataset(model, calibration, dataset, GateConfig(.9))
        altered = copy.deepcopy(dataset)
        for row in altered["rows"]:
            row["labels"].update(D=-999, H=1)
        # Internal unit probe keeps the audited identity fixed; public API rebuilds it.
        second = _fit_gate_dataset(model, calibration, altered, GateConfig(.9))
        self.assertEqual(first, second)
