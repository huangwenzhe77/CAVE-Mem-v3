"""Offline randomized numeric tests plus audited single-run integration."""
import contextlib
import copy
import io
import math
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from cave_mem_v3.common.contracts import read_json, write_json
from cave_mem_v3.interventions.inputs import seal
from cave_mem_v3.policy.dataset import build_policy_dataset, write_policy_dataset
from cave_mem_v3.policy.dr import DRConfig, dr_scores, _estimate_dataset, estimate_single_dr, load_single_dr
from cave_mem_v3.cli import main
from test_policy_dataset import single
from test_target_pair_inputs import make_prepared_fixture


class SingleDRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        fixture = make_prepared_fixture(cls.root / "inputs")
        cls.source = single(cls.root, fixture)
        cls.base = build_policy_dataset([cls.source])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def synthetic(self, count=120):
        dataset = copy.deepcopy(self.base)
        dataset["rows"] = []
        rng = random.Random(719)
        for i in range(count):
            row = copy.deepcopy(self.base["rows"][0])
            x = -1 + 2 * i / (count - 1)
            z = int(rng.random() < .5)
            row["features"]["feature_map"]["text_embedding"] = [x, 1.0]
            row["labels"] = {"Y": True, "R": .2 + .3 * x + z * (.5 + .8 * x)}
            row["label_masks"]["observed_reward"] = True
            row["assignment"] = {"kind": "randomized_single", "z": z, "arm": "remind" if z else "silent",
                "propensity": .5, "probabilities": {"silent": .5, "remind": .5}}
            row["sample_id"] = "synthetic:" + str(i)
            row["grouping"].update(target_group="g" + str(i), source_target_group="joined", target_axes=["task:t" + str(i)], source_axes=["task:history-shared"])
            dataset["rows"].append(row)
        return seal(dataset)

    def test_eq8_matches_hand_calculation_for_both_arms_and_silent_reference(self):
        p = {"silent": .8, "remind": .2}
        mu = {"silent": .1, "remind": .4}
        value = dr_scores(.9, 1, p, mu, .3)
        self.assertAlmostEqual(value["estimated_arm_utilities"]["remind"], 2.9)
        self.assertAlmostEqual(value["policy_value"], .94)
        self.assertAlmostEqual(value["improvement_over_silent"], .84)
        value = dr_scores(.2, 0, p, mu, .3)
        self.assertAlmostEqual(value["silent_value"], .225)
        self.assertAlmostEqual(value["policy_value"], .2775)
        self.assertAlmostEqual(value["improvement_over_silent"], .0525)
        self.assertEqual(dr_scores(.9, 1, p, mu, 0)["improvement_over_silent"], 0)

    def test_known_randomized_relation_recovers_value_and_advantage(self):
        dataset = self.synthetic()
        config = DRConfig(candidate_remind_probability=.3, folds=3, l2=1e-5)
        result = _estimate_dataset(dataset, config)
        self.assertEqual(result, _estimate_dataset(dataset, config))
        means = result["summary"]["equal_group_mean"]
        self.assertAlmostEqual(means["estimated_dr_advantage"], .5, delta=.001)
        self.assertAlmostEqual(means["improvement_over_silent"], .15, delta=.001)
        self.assertAlmostEqual(means["policy_value"], .35, delta=.001)
        self.assertEqual(result["summary"]["valid_rows"], 120)
        self.assertTrue(any(r["raw"]["estimated_dr_advantage"] < 0 for r in result["rows"]))
        self.assertNotIn("H", result["rows"][0])
        self.assertNotIn("D", result["rows"][0])

    def test_each_fold_excludes_heldout_groups_from_scaling_and_both_arm_fits(self):
        result = _estimate_dataset(self.synthetic(), DRConfig(.5, folds=3))
        by_id = {r["sample_id"]: r for r in result["rows"]}
        for fold in result["folds"]:
            held = set(fold["held_out_groups"])
            self.assertFalse(held.intersection(fold["normalization"]["group_ids"]))
            for head in fold["heads"].values():
                self.assertFalse(held.intersection(head["group_ids"]))
                for sid in head["sample_ids"]:
                    self.assertNotEqual(by_id[sid]["fold"], fold["fold"])
            raw = [by_id[sid]["features"]["feature_map"]["text_embedding"][0] for sid in fold["normalization"]["sample_ids"]]
            self.assertAlmostEqual(fold["normalization"]["mean"][0], sum(raw) / len(raw), places=12)

    def test_fold_assignment_is_outcome_blind_and_respects_dependencies(self):
        dataset = self.synthetic()
        first = _estimate_dataset(dataset, DRConfig(.5))
        for row in dataset["rows"]:
            row["labels"]["R"] = -999
        second = _estimate_dataset(seal(dataset), DRConfig(.5))
        self.assertEqual(first["fold_assignment"], second["fold_assignment"])
        dataset["rows"][1]["grouping"]["target_axes"] = dataset["rows"][0]["grouping"]["target_axes"]
        with self.assertRaisesRegex(ValueError, "span different"):
            _estimate_dataset(dataset, DRConfig(.5))

    def test_unknown_reward_no_features_and_paired_rows_preserve_denominators(self):
        dataset = self.synthetic()
        dataset["rows"][0]["labels"]["R"] = None
        dataset["rows"][0]["label_masks"]["observed_reward"] = False
        dataset["rows"][1]["features"] = None
        dataset["rows"][1]["label_masks"]["features"] = False
        paired = copy.deepcopy(dataset["rows"][2])
        paired["kind"] = "paired"
        paired["sample_id"] = "paired:excluded"
        dataset["rows"].append(paired)
        result = _estimate_dataset(seal(dataset), DRConfig(.5))
        self.assertEqual(result["summary"]["planned_single_rows"], 120)
        self.assertEqual(result["summary"]["valid_rows"], 118)
        self.assertEqual(result["summary"]["excluded_paired_rows"], 1)
        self.assertEqual(result["rows"][0]["reason"], "observed_reward_unavailable")
        self.assertIsNone(result["rows"][0]["raw"])
        self.assertFalse(result["summary"]["complete_planned_population"])

    def test_missing_arm_and_one_dependency_group_remain_unavailable(self):
        dataset = self.synthetic()
        for row in dataset["rows"]:
            row["assignment"].update(z=0, arm="silent")
        result = _estimate_dataset(seal(dataset), DRConfig(.5))
        self.assertEqual(result["summary"]["valid_rows"], 0)
        self.assertEqual(result["summary"]["by_reason"], {"fold_arm_support_insufficient": 120})
        self.assertIsNone(result["summary"]["equal_group_mean"]["policy_value"])
        result = _estimate_dataset(self.synthetic(), DRConfig(.5, group_axis="source_target_group"))
        self.assertEqual(result["summary"]["by_reason"], {"insufficient_independent_fold_groups": 120})

    def test_logging_propensity_not_checkpoint_inclusion_is_used(self):
        dataset = self.synthetic()
        original = _estimate_dataset(dataset, DRConfig(.5))
        for row in dataset["rows"]:
            row["inclusion_probability"] = 1e-9
        altered = _estimate_dataset(seal(dataset), DRConfig(.5))
        self.assertEqual(original["summary"], altered["summary"])
        dataset["rows"][0]["assignment"]["propensity"] = 1e-9
        with self.assertRaisesRegex(ValueError, "propensity"):
            _estimate_dataset(dataset, DRConfig(.5))

    def test_extreme_weights_clipping_and_ess_are_explicit(self):
        value = dr_scores(-1, 1, {"silent": .999999, "remind": .000001}, {"silent": 0, "remind": 0}, .5)
        clipped = dr_scores(-1, 1, {"silent": .999999, "remind": .000001}, {"silent": 0, "remind": 0}, .5, inverse_weight_cap=10)
        self.assertEqual(value["inverse_logging_weight"], 1e6)
        self.assertEqual(clipped["used_inverse_weight"], 10)
        self.assertEqual(clipped["policy_value"], -5)
        dataset = self.synthetic()
        row = dataset["rows"][0]
        row["assignment"] = {"kind": "randomized_single", "z": 1, "arm": "remind", "propensity": 1e-6,
            "probabilities": {"silent": 1 - 1e-6, "remind": 1e-6}}
        row["labels"]["R"] = -10
        result = _estimate_dataset(seal(dataset), DRConfig(.5, inverse_weight_caps=(10,)))
        self.assertLess(result["summary"]["weights"]["policy_correction_weight"]["ess"], 2)
        self.assertEqual(result["weight_sensitivity"]["10.0"]["clipped_rows"], 1)
        self.assertGreater(result["weight_sensitivity"]["10.0"]["absolute_shift_from_raw"]["policy_value"], 1)

    def test_roles_and_invalid_probabilities_are_rejected(self):
        dataset = self.synthetic()
        dataset["available_roles"] = ["validation/calibration"]
        with self.assertRaisesRegex(ValueError, "target_train"):
            _estimate_dataset(dataset, DRConfig(.5))
        for p in (-1, 1.1, True, math.nan):
            with self.assertRaises(ValueError):
                DRConfig(p)
        with self.assertRaises(ValueError):
            dr_scores(1, 1, {"silent": 1, "remind": 0}, {"silent": 0, "remind": 0}, .5)
        with self.assertRaises(ValueError):
            dr_scores(1, 1, {"silent": 1 - 1e-320, "remind": 1e-320}, {"silent": 0, "remind": 0}, .5)

    def test_public_api_cli_and_reload_reaudit_original_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "dataset.json"
            write_policy_dataset([self.source], dataset_path)
            config = DRConfig(.5)
            write_json(root / "config.json", asdict(config))
            output = root / "dr.json"
            with patch("cave_mem_v3.cli._actor", side_effect=AssertionError("no actor")), \
                    patch("cave_mem_v3.cli._env_factory", side_effect=AssertionError("no environment")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["estimate-single-dr", "--dataset", str(dataset_path), "--config", str(root / "config.json"), "--output", str(output)]), 0)
            result = load_single_dr(output)
            self.assertEqual(result, estimate_single_dr(dataset_path, config=config))
            self.assertEqual(result["summary"]["planned_single_rows"], 1)
            self.assertEqual(result["summary"]["valid_rows"], 0)
            with self.assertRaises(ValueError):
                estimate_single_dr(dataset_path, config=config, output=output)
            with self.assertRaises(ValueError):
                estimate_single_dr(dataset_path, config=config, output=Path(self.source["run"]) / "bad.json")
            result["fold_assignment"] = {key: 99 for key in result["fold_assignment"]}
            write_json(output, seal(result))
            with self.assertRaisesRegex(ValueError, "original audited"):
                load_single_dr(output)

    def test_weight_diagnostics_and_means_stay_stable_at_float_limit(self):
        from cave_mem_v3.policy.dr import _weight_summary, _mean
        stats = _weight_summary([1e308, 1e308])
        self.assertIsNone(stats["sum"])
        self.assertEqual(stats["sum_status"], "overflow_unrepresentable")
        self.assertEqual(stats["ess"], 2)
        self.assertEqual(_mean([1e308, 1e308]), 1e308)
        self.assertEqual(_mean([1e308, 1e308, -1e308, -1e308]), 0)

    def test_heldout_feature_changes_do_not_change_its_nuisance_fit(self):
        dataset = self.synthetic()
        config = DRConfig(.5)
        first = _estimate_dataset(dataset, config)
        held_out_fold = first["rows"][0]["fold"]
        dataset["rows"][0]["features"]["feature_map"]["text_embedding"][0] = 1000
        second = _estimate_dataset(seal(dataset), config)
        self.assertEqual(first["folds"][held_out_fold], second["folds"][held_out_fold])
