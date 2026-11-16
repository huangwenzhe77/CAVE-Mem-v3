"""Post-pair teacher labels stay blinded and separate from single-arm rows."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cave_mem_v3.common.contracts import StudySpec
from cave_mem_v3.interventions.auto import collect_target_auto
from cave_mem_v3.interventions.exposure import judge_paired_exposure, summarize_paired_exposure
from cave_mem_v3.policy.dataset import build_policy_dataset
from cave_mem_v3.target_data.sampling import RiskHazard
from test_single_execution import NoncloneableFixture
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture


class Teacher:
    def __init__(self, response='{"e": 1, "confidence": "high", "reason": "Observed failed action."}'):
        self.response = response
        self.calls = []

    def configuration(self):
        return {"kind": "fixture-teacher", "model": "fixture"}

    def generate(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return SimpleNamespace(text=self.response, model="fixture", finish_reason="stop",
                               usage={"model_calls": 1}, attempt={"status": "completed"})


def collect(root, *, noncloneable=False):
    source = make_prepared_fixture(root / "source")
    def factory(task):
        return NoncloneableFixture(task_id=task["task_id"]) if noncloneable else source["env_factory"](task)
    collect_target_auto(source["manifest"], bank=source["bank"], role="target_train",
        spec=StudySpec(max_steps=4, max_model_calls=4), env_factory=factory,
        actor_factory=source["actor_factory"], output_dir=root / "auto", encoder=EncoderFixture(),
        assignment_probability=.5,
        hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99))


class PosthocExposureTests(unittest.TestCase):
    def test_paired_label_and_final_x_enter_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect(root)
            teacher = Teacher()
            summary = judge_paired_exposure(root / "auto", provider=teacher, output_dir=root / "labels")
            self.assertEqual(summary["assessed"], 1)
            self.assertEqual(len(teacher.calls), 1)
            evidence = json.loads(teacher.calls[0][0][1]["content"])
            self.assertIn("target_original", evidence)
            self.assertIn("mechanism", evidence)
            self.assertNotIn("remind", json.dumps(evidence).lower())
            self.assertNotIn("success", evidence["target_original"])
            self.assertEqual(summarize_paired_exposure(root / "labels"), summary)
            dataset = build_policy_dataset([{"kind": "auto", "run": str(root / "auto"),
                                             "exposure": str(root / "labels")}])
            row = dataset["rows"][0]
            self.assertEqual(row["e_observed"]["value"], 1)
            self.assertEqual(row["x_final"]["e"], row["e_observed"])
            self.assertEqual(row["x_final"]["schema_version"], "post-pair-state.v1")
            self.assertEqual(judge_paired_exposure(root / "auto", provider=teacher,
                                                   output_dir=root / "labels"), summary)
            self.assertEqual(len(teacher.calls), 1)

    def test_invalid_teacher_response_is_preserved_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect(root)
            teacher = Teacher("not json")
            result = judge_paired_exposure(root / "auto", provider=teacher, output_dir=root / "labels")
            self.assertEqual(result["invalid_or_unavailable"], 1)
            self.assertEqual(result["labels"][0]["status"], "invalid_response")
            self.assertEqual(judge_paired_exposure(root / "auto", provider=teacher,
                                                   output_dir=root / "labels"), result)
            self.assertEqual(len(teacher.calls), 1)

    def test_single_arm_has_no_posthoc_e(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collect(root, noncloneable=True)
            teacher = Teacher()
            result = judge_paired_exposure(root / "auto", provider=teacher, output_dir=root / "labels")
            self.assertEqual(result["labels"], [])
            self.assertEqual(teacher.calls, [])
            row = build_policy_dataset([{"kind": "auto", "run": str(root / "auto"),
                                         "exposure": str(root / "labels")}])["rows"][0]
            self.assertEqual(row["kind"], "single")
            self.assertNotIn("e_observed", row)
            self.assertNotIn("x_final", row)


if __name__ == "__main__":
    unittest.main()
