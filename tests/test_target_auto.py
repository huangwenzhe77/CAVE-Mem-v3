"""One live checkpoint routes to a certified pair or an observed single arm."""
import tempfile
import unittest
from pathlib import Path

from cave_mem_v3.agents.providers import ScriptedActor
from cave_mem_v3.common.contracts import StudySpec, read_json
from cave_mem_v3.environments.base import PreparationEnvironment
from cave_mem_v3.interventions.auto import collect_target_auto, summarize_target_auto
from cave_mem_v3.policy.dataset import build_policy_dataset
from cave_mem_v3.target_data.sampling import RiskHazard
from test_single_execution import NoncloneableFixture
from test_target_pair_inputs import make_prepared_fixture
from test_target_prepare import EncoderFixture


class FailedRestoreFixture(PreparationEnvironment):
    def restore(self, checkpoint):
        raise ValueError("independent restore failed")


class AutoTargetTests(unittest.TestCase):
    def collect(self, root, env_factory=None):
        source = make_prepared_fixture(root / "source")
        return source, collect_target_auto(source["manifest"], bank=source["bank"],
            role="target_train", spec=StudySpec(max_steps=4, max_model_calls=4),
            env_factory=env_factory or source["env_factory"], actor_factory=source["actor_factory"],
            output_dir=root / "auto", encoder=EncoderFixture(), assignment_probability=.5,
            hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99))

    def test_certified_restore_produces_pair_and_training_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, summary = self.collect(root)
            self.assertEqual(summary["counts"], {"paired": 1})
            self.assertEqual(summary["observations"][0]["observation"]["status"], "completed")
            self.assertEqual(summarize_target_auto(root / "auto"), summary)
            dataset = build_policy_dataset([{"kind": "auto", "run": str(root / "auto")}])
            self.assertEqual(dataset["summary"]["label_counts"]["paired_outcomes"], 1)
            self.assertEqual(dataset["rows"][0]["kind"], "paired")
            self.assertEqual(self.collect_again(source, root), summary)

    @staticmethod
    def collect_again(source, root):
        return collect_target_auto(source["manifest"], bank=source["bank"], role="target_train",
            spec=StudySpec(max_steps=4, max_model_calls=4), env_factory=source["env_factory"],
            actor_factory=source["actor_factory"], output_dir=root / "auto",
            encoder=EncoderFixture(), assignment_probability=.5,
            hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99))

    def test_noncloneable_stays_in_one_live_session_and_has_single_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environments = []
            def factory(task):
                env = NoncloneableFixture(task_id=task["task_id"])
                environments.append(env)
                return env
            _, summary = self.collect(root, factory)
            self.assertEqual(summary["counts"], {"single": 1})
            self.assertEqual([env.resets for env in environments], [0, 1])
            self.assertEqual(summary["observations"][0]["observation"]["status"], "completed")
            dataset = build_policy_dataset([{"kind": "auto", "run": str(root / "auto")}])
            self.assertEqual(dataset["summary"]["label_counts"]["observed_reward"], 1)
            self.assertEqual(dataset["summary"]["label_counts"]["paired_outcomes"], 0)

    def test_failed_restore_audit_falls_back_before_treatment_in_same_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environments = []
            def factory(task):
                env = FailedRestoreFixture(task_id=task["task_id"])
                environments.append(env)
                return env
            _, summary = self.collect(root, factory)
            self.assertEqual(summary["counts"], {"single": 1})
            self.assertEqual(summary["observations"][0]["observation"]["status"], "completed")
            session = next((root / "auto" / "sessions").iterdir())
            route = read_json(session / "route.json")
            self.assertEqual(route["reason"], "restoration_audit_failed")
            self.assertEqual(environments[1].observe().task_id, "target-a")
            self.assertEqual(summarize_target_auto(root / "auto"), summary)

    def test_pair_arm_failure_does_not_become_single_arm(self):
        class FailingReminder(ScriptedActor):
            def propose_target(self, *args, **kwargs):
                if kwargs.get("reminder") is not None:
                    raise ValueError("reminder service failed")
                return super().propose_target(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_prepared_fixture(root / "source")
            summary = collect_target_auto(source["manifest"], bank=source["bank"],
                role="target_train", spec=StudySpec(max_steps=4, max_model_calls=4),
                env_factory=source["env_factory"], actor_factory=FailingReminder,
                output_dir=root / "auto", encoder=EncoderFixture(), assignment_probability=.5,
                hazard=RiskHazard(low_risk_probability=.99, high_risk_probability=.99))
            self.assertEqual(summary["counts"], {"paired": 1})
            self.assertFalse(summary["observations"][0]["observation"]["outcome_valid"])
            session = next((root / "auto" / "sessions").iterdir())
            self.assertFalse((session / "assignment.json").exists())

    def test_no_memory_and_no_selection_keep_excluded_denominator(self):
        for empty_bank, select, reason in ((True, True, "no_memory"),
                                           (False, False, "no_selection")):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = make_prepared_fixture(root / "source", empty_bank=empty_bank,
                                               select=select)
                probability = .99 if select else 1e-9
                summary = collect_target_auto(source["manifest"], bank=source["bank"],
                    role="target_train", spec=StudySpec(max_steps=4, max_model_calls=4),
                    env_factory=source["env_factory"], actor_factory=source["actor_factory"],
                    output_dir=root / "auto", encoder=EncoderFixture(), assignment_probability=.5,
                    hazard=RiskHazard(low_risk_probability=probability,
                                      high_risk_probability=probability))
                self.assertEqual(summary["counts"], {"none": 1})
                dataset = build_policy_dataset([{"kind": "auto", "run": str(root / "auto")}])
                self.assertEqual(dataset["summary"]["by_reason"], {reason: 1})
                self.assertEqual(dataset["summary"]["label_counts"]["observed_reward"], 0)


if __name__ == "__main__":
    unittest.main()
