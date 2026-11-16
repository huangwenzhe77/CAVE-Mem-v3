import unittest

from cave_mem_v3.common.contracts import digest
from cave_mem_v3.target_data.sampling import (
    OnlineCheckpointSampler, RiskHazard, select_first_checkpoint,
)


def step(index, observation="Workspace ready.", **extra):
    checkpoint = {"state": index}
    return {
        "index": index,
        "context": {"task_id": "task-a", "instruction": "Complete the task.",
                    "observation": observation, "history": []},
        "checkpoint_hash": digest(checkpoint),
        "checkpoint": checkpoint,
        **extra,
    }


class SequentialCheckpointSamplingTests(unittest.TestCase):
    def test_incremental_sampler_stops_sampling_but_not_the_rollout(self):
        sampler = OnlineCheckpointSampler(
            role="target_train", task_id="task-a", trace_id="trace-a", seed=1,
            hazard=RiskHazard(low_risk_probability=.5, high_risk_probability=.5))
        first = step(0)
        selected = sampler.consider(first["context"], first["checkpoint"], 0, [])
        self.assertEqual(selected["status"], "selected")

        # Invalid later inputs prove no further risk evaluation occurs. The
        # caller can continue executing the rollout and simply ignore this.
        later = sampler.consider(None, None, 1, None)
        self.assertEqual(later["status"], "already_selected")
        result = sampler.finalize()
        self.assertEqual(result["selected_step_index"], 0)
        self.assertEqual(len(result["selection_decisions"]), 1)

    def test_each_rollout_gets_a_fresh_sampler(self):
        hazard = RiskHazard(low_risk_probability=.5, high_risk_probability=.5)
        selections = []
        for trace_id in ("trace-a", "trace-b"):
            sampler = OnlineCheckpointSampler(
                role="target_train", task_id="task-a", trace_id=trace_id,
                seed=1, hazard=hazard)
            row = step(0)
            selections.append(sampler.consider(row["context"], row["checkpoint"], 0, []))
        self.assertEqual([entry["status"] for entry in selections], ["selected", "selected"])

    def test_incremental_sampler_records_hazard_times_prior_survival(self):
        sampler = OnlineCheckpointSampler(
            role="target_train", task_id="task-a", trace_id="trace-a", seed=10,
            hazard=RiskHazard(low_risk_probability=.5, high_risk_probability=.5))
        for index in range(2):
            row = step(index)
            result = sampler.consider(row["context"], row["checkpoint"], index, [])
        self.assertEqual(result["status"], "selected")
        self.assertAlmostEqual(result["decision"]["conditional_inclusion_probability"], .25)
        self.assertAlmostEqual(sampler.finalize()["inclusion_probability"], .25)

    def test_incremental_sampler_marks_hash_mismatch_incomplete(self):
        sampler = OnlineCheckpointSampler(
            role="target_train", task_id="task-a", trace_id="trace-a", seed=0)
        row = step(0)
        result = sampler.consider(row["context"], row["checkpoint"], 0, [],
                                  checkpoint_hash="wrong-hash")
        self.assertEqual(result["status"], "sampling_incomplete")
        self.assertEqual(sampler.finalize()["reason"], "invalid_checkpoint_event")

    def test_high_risk_points_have_higher_hazard_than_low_risk_points(self):
        hazard = RiskHazard()
        low = select_first_checkpoint([step(0)], role="target_train", task_id="task-a",
                                      trace_id="trace-a", seed=1, hazard=hazard)
        high = select_first_checkpoint([step(0, "Previous command failed.")],
                                       role="target_train", task_id="task-a",
                                       trace_id="trace-a", seed=1, hazard=hazard)
        self.assertGreater(high["selection_decisions"][0]["selection_hazard"],
                           low["selection_decisions"][0]["selection_hazard"])
        self.assertGreater(low["selection_decisions"][0]["selection_hazard"], 0)

    def test_selects_first_hit_and_never_consumes_future_steps(self):
        hazard = RiskHazard(low_risk_probability=.5, high_risk_probability=.5)
        consumed = []

        def steps():
            consumed.append(0)
            yield step(0)
            consumed.append(1)
            yield step(1, "Previous command failed.")
            raise AssertionError("future step was consumed after selection")

        result = select_first_checkpoint(steps(), role="target_train", task_id="task-a",
                                         trace_id="trace-a", seed=1, hazard=hazard)
        self.assertEqual(result["status"], "selected")
        self.assertEqual(result["selected_step_index"], 0)
        self.assertEqual(consumed, [0])
        self.assertEqual(result["inclusion_probability"], .5)

    def test_inclusion_probability_is_hazard_times_prior_survival(self):
        hazard = RiskHazard(low_risk_probability=.5, high_risk_probability=.5)
        result = select_first_checkpoint([step(0), step(1)], role="target_train",
                                         task_id="task-a", trace_id="trace-a",
                                         seed=10, hazard=hazard)
        self.assertEqual(result["selected_step_index"], 1)
        self.assertAlmostEqual(result["inclusion_probability"], .25)
        self.assertAlmostEqual(result["selection_decisions"][1]["survival_probability_before"], .5)

    def test_no_selection_records_probability_and_draws(self):
        result = select_first_checkpoint([step(0), step(1)], role="validation",
                                         task_id="task-a", trace_id="trace-a", seed=1)
        self.assertEqual(result["status"], "no_selection")
        self.assertEqual(result["reason"], "rollout_ended_without_selection")
        self.assertEqual(len(result["selection_decisions"]), 2)
        expected = 1.0
        for decision in result["selection_decisions"]:
            expected *= 1 - decision["selection_hazard"]
        self.assertAlmostEqual(result["no_selection_probability"], expected)

    def test_current_action_result_and_episode_outcome_cannot_change_features(self):
        baseline = select_first_checkpoint([step(0)], role="target_train", task_id="task-a",
                                           trace_id="trace-a", seed=10)
        polluted = select_first_checkpoint([step(
            0, action={"command": "danger"}, result={"valid": False, "exit_code": 1},
            success=False, future_steps=[{"result": {"valid": False}}])],
            role="target_train", task_id="task-a", trace_id="trace-a", seed=10)
        self.assertEqual(baseline["selection_decisions"][0]["risk_features"],
                         polluted["selection_decisions"][0]["risk_features"])
        self.assertEqual(baseline["selection_decisions"][0]["uniform_draw"],
                         polluted["selection_decisions"][0]["uniform_draw"])

    def test_prior_action_is_not_double_counted_from_context_history(self):
        prior = {"type": "shell", "command": "git status"}
        rows = [step(0, action=prior, result={"valid": True}),
                step(1, action={"command": "next"})]
        rows[1]["context"]["history"] = [{"action": prior}]
        result = select_first_checkpoint(rows, role="target_train", task_id="task-a",
                                         trace_id="trace-a", seed=0,
                                         hazard=RiskHazard(low_risk_probability=.5,
                                                           high_risk_probability=.5))
        second = result["selection_decisions"][1]
        self.assertFalse(second["risk_features"]["repeated_prior_action"])

    def test_prior_tool_failure_increases_the_next_checkpoint_hazard(self):
        hazard = RiskHazard(low_risk_probability=.05, high_risk_probability=.8)
        failed = select_first_checkpoint(
            [step(0, action={"command": "build"}, result={"valid": False, "tool_error": True}),
             step(1)], role="target_train", task_id="task-a", trace_id="trace-a",
            seed=1, hazard=hazard)
        clean = select_first_checkpoint(
            [step(0, action={"command": "build"}, result={"valid": True}), step(1)],
            role="target_train", task_id="task-a", trace_id="trace-a", seed=1,
            hazard=hazard)
        self.assertTrue(failed["selection_decisions"][1]["risk_features"]["prior_tool_failure"])
        self.assertGreater(failed["selection_decisions"][1]["selection_hazard"],
                           clean["selection_decisions"][1]["selection_hazard"])

    def test_missing_checkpoint_snapshot_fails_closed(self):
        missing = step(0)
        del missing["checkpoint"]
        result = select_first_checkpoint([missing], role="target_train", task_id="task-a",
                                         trace_id="trace-a", seed=0)
        self.assertEqual(result["status"], "sampling_incomplete")
        self.assertEqual(result["reason"], "invalid_checkpoint_event")

    def test_checkpoint_hash_mismatch_fails_closed(self):
        invalid = step(0)
        invalid["checkpoint"]["state"] = "tampered"
        result = select_first_checkpoint([invalid], role="target_train", task_id="task-a",
                                         trace_id="trace-a", seed=0)
        self.assertEqual(result["status"], "sampling_incomplete")
        self.assertIn("does not match", result["error"]["message"])

    def test_malformed_later_step_preserves_earlier_decision_log(self):
        invalid = step(1)
        del invalid["checkpoint_hash"]
        result = select_first_checkpoint([step(0), invalid], role="target_train",
                                         task_id="task-a", trace_id="trace-a", seed=0,
                                         hazard=RiskHazard(low_risk_probability=.05,
                                                           high_risk_probability=.05))
        self.assertEqual(result["status"], "sampling_incomplete")
        self.assertEqual(len(result["selection_decisions"]), 1)
        self.assertEqual(result["selection_decisions"][0]["step_index"], 0)

    def test_stream_error_preserves_partial_decision_log(self):
        def broken_stream():
            yield step(0)
            raise OSError("rollout transport closed")

        result = select_first_checkpoint(broken_stream(), role="target_train",
                                         task_id="task-a", trace_id="trace-a", seed=0,
                                         hazard=RiskHazard(low_risk_probability=.05,
                                                           high_risk_probability=.05))
        self.assertEqual(result["status"], "sampling_incomplete")
        self.assertEqual(result["reason"], "rollout_stream_error")
        self.assertEqual(len(result["selection_decisions"]), 1)
        self.assertEqual(result["error"]["type"], "OSError")

    def test_rejects_non_target_roles_and_invalid_hazards(self):
        for role in ("history", "test", "unknown"):
            with self.subTest(role=role), self.assertRaises(ValueError):
                select_first_checkpoint([], role=role, task_id="task-a", trace_id="trace-a", seed=0)
        with self.assertRaises(ValueError):
            RiskHazard(low_risk_probability=0)
        with self.assertRaises(ValueError):
            RiskHazard(low_risk_probability=.5, high_risk_probability=1)


if __name__ == "__main__":
    unittest.main()
