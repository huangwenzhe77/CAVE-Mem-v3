import unittest

from cave_mem_v3.target_data.risk_signals import risk_features


class TargetRiskSignalTests(unittest.TestCase):
    def test_empty_checkpoint_has_zero_risk(self):
        features = risk_features({"observation": "The workspace is ready."}, [])
        self.assertEqual(features["risk_score"], 0)
        self.assertEqual(features["risk_score_max"], 3)
        self.assertFalse(any(features[key] for key in (
            "explicit_public_error_cue", "prior_tool_failure", "repeated_prior_action")))

    def test_public_error_cue_increases_score(self):
        features = risk_features({"observation": "Previous command failed; retry."}, [])
        self.assertTrue(features["explicit_public_error_cue"])
        self.assertEqual(features["risk_score"], 1)

    def test_terminal_json_observation_with_nonzero_exit_is_high_risk(self):
        observation = '{"exit_code":1,"stderr":"","stdout":"command output"}'
        features = risk_features({"observation": observation}, [])
        self.assertTrue(features["explicit_public_error_cue"])
        self.assertEqual(features["risk_score"], 1)

    def test_nonzero_exit_wording_is_detected_but_zero_is_not(self):
        failed = risk_features({"observation": "command exited with status 2"}, [])
        succeeded = risk_features({"observation": '{"exit_code":0,"stderr":""}'}, [])
        self.assertTrue(failed["explicit_public_error_cue"])
        self.assertFalse(succeeded["explicit_public_error_cue"])

    def test_prior_tool_failure_is_used(self):
        features = risk_features({"observation": "Output available."}, [
            {"action": {"type": "shell", "command": "build"},
             "result": {"valid": True, "exit_code": 2}}
        ])
        self.assertTrue(features["prior_tool_failure"])
        self.assertEqual(features["risk_score"], 1)

    def test_repeated_prior_action_is_detected_without_current_action(self):
        prior = {"type": "shell", "command": "git status"}
        features = risk_features({"observation": "Ready."}, [
            {"action": prior, "result": {"valid": True}},
            {"action": {"command": "git status", "type": "shell"}, "result": {"valid": True}},
        ])
        self.assertTrue(features["repeated_prior_action"])
        self.assertEqual(features["risk_score"], 1)

    def test_context_history_is_fallback_not_double_counted(self):
        action = {"type": "shell", "command": "git status"}
        context = {"observation": "Ready.", "history": [
            {"action": action}, {"action": {"type": "shell", "command": "echo ready"}},
        ]}
        prior = [
            {"action": action},
            {"action": {"type": "shell", "command": "echo ready"}},
        ]
        features = risk_features(context, prior)
        self.assertFalse(features["repeated_prior_action"])

    def test_prior_history_is_used_when_step_list_is_empty(self):
        action = {"type": "shell", "command": "git status"}
        features = risk_features({"observation": "Ready.", "history": [
            {"action": action}, {"action": action},
        ]}, [])
        self.assertTrue(features["repeated_prior_action"])

    def test_public_invalid_step_is_a_prior_tool_failure(self):
        features = risk_features({"observation": "Ready."}, [{"valid": False}])
        self.assertTrue(features["prior_tool_failure"])

    def test_score_combines_only_the_three_registered_cues(self):
        features = risk_features({"observation": "Unable to open file."}, [
            {"action": {"type": "shell", "command": "fix"}, "validation_error": "invalid"},
            {"action": {"type": "shell", "command": "fix"}, "result": {"valid": True}},
        ])
        self.assertEqual(features["risk_score"], 3)
        self.assertLessEqual(features["risk_score"], features["risk_score_max"])

    def test_candidate_action_result_outcome_and_private_state_are_ignored(self):
        base = {"observation": "The workspace is ready.", "history": []}
        polluted = {
            **base,
            "action": {"type": "shell", "command": "repeated"},
            "result": {"valid": False, "exit_code": 1},
            "success": False,
            "private_checkpoint": {"failure": True},
            "memory": {"mechanism": "ignored"},
            "future_steps": [{"action": {"type": "shell", "command": "future"}}],
        }
        self.assertEqual(risk_features(base, []), risk_features(polluted, []))

    def test_prior_step_outcome_and_selected_action_do_not_enter_score(self):
        prior = [{"action": {"type": "shell", "command": "echo ready"},
                  "result": {"valid": True, "success": False, "score": 0}}]
        features = risk_features({"observation": "Ready."}, prior)
        self.assertEqual(features["risk_score"], 0)
        self.assertFalse(features["prior_tool_failure"])

    def test_rejects_malformed_inputs(self):
        with self.assertRaises(TypeError):
            risk_features([], [])
        with self.assertRaises(TypeError):
            risk_features({"observation": "Ready."}, {})
        with self.assertRaises(TypeError):
            risk_features({"observation": 3}, [])


if __name__ == "__main__":
    unittest.main()
