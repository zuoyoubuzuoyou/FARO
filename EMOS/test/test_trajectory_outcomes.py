"""Correlate real returned action observations without changing model input."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from test_trajectory_recorder import load_recorder
from trajectory_test_utils import read_episode_events


class TrajectoryOutcomeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.recorder = load_recorder()
        self.trace = self.recorder.EvaluationTrace(temporary.name)
        self.addCleanup(self.trace.close)
        self.trace.start_episode(0, SimpleNamespace(scene_id="scene", episode_id="61"),
                                 state={"initial_only": True})

    def events(self, kind):
        return [event for event in read_episode_events(self.trace.run_dir)[0]
                if event["event_type"] == kind]

    def last_event(self, kind):
        events = self.events(kind)
        self.assertTrue(events, "Missing observational event: " + kind)
        return events[-1]

    def state(self, step, distance=1.11):
        return {"agents": {"agent_0": {"base_position": [distance, 0, 0]}},
                "current_goal": {"env_step": step, "timing": "post_env_step",
                                 "any_at": {"any_targets|0": False}},
                "target_objects": [{"object_id": 7, "position": [0, 0, 0]}]}

    def test_deferred_end_records_returned_state_and_preserves_controller_status(self):
        with self.trace.policy_step():
            action_id = self.recorder.begin_action("agent_0", "nav_to_obj", {"target_obj": "any_targets|0"})
            self.recorder.end_action("agent_0", "controller_terminated", "skill_done_signal",
                                     {"call_high_level": True, "bad_terminate": False})
        snapshot = self.state(1)
        self.trace.record_step(0, [1], 0, False, {"pddl_success": False}, snapshot)
        snapshot["agents"]["agent_0"]["base_position"][0] = 99
        outcomes = self.events("action_outcome")
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertEqual(outcome["action_id"], action_id)
        self.assertEqual(outcome["termination_signal_step"], 1)
        self.assertEqual(outcome["observation_env_step"], 1)
        self.assertTrue(outcome["actual_execution_observed"])
        self.assertFalse(outcome["is_placeholder"])
        self.assertEqual(outcome["status"], "controller_terminated")
        self.assertEqual(outcome["controller_result"], {"call_high_level": True, "bad_terminate": False})
        self.assertIs(outcome["metrics"]["pddl_success"], False)
        self.assertEqual(outcome["state"]["agents"]["agent_0"]["base_position"], [1.11, 0, 0])
        self.assertLess(outcome["sequence"], self.events("step")[0]["sequence"])

    def test_terminal_outcome_is_available_in_the_terminal_detector_frame(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "habitat-mas"))
        from habitat_mas.detection.readers import iter_frames

        with self.trace.policy_step():
            action_id = self.recorder.begin_action("agent_0", "nav_to_obj", {})
        episode_dir = self.trace._episodes[0].directory
        self.trace.record_step(0, [0], 0, True, {"pddl_success": False}, self.state(1), success=False)
        frames = list(iter_frames(episode_dir, "perception"))
        self.assertEqual(len(frames), 1)
        feedback = [item for decision in frames[0]["decisions"]
                    for item in decision.get("feedback", [])]
        outcomes = [item for item in feedback if item.get("kind") == "action_outcome"]
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["action_id"], action_id)
        self.assertEqual(outcomes[0]["observation_env_step"], 1)

    def test_next_execution_request_links_prior_outcome_and_exact_placeholder(self):
        with self.trace.policy_step():
            with self.recorder.decision_scope("agent_0", "execution"):
                self.recorder.record_event("tool_response_placeholder", agent_name="agent_0",
                    phase="execution", tool_call_id="tool-nav", action_name="nav_to_obj",
                    content="Success", is_placeholder=True, actual_execution_observed=False)
                previous_id = self.recorder.begin_action("agent_0", "nav_to_obj", {})
                self.recorder.end_action("agent_0", "controller_terminated", "skill_done_signal")
        self.trace.record_step(0, [1], 0, False, {"pddl_success": False}, self.state(1))
        messages = [{"role": "system", "content": "task"},
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "tool-nav"}]},
                    {"role": "tool", "tool_call_id": "tool-nav", "name": "nav_to_obj", "content": "Success"},
                    {"role": "user", "content": "You have completed your previous action."}]
        original = deepcopy(messages)
        with self.trace.policy_step():
            with self.recorder.decision_scope("agent_0", "execution"):
                self.recorder.record_event("llm_request", agent_name="agent_0", phase="execution",
                                           call_id="next-request", messages=messages, model="test")
        feedback = self.last_event("execution_feedback")
        self.assertEqual(feedback["preceding_action_id"], previous_id)
        self.assertEqual(feedback["observation_env_step"], 1)
        self.assertEqual(feedback["call_id"], "next-request")
        self.assertTrue(feedback["actual_execution_observed"])
        self.assertEqual(feedback["model_feedback"]["tool_messages"], [{
            "tool_call_id": "tool-nav", "name": "nav_to_obj", "content": "Success", "is_placeholder": True}])
        self.assertEqual(feedback["model_feedback"]["user_message"], original[-1]["content"])
        self.assertEqual(messages, original)
        self.assertEqual(self.events("llm_request")[-1]["messages"], original)

    def test_initial_request_does_not_invent_a_previous_executed_action(self):
        with self.trace.policy_step():
            with self.recorder.decision_scope("agent_0", "execution"):
                self.recorder.record_event("llm_request", phase="execution", call_id="first",
                                           messages=[{"role": "user", "content": "start"}])
        feedback = self.last_event("execution_feedback")
        self.assertIsNone(feedback["preceding_action_id"])
        self.assertIsNone(feedback["observation_env_step"])
        self.assertFalse(feedback["actual_execution_observed"])

    def test_superseded_action_uses_its_own_last_returned_observation(self):
        with self.trace.policy_step():
            old_id = self.recorder.begin_action("agent_0", "nav_to_obj", {})
        self.trace.record_step(0, [1], 0, False, {}, self.state(1, 2.5))
        with self.trace.policy_step():
            new_id = self.recorder.begin_action("agent_0", "wait", {})
        self.trace.record_step(0, [0], 0, False, {}, self.state(2, 0.5))
        outcome = self.last_event("action_outcome")
        self.assertEqual(outcome["action_id"], old_id)
        self.assertNotEqual(old_id, new_id)
        self.assertEqual(outcome["observation_env_step"], 1)
        self.assertEqual(outcome["state"]["agents"]["agent_0"]["base_position"], [2.5, 0, 0])

    def test_action_without_returned_step_has_no_fabricated_outcome_state(self):
        with self.trace.policy_step():
            self.recorder.begin_action("agent_0", "nav_to_obj", {})
        self.trace.close(error=RuntimeError("environment step failed"))
        outcome = self.last_event("action_outcome")
        self.assertFalse(outcome["actual_execution_observed"])
        self.assertIsNone(outcome["state"])
        self.assertIsNone(outcome["observation_env_step"])

    def test_deferred_old_end_cannot_close_new_action(self):
        with self.trace.policy_step():
            old_id = self.recorder.begin_action("agent_0", "nav_to_obj", {})
        self.trace.record_step(0, [1], 0, False, {}, self.state(1))
        with self.trace.policy_step():
            self.recorder.end_action("agent_0", "timeout", "max_skill_steps")
            new_id = self.recorder.begin_action("agent_0", "wait", {})
        self.trace.record_step(0, [0], 0, False, {}, self.state(2))
        self.assertEqual([e["action_id"] for e in self.events("action_end")], [old_id])
        self.assertEqual(self.trace._episodes[0].active_actions["agent_0"]["action_id"], new_id)


if __name__ == "__main__":
    unittest.main()
