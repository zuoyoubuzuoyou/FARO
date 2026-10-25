"""Recorder-v2 timeouts retain evidence until control actually hands over."""

import unittest

from habitat_mas.detection.heuristics import TeamProgressDetector
from habitat_mas.detection.readers import FrameStream, frames_from_records


def event(kind, step, **values):
    return {"schema_version": 2, "event_type": kind, "episode_id": "test",
            "evaluation_id": "test-attempt", "env_step": step, **values}


def start_records():
    return [
        event("episode_start", 0, context={"assignments": {
            "agent_0": "Place any_targets|0 at TARGET_any_targets|0 and "
                       "any_targets|1 at TARGET_any_targets|1.",
            "agent_1": "Nothing to do.",
        }}),
        event("action_start", 1, agent_id="agent_0", action_id="reset-1",
              action_name="reset_arm", parameters={}),
        event("action_start", 1, agent_id="agent_1", action_id="wait-1",
              action_name="wait", parameters=["500"]),
    ]


def step_record(step):
    return event("step", step, done=False, state={
        "agents": {
            "agent_0": {"base_position": [0, 0, 0],
                        "base_rotation": [0, 0, 0, 1], "grasped_object_id": None},
            "agent_1": {"base_position": [5, 0, 0],
                        "base_rotation": [0, 0, 0, 1], "grasped_object_id": None},
        },
        "target_objects": [
            {"object_id": 20, "position": [0, 0, 0], "goal_position": [0, 0, 0]},
            {"object_id": 21, "position": [1, 0, 0], "goal_position": [1.664, 0, 0]},
        ],
    }, metrics={"pddl_success": False,
                "object_to_goal_distance": {"0": 0.0, "1": 0.664}})


def end_record(*, result=None, status="timeout", action_id="reset-1"):
    return event("action_end", 100, agent_id="agent_0", action_id=action_id,
                 action_name="reset_arm", status=status,
                 reason="max_skill_steps", result=result or {})


class DetectionHandoffTests(unittest.TestCase):
    def frames_after_end(self, end, *following):
        records = start_records() + [step_record(1), step_record(51), end]
        return list(frames_from_records(records + list(following), "Manipulation"))

    def test_timeout_without_handoff_still_alarms_for_unfinished_owner(self):
        frames = self.frames_after_end(
            end_record(result={"call_high_level": False, "bad_terminate": True}),
            *(step_record(step) for step in range(101, 652, 50)),
        )
        detector = TeamProgressDetector(window=10, check_every=5)
        alarms = [alarm for frame in frames
                  if (alarm := detector.observe(frame)) is not None]
        self.assertTrue(alarms, "An unfinished owner's failed handoff vanished")
        self.assertEqual(alarms[0]["status"], "system_error")
        self.assertEqual(alarms[0]["agents"], ["agent_0"])

    def test_timeout_retains_original_scope_and_explicit_termination(self):
        frames = self.frames_after_end(
            end_record(result={"call_high_level": False}), step_record(101),
        )
        agent = frames[-1]["agents"]["agent_0"]
        self.assertEqual(agent["action"]["name"], "reset_arm")
        self.assertEqual(agent["action"]["id"], "reset-1")
        self.assertEqual(agent["action"]["age"], 101)
        self.assertEqual(agent["termination"]["status"], "timeout")
        self.assertIs(agent["termination"]["call_high_level"], False)

    def test_normal_completion_does_not_remain_an_active_reset(self):
        frames = self.frames_after_end(
            end_record(status="controller_terminated",
                       result={"call_high_level": True, "bad_terminate": False}),
            *(step_record(step) for step in range(101, 652, 50)),
        )
        detector = TeamProgressDetector(window=10, check_every=5)
        for frame in frames[2:]:
            self.assertIsNone(frame["agents"]["agent_0"]["action"]["name"])
        self.assertTrue(all(detector.observe(frame) is None for frame in frames))

    def test_timeout_with_handoff_or_missing_signal_does_not_infer_failure(self):
        for result in ({"call_high_level": True}, {}, {"call_high_level": None}):
            with self.subTest(result=result):
                frames = self.frames_after_end(end_record(result=result), step_record(101))
                self.assertIsNone(frames[-1]["agents"]["agent_0"]["action"]["name"])

    def test_new_action_replaces_timeout_and_clears_termination(self):
        frames = self.frames_after_end(
            end_record(result={"call_high_level": False}),
            event("action_start", 101, agent_id="agent_0", action_id="nav-2",
                  action_name="nav_to_obj", parameters={"target_obj": "any_targets|1"}),
            step_record(101),
        )
        agent = frames[-1]["agents"]["agent_0"]
        self.assertEqual(agent["action"]["id"], "nav-2")
        self.assertEqual(agent["action"]["age"], 1)
        self.assertIsNone(agent["termination"])

    def test_stale_action_end_cannot_overwrite_new_action_or_termination(self):
        frames = self.frames_after_end(
            end_record(result={"call_high_level": False}),
            event("action_start", 101, agent_id="agent_0", action_id="nav-2",
                  action_name="nav_to_obj", parameters={"target_obj": "any_targets|1"}),
            end_record(result={"call_high_level": False}), step_record(101),
        )
        agent = frames[-1]["agents"]["agent_0"]
        self.assertEqual(agent["action"]["id"], "nav-2")
        self.assertIsNone(agent["termination"])

    def test_later_explicit_handoff_clears_retained_timeout_scope(self):
        frames = self.frames_after_end(
            end_record(result={"call_high_level": False}), step_record(101),
            event("skill_termination_signal", 102, agent_id="agent_0",
                  action_id="reset-1", call_high_level=True, bad_terminate=False),
            step_record(102),
        )
        self.assertIsNone(frames[-1]["agents"]["agent_0"]["action"]["name"])
        self.assertIs(frames[-1]["agents"]["agent_0"]["termination"]["call_high_level"], True)

    def test_stale_handoff_signal_cannot_clear_retained_timeout(self):
        frames = self.frames_after_end(
            end_record(result={"call_high_level": False}), step_record(101),
            event("skill_termination_signal", 102, agent_id="agent_0",
                  action_id="previous-action", call_high_level=True,
                  bad_terminate=False), step_record(102),
        )
        agent = frames[-1]["agents"]["agent_0"]
        self.assertEqual(agent["action"]["id"], "reset-1")
        self.assertIs(agent["termination"]["call_high_level"], False)

    def test_episode_start_clears_retained_timeout_scope(self):
        stream = FrameStream("Manipulation")
        for record in start_records() + [step_record(1),
                end_record(result={"call_high_level": False})]:
            stream.event(record)
        stream.event(event("episode_start", 0, context={}))
        frame = stream.event(step_record(1))
        self.assertIsNone(frame["agents"]["agent_0"]["action"]["name"])
        self.assertIsNone(frame["agents"]["agent_0"]["termination"])


if __name__ == "__main__":
    unittest.main()
