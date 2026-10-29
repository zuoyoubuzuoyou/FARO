"""Worker snapshot -> V3 storage -> native heuristic and LLM observations."""

from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import test_runtime_observation as collector_fixtures
from test_trajectory_recorder import load_recorder
from trajectory_test_utils import read_episode_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "habitat-mas"))
from habitat_mas.detection.heuristics import assess_goals, TeamProgressDetector
from habitat_mas.detection.llm import LLMDetector
from habitat_mas.detection.readers import iter_frames


class RuntimeObservationPipelineTests(unittest.TestCase):
    def test_real_native_state_survives_storage_and_drives_stall_detection(self):
        env = collector_fixtures.RuntimeObservationTest().make_env()
        sim = env._env.sim
        sim.agents_mgr[0].articulated_agent.base_transformation = collector_fixtures._Transform([0, 1.5, 0])
        sim.agents_mgr[1].articulated_agent.base_transformation = collector_fixtures._Transform([10, 1.5, 0])
        sim.agents_mgr[0].articulated_agent.base_pos = [0, 0, 0]
        sim.agents_mgr[1].articulated_agent.base_pos = [10, 1.02, 0]
        sim.get_targets = lambda: ([2, 0], [[0, 0, 0], [10, 0, 0]])
        sim.get_scene_pos = lambda: [[10, 1.5, 0], [99, 0, 0], [0.8, 0.8, 0]]
        info = env._env.task.pddl_problem._sim_info
        info.robot_at_thresh = 1.0
        info.obj_ids = {"any_targets|0": 2, "any_targets|1": 0}
        info.target_ids = {"TARGET_any_targets|0": 2, "TARGET_any_targets|1": 0}
        info.all_entities = {key: object() for key in (*info.obj_ids, *info.target_ids)}
        observations = {"agent_1_detected_objects": [142], "agent_0_detected_objects": [107],
                        "agent_1_head_semantic": [[142, 142, 142]],
                        "agent_0_head_semantic": [[107, 107, 107]]}
        recorder = load_recorder()
        with tempfile.TemporaryDirectory() as directory:
            with recorder.EvaluationTrace(directory) as trace:
                trace.start_episode(0, SimpleNamespace(scene_id="scene", episode_id="64"))
                with trace.policy_step():
                    recorder.record_event("task_assignment", agent_id="agent_1",
                        subtask_description="Cover and detect any_targets|0.")
                    recorder.record_event("task_assignment", agent_id="agent_0",
                        subtask_description="Cover and detect any_targets|1.")
                    with recorder.decision_scope("agent_1", "execution"):
                        recorder.record_event("tool_response_placeholder", agent_name="agent_1",
                            phase="execution", tool_call_id="nav-call", action_name="nav_to_obj",
                            content="Success", is_placeholder=True, actual_execution_observed=False)
                        recorder.begin_action("agent_1", "nav_to_obj", {"target_obj": "any_targets|0"})
                        recorder.end_action("agent_1", "controller_terminated", "skill_done_signal",
                                            {"call_high_level": True})
                messages = [{"role": "tool", "tool_call_id": "nav-call", "name": "nav_to_obj", "content": "Success"},
                            {"role": "user", "content": "You have completed your previous action."}]
                original_messages = deepcopy(messages)
                for step in range(1, 112):
                    env._env._elapsed_steps = step
                    env._trajectory_executed_action = {"action": "agent_1_wait", "action_args": {}}
                    snapshot = env.get_info(observations)["__trajectory_state"]
                    if step == 2:
                        with trace.policy_step():
                            with recorder.decision_scope("agent_1", "execution"):
                                recorder.record_event("llm_request", phase="execution", call_id="after-nav",
                                                      model="test-only", messages=messages)
                                recorder.begin_action("agent_1", "wait", ["500"])
                            recorder.begin_action("agent_0", "wait", ["500"])
                    metrics = {"pddl_success": False, "pddl_stage_goals": {"historical_all_four": 1}}
                    trace.record_step(0, [0], 0, step == 111, metrics, snapshot, success=False)
                episode_dir = next((trace.run_dir / "episodes").iterdir())
            frames = list(iter_frames(episode_dir, "perception"))
            self.assertEqual(len(frames), 111)
            goals = assess_goals(frames[1])
            coverage = next(goal for goal in goals if goal["kind"] == "coverage" and goal["object"] == "any_targets|0")
            self.assertIs(coverage["satisfied"], False)
            self.assertFalse(coverage["approximate"])
            self.assertAlmostEqual(coverage["distance"], math.sqrt(1.13))
            self.assertTrue(all(goal["satisfied"] for goal in goals if goal["kind"] == "detection"))
            detector = TeamProgressDetector(window=3, check_every=1, min_stall_steps=100)
            alarms = [alarm for frame in frames if (alarm := detector.observe(frame))]
            self.assertTrue(alarms)
            self.assertEqual(alarms[0]["reason_code"], "unresolved_team_wait")
            llm = LLMDetector(window=3, max_calls=0)
            for frame in frames[:3]:
                llm.observe(frame)
            data = json.loads(llm._request()["messages"][1]["content"])
            self.assertEqual(llm.stats["requests"], 0)
            self.assertIs(data["recent_frames"][1]["current_goal"]["any_at"]["any_targets|0"], False)
            feedback = [item for frame in data["recent_frames"] for decision in frame["decisions"]
                        for item in decision.get("feedback") or []]
            self.assertIn("action_outcome", [item.get("kind") for item in feedback])
            self.assertIn("execution_feedback", [item.get("kind") for item in feedback])
            self.assertTrue(any(item.get("kind") == "action_outcome"
                for decision in data["recent_frames"][0]["decisions"]
                for item in decision.get("feedback") or []))
            self.assertNotIn("pddl_stage_goals", json.dumps(data))
            requests = [event for event in read_episode_events(trace.run_dir)[0]
                        if event["event_type"] == "llm_request"]
            self.assertEqual(requests[0]["messages"], original_messages)
            self.assertEqual(messages, original_messages)


if __name__ == "__main__":
    unittest.main()
