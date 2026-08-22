import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from habitat_mas.fault_injection.trajectory import TrajectoryRecorder


class FaultTrajectoryTest(unittest.TestCase):
    def test_records_clean_episode_as_ordered_jsonl_events(self):
        with TemporaryDirectory() as temp_dir:
            recorder = TrajectoryRecorder(temp_dir, "clean run", "clean")
            recorder.record_episode_start(
                episode_id=7,
                scene_id="scene.glb",
                text_context={"scene_description": "cup in kitchen"},
            )
            recorder.record_step(
                episode_id=7,
                scene_id="scene.glb",
                simulator_step=0,
                joint_env_action=np.asarray([0.1, 0.2]),
                world_state_before={
                    "agents": [
                        {"agent": "agent_0", "position": [1.0, 0.0, 2.0]}
                    ]
                },
                policy_info={
                    "agent_0_decision_event": {
                        "decision_step": 0,
                        "resulting_decision": {"action": "nav_to_obj"},
                    }
                },
                reward=np.float32(0.5),
                done=False,
                metrics={
                    "pddl_success": np.float32(0.0),
                    "nested_decision": {
                        "resulting_decision": {
                            "action": "wait",
                            "arguments": ["500"],
                        }
                    },
                },
            )
            recorder.record_episode_end(
                episode_id=7,
                scene_id="scene.glb",
                simulator_step=0,
                result={"pddl_success": 0.0},
            )

            path = Path(temp_dir) / "clean_run_trajectory.jsonl"
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(
                [event["type"] for event in events],
                [
                    "trajectory_start",
                    "episode_start",
                    "simulator_step",
                    "episode_end",
                ],
            )
            self.assertEqual(
                [event["event_index"] for event in events], [0, 1, 2, 3]
            )
            self.assertEqual(events[2]["joint_env_action"], [0.1, 0.2])
            self.assertEqual(
                events[2]["world_state_before"]["agents"][0]["position"],
                [1.0, 0.0, 2.0],
            )
            self.assertEqual(events[2]["mode"], "clean")
            self.assertEqual(
                events[2]["policy_info"]["agent_0_decision_event"]
                ["resulting_decision"]["action"],
                "nav_to_obj",
            )
            self.assertEqual(
                events[2]["metrics"]["nested_decision"]
                ["resulting_decision"]["arguments"],
                ["500"],
            )

    def test_fault_event_can_be_embedded_in_faulty_step(self):
        with TemporaryDirectory() as temp_dir:
            recorder = TrajectoryRecorder(temp_dir, "faulty", "faulty")
            recorder.record_step(
                episode_id="9",
                scene_id="scene.glb",
                simulator_step=12,
                joint_env_action=[0.0],
                world_state_before={},
                policy_info={
                    "agent_1_fault_event": {
                        "fault_id": "PF_wrong_location_001",
                        "changes": [
                            {"before": "kitchen", "after": "bedroom"}
                        ],
                    }
                },
                reward=0.0,
                done=False,
                metrics={},
            )

            events = [
                json.loads(line)
                for line in recorder.path.read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            fault = events[1]["policy_info"]["agent_1_fault_event"]
            self.assertEqual(fault["changes"][0]["before"], "kitchen")
            self.assertEqual(events[1]["mode"], "faulty")


if __name__ == "__main__":
    unittest.main()
