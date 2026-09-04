import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from habitat.utils.visualizations.utils import overlay_frame

from habitat_mas.fault_injection.trajectory import (
    AgentDropoutTrajectoryRecorder,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info


class AgentDropoutTrajectoryTest(unittest.TestCase):
    def test_video_overlay_accepts_empty_fault_action_arguments(self):
        frame = np.zeros((32, 64, 3), dtype=np.uint8)

        rendered = overlay_frame(
            frame,
            {"agent_1_execution_override": {"arguments": []}},
        )

        self.assertEqual(rendered.shape, frame.shape)

    def test_non_numeric_policy_metadata_is_not_treated_as_metric(self):
        metrics = extract_scalars_from_info(
            {
                "reward_metric": np.float32(1.5),
                "agent_1_high_level_decision": {
                    "arguments": ["500"],
                    "action": "wait",
                },
            }
        )

        self.assertEqual(metrics, {"reward_metric": 1.5})

    def test_records_ordered_clean_or_faulty_steps(self):
        with TemporaryDirectory() as temp_dir:
            recorder = AgentDropoutTrajectoryRecorder(
                temp_dir,
                "dropout faulty",
                "faulty",
                agent_names=["agent_0", "agent_1"],
            )
            recorder.record_episode_start(
                episode_id=15,
                scene_id="scene.json",
                text_context={"pddl_text_goal": "detect both objects"},
            )
            recorder.record_step(
                episode_id=15,
                scene_id="scene.json",
                simulator_step=40,
                joint_env_action=np.asarray([1.0, 0.0, 0.0]),
                policy_info={
                    "agent_1_dropout_active": True,
                    "agent_1_dropout_event": {
                        "fault_subtype": "AgentDropout",
                        "trigger_domain": "simulator_step",
                        "simulator_step": 40,
                        "decision_step": 1,
                        "high_level_decisions_completed": 1,
                    },
                    "agent_1_execution_override": {
                        "agent": "agent_1",
                        "action": "agent_dropout_noop",
                        "arguments": [],
                    },
                },
                reward=np.float32(-0.0005),
                done=False,
                metrics={"pddl_success": False},
                simulator_state={
                    "timing": "post_env_step",
                    "agents": {
                        "agent_0": {
                            "position": [1.0, 0.0, 2.0],
                            "rotation": 0.5,
                            "held_object": None,
                        },
                        "agent_1": {
                            "position": [3.0, 1.5, 4.0],
                            "rotation": -0.25,
                            "held_object": {
                                "object_id": 12,
                                "handle": "clamp_:0000",
                                "goal_name": "any_targets|0",
                            },
                        },
                    },
                    "objects": [
                        {
                            "object_id": 12,
                            "handle": "clamp_:0000",
                            "goal_name": "any_targets|0",
                            "position": [3.1, 1.4, 4.1],
                        }
                    ],
                },
            )

            events = [
                json.loads(line)
                for line in Path(recorder.path)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [event["type"] for event in events],
                ["trajectory_start", "episode_start", "simulator_step"],
            )
            self.assertEqual(
                [event["event_index"] for event in events], [0, 1, 2]
            )
            self.assertEqual(events[2]["simulator_step"], 40)
            self.assertEqual(events[2]["joint_env_action"], [1.0, 0.0, 0.0])
            self.assertTrue(
                events[2]["policy_info"]["agent_1_dropout_active"]
            )
            self.assertEqual(
                events[2]["agent_step_actions"]["agent_1"]["status"],
                "fault_execution_override",
            )
            dynamic_state = events[2]["simulator_state"]
            self.assertEqual(dynamic_state["timing"], "post_env_step")
            self.assertEqual(
                dynamic_state["agents"]["agent_1"]["position"],
                [3.0, 1.5, 4.0],
            )
            self.assertTrue(
                dynamic_state["agents"]["agent_1"]["dropout_state"][
                    "active"
                ]
            )

            text_path = recorder.text_path_for_episode(15)
            text_trace = text_path.read_text(encoding="utf-8")
            self.assertIn("Task: detect both objects", text_trace)
            self.assertIn("Simulator_Step: 40", text_trace)
            self.assertIn(
                "Agent_1_Action: AgentDropoutNoOp[]",
                text_trace,
            )
            self.assertIn("Agent_0_Action: NotRecorded[]", text_trace)
            self.assertIn(
                "Agent_1_Position: [3.0000, 1.5000, 4.0000]",
                text_trace,
            )
            self.assertIn("Agent_1_Rotation: -0.2500", text_trace)
            self.assertIn(
                "Agent_1_Held_Object: any_targets|0 (object_id=12)",
                text_trace,
            )
            self.assertIn("Agent_1_Dropout_State: active", text_trace)
            self.assertIn("Object_Positions:", text_trace)
            self.assertIn(
                "Object_Position: name=any_targets|0, object_id=12",
                text_trace,
            )
            self.assertIn(
                "policy and LLM were not called", text_trace
            )
            self.assertIn(
                "Fault: StructureFault/AgentDropout injected", text_trace
            )

    def test_high_level_dropout_replaces_a_decision_action(self):
        with TemporaryDirectory() as temp_dir:
            recorder = AgentDropoutTrajectoryRecorder(
                temp_dir, "decision", "faulty"
            )
            recorder.record_episode_start(
                episode_id=15,
                scene_id="scene.json",
                text_context={"pddl_text_goal": "detect the clamp"},
            )
            event = {
                "trigger_domain": "high_level_decision",
                "phase": "PRE_HIGH_LEVEL_DECISION",
                "agent": "agent_1",
                "decision_step": 2,
                "simulator_step": 87,
            }
            recorder.record_step(
                episode_id=15,
                scene_id="scene.json",
                simulator_step=87,
                joint_env_action=[0.0, 0.0],
                policy_info={
                    "agent_1_dropout_event": event,
                    "agent_1_high_level_decision": {
                        "agent": "agent_1",
                        "action": "agent_dropout_noop",
                        "arguments": [],
                        "decision_step": 2,
                    },
                },
                reward=0.0,
                done=False,
                metrics={},
            )

            text_trace = recorder.text_path_for_episode(15).read_text(
                encoding="utf-8"
            )
            self.assertIn("Agent_1_Decision_Step: 2", text_trace)
            self.assertIn(
                "Agent_1_Action: AgentDropoutNoOp[]", text_trace
            )
            self.assertNotIn("Execution_Override", text_trace)
            self.assertIn("simulator step 87", text_trace)

    def test_text_trace_writes_every_step_and_carries_actions_forward(self):
        with TemporaryDirectory() as temp_dir:
            recorder = AgentDropoutTrajectoryRecorder(
                temp_dir, "clean", "clean", agent_names=["agent_0"]
            )
            recorder.record_episode_start(
                episode_id="7",
                scene_id="scene.json",
                text_context={
                    "pddl_text_goal": "move the clamp",
                    "scene_description": "clamp is near the table",
                },
            )
            recorder.record_step(
                episode_id="7",
                scene_id="scene.json",
                simulator_step=0,
                joint_env_action=[0.1, 0.2],
                policy_info={
                    "agent_0_high_level_decision": {
                        "agent": "agent_0",
                        "action": "nav_to_obj",
                        "arguments": {"target_obj": "clamp"},
                    }
                },
                reward=0.0,
                done=False,
                metrics={},
                text_context={
                    "scene_description": "clamp is near the table"
                },
            )
            recorder.record_step(
                episode_id="7",
                scene_id="scene.json",
                simulator_step=1,
                joint_env_action=[0.3, 0.4],
                policy_info={},
                reward=0.0,
                done=False,
                metrics={},
            )

            text_trace = recorder.text_path_for_episode("7").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "Agent_0_Action: Navigate[target_obj=clamp]", text_trace
            )
            self.assertIn("Simulator_Step: 1", text_trace)
            self.assertEqual(
                text_trace.count(
                    "Agent_0_Action: Navigate[target_obj=clamp]"
                ),
                2,
            )
            self.assertIn(
                "Agent_0_Status: new_high_level_decision", text_trace
            )
            self.assertIn("Agent_0_Status: continuing", text_trace)


if __name__ == "__main__":
    unittest.main()
