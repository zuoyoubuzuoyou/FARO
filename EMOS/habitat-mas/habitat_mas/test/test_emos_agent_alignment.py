import json
import os
import tempfile
import unittest
from unittest.mock import patch

# Match the application entrypoint's import order. Habitat registers the
# rearrangement task, which in turn imports the Habitat-MAS text sensors.
import habitat  # noqa: F401
import numpy as np
import torch

from habitat_mas.tasks.habitat_mas_sensors import RobotResumeSensor

from habitat_baselines.rl.multi_agent.multi_llm_policy import (
    MultiLLMPolicy,
    complete_robot_tasks,
    group_discussion,
)
from habitat_baselines.rl.ppo.policy import PolicyActionData


class _FakeOpenAIModel:
    def __init__(self, *args, agent_name="unknown", **kwargs):
        self.agent_name = agent_name
        self.chat_history = []
        self.token_usage = 0

    def chat(self, message):
        if self.agent_name == "leader":
            # Deliberately omit agent_0 to exercise the completion fallback.
            return "{agent_1||Detect the target object}"
        return "{{yes}}"


class _FakeWaitSkill:
    _wait_ac_idx = 2


class _FakePolicy:
    _name_to_idx = {"wait": 3}
    _idx_to_name = {1: "pick", 3: "wait"}
    _skills = {3: _FakeWaitSkill()}
    _stop_action_idx = 4

    def __init__(self):
        self._cur_call_high_level = torch.tensor([False])


class RobotResumeAlignmentTest(unittest.TestCase):
    def test_f7_overrides_environment_action_with_wait(self):
        policy = _FakePolicy()
        action_data = PolicyActionData(
            actions=torch.tensor([[1.0]]),
            take_actions=torch.ones((1, 5)),
            skill_id=torch.tensor([1]),
        )

        MultiLLMPolicy._force_wait_action(policy, action_data, True)

        self.assertEqual(action_data.env_actions.tolist(), [[0, 0, 1, 0, 0]])
        self.assertEqual(action_data.skill_id.item(), 3)
        self.assertTrue(policy._cur_call_high_level.item())

    def test_runtime_numpy_scalar_skill_id_is_supported(self):
        policy = _FakePolicy()
        action_data = PolicyActionData(
            actions=torch.tensor([[1.0]]),
            take_actions=torch.ones((1, 5)),
            skill_id=np.float64(1),
        )

        self.assertEqual(
            MultiLLMPolicy._requested_skill_name(policy, action_data), "pick"
        )
        MultiLLMPolicy._force_wait_action(policy, action_data, True)
        self.assertEqual(action_data.skill_id, 3)

    def test_f8_false_positive_sets_stop_action(self):
        policy = _FakePolicy()
        action_data = PolicyActionData(
            actions=torch.tensor([[1.0]]),
            take_actions=torch.ones((1, 5)),
            skill_id=torch.tensor([1]),
        )

        MultiLLMPolicy._force_verification_result(policy, action_data, True)

        self.assertEqual(action_data.env_actions.tolist(), [[0, 0, 0, 0, 1]])

    def test_f8_central_override_sets_every_agent_stop_action(self):
        policies = [_FakePolicy(), _FakePolicy()]
        action_data = [
            PolicyActionData(
                actions=torch.tensor([[1.0]]),
                take_actions=torch.ones((1, 5)),
                skill_id=torch.tensor([1]),
            )
            for _ in policies
        ]

        for policy, agent_action in zip(policies, action_data):
            MultiLLMPolicy._force_verification_result(
                policy, agent_action, True
            )

        self.assertTrue(
            all(action.env_actions[0, 4].item() == 1 for action in action_data)
        )
        self.assertTrue(
            all(action.env_actions.sum().item() == 1 for action in action_data)
        )

    def test_historical_spot_name_resolves_for_every_active_agent(self):
        sensor = RobotResumeSensor.__new__(RobotResumeSensor)
        sensor.robot_resume_dir = os.path.join(
            os.path.dirname(__file__), "..", "data", "robot_resume"
        )

        resumes = json.loads(
            sensor.get_observation(
                [
                    {"agent_idx": 0, "agent_type": "SpotRobot_head_arm"},
                    {"agent_idx": 1, "agent_type": "DJIDrone_default"},
                ]
            )
        )

        self.assertEqual(set(resumes), {"agent_0", "agent_1"})
        self.assertEqual(resumes["agent_0"]["robot_id"], "SpotRobot_head_arm")
        self.assertEqual(resumes["agent_0"]["robot_type"], "SpotRobot")

    def test_missing_resume_fails_with_agent_context(self):
        sensor = RobotResumeSensor.__new__(RobotResumeSensor)
        with tempfile.TemporaryDirectory() as temp_dir:
            sensor.robot_resume_dir = temp_dir
            with self.assertRaisesRegex(
                FileNotFoundError, "UnknownRobot.*agent_0"
            ):
                sensor.get_observation(
                    [{"agent_idx": 0, "agent_type": "UnknownRobot"}]
                )

    def test_missing_llm_assignment_becomes_nothing_to_do(self):
        tasks = complete_robot_tasks(
            {"agent_1": "Detect the cup"}, ["agent_0", "agent_1"]
        )

        self.assertEqual(tasks["agent_0"], "Nothing to do")
        self.assertEqual(tasks["agent_1"], "Detect the cup")

    def test_group_discussion_returns_arguments_for_both_agents(self):
        robot_resume = json.dumps(
            {
                "agent_0": {
                    "robot_type": "SpotRobot",
                    "mobility": {"summary": "legged"},
                },
                "agent_1": {
                    "robot_type": "DJIDrone",
                    "mobility": {"summary": "flying"},
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch(
            "habitat_baselines.rl.multi_agent.multi_llm_policy.OpenAIModel",
            _FakeOpenAIModel,
        ):
            arguments = group_discussion(
                robot_resume,
                "There are no objects in the scene.",
                "Detect the target object.",
                save_chat_history=True,
                save_chat_history_dir=temp_dir,
                episode_id="21",
                should_numerical=False,
            )

        self.assertEqual(set(arguments), {"agent_0", "agent_1"})
        self.assertEqual(arguments["agent_0"].subtask_description, "Nothing to do")
        self.assertEqual(
            arguments["agent_1"].subtask_description,
            "Detect the target object",
        )


if __name__ == "__main__":
    unittest.main()
