import unittest
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
from gym.spaces import Box, Dict

from habitat_mas.fault_injection.latent import ActionNoopFaultSpec, WrongSubgoalFaultSpec
from habitat_mas.fault_injection.latent_runtime import LatentFaultRuntime
from habitat_mas.fault_injection.agent_dropout import FaultInjectionError


class LatentRuntimeTest(unittest.TestCase):
    def test_only_navigation_command_masked_and_history_preserved(self):
        runtime = LatentFaultRuntime(ActionNoopFaultSpec(enabled=True, simulator_step=4, duration_steps=2), "scene", "15")
        space = Dict(OrderedDict([
            ("stop", Box(-1, 1, (1,), dtype=np.float32)),
            ("oracle_nav_action", Box(-20, 20, (1,), dtype=np.float32)),
            ("arm_action", Box(-1, 1, (2,), dtype=np.float32)),
        ]))
        original = torch.tensor([[1., 3., .2, .3]])
        history = torch.tensor([[7]])
        result = SimpleNamespace(take_actions=original, actions=history, skill_id=7, should_inserts=True)
        runtime.begin_step(3)
        runtime.apply_action("agent_0", result, space)
        self.assertIs(result.take_actions, original)
        runtime.begin_step(4)
        runtime.apply_action("agent_1", result, space)
        self.assertIs(result.take_actions, original)
        runtime.apply_action("agent_0", result, space)
        torch.testing.assert_close(result.take_actions, torch.tensor([[1., 0., .2, .3]]))
        self.assertEqual(original[0, 1], 3.)
        self.assertIs(result.actions, history)
        self.assertTrue(result.should_inserts)
        self.assertEqual(result.skill_id, 7)
        event = [e for e in runtime.drain_events() if e["type"] == "fault_applied"][0]
        self.assertEqual(event["original_action"][1], 3.)
        self.assertEqual(event["effective_action"][1], 0.)
        runtime.begin_step(6)
        result.take_actions = original
        runtime.apply_action("agent_0", result, space)
        self.assertIs(result.take_actions, original)

    def test_wait_is_triggered_but_not_effective(self):
        runtime = LatentFaultRuntime(ActionNoopFaultSpec(enabled=True), "scene", "15")
        runtime.begin_step(0)
        data = SimpleNamespace(take_actions=torch.zeros(1, 1), actions=torch.zeros(1, 1))
        runtime.apply_action("agent_0", data, Dict({"oracle_nav_action": Box(-1, 1, (1,))}))
        summary = runtime.summary()
        self.assertTrue(summary["triggered"])
        self.assertFalse(summary["effective"])
        self.assertEqual(summary["applications"], 0)

    def test_coordinate_mode_and_position_without_lookat_are_already_inactive(self):
        space = Dict(OrderedDict([("agent_0_oracle_nav_coord_action", Dict(OrderedDict([
            ("agent_0_oracle_nav_coord_action", Box(-20, 20, (3,))),
            ("agent_0_oracle_nav_lookat_action", Box(-20, 20, (3,))),
            ("agent_0_mode", Box(0, 2, (1,))),
        ])))]))
        runtime = LatentFaultRuntime(ActionNoopFaultSpec(enabled=True), "scene", "15")
        runtime.begin_step(0)
        data = SimpleNamespace(take_actions=torch.tensor([[4., 0., 2., 0., 0., 0., 2.]]), actions=None)
        runtime.apply_action("agent_0", data, space)
        self.assertFalse(runtime.summary()["effective"])
        data.take_actions = torch.tensor([[4., 0., 2., 1., 0., 1., 2.]])
        runtime.apply_action("agent_0", data, space)
        self.assertTrue(runtime.summary()["effective"])
        torch.testing.assert_close(data.take_actions, torch.tensor([[0., 0., 0., 0., 0., 0., 2.]]))

    def test_subgoal_replaces_exact_entity_and_validates_assignment(self):
        runtime = LatentFaultRuntime(WrongSubgoalFaultSpec(enabled=True), "scene", "15",
                                    valid_entities={"any_targets|0", "any_targets|1", "any_targets|10"})
        runtime.begin_step(0)
        original = "Detect any_targets|0 and any_targets|10; preserve TARGET_any_targets|0."
        changed = runtime.rewrite_assignment(original)
        self.assertEqual(changed, "Detect any_targets|1 and any_targets|10; preserve TARGET_any_targets|0.")
        with self.assertRaisesRegex(FaultInjectionError, "assignment"):
            runtime.rewrite_assignment("Nothing to do")
        invalid = LatentFaultRuntime(WrongSubgoalFaultSpec(enabled=True), "scene", "15", valid_entities={"any_targets|0"})
        invalid.begin_step(0)
        with self.assertRaisesRegex(FaultInjectionError, "entity"):
            invalid.rewrite_assignment(original)

    def test_mid_episode_planning_keeps_history_and_interrupts_old_skill(self):
        from habitat_mas.agents.crab_agent import CrabAgent
        agent = CrabAgent.__new__(CrabAgent)
        agent.robot_type, agent.name, agent.subtask_description = "Spot", "agent_0", "Detect any_targets|0"
        history = [{"role": "user", "content": "old action"}]
        agent.llm_model = SimpleNamespace(chat_history=history, set_system_message=Mock(), chat=Mock())
        agent.action_prompt, agent.start_act = "tools", True
        policy = SimpleNamespace(_cur_call_high_level=torch.tensor([False]))
        runtime = LatentFaultRuntime(WrongSubgoalFaultSpec(enabled=True, simulator_step=4), "scene", "15",
                                    valid_entities={"any_targets|0", "any_targets|1"})
        runtime.begin_step(4)
        runtime.apply_planning(agent, policy)
        self.assertIs(agent.llm_model.chat_history, history)
        self.assertIn("any_targets|1", agent.llm_model.chat.call_args.args[0])
        self.assertTrue(agent.llm_model.chat.call_args.kwargs["crab_planning"])
        self.assertTrue(policy._cur_call_high_level.item())
        self.assertFalse(agent.start_act)
        self.assertEqual(agent.subtask_description, "Detect any_targets|1")
        runtime.apply_planning(agent, policy)
        agent.llm_model.chat.assert_called_once()


if __name__ == "__main__":
    unittest.main()
