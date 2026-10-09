"""Execute the actual dispatch method with real action tensors."""

import contextlib
import copy
import io
import unittest
from types import SimpleNamespace

import torch
from gym.spaces import Box, Dict

from habitat_mas.test.test_agent_hang_policy import AgentHangPolicyTest, tensor
from habitat_mas.agents.message_bus import AgentMessageBus as Bus
from habitat_mas.fault_injection.latent import ActionNoopFaultSpec, MessageDelayFaultSpec, ObjectLocalizationFaultSpec
from habitat_mas.fault_injection.latent_runtime import LatentFaultRuntime


class LatentDispatchTest(unittest.TestCase):
    def setUp(self):
        fixture = self.fixture = AgentHangPolicyTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.namespace["copy"] = copy
        for policy in fixture.agents:
            # Policy stores a flattened space; high-level policy retains named channels.
            policy._action_space = Box(-20, 20, (1,))
            policy._high_level_policy._action_space = Dict({"oracle_nav_action": Box(-20, 20, (1,))})
            policy.act.side_effect = lambda *args, **kwargs: SimpleNamespace(
                policy_info=[{}], skill_id=1, actions=torch.tensor([[7]]),
                take_actions=torch.tensor([[2.]]), should_inserts=True,
            )

    def dispatch(self, runtime, step, starting=False, **kwargs):
        fixture = self.fixture
        masks = tensor(not starting, shape=(1, 2))
        masks.split.return_value = [tensor(not starting), tensor(not starting)]
        with contextlib.redirect_stdout(io.StringIO()):
            return fixture.policy.act(
                {"pddl_text_goal": [SimpleNamespace(tolist=lambda: [ord("x")])]},
                fixture.hidden, fixture.previous, masks,
                envs_text_context=[{"episode_id": "15", "robot_resume": "robots", "scene_description": "scene"}],
                save_chat_history=False, simulator_steps=[step], latent_fault_runtime=runtime, **kwargs,
            )

    def test_localization_expiry_publishes_restored_scene_to_target_only(self):
        runtime = LatentFaultRuntime(ObjectLocalizationFaultSpec(enabled=True, simulator_step=1, duration_steps=2), "scene", "15")

        def context(**values):
            return {"scene_description": "biased" if values["offset"] is not None else (
                "restored truth" if runtime.window.expired else None)}

        for step in range(4):
            self.dispatch(runtime, step, starting=step == 0, set_localization_fault=context)
        target = self.fixture.agents[0].act.call_args.kwargs["envs_text_context"][0]
        peer = self.fixture.agents[1].act.call_args.kwargs["envs_text_context"][0]
        self.assertEqual(target["scene_description"], "restored truth")
        self.assertTrue(target["refresh_scene_description"])
        self.assertEqual(peer["scene_description"], "scene")

    def test_noop_still_calls_target_policy_and_preserves_peer(self):
        runtime = LatentFaultRuntime(ActionNoopFaultSpec(enabled=True, agent="agent_1", simulator_step=1, duration_steps=2), "scene", "15")
        self.dispatch(runtime, 0, starting=True)
        actions = self.dispatch(runtime, 1)
        self.assertEqual([p.act.call_count for p in self.fixture.agents], [2, 2])
        self.assertEqual(actions[0].take_actions.item(), 2.)
        self.assertEqual(actions[1].take_actions.item(), 0.)
        self.assertEqual(actions[1].actions.item(), 7)
        self.assertTrue(actions[1].should_inserts)
        self.assertEqual(actions[1].skill_id, 1)
        self.assertEqual(Bus.active_members(), ["agent_0", "agent_1"])
        self.assertTrue(any(event["type"] == "fault_applied" for event in actions[1].policy_info[0]["fault_events"]))
        self.assertEqual(self.dispatch(runtime, 3)[1].take_actions.item(), 2.)

    def test_delay_independent_of_policy_dispatch_order(self):
        for target in (0, 1):
            with self.subTest(target=target):
                runtime = LatentFaultRuntime(MessageDelayFaultSpec(enabled=True, agent=f"agent_{target}", simulator_step=1, delay_steps=2), "scene", "15")
                seen = []
                sender, receiver = self.fixture.agents[1-target], self.fixture.agents[target]
                original_sender, original_receiver = sender.act.side_effect, receiver.act.side_effect

                def send(*args, **kwargs):
                    if runtime.step == 1:
                        Bus.send_request(f"agent_{1-target}", f"agent_{target}", "help")
                    return original_sender(*args, **kwargs)

                def receive(*args, **kwargs):
                    seen.append((runtime.step, list(Bus.messages.get(f"agent_{target}", []))))
                    return original_receiver(*args, **kwargs)

                sender.act.side_effect, receiver.act.side_effect = send, receive
                self.dispatch(runtime, 0, starting=True)
                self.dispatch(runtime, 1)
                self.dispatch(runtime, 2)
                self.dispatch(runtime, 3)
                self.assertEqual([messages for step, messages in seen if step in (1, 2)], [[], []])
                self.assertIn("help", [messages for step, messages in seen if step == 3][0][0])
                sender.act.side_effect, receiver.act.side_effect = original_sender, original_receiver


if __name__ == "__main__":
    unittest.main()
