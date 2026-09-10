"""Exercise the real policy dispatch with lightweight simulator/tensor doubles.

Load only dispatch and neutral-action methods, stopping before joint tensor
packing. Full Habitat imports require native simulation and GPU dependencies.
"""

import ast
import contextlib
import io
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from habitat_mas.fault_injection.agent_dropout import (
    AgentDropoutFaultSpec, FaultInjectionError, build_agent_dropout_event,
)
from habitat_mas.fault_injection.agent_hang import AgentHangFaultSpec, AgentHangState


def tensor(value=0, shape=(1, 1)):
    result = MagicMock()
    result.shape = shape
    result.device = "cpu"
    result.any.return_value.item.return_value = bool(value)
    result.item.return_value = value
    result.__getitem__.return_value = result
    return result


def load_dispatch():
    source = Path(__file__).resolve().parents[3] / "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py"
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and any(
        isinstance(method, ast.FunctionDef) and method.name == "_make_agent_fault_action"
        for method in node.body
    ))
    methods = [method for method in cls.body if isinstance(method, ast.FunctionDef)
               and method.name in {"act", "_make_agent_fault_action"}]
    act = next(method for method in methods if method.name == "act")
    end = next(index for index, node in enumerate(act.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == "policy_info" for target in node.targets))
    act.body = act.body[:end] + [ast.Return(value=ast.Name(id="agent_actions", ctx=ast.Load()))]
    cls.bases = []
    cls.decorator_list = []
    cls.body = methods
    namespace = {
        "os": os,
        "AgentDropoutFaultSpec": AgentDropoutFaultSpec,
        "AgentHangFaultSpec": AgentHangFaultSpec,
        "AgentHangState": AgentHangState,
        "FaultInjectionError": FaultInjectionError,
        "build_agent_dropout_event": build_agent_dropout_event,
        "PolicyActionData": SimpleNamespace,
        "torch": SimpleNamespace(
            zeros_like=lambda value: tensor(), zeros=lambda *a, **kw: tensor(),
            full=lambda shape, value, **kw: value, long=int,
        ),
        "np": SimpleNamespace(zeros=lambda *a, **kw: False),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(source), "exec"), namespace)
    return namespace[cls.name], namespace


class AgentHangPolicyTest(unittest.TestCase):
    def setUp(self):
        cls, self.namespace = load_dispatch()
        self.policy = cls()
        self.policy._decision_steps = {}
        self.policy._active_dropout_events = {}
        self.policy._hang_states = {}
        for flag in ("should_group_discussion", "should_agent_reflection", "should_robot_resume", "should_numerical", "should_terminate_on_wait"):
            setattr(self.policy, flag, True)
        self.agents = []
        for _ in range(2):
            llm = SimpleNamespace(initialized=True)
            llm.init_agent = lambda llm=llm, **kwargs: setattr(llm, "initialized", True)
            llm.reset_message_pipe = MagicMock()
            agent = SimpleNamespace(
                _high_level_policy=SimpleNamespace(llm_agent=llm),
                _cur_call_high_level=[tensor(1)], _name_to_idx={"wait": 0}, _stop_action_idx=0,
                act=MagicMock(side_effect=lambda *args, **kwargs: SimpleNamespace(
                    policy_info=[{"high_level_decision": {"action": "wait"}}],
                    skill_id=0, actions=tensor(),
                )),
            )
            self.agents.append(agent)
        self.policy._active_policies = self.agents
        self.policy._build_index_split = lambda *a: {
            "index_len_recurrent_hidden_states": [1, 1], "index_len_prev_actions": [1, 1]
        }
        self.policy._update_obs_with_agent_prefix_fn = lambda obs, agent: obs
        args = SimpleNamespace(robot_type="test", task_description="task", subtask_description="subtask", chat_history=[])
        self.namespace["group_discussion"] = MagicMock(return_value={"agent_0": args, "agent_1": args})
        self.hidden = tensor(shape=(1, 2))
        self.hidden_parts = [tensor(7), tensor(8)]
        self.hidden.split.return_value = self.hidden_parts
        self.previous = tensor(shape=(1, 2))
        self.previous.split.return_value = [tensor(), tensor()]

    def act(self, step, config, episode="15", starting=False, dropout=None):
        masks = tensor(not starting, shape=(1, 2))
        masks.split.return_value = [tensor(not starting), tensor(not starting)]
        with contextlib.redirect_stdout(io.StringIO()):
            return self.policy.act(
                {"pddl_text_goal": [SimpleNamespace(tolist=lambda: [ord("x")])]},
                self.hidden, self.previous, masks,
                envs_text_context=[{"episode_id": episode, "robot_resume": "robots", "scene_description": "scene"}],
                save_chat_history=False, simulator_steps=[step], agent_hang_fault=config,
                agent_dropout_fault=dropout,
            )

    def test_hang_skips_policy_freezes_state_and_does_not_end_on_wait(self):
        config = {"enabled": True, "agent": "agent_1", "simulator_step": 1}
        self.act(0, config, starting=True)
        actions = self.act(1, config)
        self.assertEqual(self.agents[0].act.call_count, 2)
        self.assertEqual(self.agents[1].act.call_count, 1)
        self.assertIs(actions[1].rnn_hidden_states, self.hidden_parts[1])
        self.assertFalse(actions[1].should_inserts)
        self.assertFalse(actions[1].take_actions.any().item())
        self.assertIn("hang_event", actions[1].policy_info[0])
        for action in actions:
            action.actions.__setitem__.assert_not_called()
        actions = self.act(999, config)
        self.assertNotIn("hang_event", actions[1].policy_info[0])
        self.assertEqual(self.policy._decision_steps[("15", "agent_1")], 1)
        self.assertEqual(self.agents[0].act.call_count, 3)
        self.assertEqual(self.agents[1].act.call_count, 1)

    def test_step_zero_does_not_repeat_discussion_and_resets_next_episode(self):
        config = {"enabled": True, "agent": "agent_1", "episode_id": "15"}
        self.act(0, config, starting=True)
        self.act(1, config)
        self.assertEqual(self.namespace["group_discussion"].call_count, 1)
        self.agents[1].act.assert_not_called()
        actions = self.act(0, config, episode="16", starting=True)
        self.assertNotIn("hang_active", actions[1].policy_info[0])
        self.assertEqual(self.agents[1].act.call_count, 1)
        self.assertEqual(self.namespace["group_discussion"].call_count, 2)

    def test_episode_boundary_resets_inter_agent_messages(self):
        reset = self.agents[0]._high_level_policy.llm_agent.reset_message_pipe

        self.act(0, {}, starting=True)
        self.act(1, {})
        self.act(0, {}, episode="16", starting=True)

        self.assertEqual(reset.call_count, 2)
        self.agents[1]._high_level_policy.llm_agent.reset_message_pipe.assert_not_called()

    def test_repeated_same_episode_reinjects_once(self):
        config = {"enabled": True, "agent": "agent_1"}
        for _ in range(2):
            actions = self.act(0, config, starting=True)
            self.assertIn("hang_event", actions[1].policy_info[0])
            self.assertNotIn("hang_event", self.act(1, config)[1].policy_info[0])

    def test_simulator_trigger_interrupts_between_decisions(self):
        config = {"enabled": True, "agent": "agent_1", "simulator_step": 87}
        self.act(0, config, starting=True)
        self.agents[1]._cur_call_high_level = [tensor(0)]
        actions = self.act(87, config)
        info = actions[1].policy_info[0]
        self.assertEqual(info["execution_override"]["action"], "agent_hang_noop")
        self.assertNotIn("high_level_decision", info)
        self.assertEqual(info["hang_event"]["simulator_step"], 87)
        self.assertEqual(self.agents[1].act.call_count, 1)

    def test_disabled_and_dropout_regression(self):
        actions = self.act(0, {}, starting=True)
        self.assertEqual([a.act.call_count for a in self.agents], [1, 1])
        for action in actions:
            action.actions.__setitem__.assert_called_once()
        actions = self.act(1, {}, dropout={"enabled": True, "agent": "agent_1", "simulator_step": 1})
        self.assertIn("dropout_event", actions[1].policy_info[0])
        self.assertNotIn("hang_active", actions[1].policy_info[0])
        self.assertEqual(self.agents[1].act.call_count, 1)

    def test_invalid_target_and_conflicting_faults_fail_before_execution(self):
        with self.assertRaisesRegex(FaultInjectionError, "unknown"):
            self.act(0, {"enabled": True, "agent": "agent_5"})
        with self.assertRaisesRegex(FaultInjectionError, "both be enabled"):
            self.act(0, {"enabled": True}, dropout={"enabled": True})
        for agent in self.agents:
            agent.act.assert_not_called()


if __name__ == "__main__":
    unittest.main()
