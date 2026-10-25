"""Exercise live recovery policy commands without a simulator or network."""

import ast
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from openai.types.chat import ChatCompletionMessage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.agents.crab_agent import CrabAgent
from habitat_mas.agents.crab_core import Action
from habitat_mas.utils import trajectory


def nav_to_obj(target_obj: str):
    """Navigate to an object."""


def pick(target_obj: str):
    """Pick an object."""


def place(target_obj: str, target_location: str):
    """Place an object at its goal."""


def wait():
    """Wait for another agent."""


def send_request(request: str, target_agent: str):
    """Ask another agent for help."""


ACTIONS = [Action.from_function(fn) for fn in
           (nav_to_obj, pick, place, wait, send_request)]


def load_policy():
    """Retain production policy methods, replacing simulator import boundaries."""
    source = ROOT / "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py"
    tree = ast.parse(source.read_text())
    body = [ast.ImportFrom(module="__future__",
                          names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in tree.body if isinstance(node, ast.FunctionDef)
             or isinstance(node, ast.ClassDef) and node.name == "MultiLLMPolicy"]
    namespace = {
        "MultiPolicy": object, "torch": torch, "np": np, "json": json, "os": os,
        "deepcopy": deepcopy, "record_event": trajectory.record_event,
        "MultiAgentPolicyActionData": SimpleNamespace,
        "_merge_list_dict": lambda infos: infos,
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace["MultiLLMPolicy"], namespace


MultiLLMPolicy, POLICY_GLOBALS = load_policy()


class ModelBoundary:
    def __init__(self):
        self.chat_history = [
            [{"role": "user", "content": "step %d" % step},
             ChatCompletionMessage(role="assistant", content="decision %d" % step)]
            for step in range(12)
        ]
        self.token_usage = 0
        self.requests = []

    def chat(self, prompt):
        self.requests.append(prompt)
        self.chat_history.append([{"role": "user", "content": prompt}])
        return "wait", {}


class EntityDomain:
    def __init__(self, names):
        self.all_entities = {name: SimpleNamespace(name=name) for name in names}

    def get_entity(self, name):
        # Match real PddlDomain.get_entity: an unknown name raises KeyError.
        return self.all_entities[name]

    def get_ordered_entities_list(self):
        return sorted(self.all_entities.values(), key=lambda entity: entity.name)


def make_policy():
    policy = MultiLLMPolicy.__new__(MultiLLMPolicy)
    policy.should_terminate_on_wait = False
    policy.should_group_discussion = True
    policy.should_agent_reflection = True
    policy.should_robot_resume = True
    policy.should_numerical = True
    policy._update_obs_with_agent_prefix_fn = lambda observations, _: observations
    policies = []
    for index in range(2):
        agent = CrabAgent("agent_%d" % index, ACTIONS)
        agent.initialized = True
        agent.robot_type = "test robot"
        agent.task_description = "Carry both cups to their goals."
        agent.subtask_description = "Carry cup %d." % index
        agent.planner_plan = "navigate, pick, navigate, place"
        agent.llm_model = ModelBoundary()
        high_level = SimpleNamespace(
            llm_agent=agent,
            _skill_name_to_idx={"nav_to_obj": 0, "pick": 1, "wait": 2},
        )
        child = SimpleNamespace(
            _high_level_policy=high_level,
            _cur_call_high_level=torch.zeros(1, dtype=torch.bool),
            _cur_skills=np.array([0]),
            _idx_to_name={0: "nav_to_obj", 1: "pick", 2: "wait"},
            _pddl=EntityDomain([
                "any_targets|0", "TARGET_any_targets|0", "cup", "agent_0", "agent_1",
            ]),
        )

        def child_act(observations, hidden, previous, masks, deterministic,
                      **kwargs):
            return SimpleNamespace(
                rnn_hidden_states=hidden, actions=torch.zeros_like(previous),
                take_actions=None, values=None, action_log_probs=None,
                policy_info=[{}], should_inserts=None,
            )

        child.act = child_act
        policies.append(child)
    policy._active_policies = policies
    return policy


class RecoveryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = make_policy()
        self.agent = self.policy._active_policies[0]._high_level_policy.llm_agent
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(CrabAgent, "message_pipe", {}))

    def test_context_exports_actual_plan_last_ten_complete_turns_and_only_executable_tools(self):
        value = self.policy.get_recovery_context()
        exported = value["agents"]["agent_0"]
        self.assertEqual(exported["task_description"], "Carry both cups to their goals.")
        self.assertEqual(exported["subtask_description"], "Carry cup 0.")
        self.assertEqual(exported["planner_plan"], "navigate, pick, navigate, place")
        self.assertEqual(exported["current_skill"], "nav_to_obj")
        self.assertEqual(len(exported["recent_history"]), 10)
        self.assertEqual(exported["recent_history"][0][0]["content"], "step 2")
        self.assertEqual(exported["recent_history"][0][1]["content"], "decision 2")
        self.assertEqual([tool["name"] for tool in exported["available_tools"]],
                         ["nav_to_obj", "pick", "wait"])
        self.assertEqual(exported["available_tools"][1]["parameters"]["required"],
                         ["target_obj"])
        self.assertEqual(exported["known_entities"], [
            "TARGET_any_targets|0", "agent_0", "agent_1", "any_targets|0", "cup",
        ])
        json.dumps(value)
        exported["recent_history"][0][0]["content"] = "changed"
        self.assertEqual(self.agent.llm_model.chat_history[2][0]["content"], "step 2")
        self.assertEqual(len(self.policy.get_recovery_context(history_turns=2)
                             ["agents"]["agent_0"]["recent_history"]), 2)

    def test_invalid_command_late_in_batch_causes_no_mutation(self):
        valid = {"type": "resume", "agent": "agent_0", "reason": "retry from current pose"}
        invalids = [
            {"type": "action", "agent": "missing", "name": "pick", "arguments": {"target_obj": "cup"}},
            {"type": "action", "agent": "agent_0", "name": "place", "arguments": {"target_obj": "cup", "target_location": "goal"}},
            {"type": "action", "agent": "agent_0", "name": "pick", "arguments": {}},
            {"type": "action", "agent": "agent_0", "name": "pick", "arguments": {"target_obj": 7}},
            {"type": "action", "agent": "agent_0", "name": "pick", "arguments": {"target_obj": "cup", "robot": "agent_1"}},
            {"type": "replan", "agent": "agent_0", "instruction": " "},
            {"type": "resume", "agent": "agent_0", "unexpected": True},
        ]
        history = deepcopy(self.agent.llm_model.chat_history)
        for invalid in invalids:
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.policy.apply_recovery_commands([valid, invalid])
                self.assertFalse(self.policy._active_policies[0]._cur_call_high_level[0])
                self.assertEqual(self.agent.llm_model.chat_history, history)
        self.assertEqual(self.agent.chat("next"), {"name": "wait", "arguments": ["500"]})
        self.assertEqual(self.agent.llm_model.requests, ["next"])

    def test_unknown_entity_in_late_command_is_rejected_before_any_queue_or_flag_changes(self):
        child = self.policy._active_policies[0]
        child._high_level_policy._skill_name_to_idx["place"] = 3
        invalids = [
            {"name": "nav_to_obj", "arguments": {"target_obj": "nonexistent"}},
            {"name": "pick", "arguments": {"target_obj": "nonexistent"}},
            {"name": "place", "arguments": {
                "target_obj": "any_targets|0", "target_location": "missing_goal"}},
        ]
        history = deepcopy(self.agent.llm_model.chat_history)
        for invalid in invalids:
            with self.subTest(invalid=invalid):
                commands = [
                    {"type": "action", "agent": "agent_0", "name": "pick",
                     "arguments": {"target_obj": "any_targets|0"}},
                    {"type": "action", "agent": "agent_0", **invalid},
                ]
                with self.assertRaisesRegex(ValueError, "entity|target"):
                    self.policy.apply_recovery_commands(commands)
                self.assertFalse(child._cur_call_high_level[0])
                self.assertEqual(self.agent._recovery_actions, [])
                self.assertEqual(self.agent._recovery_feedback, [])
                self.assertEqual(self.agent.llm_model.chat_history, history)

    def test_target_validation_uses_concrete_skill_domain_and_ordered_entity_cache(self):
        child = self.policy._active_policies[0]
        domain = EntityDomain(["any_targets|0", "TARGET_any_targets|0"])
        target = domain.get_entity("any_targets|0")
        child._skill_redirects = {1: 4}
        child._skills = {4: SimpleNamespace(
            _pddl_problem=domain, _all_entities=[target],
            on_enter=lambda *args: self.fail("Validation must not enter a controller"),
        )}
        command = {"type": "action", "agent": "agent_0", "name": "pick",
                   "arguments": {"target_obj": "any_targets|0"}}
        self.assertIsNone(self.policy.validate_recovery_commands([command]))
        for name in ("cup", "TARGET_any_targets|0"):
            with self.subTest(name=name):
                command["arguments"]["target_obj"] = name
                with self.assertRaisesRegex(ValueError, "entity|target"):
                    self.policy.validate_recovery_commands([command])
        self.assertFalse(child._cur_call_high_level[0])

    def test_entity_actions_fail_closed_when_no_live_entity_domain_is_available(self):
        del self.policy._active_policies[0]._pddl
        with self.assertRaisesRegex(ValueError, "entity|PDDL|domain"):
            self.policy.validate_recovery_commands([{
                "type": "action", "agent": "agent_0", "name": "pick",
                "arguments": {"target_obj": "any_targets|0"},
            }])

    def test_resume_and_replan_force_new_selection_and_preserve_live_model_and_history(self):
        model, history = self.agent.llm_model, self.agent.llm_model.chat_history
        commands = [
            {"type": "resume", "agent": "agent_0", "reason": "previous skill stalled"},
            {"type": "replan", "agent": "agent_1", "instruction": "Navigate to the cup before picking."},
        ]
        self.assertIsNone(self.policy.validate_recovery_commands(commands))
        self.assertFalse(self.policy._active_policies[0]._cur_call_high_level[0])
        result = self.policy.apply_recovery_commands(commands)
        self.assertEqual(result, {"applied": commands, "simulator_reset": False})
        for child in self.policy._active_policies:
            self.assertTrue(child._cur_call_high_level[0])
        self.assertIs(self.agent.llm_model, model)
        self.assertIs(self.agent.llm_model.chat_history, history)
        self.assertTrue(self.agent.initialized)
        self.agent.chat("current scene")
        self.assertIn("previous skill stalled", model.requests[-1])
        other = self.policy._active_policies[1]._high_level_policy.llm_agent
        other.chat("current scene")
        self.assertIn("Navigate to the cup before picking.", other.llm_model.requests[-1])
        self.agent.chat("later scene")
        self.assertEqual(model.requests[-1], "later scene")

    def test_queued_action_uses_normal_argument_injection_and_events_without_model_request(self):
        output_dir = self.stack.enter_context(tempfile.TemporaryDirectory())
        trace = self.stack.enter_context(trajectory.EvaluationTrace(output_dir))
        trace.start_episode(0, SimpleNamespace(scene_id="scene", episode_id="1"))
        history = deepcopy(self.agent.llm_model.chat_history)
        commands = [{"type": "action", "agent": "agent_0", "name": "pick",
                     "arguments": {"target_obj": "any_targets|0"}}]
        self.policy.apply_recovery_commands(commands)
        commands[0]["arguments"]["target_obj"] = "changed after apply"
        with trace.policy_step():
            output = self.agent.chat("current scene")
        self.assertEqual(output, {"name": "pick", "arguments": {
            "target_obj": "any_targets|0", "robot": "agent_0"}})
        self.assertEqual(self.agent.llm_model.requests, [])
        self.assertEqual(self.agent.llm_model.chat_history[:len(history)], history)
        events = [json.loads(line) for path in trace.run_dir.rglob("*.jsonl")
                  for line in path.read_text().splitlines()]
        self.assertIn("agent_tool_decision", [event["event"] for event in events])
        self.assertIn("agent_action", [event["event"] for event in events])
        self.assertNotIn("llm_request", [event["event"] for event in events])
        self.agent.chat("next normal scene")
        self.assertEqual(len(self.agent.llm_model.requests), 1)

    def test_apply_result_cannot_change_a_validated_queued_action(self):
        result = self.policy.apply_recovery_commands([{
            "type": "action", "agent": "agent_0", "name": "pick",
            "arguments": {"target_obj": "any_targets|0"},
        }])
        result["applied"][0]["name"] = "unvalidated_tool"
        result["applied"][0]["arguments"]["target_obj"] = "changed target"
        self.assertEqual(self.agent.chat("current scene"), {
            "name": "pick", "arguments": {
                "target_obj": "any_targets|0", "robot": "agent_0"},
        })

    def test_actual_planning_response_is_retained(self):
        class PlanningModel:
            def __init__(self, *args, **kwargs):
                self.chat_history = []

            def chat(self, prompt, crab_planning=False):
                self.chat_history.append([{"role": "user", "content": prompt}])
                return "Go to cup, pick it, carry it to goal."

        with patch("habitat_mas.agents.crab_agent.OpenAIModel", PlanningModel):
            self.agent.init_agent("robot", "carry cup", "deliver cup")
        self.assertEqual(self.agent.planner_plan, "Go to cup, pick it, carry it to goal.")

    def test_recovery_action_enters_new_real_skill_before_old_controller_can_act(self):
        from test_execution_trajectory import ActionData, Observations, TestSkill
        from test_execution_trajectory import make_policy as make_hierarchy

        class HighLevelBase(torch.nn.Module):
            num_recurrent_layers = 0

            def get_termination(self, *args, **kwargs):
                return torch.zeros(1, dtype=torch.bool)

        source = ROOT / "habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py"
        body = [ast.ImportFrom(module="__future__",
                              names=[ast.alias(name="annotations")], level=0)]
        body += [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == "LLMHighLevelPolicy"]
        namespace = {"torch": torch, "HighLevelPolicy": HighLevelBase,
                     "PolicyActionData": ActionData,
                     "record_event": trajectory.record_event}
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                     str(source), "exec"), namespace)
        high_level = namespace["LLMHighLevelPolicy"].__new__(namespace["LLMHighLevelPolicy"])
        torch.nn.Module.__init__(high_level)
        high_level.llm_agent = self.agent
        high_level._skill_name_to_idx = {"nav_to_obj": 0, "pick": 1, "wait": 2}
        stale, replacement = TestSkill(max_steps=100), TestSkill(max_steps=100)
        child = make_hierarchy(stale)
        child._skills[1] = replacement
        child._idx_to_name = {0: "nav_to_obj", 1: "pick"}
        child._cur_skills[0] = 0
        child._cur_call_high_level[0] = False
        child._high_level_policy = high_level
        child._pddl = self.policy._active_policies[0]._pddl
        self.policy._active_policies[0] = child
        self.policy.apply_recovery_commands([{
            "type": "action", "agent": "agent_0", "name": "pick",
            "arguments": {"target_obj": "any_targets|0"},
        }])
        child.act(Observations(), torch.zeros(1, 1), torch.ones(1, 3),
                  torch.ones(1, 1, dtype=torch.bool),
                  envs_text_context=[{"scene_description": "current scene"}],
                  agent_arguments=[object()])
        self.assertEqual(child._cur_skills.tolist(), [1])
        self.assertEqual(stale._cur_skill_step.tolist(), [0.0])
        self.assertEqual(replacement._cur_skill_step.tolist(), [1.0])
        self.assertEqual(replacement._raw_skill_args[0], ["any_targets|0", "agent_0"])
        self.assertEqual(self.agent.llm_model.requests, [])

    def test_all_zero_previous_actions_do_not_reinitialize_a_resumed_episode(self):
        discussions = []

        def group_discussion(*args, **kwargs):
            discussions.append(kwargs)
            return {"agent_%d" % index: SimpleNamespace(
                robot_type="robot", task_description="task", subtask_description="part",
                chat_history=[],
            ) for index in range(2)}

        observations = {"pddl_text_goal": torch.tensor([[ord(c) for c in "task"]])}
        context = [{"robot_resume": "robots", "scene_description": "room", "episode_id": "1"}]
        initializations = []

        def initialize(agent, **kwargs):
            initializations.append(agent.name)
            agent.initialized = True

        with patch.dict(POLICY_GLOBALS, {"group_discussion": group_discussion}):
            with patch.object(CrabAgent, "init_agent", initialize):
                model, history = self.agent.llm_model, self.agent.llm_model.chat_history
                self.policy.apply_recovery_commands([{"type": "resume", "agent": "agent_0"}])
                self.policy.act(observations, torch.ones(1, 2), torch.zeros(1, 4),
                                torch.ones(1, 2, dtype=torch.bool),
                                envs_text_context=context, save_chat_history=False)
                self.assertEqual(discussions, [])
                self.assertEqual(initializations, [])
                self.assertIs(self.agent.llm_model, model)
                self.assertIs(self.agent.llm_model.chat_history, history)
                self.policy.act(observations, torch.ones(1, 2), torch.zeros(1, 4),
                                torch.zeros(1, 2, dtype=torch.bool),
                                envs_text_context=context, save_chat_history=False)
                self.assertEqual(len(discussions), 1)
                self.assertEqual(initializations, ["agent_0", "agent_1"])


if __name__ == "__main__":
    unittest.main()
