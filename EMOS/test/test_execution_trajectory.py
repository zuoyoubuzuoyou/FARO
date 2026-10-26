"""Exercise the real skill lifecycle without scene assets or model calls."""

from trajectory_test_utils import read_events as expanded_events

import ast
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys
import tempfile
import traceback
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
from habitat_mas.utils import trajectory


class ActionData(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**{"rnn_hidden_states": None, "policy_info": None,
                            "values": None, "action_log_probs": None, **kwargs})


def load_classes():
    # Keep production methods and tensors; only imports of heavyweight Habitat
    # infrastructure are replaced. Simulator-free tests can run on a CPU.
    namespace = {
        "torch": torch, "nn": torch.nn, "np": np,
        "defaultdict": defaultdict, "Policy": type("Policy", (), {}),
        "PolicyActionData": ActionData,
        "baseline_registry": SimpleNamespace(register_policy=lambda cls: cls),
        "get_num_actions": lambda action_space: action_space,
        "logging": logging, "baselines_logger": logging.getLogger(__name__),
        "traceback": traceback,
        "NoopSkillPolicy": type("NoopSkillPolicy", (), {}),
    }
    for name in ("record_event", "begin_action", "end_action", "current_action_id"):
        if hasattr(trajectory, name):
            namespace[name] = getattr(trajectory, name)
    for relative, class_name in (
        ("habitat-baselines/habitat_baselines/rl/hrl/skills/skill.py", "SkillPolicy"),
        ("habitat-baselines/habitat_baselines/rl/hrl/hierarchical_policy.py", "HierarchicalPolicy"),
    ):
        source = ROOT / relative
        tree = ast.parse(source.read_text())
        body = [ast.ImportFrom(module="__future__",
                              names=[ast.alias(name="annotations")], level=0)]
        body += [node for node in tree.body
                 if (isinstance(node, ast.ClassDef) and node.name == class_name)
                 or isinstance(node, ast.FunctionDef)]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                     str(source), "exec"), namespace)
    return namespace["SkillPolicy"], namespace["HierarchicalPolicy"]


SkillPolicy, HierarchicalPolicy = load_classes()


class Observations:
    def slice_keys(self, *keys):
        return self

    def __getitem__(self, indices):
        return self


class TestSkill(SkillPolicy):
    def __init__(self, *, max_steps=3, force_timeout=False, done_step=None,
                 fail_phase=None):
        self._cur_skill_step = torch.zeros(1)
        self._max_skill_steps = max_steps
        self._force_end_on_timeout = force_timeout
        self._should_keep_hold_state = False
        self._apply_postconds = False
        self._cur_skill_args = [None]
        self._raw_skill_args = [None]
        self.done_step = done_step
        self.done_checks = 0
        self.fail_phase = fail_phase
        self.error = RuntimeError("original skill failure")

    def _parse_skill_arg(self, skill_name, skill_arg):
        if self.fail_phase == "on_enter":
            raise self.error
        return skill_arg

    def _internal_act(self, *args):
        if self.fail_phase == "act":
            raise self.error
        return ActionData(actions=torch.tensor([[0.25, -0.5, 1.0]]))

    def _is_skill_done(self, *args):
        self.done_checks += 1
        if self.fail_phase == "should_terminate":
            raise self.error
        done = self.done_step is not None and self._cur_skill_step[0] >= self.done_step
        return torch.tensor([bool(done)])


def make_policy(skill, interrupt_step=None):
    policy = HierarchicalPolicy.__new__(HierarchicalPolicy)
    torch.nn.Module.__init__(policy)
    policy._trajectory_agent_id = "agent_0"
    skill._trajectory_agent_id = "agent_0"
    policy._skills = {0: skill}
    policy._skill_redirects = {}
    policy._idx_to_name = {0: "navigate"}
    policy._cur_skills = np.array([-1], dtype=np.int32)
    policy._cur_call_high_level = torch.ones(1, dtype=torch.bool)
    policy._max_skill_rnn_layers = 0
    policy._action_space = 3
    policy._stop_action_idx = 2
    policy._high_level_policy = SimpleNamespace(
        num_recurrent_layers=0,
        apply_mask=lambda masks: None,
        get_next_skill=lambda *args, **kwargs: (
            torch.zeros(1), [["goal"]], torch.zeros(1, dtype=torch.bool), ActionData()
        ),
        get_termination=lambda *args: torch.tensor([
            interrupt_step is not None and skill._cur_skill_step[0] >= interrupt_step
        ]),
    )
    return policy


class ExecutionTrajectoryTests(unittest.TestCase):
    def run_steps(self, skill, count, interrupt_step=None):
        policy = make_policy(skill, interrupt_step)
        results = []
        with tempfile.TemporaryDirectory() as output:
            with trajectory.EvaluationTrace(output) as trace:
                trace.start_episode(0, SimpleNamespace(scene_id="scene", episode_id="7"))
                for step in range(count):
                    with trace.policy_step():
                        result = policy.act(Observations(), torch.zeros(1, 1),
                                            torch.zeros(1, 3), torch.ones(1, 1, dtype=torch.bool))
                    trace.submit_actions(0, result.take_actions,
                                         agent_ids=["agent_0"])
                    results.append(result.take_actions.tolist())
                    trace.record_step(0, result.take_actions, 0.0, False, {},
                                      {"position": [step + 1, 0, 0]})
            events = expanded_events(output)
        return results, events

    def test_one_action_spans_steps_and_timeout_closes_after_final_state(self):
        results, events = self.run_steps(TestSkill(), 3)
        starts = [event for event in events if event["event"] == "action_start"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0]["action_name"], "navigate")
        steps = [event for event in events if event["event"] == "step"]
        ends = [event for event in events if event["event"] == "action_end"]
        self.assertEqual(len(ends), 1)
        self.assertEqual(ends[0]["action_id"], starts[0]["action_id"])
        self.assertEqual(ends[0]["status"], "timeout")
        self.assertEqual(ends[0]["reason"], "max_skill_steps")
        self.assertLess(steps[-1]["sequence"], ends[0]["sequence"])
        self.assertEqual(results, [[[0.25, -0.5, 0.0]]] * 3)
        executions = [event for event in events if event["event"] == "skill_action"]
        self.assertEqual(len(executions), 3)
        self.assertEqual({event["action_id"] for event in executions}, {starts[0]["action_id"]})
        self.assertEqual(executions[0]["actions"], [0.25, -0.5, 1.0])
        returns = [event for event in events if event["event"] == "action_execution"]
        self.assertEqual(len(returns), 3)
        self.assertEqual({event["action_id"] for event in returns}, {starts[0]["action_id"]})

    def test_raw_done_signal_does_not_claim_physical_success_or_repeat_done_check(self):
        skill = TestSkill(max_steps=10, done_step=2)
        results, events = self.run_steps(skill, 2)
        self.assertEqual(skill.done_checks, 2)
        end = next(event for event in events if event["event"] == "action_end")
        self.assertEqual(end["status"], "controller_terminated")
        self.assertEqual(end["reason"], "skill_done_signal")
        signals = [event for event in events if event["event"] == "skill_termination_signal"]
        self.assertEqual([event["is_skill_done"] for event in signals], [False, True])
        self.assertEqual(signals[-1]["cur_skill_step"], 2)
        self.assertEqual(signals[-1]["max_skill_steps"], 10)
        self.assertFalse(signals[-1]["force_end_on_timeout"])

    def test_force_timeout_preserves_stop_action_and_logs_raw_signal(self):
        results, events = self.run_steps(TestSkill(max_steps=1, force_timeout=True), 1)
        self.assertEqual(results, [[[0.25, -0.5, 1.0]]])
        signal = next(event for event in events if event["event"] == "skill_termination_signal")
        self.assertFalse(signal["is_skill_done"])
        self.assertTrue(signal["over_max_len"])
        self.assertTrue(signal["bad_terminate"])
        self.assertTrue(signal["force_end_on_timeout"])
        self.assertEqual(next(e for e in events if e["event"] == "action_end")["status"], "timeout")

    def test_high_level_interrupt_is_logged_without_changing_actions(self):
        results, events = self.run_steps(TestSkill(max_steps=10), 2, interrupt_step=2)
        self.assertEqual(results, [[[0.25, -0.5, 0.0]]] * 2)
        end = next(event for event in events if event["event"] == "action_end")
        self.assertEqual(end["reason"], "high_level_interrupt")

    def test_disabled_trace_preserves_controller_outputs_and_done_checks(self):
        skill = TestSkill()
        policy = make_policy(skill)
        results = [policy.act(Observations(), torch.zeros(1, 1), torch.zeros(1, 3),
                              torch.ones(1, 1, dtype=torch.bool)).take_actions.tolist()
                   for _ in range(3)]
        self.assertEqual(results, [[[0.25, -0.5, 0.0]]] * 3)
        self.assertEqual(skill.done_checks, 3)

    def test_skill_errors_are_logged_and_original_exception_is_raised(self):
        for phase in ("on_enter", "act", "should_terminate"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as output:
                skill = TestSkill(fail_phase=phase)
                with self.assertRaises(RuntimeError) as caught:
                    with trajectory.EvaluationTrace(output) as trace:
                        trace.start_episode(0, SimpleNamespace(scene_id="s", episode_id="1"))
                        with trace.policy_step():
                            make_policy(skill).act(Observations(), torch.zeros(1, 1),
                                                   torch.zeros(1, 3), torch.ones(1, 1, dtype=torch.bool))
                self.assertIs(caught.exception, skill.error)
                events = expanded_events(output)
                errors = [event for event in events if event["event"] == "execution_error"]
                self.assertEqual(len(errors), 1)
                error = errors[0]
                self.assertEqual(error["stage"], phase)
                self.assertEqual(error["error"]["message"], "original skill failure")
                self.assertEqual(events[-1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
