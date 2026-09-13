"""Replay validation and real policy boundary methods without native Habitat."""

import ast
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from habitat_mas.agents.message_bus import AgentMessageBus
from habitat_mas.fault_injection.cli import prepare_run
from habitat_mas.fault_injection.replay import (
    CleanPrefixReplay, ReplayMismatchError, compare_recorded,
)
from habitat_mas.fault_injection.trajectory import AgentDropoutTrajectoryRecorder
from habitat_mas.test import test_fault_launcher, test_agent_hang_policy

ROOT = Path(__file__).resolve().parents[3]


def rows_for_episode(length=4):
    rows = []
    for step in range(length):
        decision = {"agent": "agent_0", "action": "wait", "arguments": ["500"]}
        rows.append({
            "type": "simulator_step", "mode": "clean", "scene_id": "scene_a", "episode_id": "2",
            "simulator_step": step, "done": step == length - 1,
            "joint_env_action": [0.0, 1.0], "reward": -0.0005,
            "policy_info": {"agent_0_high_level_decision": decision} if step == 0 else {},
            "simulator_state": {
                "timing": "post_env_step", "objects": [{"object_id": 1, "position": [1., 2., 3.]}],
                "agents": {"agent_0": {"position": [0., 0., float(step)], "rotation": 0., "held_object": None}},
            },
            "metrics": {"num_steps": float(step + 1), "pddl_success": float(step == length - 1)},
        })
    return rows


def load_methods(relative_path, class_name, methods, namespace):
    """Execute the production methods against small dependency doubles."""
    path = ROOT / relative_path
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    tree = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace[class_name]


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "clean.jsonl"
        self.rows = rows_for_episode()
        self.write()

    def write(self):
        self.path.write_text("\n".join(json.dumps(r) for r in self.rows))

    def start(self):
        replay = CleanPrefixReplay(self.path)
        replay.start_episode("scene_a", "2", 2, ["agent_0"])
        return replay

    def test_verified_prefix_releases_live_branch_and_stops_reading_clean(self):
        replay = self.start()
        self.assertEqual(replay.context_fidelity, "decision_history_only")
        for step in (0, 1):
            row = replay.prefix_row(step)
            replay.verify_action(row, row["joint_env_action"], row["policy_info"])
            replay.verify_result(row, row["simulator_state"], row["reward"], row["done"], row["metrics"])
        self.assertIsNone(replay.prefix_row(2))
        self.assertIsNone(replay.prefix_row(1000))  # Never force the clean suffix.
        self.assertEqual(replay.report("replay_prefix_verified")["verified_steps"], 2)
        replay.start_episode("scene_a", "2", 2, ["agent_0"])
        self.assertEqual(replay.verified_steps, 0)

    def test_no_branch_without_verification(self):
        replay = self.start()
        with self.assertRaisesRegex(ReplayMismatchError, "complete prefix"):
            replay.prefix_row(2)
        with self.assertRaisesRegex(ReplayMismatchError, "Expected replay step"):
            replay.prefix_row(1)

    def test_failure_report_is_written_before_exception_propagates(self):
        replay = self.start()
        recorder = MagicMock()
        with self.assertRaisesRegex(ReplayMismatchError, "unexpected decision"):
            with replay.record_failures(recorder):
                raise ReplayMismatchError("unexpected decision")
        event = recorder.record.call_args.args[0]
        self.assertEqual(event["type"], "replay_failed")
        self.assertEqual(event["verified_steps"], 0)
        self.assertEqual(event["error"], "unexpected decision")

    def test_context_recorded_once_without_expanding_continuing_actions(self):
        recorder = AgentDropoutTrajectoryRecorder(self.temp.name, "recorded", "clean", ["agent_0"])
        recorder.record_episode_start(episode_id="2", scene_id="scene_a", text_context={})
        context = {"chat_history": [[{"role": "user", "content": "hello"}]]}
        info = copy.deepcopy(self.rows[0]["policy_info"])
        info["agent_0_high_level_decision"]["replay_context"] = context
        for step in (0, 1):
            row = self.rows[step]
            recorder.record_step(episode_id="2", scene_id="scene_a", simulator_step=step,
                joint_env_action=row["joint_env_action"], policy_info=info if step == 0 else {},
                reward=row["reward"], done=False, metrics=row["metrics"], simulator_state=row["simulator_state"])
        recorded = [json.loads(line) for line in recorder.path.read_text().splitlines()][-2:]
        self.assertEqual(recorded[0]["policy_info"]["agent_0_high_level_decision"]["replay_context"], context)
        for row in recorded:
            self.assertNotIn("replay_context", row["agent_step_actions"]["agent_0"]["action"])
        self.assertNotIn("chat_history", recorder.text_path_for_episode("2").read_text())

    def test_action_and_decision_timing_mismatches(self):
        replay = self.start()
        row = self.rows[0]
        with self.assertRaisesRegex(ReplayMismatchError, "joint_env_action"):
            replay.verify_action(row, [1., 1.], row["policy_info"])
        with self.assertRaisesRegex(ReplayMismatchError, "timing"):
            replay.verify_action(row, row["joint_env_action"], {})
        with self.assertRaisesRegex(ReplayMismatchError, "timing"):
            replay.verify_action(self.rows[1], [0., 1.], row["policy_info"])
        info = copy.deepcopy(row["policy_info"])
        info["agent_0_high_level_decision"]["arguments"] = ["1"]
        with self.assertRaisesRegex(ReplayMismatchError, "arguments"):
            replay.verify_action(row, row["joint_env_action"], info)

    def test_physical_reward_done_and_stage_history_mismatches(self):
        row = self.rows[0]
        for field in ("position", "rotation", "held_object", "object", "reward", "done", "metric"):
            with self.subTest(field=field):
                replay = self.start()
                state = copy.deepcopy(row["simulator_state"])
                reward, done, metrics = row["reward"], False, dict(row["metrics"])
                if field == "position": state["agents"]["agent_0"][field][0] += .1
                if field == "rotation": state["agents"]["agent_0"][field] += .1
                if field == "held_object": state["agents"]["agent_0"][field] = {"object_id": 1}
                if field == "object": state["objects"][0]["position"][1] += .1
                if field == "reward": reward = 1.
                if field == "done": done = True
                if field == "metric": metrics["pddl_success"] = 1.
                with self.assertRaises(ReplayMismatchError):
                    replay.verify_result(row, state, reward, done, metrics)
                self.assertEqual(replay.verified_steps, 0)

    def test_tolerance_and_nonfinite_values(self):
        compare_recorded([1.], [1.000001])
        for actual in (float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ReplayMismatchError):
                compare_recorded(1., actual)
        for atol in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError): CleanPrefixReplay(self.path, atol=atol)

    def test_invalid_sources_and_boundaries(self):
        replay = self.start()
        for branch in (0, 4, -1):
            with self.assertRaises(ValueError): replay.validate_episode("scene_a", "2", branch)
        with self.assertRaises(ValueError): replay.validate_episode("wrong_scene", "2", 2)
        with self.assertRaises(ValueError): replay.start_episode("scene_a", "2", 2, ["agent_1"])
        original = copy.deepcopy(self.rows)
        for kind in ("faulty", "gap", "duplicate", "incomplete", "missing_action", "missing_state", "missing_decision"):
            with self.subTest(kind=kind):
                self.rows = copy.deepcopy(original)
                if kind == "faulty": self.rows[0]["mode"] = "faulty"
                if kind == "gap": self.rows[0]["simulator_step"] = 1
                if kind == "duplicate": self.rows.append(self.rows[0])
                if kind == "incomplete": self.rows[-1]["done"] = False
                if kind == "missing_action": self.rows[0].pop("joint_env_action")
                if kind == "missing_state": self.rows[0].pop("simulator_state")
                if kind == "missing_decision": self.rows[0]["policy_info"] = {}
                self.write()
                with self.assertRaises(ValueError): self.start()

    def test_recorded_context_fidelity_and_no_aliasing(self):
        self.rows[0]["policy_info"]["agent_0_high_level_decision"]["replay_context"] = {"chat_history": []}
        self.write()
        replay = self.start()
        self.assertEqual(replay.context_fidelity, "recorded")
        decision = replay.initial_decision("agent_0")
        decision["arguments"].append("changed")
        self.assertEqual(replay.initial_decision("agent_0")["arguments"], ["500"])

    def test_cli_replay_plan_and_rejected_options(self):
        fixture = test_fault_launcher.FaultLauncherTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        args = fixture.args("--episode", "2", "--step", "2", "--replay-trajectory", str(self.path))
        _, command, plan = prepare_run(args, [])
        self.assertIn(f'habitat_baselines.fault_injection.replay.path="{self.path}"', command)
        self.assertEqual(plan["replay"]["atol"], 1e-5)
        for options in (("--step", "0"), ("--step", "4"), ("--step", "2", "--no-trajectory"),
                        ("--trajectory", str(self.path)), ("--step", "2", "--replay-atol", "nan")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                prepare_run(fixture.args("--episode", "2", "--replay-trajectory", str(self.path), *options), [])


class ReplayPolicyBoundaryTest(unittest.TestCase):
    def test_dispatch_keeps_policies_running_then_only_drops_target_at_boundary(self):
        fixture = test_agent_hang_policy.AgentHangPolicyTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        initial = {"action": "wait", "arguments": ["500"]}
        replay = SimpleNamespace(initial_decision=lambda name: initial)
        fixture.namespace["json"] = json
        fixture.namespace["AgentArguments"] = SimpleNamespace
        for agent in fixture.agents:
            agent._high_level_policy.llm_agent.init_agent = MagicMock(
                side_effect=lambda agent=agent, **kw: setattr(agent._high_level_policy.llm_agent, "initialized", True)
            )
        def dispatch(step):
            masks = test_agent_hang_policy.tensor(step != 0, shape=(1, 2))
            masks.split.return_value = [test_agent_hang_policy.tensor(step != 0) for _ in range(2)]
            with contextlib.redirect_stdout(io.StringIO()):
                return fixture.policy.act(
                    {"pddl_text_goal": [SimpleNamespace(tolist=lambda: [ord("x")])]},
                    fixture.hidden, fixture.previous, masks,
                    envs_text_context=[{"episode_id": "15", "robot_resume": json.dumps({
                        "agent_0": {"robot_type": "Spot"}, "agent_1": {"robot_type": "Drone"}}),
                        "scene_description": "scene"}],
                    save_chat_history=False, simulator_steps=[step], trajectory_replay=replay,
                    replay_row={"simulator_step": step} if step < 2 else None,
                    agent_dropout_fault={"enabled": True, "agent": "agent_1", "simulator_step": 2},
                )
        dispatch(0)
        dispatch(1)
        fixture.namespace["group_discussion"].assert_not_called()
        for agent in fixture.agents:
            self.assertTrue(agent._high_level_policy.llm_agent.init_agent.call_args.kwargs["skip_planning"])
            self.assertEqual(agent.act.call_count, 2)
            self.assertEqual(agent.act.call_args.kwargs["replay_row"], {"simulator_step": 1})
        actions = dispatch(2)
        self.assertEqual(fixture.agents[0].act.call_count, 3)
        self.assertIsNone(fixture.agents[0].act.call_args.kwargs["replay_row"])
        self.assertEqual(fixture.agents[1].act.call_count, 2)
        self.assertIn("dropout_event", actions[1].policy_info[0])
        self.assertIs(actions[1].rnn_hidden_states, fixture.hidden_parts[1])

    def test_high_level_replay_bypasses_chat_and_live_branch_calls_chat(self):
        namespace = {
            "copy": copy,
            "torch": SimpleNamespace(zeros=lambda size, **kw: [0] * size, bool=bool),
            "Tuple": tuple, "List": list, "Any": object, "PolicyActionData": SimpleNamespace,
            "ReplayMismatchError": ReplayMismatchError,
        }
        # Annotations reference torch.Tensor/BoolTensor at class creation.
        namespace["torch"].Tensor = list
        namespace["torch"].BoolTensor = list
        cls = load_methods("habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py",
                           "LLMHighLevelPolicy", {"get_next_skill"}, namespace)
        policy = cls()
        policy._skill_name_to_idx = {"wait": 0, "nav_to_obj": 1}
        policy._parse_function_call_args = lambda name, args: args
        policy.llm_agent = SimpleNamespace(
            initialized=True, start_act=False, name="agent_0", replay_decision=MagicMock(),
            chat=MagicMock(return_value={"name": "nav_to_obj", "arguments": {"target_obj": "any_targets|0"}}),
            get_token_usage=lambda: 0,
        )
        def call(row):
            with contextlib.redirect_stdout(io.StringIO()):
                return policy.get_next_skill({}, None, None, SimpleNamespace(shape=(1,)), [1.], False, {},
                    envs_text_context=[{"scene_description": "scene"}], agent_arguments=[], replay_row=row)
        row = rows_for_episode()[0]
        output = call(row)
        self.assertEqual(output[1], [["500"]])
        policy.llm_agent.chat.assert_not_called()
        policy.llm_agent.replay_decision.assert_called_once()
        with self.assertRaises(ReplayMismatchError): call(rows_for_episode()[1])
        output = call(None)
        policy.llm_agent.chat.assert_called_once()
        self.assertEqual(output[0], [1])

    def test_memory_roundtrip_preserves_messages_and_does_not_alias(self):
        namespace = {"copy": copy, "json": json, "AgentMessageBus": AgentMessageBus}
        cls = load_methods("habitat-mas/habitat_mas/agents/crab_agent.py", "CrabAgent",
                           {"export_replay_context", "restore_replay_context", "replay_decision"}, namespace)
        AgentMessageBus.reset()
        self.addCleanup(AgentMessageBus.reset)
        agent = cls()
        agent.robot_type, agent.task_description, agent.subtask_description = "Spot", "task", "subtask"
        agent.start_act = True
        agent.llm_model = SimpleNamespace(system_message={"role": "system", "content": "prompt"},
            chat_history=[[{"role": "user", "content": "hello"}]], token_usage=42)
        AgentMessageBus.members = {"agent_0", "agent_1"}
        AgentMessageBus.send_request("agent_0", "agent_1", "help")
        snapshot = agent.export_replay_context()
        AgentMessageBus.reset()
        agent.llm_model.chat_history = []
        agent.replay_decision({"replay_context": snapshot})
        self.assertEqual(agent.llm_model.chat_history, snapshot["chat_history"])
        self.assertEqual(AgentMessageBus.members, {"agent_0", "agent_1"})
        self.assertIn("help", AgentMessageBus.messages["agent_1"][0])
        agent.llm_model.chat_history[0][0]["content"] = "changed"
        self.assertEqual(snapshot["chat_history"][0][0]["content"], "hello")
        agent.replay_decision({"action": "wait", "arguments": ["500"]})
        self.assertIn("reconstructed", agent.llm_model.chat_history[-1][0]["content"])


if __name__ == "__main__":
    unittest.main()
