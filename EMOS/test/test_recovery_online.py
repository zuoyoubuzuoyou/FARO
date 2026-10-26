"""Exercise recovery at the live evaluator boundary without assets or API calls."""

from trajectory_test_utils import read_events as expanded_events

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
sys.path.insert(0, str(ROOT / "test"))

from habitat_mas.utils.trajectory import EvaluationTrace


class ApplyingController:
    """Replace external diagnosis/generation with a deterministic recovery."""

    def __init__(self, config, output_dir):
        self.frames = []

    def observe(self, frame, policy, terminal=False):
        self.frames.append(deepcopy(frame))
        policy.observed.append((deepcopy(frame), terminal))
        if not terminal and len(self.frames) == 1:
            policy.apply_recovery_commands([{"operation": "replan"}])
            return {"status": "applied"}
        return {"status": "terminal" if terminal else "normal"}


class BridgePolicy:
    def __init__(self):
        self.observed = []
        self.commands = []

    def apply_recovery_commands(self, commands):
        self.commands.extend(commands)


def event(kind, step=0, **values):
    return {"event_type": kind, "env_index": 0, "env_step": step,
            "evaluation_id": "attempt-a", **values}


class OnlineBridgeTests(unittest.TestCase):
    def bridge(self, output):
        from habitat_mas.recovery.config import RecoveryConfig
        from habitat_mas.recovery.online import OnlineRecovery

        return OnlineRecovery(RecoveryConfig(enabled=True, mode="extension", sample_every=1), output)

    def test_event_callback_defers_recovery_and_detaches_redacted_payloads(self):
        from habitat_mas.recovery import online

        with tempfile.TemporaryDirectory() as output, patch.object(
                online, "RecoveryController", ApplyingController):
            bridge = self.bridge(output)
            policy = BridgePolicy()
            context = {"pddl_text_goal": "Move objects to their targets", "api_key": "private"}
            with EvaluationTrace(output, event_callback=bridge.on_event) as trace:
                trace.start_episode(0, SimpleNamespace(scene_id="scene", episode_id="0"),
                                    context=context, state={"agents": {}})
                trace.record_step(0, [0], 0, False, {}, {"agents": {}})
                self.assertEqual(policy.commands, [])
                bridge.advance(policy, [False])
                self.assertEqual(policy.commands, [{"operation": "replan"}])
                self.assertEqual(policy.observed[0][0]["step"], 1)
            self.assertNotIn("private", json.dumps(policy.observed))
            self.assertEqual(context["api_key"], "private")

    def test_terminal_frame_is_audited_without_applying_recovery(self):
        from habitat_mas.recovery import online

        with tempfile.TemporaryDirectory() as output, patch.object(
                online, "RecoveryController", ApplyingController):
            bridge = self.bridge(output)
            policy = BridgePolicy()
            bridge.on_event(event("episode_start", context={}))
            bridge.on_event(event("step", 1, state={"agents": {}}, done=True))
            bridge.advance(policy, [True])
            self.assertEqual(policy.commands, [])
            self.assertEqual([terminal for _, terminal in policy.observed], [True])

    def test_new_episode_resets_detector_roles_and_active_actions(self):
        from habitat_mas.recovery import online

        with tempfile.TemporaryDirectory() as output, patch.object(
                online, "RecoveryController", ApplyingController):
            bridge = self.bridge(output)
            policy = BridgePolicy()
            bridge.on_event(event("episode_start", context={"assignments": {"agent_0": "old role"}}))
            bridge.on_event(event("action_start", 1, agent_id="agent_0", name="pick", action_id="old"))
            bridge.on_event(event("step", 1, state={"agents": {"agent_0": {}}}))
            bridge.advance(policy, [False])
            bridge.on_event(event("episode_start", context={}, evaluation_id="attempt-b"))
            bridge.on_event(event("step", 1, state={"agents": {"agent_0": {}}}, evaluation_id="attempt-b"))
            bridge.advance(policy, [False])
            self.assertEqual(len(policy.commands), 2)  # Fresh controller gets its first frame.
            self.assertEqual(policy.observed[1][0]["context"]["assignments"], {})
            self.assertIsNone(policy.observed[1][0]["agents"]["agent_0"]["action"]["name"])

    def test_rejects_multiple_environments_instead_of_mixing_frames(self):
        from habitat_mas.recovery import online

        with tempfile.TemporaryDirectory() as output:
            bridge = self.bridge(output)
            with self.assertRaisesRegex(ValueError, "single environment"):
                bridge.advance(BridgePolicy(), [False, False])

    def test_trace_callback_cannot_mutate_the_written_record(self):
        events = []
        def callback(payload):
            events.append(deepcopy(payload))
            payload["event_type"] = "changed"
            payload.get("context", {})["task_description"] = "changed"

        with tempfile.TemporaryDirectory() as output:
            with EvaluationTrace(output, event_callback=callback) as trace:
                trace.start_episode(0, SimpleNamespace(scene_id="a", episode_id="0"),
                                    context={"task_description": "actual", "api_key": "private"})
            records = expanded_events(output)
            self.assertEqual(records[0]["event_type"], "episode_start")
            self.assertEqual(records[0]["context"]["task_description"], "actual")
            self.assertEqual(events[0]["context"]["api_key"], "[REDACTED]")


class EvaluatorRecoveryTests(unittest.TestCase):
    def run_evaluation(self, output, *, enabled=True, fail_enable=False,
                       fail_state=False, fail_returned_state=False, episodes=1,
                       trace_enabled=False):
        import gym.spaces
        import numpy as np
        import torch
        from habitat_mas.recovery.config import RecoveryConfig
        from test_mas_trajectory_integration import FakeEnvs, FakePolicy, load_evaluator

        class LiveEnvs(FakeEnvs):
            def __init__(self, items):
                super().__init__(items, fail_enable=fail_enable,
                                 fail_state_capture=fail_state)
                self.live_step = 0
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1
                return super().reset()

            def step(self, actions):
                self.submitted_actions.append(actions)
                self.live_step += 1
                done = self.live_step % 2 == 0
                info = {"configured_success": float(done)}
                if self.recording:
                    info["__trajectory_state"] = {"agents": {"agent_0": {
                        "base_position": [self.live_step, 0, 0], "grasped_object_id": 37}}}
                    if fail_returned_state:
                        info["__trajectory_state"] = None
                        info["__trajectory_state_error"] = {"type": "RuntimeError", "message": "capture failed"}
                if done:
                    self.index += 1
                return [({"rgb": np.ones((2, 2, 3))}, 1.0, done, info)]

        class LivePolicy(FakePolicy, BridgePolicy):
            def __init__(self):
                FakePolicy.__init__(self)
                BridgePolicy.__init__(self)
                self.applied_before_act = []

            def act(self, *args, **kwargs):
                self.applied_before_act.append(len(self.commands))
                return super().act(*args, **kwargs)

        items = [SimpleNamespace(scene_id="scene", episode_id=str(i)) for i in range(episodes)]
        envs, policy = LiveEnvs(items), LivePolicy()
        config = SimpleNamespace(
            habitat_baselines=SimpleNamespace(
                num_environments=1, test_episode_count=-1,
                eval=SimpleNamespace(save_trajectories=trace_enabled,
                    recovery=RecoveryConfig(enabled=enabled, mode="extension", sample_every=1),
                    trajectory_dir=output, image_option=[], video_option=[], evals_per_ep=1)),
            habitat=SimpleNamespace(seed=42,
                simulator=SimpleNamespace(agents={"agent_0": {"articulated_agent_type": "FetchRobot"}}, agents_order=["agent_0"]),
                environment=SimpleNamespace(max_episode_steps=10, max_episode_seconds=60),
                task=SimpleNamespace(task_spec="pddl/pddl_dist_man", success_measure="configured_success",
                    measurements={"configured_success": {"type": "ExistingSuccess"}}, end_on_success=False)))
        load_evaluator()().evaluate_agent(
            SimpleNamespace(actor_critic=policy, masks_shape=(1,), eval=lambda: None),
            envs, config, 7, 12, SimpleNamespace(add_scalar=lambda *a: None),
            torch.device("cpu"), [], SimpleNamespace(action_space=gym.spaces.Discrete(1)), set())
        return envs, policy

    def test_live_recovery_affects_next_action_without_resetting_simulator(self):
        from habitat_mas.recovery import online

        with tempfile.TemporaryDirectory() as output, patch.object(
                online, "RecoveryController", ApplyingController):
            envs, policy = self.run_evaluation(output)
            self.assertEqual(policy.applied_before_act, [0, 1])
            self.assertEqual(envs.reset_count, 1)
            self.assertEqual(len(envs.submitted_actions), 2)
            self.assertEqual(policy.observed[0][0]["agents"]["agent_0"]["held_object"], 37)
            self.assertEqual([terminal for _, terminal in policy.observed], [False, True])
            self.assertFalse(envs.recording)

    def test_disabled_recovery_preserves_existing_unrecorded_execution(self):
        with tempfile.TemporaryDirectory() as output:
            envs, policy = self.run_evaluation(output, enabled=False)
            self.assertEqual(policy.applied_before_act, [0, 0])
            self.assertEqual(policy.observed, [])
            self.assertEqual(list(Path(output).iterdir()), [])
            self.assertFalse(envs.recording)

    def test_multiple_episodes_recover_independently(self):
        from habitat_mas.recovery import online

        with tempfile.TemporaryDirectory() as output, patch.object(
                online, "RecoveryController", ApplyingController):
            _, policy = self.run_evaluation(output, episodes=2)
            self.assertEqual(policy.applied_before_act, [0, 1, 1, 2])
            self.assertEqual([frame["step"] for frame, _ in policy.observed], [1, 2, 1, 2])

    def test_recovery_requires_successful_state_capture_setup(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, "recovery.*state capture"):
                self.run_evaluation(output, fail_enable=True)

    def test_recovery_requires_initial_state(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, "recovery.*state"):
                self.run_evaluation(output, fail_state=True)

    def test_recovery_does_not_continue_without_returned_state(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, "recovery.*state"):
                self.run_evaluation(output, fail_returned_state=True)

    def test_recovery_does_not_silently_lose_a_step_when_trace_write_fails(self):
        from habitat_mas.utils import trajectory

        with tempfile.TemporaryDirectory() as output, patch.object(
                trajectory._EpisodeWriter, "record_step", side_effect=OSError("storage failed")):
            with self.assertRaisesRegex(RuntimeError, "recovery.*observation"):
                self.run_evaluation(output)


if __name__ == "__main__":
    unittest.main()
