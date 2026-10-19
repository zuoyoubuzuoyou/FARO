"""Exercise the MAS evaluation loop without simulator assets or API calls."""

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import gym.spaces
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def load_evaluator():
    from habitat_baselines.rl.multi_agent.habitat_mas_evaluator import HabitatMASEvaluator

    return HabitatMASEvaluator


class FakeEnvs:
    num_envs = 1

    def __init__(self, episodes, fail_step=False, success=1.0, fail_cleanup=False,
                 fail_state_capture=False, fail_enable=False):
        self.episodes = episodes
        self.number_of_episodes = [len(episodes)]
        self.index = 0
        self.fail_step = fail_step
        self.recording = False
        self.success = success
        self.fail_cleanup = fail_cleanup
        self.submitted_actions = []
        self.fail_state_capture = fail_state_capture
        self.fail_enable = fail_enable

    def reset(self):
        return [{"rgb": np.zeros((2, 2, 3))}]

    def post_step(self, observations):
        return observations

    def current_episodes(self):
        return [self.episodes[self.index % len(self.episodes)]]

    def call_at(self, index, name, function_args=None):
        if name == "set_trajectory_recording":
            if self.fail_enable and function_args["enabled"]:
                raise RuntimeError("recording setup failed")
            if self.fail_cleanup and not function_args["enabled"]:
                raise RuntimeError("cleanup failed")
            self.recording = function_args["enabled"]
        elif name == "get_task_text_context":
            return {"scene_description": f"scene at reset {self.index}"}
        elif name == "get_trajectory_state":
            if self.fail_state_capture:
                raise RuntimeError("state snapshot failed")
            return {"agents": {"agent_0": {"base_position": [self.index, 0, 0]}}}
        else:
            raise AssertionError(name)

    def call(self, names, function_args_list=None):
        args = function_args_list or [None] * len(names)
        return [self.call_at(i, name, args[i]) for i, name in enumerate(names)]

    def step(self, actions):
        self.submitted_actions.append(actions)
        if self.fail_step:
            raise RuntimeError("simulation failed")
        info = {"configured_success": self.success}
        if self.recording:
            info["__trajectory_state"] = {
                "episode_id": self.current_episodes()[0].episode_id,
                "agents": {"agent_0": {"base_position": [9, 0, 0]}},
                "termination_reason": "success" if self.success else "task_inactive",
            }
        self.index += 1  # Model VectorEnv auto-reset before returning.
        return [({"rgb": np.ones((2, 2, 3))}, 2.5, True, info)]

    def pause_at(self, index):
        self.num_envs -= 1


class FakePolicy:
    policy_action_space = gym.spaces.Discrete(1)
    hidden_state_shape = (1,)
    hidden_state_shape_lens = [1]
    policy_action_space_shape_lens = [1]

    def __init__(self, overwrite_success=False):
        self.contexts = []
        self.overwrite_success = overwrite_success
        self.pause_calls = []

    def act(self, batch, hidden, prev, masks, **kwargs):
        self.contexts.append(dict(kwargs["envs_text_context"][0]))
        try:
            from habitat_mas.utils.trajectory import begin_action, decision_scope, record_event
            with decision_scope("agent_0"):
                record_event("decision", agent_name="agent_0", action="wait")
                begin_action("agent_0", "wait", ["500"])
        except ImportError:
            pass
        return SimpleNamespace(
            should_inserts=None,
            rnn_hidden_states=hidden,
            actions=torch.zeros((1, 1), dtype=torch.long),
            env_actions=torch.tensor([0]),
        )

    def get_extra(self, data, infos, dones):
        extra = {"skill": "wait"}
        if self.overwrite_success:
            extra["configured_success"] = 1.0
        return [extra]

    def on_envs_pause(self, indices):
        self.pause_calls.append(indices)


class MASTrajectoryIntegrationTests(unittest.TestCase):
    def run_evaluation(self, output, episodes, enabled=True, fail_step=False,
                       evals_per_ep=1, success=1.0, overwrite_success=False,
                       fail_cleanup=False, success_measure="configured_success",
                       fail_state_capture=False, fail_enable=False):
        config = SimpleNamespace(
            habitat_baselines=SimpleNamespace(
                num_environments=1, test_episode_count=-1,
                eval=SimpleNamespace(
                    save_trajectories=enabled, trajectory_dir=output,
                    image_option=[], video_option=[], evals_per_ep=evals_per_ep,
                ),
            ),
            habitat=SimpleNamespace(
                seed=42,
                simulator=SimpleNamespace(
                    agents={"agent_0": {"articulated_agent_type": "FetchRobot"}},
                    agents_order=["agent_0"],
                ),
                environment=SimpleNamespace(max_episode_steps=10, max_episode_seconds=60),
                task=SimpleNamespace(success_measure=success_measure,
                                     measurements={"configured_success": {"type": "ExistingSuccess"}},
                                     end_on_success=False),
            ),
        )
        envs = FakeEnvs(episodes, fail_step, success, fail_cleanup,
                        fail_state_capture, fail_enable)
        policy = FakePolicy(overwrite_success)
        agent = SimpleNamespace(actor_critic=policy, masks_shape=(1,),
                                eval=lambda: None)
        load_evaluator()().evaluate_agent(
            agent, envs, config, 7, 12,
            SimpleNamespace(add_scalar=lambda *a: None), torch.device("cpu"), [],
            SimpleNamespace(action_space=gym.spaces.Discrete(1)), set(),
        )
        return envs, policy

    def read_episodes(self, output):
        return [[json.loads(line) for line in path.read_text().splitlines()]
                for path in sorted(Path(output).rglob("*.jsonl"))]

    def test_records_terminal_state_before_auto_reset_and_scene_identity(self):
        with tempfile.TemporaryDirectory() as output:
            episodes = [SimpleNamespace(scene_id=scene, episode_id="0")
                        for scene in ("scene-a", "scene-b")]
            self.run_evaluation(output, episodes)
            traces = self.read_episodes(output)
            self.assertEqual(len(traces), 2)
            self.assertEqual({events[0]["scene_id"] for events in traces},
                             {"scene-a", "scene-b"})
            for events in traces:
                decision = next(e for e in events if e["event"] == "decision")
                step = next(e for e in events if e["event"] == "step")
                self.assertEqual(decision["step"], step["step"])
                self.assertEqual(step["state"]["agents"]["agent_0"]["base_position"],
                                 [9, 0, 0])
                self.assertEqual(step["reward"], 2.5)
                self.assertTrue(step["done"])
                self.assertEqual(step["metrics"], {"configured_success": 1.0, "skill": "wait"})
                self.assertEqual(events[-1]["event"], "episode_end")
                self.assertEqual(events[-1]["total_reward"], 2.5)
                self.assertEqual(events[-1]["status"], "completed")

    def test_repeated_episode_logging_refreshes_without_changing_policy_context(self):
        with tempfile.TemporaryDirectory() as output:
            _, policy = self.run_evaluation(
                output, [SimpleNamespace(scene_id="a", episode_id="0")],
                evals_per_ep=2)
            traces = self.read_episodes(output)
            self.assertEqual(len(traces), 2)
            self.assertEqual({events[0]["context"]["scene_description"]
                              for events in traces},
                             {"scene at reset 0", "scene at reset 1"})
            # The original evaluator only refreshes policy context on ID change.
            self.assertEqual([context["scene_description"] for context in policy.contexts],
                             ["scene at reset 0", "scene at reset 0"])

    def test_done_failure_keeps_original_success_even_if_policy_overwrites_metric(self):
        with tempfile.TemporaryDirectory() as output:
            self.run_evaluation(output, [SimpleNamespace(scene_id="a", episode_id="0")],
                                success=0.0, overwrite_success=True)
            trace = self.read_episodes(output)[0]
            step = next(event for event in trace if event["event"] == "step")
            self.assertEqual(step["metrics"]["configured_success"], 1.0)
            summary = json.loads(next(Path(output).rglob("summary.json")).read_text())
            self.assertEqual(summary["task_success"], 0.0)
            self.assertEqual(summary["termination_reason"], "task_inactive")

    def test_unconfigured_success_is_null_instead_of_inferred_from_done(self):
        with tempfile.TemporaryDirectory() as output:
            self.run_evaluation(output, [SimpleNamespace(scene_id="a", episode_id="0")],
                                success_measure=None)
            summary = json.loads(next(Path(output).rglob("summary.json")).read_text())
            self.assertIsNone(summary["task_success"])
            self.assertIn("task_success", summary["missing_fields"])

    def test_new_layout_config_submission_correlation_and_decision_media(self):
        with tempfile.TemporaryDirectory() as output:
            envs, _ = self.run_evaluation(
                output, [SimpleNamespace(scene_id="unsafe/episode", episode_id="0/../x")])
            run = next(Path(output).iterdir())
            metadata = json.loads((run / "config.json").read_text())["metadata"]
            self.assertEqual(metadata["agent_count"], 1)
            self.assertEqual(metadata["agents_order"], ["agent_0"])
            self.assertEqual(metadata["success_measure"], "configured_success")
            self.assertTrue(metadata["version"]["source_files"])
            episode_dir = next((run / "episodes").iterdir())
            self.assertEqual(episode_dir.parent, run / "episodes")
            for filename in ("episode.json", "events.jsonl", "summary.json"):
                self.assertTrue((episode_dir / filename).is_file())
            events = self.read_episodes(output)[0]
            decision = next(event for event in events if event["event"] == "decision")
            submissions = [event for event in events if event["event"] in ("env_action_submit", "action_submit")]
            step = next(event for event in events if event["event"] == "step")
            self.assertTrue(submissions)
            self.assertTrue(all(event["sequence"] < step["sequence"] for event in submissions))
            self.assertTrue(all(event["step"] == decision["step"] for event in submissions))
            submission = next(event for event in submissions if event["event"] == "action_submit")
            start = next(event for event in events if event["event"] == "action_start")
            self.assertEqual(submission["agent_id"], "agent_0")
            self.assertEqual(submission["action_id"], start["action_id"])
            self.assertEqual(submission["env_step"], 1)
            decision_media = next(event for event in events if event["event"] == "decision_media")
            self.assertEqual(decision_media["observation_env_step"], 0)
            self.assertTrue(decision_media["media"])
            for reference in decision_media["media"]:
                self.assertTrue((episode_dir / reference["media_path"]).is_file())
                self.assertFalse(reference["seen_by_llm"])
            self.assertEqual(envs.submitted_actions, [[0]])
            self.assertTrue(list((episode_dir / "media").glob("*.png")))

    def test_simulator_error_preserves_decision_and_error_summary(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                self.run_evaluation(output, [SimpleNamespace(scene_id="a", episode_id="0")],
                                    fail_step=True)
            traces = self.read_episodes(output)
            self.assertEqual(len(traces), 1)
            self.assertIn("decision", [event["event"] for event in traces[0]])
            self.assertIn("env_action_submit", [event["event"] for event in traces[0]])
            self.assertEqual(traces[0][-1]["status"], "error")

    def test_cleanup_error_does_not_replace_original_simulator_error(self):
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(RuntimeError, "simulation failed"):
                self.run_evaluation(output, [SimpleNamespace(scene_id="a", episode_id="0")],
                                    fail_step=True, fail_cleanup=True)

    def test_state_capture_and_setup_errors_do_not_change_task_results(self):
        episode = SimpleNamespace(scene_id="a", episode_id="0")
        for flags in ({"fail_state_capture": True}, {"fail_enable": True}):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as output:
                envs, _ = self.run_evaluation(output, [episode], **flags)
                self.assertEqual(envs.submitted_actions, [[0]])
                summary = json.loads(next(Path(output).rglob("summary.json")).read_text())
                self.assertEqual(summary["task_success"], 1.0)

    def test_trace_directory_error_does_not_prevent_original_task(self):
        with tempfile.TemporaryDirectory() as output:
            with patch("habitat_baselines.rl.multi_agent.habitat_mas_evaluator.EvaluationTrace",
                       side_effect=OSError("storage failed")):
                envs, _ = self.run_evaluation(
                    output, [SimpleNamespace(scene_id="a", episode_id="0")])
            self.assertEqual(envs.submitted_actions, [[0]])
            self.assertEqual(list(Path(output).iterdir()), [])

    def test_disabled_recording_creates_no_output(self):
        with tempfile.TemporaryDirectory() as output:
            self.run_evaluation(output, [SimpleNamespace(scene_id="a", episode_id="0")],
                                enabled=False)
            self.assertEqual(list(Path(output).iterdir()), [])

    def test_logging_keeps_policy_inputs_actions_and_original_pause_predicate(self):
        with tempfile.TemporaryDirectory() as enabled, tempfile.TemporaryDirectory() as disabled:
            episodes = [SimpleNamespace(scene_id="scene-a", episode_id="0"),
                        SimpleNamespace(scene_id="scene-b", episode_id="0")]
            logged_envs, logged_policy = self.run_evaluation(enabled, episodes)
            plain_envs, plain_policy = self.run_evaluation(disabled, episodes, enabled=False)
            self.assertEqual(logged_policy.contexts, plain_policy.contexts)
            self.assertEqual(logged_envs.submitted_actions, plain_envs.submitted_actions)
            self.assertEqual(logged_policy.pause_calls, plain_policy.pause_calls)
            self.assertEqual(logged_policy.pause_calls, [])


if __name__ == "__main__":
    unittest.main()
