"""Check shared output configuration through Hydra and the real evaluator setup.

The environment and policy are doubles; no simulation or model calls are made.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gym import spaces

from habitat_baselines.config.default import get_config
from habitat_baselines.rl.multi_agent.habitat_mas_evaluator import HabitatMASEvaluator


class EvaluationReady(Exception):
    """Stop after evaluator setup, before policy/environment execution."""


class EvaluationOutputTest(unittest.TestCase):
    def config(self, directory, mode, enabled, *overrides):
        settings = [
            "habitat_baselines.num_environments=1",
            "habitat_baselines.test_episode_count=1",
            "habitat_baselines.eval.video_option=[]",
            f"habitat_baselines.trajectory.enabled={enabled}",
            f"habitat_baselines.output_dir={json.dumps(str(directory))}",
            'habitat_baselines.run_label="comparison"',
            *overrides,
        ]
        if mode != "clean":
            settings.append(f"habitat_baselines.fault_injection.structure.agent_{mode}.enabled=True")
        return get_config("multi_rearrange/llm_spot_drone_per.yaml", overrides=settings)

    def initialize(self, config):
        evaluator = HabitatMASEvaluator()
        envs = MagicMock(num_envs=1, number_of_episodes=[1])
        envs.reset.return_value = [{}]
        envs.post_step.side_effect = lambda observations: observations
        action_space = spaces.Box(low=-1, high=1, shape=(2,))
        agent = SimpleNamespace(
            actor_critic=SimpleNamespace(policy_action_space=action_space,
                                         hidden_state_shape=(1,), hidden_state_shape_lens=[1],
                                         policy_action_space_shape_lens=[2]),
            masks_shape=(1,), eval=MagicMock(side_effect=EvaluationReady),
        )
        with self.assertRaises(EvaluationReady):
            evaluator.evaluate_agent(agent, envs, config, 0, 0, MagicMock(), "cpu", [],
                                     SimpleNamespace(action_space=action_space), [])
        evaluator._crash_context["pbar"].close()
        envs.step.assert_not_called()
        return evaluator._crash_context

    def test_public_switch_and_paths_for_every_mode(self):
        for mode in ("clean", "dropout", "hang"):
            for enabled in (True, False):
                with self.subTest(mode=mode, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                    config = self.config(directory, mode, enabled)
                    context = self.initialize(config)
                    root = Path(directory)
                    stem = f"{mode}_comparison"
                    self.assertEqual(context["report_path"], root / f"{stem}_crash.json")
                    trajectory = context["trajectory_recorder"]
                    if enabled:
                        self.assertEqual(trajectory.path, root / f"{stem}_trajectory.jsonl")
                        self.assertTrue(trajectory.path.exists())
                        self.assertEqual(trajectory.mode, "clean" if mode == "clean" else "faulty")
                        trajectory.record_episode_start(episode_id="20", scene_id="scene", text_context={})
                        self.assertTrue((root / f"{stem}_trace-episode_20.txt").exists())
                    else:
                        self.assertIsNone(trajectory)
                        self.assertFalse(list(root.glob("*_trajectory.jsonl")))
                    events = context["fault_recorder"]
                    if mode == "clean":
                        self.assertIsNone(events)
                        self.assertFalse(list(root.glob("*_events.jsonl")))
                    else:
                        self.assertEqual(events.path, root / f"{stem}_events.jsonl")
                        self.assertTrue(events.path.exists())

    def test_replay_uses_public_recording_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    config = self.config(directory, "hang", enabled,
                                         'habitat_baselines.fault_injection.replay.path="clean.jsonl"')
                    with patch("habitat_baselines.rl.multi_agent.habitat_mas_evaluator.CleanPrefixReplay"):
                        if enabled:
                            self.initialize(config)
                        else:
                            with self.assertRaisesRegex(ValueError, "Replay requires"):
                                self.initialize(config)


if __name__ == "__main__":
    unittest.main()
