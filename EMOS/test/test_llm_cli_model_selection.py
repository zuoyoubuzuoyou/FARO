"""Hydra model overrides reach real API request arguments without transport."""
from copy import deepcopy
import contextlib
import io
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from habitat_baselines.config.default import get_config
from habitat_baselines import run
from habitat_baselines.rl.multi_agent.multi_llm_policy import MultiLLMPolicy
from hydra.core.hydra_config import HydraConfig
from habitat_mas.agents.crab_agent import CrabAgent
from habitat_mas.utils import models
from habitat_mas.utils.trajectory import EvaluationTrace
import torch
from test_llm_trajectory import Transport, completion, tool
from test_recovery_policy import make_policy


class LLMCLIModelSelectionTests(unittest.TestCase):
    def config(self, *overrides):
        return get_config("multi_rearrange/llm_height_per.yaml",
                          ["habitat.dataset.should_numerical=False", *overrides])

    def requests(self, config, environment=None, raw=False):
        transport = Transport([
            completion("{agent_0||task zero}{agent_1||task one}"),
            completion("{{yes}}"), completion("{{yes}}"),
            completion("plan zero"), completion("wait", [tool("wait", {})]),
            completion("plan one"), completion("wait", [tool("wait", {})]),
        ])
        client = SimpleNamespace(chat=SimpleNamespace(completions=transport))

        def execute(config, run_type):
            self.assertEqual(run_type, "eval")
            self.assertEqual(os.environ.get("EMOS_LLM_MODEL"), original)
            policy = MultiLLMPolicy.from_config(config, None, None)
            children = make_policy()._active_policies
            for child in children:
                agent, action = child._high_level_policy.llm_agent, child.act

                def act(*args, agent=agent, action=action, **kwargs):
                    agent.chat("observation")
                    return action(*args, **kwargs)

                child.act = act
            policy.set_active(children)
            robots = {"agent_0": {"robot_type": "SpotRobot"},
                      "agent_1": {"robot_type": "DJIDrone"}}
            with tempfile.TemporaryDirectory() as directory, patch.object(
                    HydraConfig, "get", return_value=SimpleNamespace(
                        job=SimpleNamespace(config_name="multi_rearrange/llm_height_per.yaml"))):
                policy.act({"pddl_text_goal": torch.tensor([[ord(c) for c in "task"]])},
                           torch.ones(1, 2), torch.zeros(1, 4),
                           torch.zeros(1, 2, dtype=torch.bool), save_chat_history=True,
                           save_chat_history_dir=directory,
                           envs_text_context=[{"robot_resume": json.dumps(robots),
                                              "scene_description": "room", "episode_id": "1"}])
                for child in children:
                    child._high_level_policy.llm_agent.llm_model.enable_logging = False

        with patch.dict(os.environ):
            os.environ.pop("EMOS_LLM_MODEL", None)
            if environment is not None:
                os.environ["EMOS_LLM_MODEL"] = environment
            original = os.environ.get("EMOS_LLM_MODEL")
            with patch.object(run, "execute_exp", side_effect=execute), patch.object(
                    models.openai, "OpenAI", return_value=client), patch.object(
                    CrabAgent, "message_pipe", {}), contextlib.redirect_stdout(io.StringIO()):
                run.main.__wrapped__(config)
            self.assertEqual(os.environ.get("EMOS_LLM_MODEL"), original)
        return transport.requests if raw else [request["model"] for request in transport.requests]

    def test_default_config_uses_default_and_ignores_environment(self):
        config = self.config()
        self.assertEqual(config.habitat_baselines.llm_model, "gpt-5.6-sol")
        self.assertEqual(self.requests(config), ["gpt-5.6-sol"] * 7)
        self.assertEqual(self.requests(deepcopy(config), "existing-model"), ["gpt-5.6-sol"] * 7)

    def test_command_model_overrides_environment_in_all_phases(self):
        config = self.config("habitat_baselines.llm_model=command-model")
        self.assertEqual(self.requests(config, "existing-model"), ["command-model"] * 7)

    def test_command_model_is_trimmed(self):
        config = self.config("habitat_baselines.llm_model='  command-model  '")
        self.assertEqual(self.requests(config), ["command-model"] * 7)

    def test_command_decoding_controls_reach_all_phases_and_trace(self):
        config = self.config("habitat_baselines.llm_temperature=0.0",
                             "habitat_baselines.llm_seed=100")
        events = []
        with tempfile.TemporaryDirectory() as directory, EvaluationTrace(
                directory, event_callback=events.append) as trace:
            trace.start_episode(0, SimpleNamespace(scene_id="scene", episode_id="test"))
            with trace.policy_step():
                requests = self.requests(config, raw=True)
        self.assertEqual(len(requests), 7)
        for request in requests:
            self.assertEqual(request["temperature"], 0.0)
            self.assertEqual(request["seed"], 100)
        configurations = [event for event in events if event["event_type"] == "model_config"]
        self.assertEqual(len(configurations), 7)
        self.assertEqual({event["phase"] for event in configurations},
                         {"discussion", "planning", "execution"})
        for event in configurations:
            self.assertEqual(event["request_kwargs"]["temperature"], 0.0)
            self.assertEqual(event["request_kwargs"]["seed"], 100)
            self.assertEqual(event["parameter_sources"]["temperature"], "explicit_request")
            self.assertEqual(event["parameter_sources"]["seed"], "explicit_request")

    def test_unspecified_command_decoding_controls_remain_omitted(self):
        config = self.config()
        self.assertIsNone(config.habitat_baselines.llm_temperature)
        self.assertIsNone(config.habitat_baselines.llm_seed)
        for request in self.requests(config, raw=True):
            self.assertNotIn("temperature", request)
            self.assertNotIn("seed", request)

    def test_blank_command_is_rejected_before_execution(self):
        config = self.config("habitat_baselines.llm_model='   '")
        with patch.dict(os.environ, {"EMOS_LLM_MODEL": "existing-model"}), patch.object(
                run, "execute_exp") as execute:
            with self.assertRaisesRegex(ValueError, "llm_model"):
                run.main.__wrapped__(config)
            execute.assert_not_called()
            self.assertEqual(os.environ["EMOS_LLM_MODEL"], "existing-model")

    def test_environment_is_untouched_when_execution_raises(self):
        config = self.config("habitat_baselines.llm_model=command-model")
        for existing in (None, "existing-model"):
            with self.subTest(existing=existing), patch.dict(os.environ):
                os.environ.pop("EMOS_LLM_MODEL", None)
                if existing is not None:
                    os.environ["EMOS_LLM_MODEL"] = existing

                def fail(*args):
                    self.assertEqual(os.environ.get("EMOS_LLM_MODEL"), existing)
                    raise RuntimeError("original execution error")

                with patch.object(run, "execute_exp", side_effect=fail):
                    with self.assertRaisesRegex(RuntimeError, "original execution error"):
                        run.main.__wrapped__(deepcopy(config))
                self.assertEqual(os.environ.get("EMOS_LLM_MODEL"), existing)


if __name__ == "__main__":
    unittest.main()
