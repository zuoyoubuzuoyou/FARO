"""Patch real EMOS model/policy source; replace only model network transports."""
import contextlib
from copy import deepcopy
import io
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

from habitat_mas.recovery import source_controller
from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.online import OnlineRecovery
from habitat_mas.recovery.source_generator import SourceRecoveryGenerator
from habitat_mas.utils import models
from habitat_baselines.rl.multi_agent.multi_llm_policy import MultiLLMPolicy
from test_recovery_policy import ACTIONS, make_policy
from test_llm_trajectory import completion, tool


POLICY_PATH = "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py"
MODEL_PATH = "habitat-mas/habitat_mas/utils/models.py"
RESUME = '''def ResumeFromSourceRevision(self, context):
    for child in self._active_policies:
        child._cur_call_high_level[0] = True
    self.source_resume_count = 1
    return {"resumed": True}
'''
PARSER = '''def _parse_tool_arguments(self, call, phase):
    try:
        return json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return json.loads(call.function.arguments + "}")
'''


class ModelTransport:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        selected = tool("nav_to_obj", {"target_obj": "any_targets|0"})
        selected["function"]["arguments"] = '{"target_obj":"any_targets|0"'
        return completion(tools=[selected])


class SourceTransport:
    def __init__(self):
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(deepcopy(request))
        context = json.loads(request["messages"][1]["content"])
        files = context["code"]["source_files"]
        proposal = {"patches": [
            {"path": POLICY_PATH, "target": "MultiLLMPolicy.ResumeFromSourceRevision",
             "base_sha256": files[POLICY_PATH]["sha256"], "code": RESUME},
            {"path": MODEL_PATH, "target": "OpenAIModel._parse_tool_arguments",
             "base_sha256": files[MODEL_PATH]["sha256"], "code": PARSER}],
            "entrypoint": {"path": POLICY_PATH, "target": "MultiLLMPolicy.ResumeFromSourceRevision"},
            "explanation": "Add original-class Resume and repair the original model parser"}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(proposal)}}]}


class OriginalSourceIntegrationTest(unittest.TestCase):
    def test_original_model_parser_exception_is_repaired_without_reinitializing_agent(self):
        original_files = {path: (ROOT / path).read_bytes() for path in (POLICY_PATH, MODEL_PATH)}
        original_parser = models.OpenAIModel._parse_tool_arguments
        robot_transport, repair_transport = ModelTransport(), SourceTransport()
        model_client = SimpleNamespace(chat=SimpleNamespace(completions=robot_transport))
        with patch.object(models.openai, "OpenAI", return_value=model_client):
            model = models.OpenAIModel("select a tool", ACTIONS, agent_name="agent_0", save_on_each_chat=False)
        model.chat_history = [[{"role": "user", "content": "prior turn %d" % i}] for i in range(12)]
        history = model.chat_history
        fixture_policy = make_policy()
        # Use the actual production class; keep only simulator import boundaries
        # represented by the same CPU fixtures used in policy lifecycle tests.
        policy = MultiLLMPolicy(None)
        policy._active_policies = fixture_policy._active_policies
        agent = policy._active_policies[0]._high_level_policy.llm_agent
        agent.llm_model = model
        cfg = RecoveryConfig(enabled=True, mode="source", sample_every=1, max_iterations=2)
        generator = SourceRecoveryGenerator(client=repair_transport)
        with tempfile.TemporaryDirectory() as output, patch.object(
                source_controller, "SourceRecoveryGenerator", return_value=generator), contextlib.redirect_stdout(io.StringIO()):
            bridge = OnlineRecovery(cfg, output, task="Mobility")
            bridge.on_event({"event_type": "episode_start", "env_index": 0, "env_step": 0,
                "evaluation_id": "real-original-source",
                "context": {"task_description": "Reach target", "assignments": {"agent_0": "target"}},
                "state": {"agents": {"agent_0": {"base_position": [1, 0, 0], "grasped_object_id": 42}}}})
            try:
                result = bridge.run_policy(lambda: agent.chat("current physical state"), policy)
                self.assertEqual(result, {"name": "nav_to_obj", "arguments": {
                    "target_obj": "any_targets|0", "robot": "agent_0"}})
                self.assertIs(agent.llm_model, model)
                self.assertIs(model.chat_history, history)
                self.assertEqual(len(history), 13)  # Failed user turn was rolled back.
                self.assertEqual(model.token_usage, 14)  # Both API responses still count.
                self.assertEqual(len(robot_transport.requests), 2)
                self.assertEqual(len(repair_transport.requests), 1)
                self.assertEqual(policy.source_resume_count, 1)
                self.assertTrue(policy._active_policies[0]._cur_call_high_level[0])
                self.assertIsNot(models.OpenAIModel._parse_tool_arguments, original_parser)
                self.assertIn("def ResumeFromSourceRevision", (ROOT / POLICY_PATH).read_text())
                self.assertIn('call.function.arguments + "}"', (ROOT / MODEL_PATH).read_text())
                context = json.loads(repair_transport.requests[0]["messages"][1]["content"])
                self.assertEqual(context["error_report"]["exception"]["type"], "JSONDecodeError")
                self.assertEqual(context["runtime"]["current_frame"]["agents"]["agent_0"]["held_object"], 42)
                self.assertTrue(list(Path(output).rglob("changes.diff")))
            finally:
                bridge.close()
        for path, expected in original_files.items():
            self.assertEqual((ROOT / path).read_bytes(), expected)
        self.assertIs(models.OpenAIModel._parse_tool_arguments, original_parser)
        self.assertFalse(hasattr(policy, "ResumeFromSourceRevision"))


if __name__ == "__main__":
    unittest.main()
