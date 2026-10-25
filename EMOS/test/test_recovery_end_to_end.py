"""Real recovery pipeline and live policy adapter; replace only LLM transport."""
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

from habitat_mas.detection.llm import LLMDetector
from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.controller import RecoveryController
from habitat_mas.recovery.generator import RecoveryGenerator
from habitat_mas.recovery import online
from habitat_mas.utils.trajectory import EvaluationTrace
from test_recovery_policy import make_policy


CODE = '''def recover(context, memory):
    commands = []
    if not memory.get("redirected", False):
        commands.append({"type":"action", "agent":"agent_0", "name":"nav_to_obj", "arguments":{"target_obj":"any_targets|0"}})
    return {"commands":commands, "memory":{"redirected":True}}
'''


class Transport:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)
        self.base_url = "https://example.invalid/v1"
        self.requests = []

    def create(self, **request):
        self.requests.append(deepcopy(request))
        data = json.loads(request["messages"][1]["content"])
        if "recent_frames" in data:
            decision = {"status": "system_error", "reason_code": "no_team_progress",
                "explanation": "team remains waiting", "agents": ["agent_0"],
                "evidence": [{"step": data["recent_frames"][-1]["step"], "observation": "waiting"}],
                "confidence": .9}
        else:
            decision = {"code": CODE, "explanation": "add a stateful redirection branch"}
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(decision)}}]}


class EndToEndRecoveryTest(unittest.TestCase):
    def test_invalid_generated_target_is_rejected_before_interrupting_any_agent(self):
        class Alarm:
            def observe(self, frame):
                return {"status": "system_error", "evidence": [{"step": frame["step"], "observation": "stalled"}]}
        class InvalidTarget:
            def generate(self, context):
                return {"code": CODE.replace("any_targets|0", "NONEXISTENT_ENTITY"), "explanation": "invalid candidate"}
        policy = make_policy()
        with tempfile.TemporaryDirectory() as output:
            controller = RecoveryController(RecoveryConfig(sample_every=1), output,
                detector=Alarm(), generator=InvalidTarget())
            result = controller.observe({"step": 1, "task": "Mobility", "context": {},
                "agents": {"agent_0": {}, "agent_1": {}}, "objects": {}, "metrics": {}}, policy)
            self.assertEqual(result["status"], "rejected")
            self.assertIsNone(controller.active_code)
            for child in policy._active_policies:
                self.assertFalse(child._cur_call_high_level[0])
                self.assertEqual(child._high_level_policy.llm_agent._recovery_actions, [])

    def test_online_alarm_generated_python_and_actual_agent_dispatch_preserve_episode(self):
        transport = Transport()
        config = RecoveryConfig(enabled=True, mode="extension", sample_every=1, window=2, check_every=1,
            cooldown_steps=1, verify_after_steps=1, max_iterations=1)
        controllers = []
        def make_controller(cfg, output):
            controller = RecoveryController(cfg, output,
                detector=LLMDetector(client=transport, window=2, check_every=1),
                generator=RecoveryGenerator(client=transport))
            controllers.append(controller)
            return controller
        policy = make_policy()
        agent = policy._active_policies[0]._high_level_policy.llm_agent
        model, history = agent.llm_model, deepcopy(agent.llm_model.chat_history)
        with tempfile.TemporaryDirectory() as output, patch.object(online, "RecoveryController", make_controller):
            bridge = online.OnlineRecovery(config, Path(output) / "recovery", task="Mobility")
            with EvaluationTrace(output, event_callback=bridge.on_event) as trace:
                trace.start_episode(0, SimpleNamespace(scene_id="room", episode_id="42"),
                    context={"task_description": "Cover target", "assignments": {"agent_0": "target"}})
                def step(position, distance):
                    state = {"agents": {"agent_0": {"base_position": [position, 0, 0], "grasped_object_id": 42}}}
                    trace.record_step(0, [0], 0, False, {"object_to_goal_distance.0": distance}, state)
                    return bridge.advance(policy, [False])
                step(0, 2)
                self.assertEqual(len(transport.requests), 0)  # A full window is required.
                result = step(0, 2)
                self.assertEqual(result[0]["status"], "applied")
                with trace.policy_step():
                    decision = agent.chat("same room, still holding object42")
                self.assertEqual(decision, {"name": "nav_to_obj", "arguments": {
                    "target_obj": "any_targets|0", "robot": "agent_0"}})
                self.assertIs(agent.llm_model, model)
                self.assertEqual(model.chat_history[:len(history)], history)
                self.assertEqual(model.requests, [])  # Recovery selected the next skill directly.
                self.assertEqual(step(1, 1)[0]["status"], "active_no_command")
                self.assertEqual(controllers[0].versions[0]["outcomes"][0]["status"], "progress_observed")
                self.assertEqual(len(controllers), 1)
                context = json.loads(next((Path(output) / "recovery").rglob("context.json")).read_text())
                self.assertEqual(context["runtime"]["current_frame"]["agents"]["agent_0"]["held_object"], 42)
                self.assertEqual(context["planner_plan"]["agent_0"], "navigate, pick, navigate, place")
                self.assertEqual(len(context["recent_history"]["agent_0"]), 10)
                self.assertIn("def", context["code"]["control"]["content"])


if __name__ == "__main__":
    unittest.main()
