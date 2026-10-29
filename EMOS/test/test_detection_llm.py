"""Causal LLM monitoring; only the external API transport is replaced."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def answer(system_error=False, **updates):
    value = {"system_error": system_error}
    value.update(updates)
    return value


def response(value):
    return SimpleNamespace(
        id="test-completion", model="served-model", object="chat.completion",
        choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(
                content=json.dumps(value, ensure_ascii=False)
                if isinstance(value, dict) else value,
                refusal=None,
            ),
        )],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )


class Transport:
    """OpenAI-compatible fake boundary, including its returned response fields."""

    def __init__(self, *values):
        self.values = iter(values)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        return response(value)


def client(*values):
    transport = Transport(*values)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=transport),
        base_url="https://example.invalid/v1",
    ), transport


def frame(step, **updates):
    value = {
        "step": step, "task": "Mobility",
        "context": {
            "task_description": "Cover both targets together.",
            "scene_description": "Two targets on the ground floor.",
            "assignments": {"agent_0": "Cover target 0"},
            "robot_capabilities": {"agent_0": "mobile robot"},
        },
        "agents": {"agent_0": {
            "position": [step / 10, 0, 0], "rotation": [0, 0, 0, 1],
            "held_object": None,
            "action": {"name": "nav_to_obj", "arguments": {"target_obj": "target_0"},
                       "id": "action-a", "age": step + 1},
            "navigation": {"path_found": True, "distance_to_goal": 5 - step / 10},
            "termination": None, "detection": None, "stop": False,
        }},
        "objects": {"target_0": {"position": [5, 0, 0], "object_id": 10}},
        "metrics": {"pddl_success": 0, "object_to_goal_distance.0": 4.0},
        "decisions": [], "messages": [],
    }
    value.update(updates)
    return value


class LLMDetectionTests(unittest.TestCase):
    def detector(self, *values, **kwargs):
        from habitat_mas.detection.llm import LLMDetector
        fake, transport = client(*values)
        return LLMDetector(client=fake, **kwargs), transport

    def feed(self, detector, start=0, stop=10):
        return [detector.observe(frame(step)) for step in range(start, stop)]

    def test_first_query_needs_full_window_then_uses_fixed_sample_cadence(self):
        monitor, api = self.detector(answer(), answer(), answer())
        values = self.feed(monitor, stop=20)
        self.assertEqual([i for i, value in enumerate(values) if value], [9, 14, 19])
        self.assertEqual([value["step"] for value in values if value], [9, 14, 19])
        self.assertEqual(monitor.stats["requests"], 3)
        self.assertEqual(len(api.requests), 3)

    def test_window_is_bounded_and_initial_task_context_is_fixed(self):
        monitor, api = self.detector(answer(), answer(), window=3, check_every=2)
        monitor.observe(frame(0))
        for step in range(1, 5):
            monitor.observe(frame(step, context={"scene_description": "changed later"}))
        first = json.loads(api.requests[0]["messages"][1]["content"])
        last = json.loads(api.requests[1]["messages"][1]["content"])
        self.assertEqual([row["step"] for row in last["recent_frames"]], [2, 3, 4])
        self.assertEqual(first["initial_context"], last["initial_context"])
        self.assertEqual(last["initial_context"]["context"]["scene_description"],
                         "Two targets on the ground floor.")

    def test_detector_input_excludes_labels_faults_terminal_outcomes_and_unknown_fields(self):
        monitor, api = self.detector(answer(), window=1)
        value = frame(0, done=True, native_success=True,
                      annotation="SECRET_ORACLE", fault_events=["SECRET_ORACLE"],
                      metrics={"error_detection.system_crash": 1,
                               "pddl_stage_goals.foo": 1,
                               "object_to_goal_distance.0": 2,
                               "native_success": "SECRET_ORACLE"})
        value["context"]["final_label"] = "SECRET_ORACLE"
        value["agents"]["agent_0"]["hang_state"] = "SECRET_ORACLE"
        value["agents"]["agent_0"]["navigation"]["fault"] = "SECRET_ORACLE"
        monitor.observe(value)
        serialized = json.dumps(api.requests[0], ensure_ascii=False)
        self.assertNotIn("SECRET_ORACLE", serialized)
        payload = json.loads(api.requests[0]["messages"][1]["content"])
        self.assertEqual(payload["recent_frames"][0]["metrics"],
                         {"object_to_goal_distance.0": 2})
        self.assertNotIn("done", payload["recent_frames"][0])

    def test_only_past_frames_are_available_even_after_caller_mutates_input(self):
        monitor, api = self.detector(answer(), window=2)
        initial = frame(0)
        monitor.observe(initial)
        initial["agents"]["agent_0"]["position"][0] = 999
        monitor.observe(frame(50))
        payload = json.loads(api.requests[0]["messages"][1]["content"])
        self.assertEqual(payload["recent_frames"][0]["agents"]["agent_0"]["position"],
                         [0.0, 0, 0])

    def test_goal_coordinates_current_reassignment_and_message_direction_survive_whitelist(self):
        monitor, api = self.detector(answer(), window=2)
        original = frame(0)
        original["context"]["goal_positions"] = {"any_targets|0": [5, 1, 2]}
        monitor.observe(original)
        changed = frame(50)
        changed["context"]["assignments"] = {"agent_0": "Cover target 1 now"}
        changed["messages"] = [{"kind": "receive", "sender": "agent_1", "recipient": "agent_0",
                                "message": "Please handle target 1", "step": 49}]
        changed["agents"]["agent_0"]["termination"] = {
            "is_skill_done": False, "cur_skill_step": 100,
            "over_max_len": True, "high_level_interrupt": False,
        }
        monitor.observe(changed)
        payload = json.loads(api.requests[0]["messages"][1]["content"])
        self.assertEqual(payload["initial_context"]["context"]["goal_positions"],
                         {"any_targets|0": [5, 1, 2]})
        latest = payload["recent_frames"][-1]
        self.assertEqual(latest["assignments"], {"agent_0": "Cover target 1 now"})
        self.assertEqual(latest["messages"][0]["kind"], "receive")
        self.assertTrue(latest["agents"]["agent_0"]["termination"]["over_max_len"])

    def test_sdk_response_types_work_without_fake_response_shortcuts(self):
        try:
            from openai.types.chat import ChatCompletion
        except ImportError:
            self.skipTest("OpenAI SDK is optional for dependency-free unit runs")
        from habitat_mas.detection.llm import LLMDetector
        value = ChatCompletion(
            id="actual-sdk-shape", object="chat.completion", created=0,
            model="served-model", choices=[{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(answer())},
            }], usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        )
        fake, _ = client()
        fake.chat.completions.create = lambda **kwargs: value
        monitor = LLMDetector(client=fake, window=1)
        self.assertEqual(monitor.observe(frame(0))["status"], "normal")
        self.assertEqual(monitor.stats["total_tokens"], 18)

    def test_boolean_alarm_gets_program_step_without_fabricating_model_details(self):
        monitor, _ = self.detector(answer(True), window=2)
        monitor.observe(frame(0))
        result = monitor.observe(frame(50))
        self.assertEqual(result, {"system_error": True, "status": "system_error",
                                  "step": 50, "detector": "llm"})
        self.assertEqual(monitor.stats["system_error"], 1)

    def test_boolean_false_is_a_valid_negative_without_extra_output_fields(self):
        monitor, _ = self.detector(answer(False), window=1)
        self.assertEqual(monitor.observe(frame(0)), {
            "system_error": False, "status": "normal", "step": 0, "detector": "llm"})
        self.assertEqual(monitor.stats["normal"], 1)

    def test_boolean_result_still_receives_original_decisions_and_messages(self):
        monitor, api = self.detector(answer(True), window=2)
        decision = {"agent": "agent_0", "tool": "wait", "step": 17}
        message = {"sender": "agent_1", "recipient": "agent_0", "message": "retry", "step": 23}
        monitor.observe(frame(0))
        result = monitor.observe(frame(50, decisions=[decision], messages=[message]))
        self.assertTrue(result["system_error"])
        recent = json.loads(api.requests[0]["messages"][1]["content"])["recent_frames"]
        self.assertEqual(recent[-1]["decisions"][0]["step"], 17)
        self.assertEqual(recent[-1]["messages"][0], message)

    def test_nonboolean_or_extra_output_fields_are_errors_not_normal(self):
        invalid = ["not json", "true", "false", "null", "[]", "{}",
                   answer(None), answer(0), answer(1), answer("false"), answer("true"),
                   answer(False, explanation="extra"), answer(True, confidence=0.9),
                   {"status": "normal"}, {"status": "uncertain"}]
        for value in invalid:
            with self.subTest(value=value):
                monitor, _ = self.detector(value, window=1)
                result = monitor.observe(frame(0))
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["reason_code"], "invalid_response")
                self.assertEqual(monitor.stats["errors"], 1)
                self.assertEqual(monitor.stats["normal"], 0)
                self.assertNotIn("system_error", result)

    def test_timeout_remains_error_and_next_window_can_still_be_checked(self):
        monitor, _ = self.detector(TimeoutError("read timed out"), answer(),
                                  window=1, check_every=1)
        self.assertEqual(monitor.observe(frame(0))["status"], "error")
        self.assertEqual(monitor.observe(frame(1))["status"], "normal")
        self.assertEqual(monitor.stats["requests"], 2)
        self.assertEqual(monitor.stats["errors"], 1)

    def test_cache_resume_reuses_only_identical_prompt_model_and_context(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first, _ = self.detector(answer(), window=1, cache_dir=cache)
            expected = first.observe(frame(0))
            second, api = self.detector(window=1, cache_dir=cache, max_calls=0)
            self.assertEqual(second.observe(frame(0)), expected)
            self.assertEqual(second.stats["cache_hits"], 1)
            self.assertEqual(second.stats["requests"], 0)
            self.assertEqual(api.requests, [])
            other, _ = self.detector(answer(), model="different", window=1, cache_dir=cache)
            other.observe(frame(0))
            scene, _ = self.detector(answer(), window=1, cache_dir=cache)
            scene.observe(frame(0, context={"scene_description": "a different scene"}))
            self.assertEqual(other.stats["requests"], 1)
            self.assertEqual(scene.stats["requests"], 1)
            self.assertEqual(len(list(cache.glob("*.json"))), 3)

    def test_equivalent_integer_timeout_reuses_existing_float_timeout_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            first, _ = self.detector(answer(), timeout=60.0, window=1, cache_dir=directory)
            expected = first.observe(frame(0))
            second, api = self.detector(timeout=60, window=1, cache_dir=directory, max_calls=0)
            self.assertEqual(second.observe(frame(0)), expected)
            self.assertEqual(second.stats["cache_hits"], 1)
            self.assertEqual(second.stats["requests"], 0)
            self.assertEqual(api.requests, [])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)

    def test_provider_endpoints_never_share_cached_model_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            first, _ = self.detector(answer(), window=1, cache_dir=directory)
            first.client.base_url = "https://provider-a.invalid/v1"
            self.assertEqual(first.observe(frame(0))["status"], "normal")
            second, _ = self.detector(answer(True), window=1, cache_dir=directory)
            second.client.base_url = "https://provider-b.invalid/v1"
            self.assertEqual(second.observe(frame(0))["status"], "system_error")
            self.assertEqual(second.stats["requests"], 1)
            self.assertEqual(second.stats["cache_hits"], 0)
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)

    def test_lazy_cache_lookup_uses_environment_endpoint_without_creating_client(self):
        from habitat_mas.detection.llm import LLMDetector
        with tempfile.TemporaryDirectory() as directory:
            first, _ = self.detector(answer(), window=1, cache_dir=directory)
            first.client.base_url = "https://provider-a.invalid/v1/"
            expected = first.observe(frame(0))
            with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://PROVIDER-A.invalid:443/v1"}):
                lazy = LLMDetector(window=1, cache_dir=directory, max_calls=0)
                self.assertEqual(lazy.observe(frame(0)), expected)
                self.assertIsNone(lazy.client)
                self.assertEqual(lazy.stats["requests"], 0)
            with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://provider-b.invalid/v1"}):
                changed = LLMDetector(window=1, cache_dir=directory, max_calls=0)
                self.assertEqual(changed.observe(frame(0))["status"], "budget_exhausted")
                self.assertIsNone(changed.client)
                self.assertEqual(changed.stats["cache_hits"], 0)

    def test_endpoint_cache_identity_normalizes_urls_without_persisting_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            first, _ = self.detector(answer(), window=1, cache_dir=directory, audit_path=audit)
            first.client.base_url = "https://user:password@PROVIDER.invalid:443/v1/?api_key=secret-a"
            expected = first.observe(frame(0))
            second, _ = self.detector(window=1, cache_dir=directory, audit_path=audit, max_calls=0)
            second.client.base_url = "https://provider.invalid/v1?api_key=secret-b"
            self.assertEqual(second.observe(frame(0)), expected)
            self.assertEqual(second.stats["cache_hits"], 1)
            for path in list(Path(directory).glob("*.json")) + [audit]:
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("user:password", content)
                self.assertNotIn("secret-a", content)
                self.assertNotIn("secret-b", content)

    def test_malformed_or_invalid_cached_content_does_not_become_a_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _ = self.detector(answer(), window=1, cache_dir=directory)
            monitor.observe(frame(0))
            path = next(Path(directory).glob("*.json"))
            path.write_text("broken", encoding="utf-8")
            second, _ = self.detector(answer(True), window=1, cache_dir=directory)
            self.assertEqual(second.observe(frame(0))["status"], "system_error")
            self.assertEqual(second.stats["cache_hits"], 0)
            self.assertEqual(second.stats["requests"], 1)

    def test_invalid_response_is_not_cached_as_normal(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _ = self.detector("{}", window=1, cache_dir=directory)
            self.assertEqual(monitor.observe(frame(0))["status"], "error")
            self.assertEqual(list(Path(directory).glob("*.json")), [])

    def test_budget_exhaustion_is_explicit_and_makes_no_api_call(self):
        monitor, api = self.detector(window=1, max_calls=0)
        result = monitor.observe(frame(0))
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertEqual(monitor.stats["budget_exhausted"], 1)
        self.assertEqual(monitor.stats["requests"], 0)
        self.assertEqual(api.requests, [])

    def test_audit_preserves_request_response_usage_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            secret = "sensitive-test-key-12345"
            with patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
                monitor, api = self.detector(answer(explanation="日志 " + secret),
                                            window=1, audit_path=audit)
                monitor.client.base_url = "https://user:password@example.invalid/v1?api_key=" + secret
                monitor.observe(frame(0))
            text = audit.read_text(encoding="utf-8")
            self.assertNotIn(secret, text)
            self.assertNotIn("user:password", text)
            row = json.loads(text)
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["request"]["model"], "gpt-6-sol")
            self.assertEqual(row["response"]["model"], "served-model")
            self.assertEqual(row["usage"]["total_tokens"], 18)
            self.assertEqual(monitor.stats["total_tokens"], 18)
            self.assertEqual(api.requests[0]["response_format"], {"type": "json_object"})
            self.assertEqual(api.requests[0]["timeout"], 60)
            self.assertNotIn("temperature", api.requests[0])

    def test_timeout_and_parse_failure_keep_original_raw_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            audit = Path(directory) / "audit.jsonl"
            monitor, _ = self.detector("not json", TimeoutError("server timeout"),
                                      window=1, check_every=1, audit_path=audit)
            self.feed(monitor, stop=2)
            rows = [json.loads(line) for line in audit.read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["error", "error"])
            self.assertEqual(rows[0]["response"]["choices"][0]["message"]["content"], "not json")
            self.assertEqual(rows[1]["error"]["type"], "TimeoutError")

    def test_invalid_configuration_and_nonincreasing_steps_are_rejected(self):
        from habitat_mas.detection.llm import LLMDetector
        for kwargs in [{"window": 0}, {"check_every": 0}, {"timeout": 0},
                       {"max_calls": -1}, {"model": ""}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    LLMDetector(**kwargs)
        monitor, _ = self.detector(window=10)
        monitor.observe(frame(5))
        with self.assertRaises(ValueError):
            monitor.observe(frame(5))


if __name__ == "__main__":
    unittest.main()
