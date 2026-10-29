"""Combined source monitoring uses one response and the real source transaction."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
sys.path.insert(0, str(ROOT / "test"))

from habitat_mas.detection.llm import LLMDetector
from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.context_compaction import expand_runtime_context
from habitat_mas.recovery.online import OnlineRecovery
from habitat_mas.recovery.source_controller import SourceRecoveryController
from habitat_mas.recovery.source_generator import SourceRecoveryGenerator
from test_recovery_source_controller import Detector, ORIGINAL, RESUME, Revisions, fixture, frame


def repair(context, code=RESUME):
    current = context["code"]["source_files"]["policy.py"]
    return {"system_error": True, "patches": [{"path": "policy.py",
        "target": "MultiLLMPolicy.Resume", "base_sha256": current["sha256"], "code": code}],
        "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"}}


class Transport:
    """Replace only network IO; request construction and decoding stay real."""
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **request):
        self.requests.append(deepcopy(request))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(json.loads(request["messages"][1]["content"]))
        content = value if isinstance(value, str) else json.dumps(value)
        return {"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}


def controller(root, runtime, transport, **values):
    settings = {"sample_every": 1, "window": 1, "check_every": 1,
                "cooldown_steps": 1, "verify_after_steps": 1, **values}
    return SourceRecoveryController(RecoveryConfig(mode="source", **settings), root / "audit",
        runtime=runtime, generator=SourceRecoveryGenerator(client=transport))


class CombinedGeneratorTests(unittest.TestCase):
    def test_negative_request_includes_runtime_and_source_without_a_repair_request(self):
        transport = Transport({"system_error": False})
        context = {"runtime": {"current_frame": {"step": 4}},
                   "code": {"source_files": {"policy.py": {"content": ORIGINAL, "sha256": "a" * 64}}}}
        result = SourceRecoveryGenerator(client=transport).detect_and_repair(context)
        self.assertFalse(result["system_error"])
        self.assertEqual(len(transport.requests), 1)
        payload = json.loads(transport.requests[0]["messages"][1]["content"])
        self.assertEqual(payload["runtime"], context["runtime"])
        self.assertEqual(payload["code"]["source_files"]["policy.py"]["view"], "methods-v1")
        self.assertEqual(payload["code"]["source_files"]["policy.py"]["sha256"], "a" * 64)
        self.assertIn("def act(self)", payload["code"]["source_files"]["policy.py"]["content"])
        self.assertNotIn("patches", result)

    def test_positive_response_requires_no_model_explanation(self):
        value = {"system_error": True, "patches": [{"path": "policy.py",
            "target": "MultiLLMPolicy.Resume", "base_sha256": "a" * 64, "code": RESUME}],
            "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"}}
        result = SourceRecoveryGenerator(client=Transport(value)).detect_and_repair({})
        self.assertEqual(result["patches"], value["patches"])
        self.assertTrue(result["system_error"])

    def test_detected_problem_can_explicitly_have_no_safe_repair(self):
        value = {"system_error": True, "patches": [], "entrypoint": None}
        result = SourceRecoveryGenerator(client=Transport(value)).detect_and_repair({})
        self.assertEqual(result["patches"], [])
        self.assertIsNone(result["entrypoint"])
        self.assertTrue(result["system_error"])

    def test_invalid_boolean_union_and_patch_shapes_preserve_raw_response(self):
        invalid = [{"system_error": "false"}, {"system_error": False, "patches": []},
            {"system_error": True}, {"system_error": True, "patches": [], "entrypoint": {}},
            {"system_error": True, "patches": [{"code": RESUME}], "entrypoint": None},
            '{"system_error": NaN}']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                SourceRecoveryGenerator(client=Transport(value)).detect_and_repair({})
            self.assertIsNotNone(caught.exception.response)
            self.assertIn("messages", caught.exception.request)

    def test_transport_failure_preserves_request_and_has_no_negative_prediction(self):
        with self.assertRaisesRegex(ValueError, "request failed") as caught:
            SourceRecoveryGenerator(client=Transport(TimeoutError("late"))).detect_and_repair({})
        self.assertIn("messages", caught.exception.request)
        self.assertIsNone(caught.exception.response)

    def test_incomplete_or_refused_model_responses_are_errors(self):
        responses = [{"choices": []},
            {"choices": [{"finish_reason": "length", "message": {"content": '{"system_error": false}'}}]},
            {"choices": [{"finish_reason": "stop", "message": {
                "content": '{"system_error": false}', "refusal": "refused"}}]}]
        for response in responses:
            def create(**request):
                return response
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            with self.subTest(response=response), self.assertRaises(ValueError) as caught:
                SourceRecoveryGenerator(client=client).detect_and_repair({})
            self.assertEqual(caught.exception.response, response)


class CombinedControllerTests(unittest.TestCase):
    def test_negative_check_has_no_source_or_policy_mutation_and_does_not_use_budget(self):
        with fixture() as (root, policy, runtime), patch.object(
                LLMDetector, "observe", side_effect=AssertionError("separate detection request")):
            (root / "worker.py").write_text("class Worker:\n    pass\n")
            runtime.targets["worker.py"] = {"module": "unloaded_worker", "classes": ["Worker"], "process": "worker"}
            transport = Transport({"system_error": False}, {"system_error": False})
            monitor = controller(root, runtime, transport, cooldown_steps=100, max_iterations=1)
            monitor._last_error = {"status": "system_error", "explanation": "stale verdict"}
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "normal")
            self.assertEqual(monitor.observe(frame(1), policy)["status"], "normal")
            self.assertEqual((root / "policy.py").read_text(), ORIGINAL)
            self.assertFalse(hasattr(policy, "Resume"))
            self.assertEqual(monitor.versions, [])
            self.assertIsNone(monitor._last_generation_step)
            self.assertEqual(len(transport.requests), 2)
            context = expand_runtime_context(json.loads(transport.requests[0]["messages"][1]["content"]))
            self.assertIsNone(context["error_report"])
            self.assertEqual(context["runtime"]["current_frame"]["step"], 0)
            self.assertIn("edit_policy", context["code"])
            self.assertIn("environment/task/simulator source", context["code"]["edit_policy"]["read_only"])
            self.assertIn("read-only", context["available_api"]["boundaries"])
            self.assertIn("code.edit_policy", context["available_api"]["boundaries"])
            self.assertEqual(set(context["available_api"]["proposal"]), {"system_error", "patches", "entrypoint"})
            self.assertIs(context["available_api"]["proposal"]["system_error"], True)
            self.assertEqual(context["available_api"]["negative"], {"system_error": False})
            self.assertEqual(context["available_api"]["unrepairable"],
                             {"system_error": True, "patches": [], "entrypoint": None})
            self.assertIn("read-only", context["runtime"]["source_targets"]["worker.py"]["worker_loading"])
            self.assertNotIn("checked transactionally", json.dumps(context["runtime"]["source_targets"]))
            rows = (root / "audit" / "source_requests.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 2)
            self.assertFalse(json.loads(rows[0])["decision"]["system_error"])

    def test_positive_response_is_installed_on_same_policy_without_another_model_call(self):
        with fixture() as (root, policy, runtime), patch.object(
                LLMDetector, "observe", side_effect=AssertionError("separate detection request")):
            history = policy.history
            transport = Transport(repair)
            monitor = controller(root, runtime, transport)
            result = monitor.observe(frame(0), policy)
            self.assertEqual(result["status"], "applied")
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(policy.act(), "move")
            self.assertIs(policy.history, history)
            self.assertEqual(policy.held_object, 42)
            self.assertEqual(policy.resume_count, 1)
            self.assertEqual(monitor.versions[0]["status"], "active")
            self.assertTrue(list((root / "audit").rglob("changes.diff")))

    def test_sampling_window_check_cadence_cooldown_and_budget_bound_calls(self):
        with fixture() as (root, policy, runtime), patch.object(
                LLMDetector, "observe", side_effect=AssertionError("separate detection request")):
            transport = Transport({"system_error": False}, repair, repair)
            monitor = controller(root, runtime, transport, sample_every=2, window=2,
                                 check_every=2, cooldown_steps=5, max_iterations=2)
            for step in range(21):
                monitor.observe(frame(step), policy)
            steps = [expand_runtime_context(json.loads(item["messages"][1]["content"]))["runtime"]["current_frame"]["step"]
                     for item in transport.requests]
            self.assertEqual(steps, [2, 6, 14])
            self.assertEqual(policy.resume_count, 2)
            self.assertEqual(len(monitor.versions), 2)
            self.assertEqual(len(monitor._recent), 2)

    def test_second_positive_check_uses_current_source_and_previous_real_feedback(self):
        with fixture() as (root, policy, runtime), patch.object(
                LLMDetector, "observe", side_effect=AssertionError("separate detection request")):
            transport = Transport(repair, repair)
            monitor = controller(root, runtime, transport, max_iterations=2)
            monitor.observe(frame(0), policy)
            monitor.observe(frame(1), policy)
            first, second = [json.loads(item["messages"][1]["content"]) for item in transport.requests]
            self.assertNotEqual(first["code"]["source_files"]["policy.py"]["sha256"],
                                second["code"]["source_files"]["policy.py"]["sha256"])
            self.assertIn("def Resume", second["code"]["source_files"]["policy.py"]["content"])
            self.assertEqual(second["recovery"]["previous_versions"][0]["outcomes"][0]["status"],
                             "no_observed_progress")
            self.assertIsNone(second["error_report"])

    def test_unrepairable_positive_is_recorded_without_installation(self):
        with fixture() as (root, policy, runtime):
            transport = Transport({"system_error": True, "patches": [], "entrypoint": None})
            monitor = controller(root, runtime, transport, max_iterations=1)
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "unrepairable")
            self.assertIsNone(monitor.observe(frame(1), policy))
            self.assertEqual(monitor.versions[0]["status"], "unrepairable")
            self.assertEqual((root / "policy.py").read_text(), ORIGINAL)
            self.assertFalse(hasattr(policy, "Resume"))
            self.assertEqual(len(transport.requests), 1)

    def test_malformed_combined_response_uses_one_bounded_rejected_attempt(self):
        with fixture() as (root, policy, runtime):
            transport = Transport({"system_error": "yes"})
            monitor = controller(root, runtime, transport, max_iterations=1)
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "rejected")
            self.assertIsNone(monitor.observe(frame(1), policy))
            self.assertEqual(monitor.versions[0]["status"], "rejected")
            self.assertTrue(list((root / "audit").rglob("generation_error.json")))
            self.assertEqual(len(transport.requests), 1)

    def test_exception_step_zero_queries_immediately_and_installs_from_that_response(self):
        with fixture() as (root, policy, runtime):
            transport = Transport(repair)
            monitor = controller(root, runtime, transport, window=10)
            result = monitor.handle_execution_error(frame(0), policy, ValueError("parser failed"))
            self.assertEqual(result["status"], "applied")
            self.assertEqual(policy.act(), "move")
            self.assertEqual(len(transport.requests), 1)
            context = json.loads(transport.requests[0]["messages"][1]["content"])
            self.assertEqual(context["runtime"]["current_frame"]["step"], 0)
            self.assertEqual(context["error_report"]["exception"]["type"], "ValueError")
            self.assertEqual(context["runtime"]["recent_frames"], [])
            self.assertEqual(context["trigger"], "execution_exception")
            self.assertEqual(context["observation_parameters"]["window"], 10)
            self.assertNotIn("status", context["error_report"])

    def test_exception_negative_or_unrepairable_stops_without_consuming_all_retries(self):
        for value, status, attempts in [({"system_error": False}, "normal", 0),
                ({"system_error": True, "patches": [], "entrypoint": None}, "unrepairable", 1)]:
            with self.subTest(status=status), fixture() as (root, policy, runtime):
                transport = Transport(value)
                monitor = controller(root, runtime, transport, max_iterations=3)
                result = monitor.handle_execution_error(frame(0), policy, ValueError("broken"))
                self.assertEqual(result["status"], status)
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(len(monitor.versions), attempts)

    def test_exception_rejects_then_revises_before_returning_to_the_same_decision(self):
        bad = 'def wrong(self, context):\n    return {}\n'
        with fixture() as (root, policy, runtime):
            transport = Transport(lambda context: repair(context, bad), repair)
            monitor = controller(root, runtime, transport, max_iterations=2)
            result = monitor.handle_execution_error(frame(0), policy, ValueError("broken"))
            self.assertEqual(result["status"], "applied")
            self.assertEqual([item["status"] for item in monitor.versions], ["rejected", "active"])
            self.assertEqual(len(transport.requests), 2)
            self.assertEqual(policy.act(), "move")

    def test_failed_combined_entrypoint_restores_original_source_and_policy_state(self):
        failing = 'def Resume(self, context):\n    self.active = True\n    raise ValueError("bad recovery")\n'
        with fixture() as (root, policy, runtime):
            transport = Transport(lambda context: repair(context, failing))
            monitor = controller(root, runtime, transport)
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "rejected")
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual((root / "policy.py").read_text(), ORIGINAL)
            self.assertFalse(hasattr(policy, "Resume"))
            self.assertFalse(policy.active)
            self.assertEqual(monitor.versions[0]["status"], "rejected")

    def test_terminal_observation_never_queries_or_installs(self):
        with fixture() as (root, policy, runtime):
            transport = Transport(repair)
            monitor = controller(root, runtime, transport)
            self.assertEqual(monitor.observe(frame(0), policy, terminal=True)["status"], "terminal_skipped")
            self.assertEqual(transport.requests, [])
            self.assertEqual(monitor.versions, [])

    def test_online_exception_retries_same_decision_after_installing_combined_response(self):
        with fixture() as (root, policy, runtime):
            transport = Transport(repair)
            generator = SourceRecoveryGenerator(client=transport)
            history = policy.history
            config = RecoveryConfig(enabled=True, mode="source", window=10)
            with patch("habitat_mas.recovery.online.SourceRuntime", return_value=runtime), patch(
                    "habitat_mas.recovery.source_controller.SourceRecoveryGenerator", return_value=generator):
                bridge = OnlineRecovery(config, root / "audit")
                bridge.on_event({"event_type": "episode_start", "env_index": 0, "env_step": 0,
                    "evaluation_id": "initial", "context": {"task_description": "Reach target"},
                    "state": {"agents": {"agent_0": {"base_position": [0, 0, 0], "grasped_object_id": 42}}}})
                try:
                    self.assertEqual(bridge.run_policy(policy.act, policy), "move")
                    self.assertEqual(len(transport.requests), 1)
                    self.assertEqual(bridge._latest_frame["step"], 0)
                    self.assertIs(policy.history, history)
                    self.assertEqual(policy.held_object, 42)
                    self.assertEqual(policy.resume_count, 1)
                finally:
                    bridge.close()
            self.assertEqual((root / "policy.py").read_text(), ORIGINAL)
            self.assertFalse(hasattr(policy, "Resume"))

    def test_online_negative_exception_response_propagates_original_error_without_retry(self):
        for value in [{"system_error": False}, {"system_error": True, "patches": [], "entrypoint": None}]:
            with self.subTest(value=value), fixture() as (root, policy, runtime):
                transport = Transport(value)
                generator = SourceRecoveryGenerator(client=transport)
                with patch("habitat_mas.recovery.online.SourceRuntime", return_value=runtime), patch(
                        "habitat_mas.recovery.source_controller.SourceRecoveryGenerator", return_value=generator):
                    bridge = OnlineRecovery(RecoveryConfig(enabled=True, mode="source"), root / "audit")
                    bridge.on_event({"event_type": "episode_start", "env_index": 0, "env_step": 0,
                        "evaluation_id": "initial", "context": {}, "state": {"agents": {}}})
                    try:
                        with self.assertRaisesRegex(RuntimeError, "no resume mechanism"):
                            bridge.run_policy(policy.act, policy)
                        self.assertEqual(len(transport.requests), 1)
                        self.assertEqual(bridge._latest_frame["step"], 0)
                        self.assertEqual((root / "policy.py").read_text(), ORIGINAL)
                    finally:
                        bridge.close()

    def test_explicit_detector_retains_legacy_source_generation_route(self):
        with fixture() as (root, policy, runtime):
            detector = Detector()
            revisions = Revisions([("MultiLLMPolicy.Resume", RESUME)])
            monitor = SourceRecoveryController(RecoveryConfig(sample_every=1), root / "audit",
                runtime=runtime, detector=detector, generator=revisions)
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "applied")
            self.assertEqual(detector.calls, [0])
            self.assertEqual(len(revisions.contexts), 1)
            self.assertNotIn("edit_policy", revisions.contexts[0]["code"])
            self.assertIn("installed via RPC", revisions.contexts[0]["available_api"]["boundaries"])
            self.assertEqual(set(revisions.contexts[0]["available_api"]["proposal"]),
                             {"patches", "entrypoint", "explanation"})

    def test_explicit_detector_with_model_generator_uses_source_only_protocol(self):
        def legacy_proposal(context):
            value = repair(context)
            value.pop("system_error")
            value["explanation"] = "Resume original execution"
            return value
        with fixture() as (root, policy, runtime):
            detector, transport = Detector(), Transport(legacy_proposal)
            monitor = SourceRecoveryController(RecoveryConfig(sample_every=1), root / "audit",
                runtime=runtime, detector=detector, generator=SourceRecoveryGenerator(client=transport))
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "applied")
            self.assertEqual(detector.calls, [0])
            self.assertEqual(policy.act(), "move")
            self.assertEqual(len(transport.requests), 1)
            context = json.loads(transport.requests[0]["messages"][1]["content"])
            self.assertEqual(set(context["available_api"]["proposal"]), {"patches", "entrypoint", "explanation"})


if __name__ == "__main__":
    unittest.main()
