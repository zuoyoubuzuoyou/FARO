"""Research trace contract: identity, action links, summaries and media."""

import base64
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
from habitat_mas.utils import trajectory as trace_module


def read(path):
    return json.loads(Path(path).read_text())


class ResearchTraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.trace = trace_module.EvaluationTrace(self.temp.name, {
            "benchmark": "Habitat-MAS", "api_key": "private-key",
            "success_measure": {"name": "pddl_success"},
        })
        self.addCleanup(self.trace.close)
        self.trace.start_episode(0, SimpleNamespace(scene_id="scene-a", episode_id="../7"),
                                 context={"goal": "move cup"}, state={"agents": {}})

    def episode_dir(self):
        return next((self.trace.run_dir / "episodes").iterdir())

    def events(self):
        return [json.loads(line) for line in
                (self.episode_dir() / "events.jsonl").read_text().splitlines()]

    def test_action_lifecycle_and_distinct_agent_decision_counts(self):
        with self.trace.policy_step():
            with trace_module.decision_scope("agent_0", "execution"):
                with trace_module.decision_scope("agent_0", "execution"):
                    trace_module.record_event("tool_selection", agent_name="agent_0", action_name="wait")
            action_id = trace_module.begin_action("agent_0", "wait", ["500"])
        self.trace.submit_actions(0, [0.0, 1.0], [2], ["agent_0"])
        self.trace.record_step(0, [0.0, 1.0], 0.0, False, {}, {"agents": {}})
        with self.trace.policy_step():
            trace_module.record_event("controller_signal", agent_name="agent_0")
            trace_module.end_action("agent_0", "controller_terminated", reason="skill_done_signal")
        self.trace.submit_actions(0, [0.0, 0.0], [2], ["agent_0"])
        self.trace.record_step(0, [0.0, 0.0], 1.0, True,
                               {"pddl_success": True}, {"agents": {"agent_0": {"base_position": [1, 0, 0]}}},
                               success=True, termination_reason="success")
        events = self.events()
        linked = [e for e in events if e["agent_id"] == "agent_0"]
        self.assertTrue(linked)
        self.assertEqual({e["action_id"] for e in linked}, {action_id})
        self.assertEqual({e["decision_step"] for e in linked}, {1})
        self.assertEqual([e["event_type"] for e in linked].count("action_start"), 1)
        self.assertEqual([e["event_type"] for e in linked].count("action_end"), 1)
        self.assertEqual(next(e for e in linked if e["event_type"] == "action_end")["env_step"], 2)
        self.assertEqual(len({e["event_id"] for e in events}), len(events))
        self.assertEqual([e["sequence"] for e in events], list(range(len(events))))
        summary = read(self.episode_dir() / "summary.json")
        self.assertTrue(summary["task_success"])
        self.assertEqual(summary["num_env_steps"], 2)
        self.assertEqual(summary["decision_counts"], {"agent_0": 1})
        self.assertEqual(summary["termination_reason"], "success")
        self.assertEqual(summary["last_actions"]["agent_0"]["action_id"], action_id)
        self.assertTrue((self.episode_dir() / summary["files"]["events"]).is_file())

    def test_pending_submission_survives_exception_with_same_action_id(self):
        with self.trace.policy_step():
            with trace_module.decision_scope("agent_0"):
                trace_module.record_event("llm_request", agent_name="agent_0", messages=[])
            action_id = trace_module.begin_action("agent_0", "nav_to_obj", {"target": "cup"})
        self.trace.submit_actions(0, [1], [1], ["agent_0"])
        self.trace.close("error", RuntimeError("worker failed"))
        summary = read(self.episode_dir() / "summary.json")
        self.assertEqual(summary["status"], "error")
        self.assertIsNone(summary["task_success"])
        self.assertIn("task_success", summary["missing_fields"])
        self.assertEqual(summary["exception_count"], 1)
        end = next(e for e in self.events() if e["event_type"] == "action_end")
        self.assertEqual(end["action_id"], action_id)
        self.assertEqual(end["status"], "error")

    def test_config_redacts_credentials_and_summary_does_not_create_labels(self):
        config = read(self.trace.run_dir / "config.json")
        self.assertNotIn("private-key", json.dumps(config))
        self.assertEqual(config["metadata"]["api_key"], "[REDACTED]")
        self.trace.record_step(0, "stop", 0, True, {"pddl_success": False}, None,
                               success=False, termination_reason="time_limit")
        summary = read(self.episode_dir() / "summary.json")
        self.assertFalse(summary["task_success"])
        self.assertNotIn("system_error", summary)
        self.assertNotIn("label", summary)
        self.assertIsNone(summary["final_state"])
        self.assertIn("final_state", summary["missing_fields"])
        self.assertTrue((self.episode_dir() / "episode.json").is_file())
        self.assertTrue((self.episode_dir() / "media").is_dir())

    def test_actual_image_input_is_externalized_without_mutating_request(self):
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a6c8AAAAASUVORK5CYII=")
        uri = "data:image/png;base64," + base64.b64encode(png).decode()
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}]
        with self.trace.policy_step():
            with trace_module.decision_scope("agent_0"):
                trace_module.record_event("llm_request", agent_name="agent_0", messages=messages)
        self.assertEqual(messages[0]["content"][0]["image_url"]["url"], uri)
        event = next(e for e in self.events() if e["event_type"] == "llm_request")
        reference = event["messages"][0]["content"][0]["image_url"]
        self.assertIsNone(reference["url"])
        self.assertEqual((self.episode_dir() / reference["media_path"]).read_bytes(), png)
        self.assertNotIn(uri, json.dumps(event))

    def test_non_image_urls_do_not_trigger_network_or_media(self):
        with self.trace.policy_step(), patch("urllib.request.urlopen", side_effect=AssertionError("network")):
            trace_module.record_event("tool_execution", result={"url": "https://example.invalid/docs"})
        event = next(e for e in self.events() if e["event_type"] == "tool_execution")
        self.assertEqual(event["result"], {"url": "https://example.invalid/docs"})

    def test_parse_errors_are_counted_once_when_exception_propagates(self):
        with self.trace.policy_step():
            trace_module.record_event("llm_parse", agent_name="agent_0", status="error",
                                      error_type="ValueError", error_message="invalid action", handling="rethrow")
        self.trace.close("error", ValueError("invalid action"))
        summary = read(self.episode_dir() / "summary.json")
        self.assertEqual(summary["exception_count"], 1)
        self.assertEqual(summary["exception_counts"], {"llm_parse": 1})

    def test_latest_failed_decision_is_preserved_after_an_earlier_action(self):
        with self.trace.policy_step():
            with trace_module.decision_scope("agent_0"):
                earlier = trace_module.begin_action("agent_0", "wait", [1])
            trace_module.end_action("agent_0", "controller_terminated", defer=False)
        self.trace.record_step(0, [0], 0, False, {}, {})
        with self.trace.policy_step():
            with trace_module.decision_scope("agent_0"):
                latest = trace_module.current_action_id("agent_0")
                trace_module.record_event("llm_error", error_type="RuntimeError", error_message="offline")
        self.trace.close("error", RuntimeError("offline"))
        summary = read(self.episode_dir() / "summary.json")
        self.assertNotEqual(earlier, latest)
        self.assertEqual(summary["last_actions"]["agent_0"]["action_id"], latest)
        self.assertEqual(summary["last_actions"]["agent_0"]["status"], "not_entered")
        self.assertTrue(any(e["event_type"] == "action_end" and e["action_id"] == latest for e in self.events()))

    def test_propagated_error_is_not_counted_at_multiple_catch_sites(self):
        with self.trace.policy_step():
            with trace_module.decision_scope("agent_0"):
                trace_module.record_event("llm_error", error_type="RuntimeError", error_message="offline")
                trace_module.record_event("execution_error", error={"type": "RuntimeError", "message": "offline"})
        self.trace.close("error", RuntimeError("offline"))
        self.assertEqual(read(self.episode_dir() / "summary.json")["exception_count"], 1)

    def test_credential_urls_are_redacted_inside_error_text(self):
        text = "Image URL: https://fakeuser:fakepass@example.test/a?X-Amz-Credential=FAKEACCESS&X-Amz-Security-Token=FAKETOKEN"
        with self.trace.policy_step():
            trace_module.record_event("exception", error={"message": text})
        saved = json.dumps(self.events())
        for secret in ("fakeuser", "fakepass", "FAKEACCESS", "FAKETOKEN"):
            self.assertNotIn(secret, saved)

    def test_idle_agents_and_output_images_are_not_misrepresented(self):
        self.trace.metadata["agents"] = [{"agent_id": "idle_agent", "robot_type": "SpotRobot"}]
        uri = "data:image/png;base64," + base64.b64encode(b"image bytes").decode()
        with self.trace.policy_step():
            trace_module.record_event("tool_execution", result={"image_url": {"url": uri}})
        self.trace.close()
        event = next(e for e in self.events() if e["event_type"] == "tool_execution")
        self.assertFalse(event["result"]["image_url"]["seen_by_llm"])
        summary = read(self.episode_dir() / "summary.json")
        self.assertIsNone(summary["last_actions"]["idle_agent"])
        self.assertIn("last_actions.idle_agent", summary["missing_fields"])

    def test_manifest_keeps_observed_decode_and_response_versions(self):
        messages = [{"role": "system", "content": "same prompt"}]
        with self.trace.policy_step():
            for call_id, temperature, response_model in (("a", 0.0, "version-a"), ("b", 1.0, "version-b"), ("c", 0.0, "version-c")):
                trace_module.record_event("llm_request", agent_name="agent_0", phase="execution",
                                          call_id=call_id, model="alias", temperature=temperature, messages=messages)
                trace_module.record_event("llm_response", agent_name="agent_0", phase="execution",
                                          call_id=call_id, model="alias", response_model=response_model)
        manifest = read(self.trace.run_dir / "config.json")["models_used"]
        self.assertEqual(len(manifest), 2)
        by_temperature = {entry["decoding_parameters"]["temperature"]: entry for entry in manifest}
        self.assertEqual(by_temperature[0.0]["response_model_ids"], ["version-a", "version-c"])
        self.assertEqual(by_temperature[1.0]["response_model_ids"], ["version-b"])

    def test_missing_episode_after_logging_setup_failure_does_not_break_task(self):
        self.trace.close()
        other = trace_module.EvaluationTrace(self.temp.name)
        self.addCleanup(other.close)
        other.submit_actions(0, [1], [1], ["agent_0"])
        other.record_step(0, [1], 0.0, True, {}, None)


if __name__ == "__main__":
    unittest.main()
