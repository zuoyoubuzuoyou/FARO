"""Recovery diagnostics remain visible without exposing model/source payloads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
sys.path.insert(0, str(ROOT / "test"))

from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.controller import RecoveryController
from test_recovery_combined import Transport, controller
from test_recovery_source_controller import fixture, frame


class RecoveryLoggingTests(unittest.TestCase):
    def test_detection_errors_are_not_logged_as_negative_predictions(self):
        with tempfile.TemporaryDirectory() as output:
            monitor = RecoveryController(RecoveryConfig(), output,
                                         detector=object(), generator=object())
            monitor._last_step = 451
            with self.assertLogs("habitat_mas.recovery", level="INFO") as captured:
                monitor._event("detection", decision={"status": "error"})
            self.assertIn("decision_status=error", captured.output[0])
            self.assertNotIn("system_error=False", captured.output[0])

    def test_events_reach_stderr_with_no_application_logging_configuration(self):
        script = """
import sys
sys.path.insert(0, sys.argv[1])
from habitat_mas.recovery.controller import RecoveryController
from habitat_mas.recovery.config import RecoveryConfig
monitor = RecoveryController(RecoveryConfig(), sys.argv[2], detector=object(), generator=object())
monitor._last_step = 451
monitor._event('detection', decision={'system_error': True, 'status': 'system_error'})
"""
        with tempfile.TemporaryDirectory() as output:
            result = subprocess.run(
                [sys.executable, "-c", script, str(ROOT / "habitat-mas"), output],
                capture_output=True, text=True, check=True,
            )
        self.assertIn("detection", result.stderr)
        self.assertIn("step=451", result.stderr)
        self.assertIn("system_error=True", result.stderr)
        self.assertEqual(result.stderr.count("detection"), 1)

    def test_recovery_lifecycle_summaries_preserve_audit_without_payload_logging(self):
        cases = [
            ("detection", {"decision": {"system_error": False, "status": "normal"}}, "system_error=False"),
            ("generation_started", {"iteration": 1}, "iteration=1"),
            ("source_generation_started", {"iteration": 2}, "iteration=2"),
            ("generation_rejected", {"iteration": 1, "error": "invalid commands"}, "invalid commands"),
            ("source_generation_rejected", {"iteration": 2, "error": "bad patch sk-example_secret123\nSOURCE_PAYLOAD"}, "bad patch"),
            ("application", {"iteration": 1, "status": "applied", "result": {"source": "SOURCE_PAYLOAD"}}, "status=applied"),
            ("source_application", {"iteration": 2, "result": {"status": "applied", "result": "SOURCE_PAYLOAD"}}, "result_status=applied"),
            ("verification", {"iteration": 2, "status": "no_observed_progress"}, "status=no_observed_progress"),
            ("source_budget_exhausted", {"attempts": 3}, "attempts=3"),
        ]
        with tempfile.TemporaryDirectory() as output:
            monitor = RecoveryController(RecoveryConfig(), output, detector=object(), generator=object())
            monitor._last_step = 701
            for event, data, expected in cases:
                with self.subTest(event=event), self.assertLogs("habitat_mas.recovery", level="INFO") as captured:
                    monitor._event(event, **data, context="CONTEXT_PAYLOAD", prompt="PROMPT_PAYLOAD")
                message = "\n".join(captured.output)
                self.assertIn(event, message)
                self.assertIn("step=701", message)
                self.assertIn(expected, message)
                for forbidden in ("SOURCE_PAYLOAD", "CONTEXT_PAYLOAD", "PROMPT_PAYLOAD", "sk-example_secret123"):
                    self.assertNotIn(forbidden, message)
            rows = [json.loads(line) for line in (Path(output) / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), len(cases))
        self.assertEqual(rows[4]["error"], "bad patch [REDACTED]\nSOURCE_PAYLOAD")
        self.assertEqual(rows[6]["result"]["result"], "SOURCE_PAYLOAD")

    def test_combined_exhausted_budget_is_reported_once_without_another_request(self):
        with fixture() as (root, policy, runtime):
            transport = Transport({"system_error": "invalid response"})
            monitor = controller(root, runtime, transport, max_iterations=1)
            with self.assertLogs("habitat_mas.recovery", level="INFO") as captured:
                self.assertEqual(monitor.observe(frame(0), policy)["status"], "rejected")
                for step in range(1, 8):
                    self.assertIsNone(monitor.observe(frame(step), policy))
            rows = [json.loads(line) for line in (root / "audit" / "events.jsonl").read_text().splitlines()]
            exhausted = [row for row in rows if row["event"] == "source_budget_exhausted"]
            self.assertEqual(len(exhausted), 1)
            self.assertEqual(exhausted[0]["attempts"], 1)
            self.assertEqual(exhausted[0]["step"], 1)
            self.assertEqual(sum("source_budget_exhausted" in row for row in captured.output), 1)
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(len(monitor.versions), 1)

    def test_normal_combined_checks_keep_running_without_using_repair_budget(self):
        with fixture() as (root, policy, runtime):
            transport = Transport({"system_error": False}, {"system_error": False})
            monitor = controller(root, runtime, transport, max_iterations=1)
            for step in range(2):
                self.assertEqual(monitor.observe(frame(step), policy)["status"], "normal")
            rows = [json.loads(line) for line in (root / "audit" / "events.jsonl").read_text().splitlines()]
            self.assertFalse(any(row["event"] == "source_budget_exhausted" for row in rows))
            self.assertEqual(len(transport.requests), 2)
            self.assertEqual(monitor.versions, [])


if __name__ == "__main__":
    unittest.main()
