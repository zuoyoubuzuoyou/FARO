"""The default recovery run tries one repair and still evaluates its outcome."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.source_controller import SourceRecoveryController
from habitat_mas.recovery.source_generator import SourceRecoveryGenerator
from test_recovery_combined import Transport, repair
from test_recovery_source_controller import fixture, frame


class SingleRepairTests(unittest.TestCase):
    def monitor(self, root, runtime, transport):
        return SourceRecoveryController(RecoveryConfig(
            sample_every=1, window=1, check_every=1,
            cooldown_steps=1, verify_after_steps=250), root / "audit", runtime=runtime,
            generator=SourceRecoveryGenerator(client=transport))

    def test_default_applies_once_then_verifies_without_requesting_another_repair(self):
        with fixture() as (root, policy, runtime):
            transport = Transport(repair)
            monitor = self.monitor(root, runtime, transport)
            self.assertEqual(monitor.observe(frame(451), policy)["status"], "applied")
            self.assertIsNone(monitor.observe(frame(701), policy))
            self.assertIsNone(monitor.observe(frame(951), policy))
            rows = [json.loads(line) for line in (root / "audit/events.jsonl").read_text().splitlines()]
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(len(monitor.versions), 1)
            self.assertEqual(policy.resume_count, 1)
            verification = [row for row in rows if row["event"] == "verification"]
            self.assertEqual(len(verification), 1)
            self.assertEqual(verification[0]["step"], 701)
            self.assertEqual(verification[0]["status"], "no_observed_progress")
            self.assertEqual(sum(row["event"] == "source_budget_exhausted" for row in rows), 1)

    def test_default_unrepairable_response_is_not_followed_by_another_attempt(self):
        with fixture() as (root, policy, runtime):
            transport = Transport({"system_error": True, "patches": [], "entrypoint": None})
            monitor = self.monitor(root, runtime, transport)
            self.assertEqual(monitor.observe(frame(451), policy)["status"], "unrepairable")
            self.assertIsNone(monitor.observe(frame(701), policy))
            self.assertEqual(len(transport.requests), 1)
            self.assertEqual(len(monitor.versions), 1)
            self.assertEqual(policy.resume_count, 0)


if __name__ == "__main__":
    unittest.main()
