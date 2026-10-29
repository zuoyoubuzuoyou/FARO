"""The real request boundary sends method views and retains full local audit."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.context_compaction import expand_runtime_context
from habitat_mas.recovery.model_input import prepare_model_context
from habitat_mas.recovery.source_generator import SourceRecoveryGenerator
from test_recovery_combined import Transport, controller
from test_recovery_source_controller import ORIGINAL, fixture, frame


class ModelInputTests(unittest.TestCase):
    def test_prepared_method_view_is_idempotent_and_rejects_mixed_source_views(self):
        context = {"code": {"source_files": {"policy.py": {
            "content": ORIGINAL, "sha256": hashlib.sha256(ORIGINAL.encode()).hexdigest(),
            "classes": ["MultiLLMPolicy"], "process": "parent", "module": "fixture"}}}}
        prepared = prepare_model_context(context)
        self.assertEqual(prepare_model_context(prepared), prepared)
        prepared["code"]["source_files"]["other.py"] = {"content": "x = 1", "sha256": "a" * 64}
        with self.assertRaisesRegex(ValueError, "mix"):
            prepare_model_context(prepared)

    def test_unknown_partial_source_view_is_not_treated_as_a_complete_file(self):
        context = {"code": {"source_files": {"policy.py": {
            "content": ORIGINAL, "sha256": hashlib.sha256(ORIGINAL.encode()).hexdigest(),
            "view": "partial-lines-v0"}}}}
        with self.assertRaisesRegex(ValueError, "view"):
            prepare_model_context(context)

    def test_request_uses_method_view_and_exact_dedup_without_mutating_original(self):
        source = ORIGINAL + '\ndef unrelated_helper():\n    return "UNRELATED_BODY_MARKER"\n'
        current = {"step": 451, "agents": {}, "current_goal": {"seen": False, "at": None}}
        context = {"task": {"description": "Observe the target"},
                   "runtime": {"current_frame": current, "recent_frames": [deepcopy(current)]},
                   "code": {"source_files": {"policy.py": {"content": source,
                       "sha256": hashlib.sha256(source.encode()).hexdigest(),
                       "classes": ["MultiLLMPolicy"], "process": "parent", "module": "fixture"}}}}
        before = deepcopy(context)
        transport = Transport({"system_error": False})

        SourceRecoveryGenerator(client=transport).detect_and_repair(context)

        payload = json.loads(transport.requests[0]["messages"][1]["content"])
        viewed = payload["code"]["source_files"]["policy.py"]
        self.assertEqual(viewed["view"], "methods-v1")
        self.assertEqual(viewed["sha256"], context["code"]["source_files"]["policy.py"]["sha256"])
        self.assertNotIn("UNRELATED_BODY_MARKER", viewed["content"])
        self.assertIn("def act(self)", viewed["content"])
        self.assertNotIn("current_frame", payload["runtime"])
        self.assertEqual(expand_runtime_context(payload)["runtime"], context["runtime"])
        self.assertEqual(context, before)

    def test_negative_check_archives_original_full_context_outside_model_payload(self):
        with fixture() as (root, policy, runtime):
            transport = Transport({"system_error": False})
            monitor = controller(root, runtime, transport)
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "normal")
            audit = json.loads((root / "audit/source_requests.jsonl").read_text().splitlines()[0])
            self.assertIn("input_snapshot", audit)
            with gzip.open(root / "audit" / audit["input_snapshot"], "rt", encoding="utf-8") as stream:
                full = json.load(stream)
            self.assertEqual(full["code"]["source_files"]["policy.py"]["content"], ORIGINAL)
            payload = json.loads(transport.requests[0]["messages"][1]["content"])
            self.assertEqual(payload["code"]["source_files"]["policy.py"]["view"], "methods-v1")
            self.assertEqual(expand_runtime_context(payload)["runtime"], full["runtime"])
            self.assertNotIn("input_snapshot", payload)
            self.assertNotIn("full_context", payload)
            self.assertEqual(monitor.versions, [])

    def test_failed_request_keeps_full_snapshot_and_exact_compact_request(self):
        with fixture() as (root, policy, runtime):
            transport = Transport(TimeoutError("offline timeout"))
            monitor = controller(root, runtime, transport)
            self.assertEqual(monitor.observe(frame(0), policy)["status"], "rejected")
            version = json.loads((root / "audit/versions/001/version.json").read_text())
            self.assertIn("input_snapshot", version)
            with gzip.open(root / "audit" / version["input_snapshot"], "rt", encoding="utf-8") as stream:
                full = json.load(stream)
            error = json.loads((root / "audit/versions/001/generation_error.json").read_text())
            self.assertEqual(error["request"], transport.requests[0])
            payload = json.loads(error["request"]["messages"][1]["content"])
            self.assertEqual(expand_runtime_context(payload)["runtime"], full["runtime"])
            self.assertEqual(full["code"]["source_files"]["policy.py"]["content"], ORIGINAL)


if __name__ == "__main__":
    unittest.main()
