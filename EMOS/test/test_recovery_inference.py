"""Sampled source recovery must handle the policy's real inference tensors."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.source_patch import SourcePatchManager
from test_recovery_source_controller import fixture


RESUME = """def Resume(self, context):
    self.source_resume_count = 1
    for child in self._active_policies:
        child._cur_call_high_level[0] = True
"""


class InferenceResumeTests(unittest.TestCase):
    def prepare(self, root, runtime, code):
        runtime.require_restricted_patches()
        path = root / "policy.py"
        manager = SourcePatchManager(root, root / "artifacts", targets=runtime.targets,
                                     restricted=True)
        return manager.prepare({
            "patches": [{"path": "policy.py", "target": "MultiLLMPolicy.Resume",
                         "base_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "code": code}],
            "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"},
            "explanation": "Request the next high-level decision on the existing policy."})

    def add_inference_flag(self, policy):
        with torch.inference_mode():
            flag = torch.zeros(1, dtype=torch.bool)
        policy._active_policies = [SimpleNamespace(_cur_call_high_level=flag)]
        self.assertTrue(torch.is_inference(flag))
        self.assertFalse(torch.is_inference_mode_enabled())
        return flag

    def test_sampled_resume_updates_original_inference_flag_outside_policy_call(self):
        with fixture() as (root, policy, runtime):
            original = (root / "policy.py").read_bytes()
            flag = self.add_inference_flag(policy)
            history = policy.history
            transaction = self.prepare(root, runtime, RESUME)

            result = runtime.apply(transaction, policy, {})

            self.assertEqual(result["status"], "applied")
            self.assertIs(policy._active_policies[0]._cur_call_high_level, flag)
            self.assertEqual(flag.tolist(), [True])
            self.assertEqual(policy.source_resume_count, 1)
            self.assertIs(policy.history, history)
            self.assertFalse(torch.is_inference_mode_enabled())
            runtime.close()
            self.assertEqual((root / "policy.py").read_bytes(), original)
            self.assertFalse(hasattr(policy, "Resume"))

    def test_failed_resume_restores_inference_flag_and_source_without_fatal_state(self):
        with fixture() as (root, policy, runtime):
            original = (root / "policy.py").read_bytes()
            flag = self.add_inference_flag(policy)
            transaction = self.prepare(root, runtime, RESUME + '    raise ValueError("resume failed")\n')

            with self.assertRaisesRegex(RuntimeError, "resume failed"):
                runtime.apply(transaction, policy, {})

            self.assertIsNone(runtime._consistency_error)
            self.assertEqual(flag.tolist(), [False])
            self.assertIs(policy._active_policies[0]._cur_call_high_level, flag)
            self.assertFalse(hasattr(policy, "source_resume_count"))
            self.assertFalse(hasattr(policy, "Resume"))
            self.assertEqual((root / "policy.py").read_bytes(), original)
            self.assertFalse(torch.is_inference_mode_enabled())


if __name__ == "__main__":
    unittest.main()
