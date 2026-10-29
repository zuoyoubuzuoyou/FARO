"""Source Resume can repair declared transient skill state without changing goals."""
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


RESUME = '''def Resume(self, context):
    for child in self._active_policies:
        child._cur_call_high_level[0] = True
        for skill in child._skills.values():
            skill._cur_skill_step[:] = 0
            skill._was_running_on_prev_step = False
'''


class ExecutionResumeTests(unittest.TestCase):
    def prepare(self, root, runtime, code):
        manager = SourcePatchManager(root, root / "audit", targets=runtime.targets,
                                     restricted=True)
        return manager.prepare({"patches": [{"path": "policy.py",
            "target": "MultiLLMPolicy.Resume",
            "base_sha256": hashlib.sha256((root / "policy.py").read_bytes()).hexdigest(),
            "code": code}], "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"},
            "explanation": "Clear transient state and return control without changing the task."})

    def test_resume_clears_original_skill_counters_and_preserves_arguments_and_model(self):
        with fixture() as (root, policy, runtime):
            runtime.require_restricted_patches()
            with torch.inference_mode():
                counter = torch.tensor([9.0])
                flags = torch.tensor([False])
            arguments = {"target_obj": "original target"}
            model = object()
            skill = SimpleNamespace(_cur_skill_step=counter, _was_running_on_prev_step=True,
                                    _cur_skill_args=arguments)
            child = SimpleNamespace(_cur_call_high_level=flags, _skills={0: skill},
                                    llm_model=model)
            policy._active_policies = [child]

            result = runtime.apply(self.prepare(root, runtime, RESUME), policy, {})

            self.assertEqual(result["status"], "applied")
            self.assertEqual(counter.tolist(), [0.0])
            self.assertIs(skill._cur_skill_step, counter)
            self.assertIs(skill._cur_skill_args, arguments)
            self.assertEqual(arguments, {"target_obj": "original target"})
            self.assertIs(child.llm_model, model)
            self.assertFalse(skill._was_running_on_prev_step)
            self.assertEqual(flags.tolist(), [True])

    def test_resume_rejects_goal_selection_physics_and_fabricated_completion_writes(self):
        extra = ["skill._cur_skill_args = {}", "skill._cur_skill_step[:] = 9999",
                 "skill._sim = None", "skill.skill_done = True",
                 "skill._wait_time = 0", "child._cur_skills[:] = 2"]
        for statement in extra:
            with self.subTest(statement=statement), fixture() as (root, _, runtime):
                with self.assertRaises(ValueError):
                    self.prepare(root, runtime, RESUME + "            " + statement + "\n")


if __name__ == "__main__":
    unittest.main()
