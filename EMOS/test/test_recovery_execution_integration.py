"""Execution permissions are enforced at prepare and live installation."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest
from uuid import uuid4

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.source_patch import (
    RESTRICTED_SOURCE_POLICY, RuntimeSourcePatcher, SourcePatchManager,
)
from test_recovery_source_controller import fixture


WAIT = "habitat-baselines/habitat_baselines/rl/hrl/skills/wait.py"


class ExecutionBoundaryTests(unittest.TestCase):
    def setup_wait(self, root, runtime):
        source = (ROOT / WAIT).read_text().replace(
            "self._cur_skill_step >= self._wait_time", "self._cur_skill_step > self._wait_time")
        path = root / WAIT
        path.parent.mkdir(parents=True)
        path.write_text(source)
        name = "_execution_wait_" + uuid4().hex
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        self.addCleanup(sys.modules.pop, name, None)
        spec.loader.exec_module(module)
        targets = dict(runtime.targets)
        targets[WAIT] = {"module": name, "classes": ["WaitSkillPolicy"], "process": "parent"}
        tree = ast.parse(source)
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_is_skill_done")
        return path, source, targets, method, module

    def proposal(self, root, source, method):
        return {"patches": [
            {"path": "policy.py", "target": "MultiLLMPolicy.Resume",
             "base_sha256": hashlib.sha256((root / "policy.py").read_bytes()).hexdigest(),
             "code": "def Resume(self, context):\n    pass\n"},
            {"path": WAIT, "target": "WaitSkillPolicy._is_skill_done",
             "base_sha256": hashlib.sha256(source.encode()).hexdigest(), "code": ast.unparse(method)}],
            "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"},
            "explanation": "Correct a wait boundary while preserving its requested duration."}

    def test_wait_boundary_fix_installs_on_original_class_and_preserves_duration(self):
        with fixture() as (root, _, runtime):
            path, source, targets, method, module = self.setup_wait(root, runtime)
            skill = module.WaitSkillPolicy.__new__(module.WaitSkillPolicy)
            skill._wait_time = 5
            skill._cur_skill_step = torch.tensor([5])
            args = ({}, None, None, None, [0])
            self.assertEqual(skill._is_skill_done(*args).tolist(), [False])
            for node in ast.walk(method):
                if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Attribute)
                        and node.left.attr == "_cur_skill_step"):
                    node.ops = [ast.GtE() if isinstance(op, ast.Gt) else op for op in node.ops]
            manager = SourcePatchManager(root, root / "audit", targets=targets, restricted=True)
            transaction = manager.prepare(self.proposal(root, source, method))
            patcher = RuntimeSourcePatcher(root, targets=targets, restricted=True)
            try:
                result = patcher.install(transaction.manifest("parent"))
                self.assertEqual(result["status"], "installed")
                self.assertEqual(skill._is_skill_done(*args).tolist(), [True])
                skill._cur_skill_step[0] = 4
                self.assertEqual(skill._is_skill_done(*args).tolist(), [False])
                self.assertEqual(skill._wait_time, 5)
            finally:
                if patcher._stack:
                    patcher.rollback(transaction.id)
            skill._cur_skill_step[0] = 5
            self.assertEqual(skill._is_skill_done(*args).tolist(), [False])
            self.assertEqual(path.read_text(), source)

    def test_constant_completion_is_rejected_by_manager_and_forged_manifest_install(self):
        with fixture() as (root, _, runtime):
            _, source, targets, method, _ = self.setup_wait(root, runtime)
            method.body[-1] = ast.Return(value=ast.Constant(value=True))
            proposal = self.proposal(root, source, method)
            with self.assertRaises(ValueError):
                SourcePatchManager(root, root / "audit", targets=targets,
                                   restricted=True).prepare(proposal)
            transaction = SourcePatchManager(root, root / "forged", targets=targets).prepare(proposal)
            manifest = transaction.manifest("parent")
            manifest["validation_profile"] = RESTRICTED_SOURCE_POLICY["profile"]
            with self.assertRaises(ValueError):
                RuntimeSourcePatcher(root, targets=targets, restricted=True).install(manifest)


if __name__ == "__main__":
    unittest.main()
