"""The source runtime pins validation before any automatic installation."""
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'habitat-mas'))

from habitat_mas.recovery.source_patch import SourcePatchManager
from habitat_mas.recovery.source_runtime import SourceRuntime


class RestrictedRuntimeTests(unittest.TestCase):
    def test_restricted_host_rejects_legacy_manifest_even_with_a_cached_validator(self):
        source = 'class Policy:\n    def __init__(self):\n        self.active = False\n'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'policy.py'
            path.write_text(source)
            name = '_restricted_runtime_' + uuid4().hex
            module = ModuleType(name)
            module.__file__ = str(path)
            exec(compile(source, str(path), 'exec'), module.__dict__)
            sys.modules[name] = module
            policy = module.Policy()
            targets = {'policy.py': {'module': name, 'classes': ['Policy'], 'process': 'parent'}}
            runtime = SourceRuntime(root=root, targets=targets)
            try:
                runtime._patcher()  # A previously cached legacy validator must not bypass the pin.
                runtime.restricted = True
                manager = SourcePatchManager(root, root / 'artifacts', targets=targets)
                transaction = manager.prepare({
                    'patches': [{'path': 'policy.py', 'target': 'Policy.Resume',
                                 'base_sha256': hashlib.sha256(source.encode()).hexdigest(),
                                 'code': 'def Resume(self, context):\n    self._metric = True\n'}],
                    'entrypoint': {'path': 'policy.py', 'target': 'Policy.Resume'},
                    'explanation': 'This unprofiled candidate must never execute automatically.'})
                with self.assertRaisesRegex(RuntimeError, 'restricted|profile'):
                    runtime.apply(transaction, policy, {})
                self.assertFalse(hasattr(policy, '_metric'))
                self.assertFalse(hasattr(policy, 'Resume'))
                self.assertEqual(path.read_text(), source)
            finally:
                runtime.close()
                sys.modules.pop(name, None)


if __name__ == '__main__':
    unittest.main()
