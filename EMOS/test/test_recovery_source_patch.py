"""Source transactions change existing Python objects without a simulator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
try:
    from habitat_mas.recovery import source_patch
except ImportError:
    source_patch = None


SOURCE = '''from __future__ import annotations

class Parent:
    def act(self, context):
        return "parent:" + context["value"]

class Agent(Parent):
    def act(self, context):
        return "original:" + context["value"]

    def configured(self, value: MissingType = object()) -> MissingType:
        return value
'''


class SourcePatchTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(source_patch, "source patch implementation is missing")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "agent.py"
        self.path.write_text(SOURCE)
        self.module_name = "_emos_patch_test_" + uuid4().hex
        spec = importlib.util.spec_from_file_location(self.module_name, self.path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self.module
        self.addCleanup(sys.modules.pop, self.module_name, None)
        spec.loader.exec_module(self.module)
        self.alias = self.module.Agent
        self.instance = self.alias()
        self.targets = {"agent.py": {"module": self.module_name,
            "classes": ["Agent"], "process": "parent"}}
        self.manager = source_patch.SourcePatchManager(
            self.root, self.root / "artifacts", targets=self.targets)
        self.runtime = source_patch.RuntimeSourcePatcher(self.root, targets=self.targets)

    def patch(self, target="Agent.act", code=None, path="agent.py", base=None):
        if code is None:
            code = 'def act(self, context):\n    return "patched:" + context["value"]\n'
        return {"path": path, "target": target,
            "base_sha256": base or hashlib.sha256((self.root / path).read_bytes()).hexdigest(),
            "code": code}

    def proposal(self, *patches, entrypoint=None):
        return {"patches": list(patches or [self.patch()]),
            "entrypoint": entrypoint or {"path": "agent.py", "target": "Agent.act"},
            "explanation": "Repair the live decision."}

    def test_source_commit_and_existing_instance_use_revised_original_method(self):
        transaction = self.manager.prepare(self.proposal())
        self.assertEqual(self.instance.act({"value": "cup"}), "original:cup")
        self.assertEqual(self.path.read_text(), SOURCE)
        result = self.runtime.install(transaction.manifest("parent"))
        transaction.commit_files()
        self.assertEqual(result["status"], "installed")
        self.assertIs(self.module.Agent, self.alias)
        self.assertEqual(self.instance.act({"value": "cup"}), "patched:cup")
        reloaded = {}
        exec(compile(self.path.read_text(), str(self.path), "exec"), reloaded)
        self.assertEqual(reloaded["Agent"]().act({"value": "cup"}), "patched:cup")
        self.assertTrue(list((self.root / "artifacts").rglob("*.diff")))
        json.dumps(transaction.files)

    def test_added_resume_is_callable_on_preexisting_instance_and_rollback_removes_it(self):
        code = 'def Resume(self, context):\n    self.resumed = context["value"]\n    return self.resumed\n'
        transaction = self.manager.prepare(self.proposal(
            self.patch("Agent.Resume", code),
            entrypoint={"path": "agent.py", "target": "Agent.Resume"}))
        self.runtime.install(transaction.manifest("parent"))
        transaction.commit_files()
        self.assertEqual(self.instance.Resume({"value": "live"}), "live")
        self.runtime.rollback(transaction.id)
        transaction.rollback_files()
        self.assertFalse(hasattr(self.instance, "Resume"))
        self.assertEqual(self.path.read_text(), SOURCE)

    def test_replacement_reuses_original_defaults_and_annotations_without_evaluating_them(self):
        original = self.alias.configured
        code = '''def configured(self, value: MissingType = object()) -> MissingType:
    return ("configured", value)
'''
        transaction = self.manager.prepare(self.proposal(self.patch(),
            self.patch("Agent.configured", code)))
        self.runtime.install(transaction.manifest("parent"))
        self.assertEqual(self.instance.configured()[0], "configured")
        self.assertIs(self.instance.configured()[1], original.__defaults__[0])
        self.assertEqual(self.alias.configured.__annotations__, original.__annotations__)

    def test_zero_argument_super_uses_original_owner_class(self):
        transaction = self.manager.prepare(self.proposal(self.patch(code=
            'def act(self, context):\n    return super().act(context) + ":recovered"\n')))
        self.runtime.install(transaction.manifest("parent"))
        self.assertEqual(self.instance.act({"value": "cup"}), "parent:cup:recovered")

    def test_original_multi_llm_act_nested_helper_can_be_replaced(self):
        relative = "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py"
        production = (ROOT / relative).read_text()
        self.path.write_text(production)
        self.targets["agent.py"]["classes"] = ["MultiLLMPolicy"]
        manager = source_patch.SourcePatchManager(self.root, self.root / "artifacts", targets=self.targets)
        import ast
        owner = next(node for node in ast.parse(production).body
                     if isinstance(node, ast.ClassDef) and node.name == "MultiLLMPolicy")
        method = next(node for node in owner.body
                      if isinstance(node, ast.FunctionDef) and node.name == "act")
        code = textwrap.dedent(ast.get_source_segment(production, method)).replace(
            'kwargs.get("save_chat_history", True)', 'kwargs.get("save_chat_history", False)')
        transaction = manager.prepare(self.proposal(
            self.patch("MultiLLMPolicy.act", code),
            self.patch("MultiLLMPolicy.Resume", "def Resume(self, context):\n    return context\n"),
            entrypoint={"path": "agent.py", "target": "MultiLLMPolicy.Resume"}))
        transaction.commit_files()
        self.assertIn('kwargs.get("save_chat_history", False)', self.path.read_text())

    def test_existing_classmethod_and_staticmethod_descriptors_survive_replacement(self):
        original = SOURCE + '''
    @staticmethod
    def utility(value):
        return value + 1

    @classmethod
    def create(cls, context):
        return cls().act(context)
'''
        self.path.write_text(original)
        exec(compile(original, str(self.path), "exec"), self.module.__dict__)
        transaction = self.manager.prepare(self.proposal(self.patch(),
            self.patch("Agent.utility", "@staticmethod\ndef utility(value):\n    return value + 2\n"),
            self.patch("Agent.create", "@classmethod\ndef create(cls, context):\n    return 'new:' + cls().act(context)\n")))
        self.runtime.install(transaction.manifest("parent"))
        self.assertEqual(self.module.Agent.utility(3), 5)
        self.assertEqual(self.module.Agent.create({"value": "x"}), "new:patched:x")

    def test_keyword_default_object_identity_is_preserved(self):
        original = SOURCE + '''
    def keyword(self, *, value=object()):
        return value
'''
        self.path.write_text(original)
        exec(compile(original, str(self.path), "exec"), self.module.__dict__)
        expected = self.module.Agent().keyword()
        transaction = self.manager.prepare(self.proposal(self.patch(), self.patch(
            "Agent.keyword", "def keyword(self, *, value=object()):\n    return ('new', value)\n")))
        self.runtime.install(transaction.manifest("parent"))
        self.assertIs(self.module.Agent().keyword()[1], expected)

    def test_class_private_names_keep_the_original_name_mangling(self):
        original = SOURCE + '''
    __secret = "kept"

    def private(self, context):
        return self.__secret
'''
        self.path.write_text(original)
        exec(compile(original, str(self.path), "exec"), self.module.__dict__)
        transaction = self.manager.prepare(self.proposal(self.patch(), self.patch(
            "Agent.private", 'def private(self, context):\n    return "new:" + self.__secret\n')))
        self.runtime.install(transaction.manifest("parent"))
        self.assertEqual(self.module.Agent().private({}), "new:kept")

    def test_late_bad_hash_rejects_entire_batch_without_writes_or_runtime_changes(self):
        other = self.root / "other.py"
        other.write_text("class Helper:\n    def act(self, context):\n        return 1\n")
        self.targets["other.py"] = {"module": "not_loaded", "classes": ["Helper"], "process": "worker"}
        self.manager = source_patch.SourcePatchManager(self.root, self.root / "artifacts", targets=self.targets)
        bad = self.patch("Helper.act", "def act(self, context):\n    return 2\n",
                         path="other.py", base="0" * 64)
        with self.assertRaisesRegex(ValueError, "hash|base"):
            self.manager.prepare(self.proposal(self.patch(), bad))
        self.assertEqual(self.path.read_text(), SOURCE)
        self.assertEqual(self.instance.act({"value": "cup"}), "original:cup")

    def test_commit_rechecks_all_hashes_before_any_file_write(self):
        transaction = self.manager.prepare(self.proposal())
        self.path.write_text(SOURCE + "\n# concurrent user edit\n")
        with self.assertRaisesRegex(ValueError, "hash|conflict|changed"):
            transaction.commit_files()
        self.assertIn("concurrent user edit", self.path.read_text())

    def test_partial_multifile_write_failure_restores_already_replaced_file(self):
        other = self.root / "other.py"
        other_source = "class Helper:\n    def act(self, context):\n        return 1\n"
        other.write_text(other_source)
        self.targets["other.py"] = {"module": "unloaded_helper", "classes": ["Helper"], "process": "worker"}
        manager = source_patch.SourcePatchManager(self.root, self.root / "artifacts", targets=self.targets)
        transaction = manager.prepare(self.proposal(self.patch(), self.patch(
            "Helper.act", "def act(self, context):\n    return 2\n", path="other.py")))
        real_replace = source_patch.os.replace
        def fail_second(source, destination):
            if Path(destination) == other:
                raise OSError("disk failure on second file")
            return real_replace(source, destination)
        with patch.object(source_patch.os, "replace", fail_second):
            with self.assertRaisesRegex(OSError, "disk failure"):
                transaction.commit_files()
        self.assertEqual(self.path.read_text(), SOURCE)
        self.assertEqual(other.read_text(), other_source)

    def test_rollback_refuses_to_overwrite_user_edits(self):
        transaction = self.manager.prepare(self.proposal())
        transaction.commit_files()
        self.path.write_text(self.path.read_text() + "\n# user edit\n")
        with self.assertRaisesRegex(ValueError, "hash|conflict|changed"):
            transaction.rollback_files()
        self.assertIn("user edit", self.path.read_text())

    def test_runtime_rollback_requires_reverse_installation_order(self):
        first = self.manager.prepare(self.proposal())
        self.runtime.install(first.manifest("parent"))
        first.commit_files()
        second = self.manager.prepare(self.proposal(self.patch(code=
            'def act(self, context):\n    return "second:" + context["value"]\n')))
        self.runtime.install(second.manifest("parent"))
        second.commit_files()
        with self.assertRaisesRegex(ValueError, "order|LIFO|latest"):
            self.runtime.rollback(first.id)
        self.assertEqual(self.instance.act({"value": "x"}), "second:x")
        self.runtime.rollback(second.id)
        second.rollback_files()
        self.assertEqual(self.instance.act({"value": "x"}), "patched:x")
        self.runtime.rollback(first.id)
        first.rollback_files()
        self.assertEqual(self.instance.act({"value": "x"}), "original:x")

    def test_unloaded_modules_are_reported_pending_without_importing_them(self):
        sys.modules.pop(self.module_name)
        transaction = self.manager.prepare(self.proposal())
        result = self.runtime.install(transaction.manifest("parent"))
        self.assertEqual(result["pending_modules"], [self.module_name])
        self.assertNotIn(self.module_name, sys.modules)

    def test_bad_late_runtime_target_leaves_earlier_target_unchanged(self):
        transaction = self.manager.prepare(self.proposal(self.patch(),
            self.patch("Agent.Resume", "def Resume(self, context):\n    return context\n")))
        manifest = transaction.manifest("parent")
        del self.module.Agent.act
        try:
            with self.assertRaises(ValueError):
                self.runtime.install(manifest)
            self.assertNotIn("Resume", vars(self.module.Agent))
        finally:
            exec(compile(SOURCE, str(self.path), "exec"), self.module.__dict__)

    def test_runtime_rejects_manifest_changes_outside_declared_methods(self):
        transaction = self.manager.prepare(self.proposal())
        manifest = transaction.manifest("parent")
        manifest["files"][0]["after_source"] += "\nUNDECLARED_CHANGE = True\n"
        manifest["files"][0]["after_sha256"] = hashlib.sha256(
            manifest["files"][0]["after_source"].encode()).hexdigest()
        with self.assertRaisesRegex(ValueError, "declared|method|structured"):
            self.runtime.install(manifest)
        self.assertEqual(self.instance.act({"value": "x"}), "original:x")

    def test_rejects_signature_changes_duplicate_targets_and_bad_entrypoints(self):
        invalids = [
            self.proposal(self.patch(code="def act(self, context, extra=None):\n    return context\n")),
            self.proposal(self.patch(), self.patch()),
            self.proposal(entrypoint={"path": "agent.py", "target": "Agent.missing"}),
            self.proposal(self.patch("Agent.Resume", "def Resume(self):\n    return None\n"),
                entrypoint={"path": "agent.py", "target": "Agent.Resume"}),
            self.proposal(self.patch("Agent.Resume", "@staticmethod\ndef Resume(context):\n    return context\n"),
                entrypoint={"path": "agent.py", "target": "Agent.Resume"}),
        ]
        for proposal in invalids:
            with self.subTest(proposal=proposal), self.assertRaises(ValueError):
                self.manager.prepare(proposal)
        self.assertEqual(self.path.read_text(), SOURCE)

    def test_rejects_escapes_symlinks_nonallowlisted_classes_and_lifecycle_methods(self):
        link = self.root / "linked.py"
        link.symlink_to(self.path)
        self.targets["linked.py"] = self.targets["agent.py"]
        manager = source_patch.SourcePatchManager(self.root, self.root / "artifacts", targets=self.targets)
        invalids = [
            self.patch(path="../escape.py", base="0" * 64),
            self.patch(path=str(self.path), base="0" * 64),
            self.patch(path="linked.py"),
            self.patch("Parent.act"),
            self.patch("Agent.reset", "def reset(self, context):\n    return context\n"),
            self.patch("Agent.__call__", "def __call__(self, context):\n    return context\n"),
            self.patch("Agent.get_metrics", "def get_metrics(self, context):\n    return {}\n"),
            self.patch("Agent.apply_runtime_source_patch", "def apply_runtime_source_patch(self, context):\n    return {}\n"),
            self.patch("Agent.rollback_runtime_source_patch", "def rollback_runtime_source_patch(self, context):\n    return {}\n"),
            self.patch("Agent.get_trajectory_state", "def get_trajectory_state(self, context):\n    return {}\n"),
            self.patch("Agent._recovery_agents", "def _recovery_agents(self, context):\n    return {}\n"),
            self.patch("Agent.validate_recovery_commands", "def validate_recovery_commands(self, context):\n    return {}\n"),
        ]
        for value in invalids:
            with self.subTest(patch=value), self.assertRaises(ValueError):
                manager.prepare(self.proposal(value,
                    entrypoint={"path": value["path"], "target": value["target"]}))

    def test_generated_code_cannot_add_import_file_network_or_dynamic_execution(self):
        bodies = ["import os\n    return context", "return open('/tmp/private').read()",
            "return eval('1')", "return __import__('os')", "return self.client.get('https://example.org')",
            "return self.path.write_text('changed')", "return self.__class__.__mro__",
            "return getattr(self, 'act')(context)", "return self.env.reset()"]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.manager.prepare(self.proposal(self.patch(code="def act(self, context):\n    " + body + "\n")))

    def test_manifest_partition_and_source_snapshot_have_exact_current_hashes(self):
        snapshot = source_patch.source_snapshot(self.root, targets=self.targets)
        self.assertEqual(snapshot["agent.py"]["content"], SOURCE)
        self.assertEqual(snapshot["agent.py"]["sha256"], hashlib.sha256(SOURCE.encode()).hexdigest())
        transaction = self.manager.prepare(self.proposal())
        self.assertEqual(transaction.manifest("worker")["files"], [])
        self.assertEqual(transaction.entrypoint, {"path": "agent.py", "target": "Agent.act"})
        self.assertEqual(self.manager.current_manifest("parent")["transaction_id"], transaction.id)


if __name__ == "__main__":
    unittest.main()
