"""Original-source revisions: actual methods, same initialized policy instance."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.source_controller import SourceRecoveryController
from habitat_mas.recovery.source_generator import SourceRecoveryGenerator
from habitat_mas.recovery.source_runtime import SourceRuntime


ORIGINAL = '''class MultiLLMPolicy:
    def __init__(self):
        self.active = False
        self.resume_count = 0
        self.history = [[{"role": "user", "content": str(i)}] for i in range(12)]
        self.held_object = 42

    def act(self):
        if not self.active:
            raise RuntimeError("no resume mechanism")
        return "move"

    def get_recovery_context(self, history_turns=10):
        return {"agents": {"agent_0": {"planner_plan": "go to target", "recent_history": self.history[-history_turns:], "available_tools": []}}}
'''

RESUME = '''def Resume(self, context):
    self.active = True
    self.resume_count += 1
    return {"resumed": True}
'''


def frame(step):
    return {"step": step, "task": "Mobility", "context": {"task_description": "Reach target"},
            "agents": {"agent_0": {"position": [0, 0, 0], "held_object": 42}},
            "objects": {}, "metrics": {"distance_to_goal": 10}}


class Detector:
    def __init__(self):
        self.calls = []

    def observe(self, value):
        self.calls.append(value["step"])
        return {"status": "system_error", "reason_code": "stalled", "evidence": [
            {"step": value["step"], "observation": "waiting"}]}


class Revisions:
    def __init__(self, *methods):
        self.methods = iter(methods)
        self.contexts = []

    def generate(self, context):
        self.contexts.append(context)
        methods = next(self.methods)
        current = context["code"]["source_files"]["policy.py"]
        return {"patches": [{"path": "policy.py", "target": target,
                              "base_sha256": current["sha256"], "code": code}
                             for target, code in methods],
                "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"},
                "explanation": "add Resume and improve original control code"}


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory() as output:
        root = Path(output)
        path = root / "policy.py"
        path.write_text(ORIGINAL)
        name = "recovery_fixture_" + uuid4().hex
        module = ModuleType(name)
        module.__file__ = str(path)
        exec(compile(ORIGINAL, str(path), "exec"), module.__dict__)
        sys.modules[name] = module
        targets = {"policy.py": {"module": name, "classes": ["MultiLLMPolicy"], "process": "parent"}}
        runtime = SourceRuntime(root=root, targets=targets)
        try:
            yield root, module.MultiLLMPolicy(), runtime
        finally:
            runtime.close()
            sys.modules.pop(name, None)


class SourceGeneratorTests(unittest.TestCase):
    def test_model_must_return_original_source_edits_instead_of_command_function(self):
        requests = []
        proposal = {"patches": [{"path": "policy.py", "target": "MultiLLMPolicy.Resume",
            "base_sha256": "a" * 64, "code": RESUME}],
            "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"}, "explanation": "new resume"}
        def create(**request):
            requests.append(request)
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(proposal)}}]}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = SourceRecoveryGenerator(client=client).generate({"code": {"source_files": {}}})
        self.assertEqual(result["patches"], proposal["patches"])
        self.assertIn("base_sha256", requests[0]["messages"][0]["content"])
        self.assertIn("原始", requests[0]["messages"][0]["content"])

    def test_legacy_extension_output_is_rejected_and_preserved(self):
        def create(**request):
            return {"choices": [{"message": {"content": json.dumps({"code": "def recover(context,memory): pass", "explanation": "wrong mode"})}}]}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaises(ValueError) as caught:
            SourceRecoveryGenerator(client=client).generate({})
        self.assertIsNotNone(caught.exception.response)


class SourceControllerTests(unittest.TestCase):
    def controller(self, root, runtime, generator, **values):
        config = RecoveryConfig(mode="source", sample_every=1, cooldown_steps=1, verify_after_steps=1, **values)
        return SourceRecoveryController(config, root / "artifacts", runtime=runtime,
            detector=Detector(), generator=generator)

    def test_generated_resume_is_added_to_original_class_and_invoked_on_same_instance(self):
        with fixture() as (root, policy, runtime):
            self.assertFalse(hasattr(policy, "Resume"))
            with self.assertRaises(RuntimeError):
                policy.act()
            history = policy.history
            generator = Revisions([("MultiLLMPolicy.Resume", RESUME)])
            controller = self.controller(root, runtime, generator)
            result = controller.observe(frame(0), policy)
            self.assertEqual(result["status"], "applied")
            self.assertEqual(policy.act(), "move")
            self.assertIs(policy.history, history)
            self.assertEqual(policy.held_object, 42)
            self.assertEqual(policy.resume_count, 1)
            self.assertIn("def Resume", (root / "policy.py").read_text())
            self.assertTrue(list((root / "artifacts").rglob("*.diff")))

    def test_second_revision_changes_original_action_method_and_sees_updated_source(self):
        replacement = 'def act(self):\n    return "fly"\n'
        with fixture() as (root, policy, runtime):
            generator = Revisions([("MultiLLMPolicy.Resume", RESUME)],
                [("MultiLLMPolicy.Resume", RESUME), ("MultiLLMPolicy.act", replacement)])
            controller = self.controller(root, runtime, generator)
            controller.observe(frame(0), policy)
            controller.observe(frame(1), policy)
            self.assertEqual(policy.act(), "fly")
            self.assertIn("def Resume", generator.contexts[1]["code"]["source_files"]["policy.py"]["content"])
            self.assertNotEqual(generator.contexts[0]["code"]["source_files"]["policy.py"]["sha256"],
                                generator.contexts[1]["code"]["source_files"]["policy.py"]["sha256"])
            self.assertEqual(len(generator.contexts[1]["recent_history"]["agent_0"]), 10)
            self.assertEqual(generator.contexts[1]["planner_plan"]["agent_0"], "go to target")

    def test_entrypoint_exception_restores_original_source_and_logical_state(self):
        failing = 'def Resume(self, context):\n    self.active = True\n    raise RuntimeError("bad revision")\n'
        with fixture() as (root, policy, runtime):
            controller = self.controller(root, runtime, Revisions([("MultiLLMPolicy.Resume", failing)]))
            result = controller.observe(frame(0), policy)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual((root / "policy.py").read_text(), ORIGINAL)
            self.assertFalse(hasattr(policy, "Resume"))
            self.assertFalse(policy.active)

    def test_exception_before_first_completed_step_can_generate_resume_without_fake_step(self):
        with fixture() as (root, policy, runtime):
            controller = self.controller(root, runtime, Revisions([("MultiLLMPolicy.Resume", RESUME)]))
            result = controller.handle_execution_error(frame(0), policy, ValueError("parser failed"))
            self.assertEqual(result["status"], "applied")
            self.assertEqual(controller.detector.calls, [])
            self.assertEqual(policy.act(), "move")
            self.assertEqual(controller.versions[0]["step"], 0)

    def test_failed_source_revision_is_rolled_back_before_next_exception_revision(self):
        failing_act = 'def act(self):\n    raise RuntimeError("new method broke")\n'
        fixed_act = 'def act(self):\n    return "repaired"\n'
        with fixture() as (root, policy, runtime):
            generator = Revisions([("MultiLLMPolicy.Resume", RESUME), ("MultiLLMPolicy.act", failing_act)],
                [("MultiLLMPolicy.Resume", RESUME), ("MultiLLMPolicy.act", fixed_act)])
            controller = self.controller(root, runtime, generator)
            controller.observe(frame(0), policy)
            result = controller.handle_execution_error(frame(0), policy, RuntimeError("new method broke"))
            self.assertEqual(result["status"], "applied")
            self.assertEqual(controller.versions[0]["status"], "rolled_back")
            self.assertEqual(policy.act(), "repaired")
            self.assertEqual(generator.contexts[1]["code"]["source_files"]["policy.py"]["content"], ORIGINAL)

    def test_exception_budget_and_terminal_boundary_prevent_extra_generation(self):
        with fixture() as (root, policy, runtime):
            generator = Revisions([("MultiLLMPolicy.Resume", RESUME)])
            controller = self.controller(root, runtime, generator, max_iterations=1)
            controller.observe(frame(0), policy, terminal=True)
            self.assertEqual(generator.contexts, [])
            controller.handle_execution_error(frame(0), policy, RuntimeError("failed"))
            result = controller.handle_execution_error(frame(0), policy, RuntimeError("failed again"))
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertEqual(len(generator.contexts), 1)

    def test_first_exception_retries_a_rejected_revision_without_waiting_for_another_step(self):
        with fixture() as (root, policy, runtime):
            generator = Revisions([("MultiLLMPolicy.Resume", "def wrong(self, context):\n    return {}\n")],
                                  [("MultiLLMPolicy.Resume", RESUME)])
            controller = self.controller(root, runtime, generator, max_iterations=2)
            result = controller.handle_execution_error(frame(0), policy, ValueError("parser failed"))
            self.assertEqual(result["status"], "applied")
            self.assertEqual([version["status"] for version in controller.versions], ["rejected", "active"])
            self.assertEqual(policy.act(), "move")
            self.assertEqual(controller.detector.calls, [])

    def test_failed_generated_resume_is_revised_before_retrying_the_original_decision(self):
        bad = 'def Resume(self, context):\n    self.active = True\n    raise ValueError("wrong branch")\n'
        with fixture() as (root, policy, runtime):
            generator = Revisions([("MultiLLMPolicy.Resume", bad)], [("MultiLLMPolicy.Resume", RESUME)])
            controller = self.controller(root, runtime, generator, max_iterations=2)
            result = controller.handle_execution_error(frame(0), policy, ValueError("parser failed"))
            self.assertEqual(result["status"], "applied")
            self.assertEqual(policy.resume_count, 1)
            self.assertIn("wrong branch", generator.contexts[1]["recovery"]["previous_versions"][0]["error"])

    def test_post_install_audit_failure_propagates_instead_of_reporting_rejection(self):
        with fixture() as (root, policy, runtime):
            controller = self.controller(root, runtime, Revisions([("MultiLLMPolicy.Resume", RESUME)]))
            event = controller._event
            def record(kind, **data):
                if kind == "source_application":
                    raise OSError("audit storage unavailable")
                return event(kind, **data)
            with patch.object(controller, "_event", side_effect=record):
                with self.assertRaisesRegex(OSError, "audit storage unavailable"):
                    controller.observe(frame(0), policy)
            self.assertEqual(controller.versions[0]["status"], "active")


if __name__ == "__main__":
    unittest.main()
