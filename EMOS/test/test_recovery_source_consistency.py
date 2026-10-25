"""A source rollback conflict must halt recovery before changing other components."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.source_controller import SourceRecoveryController
from habitat_mas.recovery.source_patch import RuntimeSourcePatcher, SourcePatchManager
from habitat_mas.recovery.source_runtime import ExecutionCheckpoint, SourceRuntime


POLICY = '''class Policy:
    def __init__(self):
        self.history = []

    def act(self):
        return "original"

    def get_recovery_context(self, history_turns=10):
        return {"agents": {}}
'''
WORKER = '''class Worker:
    def act(self):
        return "original worker"
'''
RESUME = 'def Resume(self, context):\n    self.history.append("resumed")\n    return True\n'


class WorkerConnection:
    def __init__(self, patcher):
        self.patcher = patcher
        self.calls = []
        self.fail_check = False
        self.fail_rollback = False

    def call_at(self, index, name, args):
        self.calls.append(name)
        if name == "apply_runtime_source_patch":
            return self.patcher.install(args["manifest"])
        if name == "check_runtime_source_rollback":
            if self.fail_check:
                return {"status": "error", "error": "worker preflight conflict"}
            return self.patcher.check_rollback(args["transaction_id"])
        if self.fail_rollback:
            return {"status": "error", "error": "worker rollback failed"}
        return self.patcher.rollback(args["transaction_id"])


@contextmanager
def fixture(worker=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        targets, modules = {}, []
        for filename, source, owner, process in [
                ("policy.py", POLICY, "Policy", "parent"),
                ("worker.py", WORKER, "Worker", "worker")]:
            path = root / filename
            path.write_text(source)
            module = ModuleType("_source_consistency_" + uuid4().hex)
            module.__file__ = str(path)
            exec(compile(source, str(path), "exec"), module.__dict__)
            sys.modules[module.__name__] = module
            modules.append(module)
            targets[filename] = {"module": module.__name__, "classes": [owner], "process": process}
        connection = WorkerConnection(RuntimeSourcePatcher(root, targets)) if worker else None
        runtime = SourceRuntime(root=root, targets=targets, envs=connection)
        manager = SourcePatchManager(root, root / "patches", targets)
        try:
            yield root, modules[0].Policy(), runtime, manager, connection
        finally:
            for module in modules:
                sys.modules.pop(module.__name__, None)


def proposal(root, resume=RESUME, worker=False):
    changes = [("policy.py", "Policy.Resume", resume),
               ("policy.py", "Policy.act", 'def act(self):\n    return "revised"\n')]
    if worker:
        changes.append(("worker.py", "Worker.act", 'def act(self):\n    return "revised worker"\n'))
    return {"explanation": "repair the original runtime methods",
            "entrypoint": {"path": "policy.py", "target": "Policy.Resume"},
            "patches": [{"path": path, "target": target,
                         "base_sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
                         "code": code} for path, target, code in changes]}


class SourceConsistencyTests(unittest.TestCase):
    def assertFatal(self, error):
        self.assertEqual(type(error).__name__, "SourceConsistencyError")
        self.assertTrue(error.rollback_failed)

    def test_checkpoint_restores_numpy_scalar_attributes_without_array_assignment(self):
        import numpy as np
        from types import SimpleNamespace
        policy = SimpleNamespace(counter=np.int32(4), distance=np.float64(1.5),
                                 positions=np.array([2, 3]))
        checkpoint = ExecutionCheckpoint(policy)
        policy.counter = np.int32(9)
        policy.distance = np.float64(7)
        policy.positions[:] = 99
        checkpoint.restore()
        self.assertEqual(policy.counter, np.int32(4))
        self.assertEqual(policy.distance, np.float64(1.5))
        self.assertEqual(policy.positions.tolist(), [2, 3])

    def test_file_conflict_is_checked_before_restoring_live_methods(self):
        with fixture() as (root, policy, runtime, manager, _):
            transaction = manager.prepare(proposal(root))
            runtime.apply(transaction, policy, {})
            installed = type(policy).act
            user_source = (root / "policy.py").read_text() + "# concurrent user edit\n"
            (root / "policy.py").write_text(user_source)
            policy.history.append("real progress")
            with self.assertRaises(RuntimeError) as caught:
                runtime.rollback(transaction)
            self.assertIs(type(policy).act, installed)
            self.assertEqual(policy.act(), "revised")
            self.assertEqual((root / "policy.py").read_text(), user_source)
            self.assertEqual(policy.history, ["resumed", "real progress"])
            self.assertFatal(caught.exception)
            self.assertEqual(runtime._active[-1][0], transaction)
            with self.assertRaises(RuntimeError) as repeated:
                runtime.apply(transaction, policy, {})
            self.assertFatal(repeated.exception)

    def test_parent_method_conflict_prevents_worker_and_disk_rollback(self):
        with fixture(worker=True) as (root, policy, runtime, manager, connection):
            transaction = manager.prepare(proposal(root, worker=True))
            runtime.apply(transaction, policy, {})
            installed_source = (root / "policy.py").read_bytes()
            user_method = lambda self: "user edit"
            type(policy).act = user_method
            with self.assertRaises(RuntimeError) as caught:
                runtime.rollback(transaction)
            self.assertNotIn("rollback_runtime_source_patch", connection.calls)
            self.assertIs(type(policy).act, user_method)
            self.assertEqual((root / "policy.py").read_bytes(), installed_source)
            self.assertFatal(caught.exception)

    def test_worker_preflight_conflict_prevents_parent_and_disk_rollback(self):
        with fixture(worker=True) as (root, policy, runtime, manager, connection):
            transaction = manager.prepare(proposal(root, worker=True))
            runtime.apply(transaction, policy, {})
            installed_source = (root / "policy.py").read_bytes()
            installed = type(policy).act
            connection.fail_check = True
            with self.assertRaises(RuntimeError) as caught:
                runtime.rollback(transaction)
            self.assertIs(type(policy).act, installed)
            self.assertEqual((root / "policy.py").read_bytes(), installed_source)
            self.assertNotIn("rollback_runtime_source_patch", connection.calls)
            self.assertFatal(caught.exception)

    def test_failed_worker_rollback_during_apply_is_fatal_to_controller(self):
        with fixture(worker=True) as (root, policy, runtime, _, connection):
            failing = 'def Resume(self, context):\n    self.history.append("failed")\n    raise RuntimeError("bad entrypoint")\n'
            class Generator:
                def __init__(self):
                    self.calls = 0
                def generate(self, context):
                    self.calls += 1
                    return proposal(root, resume=failing, worker=True)
            connection.fail_rollback = True
            generator = Generator()
            controller = SourceRecoveryController(RecoveryConfig(mode="source"), root / "audit",
                                                  runtime=runtime, generator=generator)
            frame = {"step": 0, "task": "Mobility", "context": {},
                     "agents": {}, "objects": {}, "metrics": {}}
            with self.assertRaises(RuntimeError) as caught:
                controller.handle_execution_error(frame, policy, ValueError("original failure"))
            self.assertFatal(caught.exception)
            self.assertEqual(controller.versions[-1]["status"], "fatal")
            self.assertTrue(runtime._active)
            self.assertEqual(policy.history, [])
            with self.assertRaises(RuntimeError) as repeated:
                runtime.apply(runtime._active[-1][0], policy, {})
            self.assertFatal(repeated.exception)
            with self.assertRaises(RuntimeError) as repeated_controller:
                controller.handle_execution_error(frame, policy, ValueError("caller tried again"))
            self.assertFatal(repeated_controller.exception)
            self.assertEqual(generator.calls, 1)

    def test_clean_rollback_retains_real_control_history(self):
        with fixture(worker=True) as (root, policy, runtime, manager, connection):
            transaction = manager.prepare(proposal(root, worker=True))
            runtime.apply(transaction, policy, {})
            policy.history.append("real progress")
            self.assertEqual(runtime.rollback(transaction)["status"], "rolled_back")
            self.assertEqual(policy.act(), "original")
            self.assertEqual(policy.history, ["resumed", "real progress"])
            self.assertEqual((root / "policy.py").read_text(), POLICY)
            self.assertIn("check_runtime_source_rollback", connection.calls)

    def test_unloaded_targets_are_rejected_before_files_are_committed(self):
        for process in ("parent", "worker"):
            with self.subTest(process=process), fixture(worker=True) as (root, policy, runtime, manager, connection):
                worker_target = runtime.targets["worker.py"]
                sys.modules.pop(worker_target["module"])
                worker_target["process"] = process
                manager.targets["worker.py"]["process"] = process
                transaction = manager.prepare(proposal(root, worker=True))
                with self.assertRaisesRegex(RuntimeError, "unloaded|loaded|pending"):
                    runtime.apply(transaction, policy, {})
                self.assertEqual((root / "policy.py").read_text(), POLICY)
                self.assertEqual((root / "worker.py").read_text(), WORKER)
                self.assertEqual(policy.act(), "original")
                self.assertFalse(hasattr(policy, "Resume"))
                self.assertEqual(runtime._active, [])
                self.assertEqual(runtime._parent_patcher._stack, [])
                self.assertEqual(connection.patcher._stack, [])

    def test_apply_file_conflict_preserves_live_revision_and_halts_next_apply(self):
        with fixture() as (root, policy, runtime, manager, _):
            failing = ('def Resume(self, context):\n    self.history.append("failed")\n'
                       '    context["edit"]()\n    raise RuntimeError("bad entrypoint")\n')
            transaction = manager.prepare(proposal(root, resume=failing))
            def edit():
                path = root / "policy.py"
                path.write_text(path.read_text() + "# concurrent user edit\n")
            with self.assertRaises(RuntimeError) as caught:
                runtime.apply(transaction, policy, {"edit": edit})
            self.assertFatal(caught.exception)
            self.assertEqual(policy.act(), "revised")
            self.assertIn("# concurrent user edit", (root / "policy.py").read_text())
            self.assertEqual(policy.history, [])
            self.assertEqual(runtime._active[-1][0], transaction)

    def test_entrypoint_result_must_be_finite_bounded_json_before_revision_activation(self):
        for result in ("object()", 'float("nan")', '"x" * (256 * 1024 + 1)'):
            with self.subTest(result=result), fixture() as (root, policy, runtime, manager, _):
                resume = ('def Resume(self, context):\n    self.history.append("partial")\n'
                          '    return ' + result + '\n')
                transaction = manager.prepare(proposal(root, resume=resume))
                with self.assertRaisesRegex(RuntimeError, "JSON|256"):
                    runtime.apply(transaction, policy, {})
                self.assertEqual(runtime._active, [])
                self.assertEqual(policy.history, [])
                self.assertEqual(policy.act(), "original")
                self.assertFalse(hasattr(policy, "Resume"))
                self.assertEqual((root / "policy.py").read_text(), POLICY)

    def test_worker_preflight_rpc_reports_errors_without_mutation(self):
        from habitat.core.environments import RLTaskEnv
        self.assertTrue(hasattr(RLTaskEnv, "check_runtime_source_rollback"))
        env = RLTaskEnv.__new__(RLTaskEnv)
        result = env.check_runtime_source_rollback("unknown")
        self.assertEqual(result["status"], "error")
        self.assertIn("installed", result["error"])

    def test_generated_patch_cannot_replace_rollback_preflight_rpc(self):
        with fixture() as (root, _, _, manager, _):
            value = proposal(root)
            value["patches"][0].update(target="Policy.check_runtime_source_rollback",
                code="def check_runtime_source_rollback(self, context):\n    return {}\n")
            value["entrypoint"]["target"] = "Policy.check_runtime_source_rollback"
            with self.assertRaisesRegex(ValueError, "cannot be patched"):
                manager.prepare(value)


if __name__ == "__main__":
    unittest.main()
