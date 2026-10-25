"""Real parent/worker source installation over the existing RLTaskEnv RPC API."""
import hashlib
import importlib.util
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker_main(connection, root, targets):
    from habitat.core.environments import RLTaskEnv
    from habitat_mas.recovery.source_patch import RuntimeSourcePatcher
    worker = load_module("_source_rpc_worker", Path(root) / "worker.py")
    control = worker.Worker()
    environment = RLTaskEnv.__new__(RLTaskEnv)
    environment._runtime_source_patcher = RuntimeSourcePatcher(root, targets=targets)
    physical_state = {"episode": "same-episode", "position": [1, 0, 0], "held_object": 42}
    while True:
        name, arguments = connection.recv()
        if name == "quit":
            connection.send({"status": "closed"})
            break
        if name == "snapshot":
            connection.send({"value": control.calc(5), "identity": id(control), "state": physical_state})
        else:
            connection.send(getattr(environment, name)(**arguments))
    connection.close()


class RemoteEnvironment:
    def __init__(self, root, targets):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child, str(root), targets))
        self.process.start()
        child.close()

    def call_at(self, index, name, arguments=None):
        if index != 0:
            raise AssertionError("single worker")
        self.connection.send((name, arguments or {}))
        if not self.connection.poll(20):
            raise TimeoutError("worker RPC did not return")
        return self.connection.recv()

    def close(self):
        if self.process.is_alive():
            self.call_at(0, "quit")
        self.process.join(5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        self.connection.close()


class SourceWorkerIntegrationTest(unittest.TestCase):
    def test_patch_changes_existing_worker_control_without_restarting_its_state(self):
        from habitat_mas.recovery.source_patch import SourcePatchManager
        from habitat_mas.recovery.source_runtime import SourceRuntime
        policy_source = "class Policy:\n    def __init__(self):\n        self.active = False\n"
        worker_source = "class Worker:\n    def calc(self, value):\n        return value\n"
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "policy.py").write_text(policy_source)
            (root / "worker.py").write_text(worker_source)
            targets = {
                "policy.py": {"module": "_source_rpc_policy", "classes": ["Policy"], "process": "parent"},
                "worker.py": {"module": "_source_rpc_worker", "classes": ["Worker"], "process": "worker"},
            }
            module = load_module("_source_rpc_policy", root / "policy.py")
            self.addCleanup(sys.modules.pop, "_source_rpc_policy", None)
            policy = module.Policy()
            remote = RemoteEnvironment(root, targets)
            runtime = SourceRuntime(envs=remote, root=root, targets=targets)
            try:
                before = remote.call_at(0, "snapshot")
                manager = SourcePatchManager(root, root / "artifacts", targets=targets)
                transaction = manager.prepare({"patches": [
                    {"path": "policy.py", "target": "Policy.Resume", "base_sha256": hashlib.sha256(policy_source.encode()).hexdigest(),
                     "code": "def Resume(self, context):\n    self.active = True\n    return {\"resumed\": True}\n"},
                    {"path": "worker.py", "target": "Worker.calc", "base_sha256": hashlib.sha256(worker_source.encode()).hexdigest(),
                     "code": "def calc(self, value):\n    return value + 3\n"}],
                    "entrypoint": {"path": "policy.py", "target": "Policy.Resume"}, "explanation": "repair original worker control"})
                result = runtime.apply(transaction, policy, {})
                after = remote.call_at(0, "snapshot")
                self.assertEqual(result["status"], "applied")
                self.assertTrue(policy.active)
                self.assertEqual((before["value"], after["value"]), (5, 8))
                self.assertEqual(before["identity"], after["identity"])
                self.assertEqual(before["state"], after["state"])
                self.assertEqual(after["state"]["held_object"], 42)
                self.assertIn("value + 3", (root / "worker.py").read_text())
                runtime.close()
                restored = remote.call_at(0, "snapshot")
                self.assertEqual(restored, before)
                self.assertEqual((root / "worker.py").read_text(), worker_source)
                self.assertFalse(hasattr(policy, "Resume"))
            finally:
                runtime.close()
                remote.close()


if __name__ == "__main__":
    unittest.main()
