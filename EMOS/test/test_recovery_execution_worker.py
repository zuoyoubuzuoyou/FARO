"""Restricted navigation repairs cross the real RLTaskEnv worker RPC boundary."""
from copy import deepcopy
import hashlib
import importlib.util
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

POLICY_PATH = "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py"
WORKER_PATH = "habitat-lab/habitat/tasks/rearrange/actions/oracle_nav_action.py"
POLICY_MODULE = "_execution_rpc_policy"
WORKER_MODULE = "_execution_rpc_worker"
POLICY_SOURCE = '''from types import SimpleNamespace

class MultiLLMPolicy:
    def __init__(self):
        self._active_policies = [SimpleNamespace(_cur_call_high_level=[False])]
'''
WORKER_SOURCE = '''from types import SimpleNamespace

class OracleNavCoordinateAction:
    def __init__(self):
        self.skill_done = False
        self.motion_calls = 0
        self.reset_calls = 0
        self.last_velocity = None
        self._config = SimpleNamespace(dist_thresh=0.2, turn_thresh=0.1)

    def motion(self, velocity):
        self.motion_calls += 1
        self.last_velocity = velocity

    def reset(self):
        self.reset_calls += 1

    def step(self, *args, **kwargs):
        self.skill_done = False
        dist_to_final_nav_targ = kwargs["distance"]
        angle_to_obj = kwargs["angle"]
        at_goal = (
            dist_to_final_nav_targ < self._config.dist_thresh
            and angle_to_obj < self._config.turn_thresh
        )
        if not at_goal:
            vel = [1, 0]
        else:
            vel = [0, 0]
        self.motion(vel)
'''
REPAIRED_STEP = '''def step(self, *args, **kwargs):
    self.skill_done = False
    dist_to_final_nav_targ = kwargs["distance"]
    angle_to_obj = kwargs["angle"]
    at_goal = (
        dist_to_final_nav_targ < self._config.dist_thresh
        and angle_to_obj < self._config.turn_thresh
    )
    if not at_goal:
        vel = [1, 0]
    else:
        vel = [0, 0]
        self.skill_done = True
    self.motion(vel)
'''
RESUME = '''def Resume(self, context):
    for child in self._active_policies:
        child._cur_call_high_level[0] = True
    return {"resumed": True}
'''
TARGETS = {
    POLICY_PATH: {"module": POLICY_MODULE, "classes": ["MultiLLMPolicy"], "process": "parent"},
    WORKER_PATH: {"module": WORKER_MODULE, "classes": ["OracleNavCoordinateAction"], "process": "worker"},
}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def worker_main(connection, root):
    from habitat.core.environments import RLTaskEnv
    from habitat_mas.recovery.source_patch import RuntimeSourcePatcher

    path = Path(root) / WORKER_PATH
    module = load_module(WORKER_MODULE, path)
    action = module.OracleNavCoordinateAction()
    environment = RLTaskEnv.__new__(RLTaskEnv)
    environment._runtime_source_patcher = RuntimeSourcePatcher(root, targets=TARGETS, restricted=True)
    physical_state = {"episode": "same-episode", "position": [1, 0, 0], "held_object": 42}
    rpc_counts = {"install": 0, "rollback": 0}

    def snapshot():
        return {
            "action": {"identity": id(action), "skill_done": action.skill_done,
                       "motion_calls": action.motion_calls, "reset_calls": action.reset_calls,
                       "last_velocity": action.last_velocity, "physical_state": physical_state},
            "rpc_counts": dict(rpc_counts),
        }

    try:
        while True:
            name, arguments = connection.recv()
            if name == "quit":
                connection.send({"status": "closed"})
                break
            if name == "snapshot":
                result = snapshot()
            elif name == "run_action":
                action.step(**arguments)
                result = snapshot()
            elif name == "invalidate_module_path":
                module.__file__ = str(path.with_name("different_source.py"))
                result = {"status": "invalidated"}
            else:
                if name == "apply_runtime_source_patch":
                    rpc_counts["install"] += 1
                elif name == "rollback_runtime_source_patch":
                    rpc_counts["rollback"] += 1
                result = getattr(environment, name)(**arguments)
            connection.send(result)
    finally:
        connection.close()


class RemoteEnvironment:
    def __init__(self, root):
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=worker_main, args=(child, str(root)))
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
        try:
            if self.process.is_alive():
                self.call_at(0, "quit")
        finally:
            self.process.join(5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
            self.connection.close()


class RestrictedExecutionWorkerTests(unittest.TestCase):
    def setUp(self):
        from habitat_mas.recovery.source_patch import SourcePatchManager
        from habitat_mas.recovery.source_runtime import SourceRuntime

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for relative, source in ((POLICY_PATH, POLICY_SOURCE), (WORKER_PATH, WORKER_SOURCE)):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        module = load_module(POLICY_MODULE, self.root / POLICY_PATH)
        self.addCleanup(sys.modules.pop, POLICY_MODULE, None)
        self.policy = module.MultiLLMPolicy()
        self.remote = RemoteEnvironment(self.root)
        self.addCleanup(self.remote.close)
        self.runtime = SourceRuntime(envs=self.remote, root=self.root, targets=TARGETS, restricted=True)
        self.addCleanup(self.runtime.close)
        self.manager = SourcePatchManager(self.root, self.root / "artifacts", targets=TARGETS, restricted=True)

    def prepare(self):
        return self.manager.prepare({
            "patches": [
                {"path": POLICY_PATH, "target": "MultiLLMPolicy.Resume",
                 "base_sha256": hashlib.sha256(POLICY_SOURCE.encode()).hexdigest(), "code": RESUME},
                {"path": WORKER_PATH, "target": "OracleNavCoordinateAction.step",
                 "base_sha256": hashlib.sha256(WORKER_SOURCE.encode()).hexdigest(), "code": REPAIRED_STEP},
            ],
            "entrypoint": {"path": POLICY_PATH, "target": "MultiLLMPolicy.Resume"},
            "explanation": "Report completion at the existing geometric goal without changing motion.",
        })

    def test_restricted_repair_updates_existing_action_only_on_its_next_step_and_close_restores_methods(self):
        before = self.remote.call_at(0, "run_action", {"distance": 0.05, "angle": 0.0})
        self.assertFalse(before["action"]["skill_done"])
        transaction = self.prepare()
        result = self.runtime.apply(transaction, self.policy, {})
        installed = self.remote.call_at(0, "snapshot")
        self.assertEqual(result["status"], "applied")
        self.assertEqual(result["result"], {"resumed": True})
        self.assertTrue(self.policy._active_policies[0]._cur_call_high_level[0])
        self.assertEqual(installed["action"], before["action"])
        self.assertEqual(installed["rpc_counts"], {"install": 1, "rollback": 0})
        travelling = self.remote.call_at(0, "run_action", {"distance": 1.0, "angle": 0.0})
        self.assertFalse(travelling["action"]["skill_done"])
        self.assertEqual(travelling["action"]["last_velocity"], [1, 0])
        completed = self.remote.call_at(0, "run_action", {"distance": 0.05, "angle": 0.0})
        self.assertTrue(completed["action"]["skill_done"])
        self.assertEqual(completed["action"]["last_velocity"], [0, 0])
        self.assertEqual(completed["action"]["motion_calls"], 3)
        self.assertEqual(completed["action"]["identity"], before["action"]["identity"])
        self.assertEqual(completed["action"]["physical_state"], before["action"]["physical_state"])
        self.runtime.close()
        restored = self.remote.call_at(0, "snapshot")
        self.assertEqual(restored["action"], completed["action"])
        self.assertEqual(restored["rpc_counts"], {"install": 1, "rollback": 1})
        self.assertFalse(hasattr(self.policy, "Resume"))
        self.assertEqual((self.root / POLICY_PATH).read_text(), POLICY_SOURCE)
        self.assertEqual((self.root / WORKER_PATH).read_text(), WORKER_SOURCE)
        original = self.remote.call_at(0, "run_action", {"distance": 0.05, "angle": 0.0})
        self.assertFalse(original["action"]["skill_done"])
        self.assertEqual(original["action"]["motion_calls"], 4)
        self.assertEqual(original["action"]["reset_calls"], 0)

    def test_worker_validation_failure_restores_parent_method_and_original_sources(self):
        transaction = self.prepare()
        before = self.remote.call_at(0, "snapshot")
        self.remote.call_at(0, "invalidate_module_path")
        with self.assertRaisesRegex(RuntimeError, "Loaded module does not match"):
            self.runtime.apply(transaction, self.policy, {})
        after = self.remote.call_at(0, "snapshot")
        self.assertEqual(after["action"], before["action"])
        self.assertEqual(after["rpc_counts"], {"install": 1, "rollback": 0})
        self.assertFalse(hasattr(self.policy, "Resume"))
        self.assertFalse(self.policy._active_policies[0]._cur_call_high_level[0])
        self.assertEqual((self.root / POLICY_PATH).read_text(), POLICY_SOURCE)
        self.assertEqual((self.root / WORKER_PATH).read_text(), WORKER_SOURCE)

    def test_restricted_runtime_rejects_environment_method_before_installing_anything(self):
        from habitat_mas.recovery.source_patch import SourceTransaction

        transaction = SourceTransaction(self.root, [{
            "path": "habitat-lab/habitat/core/env.py", "process": "worker",
            "changes": [{"target": "Env.step"}],
        }], {"path": POLICY_PATH, "target": "MultiLLMPolicy.Resume"}, [], restricted=True)
        before = self.remote.call_at(0, "snapshot")
        with self.assertRaisesRegex(RuntimeError, "allowlisted|execution|worker"):
            self.runtime.apply(transaction, self.policy, {})
        self.assertEqual(self.remote.call_at(0, "snapshot"), before)
        self.assertFalse(hasattr(self.policy, "Resume"))

    def test_worker_rpc_rejects_missing_profile_and_tampered_source_without_running_action(self):
        transaction = self.prepare()
        before = self.remote.call_at(0, "snapshot")
        missing_profile = transaction.manifest("worker")
        missing_profile.pop("validation_profile")
        result = self.remote.call_at(0, "apply_runtime_source_patch", {"manifest": missing_profile})
        self.assertEqual(result["status"], "error")
        self.assertIn("profile", result["error"])
        tampered = deepcopy(transaction.manifest("worker"))
        tampered["files"][0]["after_source"] += "\n# different bytes\n"
        result = self.remote.call_at(0, "apply_runtime_source_patch", {"manifest": tampered})
        self.assertEqual(result["status"], "error")
        self.assertIn("hash", result["error"])
        unsafe = deepcopy(transaction.manifest("worker"))
        file = unsafe["files"][0]
        file["changes"][0]["code"] = REPAIRED_STEP.replace(
            "        self.skill_done = True",
            "        self._config.dist_thresh = 100.0\n        self.skill_done = True")
        file["after_source"] = file["after_source"].replace(
            "self.skill_done = True", "self._config.dist_thresh = 100.0\n            self.skill_done = True")
        file["after_sha256"] = hashlib.sha256(file["after_source"].encode()).hexdigest()
        result = self.remote.call_at(0, "apply_runtime_source_patch", {"manifest": unsafe})
        self.assertEqual(result["status"], "error")
        after = self.remote.call_at(0, "snapshot")
        self.assertEqual(after["action"], before["action"])
        self.assertEqual(after["rpc_counts"], {"install": 3, "rollback": 0})
        self.assertEqual((self.root / WORKER_PATH).read_text(), WORKER_SOURCE)


if __name__ == "__main__":
    unittest.main()
