"""Source recovery preserves live objects and retries only the policy attempt."""
import importlib.util
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def runtime_module(test):
    name = "habitat_mas.recovery.source_runtime"
    test.assertIsNotNone(importlib.util.find_spec(name), "source runtime is required")
    return __import__(name, fromlist=["ExecutionCheckpoint"])


class RuntimePolicy:
    def __init__(self):
        self.history = [[{"role": "user", "content": "existing"}]]
        self.agent = SimpleNamespace(llm_model=SimpleNamespace(
            chat_history=self.history, client=object()), actions=[])
        self._active_policies = [SimpleNamespace(
            _high_level_policy=SimpleNamespace(llm_agent=self.agent),
            _cur_skills=np.array([4]), _call_high_level=torch.tensor([False]))]
        self.mode = "waiting"

    def Resume(self, context):
        self.mode = context["mode"]
        return {"continued": True}


class CheckpointTests(unittest.TestCase):
    def test_registered_torch_high_level_module_history_is_restored_without_copying_weights(self):
        from habitat_mas.agents.crab_agent import CrabAgent
        from habitat_mas.utils.models import OpenAIModel
        module = runtime_module(self)
        high = torch.nn.Module()
        high.llm_agent = CrabAgent("agent_0", [])
        high.llm_agent.llm_model = OpenAIModel.__new__(OpenAIModel)
        high.llm_agent.llm_model.enable_logging = False
        high.llm_agent.llm_model.chat_history = [[{"content": "before"}]]
        high.llm_agent.llm_model.token_usage = 10
        child = torch.nn.Module()
        child._high_level_policy = high  # nn.Module stores this in _modules.
        child.weight = torch.nn.Parameter(torch.tensor([2.0]))
        policy = torch.nn.Module()
        policy._active_policies = torch.nn.ModuleList([child])
        checkpoint = module.ExecutionCheckpoint(policy)
        high.llm_agent.llm_model.chat_history.append([{"content": "failed"}])
        high.llm_agent.llm_model.token_usage += 25  # The failed response was billed.
        child._high_level_policy = torch.nn.Module()
        child.weight.data.fill_(7)  # Model weights are outside logical control.
        checkpoint.restore()
        self.assertIs(child._high_level_policy, high)
        self.assertEqual(high.llm_agent.llm_model.chat_history, [[{"content": "before"}]])
        self.assertEqual(high.llm_agent.llm_model.token_usage, 35)
        self.assertEqual(child.weight.tolist(), [7.0])

    def test_restore_failed_partial_decisions_in_place_without_copying_clients(self):
        module = runtime_module(self)
        policy = RuntimePolicy()
        agent, history, client = policy.agent, policy.history, policy.agent.llm_model.client
        child = policy._active_policies[0]
        skill_ids, flags = child._cur_skills, child._call_high_level
        previous = torch.tensor([[3.0]])
        checkpoint = module.ExecutionCheckpoint(policy, tensors=(previous,))
        history[0][0]["content"] = "failed mutation"
        history.append([{"role": "assistant", "content": "failed decision"}])
        policy.agent = SimpleNamespace()
        policy.failed_attribute = True
        child._cur_skills[:] = 99
        child._call_high_level[:] = True
        previous.add_(10)
        checkpoint.restore()
        self.assertIs(policy.agent, agent)
        self.assertIs(policy.history, history)
        self.assertIs(agent.llm_model.client, client)
        self.assertEqual(history, [[{"role": "user", "content": "existing"}]])
        self.assertFalse(hasattr(policy, "failed_attribute"))
        self.assertIs(child._cur_skills, skill_ids)
        self.assertIs(child._call_high_level, flags)
        self.assertEqual(skill_ids.tolist(), [4])
        self.assertEqual(flags.tolist(), [False])
        self.assertEqual(previous.tolist(), [[3.0]])

    def test_restore_shared_crab_message_pipe(self):
        from habitat_mas.agents.crab_agent import CrabAgent
        module = runtime_module(self)
        previous = CrabAgent.message_pipe
        try:
            CrabAgent.message_pipe = {"agent_1": ["existing"]}
            policy = RuntimePolicy()
            policy.agent = CrabAgent("agent_0", [])
            checkpoint = module.ExecutionCheckpoint(policy)
            CrabAgent.message_pipe["agent_1"].append("failed send")
            checkpoint.restore()
            self.assertEqual(CrabAgent.message_pipe, {"agent_1": ["existing"]})
        finally:
            CrabAgent.message_pipe = previous


class Transaction:
    id = "version-a"
    entrypoint = {"target": "MultiLLMPolicy.Resume"}

    def __init__(self, root, worker=False, fail_commit=False):
        self.path = Path(root) / "original.py"
        self.path.write_text("before")
        self.worker = worker
        self.fail_commit = fail_commit

    def manifest(self, process):
        return {"transaction_id": self.id, "files": [{"path": "original.py"}]
                if process == "parent" or self.worker else []}

    def commit_files(self):
        self.path.write_text("after")
        if self.fail_commit:
            raise RuntimeError("file commit failed")

    def rollback_files(self):
        self.path.write_text("before")


class Patcher:
    def __init__(self):
        self.active = []

    def install(self, manifest):
        self.active.append(manifest["transaction_id"])
        return {"status": "installed", "transaction_id": manifest["transaction_id"]}

    def rollback(self, identity):
        self.active.remove(identity)
        return {"status": "rolled_back"}


class SourceRuntimeTests(unittest.TestCase):
    def make_runtime(self, root, envs=None):
        module = runtime_module(self)
        runtime = module.SourceRuntime(envs=envs, root=root)
        runtime._parent_patcher = Patcher()
        return runtime

    def test_parent_only_revision_calls_original_policy_and_restores_files_on_close(self):
        with tempfile.TemporaryDirectory() as root:
            runtime, policy, txn = self.make_runtime(root), RuntimePolicy(), Transaction(root)
            history = policy.history
            result = runtime.apply(txn, policy, {"mode": "moving"})
            self.assertEqual(result["status"], "applied")
            self.assertIs(result["simulator_reset"], False)
            self.assertEqual(policy.mode, "moving")
            self.assertIs(policy.history, history)
            self.assertEqual(txn.path.read_text(), "after")
            runtime.close()
            self.assertEqual(txn.path.read_text(), "before")
            self.assertEqual(runtime._parent_patcher.active, [])

    def test_worker_failure_restores_parent_methods_files_and_policy_without_reset(self):
        class Envs:
            def __init__(self):
                self.calls = []

            def call_at(self, index, name, args):
                self.calls.append(name)
                return {"status": "error", "error": "worker rejected patch"}

        with tempfile.TemporaryDirectory() as root:
            envs = Envs()
            runtime, policy = self.make_runtime(root, envs), RuntimePolicy()
            txn = Transaction(root, worker=True)
            with self.assertRaisesRegex(RuntimeError, "worker rejected"):
                runtime.apply(txn, policy, {"mode": "moving"})
            self.assertEqual(policy.mode, "waiting")
            self.assertEqual(txn.path.read_text(), "before")
            self.assertEqual(runtime._parent_patcher.active, [])
            self.assertEqual(envs.calls, ["apply_runtime_source_patch"])

    def test_entrypoint_failure_restores_mutated_history_and_committed_files(self):
        class BrokenPolicy(RuntimePolicy):
            def Resume(self, context):
                self.history.append([{"content": "partial"}])
                self.mode = "broken"
                raise ValueError("generated branch failed")

        with tempfile.TemporaryDirectory() as root:
            runtime, policy, txn = self.make_runtime(root), BrokenPolicy(), Transaction(root)
            with self.assertRaisesRegex(RuntimeError, "generated branch failed"):
                runtime.apply(txn, policy, {})
            self.assertEqual(policy.mode, "waiting")
            self.assertEqual(len(policy.history), 1)
            self.assertEqual(txn.path.read_text(), "before")
            self.assertEqual(runtime._parent_patcher.active, [])

    def test_real_original_source_revision_adds_resume_to_existing_policy_then_reverts_it(self):
        from habitat_mas.recovery.source_patch import SourcePatchManager
        module = runtime_module(self)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "policy.py"
            source = "class MultiLLMPolicy:\n    def __init__(self):\n        self.steps = 0\n"
            path.write_text(source)
            name = "_emos_runtime_original"
            spec = importlib.util.spec_from_file_location(name, path)
            live_module = importlib.util.module_from_spec(spec)
            sys.modules[name] = live_module
            self.addCleanup(sys.modules.pop, name, None)
            spec.loader.exec_module(live_module)
            owner, policy = live_module.MultiLLMPolicy, live_module.MultiLLMPolicy()
            targets = {"policy.py": {"module": name, "classes": ["MultiLLMPolicy"], "process": "parent"}}
            manager = SourcePatchManager(root, Path(root) / "artifacts", targets=targets)
            transaction = manager.prepare({"explanation": "Add live continuation",
                "entrypoint": {"path": "policy.py", "target": "MultiLLMPolicy.Resume"},
                "patches": [{"path": "policy.py", "target": "MultiLLMPolicy.Resume",
                    "base_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "code": "def Resume(self, context):\n    self.steps += 1\n    return self.steps\n"}]})
            runtime = module.SourceRuntime(root=root, targets=targets)
            self.assertFalse(hasattr(policy, "Resume"))
            with self.assertRaisesRegex(RuntimeError, "entrypoint.*policy|policy.*entrypoint"):
                runtime.apply(transaction, RuntimePolicy(), {"mode": "wrong original class"})
            self.assertEqual(path.read_text(), source)
            result = runtime.apply(transaction, policy, {})
            self.assertIs(live_module.MultiLLMPolicy, owner)
            self.assertEqual(result["result"], 1)
            self.assertIn("def Resume", path.read_text())
            policy.steps = 12  # Real progress after the installation is retained.
            runtime.close()
            self.assertEqual(policy.steps, 12)
            self.assertFalse(hasattr(policy, "Resume"))
            self.assertEqual(path.read_text(), source)


class OnlineSourceTests(unittest.TestCase):
    def test_failed_policy_restores_first_agent_state_before_source_recovery_and_retry(self):
        from habitat_mas.recovery import online
        self.assertTrue(hasattr(online.OnlineRecovery, "run_policy"), "policy failures need a retry boundary")
        policy = RuntimePolicy()
        observed = []

        class Controller:
            def __init__(self, config, output_dir, runtime=None):
                pass

            def handle_execution_error(self, frame, live_policy, error):
                observed.append((frame["step"], len(live_policy.history), str(error)))
                live_policy.mode = "fixed"
                return {"status": "applied"}

        with tempfile.TemporaryDirectory() as root, patch.object(online, "SourceRecoveryController", Controller):
            bridge = online.OnlineRecovery(SimpleNamespace(mode="source", max_iterations=2), root)
            bridge.on_event({"event_type": "episode_start", "env_index": 0,
                             "context": {}, "state": {"agents": {}}, "env_step": 0})
            calls = []

            def act():
                calls.append(True)
                policy.history.append([{"content": "agent0"}])
                if policy.mode != "fixed":
                    raise ValueError("agent1 malformed arguments")
                return "one complete decision"

            self.assertEqual(bridge.run_policy(act, policy), "one complete decision")
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(policy.history), 2)
            self.assertEqual(observed, [(0, 1, "agent1 malformed arguments")])

    def test_retry_budget_never_hides_unrecoverable_policy_exception(self):
        from habitat_mas.recovery import online
        self.assertTrue(hasattr(online.OnlineRecovery, "run_policy"), "policy failures need a retry boundary")
        class Controller:
            def __init__(self, *args, **kwargs):
                self.calls = 0

            def handle_execution_error(self, *args):
                self.calls += 1
                return {"status": "applied"}

        with tempfile.TemporaryDirectory() as root, patch.object(online, "SourceRecoveryController", Controller):
            bridge = online.OnlineRecovery(SimpleNamespace(mode="source", max_iterations=2), root)
            bridge.on_event({"event_type": "episode_start", "context": {}, "state": {"agents": {}}})
            def fail():
                raise ValueError("still broken")
            with self.assertRaisesRegex(ValueError, "still broken"):
                bridge.run_policy(fail, RuntimePolicy())
            self.assertEqual(bridge._controller.calls, 2)


class WorkerRPCTests(unittest.TestCase):
    def test_worker_patch_errors_return_status_and_leave_rpc_object_usable(self):
        from habitat.core.environments import RLTaskEnv
        self.assertTrue(hasattr(RLTaskEnv, "apply_runtime_source_patch"), "worker needs a live patch RPC")
        env = RLTaskEnv.__new__(RLTaskEnv)
        class FailingPatcher:
            def install(self, manifest):
                raise ValueError("invalid generated method")
            def rollback(self, transaction_id):
                raise ValueError("unknown transaction")
        env._runtime_source_patcher = FailingPatcher()
        result = env.apply_runtime_source_patch({"transaction_id": "bad", "files": []})
        self.assertEqual(result["status"], "error")
        self.assertIn("invalid generated method", result["error"])
        result = env.rollback_runtime_source_patch("bad")
        self.assertEqual(result["status"], "error")
        self.assertIn("unknown transaction", result["error"])
        self.assertIsNotNone(env._runtime_source_patcher)


class EvaluatorSourceTests(unittest.TestCase):
    def test_first_policy_failure_retries_before_the_single_environment_step(self):
        from habitat_mas.recovery import online
        self.assertTrue(hasattr(online.OnlineRecovery, "run_policy"), "evaluator needs a policy retry boundary")
        import gym.spaces
        sys.path.insert(0, str(ROOT / "test"))
        from test_mas_trajectory_integration import FakeEnvs, FakePolicy, load_evaluator
        from habitat_mas.recovery.config import RecoveryConfig
        calls, recovered_frames = [], []

        class Envs(FakeEnvs):
            def __init__(self):
                super().__init__([SimpleNamespace(scene_id="scene", episode_id="0")])
                self.resets = 0
            def reset(self):
                self.resets += 1
                return super().reset()

        class Policy(FakePolicy):
            def __init__(self):
                super().__init__()
                self.mode = "broken"
            def act(self, *args, **kwargs):
                calls.append(self.mode)
                if self.mode == "broken":
                    self.contexts.append("failed agent0 decision")
                    raise ValueError("agent1 parse failed")
                return super().act(*args, **kwargs)

        class Controller:
            def __init__(self, *args, **kwargs):
                pass
            def handle_execution_error(self, frame, policy, error):
                recovered_frames.append((frame["step"], list(policy.contexts)))
                policy.mode = "fixed"
                return {"status": "applied"}
            def observe(self, frame, policy, terminal=False):
                return {"status": "terminal"}

        with tempfile.TemporaryDirectory() as output, patch.object(online, "SourceRecoveryController", Controller):
            envs, policy = Envs(), Policy()
            config = SimpleNamespace(
                habitat_baselines=SimpleNamespace(num_environments=1, test_episode_count=-1,
                    eval=SimpleNamespace(save_trajectories=True,
                        recovery=RecoveryConfig(enabled=True, mode="source"),
                        trajectory_dir=output, image_option=[], video_option=[], evals_per_ep=1)),
                habitat=SimpleNamespace(seed=1,
                    simulator=SimpleNamespace(agents={"agent_0": {}}, agents_order=["agent_0"]),
                    environment=SimpleNamespace(max_episode_steps=10, max_episode_seconds=60),
                    task=SimpleNamespace(task_spec="mobility", success_measure="configured_success",
                        measurements={}, end_on_success=False)))
            load_evaluator()().evaluate_agent(
                SimpleNamespace(actor_critic=policy, masks_shape=(1,), eval=lambda: None), envs,
                config, 0, 0, SimpleNamespace(add_scalar=lambda *args: None), torch.device("cpu"), [],
                SimpleNamespace(action_space=gym.spaces.Discrete(1)), set())
            self.assertEqual(calls, ["broken", "fixed"])
            self.assertEqual(recovered_frames, [(0, [])])
            self.assertEqual(envs.resets, 1)
            self.assertEqual(len(envs.submitted_actions), 1)
            self.assertEqual(len(policy.contexts), 1)


if __name__ == "__main__":
    unittest.main()
