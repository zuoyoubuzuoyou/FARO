"""Install source revisions on existing Python objects at a stopped policy boundary."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class SourceConsistencyError(RuntimeError):
    """A source revision could not be restored safely; execution must stop."""

    rollback_failed = True


class ExecutionCheckpoint:
    """Restore logical control state in place, retaining clients and live objects.

    Containers are saved by reference and restored recursively. Simulator and
    transport objects are opaque: neither copying nor reconstructing them is
    necessary to retry a policy decision before the next environment step.
    """

    _opaque_attributes = frozenset({
        "client", "interpreter", "_sim", "sim", "simulator", "_env", "env",
        "envs", "_parameters", "_buffers", "_modules",
    })
    _control_links = ("_active_policies", "_high_level_policy", "_skills", "llm_agent", "llm_model")

    def __init__(self, policy, tensors=()):
        self._states = []
        self._seen = set()
        self._capture(policy, force=True)
        for tensor in tensors:
            self._capture(tensor)
        # Agent communication is class state, rather than an instance field.
        from habitat_mas.agents.crab_agent import CrabAgent
        self._message_owner = CrabAgent
        self._message_pipe = CrabAgent.message_pipe
        self._capture(self._message_pipe)
        # Keep the failed attempt in the audit, while restoring the action
        # bookkeeping that will be associated with the successful env.step.
        from habitat_mas.utils.trajectory import _ACTIVE_WRITER
        binding = _ACTIVE_WRITER.get()
        self._writer_state = None
        if binding is not None:
            writer = binding[0]
            names = ("decision_counts", "decisions_by_phase", "active_actions",
                     "pending_actions", "deferred_ends", "last_actions")
            self._writer_state = (writer, {name: getattr(writer, name) for name in names})
            for value in self._writer_state[1].values():
                self._capture(value)

    def _capture(self, value, force=False):
        identity = id(value)
        if identity in self._seen:
            return
        self._seen.add(identity)
        module = type(value).__module__
        if module.startswith("torch") and hasattr(value, "detach") and hasattr(value, "copy_"):
            self._states.append(("tensor", value, value.detach().clone()))
        elif (module.startswith("numpy") and hasattr(value, "shape")
              and hasattr(value, "copy") and hasattr(value, "__setitem__")):
            self._states.append(("array", value, value.copy()))
        elif isinstance(value, dict):
            self._states.append(("dict", value, dict(value)))
            for item in value.values():
                self._capture(item)
        elif isinstance(value, list):
            self._states.append(("list", value, list(value)))
            for item in value:
                self._capture(item)
        elif isinstance(value, set):
            self._states.append(("set", value, set(value)))
        elif isinstance(value, tuple):
            for item in value:
                self._capture(item)
        elif hasattr(value, "__dict__") and (force or isinstance(value, SimpleNamespace)
                or module.startswith(("habitat_baselines.rl.", "habitat_mas.agents.",
                                      "habitat_mas.utils.models"))):
            attrs = vars(value)
            self._states.append(("object", value, dict(attrs)))
            for name, item in attrs.items():
                if name not in self._opaque_attributes:
                    self._capture(item)
            # nn.Module.__setattr__ stores registered children outside vars().
            # Preserve registrations shallowly, then visit only control links;
            # model weights and arbitrary neural submodules stay opaque.
            registered = attrs.get("_modules")
            if isinstance(registered, dict) and id(registered) not in self._seen:
                self._seen.add(id(registered))
                self._states.append(("dict", registered, dict(registered)))
            if module.startswith("torch.nn") and type(value).__name__ in {"ModuleList", "ModuleDict"}:
                for child in (registered or {}).values():
                    self._capture(child, force=True)
            for name in self._control_links:
                linked = getattr(value, name, None)
                if linked is not None:
                    self._capture(linked, force=True)

    def restore(self):
        for kind, value, saved in self._states:
            if kind == "tensor":
                # detach avoids autograd tracking during rollback.
                value.detach().copy_(saved)
            elif kind == "array":
                value[...] = saved
            elif kind == "dict":
                value.clear()
                value.update(saved)
            elif kind == "list":
                value[:] = saved
            elif kind == "set":
                value.clear()
                value.update(saved)
            elif kind == "object":
                billed_usage = None
                if type(value).__module__.startswith("habitat_mas.utils.models"):
                    current = vars(value).get("token_usage")
                    previous = saved.get("token_usage")
                    if isinstance(current, (int, float)) and isinstance(previous, (int, float)):
                        billed_usage = max(current, previous)
                vars(value).clear()
                vars(value).update(saved)
                if billed_usage is not None:
                    value.token_usage = billed_usage
        self._message_owner.message_pipe = self._message_pipe
        if self._writer_state is not None:
            writer, attrs = self._writer_state
            for name, value in attrs.items():
                setattr(writer, name, value)


class SourceRuntime:
    """Coordinate parent, worker and original files as one reversible revision."""

    def __init__(self, envs=None, root=None, targets=None, persist_source_changes=False):
        self.envs = envs
        self.root = Path(root or REPOSITORY_ROOT).resolve()
        self.targets = targets
        self.persist_source_changes = bool(persist_source_changes)
        self._parent_patcher = None
        self._active = []
        self._consistency_error = None

    def _patcher(self):
        if self._parent_patcher is None:
            from .source_patch import RuntimeSourcePatcher
            self._parent_patcher = RuntimeSourcePatcher(self.root, targets=self.targets)
        return self._parent_patcher

    @staticmethod
    def _require(result, expected, location):
        if not isinstance(result, dict) or result.get("status") != expected:
            detail = result.get("error", result) if isinstance(result, dict) else result
            raise RuntimeError("%s: %s" % (location, detail))

    def _worker(self, name, args, expected):
        if self.envs is None:
            raise RuntimeError("Worker source patches require the live environment connection")
        result = self.envs.call_at(0, name, args)
        self._require(result, expected, "Worker source patch")
        return result

    def apply(self, transaction, policy, context):
        if self._consistency_error is not None:
            raise self._consistency_error
        checkpoint = ExecutionCheckpoint(policy)
        parent = transaction.manifest("parent")
        worker = transaction.manifest("worker")
        parent_installed = worker_installed = False
        try:
            entrypoint = transaction.entrypoint
            target = entrypoint["target"] if isinstance(entrypoint, dict) else entrypoint
            if isinstance(entrypoint, dict) and "path" in entrypoint:
                from .source_patch import SOURCE_TARGETS
                registry = SOURCE_TARGETS if self.targets is None else self.targets
                declaration = registry.get(entrypoint["path"])
                owner_name = target.split(".", 1)[0]
                module = sys.modules.get(declaration["module"]) if declaration else None
                owner = vars(module).get(owner_name) if module else None
                if not isinstance(owner, type) or not isinstance(policy, owner):
                    raise ValueError("Source entrypoint does not belong to the original live policy")
            if parent["files"]:
                result = self._patcher().install(parent)
                self._require(result, "installed", "Parent source patch")
                parent_installed = True
                if result.get("pending_modules"):
                    raise RuntimeError("Parent source targets must already be loaded: "
                                       + ", ".join(result["pending_modules"]))
            if worker["files"]:
                result = self._worker("apply_runtime_source_patch", {"manifest": worker}, "installed")
                worker_installed = True
                if result.get("pending_modules"):
                    raise RuntimeError("Worker source targets must already be loaded: "
                                       + ", ".join(result["pending_modules"]))
            transaction.commit_files()
            method = target.rsplit(".", 1)[-1]
            result = getattr(policy, method)(context)
            try:
                serialized = json.dumps(result, allow_nan=False, ensure_ascii=False)
            except (TypeError, ValueError, OverflowError, RecursionError) as result_error:
                raise ValueError("Source entrypoint must return finite JSON data or None") from result_error
            if len(serialized.encode("utf-8")) > 256 * 1024:
                raise ValueError("Source entrypoint JSON result exceeds the 256KB limit")
            self._active.append((transaction, parent_installed, worker_installed))
            return {"status": "applied", "transaction_id": transaction.id,
                    "simulator_reset": False, "result": result}
        except Exception as error:
            failures = self._undo(transaction, parent_installed, worker_installed)
            try:
                checkpoint.restore()
            except Exception as restore_error:
                failures.append("Control state rollback: " + str(restore_error))
            detail = str(error)
            if failures:
                detail += "; rollback failures: " + "; ".join(failures)
                self._active.append((transaction, parent_installed, worker_installed))
                self._consistency_error = SourceConsistencyError(detail)
                raise self._consistency_error from error
            raise RuntimeError(detail) from error

    def _undo(self, transaction, parent_installed, worker_installed):
        failures = []
        checks = []
        operations = []
        # A failed commit already restores its writes. Only a committed source
        # transaction owns bytes that may be reverted by this revision.
        if getattr(transaction, "_committed", True):
            check_files = getattr(transaction, "check_rollback_files", None)
            if check_files is not None:
                checks.append(check_files)
        if worker_installed:
            checks.append(lambda: self._worker("check_runtime_source_rollback",
                {"transaction_id": transaction.id}, "checked"))
            operations.append(lambda: self._worker("rollback_runtime_source_patch",
                {"transaction_id": transaction.id}, "rolled_back"))
        if parent_installed:
            check_parent = getattr(self._patcher(), "check_rollback", None)
            if check_parent is not None:
                checks.append(lambda: self._require(check_parent(transaction.id),
                                                    "checked", "Parent source rollback check"))
            operations.append(lambda: self._require(self._patcher().rollback(transaction.id),
                                                   "rolled_back", "Parent source rollback"))
        if getattr(transaction, "_committed", True):
            operations.append(transaction.rollback_files)
        # Check every process and every original file before reverting anything.
        # In particular, a user's file edit must not leave old live methods beside
        # newer source bytes. Keep the revision recorded when a check fails.
        for check in checks:
            try:
                check()
            except Exception as error:
                failures.append(str(error))
        if failures:
            return failures
        for operation in operations:
            try:
                operation()
            except Exception as error:
                failures.append(str(error))
                break
        return failures

    def rollback(self, transaction):
        if not self._active or self._active[-1][0] is not transaction:
            raise ValueError("Source revisions must be rolled back in reverse installation order")
        _, parent_installed, worker_installed = self._active[-1]
        failures = self._undo(transaction, parent_installed, worker_installed)
        if failures:
            self._consistency_error = SourceConsistencyError("; ".join(failures))
            raise self._consistency_error
        self._active.pop()
        return {"status": "rolled_back", "transaction_id": transaction.id,
                "simulator_reset": False}

    def close(self):
        if not self.persist_source_changes:
            while self._active:
                self.rollback(self._active[-1][0])
