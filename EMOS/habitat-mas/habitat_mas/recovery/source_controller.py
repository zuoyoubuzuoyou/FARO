"""Version original runtime source and invoke generated Resume on live objects."""
from copy import deepcopy
from pathlib import Path
import sys
import traceback

from ..detection.llm import _plain
from ..utils.trajectory import redact
from .context import build_context
from .controller import RecoveryController, _write
from .source_generator import SourceRecoveryGenerator
from .source_patch import SourcePatchManager, source_snapshot
from .source_runtime import SourceConsistencyError, SourceRuntime


SOURCE_API = {
    "proposal": {"patches": [{"path": "source_files relative path", "target": "Class.method",
                  "base_sha256": "exact current file SHA256", "code": "complete def source"}],
                 "entrypoint": {"path": "parent policy path", "target": "PolicyClass.Resume"},
                 "explanation": "why the original runtime code needs these changes"},
    "entrypoint_signature": "def METHOD(self, context)",
    "entrypoint_execution": "Invoked on the original live policy after source/method installation; implement Resume directly, not through predefined recovery command helpers.",
    "source_effect": "Original .py files and methods on original loaded classes change; already-running frames retain old bytecode until they return.",
    "boundaries": "No imports, top-level execution, metric/validator changes, lifecycle reset or extra env.step. Worker edits are installed via RPC in the existing worker.",
}


def _bindings(policy):
    result = []
    def add(role, value):
        if value is not None:
            cls = type(value)
            result.append({"role": role, "module": cls.__module__, "class": cls.__qualname__})
    add("policy", policy)
    for index, child in enumerate(getattr(policy, "_active_policies", [])):
        add("agent_%d.control" % index, child)
        high = getattr(child, "_high_level_policy", None)
        add("agent_%d.high_level" % index, high)
        agent = getattr(high, "llm_agent", None)
        add("agent_%d.agent" % index, agent)
        add("agent_%d.model" % index, getattr(agent, "llm_model", None))
        for skill_id, skill in getattr(child, "_skills", {}).items():
            add("agent_%d.skill_%s" % (index, skill_id), skill)
    return result


class SourceRecoveryController(RecoveryController):
    def __init__(self, config, output_dir, runtime=None, detector=None, generator=None):
        if generator is None:
            from .config import RecoveryConfig
            cfg = RecoveryConfig.from_value(config)
            generator = SourceRecoveryGenerator(model=cfg.model, timeout=cfg.model_timeout)
        super().__init__(config, output_dir, detector=detector, generator=generator)
        self.runtime = runtime if runtime is not None else SourceRuntime(
            persist_source_changes=self.config.persist_source_changes)
        self.manager = SourcePatchManager(self.runtime.root, self.output_dir / "source",
                                          targets=getattr(self.runtime, "targets", None))
        self._source_revisions = []

    def _context(self, frame, policy):
        # A previous version changed originals: never reuse a cached source snapshot.
        sources = source_snapshot(self.runtime.root, targets=getattr(self.runtime, "targets", None))
        context = build_context(frame, list(self._recent), self._last_error, policy, self.config,
                                self.versions, {}, sources={"source_files": sources})
        context["schema_version"] = "emos-source-recovery-v1"
        context["available_api"] = deepcopy(SOURCE_API)
        context["runtime"]["class_bindings"] = _bindings(policy)
        context["runtime"]["source_targets"] = {
            path: {"module": info["module"], "process": info["process"],
                   "loaded_parent_classes": [name for name in info["classes"]
                       if isinstance(vars(sys.modules[info["module"]]).get(name), type)]
                       if info["process"] == "parent" and info["module"] in sys.modules else [],
                   "worker_loading": "checked transactionally by the worker before file commit"}
            for path, info in sources.items()}
        context["recovery"]["active_source_revisions"] = [version["iteration"] for _, version in self._source_revisions]
        return redact(context)

    def _generate(self, frame, policy):
        consistency_error = getattr(self.runtime, "_consistency_error", None)
        if consistency_error is not None:
            raise consistency_error
        self._last_generation_step = frame["step"]
        version = {"iteration": len(self.versions) + 1, "step": frame["step"],
                   "mode": "source", "status": "generating", "outcomes": []}
        directory = self.output_dir / "versions" / ("%03d" % version["iteration"])
        context = self._context(frame, policy)
        self.versions.append(version)
        _write(directory / "context.json", context)
        self._event("source_generation_started", iteration=version["iteration"])
        try:
            proposal = self.generator.generate(context)
            _write(directory / "generation.json", proposal)
            version.update(patches=deepcopy(proposal["patches"]), entrypoint=deepcopy(proposal["entrypoint"]),
                           explanation=proposal.get("explanation"))
            transaction = self.manager.prepare(proposal)
            applied = self.runtime.apply(transaction, policy, context)
        except SourceConsistencyError as error:
            version.update(status="fatal", error=redact(str(error)), error_type=type(error).__name__,
                           rollback_failed=True)
            self._event("source_consistency_error", iteration=version["iteration"], error=version["error"])
            self._save_version(version)
            raise
        except Exception as error:
            version.update(status="rejected", error=redact(str(error)), error_type=type(error).__name__)
            if hasattr(error, "response"):
                _write(directory / "generation_error.json", {"request": getattr(error, "request", None), "response": error.response})
            self._event("source_generation_rejected", iteration=version["iteration"], error=version["error"])
            result = {"status": "rejected", "iteration": version["iteration"], "error": version["error"]}
            self._save_version(version)
            return result
        # The transaction is now installed. An audit/storage failure must halt
        # evaluation and reach its cleanup; it is not a rejected candidate.
        version.update(status="active", transaction_id=transaction.id, application=_plain(applied))
        self._source_revisions.append((transaction, version))
        self._last_apply_step = frame["step"]
        self._pending_assessments.append({"before": deepcopy(frame), "version": version})
        self._save_version(version)
        self._event("source_application", iteration=version["iteration"], transaction_id=transaction.id,
                    result=applied, simulator_reset=False)
        return {"status": "applied", "iteration": version["iteration"], "transaction_id": transaction.id,
                "entrypoint": version["entrypoint"], "result": applied}

    def handle_execution_error(self, frame, policy, error):
        """The failed decision did not advance physics or the observation clock."""
        self._last_step = frame["step"]
        frames = []
        for item in traceback.extract_tb(error.__traceback__)[-10:]:
            path = Path(item.filename)
            try:
                location = str(path.resolve().relative_to(self.runtime.root))
            except ValueError:
                location = path.name
            frames.append({"path": location, "line": item.lineno, "function": item.name})
        failure = {"type": type(error).__name__, "message": redact(str(error)), "traceback": frames}
        if self._source_revisions:
            transaction, version = self._source_revisions[-1]
            # Fail closed if coordinated rollback cannot complete; do not hide a
            # user edit conflict or install further code on inconsistent state.
            self.runtime.rollback(transaction)
            self._source_revisions.pop()
            version.update(status="rolled_back", execution_error=failure)
            version["outcomes"].append({"step": frame["step"], "status": "execution_error", "error": failure})
            self._save_version(version)
            self._event("source_rollback", iteration=version["iteration"], error=failure)
        self._last_error = {"status": "system_error", "reason_code": "execution_exception",
            "explanation": "An original runtime method raised before a simulator transition completed.",
            "exception": failure, "agents": list(frame.get("agents") or {}),
            "evidence": [{"step": frame["step"], "observation": "Last captured physical state before failed decision; exception is recorded separately."}]}
        self._event("execution_exception", error=failure)
        if len(self.versions) >= self.config.max_iterations:
            self._event("source_budget_exhausted", attempts=len(self.versions))
            return {"status": "budget_exhausted"}
        # A failed first decision has no later observation on which to retry a
        # rejected proposal. Revise immediately within the same finite budget.
        last_result = None
        while len(self.versions) < self.config.max_iterations:
            last_result = self._generate(frame, policy)
            if last_result.get("status") == "applied":
                return last_result
        return {"status": "budget_exhausted", "last_result": last_result}
