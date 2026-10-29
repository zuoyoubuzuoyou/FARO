"""Version original runtime source and invoke generated Resume on live objects."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import sys
import traceback
from uuid import uuid4

from ..detection.llm import _plain
from ..utils.trajectory import redact
from .context import build_context
from .controller import RecoveryController, _write
from .source_generator import SourceRecoveryGenerator
from .source_patch import RESTRICTED_SOURCE_POLICY, SourcePatchManager, source_snapshot
from .source_runtime import SourceConsistencyError, SourceRuntime


SOURCE_API = {
    "proposal": {"patches": [{"path": "source_files relative path", "target": "Class.method",
                  "base_sha256": "exact current file SHA256", "code": "complete def source"}],
                 "entrypoint": {"path": "parent policy path", "target": "PolicyClass.Resume"},
                 "explanation": "why the original runtime code needs these changes"},
    "entrypoint_signature": "def METHOD(self, context)",
    "entrypoint_execution": "Invoked on the original live policy after source/method installation; implement Resume directly, not through predefined recovery command helpers.",
    "entrypoint_context": "The live Resume receives the original expanded context, including runtime.current_frame, agent fields and full source snapshots. input_layout only describes the compact model view, not extra runtime API fields.",
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
        # Explicit generate-only developer integrations retain their legacy
        # transaction contract. The production model generator is restricted.
        self._restricted = callable(getattr(self.generator, "detect_and_repair", None))
        self.runtime = runtime if runtime is not None else SourceRuntime(
            persist_source_changes=self.config.persist_source_changes)
        if self._restricted and hasattr(self.runtime, "require_restricted_patches"):
            self.runtime.require_restricted_patches()
        self.manager = SourcePatchManager(self.runtime.root, self.output_dir / "source",
                                          targets=getattr(self.runtime, "targets", None), restricted=self._restricted)
        self._source_revisions = []
        self._combined = (detector is None and self.config.detector == "llm"
                          and self._restricted)
        self._source_samples = 0
        self._budget_reported = False

    def _context(self, frame, policy):
        # A previous version changed originals: never reuse a cached source snapshot.
        sources = source_snapshot(self.runtime.root, targets=getattr(self.runtime, "targets", None))
        context = build_context(frame, list(self._recent), self._last_error, policy, self.config,
                                self.versions, {}, sources={"source_files": sources})
        context["schema_version"] = "emos-source-recovery-v1"
        if self._restricted:
            context["code"]["edit_policy"] = deepcopy(RESTRICTED_SOURCE_POLICY)
        context["available_api"] = deepcopy(SOURCE_API)
        context["runtime"]["class_bindings"] = _bindings(policy)
        context["runtime"]["source_targets"] = {
            path: {"module": info["module"], "process": info["process"],
                   "loaded_parent_classes": [name for name in info["classes"]
                       if isinstance(vars(sys.modules[info["module"]]).get(name), type)]
                       if info["process"] == "parent" and info["module"] in sys.modules else [],
                   "worker_loading": "checked transactionally by the worker before file commit"}
            for path, info in sources.items()}
        if self._restricted:
            context["available_api"]["boundaries"] = (
                "Only methods and edit regions declared by code.edit_policy are editable. "
                "Declared navigation completion edits are installed transactionally in the existing worker via RPC; "
                "other environment/task/simulator and motion-control source remains read-only. "
                "Preserve geometry, motion calls, arguments, configured thresholds, diagnostics and normal paths. "
                "No imports, top-level execution, goal/capability/observation/scoring changes, reset, teleport or extra env.step.")
            for path, target in context["runtime"]["source_targets"].items():
                target["worker_loading"] = (
                    "declared navigation completion edits are checked and installed by the existing worker before file commit"
                    if target["process"] == "worker" and any(name in RESTRICTED_SOURCE_POLICY["editable"]
                        for name in sources[path].get("classes", []))
                    else "read-only in this profile" if target["process"] == "worker"
                    else "not applicable; editable parent methods are declared by code.edit_policy")
        context["recovery"]["active_source_revisions"] = [version["iteration"] for _, version in self._source_revisions]
        return redact(context)

    def _observe_sample(self, frame, policy):
        if not self._combined:
            return super()._observe_sample(frame, policy)
        self._source_samples += 1
        count, cfg = self._source_samples, self.config
        if count < cfg.window or (count - cfg.window) % cfg.check_every:
            return None
        if len(self.versions) >= cfg.max_iterations:
            self._report_budget_exhausted()
            return None
        can_apply = self._last_apply_step is None or frame["step"] - self._last_apply_step >= cfg.cooldown_steps
        can_generate = self._last_generation_step is None or frame["step"] - self._last_generation_step >= cfg.cooldown_steps
        if not can_apply or not can_generate:
            return None
        return self._detect_and_repair(frame, policy)

    def _report_budget_exhausted(self):
        if not self._budget_reported:
            self._event("source_budget_exhausted", attempts=len(self.versions))
            self._budget_reported = True

    def _save_input_snapshot(self, context):
        """Keep complete local evidence even for normal or failed requests."""
        relative = "input_snapshots/%s.full.json.gz" % uuid4().hex
        path = self.output_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(redact(context), ensure_ascii=False, allow_nan=False).encode("utf-8")
        with path.open("xb") as stream:
            stream.write(gzip.compress(data, mtime=0))
        return relative

    def _detect_and_repair(self, frame, policy, execution_error=None):
        consistency_error = getattr(self.runtime, "_consistency_error", None)
        if consistency_error is not None:
            raise consistency_error
        context = self._context(frame, policy)
        context["schema_version"] = "emos-source-detect-repair-v1"
        source_api = context["available_api"]["proposal"]
        context["available_api"]["proposal"] = {"system_error": True,
            "patches": source_api["patches"], "entrypoint": source_api["entrypoint"]}
        context["available_api"]["negative"] = {"system_error": False}
        context["available_api"]["unrepairable"] = {"system_error": True, "patches": [], "entrypoint": None}
        # A fresh check gets real execution evidence, never a previous model's
        # classification presented as the conclusion of this check.
        context["error_report"] = deepcopy(execution_error)
        context["trigger"] = "execution_exception" if execution_error is not None else "sampled_observation"
        context["observation_parameters"] = {"sample_every": self.config.sample_every,
            "window": self.config.window, "check_every": self.config.check_every,
            "window_unit": "sampled observations"}
        input_snapshot = self._save_input_snapshot(context)
        try:
            proposal = self.generator.detect_and_repair(context)
        except Exception as error:
            # Invalid responses and failed requests are bounded repair attempts,
            # not negative predictions, and must not trigger another API call.
            return self._generate(frame, policy, context=context, generation_error=error,
                                  input_snapshot=input_snapshot)
        decision = {"system_error": proposal["system_error"], "step": frame["step"],
                    "status": "system_error" if proposal["system_error"] else "normal", "detector": "llm"}
        audit = redact({"step": frame["step"], "trigger": context["trigger"], "decision": decision,
                        "input_snapshot": input_snapshot,
                        "request": proposal.get("request"), "response": proposal.get("response"),
                        "usage": proposal.get("usage")})
        with (self.output_dir / "source_requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(audit, ensure_ascii=False, allow_nan=False) + "\n")
        self._event("detection", decision=decision)
        if not proposal["system_error"]:
            return {"status": "normal", "system_error": False}
        proposal["explanation"] = ("Source repair proposed from current runtime evidence." if proposal["patches"]
                                   else "A runtime problem was detected without a permitted source repair.")
        return self._generate(frame, policy, proposal=proposal, context=context,
                              input_snapshot=input_snapshot)

    def _generate(self, frame, policy, proposal=None, context=None, generation_error=None,
                  input_snapshot=None):
        consistency_error = getattr(self.runtime, "_consistency_error", None)
        if consistency_error is not None:
            raise consistency_error
        self._last_generation_step = frame["step"]
        version = {"iteration": len(self.versions) + 1, "step": frame["step"],
                   "mode": "source", "status": "generating", "outcomes": []}
        if input_snapshot is not None:
            version["input_snapshot"] = input_snapshot
        directory = self.output_dir / "versions" / ("%03d" % version["iteration"])
        context = self._context(frame, policy) if context is None else context
        self.versions.append(version)
        _write(directory / "context.json", context)
        self._event("source_generation_started", iteration=version["iteration"])
        try:
            if generation_error is not None:
                raise generation_error
            if proposal is None:
                proposal = self.generator.generate(context)
            _write(directory / "generation.json", proposal)
            version.update(patches=deepcopy(proposal["patches"]), entrypoint=deepcopy(proposal["entrypoint"]),
                           explanation=proposal.get("explanation"))
            if proposal.get("system_error") is True and not proposal["patches"] and proposal["entrypoint"] is None:
                version["status"] = "unrepairable"
                self._save_version(version)
                self._event("source_unrepairable", iteration=version["iteration"])
                return {"status": "unrepairable", "system_error": True, "iteration": version["iteration"]}
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
        consistency_error = getattr(self.runtime, "_consistency_error", None)
        if consistency_error is not None:
            raise consistency_error
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
            self._report_budget_exhausted()
            return {"status": "budget_exhausted"}
        # A failed first decision has no later observation on which to retry a
        # rejected proposal. Revise immediately within the same finite budget.
        last_result = None
        while len(self.versions) < self.config.max_iterations:
            if self._combined:
                evidence = {key: deepcopy(value) for key, value in self._last_error.items()
                            if key not in {"status", "reason_code"}}
                last_result = self._detect_and_repair(frame, policy, execution_error=evidence)
            else:
                last_result = self._generate(frame, policy)
            if last_result.get("status") == "applied":
                return last_result
            if last_result.get("status") in {"normal", "unrepairable"}:
                return last_result
        self._report_budget_exhausted()
        return {"status": "budget_exhausted", "last_result": last_result}
