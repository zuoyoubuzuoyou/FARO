"""Detect, generate, validate and resume; never reset or directly edit the sim."""
from collections import deque
from copy import deepcopy
import json
import logging
import math
from pathlib import Path

from ..detection.heuristics import TeamProgressDetector
from ..detection.llm import LLMDetector, _frame
from ..utils.trajectory import redact
from .config import RecoveryConfig
from .context import build_context, source_snapshot
from .generator import RecoveryGenerator
from .program import RecoveryProgram


_LOG = logging.getLogger("habitat_mas.recovery")
if not _LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    _LOG.addHandler(_handler)
_LOG.setLevel(logging.INFO)
_LOG.propagate = False


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _distance(a, b):
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != len(b):
        return 0.0
    try:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    except (TypeError, ValueError):
        return 0.0


def _progress(before, after):
    reduced = []
    for key, initial in before.get("metrics", {}).items():
        current = after.get("metrics", {}).get(key)
        if ("distance" in key and isinstance(initial, (int, float)) and not isinstance(initial, bool)
                and isinstance(current, (int, float)) and not isinstance(current, bool)
                and initial - current > .05):
            reduced.append({"metric": key, "before": initial, "after": current})
    moved = [name for name, value in before.get("agents", {}).items()
             if value and _distance(value.get("position"), (after.get("agents", {}).get(name) or {}).get("position")) > .05]
    success = after.get("metrics", {}).get("pddl_success")
    new_success = success in (True, 1) and before.get("metrics", {}).get("pddl_success") not in (True, 1)
    status = "progress_observed" if reduced or new_success else "state_changed" if moved else "no_observed_progress"
    return {"status": status, "goal_distance_reduced": reduced, "robots_moved": moved,
            "current_success_predicate": success,
            "interpretation": "Observable progress is feedback, not proof of recovery or causal attribution."}


class RecoveryController:
    def __init__(self, config, output_dir, detector=None, generator=None):
        self.config = RecoveryConfig.from_value(config)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        if detector is None:
            if cfg.detector == "llm":
                detector = LLMDetector(model=cfg.model, window=cfg.window, check_every=cfg.check_every,
                    timeout=cfg.model_timeout, audit_path=self.output_dir / "detection_requests.jsonl",
                    cache_dir=self.output_dir / "detection_cache")
            else:
                detector = TeamProgressDetector(window=cfg.window, check_every=cfg.check_every,
                    min_stall_steps=cfg.min_stall_steps, reset_budget=cfg.reset_budget)
        self.detector = detector
        self.generator = generator if generator is not None else RecoveryGenerator(model=cfg.model, timeout=cfg.model_timeout)
        self.versions = []
        self.memory = {}
        self.active_code = None
        self._active_program = None
        self._active_version = None
        self._previous = []
        self._recent = deque(maxlen=cfg.window)
        self._pending_events = {"decisions": [], "messages": []}
        self._pending_assessments = []
        self._count = 0
        self._last_step = None
        self._last_apply_step = None
        self._last_generation_step = None
        self._last_error = None
        self._sources = None

    def _event(self, event, **data):
        row = redact({"event": event, "step": self._last_step, **data})
        with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        # Exclude request/source/context fields from console diagnostics; only
        # event metadata and a bounded, redacted error first line are shown.
        visible = {"detection", "generation_started", "generation_rejected", "application",
                   "source_generation_started", "source_generation_rejected", "source_application",
                   "source_unrepairable", "verification", "source_budget_exhausted",
                   "execution_exception", "source_rollback", "source_consistency_error", "rollback"}
        if event not in visible:
            return
        summary = [event]
        for name in ("step", "iteration", "status", "attempts"):
            if name in row:
                summary.append("%s=%s" % (name, row[name]))
        if event == "detection":
            decision = row.get("decision") or {}
            prediction = decision.get("system_error")
            if isinstance(prediction, bool):
                summary.append("system_error=%s" % prediction)
            elif decision.get("status") in {"normal", "system_error"}:
                summary.append("system_error=%s" % (decision["status"] == "system_error"))
            else:
                summary.append("decision_status=%s" % decision.get("status", "unknown"))
        result = row.get("result")
        if isinstance(result, dict) and "status" in result:
            summary.append("result_status=%s" % result["status"])
        if row.get("error"):
            error = row["error"]
            if isinstance(error, dict):
                error = "%s: %s" % (error.get("type", "Error"), error.get("message", ""))
            summary.append("error=%s" % str(error).splitlines()[0][:200])
        level = logging.WARNING if ("rejected" in event or event in {
            "source_budget_exhausted", "source_unrepairable", "execution_exception",
            "source_rollback", "source_consistency_error", "rollback"}) else logging.INFO
        _LOG.log(level, "Recovery %s", " ".join(summary))

    def _save_version(self, version):
        _write(self.output_dir / "versions" / ("%03d" % version["iteration"]) / "version.json", version)

    def _assess(self, frame, terminal):
        waiting = []
        for item in self._pending_assessments:
            if not terminal and frame["step"] - item["before"]["step"] < self.config.verify_after_steps:
                waiting.append(item)
                continue
            outcome = {"step": frame["step"], "applied_step": item["before"]["step"],
                       "terminal": terminal, **_progress(item["before"], frame)}
            item["version"].setdefault("outcomes", []).append(outcome)
            self._save_version(item["version"])
            self._event("verification", iteration=item["version"]["iteration"], **outcome)
        self._pending_assessments = waiting

    def _context(self, frame, policy):
        if self._sources is None:
            self._sources = source_snapshot()
        context = build_context(frame, list(self._recent), self._last_error, policy, self.config,
                                self.versions, self.memory, sources=self._sources)
        context["code"]["active_recovery"] = self.active_code
        context["recovery"]["active_iteration"] = self._active_version["iteration"] if self._active_version else None
        return context

    def _apply(self, result, frame, policy, version):
        commands = result["commands"]
        if commands:
            policy.validate_recovery_commands(commands)
        applied = policy.apply_recovery_commands(commands) if commands else {"applied": [], "simulator_reset": False}
        self.memory = deepcopy(result["memory"])
        if commands:
            self._last_apply_step = frame["step"]
            self._pending_assessments.append({"before": deepcopy(frame), "version": version})
        status = "applied" if commands else "active_no_command"
        self._event("application", iteration=version["iteration"], status=status, commands=commands, result=applied)
        return {"status": status, "iteration": version["iteration"], "commands": commands, "result": applied}

    def _generate(self, frame, policy):
        self._last_generation_step = frame["step"]
        version = {"iteration": len(self.versions) + 1, "step": frame["step"], "status": "generating", "outcomes": []}
        directory = self.output_dir / "versions" / ("%03d" % version["iteration"])
        context = self._context(frame, policy)
        self.versions.append(version)
        _write(directory / "context.json", context)
        self._event("generation_started", iteration=version["iteration"])
        try:
            proposal = self.generator.generate(context)
            _write(directory / "generation.json", proposal)
            version.update(code=proposal["code"], explanation=proposal.get("explanation"))
            (directory / "recovery.py").write_text(proposal["code"], encoding="utf-8")
            program = RecoveryProgram(proposal["code"])
            result = program.run(context, self.memory, timeout=self.config.execution_timeout)
            _write(directory / "candidate_result.json", result)
            if result["commands"]:
                policy.validate_recovery_commands(result["commands"])
            # Only install after code execution and whole-batch validation.
            previous = (self.active_code, self._active_program, self._active_version, deepcopy(self.memory))
            applied = self._apply(result, frame, policy, version)
            if previous[1] is not None:
                self._previous.append(previous)
                previous[2]["status"] = "superseded"
                self._save_version(previous[2])
            self.active_code, self._active_program, self._active_version = proposal["code"], program, version
            version["status"] = "active"
        except Exception as error:
            version.update(status="rejected", error=redact(str(error)), error_type=type(error).__name__)
            if hasattr(error, "response"):
                _write(directory / "generation_error.json", {"request": getattr(error, "request", None), "response": error.response})
            self._event("generation_rejected", iteration=version["iteration"], error=version["error"])
            applied = {"status": "rejected", "iteration": version["iteration"], "error": version["error"]}
        self._save_version(version)
        return applied

    def _run_active(self, frame, policy):
        try:
            result = self._active_program.run(self._context(frame, policy), self.memory, timeout=self.config.execution_timeout)
            return self._apply(result, frame, policy, self._active_version)
        except Exception as error:
            failed = self._active_version
            failed.update(status="rolled_back", runtime_error=redact(str(error)))
            self._save_version(failed)
            if self._previous:
                self.active_code, self._active_program, self._active_version, self.memory = self._previous.pop()
                self._active_version["status"] = "active"
                self._save_version(self._active_version)
            else:
                self.active_code = self._active_program = self._active_version = None
                self.memory = {}
            self._event("rollback", failed_iteration=failed["iteration"], error=redact(str(error)))
            return {"status": "rolled_back", "error": redact(str(error))}

    def observe(self, frame, policy, terminal=False):
        clean = _frame(frame)
        if self._last_step is not None and clean["step"] <= self._last_step:
            raise ValueError("Recovery frames must strictly increase within an episode")
        self._last_step = clean["step"]
        # _frame deliberately excludes initial task context; retain its allowed
        # context separately for the structured recovery prompt.
        clean["task"] = frame.get("task")
        clean["context"] = deepcopy(frame.get("context") or {})
        self._assess(clean, terminal)
        if terminal:
            self._event("terminal_observation", handling="no_recovery_after_auto_reset")
            return {"status": "terminal_skipped"}
        for key in self._pending_events:
            self._pending_events[key].extend(clean[key])
        index, self._count = self._count, self._count + 1
        if index % self.config.sample_every:
            return None
        for key in self._pending_events:
            clean[key], self._pending_events[key] = self._pending_events[key], []
        self._recent.append(deepcopy(clean))
        return self._observe_sample(clean, policy)

    def _observe_sample(self, clean, policy):
        decision = self.detector.observe(clean)
        if decision:
            self._event("detection", decision=decision)
        can_apply = self._last_apply_step is None or clean["step"] - self._last_apply_step >= self.config.cooldown_steps
        can_generate = self._last_generation_step is None or clean["step"] - self._last_generation_step >= self.config.cooldown_steps
        if decision and decision.get("status") == "system_error":
            self._last_error = decision
            if can_apply and can_generate and len(self.versions) < self.config.max_iterations:
                return self._generate(clean, policy)
        if self._active_program is not None and can_apply:
            return self._run_active(clean, policy)
        return None
