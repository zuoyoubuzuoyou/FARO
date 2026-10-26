"""Incremental research traces; hooks observe execution and never run detectors.

``env_step`` identifies the pending/returned env.step transition (1 based).
``decision_step`` counts top-level chat decisions separately for each agent.
A controller action can span many transitions and keeps one ``action_id``.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path, PurePath
import re
import time
from typing import Any, Iterator
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from uuid import uuid4
from zoneinfo import ZoneInfo

# Also support dependency-free loading via importlib.spec_from_file_location.
if __package__:
    from .trajectory_storage import StepTraceStore
else:
    import importlib.util
    _storage_spec = importlib.util.spec_from_file_location(
        "_standalone_trajectory_storage", Path(__file__).with_name("trajectory_storage.py"))
    _storage_module = importlib.util.module_from_spec(_storage_spec)
    _storage_spec.loader.exec_module(_storage_module)
    StepTraceStore = _storage_module.StepTraceStore

SCHEMA_VERSION = 3
_ACTIVE_WRITER = ContextVar("evaluation_trace_writer", default=None)
_DECISION = ContextVar("evaluation_trace_decision", default=None)
_OBSERVATIONS = ContextVar("evaluation_trace_observations", default=None)
_ENVELOPE_KEYS = frozenset({
    "schema_version", "run_id", "scene_id", "episode_id", "env_index",
    "evaluation_id", "sequence", "event", "event_id", "event_type", "step",
    "env_step", "timestamp", "elapsed_seconds",
})
_SECRET_KEYS = {
    "api_key", "apikey", "authorization", "password", "passwd", "secret",
    "client_secret", "access_token", "refresh_token", "credentials", "credential",
    "token", "signature", "x_amz_signature", "sig",
    "private_key", "aws_access_key_id", "aws_secret_access_key",
    "aws_session_token", "auth_token", "bearer_token",
}
_LOG = logging.getLogger(__name__)


def _secret_key(key):
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    normalized = re.sub(r"[^a-z0-9]+", "_", separated.lower()).strip("_")
    return normalized in _SECRET_KEYS or normalized.endswith(("_api_key", "_password", "_secret", "_access_token", "_credential", "_security_token"))


def redact(value):
    """Remove credential fields, URL credentials and known environment secrets."""
    secrets = [v for k, v in os.environ.items() if _secret_key(k) and len(v) >= 4]

    def visit(item):
        if isinstance(item, Mapping):
            return {str(k): "[REDACTED]" if _secret_key(k) else visit(v)
                    for k, v in item.items()}
        if isinstance(item, (list, tuple, set, frozenset)):
            return [visit(v) for v in item]
        if not isinstance(item, str):
            return item
        for secret in secrets:
            item = item.replace(secret, "[REDACTED]")
        item = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[REDACTED]", item)
        item = re.sub(r"(?i)Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", item)
        def scrub_url(match):
            try:
                parts = urlsplit(match.group())
                host = parts.netloc.rsplit("@", 1)[-1]
                query = [(k, "[REDACTED]" if _secret_key(k) else v)
                         for k, v in parse_qsl(parts.query, keep_blank_values=True)]
                return urlunsplit((parts.scheme, host, parts.path, urlencode(query), parts.fragment))
            except ValueError:
                return "[REDACTED_URL]"
        return re.sub(r"https?://[^\s\"'<>]+", scrub_url, item)
    return visit(value)


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, PurePath):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_value(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _json_value(dump())
    converted = value
    for name in ("detach", "cpu"):
        method = getattr(converted, name, None)
        if callable(method):
            converted = method()
    for name in ("tolist", "item"):
        method = getattr(converted, name, None)
        if callable(method):
            result = method()
            if result is not value and result is not converted:
                return _json_value(result)
    raise TypeError(f"Unsupported trajectory value: {type(value).__name__}")


def _safe_value(value, missing, path=""):
    try:
        return _json_value(value)
    except (TypeError, ValueError, RuntimeError):
        if isinstance(value, Mapping):
            return {str(k): _safe_value(v, missing, f"{path}.{k}".strip(".")) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_safe_value(v, missing, f"{path}[{i}]") for i, v in enumerate(value)]
        missing[path or "value"] = f"Unavailable JSON representation for {type(value).__name__}"
        return None


def _encode(value):
    return json.dumps(redact(_json_value(value)), ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _allocate_directory(parent, name, suffix="_"):
    """Reserve a readable directory atomically, including colliding attempts."""
    attempt = 1
    while True:
        candidate = parent / (name if attempt == 1 else f"{name}{suffix}{attempt:02d}")
        try:
            candidate.mkdir()
        except FileExistsError:
            attempt += 1
        else:
            return candidate


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_encode(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _guard(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        # Logging failures must not replace task exceptions or choose actions.
        _LOG.warning("Trajectory hook failed (%s): %s", type(error).__name__, redact(str(error)))
        return None


def record_event(event_type, **data):
    binding = _ACTIVE_WRITER.get()
    if binding is not None and not binding[0].closed:
        writer, step = binding
        return _guard(writer.write, event_type, step, **data)
    return None


def current_action_id(agent_id):
    binding = _ACTIVE_WRITER.get()
    if binding is None:
        return None
    writer = binding[0]
    action = writer.pending_actions.get(agent_id) or writer.active_actions.get(agent_id)
    return action["action_id"] if action else None


@contextmanager
def decision_scope(agent_id, phase="execution"):
    binding = _ACTIVE_WRITER.get()
    if binding is None:
        yield
        return
    writer, step = binding
    previous = _DECISION.get()
    if previous and previous[:3] == (writer, agent_id, phase):
        yield
        return
    writer.decision_counts[agent_id] = writer.decision_counts.get(agent_id, 0) + 1
    phases = writer.decisions_by_phase.setdefault(agent_id, {})
    phases[phase] = phases.get(phase, 0) + 1
    if phase == "execution":
        writer.pending_actions[agent_id] = {
            "action_id": str(uuid4()), "decision_step": writer.decision_counts[agent_id],
            "action_name": None, "parameters": None,
        }
    token = _DECISION.set((writer, agent_id, phase))
    _guard(writer.capture_decision_media, agent_id, step, _OBSERVATIONS.get())
    record_event("decision_start", agent_name=agent_id, phase=phase)
    started = time.monotonic()
    try:
        yield
    except BaseException:
        record_event("decision_end", agent_name=agent_id, phase=phase, status="error",
                     duration_seconds=time.monotonic() - started, handling="propagate_original_exception")
        raise
    else:
        record_event("decision_end", agent_name=agent_id, phase=phase, status="returned",
                     duration_seconds=time.monotonic() - started)
    finally:
        _DECISION.reset(token)


def begin_action(agent_id, action_name, parameters, **data):
    binding = _ACTIVE_WRITER.get()
    if binding is None:
        return None
    return _guard(binding[0].begin_action, agent_id, action_name, parameters, binding[1], **data)


def end_action(agent_id, status, reason=None, result=None, defer=True):
    binding = _ACTIVE_WRITER.get()
    if binding is not None:
        return _guard(binding[0].end_action, agent_id, status, reason, result, binding[1], defer)
    return None


class _EpisodeWriter:
    def __init__(self, owner, env_index, episode, context, state):
        self.owner = owner
        self.identity = {
            "schema_version": SCHEMA_VERSION, "run_id": owner.run_id,
            "scene_id": episode.scene_id, "episode_id": episode.episode_id,
            "env_index": env_index, "evaluation_id": str(uuid4()),
        }
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(episode.episode_id))[:80].strip(".") or "episode"
        identifier = str(episode.episode_id)
        if re.fullmatch(r"[0-9]+", identifier):
            safe_id = identifier.lstrip("0")[:80].zfill(4)
        self.directory = _allocate_directory(
            owner.run_dir / "episodes", f"episode_{safe_id}", suffix="_attempt_")
        (self.directory / "media").mkdir()
        self.num_steps = 0
        self.total_reward = 0.0
        self.metrics = {}
        self.state = state
        self.sequence = 0
        self.last_event_step = 0
        self.started = time.monotonic()
        self.closed = False
        self.decision_counts = {}
        self.decisions_by_phase = {}
        self.active_actions = {}
        self.pending_actions = {}
        self.deferred_ends = {}
        self.last_actions = {}
        self.submissions = {}
        self.media = []
        self.exception_count = 0
        self.exception_counts = {}
        self.error_signatures = set()
        self.error_contexts = set()
        self.exception_event_count = 0
        self.logging_error_count = 0
        self.success = None
        self.termination_reason = None
        self.missing_fields = {}
        if state is None:
            self.missing_fields["initial_state"] = "Environment did not supply an initial state"
        self.files = {"config": "../../config.json", "episode": "episode.json",
                      "steps": "steps.jsonl", "summary": "summary.json", "media": "media/"}
        snapshot = {**self.identity, "timestamp": _timestamp(), "context": context,
                    "initial_state": state, "success_measure": owner.metadata.get("success_measure"),
                    "missing_fields": dict(self.missing_fields), "files": self.files}
        _write_json(self.directory / "episode.json", self.prepare(snapshot))
        self.store = StepTraceStore(self.directory, self.identity)
        self.write("episode_start", 0, context=context, state=state)

    def _save_bytes(self, content, suffix, source, seen_by_llm):
        digest = hashlib.sha256(content).hexdigest()
        relative = f"media/{digest}.{suffix}"
        path = self.directory / relative
        if not path.exists():
            path.write_bytes(content)
        reference = {"media_path": relative, "sha256": digest, "source": source,
                     "seen_by_llm": seen_by_llm}
        if reference not in self.media:
            self.media.append(reference)
        return reference

    def prepare(self, value, model_input=False):
        missing = {}
        value = _safe_value(value, missing)

        def images(item, image_reference=False):
            if isinstance(item, list):
                return [images(v) for v in item]
            if not isinstance(item, dict):
                return item
            result = {k: images(v, image_reference=(k == "image_url")) for k, v in item.items()}
            url = result.get("url")
            if image_reference and isinstance(url, str) and url.startswith("data:image/"):
                try:
                    header, encoded = url.split(",", 1)
                    mime = header[5:].split(";", 1)[0]
                    if ";base64" not in header:
                        raise ValueError("Only base64 image data URLs are supported")
                    content = base64.b64decode(encoded, validate=True)
                    suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}.get(mime, "image")
                    result.update(self._save_bytes(content, suffix, "model_input" if model_input else "logged_image", model_input))
                    result["url"] = None
                    result["url_storage"] = "Original image bytes saved at media_path"
                except Exception as error:
                    result["url"] = None
                    result["media_path"] = None
                    result["missing_reason"] = f"Image could not be saved: {type(error).__name__}"
            elif image_reference and isinstance(url, str) and url.startswith(("http://", "https://")):
                # Fetch an actual model input image, never substitute a simulator view.
                try:
                    from urllib.request import urlopen
                    with urlopen(url, timeout=10) as response:
                        content = response.read(32 * 1024 * 1024 + 1)
                        if len(content) > 32 * 1024 * 1024:
                            raise ValueError("Image exceeded 32 MiB")
                        mime = response.headers.get_content_type()
                    suffix = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "image")
                    result.update(self._save_bytes(content, suffix, "model_input_url" if model_input else "logged_image_url", model_input))
                except Exception as error:
                    result["media_path"] = None
                    result["missing_reason"] = f"Remote model image unavailable: {type(error).__name__}"
            return result
        value = images(value)
        if missing and isinstance(value, dict):
            value.setdefault("missing_fields", {}).update(missing)
        return redact(value)

    def capture_decision_media(self, agent_id, step, observations):
        references = []
        missing = {}
        if observations is not None:
            for name, tensor in observations.items():
                if "rgb" not in name.lower():
                    continue
                if name.startswith("agent_") and not name.startswith(agent_id + "_"):
                    continue
                try:
                    import numpy as np
                    from PIL import Image
                    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
                    if array.ndim == 4:
                        array = array[0]
                    if array.ndim != 3:
                        raise ValueError("RGB observation is not an HWC/CHW image")
                    if array.shape[-1] not in (3, 4) and array.shape[0] in (3, 4):
                        array = array.transpose(1, 2, 0)
                    if np.issubdtype(array.dtype, np.floating) and array.size and array.min() >= 0 and array.max() <= 1:
                        array = array * 255
                    array = np.clip(array, 0, 255).astype(np.uint8)
                    output = io.BytesIO()
                    Image.fromarray(array).save(output, format="PNG")
                    references.append(self._save_bytes(output.getvalue(), "png", name, False))
                except Exception as error:
                    missing[name] = f"Decision observation image unavailable: {type(error).__name__}"
        if not references and not missing:
            missing["decision_observation_images"] = "No RGB observation supplied to the policy tracing scope"
        self.write("decision_media", step, agent_name=agent_id, media=references,
                   observation_env_step=max(0, step - 1), missing_fields=missing)

    def write(self, event_type, step, /, **data):
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("An event type must be a nonempty string")
        reserved = _ENVELOPE_KEYS.intersection(data)
        if reserved:
            raise ValueError(f"Reserved trajectory fields: {', '.join(sorted(reserved))}")
        agent_id = data.pop("agent_id", data.get("agent_name"))
        if agent_id is None and _DECISION.get() and _DECISION.get()[0] is self:
            agent_id = _DECISION.get()[1]
        action = self.pending_actions.get(agent_id) or self.active_actions.get(agent_id)
        action_id = data.pop("action_id", action["action_id"] if action else None)
        decision = data.pop("decision_step", self.decision_counts.get(agent_id) if agent_id is not None else None)
        payload = self.prepare({
            **self.identity, "event_id": str(uuid4()), "sequence": self.sequence,
            "event_type": event_type, "event": event_type, "timestamp": _timestamp(),
            "env_step": step, "step": step, "agent_id": agent_id,
            "decision_step": decision, "action_id": action_id,
            "elapsed_seconds": time.monotonic() - self.started, **data,
        }, model_input=(event_type == "llm_request"))
        if agent_id is None or action_id is None or decision is None:
            absent = payload.setdefault("missing_fields", {})
            if agent_id is None:
                absent.setdefault("agent_id", "Episode/environment event has no single agent")
            if action_id is None:
                absent.setdefault("action_id", "No execution action is associated with this event")
            if decision is None:
                absent.setdefault("decision_step", "No agent decision is associated with this event")
        self.store.append(payload)
        self.sequence += 1
        self.last_event_step = step
        has_error = bool(payload.get("error_type") or payload.get("error"))
        if event_type.endswith("_error") or event_type in ("parse_failure", "exception", "action_failure") or (payload.get("status") == "error" and has_error):
            error = payload.get("error") or {}
            signature = (step, payload.get("error_type", error.get("type")), payload.get("error_message", error.get("message")))
            context = (signature, agent_id, action_id, decision)
            if payload.get("error_origin") == "logging":
                self.logging_error_count += 1
            else:
                self.exception_event_count += 1
                if context not in self.error_contexts:
                    self.exception_count += 1
                    self.exception_counts[event_type] = self.exception_counts.get(event_type, 0) + 1
                    self.error_contexts.add(context)
                self.error_signatures.add(signature)
        if event_type in ("model_config", "llm_request", "llm_response"):
            self.owner.register_model(payload)
        if agent_id is not None and event_type == "tool_selection" and agent_id in self.pending_actions:
            self.pending_actions[agent_id].update(
                action_name=payload.get("action_name"), parameters=payload.get("parameters"),
                status="selected_not_entered",
            )
        if self.owner.event_callback is not None:
            # Callbacks only enqueue detached events; online work runs after
            # the returned environment transition, outside logging hooks.
            try:
                self.owner.event_callback(deepcopy(payload))
            except Exception as error:
                # Observer failures must not interrupt a completed transition
                # or strand the next step behind an uncommitted packet.
                self.logging_error_count += 1
                _LOG.warning("Trajectory observer failed (%s): %s", type(error).__name__, redact(str(error)))
        return payload

    def begin_action(self, agent_id, name, parameters, step, **data):
        if agent_id in self.active_actions:
            self._finish_action(agent_id, "superseded", "New controller action entered", None, step)
        action = self.pending_actions.pop(agent_id, None) or {"action_id": str(uuid4()), "decision_step": self.decision_counts.get(agent_id)}
        action.update(action_name=name, parameters=parameters, status="started", started_step=step)
        self.active_actions[agent_id] = action
        self.last_actions[agent_id] = dict(action)
        self.write("action_start", step, agent_name=agent_id, **data, **action)
        return action["action_id"]

    def _finish_action(self, agent_id, status, reason, result, step):
        action = self.active_actions.pop(agent_id, None)
        if action is None:
            return
        action = {**action, "status": status, "reason": reason, "result": result}
        self.last_actions[agent_id] = action
        self.write("action_end", step, agent_name=agent_id, **action)

    def end_action(self, agent_id, status, reason, result, step, defer):
        if defer:
            self.deferred_ends[agent_id] = (status, reason, result)
        else:
            self._finish_action(agent_id, status, reason, result, step)

    def submit_actions(self, action, lengths, agent_ids):
        step = self.num_steps + 1
        self.write("env_action_submit", step, action=action,
                   active_action_ids={a: v["action_id"] for a, v in self.active_actions.items()})
        normalized = _json_value(action)
        offset = 0
        for index, agent_id in enumerate(agent_ids or self.active_actions):
            size = lengths[index] if lengths is not None and index < len(lengths) else None
            if size is not None and isinstance(normalized, list):
                command = normalized[offset:offset + size]
                offset += size
            elif len(agent_ids or self.active_actions) == 1:
                command = normalized
            else:
                command = None
            self.submissions[agent_id] = command
            self.write("action_submit", step, agent_name=agent_id, action=command,
                       execution_status="submitted", missing_fields={} if command is not None else {"action": "Per-agent vector layout unavailable; full command is in env_action_submit"})

    def record_step(self, action, reward, done, metrics, state, success=None, termination_reason=None):
        step = self.num_steps + 1
        payload = self.write("step", step, action=action, reward=reward,
                             done=bool(done), metrics=metrics, state=state)
        self.num_steps = step
        value = payload["reward"]
        self.total_reward += float(value) if isinstance(value, (int, float)) else math.nan
        self.metrics = payload["metrics"]
        self.state = payload["state"]
        for agent_id, command in self.submissions.items():
            self.write("action_execution", step, agent_name=agent_id, action=command,
                       execution_status="returned", result={"reward": reward, "done": bool(done), "metrics": metrics},
                       state=state.get("agents", {}).get(agent_id) if state else None,
                       executed_action=state.get("executed_action") if state else None)
        self.submissions.clear()
        for agent_id, (status, reason, result) in list(self.deferred_ends.items()):
            self._finish_action(agent_id, status, reason,
                                result if result is not None else {"metrics": self.metrics, "state": self.state}, step)
        self.deferred_ends.clear()
        if done:
            self.success = _json_value(success)
            self.termination_reason = termination_reason or (state or {}).get("termination_reason")
            self.finish("completed")
        else:
            self.store.commit_step()

    def finish(self, status, error=None):
        if self.closed:
            return
        if error is not None:
            error = {"type": type(error).__name__, "message": str(error)} if isinstance(error, BaseException) else error
            error = redact(error)
            signature = (self.num_steps + 1, error.get("type"), error.get("message"))
            if signature not in self.error_signatures:
                self.write("exception", self.num_steps + 1, error=error, handling="propagate_original_exception")
        for agent_id in list(self.active_actions):
            self._finish_action(agent_id, "error" if error else "episode_terminated",
                                self.termination_reason or status, {"state": self.state, "metrics": self.metrics, "error": error},
                                max(self.last_event_step, self.num_steps + (1 if self.submissions else 0)))
        for agent_id, action in self.pending_actions.items():
            final = {**action, "status": "not_entered", "controller_entered": False,
                     "reason": "Exception before controller entry" if error else "Episode ended before controller entry",
                     "result": None}
            self.last_actions[agent_id] = final
            self.write("action_end", self.num_steps + 1, agent_name=agent_id, **final)
        self.pending_actions.clear()
        missing = dict(self.missing_fields)
        if self.success is None:
            missing["task_success"] = "Original evaluator success measure was not returned before termination"
        if self.state is None:
            missing["final_state"] = "Environment did not return a state snapshot"
        if self.termination_reason is None:
            missing["termination_reason"] = "No original environment termination reason; see status/error"
        agent_states = (self.state or {}).get("agents", {})
        known_agents = set(self.decision_counts) | set(agent_states)
        known_agents.update(a.get("agent_id", a.get("id")) for a in self.owner.metadata.get("agents", []) if isinstance(a, dict))
        for agent_id in known_agents - {None}:
            if agent_id not in self.last_actions:
                self.last_actions[agent_id] = None
                missing[f"last_actions.{agent_id}"] = "No controller action was entered for this agent"
            if agent_id not in agent_states:
                agent_states[agent_id] = None
                missing[f"last_agent_states.{agent_id}"] = "No final state was returned for this agent"
        summary = self.prepare({
            **self.identity, "status": status, "task_success": self.success,
            "num_env_steps": self.num_steps, "num_steps": self.num_steps,
            "total_reward": self.total_reward, "decision_counts": self.decision_counts,
            "decisions_by_phase": self.decisions_by_phase,
            "total_decisions": sum(self.decision_counts.values()),
            "elapsed_seconds": time.monotonic() - self.started,
            "termination_reason": self.termination_reason,
            "last_actions": self.last_actions, "final_state": self.state,
            "last_agent_states": agent_states,
            "exception_count": self.exception_count, "exception_counts": self.exception_counts,
            "exception_event_count": self.exception_event_count, "logging_error_count": self.logging_error_count,
            "metrics": self.metrics, "error": error, "files": self.files,
            "media": self.media, "missing_fields": missing,
        })
        try:
            # Persist the terminal event before committing its returned step.
            fields = {k: v for k, v in summary.items() if k not in self.identity and k != "elapsed_seconds"}
            self.write("episode_end", max(self.num_steps, self.last_event_step), **fields)
            if status == "completed":
                self.store.commit_step()
            summary["logging_error_count"] = self.logging_error_count
            self.store.write_summary(summary)
        finally:
            self.closed = True
            self.store.close()


class EvaluationTrace:
    def __init__(self, output_dir, metadata=None, event_callback=None):
        missing = {}
        self.metadata = redact(_safe_value(metadata or {}, missing))
        if missing:
            self.metadata.setdefault("missing_fields", {}).update(missing)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = str(uuid4())
        run_name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = _allocate_directory(output_dir, run_name)
        (self.run_dir / "episodes").mkdir()
        self._episodes = {}
        self._closed = False
        self._models = {}
        self._model_call_keys = {}
        self.event_callback = event_callback
        self.config = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                       "timestamp": _timestamp(), "metadata": self.metadata, "models_used": [],
                       "episodes": [],
                       "logging_semantics": {
                           "env_step": "1-based pending/returned env.step transition; episode_start=0",
                           "decision_step": "Per-agent top-level chat call; nested calls share a counter; phase disambiguates discussion/planning/execution",
                           "action_id": "One decision/controller action across many transitions, including aborted decisions",
                           "exception_count": "Distinct (transition, agent, action/decision, error type, message) signatures; raw catch events also retained",
                           "model_events": "Associated for the single environment supported by MultiLLMPolicy",
                       }}
        self._save_config()

    def _save_config(self):
        _write_json(self.run_dir / "config.json", self.config)

    def register_model(self, event):
        if event["event_type"] == "llm_response":
            item = self._models.get(self._model_call_keys.get(event.get("call_id")))
            response_model = event.get("response_model")
            if item is not None and response_model is not None and response_model not in item["response_model_ids"]:
                item["response_model_ids"].append(response_model)
                self.config["models_used"] = list(self._models.values())
                _guard(self._save_config)
            return
        if event["event_type"] == "llm_request" and event.get("call_id") in self._model_call_keys:
            return
        names = ("temperature", "top_p", "max_tokens", "max_completion_tokens", "frequency_penalty", "presence_penalty", "seed", "stop", "n", "response_format", "reasoning_effort", "logprobs", "top_logprobs", "parallel_tool_calls", "stream")
        decoding = event.get("decoding_parameters") or {k: event.get(k) for k in names}
        prompt = next((m.get("content") for m in event.get("messages", []) if isinstance(m, dict) and m.get("role") == "system"), None)
        item = {"agent_id": event["agent_id"], "phase": event.get("phase"), "model_id": event.get("model"),
                "decoding_parameters": decoding,
                "response_model_ids": [],
                "api_timeout_seconds": event.get("api_timeout_seconds"),
                "api_max_retries": event.get("api_max_retries"),
                "system_prompt_sha256": event.get("system_prompt_sha256") or (hashlib.sha256(prompt.encode("utf-8")).hexdigest() if isinstance(prompt, str) else None),
                "missing_fields": {f"decoding_parameters.{k}": "Not passed by existing code; provider default unavailable" for k, v in decoding.items() if v is None}}
        for name in ("api_timeout_seconds", "api_max_retries"):
            if item[name] is None:
                item["missing_fields"][name] = "SDK client configuration was not exposed by the observed call"
        # model_config and llm_request describe the same call. The latter
        # carries the complete prompt, while the former supplies its hash.
        key = json.dumps({k: item[k] for k in ("agent_id", "phase", "model_id", "system_prompt_sha256", "decoding_parameters", "api_timeout_seconds", "api_max_retries")}, sort_keys=True)
        if event.get("call_id") is not None:
            self._model_call_keys[event["call_id"]] = key
        if key not in self._models:
            self._models[key] = item
            self.config["models_used"] = list(self._models.values())
            _guard(self._save_config)

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        _guard(self.close, "error" if exc_type else "interrupted", exc)
        return False

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("This evaluation trace is closed")

    def has_episode(self, env_index):
        return env_index in self._episodes

    def start_episode(self, env_index, episode, context=None, state=None):
        self._ensure_open()
        if env_index in self._episodes:
            raise ValueError(f"Environment {env_index} already has an active episode")
        writer = _EpisodeWriter(self, env_index, episode, context, state)
        self._episodes[env_index] = writer
        self.config["episodes"].append({
            **writer.identity,
            "directory": str(writer.directory.relative_to(self.run_dir)),
        })
        _guard(self._save_config)

    @contextmanager
    def policy_step(self, observations=None) -> Iterator[None]:
        self._ensure_open()
        writer = self._episodes.get(0) if len(self._episodes) == 1 else None
        token = _ACTIVE_WRITER.set((writer, writer.num_steps + 1) if writer else None)
        observation_token = _OBSERVATIONS.set(observations)
        try:
            yield
        finally:
            _OBSERVATIONS.reset(observation_token)
            _ACTIVE_WRITER.reset(token)

    def submit_actions(self, env_index, action, action_lengths=None, agent_ids=None):
        self._ensure_open()
        writer = self._episodes.get(env_index)
        if writer is not None:
            _guard(writer.submit_actions, action, action_lengths, agent_ids)

    def record_environment_event(self, env_index, event_type, **data):
        writer = self._episodes.get(env_index)
        if writer is not None:
            return _guard(writer.write, event_type, writer.num_steps + 1, **data)

    def record_step(self, env_index, action, reward, done, metrics, state, success=None, termination_reason=None):
        self._ensure_open()
        writer = self._episodes.get(env_index)
        if writer is None:
            return
        _guard(writer.record_step, action, reward, done, metrics, state, success, termination_reason)
        if writer.closed:
            del self._episodes[env_index]

    def pause_envs(self, indices):
        self._ensure_open()
        removed = set(indices)
        if any(not isinstance(i, int) or i < 0 for i in removed):
            raise ValueError("Paused environment indices must be nonnegative integers")
        for index in removed:
            writer = self._episodes.get(index)
            if writer is not None:
                _guard(writer.finish, "paused")
        self._episodes = {i - sum(r < i for r in removed): w for i, w in self._episodes.items() if i not in removed}

    def close(self, status="interrupted", error=None):
        if self._closed:
            return
        self._closed = True
        if error is not None and status == "interrupted":
            status = "error"
        for writer in self._episodes.values():
            _guard(writer.finish, status, error)
        self._episodes.clear()
