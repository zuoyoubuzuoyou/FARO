"""Bridge detached trace observations into recovery between live policy steps."""
from collections import deque
from copy import deepcopy
from pathlib import Path
import re

from habitat_mas.detection.readers import FrameStream
from habitat_mas.recovery.controller import RecoveryController
from habitat_mas.recovery.source_controller import SourceRecoveryController
from habitat_mas.recovery.source_runtime import ExecutionCheckpoint, SourceRuntime
from habitat_mas.utils.trajectory import redact


def _task_name(hint, context):
    value = str(hint or "auto").lower()
    if value == "auto":
        value = str(context.get("task_description", context.get("pddl_text_goal", ""))).lower()
    if re.search(r"perception|(?:^|[_/])per(?:$|[_/])|detect|recogniz|identify", value):
        return "Perception"
    if re.search(r"mobility|(?:^|[_/])mob(?:$|[_/])|visit|navigate|reach", value):
        return "Mobility"
    return "Manipulation"


class OnlineRecovery:
    """Recover between steps, or retry a failed decision before any env.step.

    A new controller is created for each episode attempt. The recorder callback
    never queries a model, executes generated code, or mutates a live policy.
    """

    def __init__(self, config, output_dir, task="auto", envs=None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.task = task
        self._events = deque()
        self._stream = None
        self._controller = None
        self._latest_frame = None
        self._terminal = False
        self.mode = getattr(config, "mode", "source")
        self.runtime = SourceRuntime(envs=envs,
            persist_source_changes=getattr(config, "persist_source_changes", False)) if self.mode == "source" else None

    def on_event(self, payload):
        self._events.append(deepcopy(redact(payload)))

    def _drain(self, policy, dones=None):
        results = []
        observations = 0
        while self._events:
            payload = self._events.popleft()
            if payload.get("env_index", 0) != 0:
                raise ValueError("Online recovery currently requires a single environment")
            kind = payload.get("event_type", payload.get("event"))
            if kind == "episode_start":
                context = payload.get("context") or {}
                self._stream = FrameStream(_task_name(self.task, context))
                identity = payload.get("evaluation_id", payload.get("episode_id", "episode"))
                # Recorder identities are UUIDs; sanitize alternate event sources.
                identity = re.sub(r"[^A-Za-z0-9_.-]", "_", str(identity))
                if self.mode == "source":
                    self._controller = SourceRecoveryController(
                        self.config, self.output_dir / identity, runtime=self.runtime)
                else:
                    self._controller = RecoveryController(self.config, self.output_dir / identity)
                self._latest_frame = None
                self._terminal = False
            if self._stream is None:
                continue
            frame = self._stream.event(payload)
            if kind == "episode_start" and isinstance(payload.get("state"), dict):
                # Initial state is a real observation; no completed transition
                # is invented when the first model decision raises.
                self._latest_frame = self._stream.frame(payload, payload["state"], 0, 2)
            if frame is not None:
                observations += 1
                terminal = bool(payload.get("done", dones[0] if dones else False))
                self._latest_frame, self._terminal = deepcopy(frame), terminal
                results.append(self._controller.observe(frame, policy, terminal=terminal))
        return results, observations

    def before_policy_step(self, policy):
        if self.mode == "source":
            self._drain(policy)

    def handle_policy_error(self, policy, error):
        if self.mode != "source" or self._controller is None or self._latest_frame is None or self._terminal:
            return False
        result = self._controller.handle_execution_error(deepcopy(self._latest_frame), policy, error)
        return result is True or isinstance(result, dict) and result.get("status") == "applied"

    def run_policy(self, call, policy, tensors=()):
        if self.mode != "source":
            return call()
        self.before_policy_step(policy)
        retries = 0
        while True:
            checkpoint = ExecutionCheckpoint(policy, tensors=tensors)
            queued = list(self._events)
            try:
                return call()
            except Exception as error:
                checkpoint.restore()
                self._events.clear()
                self._events.extend(queued)
                limit = getattr(self.config, "max_policy_retries", getattr(self.config, "max_iterations", 3))
                if retries >= limit or not self.handle_policy_error(policy, error):
                    raise
                retries += 1

    def advance(self, policy, dones):
        if len(dones) != 1:
            raise ValueError("Online recovery currently requires a single environment")
        results, observations = self._drain(policy, dones)
        if observations == 0:
            raise RuntimeError("Online recovery requires a recorded observation for every simulator step")
        return results

    def close(self):
        if self.runtime is not None:
            self.runtime.close()
