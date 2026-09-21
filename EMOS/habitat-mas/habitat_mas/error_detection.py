"""External runtime watchdog for the single-environment EMOS evaluator.

Consumes agent availability/response telemetry and actual sent messages. It does
not read injection schedules or ask the task LLM whether evaluation should stop.
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass
from typing import Mapping


class DetectedSystemCrash(RuntimeError):
    """A watchdog rule has latched; no subsequent policy/env step is allowed."""

    def __init__(self, event):
        self.event = event
        self.fault_events = []
        super().__init__(f"{event['reason']}: {event['agent']} at step {event['simulator_step']}")


@dataclass(frozen=True)
class ErrorDetectionSpec:
    enabled: bool = True
    hang_steps: int = 10
    repeated_message_limit: int = 10

    @classmethod
    def from_config(cls, config):
        get = config.get if isinstance(config, Mapping) else lambda k, d: getattr(config, k, d)
        spec = cls(**{k: get(k, v) for k, v in vars(cls()).items()})
        if not isinstance(spec.enabled, bool):
            raise ValueError("error_detection.enabled must be a boolean")
        for field in ("hang_steps", "repeated_message_limit"):
            value = getattr(spec, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"error_detection.{field} must be a positive integer")
        return spec


class ErrorDetector:
    def __init__(self, spec=ErrorDetectionSpec()):
        self.spec = spec
        self.reset("", "", [])

    def reset(self, scene_id, episode_id, agent_names):
        self.scene_id, self.episode_id = str(scene_id), str(episode_id)
        self.names = tuple(agent_names)
        self.health = {name: {"online": True, "responsive": True} for name in self.names}
        self.unresponsive = {name: 0 for name in self.names}
        self.message_runs = {}
        self.last_progress = None
        self.last_step = -1
        self.crash = None
        self.recovered_dropouts = set()

    def acknowledge_dropout(self, names):
        """Clear only a dropout latch after its task transfer and replans succeed."""
        names = set(names)
        if (self.crash is None or self.crash.event["reason"] != "agent_dropout"
                or self.crash.event["agent"] not in names
                or any(name not in self.health or self.health[name]["online"] for name in names)):
            raise ValueError("Only the currently detected offline agents can be acknowledged")
        self.recovered_dropouts.update(names)
        for name in names:
            self.unresponsive[name] = 0
            self.message_runs.pop(name, None)
        self.crash = None

    def _check_latch(self):
        if self.crash is not None:
            raise self.crash

    def _fail(self, reason, name, step, phase, count, evidence):
        event = {
            "type": "system_crash", "scene_id": self.scene_id, "episode_id": self.episode_id,
            "reason": reason, "agent": name, "simulator_step": step, "phase": phase,
            "consecutive_count": count, "evidence": evidence,
            "agent_status": copy.deepcopy(self.health), "simulated_crash": True,
        }
        self.crash = DetectedSystemCrash(event)
        raise self.crash

    def before_actions(self, step, health):
        if not self.spec.enabled:
            return
        self._check_latch()
        if set(health) != set(self.names):
            raise ValueError("Error detector requires status for every configured agent")
        self.health = health
        for name in self.names:
            if not health[name]["online"] and name not in self.recovered_dropouts:
                self._fail("agent_dropout", name, step, "PRE_ACTION", 1,
                           "Agent is no longer an online team member")

    def check_messages(self, step, messages, progress=None):
        if not self.spec.enabled:
            return
        self._check_latch()
        # Physical/task progress breaks a suspected message loop. Repeated log
        # rows and carried-forward high-level actions are NOT message events.
        if progress is not None and progress != self.last_progress:
            self.message_runs.clear()
            self.last_progress = progress
        for message in messages:
            source = message["source"]
            signature = (message["target"], " ".join(message["request"].split()))
            previous, count, start = self.message_runs.get(source, (None, 0, step))
            count, start = (count + 1, start) if previous == signature else (1, step)
            self.message_runs[source] = (signature, count, start)
            if count >= self.spec.repeated_message_limit:
                self._fail("message_loop", source, step, "PRE_ENV_STEP", count, {
                    "target": signature[0], "message": signature[1], "since_simulator_step": start,
                })

    def after_step(self, step):
        if not self.spec.enabled:
            return
        self._check_latch()
        if step != self.last_step + 1:
            raise ValueError(f"Watchdog expected completed step {self.last_step + 1}, got {step}")
        self.last_step = step
        for name in self.names:
            if name in self.recovered_dropouts and not self.health[name]["online"]:
                continue
            count = self.unresponsive[name] + 1 if not self.health[name]["responsive"] else 0
            self.unresponsive[name] = count
            if count >= self.spec.hang_steps:
                self._fail("agent_hang", name, step, "POST_ENV_STEP", count,
                           f"Repeated unresponsive agent status for {count} completed simulator steps")

    def metrics(self):
        reason = None if self.crash is None else self.crash.event["reason"]
        return {
            "error_detection.enabled": float(self.spec.enabled),
            "error_detection.all_agents_online": float(all(v["online"] for v in self.health.values())),
            "error_detection.all_agents_responsive": float(all(v["responsive"] for v in self.health.values())),
            "error_detection.no_message_loop": float(reason != "message_loop"),
            "error_detection.max_unresponsive_steps": float(max(self.unresponsive.values(), default=0)),
            "error_detection.system_crash": float(self.crash is not None),
        }


def progress_signature(state, metrics):
    """Coarse physical/task progress, excluding time, reward and watchdog state."""
    positions = {
        name: {"position": [round(x, 4) for x in agent["position"]],
               "rotation": round(agent["rotation"], 4), "held_object": agent["held_object"]}
        for name, agent in state.get("agents", {}).items()
    }
    goals = {k: v for k, v in metrics.items() if k == "pddl_success" or k.startswith("pddl_stage_goals.")}
    return json.dumps([positions, state.get("objects", []), goals], sort_keys=True)
