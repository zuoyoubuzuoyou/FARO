"""Validated simulator-step latent faults, without simulator dependencies."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, fields
from typing import Any, ClassVar

from .agent_dropout import AgentDropoutFaultSpec, FaultInjectionError, _config_value


@dataclass(frozen=True)
class LatentFaultSpec:
    enabled: bool = False
    fault_id: str = ""
    agent: str = "agent_0"
    episode_id: str = ""
    simulator_step: int = 0
    simulator_step_schedule: str = ""
    duration_steps: int = -1
    object: str = "any_targets|0"
    replacement_object: str = "any_targets|1"
    delay_steps: int = 5
    position_offset: tuple[float, ...] = (1.0, 0.0, 0.0)
    name: ClassVar[str]
    category: ClassVar[str]
    subtype: ClassVar[str]
    config_path: ClassVar[str]

    @classmethod
    def from_config(cls, config: Any):
        default = cls()
        values = {field.name: _config_value(config, field.name, getattr(default, field.name))
                  for field in fields(cls)}
        try:
            values["position_offset"] = tuple(float(x) for x in values["position_offset"])
            for key in ("simulator_step", "duration_steps", "delay_steps"):
                value = values[key]
                if isinstance(value, bool) or int(value) != value:
                    raise ValueError(f"{key} must be an integer")
                values[key] = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise FaultInjectionError(f"Invalid {cls.name} parameters: {error}") from error
        return cls(**values)

    def _trigger(self):
        return AgentDropoutFaultSpec(
            enabled=self.enabled, agent=self.agent, episode_id=self.episode_id,
            simulator_step=self.simulator_step,
            simulator_step_schedule=self.simulator_step_schedule,
        )

    def validate(self):
        if not self.enabled:
            return
        try:
            self._trigger().validate()
        except FaultInjectionError as error:
            raise FaultInjectionError(str(error).replace("agent_dropout", self.name)) from error
        if self.duration_steps != -1 and self.duration_steps <= 0:
            raise FaultInjectionError("duration_steps must be positive or -1")
        if self.name == "wrong_subgoal" and self.duration_steps != -1:
            raise FaultInjectionError("wrong_subgoal persists until episode end; duration_steps must be -1")
        if self.name in ("wrong_subgoal", "object_localization"):
            if not re.fullmatch(r"any_targets\|\d+", self.object):
                raise FaultInjectionError("object must be an exact any_targets|N entity")
        if self.name == "wrong_subgoal":
            if (not re.fullmatch(r"any_targets\|\d+", self.replacement_object)
                    or self.object == self.replacement_object):
                raise FaultInjectionError("replacement_object must be a different valid entity")
        if self.name == "message_delay" and self.delay_steps <= 0:
            raise FaultInjectionError("delay_steps must be positive")
        if self.name == "object_localization":
            if (len(self.position_offset) != 3
                    or not all(math.isfinite(x) for x in self.position_offset)
                    or not any(self.position_offset)):
                raise FaultInjectionError("position_offset must contain three finite values and a nonzero offset")

    def simulator_step_schedule_map(self):
        return self._trigger().simulator_step_schedule_map()

    def episode_matches(self, episode_id):
        return self._trigger().episode_matches(episode_id)

    def target_simulator_step(self, episode_id):
        return self._trigger().target_simulator_step(episode_id)

    def event(self, kind, scene_id, episode_id, step, **evidence):
        return {
            "schema_version": 1, "type": kind, "fault_id": self.fault_id or f"{self.name}_001",
            "fault_name": self.name, "fault_type": self.category, "fault_subtype": self.subtype,
            "scene_id": str(scene_id), "episode_id": str(episode_id), "agent": self.agent,
            "simulator_step": int(step), "trigger_domain": "simulator_step",
            "phase": "PRE_ACTION", "logical_agent_active": True,
            "logical_agent_responsive": True, **evidence,
        }


class ActionNoopFaultSpec(LatentFaultSpec):
    name = "action_noop"
    category, subtype = "ActionFault", "ActionNoop"
    config_path = "action.action_noop"


class WrongSubgoalFaultSpec(LatentFaultSpec):
    name = "wrong_subgoal"
    category, subtype = "PlanningFault", "WrongSubgoal"
    config_path = "planning.wrong_subgoal"


class MessageDelayFaultSpec(LatentFaultSpec):
    name = "message_delay"
    category, subtype = "CommunicationFault", "MessageDelay"
    config_path = "communication.message_delay"


class ObjectLocalizationFaultSpec(LatentFaultSpec):
    name = "object_localization"
    category, subtype = "PerceptionFault", "ObjectLocalization"
    config_path = "perception.object_localization"


LATENT_SPECS = (ActionNoopFaultSpec, WrongSubgoalFaultSpec,
                MessageDelayFaultSpec, ObjectLocalizationFaultSpec)


def read_latent_spec(config: Any):
    selected = []
    for cls in LATENT_SPECS:
        category, name = cls.config_path.split(".")
        spec = cls.from_config(_config_value(_config_value(config, category, None), name, None))
        spec.validate()
        if spec.enabled:
            selected.append(spec)
    if len(selected) > 1:
        raise FaultInjectionError("Only one fault can be enabled per run")
    return selected[0] if selected else None


class FaultWindow:
    def __init__(self, spec: LatentFaultSpec, scene_id: str, episode_id: str):
        spec.validate()
        self.spec, self.scene_id, self.episode_id = spec, scene_id, episode_id
        self.active = self.triggered = self.expired = False

    def advance(self, step: int):
        spec = self.spec
        events = []
        if not spec.enabled or not spec.episode_matches(self.episode_id):
            return events
        start = spec.target_simulator_step(self.episode_id)
        if step >= start and not self.triggered:
            self.triggered = self.active = True
            events.append(spec.event("fault_injected", self.scene_id, self.episode_id, step,
                                     configured_simulator_step=start, duration_steps=spec.duration_steps))
        if (self.active and spec.duration_steps != -1
                and step >= start + spec.duration_steps):
            self.active, self.expired = False, True
            events.append(spec.event("fault_expired", self.scene_id, self.episode_id, step))
        return events
