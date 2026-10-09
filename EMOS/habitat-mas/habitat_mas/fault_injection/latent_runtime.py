"""Per-episode latent fault execution and evidence collection."""

from __future__ import annotations

import math
import re

from habitat_mas.agents.message_bus import AgentMessageBus
from .agent_dropout import FaultInjectionError
from .latent import FaultWindow


def navigation_ranges(action_space):
    """Flatten the actual agent action space without touching stop/arm channels."""
    index = 0
    ranges = []
    navigation_names = {"oracle_nav_action", "oracle_nav_coord_action", "oracle_nav_lookat_action",
                        "base_velocity", "base_velocity_non_cylinder", "base_vel"}

    def visit(space, path):
        nonlocal index
        if hasattr(space, "spaces"):
            for key, value in space.spaces.items():
                visit(value, path + [key])
        else:
            size = math.prod(space.shape) if getattr(space, "shape", ()) else getattr(space, "n", 1)
            nav = any(path[-1] == name or path[-1].endswith("_" + name) for name in navigation_names)
            if nav:
                ranges.append((index, index + size, "/".join(path)))
            index += size

    visit(action_space, [])
    if not ranges:
        raise FaultInjectionError("action_noop requires a supported navigation action channel")
    return ranges


class LatentFaultRuntime:
    def __init__(self, spec, scene_id, episode_id, valid_entities=None):
        self.spec = spec
        self.window = FaultWindow(spec, str(scene_id), str(episode_id))
        self.valid_entities = set(valid_entities) if valid_entities is not None else None
        self.step = 0
        self.events = []
        self.applications = 0
        self.planning_applied = False
        self.agent_scene_description = None
        self.finished_summary = None

    def emit(self, kind, **evidence):
        event = self.spec.event(kind, self.window.scene_id, self.window.episode_id, self.step, **evidence)
        self.events.append(event)
        if kind == "fault_applied":
            self.applications += 1
        return event

    def begin_step(self, step):
        self.step = int(step)
        self.events.extend(self.window.advance(self.step))
        delay = None
        if self.window.active and self.spec.name == "message_delay":
            delay = {"agent": self.spec.agent, "delay_steps": self.spec.delay_steps,
                     "fault_id": self.spec.fault_id or f"{self.spec.name}_001"}
        AgentMessageBus.begin_step(self.step, delay)

    def collect_messages(self):
        for message in AgentMessageBus.drain_delivery_events():
            kind = message["type"]
            evidence = {key: value for key, value in message.items() if key not in ("type", "simulator_step", "fault_id")}
            self.emit(kind, **evidence)
            if kind == "message_queued":
                self.emit("fault_applied", effect="message_delay", **evidence)

    def apply_action(self, agent_name, action_data, action_space):
        if not (self.window.active and self.spec.name == "action_noop" and agent_name == self.spec.agent):
            return
        original = action_data.take_actions if action_data.take_actions is not None else action_data.actions
        effective = original.clone()
        channels = navigation_ranges(action_space)
        changed = False
        for start, end, name in channels:
            if name.endswith("oracle_nav_coord_action"):
                # Coordinate navigation is gated by the look-at vector in
                # OracleNavCoordinateAction.step, irrespective of position/mode.
                lookat = [(a, b) for a, b, key in channels
                          if key.rsplit("/", 1)[0] == name.rsplit("/", 1)[0]
                          and key.endswith("oracle_nav_lookat_action")]
                active = any(bool(original[:, a:b].ne(0).any().item()) for a, b in lookat)
            elif name.endswith("oracle_nav_action"):
                active = bool(original[:, start:end].gt(0).any().item())
            else:
                active = bool(original[:, start:end].ne(0).any().item())
            changed |= active
            effective[:, start:end] = 0
        if changed:
            action_data.take_actions = effective
            self.emit("fault_applied", phase="POST_POLICY_PRE_ENV_STEP", effect="navigation_suppressed",
                      original_action=original[0].detach().cpu().tolist(),
                      effective_action=effective[0].detach().cpu().tolist(),
                      masked_channels=[name for _, _, name in channels])

    def rewrite_assignment(self, assignment):
        if self.valid_entities is None or not {self.spec.object, self.spec.replacement_object} <= self.valid_entities:
            raise FaultInjectionError("wrong_subgoal references an unavailable scene entity")
        pattern = r"(?<![\w|])" + re.escape(self.spec.object) + r"(?![\w|])"
        changed, count = re.subn(pattern, lambda _: self.spec.replacement_object, assignment)
        if not count:
            raise FaultInjectionError(f"wrong_subgoal object {self.spec.object} is absent from the target agent assignment")
        return changed

    def initial_assignment(self, assignment):
        if not (self.window.active and self.spec.name == "wrong_subgoal" and not self.planning_applied):
            return assignment
        changed = self.rewrite_assignment(assignment)
        self.planning_applied = True
        self.emit("fault_applied", effect="planning_input_replaced", original_subtask=assignment,
                  effective_subtask=changed)
        return changed

    def apply_planning(self, llm_agent, policy):
        if not (self.window.active and self.spec.name == "wrong_subgoal" and not self.planning_applied):
            return
        original = llm_agent.subtask_description
        changed = self.rewrite_assignment(original)
        llm_agent.replace_subtask(changed)
        policy._cur_call_high_level.fill_(True)
        self.planning_applied = True
        self.emit("fault_applied", effect="planning_input_replaced", original_subtask=original,
                  effective_subtask=changed)

    def record_simulator_events(self, events):
        for event in events:
            if event.get("fault_id") == (self.spec.fault_id or f"{self.spec.name}_001"):
                self.events.append(event)
                self.applications += int(event["type"] == "fault_applied")

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def finish_episode(self, termination_reason="episode_done", **evidence):
        if self.finished_summary is not None:
            return self.finished_summary
        AgentMessageBus.finish_episode(self.step)
        self.collect_messages()
        self.finished_summary = self.emit("fault_episode_summary", **self.summary(),
                                          termination_reason=termination_reason, **evidence)
        return self.finished_summary

    def summary(self):
        return {"triggered": self.window.triggered, "effective": self.applications > 0,
                "applications": self.applications,
                "ineffective_reason": None if self.applications else (
                    "no_matching_command_message_or_navigation" if self.window.triggered else "trigger_not_reached")}
