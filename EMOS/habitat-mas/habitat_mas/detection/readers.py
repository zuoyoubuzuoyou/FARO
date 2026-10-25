"""Streaming, causal adapters for historical frames and recorder-v2 events.

Only an environment frame yields an observation. Episode summaries, injected
fault state, detector outputs, rewards and historical/latched stage metrics
never enter the normalized frame. Missing diagnostics remain ``None``.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Iterable, Iterator, Optional


_NAV_FIELDS = frozenset(("available", "timing", "active", "target", "reason",
    "path_found", "distance_to_goal", "distance_threshold", "angle_to_object",
    "angle_to_waypoint", "angle_threshold", "distance_satisfied", "orientation_satisfied",
    "at_goal", "control", "motion_type", "skill_done", "prev_nav_done", "target_cache_hit",
    "true_object_position", "estimated_object_position", "final_navigation_position",
    "robot_position_before_control", "next_waypoint"))
_TERM_FIELDS = frozenset(("skill_steps", "max_skill_steps", "oracle_finished", "timeout_reached",
    "force_end_on_timeout", "high_level_requested_termination", "skill_terminated",
    "episode_terminated", "reason", "status", "is_skill_done", "cur_skill_step",
    "over_max_len", "high_level_interrupt", "call_high_level", "bad_terminate"))
_REQUEST = re.compile(r'"(agent_\d+)" agent sent you requests: "(.*?)"\.', re.DOTALL)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _metric_subset(metrics):
    """Allow current native predicates/distances, never historical stages."""
    result = {}
    for key, value in _mapping(metrics).items():
        if key == "pddl_success" or re.fullmatch(r"object_to_goal_distance\.\d+", key):
            if value is None or isinstance(value, (int, float, bool)):
                result[key] = value
        elif key == "object_to_goal_distance" and isinstance(value, dict):
            for index, distance in value.items():
                if str(index).isdigit() and (distance is None or isinstance(distance, (int, float))):
                    result[f"object_to_goal_distance.{index}"] = distance
    return result


def _json_or_original(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return value


def _flatten(history):
    if isinstance(history, dict):
        return [history] if history.get("role") else []
    if not isinstance(history, list):
        return []
    return [entry for group in history for entry in _flatten(group)]


def _tool(messages, fallback):
    for message in reversed(messages):
        if message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls") or []
        function = _mapping(calls[0]).get("function") if calls else message.get("function_call")
        if isinstance(function, dict):
            return function.get("name"), _json_or_original(function.get("arguments"))
    return fallback.get("action"), fallback.get("arguments")


def _navigation(state):
    raw = state.get("navigation")
    if not isinstance(raw, dict):
        return None
    if "active" in raw or "available" in raw:
        selected = raw
    else:
        candidates = [v for v in raw.values() if isinstance(v, dict)]
        selected = next((v for v in candidates if v.get("active") is True), None)
        if selected is None:
            selected = next((v for v in candidates if v.get("available") is True), None)
    if selected is None:
        return None
    return {k: v for k, v in selected.items() if k in _NAV_FIELDS}


def _termination(value):
    return {k: v for k, v in value.items() if k in _TERM_FIELDS} if isinstance(value, dict) else None


class FrameStream:
    """Incrementally normalize recorder events without reading future records."""

    def __init__(self, task, goal_positions=None):
        self.task = task
        self.initial_goals = deepcopy(goal_positions or {})
        self.reset()

    def reset(self):
        self.context = {"task_description": None, "scene_description": None,
                        "assignments": {}, "robot_capabilities": None,
                        "goal_positions": deepcopy(self.initial_goals)}
        self.action_scopes = {}
        self.active_actions = {}
        self.pending_handoffs = set()
        self.histories = {}
        self.sent_counts = Counter()
        self.terminations = {}
        self.pending_decisions = []
        self.pending_decision_priorities = {}
        self.pending_messages = []

    def update_context(self, raw):
        raw = _mapping(raw)
        for key in ("task_description", "scene_description"):
            if key in raw and raw[key] is not None:
                self.context[key] = raw[key]
        if raw.get("pddl_text_goal") is not None and raw.get("task_description") is None:
            self.context["task_description"] = raw["pddl_text_goal"].strip()
        if raw.get("robot_capabilities") is not None or raw.get("robot_resume") is not None:
            self.context["robot_capabilities"] = _json_or_original(
                raw.get("robot_capabilities", raw.get("robot_resume")))
        if isinstance(raw.get("assignments"), dict):
            self.context["assignments"].update(raw["assignments"])

    def action(self, agent, raw, step):
        raw = _mapping(raw)
        name = raw.get("action", raw.get("action_name", raw.get("name")))
        args = raw.get("arguments", raw.get("parameters"))
        identity = raw.get("action_id", raw.get("decision_step"))
        scope = (identity, name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        if name is None:
            self.action_scopes.pop(agent, None)
            return {"name": None, "arguments": None, "id": None, "age": None}
        previous = self.action_scopes.get(agent)
        if previous is None or previous[0] != scope:
            self.action_scopes[agent] = (scope, step)
        age = step - self.action_scopes[agent][1] + 1
        return {"name": name, "arguments": args, "id": identity, "age": age}

    def old_decisions(self, record, step):
        decisions, messages = [], []
        for key, decision in _mapping(record.get("policy_info")).items():
            if not key.endswith("_high_level_decision") or not isinstance(decision, dict):
                continue
            agent = decision.get("agent", key.removesuffix("_high_level_decision"))
            rc = _mapping(decision.get("replay_context"))
            history = _flatten(rc.get("chat_history"))
            previous = self.histories.get(agent, [])
            shared = 0
            for old, new in zip(previous, history):
                if old != new:
                    break
                shared += 1
            new_history = history[shared:]
            self.histories[agent] = deepcopy(history)
            self.update_context({k: rc[k] for k in ("task_description", "scene_description", "robot_resume") if k in rc})
            subtask = rc.get("subtask_description")
            if subtask is not None:
                self.context["assignments"][agent] = subtask
            tool, arguments = _tool(history, decision)
            feedback = []
            for message in new_history:
                role, content = message.get("role"), message.get("content")
                if role in ("tool", "user") and content is not None:
                    feedback.append({"role": role, "content": content,
                        "tool_call_id": message.get("tool_call_id"),
                        # Historical tool responses were written before env.step.
                        "placeholder": role == "tool"})
                if role == "user" and isinstance(content, str):
                    for sender, request in _REQUEST.findall(content):
                        messages.append({"kind": "receive", "sender": sender, "recipient": agent,
                                         "message": request, "step": step})
            decisions.append({"agent": agent, "tool": tool, "arguments": arguments,
                              "feedback": feedback, "subtask": subtask})
            # These snapshots are cumulative and shared between agent contexts.
            occurrence = Counter()
            for sent in rc.get("sent_messages") or []:
                if not isinstance(sent, dict):
                    continue
                fingerprint = json.dumps(sent, sort_keys=True, ensure_ascii=False)
                occurrence[fingerprint] += 1
                if occurrence[fingerprint] > self.sent_counts[fingerprint]:
                    messages.append({"kind": "send", "sender": sent.get("source", sent.get("sender")),
                        "recipient": sent.get("target", sent.get("recipient")),
                        "message": sent.get("request", sent.get("message")), "step": step})
                    self.sent_counts[fingerprint] = occurrence[fingerprint]
        return decisions, messages

    def frame(self, record, state, step, schema):
        agents = {}
        state_agents = _mapping(state.get("agents"))
        step_actions = _mapping(record.get("agent_step_actions"))
        agent_names = set(state_agents) | set(step_actions) | set(self.active_actions)
        for agent in sorted(agent_names):
            raw = _mapping(state_agents.get(agent))
            if schema == 1:
                command = raw.get("current_high_level_action") or _mapping(step_actions.get(agent)).get("action")
                term = _mapping(record.get("policy_info")).get(agent + "_navigation_termination")
            else:
                command = self.active_actions.get(agent)
                term = self.terminations.get(agent)
            agents[agent] = {
                "position": raw.get("position", raw.get("base_position")),
                "rotation": raw.get("rotation", raw.get("base_rotation")),
                "held_object": raw.get("held_object", raw.get("grasped_object_id")),
                "action": self.action(agent, command, step),
                "navigation": _navigation(raw), "termination": _termination(term),
                "detection": raw.get("detection"), "stop": raw.get("stop"),
                "nav_finished": raw.get("nav_finished"),
            }
        objects = {}
        object_records = state.get("objects", state.get("target_objects")) or []
        if isinstance(object_records, dict):
            object_records = [{"goal_name": name, **value} for name, value in object_records.items() if isinstance(value, dict)]
        for index, raw in enumerate(object_records):
            if not isinstance(raw, dict):
                continue
            name = raw.get("goal_name", f"any_targets|{index}")
            objects[name] = {"position": raw.get("position"), "object_id": raw.get("object_id")}
            if raw.get("goal_position") is not None and name not in self.context["goal_positions"]:
                self.context["goal_positions"][name] = raw["goal_position"]
        metrics = _metric_subset(record.get("metrics", state.get("task_progress")))
        if schema == 1:
            decisions, messages = self.old_decisions(record, step)
        else:
            decisions, messages = self.pending_decisions, self.pending_messages
            self.pending_decisions, self.pending_messages = [], []
            self.pending_decision_priorities = {}
        return deepcopy({"step": step, "task": self.task, "context": self.context,
                         "agents": agents, "objects": objects, "metrics": metrics,
                         "decisions": decisions, "messages": messages})

    def pending_decision(self, agent, tool, arguments, priority):
        """Merge one environment step's model/controller events losslessly.

        The recorder emits placeholder feedback before tool_selection and
        agent_tool_decision. The placeholder establishes a partial observation;
        later decisions fill its tool/arguments without replacing feedback.
        CrabAgent's parsed decision takes priority over a model selection.
        """
        prior = next((d for d in self.pending_decisions if d["agent"] == agent), None)
        if prior is None:
            prior = {"agent": agent, "tool": tool, "arguments": arguments,
                     "feedback": [], "subtask": self.context["assignments"].get(agent)}
            self.pending_decisions.append(prior)
            self.pending_decision_priorities[agent] = priority
        elif priority > 0 and priority >= self.pending_decision_priorities.get(agent, 0):
            prior.update(tool=tool, arguments=arguments,
                         subtask=self.context["assignments"].get(agent))
            self.pending_decision_priorities[agent] = priority
        return prior

    def event(self, record):
        kind = record.get("event_type", record.get("event"))
        agent = record.get("agent_id", record.get("agent_name"))
        step = record.get("env_step", record.get("step"))
        if kind == "episode_start":
            self.reset()
            self.update_context(record.get("context"))
        elif kind == "task_assignment":
            self.update_context(record)
            if agent is not None and record.get("subtask_description") is not None:
                self.context["assignments"][agent] = record["subtask_description"]
        elif kind == "action_start" and agent is not None:
            self.active_actions[agent] = record
            self.pending_handoffs.discard(agent)
            self.terminations.pop(agent, None)
        elif kind == "action_end" and agent is not None:
            active = self.active_actions.get(agent)
            if active is not None and active.get("action_id") != record.get("action_id"):
                return None
            termination = _termination({**record, **_mapping(record.get("result"))})
            if (active is not None and record.get("status") == "timeout"
                    and termination.get("call_high_level") is False):
                # Keep the last scope as failed-handoff evidence, not proof of
                # physical execution. Its termination remains explicitly timeout.
                self.pending_handoffs.add(agent)
            else:
                self.active_actions.pop(agent, None)
                self.pending_handoffs.discard(agent)
            self.terminations[agent] = termination
        elif kind == "skill_termination_signal" and agent is not None:
            active = self.active_actions.get(agent)
            if (active is not None and record.get("action_id") is not None
                    and active.get("action_id") != record["action_id"]):
                return None
            termination = _termination(record)
            if agent in self.pending_handoffs:
                if record.get("call_high_level") is True:
                    self.active_actions.pop(agent, None)
                    self.pending_handoffs.discard(agent)
                else:
                    termination = {**self.terminations.get(agent, {}), **termination}
            self.terminations[agent] = termination
        elif kind in ("agent_tool_decision", "tool_selection") and record.get("phase", "execution") == "execution":
            self.pending_decision(agent, record.get("action_name"), record.get("parameters"),
                                  2 if kind == "agent_tool_decision" else 1)
        elif kind == "tool_response_placeholder" and record.get("phase", "execution") == "execution":
            previous = self.pending_decision(agent, record.get("action_name"), None, 0)
            previous["feedback"].append({"role": "tool", "content": record.get("content"),
                                        "tool_call_id": record.get("tool_call_id"), "placeholder": True})
        elif kind in ("communication_send", "communication_receive") and record.get("status") != "suppressed":
            self.pending_messages.append({"kind": "send" if kind.endswith("send") else "receive",
                "sender": record.get("sender"), "recipient": record.get("recipient"),
                "message": record.get("message"), "step": step})
        elif kind == "step":
            return self.frame(record, _mapping(record.get("state")), int(step), 2)
        return None


def frames_from_records(records: Iterable[dict], task: str,
                        goal_positions: Optional[dict] = None) -> Iterator[dict]:
    """Yield detached observation dictionaries without consuming later records.

    Source step units are preserved: historical simulator_step is zero-based;
    recorder-v2 env_step is one-based. Actions have a one-based observed age.
    A frame is a mutable JSON dictionary detached from every other frame and
    caller input. An episode_start clears all state, including previous roles.
    """
    stream = FrameStream(task, goal_positions)
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Trajectory record must be a JSON object")
        if record.get("type") == "episode_start":
            stream.reset()
            stream.update_context(record.get("text_context"))
        elif record.get("type") == "simulator_step":
            yield stream.frame(record, _mapping(record.get("simulator_state")),
                               int(record["simulator_step"]), 1)
        elif "event_type" in record or "event" in record:
            frame = stream.event(record)
            if frame is not None:
                yield frame


def iter_frames(path, task: str, goal_positions: Optional[dict] = None) -> Iterator[dict]:
    """Read JSONL lazily; malformed input includes the file and line number."""
    path = Path(path)
    def records():
        with path.open(encoding="utf-8-sig") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError as error:
                    raise ValueError(f"{path}: line {line_number}: {error}") from error
                yield record
    yield from frames_from_records(records(), task, goal_positions)
