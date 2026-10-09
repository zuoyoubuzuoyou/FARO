"""Machine-readable and human-readable structure-fault trajectories."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from habitat_mas.fault_injection.naming import output_run_stem


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return repr(value)


class AgentDropoutTrajectoryRecorder:
    """Record exact JSONL steps plus a concise high-level text trace.

    JSONL remains the source of truth for every simulator step.  The text file
    is episode-scoped and records every simulator step.  Since a high-level
    skill normally spans many simulator steps, the last selected action is
    carried forward and marked as ``continuing`` until a new decision or an
    execution override replaces it.
    """

    schema_version = 1

    def __init__(
        self,
        output_dir: str,
        run_label: str,
        mode: str,
        agent_names: Optional[Iterable[str]] = None,
        *,
        fault_name: str = "agent_dropout",
    ) -> None:
        self.run_label = str(run_label)
        self.mode = str(mode)
        output_type = "clean" if self.mode == "clean" else fault_name
        run_stem = output_run_stem(run_label, output_type)
        self.path = (
            Path(output_dir)
            / f"{run_stem}_trajectory.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._run_stem = run_stem
        self._text_paths: Dict[str, Path] = {}
        self._agent_names = [str(name) for name in (agent_names or [])]
        self._active_actions: Dict[str, Dict[str, Mapping[str, Any]]] = {}
        self._event_index = 0
        self.record({"type": "trajectory_start"})

    @staticmethod
    def _safe_component(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))

    def text_path_for_episode(self, episode_id: Any) -> Path:
        """Return the readable trace path for an episode."""

        episode_key = str(episode_id)
        if episode_key not in self._text_paths:
            safe_episode = self._safe_component(episode_key)
            self._text_paths[episode_key] = self.path.parent / (
                f"{self._run_stem}_trace-episode_{safe_episode}.txt"
            )
        return self._text_paths[episode_key]

    def _append_text(self, episode_id: Any, text: str) -> None:
        with self.text_path_for_episode(episode_id).open(
            "a", encoding="utf-8"
        ) as output:
            output.write(text)
            output.flush()

    @staticmethod
    def _format_action(decision: Mapping[str, Any]) -> str:
        action_names = {
            "nav_to_obj": "Navigate",
            "nav_to_goal": "NavigateToGoal",
            "nav_to_robot": "NavigateToRobot",
            "pick": "Pick",
            "place": "Place",
            "reset_arm": "ResetArm",
            "send_request": "SendRequest",
            "wait": "Wait",
            "agent_dropout_noop": "AgentDropoutNoOp",
            "agent_hang_noop": "AgentHangNoOp",
            "not_recorded": "NotRecorded",
        }
        raw_name = str(decision.get("action", "unknown"))
        name = action_names.get(raw_name, raw_name)
        arguments = _jsonable(decision.get("arguments", []))
        if isinstance(arguments, Mapping):
            rendered_args = ", ".join(
                f"{key}={value}" for key, value in arguments.items()
            )
        elif isinstance(arguments, list):
            rendered_args = ", ".join(str(value) for value in arguments)
        elif arguments is None:
            rendered_args = ""
        else:
            rendered_args = str(arguments)
        return f"{name}[{rendered_args}]"

    @staticmethod
    def is_high_level_trace_event(
        policy_info: Optional[Mapping[str, Any]], done: bool = False
    ) -> bool:
        """Whether a step needs a refreshed symbolic scene description."""

        if done:
            return True
        if policy_info is None:
            return False
        return any(
            key.endswith(
                (
                    "_high_level_decision",
                    "_execution_override",
                    "_pddl_action",
                    "_dropout_event",
                    "_hang_event",
                    "_fault_events",
                )
            )
            for key in policy_info
        )

    def record(self, event: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": self.schema_version,
            "event_index": self._event_index,
            "run_label": self.run_label,
            "mode": self.mode,
            **dict(event),
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(
                json.dumps(_jsonable(payload), ensure_ascii=False)
            )
            output.write("\n")
            output.flush()
        self._event_index += 1

    def record_episode_start(
        self,
        *,
        episode_id: Any,
        scene_id: str,
        text_context: Mapping[str, Any],
    ) -> None:
        self.record(
            {
                "type": "episode_start",
                "episode_id": str(episode_id),
                "scene_id": str(scene_id),
                "simulator_step": 0,
                "text_context": text_context,
            }
        )
        self._active_actions[str(episode_id)] = {}
        text_path = self.text_path_for_episode(episode_id)
        text_path.write_text("", encoding="utf-8")
        task = str(text_context.get("pddl_text_goal", "")).strip()
        scene_description = str(
            text_context.get("scene_description", "")
        ).strip()
        header = (
            f"Task: {task}\n"
            f"Episode_ID: {episode_id}\n"
            f"Scene_ID: {scene_id}\n"
            f"Mode: {self.mode}\n"
        )
        if scene_description:
            header += f"Initial_Scene_Description:\n{scene_description}\n"
        header += "\n"
        self._append_text(episode_id, header)

    @staticmethod
    def _agent_from_policy_key(key: str) -> Optional[str]:
        match = re.match(r"(agent_\d+)_", key)
        return None if match is None else match.group(1)

    def _agent_step_actions(
        self,
        episode_id: Any,
        policy_info: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve every agent's effective high-level action for this step."""

        episode_key = str(episode_id)
        active_actions = self._active_actions.setdefault(episode_key, {})
        discovered_agents = set(self._agent_names)
        discovered_agents.update(active_actions)
        for key in policy_info:
            agent_name = self._agent_from_policy_key(str(key))
            if agent_name is not None:
                discovered_agents.add(agent_name)

        step_actions: Dict[str, Dict[str, Any]] = {}
        for agent_name in sorted(discovered_agents):
            decision_key = f"{agent_name}_high_level_decision"
            override_key = f"{agent_name}_execution_override"
            dropout_key = f"{agent_name}_dropout_active"
            status = "continuing"

            decision = policy_info.get(decision_key)
            override = policy_info.get(override_key)
            # Hang is an execution override; no new policy decision is made.
            if bool(policy_info.get(f"{agent_name}_hang_active", False)):
                step_actions[agent_name] = {
                    "action": {
                        "agent": agent_name,
                        "action": "agent_hang_noop",
                        "arguments": [],
                    },
                    "status": "hang_noop",
                }
                continue
            if isinstance(override, Mapping):
                active_actions[agent_name] = dict(override)
                status = "fault_execution_override"
            elif isinstance(decision, Mapping):
                # Full model memory is stored once in policy_info at the
                # decision boundary, not repeated in every continuing step.
                active_actions[agent_name] = {
                    key: value for key, value in decision.items()
                    if key != "replay_context"
                }
                status = "new_high_level_decision"
            elif bool(policy_info.get(dropout_key, False)):
                status = "dropout_noop"

            action = active_actions.get(agent_name)
            if action is None:
                action = {
                    "agent": agent_name,
                    "action": "not_recorded",
                    "arguments": [],
                }
                status = "not_recorded"
            step_actions[agent_name] = {
                "action": dict(action),
                "status": status,
            }
        return step_actions

    @staticmethod
    def _format_state_value(value: Any) -> str:
        value = _jsonable(value)
        if isinstance(value, list):
            components = []
            for component in value:
                if isinstance(component, float):
                    components.append(f"{component:.4f}")
                else:
                    components.append(str(component))
            return f"[{', '.join(components)}]"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    @staticmethod
    def _format_held_object(value: Any) -> str:
        if not isinstance(value, Mapping):
            return "None" if value is None else str(_jsonable(value))
        name = value.get("goal_name") or value.get("handle") or "unknown"
        return f"{name} (object_id={value.get('object_id')})"

    @staticmethod
    def _dropout_state(
        agent_name: str, policy_info: Mapping[str, Any]
    ) -> Dict[str, Any]:
        active = bool(
            policy_info.get(f"{agent_name}_dropout_active", False)
        )
        return {
            "active": active,
            "trigger_domain": policy_info.get(
                f"{agent_name}_dropout_trigger_domain"
            ),
            "since_simulator_step": policy_info.get(
                f"{agent_name}_dropout_since_simulator_step"
            ),
            "since_decision_step": policy_info.get(
                f"{agent_name}_dropout_since_decision_step"
            ),
        }

    def _enrich_simulator_state(
        self,
        simulator_state: Optional[Mapping[str, Any]],
        agent_step_actions: Mapping[str, Mapping[str, Any]],
        policy_info: Mapping[str, Any],
    ) -> Dict[str, Any]:
        raw_state = _jsonable(simulator_state or {})
        raw_agents = raw_state.get("agents", {})
        agent_names = set(raw_agents) | set(agent_step_actions)
        agents: Dict[str, Any] = {}
        for agent_name in sorted(agent_names):
            physical_state = raw_agents.get(agent_name, {})
            if not isinstance(physical_state, Mapping):
                physical_state = {}
            action_state = agent_step_actions.get(agent_name, {})
            agents[agent_name] = {
                **dict(physical_state),
                "current_high_level_action": action_state.get("action"),
                "action_status": action_state.get("status", "not_recorded"),
                "dropout_state": self._dropout_state(
                    agent_name, policy_info
                ),
                "hang_state": {
                    "active": bool(
                        policy_info.get(f"{agent_name}_hang_active", False)
                    ),
                    "since_simulator_step": policy_info.get(
                        f"{agent_name}_hang_since_simulator_step"
                    ),
                    "since_decision_step": policy_info.get(
                        f"{agent_name}_hang_since_decision_step"
                    ),
                    "trigger_domain": policy_info.get(
                        f"{agent_name}_hang_trigger_domain"
                    ),
                },
            }
        return {
            "timing": raw_state.get("timing", "post_env_step"),
            "agents": agents,
            "objects": raw_state.get("objects", []),
        }

    def record_recovery_event(self, event):
        """Persist the same recovery audit in JSONL and the readable episode trace."""
        self.record(event)
        self._append_text(
            event["episode_id"],
            "Recovery_Event: " + json.dumps(_jsonable(event), ensure_ascii=False) + "\n\n",
        )

    def record_system_crash(self, report, *, terminal_step_recorded=False):
        """A pre-action abort is a terminal event, never a fabricated env step."""
        self.record(report)
        lines = [f"System_Crash: reason={report['reason']}, agent={report['agent']}, "
                 f"detected_step={report['simulator_step']}, phase={report['phase']}, "
                 f"completed_steps={report['completed_steps']}, consecutive_count={report['consecutive_count']}"]
        if not terminal_step_recorded:
            lines.append("Episode_Result: done=True, termination_reason=error_detection, "
                         + ", ".join(f"{key}={value}" for key, value in report["metrics"].items()))
        self._append_text(report["episode_id"], "\n".join(lines) + "\n\n")

    def record_step(
        self,
        *,
        episode_id: Any,
        scene_id: str,
        simulator_step: int,
        joint_env_action: Any,
        policy_info: Optional[Mapping[str, Any]],
        reward: Any,
        done: bool,
        metrics: Mapping[str, Any],
        text_context: Optional[Mapping[str, Any]] = None,
        simulator_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        info = {} if policy_info is None else policy_info
        agent_step_actions = self._agent_step_actions(episode_id, info)
        dynamic_state = self._enrich_simulator_state(
            simulator_state, agent_step_actions, info
        )
        self.record(
            {
                "type": "simulator_step",
                "episode_id": str(episode_id),
                "scene_id": str(scene_id),
                "simulator_step": int(simulator_step),
                "joint_env_action": joint_env_action,
                "policy_info": info,
                "fault_events": [event for key, events in info.items() if key.endswith("_fault_events")
                                 for event in events],
                "agent_step_actions": agent_step_actions,
                "simulator_state": dynamic_state,
                "reward": reward,
                "done": bool(done),
                "metrics": metrics,
                "text_context": text_context,
            }
        )

        lines = []
        for agent_name, step_action in agent_step_actions.items():
            agent_id = agent_name.removeprefix("agent_")
            agent_state = dynamic_state["agents"].get(agent_name, {})
            if "position" in agent_state:
                lines.append(
                    f"Agent_{agent_id}_Position: "
                    f"{self._format_state_value(agent_state['position'])}"
                )
            if "rotation" in agent_state:
                lines.append(
                    f"Agent_{agent_id}_Rotation: "
                    f"{self._format_state_value(agent_state['rotation'])}"
                )
            if "held_object" in agent_state:
                lines.append(
                    f"Agent_{agent_id}_Held_Object: "
                    f"{self._format_held_object(agent_state['held_object'])}"
                )
            lines.append(
                f"Agent_{agent_id}_Action: "
                f"{self._format_action(step_action['action'])}"
            )
            lines.append(
                f"Agent_{agent_id}_Current_High_Level_Action: "
                f"{self._format_action(step_action['action'])}"
            )
            lines.append(
                f"Agent_{agent_id}_Status: {step_action['status']}"
            )
            dropout_state = agent_state.get("dropout_state", {})
            if dropout_state.get("active"):
                lines.append(
                    f"Agent_{agent_id}_Dropout_State: active "
                    f"(trigger_domain={dropout_state.get('trigger_domain')}, "
                    "since_simulator_step="
                    f"{dropout_state.get('since_simulator_step')}, "
                    "since_decision_step="
                    f"{dropout_state.get('since_decision_step')})"
                )
            else:
                lines.append(f"Agent_{agent_id}_Dropout_State: inactive")
            hang_state = agent_state.get("hang_state", {})
            if hang_state.get("active"):
                lines.append(
                    f"Agent_{agent_id}_Hang_State: active "
                    f"(since_simulator_step={hang_state.get('since_simulator_step')}, "
                    f"since_decision_step={hang_state.get('since_decision_step')})"
                )
            else:
                lines.append(f"Agent_{agent_id}_Hang_State: inactive")

        objects = dynamic_state.get("objects", [])
        if objects:
            lines.append("Object_Positions:")
            for object_state in objects:
                if not isinstance(object_state, Mapping):
                    continue
                object_name = (
                    object_state.get("goal_name")
                    or object_state.get("handle")
                    or "unknown"
                )
                lines.append(
                    f"  Object_Position: name={object_name}, "
                    f"object_id={object_state.get('object_id')}, "
                    f"handle={object_state.get('handle')}, position="
                    f"{self._format_state_value(object_state.get('position'))}"
                )

        for key, value in info.items():
            if key.endswith("_fault_events"):
                for event in value:
                    lines.append(
                        f"Fault: {event.get('fault_type')}/{event.get('fault_subtype')} "
                        + json.dumps(_jsonable(event), ensure_ascii=False)
                    )
            if key.endswith("_hang_event") and isinstance(value, Mapping):
                lines.append(
                    f"Fault: StructureFault/AgentHang {value.get('type')} "
                    f"at simulator step {value.get('simulator_step')} "
                    f"for {value.get('agent')} "
                    f"(decision_step={value.get('decision_step')}; until episode end)."
                )
            match = re.match(r"agent_(\d+)_high_level_decision$", key)
            if match and isinstance(value, Mapping):
                agent_id = match.group(1)
                if value.get("decision_step") is not None:
                    lines.append(
                        f"Agent_{agent_id}_Decision_Step: "
                        f"{value['decision_step']}"
                    )
                if value.get("action") in {
                    "agent_dropout_noop", "agent_hang_noop"
                }:
                    lines.append(
                        f"Agent_{agent_id}_Observation: Policy and LLM were "
                        "not called; a zero no-op action was sent."
                    )
                else:
                    lines.append(
                        f"Agent_{agent_id}_Observation: Action accepted; "
                        "low-level execution started."
                    )

            match = re.match(r"agent_(\d+)_pddl_action$", key)
            if match:
                lines.append(
                    f"Agent_{match.group(1)}_Observation: Successful "
                    f"execution! Applied {value}."
                )

            match = re.match(r"agent_(\d+)_execution_override$", key)
            if match and isinstance(value, Mapping):
                agent_id = match.group(1)
                lines.append(
                    f"Agent_{agent_id}_Observation: The running low-level "
                    "skill was interrupted; policy and LLM were not called, "
                    "and a zero no-op action was sent."
                )

            match = re.match(r"agent_(\d+)_dropout_event$", key)
            if match and isinstance(value, Mapping):
                agent_name = f"agent_{match.group(1)}"
                if value.get("trigger_domain") == "high_level_decision":
                    lines.append(
                        "Fault: StructureFault/AgentDropout injected at "
                        f"PRE_HIGH_LEVEL_DECISION for {agent_name} "
                        f"decision step {value.get('decision_step')} "
                        f"(simulator step {value.get('simulator_step')})."
                    )
                else:
                    lines.append(
                        "Fault: StructureFault/AgentDropout injected at "
                        f"PRE_ACTION simulator step "
                        f"{value.get('simulator_step', simulator_step)} for "
                        f"{agent_name}; {value.get('high_level_decisions_completed')} "
                        "high-level decisions had completed."
                    )

        if done:
            scalar_metrics = _jsonable(metrics)
            metric_text = ", ".join(
                f"{key}={value}" for key, value in scalar_metrics.items()
            )
            lines.append(
                f"Episode_Result: done=True, reward={_jsonable(reward)}"
                + (f", {metric_text}" if metric_text else "")
            )

        block = f"Simulator_Step: {int(simulator_step)}\n"
        block += "\n".join(lines) + "\n"
        if text_context is not None:
            scene_description = str(
                text_context.get("scene_description", "")
            ).strip()
            if scene_description:
                block += f"Scene_Description:\n{scene_description}\n"
        block += "\n"
        self._append_text(episode_id, block)
