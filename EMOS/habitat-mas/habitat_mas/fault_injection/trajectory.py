"""Per-step trajectory recording and paired trajectory comparison for EMOS."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


_LOCALIZATION_KEY = re.compile(r"^(agent_\d+)_localization_sensor$")


def _json_value(value: Any) -> Any:
    """Convert tensors, arrays, and scalar wrappers to JSON-compatible data."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _atomic_json_dump(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        temporary_path = output_file.name
    os.replace(temporary_path, path)


class TrajectoryRecorder:
    """Collect one pre-action state for every Habitat environment step."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self._episodes: Dict[str, Dict[str, Any]] = {}

    def start_episode(self, episode_id: Any, scene_id: Any) -> None:
        episode_key = str(episode_id)
        self._episodes[episode_key] = {
            "schema_version": 1,
            "episode_id": episode_key,
            "scene_id": str(scene_id),
            "coordinate_format": "[x, y, z] in Habitat world coordinates",
            "step_semantics": (
                "Each entry stores the agent state immediately before the "
                "executed action for that Habitat step."
            ),
            "steps": [],
        }

    def make_step(
        self,
        *,
        episode_id: Any,
        step: int,
        batch: Mapping[str, Any],
        env_index: int,
        action_data: Any,
    ) -> Dict[str, Any]:
        agents: Dict[str, Dict[str, Any]] = {}
        for key, values in batch.items():
            match = _LOCALIZATION_KEY.match(key)
            if match is None:
                continue
            localization = _json_value(values[env_index])
            if not isinstance(localization, list) or len(localization) < 4:
                continue
            agents[match.group(1)] = {
                "position": localization[:3],
                "heading": localization[3],
            }

        executed_action = _json_value(action_data.env_actions[env_index])
        agent_actions: Dict[str, Any] = {}
        action_lengths = getattr(action_data, "length_take_actions", None)
        if action_lengths is None:
            action_lengths = getattr(action_data, "length_actions", None)
        if action_lengths is not None and isinstance(executed_action, list):
            offset = 0
            for agent_index, length in enumerate(action_lengths):
                length = int(length)
                agent_actions[f"agent_{agent_index}"] = executed_action[
                    offset : offset + length
                ]
                offset += length

        policy_info = None
        if action_data.policy_info is not None:
            policy_info = _json_value(action_data.policy_info[env_index])

        return {
            "step": int(step),
            "agents": agents,
            "executed_action": executed_action,
            "agent_actions": agent_actions,
            "policy_info": policy_info,
        }

    def finish_step(
        self,
        record: Dict[str, Any],
        *,
        episode_id: Any,
        reward: Any,
        done: bool,
        info: Mapping[str, Any],
    ) -> None:
        episode_key = str(episode_id)
        if episode_key not in self._episodes:
            raise RuntimeError(
                f"Trajectory episode {episode_key!r} was not initialized."
            )
        record["reward"] = float(reward)
        record["done"] = bool(done)
        record["metrics"] = {
            str(key): _json_value(value)
            for key, value in info.items()
            if isinstance(value, (int, float, bool))
            or hasattr(value, "item")
        }
        self._episodes[episode_key]["steps"].append(record)
        if done:
            self.finish_episode(episode_key)

    def finish_episode(self, episode_id: Any) -> Path:
        episode_key = str(episode_id)
        episode = self._episodes.pop(episode_key)
        episode["num_steps"] = len(episode["steps"])
        output_path = self.output_dir / f"episode_{episode_key}.json"
        _atomic_json_dump(output_path, episode)
        print(
            f"[Trajectory] episode={episode_key} steps={episode['num_steps']} "
            f"path={output_path}"
        )
        return output_path


def _load_trajectory(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as trajectory_file:
        trajectory = json.load(trajectory_file)
    if not isinstance(trajectory.get("steps"), list):
        raise ValueError(f"Trajectory {path!r} has no steps list.")
    return trajectory


def _distance(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)))


def _path_length(steps: Sequence[Mapping[str, Any]], agent: str) -> float:
    positions = [
        step["agents"][agent]["position"]
        for step in steps
        if agent in step.get("agents", {})
    ]
    return sum(
        _distance(first, second)
        for first, second in zip(positions, positions[1:])
    )


def compare_trajectories(
    baseline_path: str,
    fault_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Align two trajectories by step and write complete JSON/CSV differences."""

    baseline = _load_trajectory(baseline_path)
    fault = _load_trajectory(fault_path)
    if baseline.get("episode_id") != fault.get("episode_id"):
        raise ValueError(
            "Trajectory episode IDs differ: "
            f"{baseline.get('episode_id')!r} != {fault.get('episode_id')!r}."
        )

    baseline_steps = {int(step["step"]): step for step in baseline["steps"]}
    fault_steps = {int(step["step"]): step for step in fault["steps"]}
    all_steps = sorted(set(baseline_steps) | set(fault_steps))
    agents = sorted(
        {
            agent
            for trajectory_step in list(baseline_steps.values())
            + list(fault_steps.values())
            for agent in trajectory_step.get("agents", {})
        }
    )

    aligned_steps = []
    distances_by_agent: Dict[str, list] = {agent: [] for agent in agents}
    for step_index in all_steps:
        baseline_step = baseline_steps.get(step_index)
        fault_step = fault_steps.get(step_index)
        comparison: Dict[str, Any] = {
            "step": step_index,
            "baseline_present": baseline_step is not None,
            "fault_present": fault_step is not None,
            "agents": {},
        }
        for agent in agents:
            baseline_agent = (
                baseline_step.get("agents", {}).get(agent)
                if baseline_step is not None
                else None
            )
            fault_agent = (
                fault_step.get("agents", {}).get(agent)
                if fault_step is not None
                else None
            )
            distance = None
            heading_difference = None
            if baseline_agent is not None and fault_agent is not None:
                distance = _distance(
                    baseline_agent["position"], fault_agent["position"]
                )
                heading_difference = abs(
                    float(baseline_agent["heading"])
                    - float(fault_agent["heading"])
                )
                distances_by_agent[agent].append((step_index, distance))
            comparison["agents"][agent] = {
                "baseline": baseline_agent,
                "fault": fault_agent,
                "position_distance": distance,
                "heading_difference": heading_difference,
            }
        aligned_steps.append(comparison)

    agent_summary = {}
    for agent in agents:
        distances = distances_by_agent[agent]
        values = [distance for _, distance in distances]
        first_divergent_step: Optional[int] = next(
            (step for step, distance in distances if distance > 1e-4), None
        )
        agent_summary[agent] = {
            "baseline_path_length": _path_length(baseline["steps"], agent),
            "fault_path_length": _path_length(fault["steps"], agent),
            "first_divergent_step": first_divergent_step,
            "mean_position_distance": (
                sum(values) / len(values) if values else None
            ),
            "max_position_distance": max(values) if values else None,
            "final_position_distance": values[-1] if values else None,
        }

    result = {
        "schema_version": 1,
        "episode_id": baseline.get("episode_id"),
        "baseline_num_steps": len(baseline["steps"]),
        "fault_num_steps": len(fault["steps"]),
        "agent_summary": agent_summary,
        "steps": aligned_steps,
    }
    output_path = Path(output_dir)
    _atomic_json_dump(output_path / "trajectory_comparison.json", result)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(
        output_path / "trajectory_comparison.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        fieldnames = ["step"]
        for agent in agents:
            fieldnames.extend(
                [
                    f"{agent}_baseline_x",
                    f"{agent}_baseline_y",
                    f"{agent}_baseline_z",
                    f"{agent}_fault_x",
                    f"{agent}_fault_y",
                    f"{agent}_fault_z",
                    f"{agent}_position_distance",
                    f"{agent}_heading_difference",
                ]
            )
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for comparison in aligned_steps:
            row: Dict[str, Any] = {"step": comparison["step"]}
            for agent in agents:
                agent_comparison = comparison["agents"][agent]
                for variant in ("baseline", "fault"):
                    state = agent_comparison[variant]
                    if state is not None:
                        for axis, value in zip("xyz", state["position"]):
                            row[f"{agent}_{variant}_{axis}"] = value
                row[f"{agent}_position_distance"] = agent_comparison[
                    "position_distance"
                ]
                row[f"{agent}_heading_difference"] = agent_comparison[
                    "heading_difference"
                ]
            writer.writerow(row)
    return result
