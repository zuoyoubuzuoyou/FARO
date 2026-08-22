"""JSONL trajectory recording for EMOS fault-injection experiments."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Mapping, Optional


def _jsonable(value: Any) -> Any:
    """Convert common evaluator values to JSON-compatible Python values."""

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

    # Torch tensors and NumPy arrays/scalars expose one or both of these APIs.
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

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump())
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _jsonable(as_dict())

    return repr(value)


class TrajectoryRecorder:
    """Append a complete evaluator trace to one run-specific JSONL file.

    The trace is event based because high-level decisions occur asynchronously
    and much less frequently than Habitat environment steps.
    """

    schema_version = 1

    def __init__(self, output_dir: str, run_label: str, mode: str) -> None:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
        self.run_label = str(run_label)
        self.mode = str(mode)
        self.path = Path(output_dir) / f"{safe_label}_trajectory.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._event_index = 0
        self.record(
            {
                "type": "trajectory_start",
                "run_label": self.run_label,
                "mode": self.mode,
            }
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
                json.dumps(_jsonable(payload), ensure_ascii=False, sort_keys=True)
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

    def record_step(
        self,
        *,
        episode_id: Any,
        scene_id: str,
        simulator_step: int,
        joint_env_action: Any,
        world_state_before: Mapping[str, Any],
        policy_info: Optional[Mapping[str, Any]],
        reward: Any,
        done: bool,
        metrics: Mapping[str, Any],
    ) -> None:
        self.record(
            {
                "type": "simulator_step",
                "episode_id": str(episode_id),
                "scene_id": str(scene_id),
                "simulator_step": int(simulator_step),
                "joint_env_action": joint_env_action,
                "world_state_before": world_state_before,
                "policy_info": {} if policy_info is None else policy_info,
                "reward": reward,
                "done": bool(done),
                "metrics": metrics,
            }
        )

    def record_episode_end(
        self,
        *,
        episode_id: Any,
        scene_id: str,
        simulator_step: int,
        result: Mapping[str, Any],
    ) -> None:
        self.record(
            {
                "type": "episode_end",
                "episode_id": str(episode_id),
                "scene_id": str(scene_id),
                "simulator_step": int(simulator_step),
                "result": result,
            }
        )
