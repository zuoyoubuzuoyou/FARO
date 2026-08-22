"""MVP object-localization fault for symbolic EMOS observations.

The injector changes only the natural-language scene description consumed by
one LLM agent.  It never changes Habitat's simulator state or the context seen
by other agents.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple


class FaultInjectionError(ValueError):
    """Raised when an enabled fault cannot be applied unambiguously."""


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True)
class ObjectLocalizationFaultSpec:
    """Validated runtime view of the Hydra fault configuration."""

    enabled: bool = False
    fault_id: str = "PF_object_localization_001"
    episode_id: str = ""
    agent: str = "agent_0"
    decision_step: int = 0
    object_name: str = ""
    replacement_region: str = ""
    replacement_floor: str = ""
    replacement_height: str = ""
    replacement_horizontal_distance: str = ""

    @classmethod
    def from_config(cls, config: Any) -> "ObjectLocalizationFaultSpec":
        return cls(
            enabled=bool(_config_value(config, "enabled", False)),
            fault_id=str(
                _config_value(
                    config, "fault_id", "PF_object_localization_001"
                )
            ),
            episode_id=str(_config_value(config, "episode_id", "")),
            agent=str(_config_value(config, "agent", "agent_0")),
            decision_step=int(_config_value(config, "decision_step", 0)),
            object_name=str(_config_value(config, "object_name", "")),
            replacement_region=str(
                _config_value(config, "replacement_region", "")
            ),
            replacement_floor=str(
                _config_value(config, "replacement_floor", "")
            ),
            replacement_height=str(
                _config_value(config, "replacement_height", "")
            ),
            replacement_horizontal_distance=str(
                _config_value(
                    config, "replacement_horizontal_distance", ""
                )
            ),
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.object_name:
            raise FaultInjectionError(
                "object_localization.object_name must be set"
            )
        if self.decision_step < 0:
            raise FaultInjectionError(
                "object_localization.decision_step must be >= 0"
            )
        if not any(
            (
                self.replacement_region,
                self.replacement_height,
                self.replacement_horizontal_distance,
            )
        ):
            raise FaultInjectionError(
                "set replacement_region, replacement_height, or "
                "replacement_horizontal_distance"
            )

    def matches(
        self, *, episode_id: Any, agent: str, decision_step: int
    ) -> bool:
        if not self.enabled:
            return False
        episode_matches = not self.episode_id or self.episode_id == str(
            episode_id
        )
        return (
            episode_matches
            and self.agent == str(agent)
            and self.decision_step == int(decision_step)
        )


def inject_object_localization_error(
    scene_description: str,
    spec: ObjectLocalizationFaultSpec,
    *,
    episode_id: Any,
    agent: str,
    decision_step: int,
    simulator_step: Optional[int] = None,
) -> Tuple[str, dict[str, Any]]:
    """Replace one object's symbolic location facts and return an audit event.

    MP3D descriptions expose a semantic region and floor. HSSD descriptions
    currently expose height and horizontal distance instead, so the MVP
    supports both representations. At least one configured fact must match;
    otherwise the run fails loudly rather than claiming a fault was injected.
    """

    spec.validate()
    if not spec.matches(
        episode_id=episode_id,
        agent=agent,
        decision_step=decision_step,
    ):
        raise FaultInjectionError(
            "inject_object_localization_error called for a non-matching trigger"
        )

    object_pattern = re.escape(spec.object_name)
    mutated = str(scene_description)
    changes: list[dict[str, str]] = []

    if spec.replacement_region:
        region_pattern = re.compile(
            rf'The object "{object_pattern}" is located in '
            r'(?P<region>.+?) on (?P<floor>.+?) floor\.'
        )

        def replace_region(match: re.Match[str]) -> str:
            old_region = match.group("region")
            old_floor = match.group("floor")
            new_floor = spec.replacement_floor or old_floor
            before = match.group(0)
            after = (
                f'The object "{spec.object_name}" is located in '
                f"{spec.replacement_region} on {new_floor} floor."
            )
            changes.append(
                {
                    "field": "region",
                    "before": old_region,
                    "after": spec.replacement_region,
                    "fact_before": before,
                    "fact_after": after,
                }
            )
            if new_floor != old_floor:
                changes.append(
                    {
                        "field": "floor",
                        "before": old_floor,
                        "after": new_floor,
                        "fact_before": before,
                        "fact_after": after,
                    }
                )
            return after

        region_matches = list(region_pattern.finditer(mutated))
        if not region_matches:
            raise FaultInjectionError(
                f'object "{spec.object_name}" has no MP3D region fact in '
                "scene_description"
            )
        if len(region_matches) > 1:
            raise FaultInjectionError(
                f'object "{spec.object_name}" has multiple MP3D region facts; '
                "use a unique object identifier"
            )
        mutated = region_pattern.sub(replace_region, mutated, count=1)

    if spec.replacement_height:
        height_pattern = re.compile(
            rf'The height of (?:object )?"{object_pattern}"(?: from the floor)? '
            r'is (?P<value>[-+]?\d+(?:\.\d+)?|inf)\.'
        )

        def replace_height(match: re.Match[str]) -> str:
            before = match.group(0)
            old_value = match.group("value")
            value_start, value_end = match.span("value")
            relative_start = value_start - match.start()
            relative_end = value_end - match.start()
            after = (
                before[:relative_start]
                + spec.replacement_height
                + before[relative_end:]
            )
            changes.append(
                {
                    "field": "height",
                    "before": old_value,
                    "after": spec.replacement_height,
                    "fact_before": before,
                    "fact_after": after,
                }
            )
            return after

        height_matches = list(height_pattern.finditer(mutated))
        if not height_matches:
            raise FaultInjectionError(
                f'object "{spec.object_name}" has no height fact in '
                "scene_description"
            )
        if len(height_matches) > 1:
            raise FaultInjectionError(
                f'object "{spec.object_name}" has multiple height facts; '
                "use a unique object identifier"
            )
        mutated = height_pattern.sub(replace_height, mutated, count=1)

    if spec.replacement_horizontal_distance:
        distance_pattern = re.compile(
            rf'The horizontal distance of "{object_pattern}" to the nearest '
            r'navigable point is (?P<value>[-+]?\d+(?:\.\d+)?|inf)\.'
        )

        def replace_distance(match: re.Match[str]) -> str:
            before = match.group(0)
            old_value = match.group("value")
            value_start, value_end = match.span("value")
            relative_start = value_start - match.start()
            relative_end = value_end - match.start()
            after = (
                before[:relative_start]
                + spec.replacement_horizontal_distance
                + before[relative_end:]
            )
            changes.append(
                {
                    "field": "horizontal_distance",
                    "before": old_value,
                    "after": spec.replacement_horizontal_distance,
                    "fact_before": before,
                    "fact_after": after,
                }
            )
            return after

        distance_matches = list(distance_pattern.finditer(mutated))
        if not distance_matches:
            raise FaultInjectionError(
                f'object "{spec.object_name}" has no horizontal-distance '
                "fact in scene_description"
            )
        if len(distance_matches) > 1:
            raise FaultInjectionError(
                f'object "{spec.object_name}" has multiple '
                "horizontal-distance facts; use a unique object identifier"
            )
        mutated = distance_pattern.sub(replace_distance, mutated, count=1)

    if not changes or mutated == scene_description:
        raise FaultInjectionError(
            "configured object-localization fault produced no change"
        )

    event = {
        "schema_version": 1,
        "type": "fault_injected",
        "fault_id": spec.fault_id,
        "fault_type": "PerceptionFault",
        "fault_subtype": "ObjectLocalizationError",
        "phase": "POST_PERCEPTION",
        "episode_id": str(episode_id),
        "agent": str(agent),
        "decision_step": int(decision_step),
        "simulator_step": (
            None if simulator_step is None else int(simulator_step)
        ),
        "target": {"object": spec.object_name},
        "changes": changes,
        "ground_truth_preserved": True,
    }
    return mutated, event


class FaultEventRecorder:
    """Append fault events as JSON Lines for later clean/fault alignment."""

    def __init__(self, output_dir: str, run_label: str) -> None:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_label))
        self.path = Path(output_dir) / f"{safe_label}_fault_events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.count = 0

    def record(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(dict(event), ensure_ascii=False))
            output.write("\n")
            output.flush()
        self.count += 1
