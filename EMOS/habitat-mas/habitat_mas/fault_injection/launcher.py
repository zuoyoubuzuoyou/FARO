"""Plan reproducible, simulator-step fault runs without importing Habitat."""

from __future__ import annotations

import gzip
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from habitat_mas.fault_injection.agent_dropout import AgentDropoutFaultSpec
from habitat_mas.fault_injection.agent_hang import AgentHangFaultSpec


@dataclass(frozen=True)
class FaultDefinition:
    config_path: str
    spec_type: type
    aliases: tuple[str, ...] = ()
    overrides: tuple[str, ...] = ()


# Add future injectors here after implementing their runtime/config support.
# Each injector accepts agent, episode_id and simulator_step_schedule.
FAULTS = {
    "agent_dropout": FaultDefinition(
        "habitat_baselines.fault_injection.structure.agent_dropout",
        AgentDropoutFaultSpec,
        aliases=("dropout",),
        overrides=("trigger_domain=simulator_step",),
    ),
    "agent_hang": FaultDefinition(
        "habitat_baselines.fault_injection.structure.agent_hang",
        AgentHangFaultSpec,
        aliases=("hang",),
    ),
}


def normalize_fault(value: str) -> str:
    name = re.sub(r"[\s-]+", "_", value.strip().lower())
    for canonical, definition in FAULTS.items():
        if name == canonical or name in definition.aliases:
            return canonical
    raise ValueError(f"Unknown fault {value!r}; choose: {', '.join(FAULTS)}")


def read_dataset(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as source:
        data = json.load(source)
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError("Dataset must be a JSON/JSON.gz object with episodes")
    if not data["episodes"]:
        raise ValueError("Dataset has no inline episodes; use a monolithic episode dataset")
    for episode in data["episodes"]:
        if not isinstance(episode, dict) or not {"episode_id", "scene_id"} <= episode.keys():
            raise ValueError("Every episode must contain episode_id and scene_id")
        episode_id = str(episode["episode_id"])
        if not episode_id or episode_id.strip() != episode_id or any(c in episode_id for c in "@;"):
            raise ValueError(f"Episode ID cannot be encoded in a fault schedule: {episode_id!r}")
    return data


def episode_key(episode: dict[str, Any]) -> tuple[str, str]:
    return str(episode["scene_id"]), str(episode["episode_id"])


def select_episodes(
    episodes: list[dict[str, Any]],
    *,
    episode_ids: list[str] | None,
    random_count: int | None,
    seed: int,
    scene: str | None = None,
    content_scenes: tuple[str, ...] = ("*",),
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        scene_id, episode_id = episode_key(episode)
        scene_name = Path(scene_id).stem
        if "*" not in content_scenes and scene_name not in content_scenes:
            continue
        if scene is not None and scene not in (scene_id, scene_name):
            continue
        groups.setdefault(episode_id, []).append(episode)
    if not groups:
        raise ValueError("No episodes match the scene selection")
    if (episode_ids is None) == (random_count is None):
        raise ValueError("Select episode IDs or a random episode count")
    if episode_ids is not None:
        if len(episode_ids) != len(set(episode_ids)) or not all(episode_ids):
            raise ValueError("Episode IDs must be nonempty and unique")
        selected = []
        for episode_id in episode_ids:
            matches = groups.get(episode_id, [])
            if not matches:
                raise ValueError(f"Episode {episode_id!r} not found in the selected dataset/scenes")
            if len(matches) > 1:
                scenes = ', '.join(str(ep['scene_id']) for ep in matches)
                raise ValueError(
                    f"Episode {episode_id!r} is ambiguous; use --scene to select one: {scenes}"
                )
            selected.append(matches[0])
        return selected
    if random_count is None or not 1 <= random_count <= len(groups):
        raise ValueError(f"--random-episodes must be between 1 and {len(groups)} unique episode IDs")
    rng = random.Random(seed)
    # Schedules currently address episode IDs, so choose each ID once, then
    # choose one of its scene variants. Never renumber IDs (robot configs use them).
    return [rng.choice(groups[key]) for key in rng.sample(list(groups), random_count)]


def read_trace_steps(path: Path) -> dict[tuple[str, str], list[int]]:
    """Read one complete clean trace per scene/episode; reject partial runs."""
    steps: dict[tuple[str, str], list[int]] = {}
    completed = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "simulator_step":
                continue
            if row.get("mode") != "clean":
                raise ValueError("--trajectory requires a clean trajectory")
            key = episode_key(row)
            step = row["simulator_step"]
            sequence = steps.setdefault(key, [])
            if key in completed or type(step) is not int or step != len(sequence):
                raise ValueError(f"Trace for {key} repeats or has missing/unordered steps")
            sequence.append(step)
            if row.get("done"):
                completed.add(key)
    return {key: sequence for key, sequence in steps.items() if key in completed}


def make_schedule(
    episodes: list[dict[str, Any]],
    *,
    seed: int,
    max_episode_steps: int,
    step: int | None = None,
    trace_steps: dict[tuple[str, str], list[int]] | None = None,
) -> dict[str, int]:
    if max_episode_steps <= 0:
        raise ValueError("A positive max_episode_steps is required")
    if (step is None) == (trace_steps is None):
        raise ValueError("Choose a fixed step or a clean trajectory")
    if step is not None and not 0 <= step < max_episode_steps:
        raise ValueError(f"--step must be in [0, {max_episode_steps - 1}]")
    rng = random.Random(seed)
    schedule = {}
    for episode in episodes:
        key = episode_key(episode)
        if key[1] in schedule:
            raise ValueError("Selected episodes must have unique episode IDs")
        if step is not None:
            chosen = step
        else:
            choices = [s for s in trace_steps.get(key, []) if s < max_episode_steps]
            if not choices:
                raise ValueError(f"No complete clean trajectory for scene/episode {key}")
            chosen = rng.choice(choices)
        schedule[key[1]] = chosen
    return schedule


def fault_overrides(
    fault: str, agent: str, schedule: dict[str, int],
    output_dir: Path, run_label: str, record_trajectory: bool,
) -> list[str]:
    definition = FAULTS[fault]
    encoded = ";".join(f"{episode}@{step}" for episode, step in schedule.items())
    definition.spec_type.from_config({
        "enabled": True, "agent": agent, "simulator_step_schedule": encoded,
    }).validate()
    # JSON quoted strings are Hydra values, not shell quoting. The launcher
    # passes each argument directly to subprocess without invoking a shell.
    values = [f"{entry.config_path}.enabled=False" for name, entry in FAULTS.items() if name != fault]
    prefix = definition.config_path
    values.extend([
        f"{prefix}.enabled=True",
        f"{prefix}.agent={json.dumps(agent)}",
        f'{prefix}.episode_id=""',
        f"{prefix}.simulator_step_schedule={json.dumps(encoded)}",
        f"{prefix}.record_trajectory={record_trajectory}",
        f"{prefix}.output_dir={json.dumps(str(output_dir))}",
        f"{prefix}.run_label={json.dumps(run_label)}",
    ])
    values.extend(f"{prefix}.{setting}" for setting in definition.overrides)
    return values
