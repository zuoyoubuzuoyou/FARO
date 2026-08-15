from __future__ import annotations

import re
from collections import Counter


DETECTION_GOAL_RE = re.compile(
    r"^\s*\d+\.\s+The object ['\"]([^'\"]+)['\"] "
    r"has been detected by any robot\.\s*$",
    re.MULTILINE,
)
OBJECT_ID_RE = re.compile(r"(?:TARGET_)?any_targets\|\d+")
ASSIGNMENT_RE = re.compile(r"\{([^{}|]+)\|\|([^{}]*)\}")


class QwenAssignmentValidationError(ValueError):
    pass


class QwenPerceptionConfigError(ValueError):
    pass


def _unique(items):
    return tuple(dict.fromkeys(items))


def extract_detection_goal_objects(task_description: str) -> tuple[str, ...]:
    return _unique(DETECTION_GOAL_RE.findall(task_description))


def extract_object_ids(text: str) -> tuple[str, ...]:
    return _unique(OBJECT_ID_RE.findall(text))


def build_assignment_contract(
    robot_ids: tuple[str, ...],
    goal_objects: tuple[str, ...],
) -> str:
    return (
        " Qwen assignment contract: The only valid agent IDs are: "
        f"{', '.join(robot_ids)}. The only objects required by the explicit "
        f"goal conditions are: {', '.join(goal_objects)}. Assign every required "
        "object to at least one valid agent. TARGET_any_targets|* identifiers are "
        "rearrangement destination markers, not implicit detection goals. Do not "
        "invent agents, sensors, camera poses, field-of-view values, altitudes, "
        "or additional target objects."
    )


def parse_qwen_assignment(text: str) -> dict[str, str]:
    return {
        robot_id.strip(): subtask.strip()
        for robot_id, subtask in ASSIGNMENT_RE.findall(text)
    }


def validate_assignment(
    text: str,
    tasks: dict[str, str],
    robot_ids: tuple[str, ...],
    goal_objects: tuple[str, ...],
) -> tuple[str, ...]:
    entries = [
        (robot_id.strip(), subtask.strip())
        for robot_id, subtask in ASSIGNMENT_RE.findall(text)
    ]
    entry_counts = Counter(robot_id for robot_id, _ in entries)
    valid_robot_ids = set(robot_ids)
    valid_goal_objects = set(goal_objects)
    violations = []

    invalid_ids = _unique(
        robot_id for robot_id, _ in entries if robot_id not in valid_robot_ids
    )
    if invalid_ids:
        violations.append(f"invalid agent IDs: {', '.join(invalid_ids)}")

    duplicate_ids = tuple(
        robot_id for robot_id in robot_ids if entry_counts[robot_id] > 1
    )
    if duplicate_ids:
        violations.append(f"duplicate agent IDs: {', '.join(duplicate_ids)}")

    missing_ids = tuple(
        robot_id for robot_id in robot_ids if entry_counts[robot_id] == 0
    )
    if missing_ids:
        violations.append(f"missing agent IDs: {', '.join(missing_ids)}")

    assigned_objects = []
    for robot_id in robot_ids:
        subtask = tasks.get(robot_id, "")
        if subtask.strip().lower() == "nothing to do":
            continue
        assigned_objects.extend(extract_object_ids(subtask))

    irrelevant_objects = _unique(
        object_id
        for object_id in assigned_objects
        if object_id not in valid_goal_objects
    )
    if irrelevant_objects:
        violations.append(
            "not explicit goal objects: " + ", ".join(irrelevant_objects)
        )

    assigned_goal_objects = set(assigned_objects) & valid_goal_objects
    uncovered_objects = tuple(
        object_id
        for object_id in goal_objects
        if object_id not in assigned_goal_objects
    )
    if uncovered_objects:
        violations.append(
            "uncovered goal objects: " + ", ".join(uncovered_objects)
        )

    return tuple(violations)


def build_assignment_correction(
    violations: tuple[str, ...],
    robot_ids: tuple[str, ...],
    goal_objects: tuple[str, ...],
) -> str:
    return (
        "Your previous assignment is invalid:\n- "
        + "\n- ".join(violations)
        + f"\nValid agent IDs: {', '.join(robot_ids)}."
        + f"\nRequired goal objects: {', '.join(goal_objects)}."
        + "\nReturn a corrected assignment in the required format. Decide the "
        "agent-to-object mapping yourself."
    )


def validate_qwen_perception_config(
    config_name: str,
    goal_objects: tuple[str, ...],
    exposed_actions: dict[str, tuple[str, ...]],
) -> None:
    if not goal_objects:
        return

    forbidden_actions = {"pick", "place", "reset_arm"}
    exposed_forbidden = {
        agent_id: tuple(
            action for action in actions if action in forbidden_actions
        )
        for agent_id, actions in exposed_actions.items()
    }
    exposed_forbidden = {
        agent_id: actions
        for agent_id, actions in exposed_forbidden.items()
        if actions
    }
    if exposed_forbidden:
        required = "multi_rearrange/llm_spot_drone_per_qwen.yaml"
        raise QwenPerceptionConfigError(
            "Qwen detection-only run exposes manipulation actions "
            f"{exposed_forbidden} under config {config_name!r}. "
            f"Use --config-name={required}."
        )
