import pytest

from habitat_baselines.rl.multi_agent.qwen_perception_compat import (
    build_assignment_contract,
    extract_detection_goal_objects,
    parse_qwen_assignment,
    validate_assignment,
)


TASK = """Goal of this episode is the logical operation and of the following conditions:
0. The object 'any_targets|1' has been detected by any robot.
1. The object 'any_targets|0' has been detected by any robot."""
ROBOT_IDS = ("agent_0", "agent_1")
GOAL_OBJECTS = ("any_targets|1", "any_targets|0")


def test_extracts_only_explicit_detection_goal_objects():
    task_with_scene_identifiers = (
        TASK
        + "\nThe scene also contains 'TARGET_any_targets|0' and "
        "'TARGET_any_targets|1'."
    )

    assert extract_detection_goal_objects(task_with_scene_identifiers) == GOAL_OBJECTS


def test_assignment_contract_identifies_real_agents_and_goal_vocabulary():
    contract = build_assignment_contract(ROBOT_IDS, GOAL_OBJECTS)

    assert "agent_0, agent_1" in contract
    assert "any_targets|1, any_targets|0" in contract
    assert "TARGET_any_targets" in contract
    assert "do not invent" in contract.lower()


@pytest.mark.parametrize(
    ("response", "expected_fragment"),
    [
        (
            "{agent_0||Nothing to do}"
            "{agent_1||Detect TARGET_any_targets|0 and TARGET_any_targets|1 "
            "with a second robot}",
            "not explicit goal objects",
        ),
        (
            "{agent_0||Detect TARGET_any_targets|1}"
            "{agent_1||Detect any_targets|1 and TARGET_any_targets|0}",
            "uncovered goal objects: any_targets|0",
        ),
        (
            "{agent_0||Nothing to do}{agent_1||Nothing to do}",
            "uncovered goal objects",
        ),
        (
            "{agent_0||Detect any_targets|0}"
            "{agent_1||Detect any_targets|1}"
            "{agent_2||Assist agent_1}",
            "invalid agent IDs: agent_2",
        ),
    ],
)
def test_rejects_observed_invalid_assignments(response, expected_fragment):
    tasks = parse_qwen_assignment(response)

    violations = validate_assignment(
        response,
        tasks,
        ROBOT_IDS,
        GOAL_OBJECTS,
    )

    assert any(expected_fragment in item for item in violations)


def test_rejects_duplicate_and_missing_real_agent_entries():
    response = (
        "{agent_0||Detect any_targets|0}"
        "{agent_0||Detect any_targets|1}"
    )

    violations = validate_assignment(
        response,
        parse_qwen_assignment(response),
        ROBOT_IDS,
        GOAL_OBJECTS,
    )

    assert "duplicate agent IDs: agent_0" in violations
    assert "missing agent IDs: agent_1" in violations


def test_accepts_complete_assignment_generated_by_the_model():
    response = (
        "{agent_0||Detect object any_targets|0}"
        "{agent_1||Detect object any_targets|1}"
    )
    tasks = parse_qwen_assignment(response)

    assert validate_assignment(
        response,
        tasks,
        ROBOT_IDS,
        GOAL_OBJECTS,
    ) == ()
