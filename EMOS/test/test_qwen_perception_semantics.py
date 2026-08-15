from unittest.mock import Mock

import pytest

from habitat_baselines.rl.multi_agent.qwen_perception_compat import (
    QwenAssignmentValidationError,
    QwenPerceptionConfigError,
    build_assignment_contract,
    extract_detection_goal_objects,
    parse_qwen_assignment,
    validate_assignment,
    validate_qwen_perception_config,
)
from habitat_baselines.rl.multi_agent.multi_llm_policy import (
    request_qwen_assignment,
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
    assert "do not leave" in contract.lower()
    assert "exactly one valid agent" in contract.lower()


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
            "{agent_0||Detect any_targets|0 and any_targets|1}"
            "{agent_1||Nothing to do}",
            "idle agents",
        ),
        (
            "{agent_0||Detect any_targets|0 and any_targets|1}"
            "{agent_1||Detect any_targets|0 and any_targets|1}",
            "duplicate goal assignments",
        ),
        (
            "{agent_0||Detect any_targets|0}"
            "{agent_1||Detect any_targets|1}"
            "{agent_2||Assist agent_1}",
            "invalid agent IDs: agent_2",
        ),
        (
            "{agent_0||Detect any_targets|0 and rearrange it}"
            "{agent_1||Detect any_targets|1}",
            "manipulation actions",
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


def test_accepts_qwen_braceless_line_assignment_format():
    response = (
        "agent_0||Detect object any_targets|0\n"
        "agent_1||Detect object any_targets|1"
    )

    tasks = parse_qwen_assignment(response)

    assert tasks == {
        "agent_0": "Detect object any_targets|0",
        "agent_1": "Detect object any_targets|1",
    }
    assert validate_assignment(response, tasks, ROBOT_IDS, GOAL_OBJECTS) == ()


def test_braceless_assignment_reports_semantics_instead_of_missing_agents():
    response = (
        "agent_0||Detect and rearrange any_targets|0 to TARGET_any_targets|0\n"
        "agent_1||Detect any_targets|0 and any_targets|1"
    )

    violations = validate_assignment(
        response,
        parse_qwen_assignment(response),
        ROBOT_IDS,
        GOAL_OBJECTS,
    )

    assert not any("missing agent IDs" in item for item in violations)
    assert any("duplicate goal assignments" in item for item in violations)
    assert any("manipulation actions" in item for item in violations)


def test_qwen_leader_retries_invalid_assignment_and_accepts_model_correction():
    leader = Mock()
    leader.chat.side_effect = [
        "{agent_0||Nothing to do}{agent_1||Detect TARGET_any_targets|0}",
        "{agent_0||Detect any_targets|0}{agent_1||Detect any_targets|1}",
    ]

    response, tasks = request_qwen_assignment(
        leader,
        "initial prompt",
        ROBOT_IDS,
        GOAL_OBJECTS,
    )

    assert response.endswith("{agent_1||Detect any_targets|1}")
    assert tasks == {
        "agent_0": "Detect any_targets|0",
        "agent_1": "Detect any_targets|1",
    }
    assert leader.chat.call_count == 2
    correction = leader.chat.call_args_list[1].args[0]
    assert "uncovered goal objects" in correction
    assert "Decide the agent-to-object mapping yourself" in correction


def test_qwen_leader_exhaustion_never_synthesizes_assignment():
    leader = Mock()
    leader.chat.return_value = (
        "{agent_0||Nothing to do}{agent_1||Nothing to do}"
    )

    with pytest.raises(
        QwenAssignmentValidationError,
        match="uncovered goal objects",
    ):
        request_qwen_assignment(
            leader,
            "initial prompt",
            ROBOT_IDS,
            GOAL_OBJECTS,
        )

    assert leader.chat.call_count == 3


def test_qwen_perception_base_config_fails_fast():
    with pytest.raises(
        QwenPerceptionConfigError, match="llm_spot_drone_per_qwen"
    ):
        validate_qwen_perception_config(
            "multi_rearrange/llm_spot_drone_per",
            ("any_targets|0",),
            {"agent_0": ("nav_to_obj", "pick", "place", "wait")},
        )


def test_qwen_overlay_perception_actions_are_accepted():
    validate_qwen_perception_config(
        "multi_rearrange/llm_spot_drone_per_qwen",
        ("any_targets|0",),
        {
            "agent_0": ("send_request", "nav_to_obj", "wait"),
            "agent_1": ("send_request", "nav_to_obj", "wait"),
        },
    )
