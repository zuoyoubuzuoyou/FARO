import hashlib
import os
from unittest.mock import Mock, patch

from habitat_baselines.rl.multi_agent.multi_llm_policy import (
    create_leader_prompt,
    create_robot_prompt,
    create_robot_start_message,
    parse_agent_response,
    request_qwen_reflection,
)


def test_qwen_numerical_prompt_adds_contract_without_changing_gpt_prompt():
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "gpt"}, clear=False):
        gpt_prompt = create_robot_prompt(
            "SpotRobot", "agent_0", "Perception capability: test", True
        )
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "qwen"}, clear=False):
        qwen_prompt = create_robot_prompt(
            "SpotRobot", "agent_0", "Perception capability: test", True
        )

    assert qwen_prompt.startswith(gpt_prompt)
    assert "exactly one self-contained Python code block" in qwen_prompt
    assert "Put all imports first" in qwen_prompt
    assert "Python standard library" in qwen_prompt
    assert "name: value" in qwen_prompt
    assert "without another code block" in qwen_prompt
    assert "Correct only the reported error" in qwen_prompt
    assert "exactly one self-contained Python code block" not in gpt_prompt


def test_qwen_parser_accepts_family_format_variants():
    cases = [
        ("{{yes}}", ("yes", None)),
        ("{yes}", ("yes", None)),
        ("YES", ("yes", None)),
        ('"{{YeS}}"', ("yes", None)),
        ("```text\nYES\n```", ("yes", None)),
        ("{{no||sensor range is too short}}", ("no", "sensor range is too short")),
        ("{NO||height is out of range}", ("no", "height is out of range")),
        ("analysis\nno||cannot verify", ("no", "cannot verify")),
    ]
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "qwen"}, clear=False):
        for text, expected in cases:
            assert parse_agent_response(text) == expected, text


def test_qwen_parser_uses_last_braced_marker_and_only_final_bare_line():
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "qwen"}, clear=False):
        assert parse_agent_response("{no||first estimate}\n{{YES}}") == (
            "yes",
            None,
        )
        reasoning = "yes, the first calculation looks plausible\ncalculation incomplete"
        assert parse_agent_response(reasoning) == ("no", reasoning)


def test_gpt_parser_keeps_existing_case_sensitive_first_match_behavior():
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "gpt"}, clear=False):
        assert parse_agent_response("{no||first} then {yes}") == ("no", "first")
        assert parse_agent_response("{YES}") == ("no", "{YES}")


def test_gpt_leader_and_robot_start_prompts_remain_exactly_unchanged():
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "gpt"}, clear=False):
        leader = create_leader_prompt("resume")
        robot = create_robot_start_message("task", "scene")

    assert hashlib.sha256(leader.encode()).hexdigest() == (
        "f9db97b5e3c99340501ab52f4fd273fef5c06b147b888307422a31817bc749d7"
    )
    assert hashlib.sha256(robot.encode()).hexdigest() == (
        "91561a1c94ecd933e174e461ddfda672317b73c7d719886db6c8e97ca8d8a15f"
    )


def test_qwen_robot_reflection_forbids_invented_detection_geometry():
    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "qwen"}, clear=False):
        prompt = create_robot_start_message(
            "Detect any_targets|0",
            "scene",
            goal_objects=("any_targets|0",),
        )

    assert "Do not apply manipulation workspace" in prompt
    assert "Do not invent" in prompt
    assert "nearest navigable" in prompt
    assert "any_targets|0" in prompt


def test_qwen_reflection_retries_unsupported_geometry_and_accepts_correction():
    robot = Mock()
    robot.chat.side_effect = [
        "{{no||Assuming vertical FOV is plus or minus 45 degrees, the target is outside range.}}",
        "{{yes}}",
    ]

    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "qwen"}, clear=False):
        response, verdict = request_qwen_reflection(
            robot, "initial reflection", ("any_targets|0",)
        )

    assert response == "{{yes}}"
    assert verdict == ("yes", None)
    assert robot.chat.call_count == 2
    correction = robot.chat.call_args_list[1].args[0]
    assert "unsupported detection geometry" in correction


def test_qwen_reflection_retries_non_protocol_response():
    robot = Mock()
    robot.chat.side_effect = ["not protocol response", "{{yes}}"]

    with patch.dict(os.environ, {"EMOS_LLM_BACKEND": "qwen"}, clear=False):
        response, verdict = request_qwen_reflection(
            robot, "initial reflection", ("any_targets|0",)
        )

    assert response == "{{yes}}"
    assert verdict == ("yes", None)
    assert "not protocol response" in robot.chat.call_args_list[1].args[0]
