import os
from unittest.mock import patch

from habitat_baselines.rl.multi_agent.multi_llm_policy import (
    create_robot_prompt,
    parse_agent_response,
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
