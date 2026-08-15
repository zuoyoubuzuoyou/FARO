from types import SimpleNamespace
from unittest.mock import Mock

import torch

from habitat_baselines.rl.hrl.hl.llm_policy import (
    ACTION_POOL,
    LLMHighLevelPolicy,
    get_llm_actions,
    get_true_detected_objects,
)


def test_gpt_action_pool_keeps_existing_identity_and_contents():
    selected = get_llm_actions(ACTION_POOL, {"wait": 0}, backend="gpt")

    assert selected is ACTION_POOL


def test_qwen_action_pool_matches_each_agents_real_skills():
    spot = get_llm_actions(
        ACTION_POOL,
        {"wait": 0, "nav_to_obj": 1},
        backend="qwen",
    )
    drone = get_llm_actions(ACTION_POOL, {"wait": 0}, backend="qwen")

    assert [action.name for action in spot] == [
        "send_request",
        "nav_to_obj",
        "wait",
    ]
    assert [action.name for action in drone] == ["send_request", "wait"]
    assert {"pick", "place", "reset_arm"}.isdisjoint(
        {action.name for action in spot + drone}
    )


def test_qwen_forbidden_action_is_rejected_before_argument_parsing(monkeypatch):
    monkeypatch.setenv("EMOS_LLM_BACKEND", "qwen")
    policy = object.__new__(LLMHighLevelPolicy)
    policy._skill_name_to_idx = {"wait": 7}
    policy._qwen_predicates_list = []
    policy.llm_agent = Mock(
        initialized=True,
        start_act=True,
        name="agent_0",
    )
    policy.llm_agent.chat.return_value = {"name": "place", "arguments": {}}
    policy.llm_agent.get_token_usage.return_value = 0
    policy._parse_function_call_args = Mock(side_effect=AssertionError("must not parse"))

    next_skill, skill_args, _, _ = policy.get_next_skill(
        observations={"all_predicates": torch.empty((1, 0))},
        rnn_hidden_states=None,
        prev_actions=None,
        masks=torch.ones(1),
        plan_masks=torch.ones(1),
        deterministic=True,
        log_info=[],
        envs_text_context=[{"scene_description": "test scene"}],
        agent_arguments=object(),
    )

    assert next_skill.tolist() == [7.0]
    assert skill_args == [["500"]]
    policy._parse_function_call_args.assert_not_called()


def test_qwen_detected_objects_come_from_true_pddl_predicates():
    object_0 = SimpleNamespace(name="any_targets|0")
    object_1 = SimpleNamespace(name="any_targets|1")
    predicates = [
        SimpleNamespace(name="is_detected", _arg_values=[object_0]),
        SimpleNamespace(name="is_detected", _arg_values=[object_1]),
        SimpleNamespace(name="robot_at", _arg_values=[object_0]),
    ]

    assert get_true_detected_objects(predicates, torch.tensor([1.0, 0.0, 1.0])) == (
        "any_targets|0",
    )
