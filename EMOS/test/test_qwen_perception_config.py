from omegaconf import OmegaConf

from habitat_baselines.config.default import get_config


BASE_CONFIG = "multi_rearrange/llm_spot_drone_per.yaml"
QWEN_CONFIG = "multi_rearrange/llm_spot_drone_per_qwen.yaml"


def test_qwen_overlay_removes_only_spot_manipulation_skills():
    base = get_config(BASE_CONFIG)
    qwen = get_config(QWEN_CONFIG)

    base_spot = base.habitat_baselines.rl.policy.agent_0.hierarchical_policy
    qwen_spot = qwen.habitat_baselines.rl.policy.agent_0.hierarchical_policy
    excluded_manipulation = {
        "pick",
        "pick_at_position",
        "place",
        "place_at_position",
        "reset_arm",
    }

    assert "pick" in base_spot.high_level_policy.allowed_actions
    assert "place" in base_spot.high_level_policy.allowed_actions
    assert excluded_manipulation.isdisjoint(set(base_spot.ignore_skills))
    assert excluded_manipulation.issubset(set(qwen_spot.ignore_skills))

    base_drone = OmegaConf.to_container(
        base.habitat_baselines.rl.policy.agent_1, resolve=True
    )
    qwen_drone = OmegaConf.to_container(
        qwen.habitat_baselines.rl.policy.agent_1, resolve=True
    )
    assert qwen_drone == base_drone
