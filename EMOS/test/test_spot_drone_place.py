"""Check that the Spot/Drone config supplies observations needed by place."""

from types import SimpleNamespace

import numpy as np
import unittest
import torch

from habitat.core.registry import registry
from habitat.core.simulator import SensorSuite
from habitat.tasks.rearrange.rearrange_sensors import (
    ObjectToGoalDistance,
    ObjectToGoalDistanceSensor,
)
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.hrl.skills.oracle_arm_policies import OraclePlacePolicy
from habitat_baselines.rl.multi_agent.utils import update_dict_with_agent_prefix
from habitat_baselines.utils.common import batch_obs


def place_action_at_distance(distance):
    config = get_config("multi_rearrange/llm_spot_drone_per.yaml")
    # Only native simulation is replaced; measurement, sensor, batching and
    # place action generation all use the real implementations.
    sim = SimpleNamespace(
        get_targets=lambda: (np.array([0]), np.zeros((1, 3))),
        get_scene_pos=lambda: np.array([[distance, 0.0, 0.0]]),
        agents_mgr=SimpleNamespace(agent_names=["agent_0", "agent_1"]),
    )
    measures = {
        name: registry.get_measure(settings.type)(sim=sim, config=settings)
        for name, settings in config.habitat.task.measurements.items()
        if settings.type == ObjectToGoalDistance.__name__
    }
    task = SimpleNamespace(
        _sim=sim, targ_idx=0, measurements=SimpleNamespace(measures=measures)
    )
    sensors = SensorSuite([
        registry.get_sensor(settings.type)(sim=sim, config=settings)
        for settings in config.habitat.task.lab_sensors.values()
        if settings.type == ObjectToGoalDistanceSensor.__name__
    ])
    RearrangeTask._duplicate_sensor_suite(task, sensors)
    observations = batch_obs([
        sensors.get_observations(observations={}, episode=None, task=task)
    ])
    observations = update_dict_with_agent_prefix(observations, 0)

    policy = OraclePlacePolicy.__new__(OraclePlacePolicy)
    policy._full_ac_size = 3
    policy._place_srt_idx = 0
    policy._place_end_idx = 3
    policy._cur_skill_args = {
        0: OraclePlacePolicy.OraclePlaceActionArgs(4, 0)
    }
    action = policy._internal_act(
        observations, None, None, torch.ones((1, 1)), [0]
    )

    return action.actions.tolist()


class SpotDronePlaceTest(unittest.TestCase):
    def test_place_keeps_holding_until_close_to_goal(self):
        self.assertEqual(place_action_at_distance(0.5), [[4.0, 2.0, 0.0]])

    def test_place_releases_close_to_goal(self):
        self.assertEqual(place_action_at_distance(0.01), [[4.0, 2.0, -1.0]])


if __name__ == "__main__":
    unittest.main()
