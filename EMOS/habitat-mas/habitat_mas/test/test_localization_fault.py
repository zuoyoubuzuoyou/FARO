import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.actions.oracle_nav_action import OracleNavAction
from habitat.tasks.rearrange.actions.habitat_mas_actions import OracleNavDiffBaseAction
from habitat_mas.scene_graph.utils import generate_objects_description


class LocalizationFaultTest(unittest.TestCase):
    def setUp(self):
        self.entity = SimpleNamespace(name="any_targets|0")
        self.true_position = np.array([2., 1., 3.])
        self.task = RearrangeTask.__new__(RearrangeTask)
        self.task._episode_id = "15"
        self.task._sim = SimpleNamespace(
            agents_mgr=SimpleNamespace(agent_names=["agent_0", "agent_1"]),
            ep_info=SimpleNamespace(scene_id="scene"),
        )
        self.task.pddl_problem = SimpleNamespace(
            get_entity=lambda name: self.entity if name == self.entity.name else None,
            sim_info=SimpleNamespace(get_entity_pos=lambda entity: self.true_position),
        )
        self.action = SimpleNamespace(_agent_index=1, _targets={0: "cached"})
        self.peer = SimpleNamespace(_agent_index=0, _targets={0: "peer_cached"})
        self.task.actions = {"agent_1_oracle_nav_action": self.action, "agent_0_oracle_nav_action": self.peer}
        self.task.get_task_text_context = Mock(return_value={"scene_description": "biased scene"})

    def test_estimate_is_agent_specific_and_invalidates_cache_on_entry_expiry(self):
        self.task.set_localization_fault("agent_1", "any_targets|0", [1, .5, -2], "pf", 4)
        self.assertEqual(self.action._targets, {})
        self.assertEqual(self.peer._targets, {0: "peer_cached"})
        np.testing.assert_allclose(self.task.get_navigation_entity_position(1, self.entity), [3, 1.5, 1])
        np.testing.assert_allclose(self.task.get_navigation_entity_position(0, self.entity), [2, 1, 3])
        np.testing.assert_allclose(self.true_position, [2, 1, 3])
        self.assertEqual(self.task._localization_events[0]["type"], "fault_applied")
        self.assertEqual(self.task._localization_events[0]["simulator_step"], 4)
        self.action._targets[0] = "biased_cached"
        self.task.set_localization_fault("agent_1", "any_targets|0", [1, .5, -2], "pf", 5)
        self.assertEqual(self.action._targets[0], "biased_cached")
        self.task.set_localization_fault("agent_1", "any_targets|0", None, "pf", 6)
        self.assertEqual(self.action._targets, {})
        np.testing.assert_allclose(self.task.get_navigation_entity_position(1, self.entity), [2, 1, 3])

    def test_oracle_reads_task_estimate_instead_of_direct_truth(self):
        self.check_oracle_estimate(OracleNavAction)

    def test_native_diff_base_oracle_reads_estimate_and_restores_truth(self):
        self.check_oracle_estimate(OracleNavDiffBaseAction)

    def check_oracle_estimate(self, action_type):
        import habitat.tasks.rearrange.actions.oracle_nav_action as module
        from unittest.mock import patch
        action = action_type.__new__(action_type)
        action._targets, action._poss_entities = {}, [self.entity]
        action._task, action._agent_index = self.task, 1
        action._sim = SimpleNamespace(agents_mgr=[None, SimpleNamespace(articulated_agent=SimpleNamespace())])
        action._config = SimpleNamespace(spawn_max_dist_to_obj=1, num_spawn_attempts=1)
        action.motion_type = "base_velocity"
        self.task.set_localization_fault("agent_1", "any_targets|0", [1, 0, -2], "pf", 4)
        with patch.object(module, "place_agent_at_dist_from_pos", side_effect=lambda pos, *args: (pos, 0, False)):
            with patch("habitat.tasks.rearrange.actions.habitat_mas_actions.place_agent_at_dist_from_pos",
                       side_effect=lambda pos, *args: (pos, 0, False)):
                _, looked_at = action._get_target_for_idx(0)
        np.testing.assert_allclose(looked_at, [3, 1, 1])
        self.task.actions["agent_1_oracle_nav_action"] = action
        self.task.set_localization_fault("agent_1", "any_targets|0", None, "pf", 9)
        with patch.object(module, "place_agent_at_dist_from_pos", side_effect=lambda pos, *args: (pos, 0, False)):
            # The subclass imports its own placement function.
            with patch("habitat.tasks.rearrange.actions.habitat_mas_actions.place_agent_at_dist_from_pos",
                       side_effect=lambda pos, *args: (pos, 0, False)):
                _, looked_at = action._get_target_for_idx(0)
        np.testing.assert_allclose(looked_at, [2, 1, 3])
        self.assertEqual(self.task._localization_events[-1]["type"], "localization_target_restored")

    def test_expiry_keeps_explicit_true_position_for_later_model_decisions(self):
        self.task.get_task_text_context = Mock(side_effect=lambda **values: {"scene_description": str(values.get("position_estimates"))})
        self.task.set_localization_fault("agent_1", "any_targets|0", [1, 0, -2], "pf", 4)
        restored = self.task.set_localization_fault("agent_1", "any_targets|0", None, "pf", 6)
        self.task.get_task_text_context.assert_called_with(agent="agent_1", position_estimates={"any_targets|0": [2., 1., 3.]})
        self.assertIn("[2.0, 1.0, 3.0]", restored["scene_description"])
        self.assertEqual(restored, self.task.set_localization_fault("agent_1", "any_targets|0", None, "pf", 7))

    def test_real_step_reopens_completed_navigation_on_localization_boundaries(self):
        action = OracleNavDiffBaseAction.__new__(OracleNavDiffBaseAction)
        action._task, action._agent_index, action._multi_agent = self.task, 1, True
        action.ep_id = "15"
        action._sim = SimpleNamespace(ep_info=SimpleNamespace(episode_id="15"))
        action._poss_entities, action._targets = [self.entity], {0: "cached"}
        action.prev_nav_done, action.prev_match_target_id = True, 0
        self.task.actions["agent_1_oracle_nav_action"] = action
        self.peer.prev_nav_done = True
        reads = []

        class ReadCompleted(Exception):
            pass

        def read_target(index):
            reads.append(self.task.get_navigation_entity_position(1, self.entity))
            raise ReadCompleted()  # Stop before the unrelated physical controller.

        action._get_target_for_idx = read_target
        action.step(agent_1_oracle_nav_action=np.array([1]))
        self.assertEqual(reads, [])
        self.task.set_localization_fault("agent_1", "any_targets|0", [1, 0, -2], "pf", 4)
        with self.assertRaises(ReadCompleted):
            action.step(agent_1_oracle_nav_action=np.array([1]))
        np.testing.assert_allclose(reads[-1], [3, 1, 1])
        action.prev_nav_done, action.prev_match_target_id = True, 0
        self.task.set_localization_fault("agent_1", "any_targets|0", None, "pf", 6)
        with self.assertRaises(ReadCompleted):
            action.step(agent_1_oracle_nav_action=np.array([1]))
        np.testing.assert_allclose(reads[-1], [2, 1, 3])
        self.assertTrue(self.peer.prev_nav_done)

    def test_scene_height_and_distance_use_same_estimate(self):
        obj = SimpleNamespace(full_name="cup", label="cup", center=np.array([2., 1., 3.]))
        sim = SimpleNamespace(_handle_to_goal_name={"cup": "any_targets|0"},
                              pathfinder=SimpleNamespace(num_islands=1, island_area=lambda **kwargs: 1,
                                  snap_point=lambda pos, island: np.array([pos[0], 0., pos[2]])))
        layer = SimpleNamespace(obj_ids=[0], obj_dict={0: obj})
        text = generate_objects_description(sim, layer, position_estimates={"any_targets|0": [3, 2, 1]})
        self.assertIn('height of "any_targets|0" is 2.0', text)
        self.assertIn('[3.000, 2.000, 1.000]', text)
        np.testing.assert_allclose(obj.center, [2, 1, 3])


if __name__ == "__main__":
    unittest.main()
