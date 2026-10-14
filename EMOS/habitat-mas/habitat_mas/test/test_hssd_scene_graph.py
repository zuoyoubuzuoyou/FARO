"""Goal descriptions must survive furniture restoration between episodes."""

from types import SimpleNamespace
import unittest

import numpy as np
import habitat

from habitat_mas.scene_graph.scene_graph_hssd import SceneGraphHSSD


class HSSDGoalReceptacleTest(unittest.TestCase):
    def scene_graph(self, handles, goals, mapping=None):
        objects = {
            handle: SimpleNamespace(object_id=object_id)
            for handle, object_id in handles.items()
        }
        manager = SimpleNamespace(get_object_by_handle=objects.get)
        pathfinder = SimpleNamespace(
            get_bounds=lambda: np.array([[0., 0., 0.], [4., 2., 4.]]),
            build_navmesh_vertices=lambda: np.array(
                [[0., 0., 0.], [0., 0., 4.], [4., 0., 0.]]
            ),
            build_navmesh_vertex_indices=lambda: [0, 1, 2],
        )
        poses = [
            SimpleNamespace(translation=np.array([i + 1., 1., 2.]),
                            rotation=np.eye(3))
            for i in range(len(goals))
        ]
        sim = SimpleNamespace(
            pathfinder=pathfinder,
            get_agent=lambda index: SimpleNamespace(
                state=SimpleNamespace(position=np.zeros(3))
            ),
            get_rigid_object_manager=lambda: manager,
            handle_to_object_id={},
            _get_target_trans=lambda: list(enumerate(poses)),
            ep_info=SimpleNamespace(
                episode_id="106", scene_id="hssd_scene",
                goal_receptacles=[(handle, None, None) for handle in goals],
            ),
            habitat_config=SimpleNamespace(object_ids_start=100),
            agents_mgr=SimpleNamespace(agents_order=[]),
        )
        if mapping is not None:
            sim.prev_recep = {"handle": mapping}
        graph = SceneGraphHSSD()
        graph.load_gt_scene_graph(sim)
        return graph

    def test_restored_furniture_keeps_episode_goal_name_and_pose(self):
        graph = self.scene_graph(
            {"cabinet_:0002": 12}, ["cabinet_:0000"],
            {"cabinet_:0000": "cabinet_:0002"},
        )
        node = graph.object_layer.obj_dict[112]
        self.assertEqual(node.full_name, "cabinet_:0000")
        self.assertEqual(node.label, "TARGET_any_targets|0")
        np.testing.assert_array_equal(node.center, [1., 1., 2.])

    def test_mapping_selects_correct_instance_when_old_handle_is_reused(self):
        graph = self.scene_graph(
            {"cabinet_:0000": 8, "cabinet_:0002": 12},
            ["cabinet_:0000"], {"cabinet_:0000": "cabinet_:0002"},
        )
        self.assertEqual(graph.object_layer.obj_ids, [112])

    def test_same_template_instances_remain_distinct_goals(self):
        graph = self.scene_graph(
            {"cabinet_:0001": 11, "cabinet_:0002": 12},
            ["cabinet_:0000", "cabinet_:0001"],
            {"cabinet_:0000": "cabinet_:0002"},
        )
        first, second = (graph.object_layer.obj_dict[i] for i in [112, 111])
        self.assertEqual(first.label, "TARGET_any_targets|0")
        self.assertEqual(second.label, "TARGET_any_targets|1")
        self.assertEqual(second.full_name, "cabinet_:0001")
        np.testing.assert_array_equal(second.center, [2., 1., 2.])

    def test_existing_goal_without_restoration_mapping_still_loads(self):
        graph = self.scene_graph({"cabinet_:0000": 8}, ["cabinet_:0000"])
        self.assertEqual(graph.object_layer.obj_dict[108].full_name,
                         "cabinet_:0000")

    def test_missing_restored_instance_reports_episode_and_both_handles(self):
        with self.assertRaisesRegex(
            ValueError, r"106.*cabinet_:0000.*cabinet_:0002"
        ):
            self.scene_graph(
                {}, ["cabinet_:0000"],
                {"cabinet_:0000": "cabinet_:0002"},
            )


if __name__ == "__main__":
    unittest.main()
