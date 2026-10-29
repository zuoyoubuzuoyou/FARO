"""Scene descriptions preserve the target IDs used by task assignments."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "habitat-mas/habitat_mas/scene_graph/utils.py"


class SceneObjectDescriptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("scene_graph_utils_test", SOURCE)
        cls.utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.utils)

    def describe(self, objects, goal_names):
        pathfinder = SimpleNamespace(
            num_islands=1,
            island_area=lambda island_index: 1.0,
            snap_point=lambda point, island: np.asarray(point) + [0.3, 0.0, 0.4],
        )
        sim = SimpleNamespace(_handle_to_goal_name=goal_names, pathfinder=pathfinder)
        layer = SimpleNamespace(obj_ids=list(objects), obj_dict=objects)
        return self.utils.generate_objects_description(sim, layer)

    def test_target_ids_keep_their_heights_when_scene_order_differs(self):
        objects = {
            235: SimpleNamespace(
                full_name="017_orange_:0000", label=None, center=[0.0, 0.572, 0.0]
            ),
            236: SimpleNamespace(
                full_name="050_medium_clamp_:0000", label=None, center=[0.0, 1.396, 0.0]
            ),
        }
        description = self.describe(objects, {
            "050_medium_clamp_:0000": "any_targets|0",
            "017_orange_:0000": "any_targets|1",
        })

        self.assertIn('The height of "any_targets|0" is 1.4.', description)
        self.assertIn('The height of "any_targets|1" is 0.6.', description)
        self.assertIn(
            'The horizontal distance of "any_targets|0" to the nearest navigable point is 0.5.',
            description,
        )
        self.assertNotIn("017_orange_:0000", description)
        self.assertNotIn("050_medium_clamp_:0000", description)

    def test_description_uses_the_highest_priority_available_name(self):
        cases = [
            ("asset", "other_label", {"asset": "TARGET_any_targets|0"}, "TARGET_any_targets|0"),
            ("asset", "any_targets|1", {}, "any_targets|1"),
            (None, "any_targets|1", {}, "any_targets|1"),
            ("asset", None, {}, "asset"),
            (None, None, {}, "any_targets|42"),
        ]
        for full_name, label, goal_names, expected_name in cases:
            with self.subTest(full_name=full_name, label=label, goal_names=goal_names):
                description = self.describe({42: SimpleNamespace(
                    full_name=full_name, label=label, center=[0.0, 1.4, 0.0]
                )}, goal_names)
                self.assertIn(f'The height of "{expected_name}" is 1.4.', description)


if __name__ == "__main__":
    unittest.main()
