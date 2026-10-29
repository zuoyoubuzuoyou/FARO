"""Runtime evidence uses only the observation already returned by env.step."""

import json
from collections import UserList
from types import SimpleNamespace
import unittest

import test_trajectory_state


class _Transform:
    def __init__(self, origin):
        self.origin = origin

    def inverted(self):
        return self

    def transform_point(self, point):
        return [float(value) - origin for value, origin in zip(point, self.origin)]


def _forbidden(*args, **kwargs):
    _forbidden.calls += 1
    raise AssertionError("logging must not evaluate sensors, predicates, or controls")


_forbidden.calls = 0


class RuntimeObservationTest(unittest.TestCase):
    def tearDown(self):
        self.assertEqual(_forbidden.calls, 0, "observer executed a live query")

    def make_env(self):
        _forbidden.calls = 0
        env = test_trajectory_state.TrajectoryStateTest().make_env()
        env.set_trajectory_recording(True)
        sim = env._env.sim
        sim.get_targets = lambda: ([2], [[4, 1, 0]])
        sim.get_scene_pos = lambda: [[99, 0, 0], [88, 0, 0], [2, 1, 0]]
        sim.scene_obj_ids = [7, 8, 42]
        sim.habitat_config = SimpleNamespace(object_ids_start=100)
        sim.get_rigid_object_manager = lambda: SimpleNamespace(
            get_object_by_id=lambda object_id: SimpleNamespace(handle="mug-handle")
        )
        sim.handle_to_object_id = {"mug-handle": 2, "any_targets|7": 2}
        sim.get_sensor_observations = _forbidden
        sim.sensor_suite = SimpleNamespace(get_observations=_forbidden)
        # Native root is y=1, even though robot ground/base_pos is unrelated.
        sim.agents_mgr[0].articulated_agent.base_transformation = _Transform([0, 1, 0])
        sim.agents_mgr[1].articulated_agent.base_transformation = _Transform([10, 1, 0])
        sim_info = SimpleNamespace(
            robot_at_thresh=2.0,
            obj_ids={"any_targets|7": 2},
            target_ids={"TARGET_any_targets|7": 2},
            all_entities={"any_targets|7": object(), "TARGET_any_targets|7": object()},
            pred_truth_cache={"is_detected(any_targets|7)": True},
            get_entity_pos=_forbidden,
        )
        env._env.task = SimpleNamespace(
            pddl_problem=SimpleNamespace(_sim_info=sim_info),
            sensor_suite=SimpleNamespace(
                sensors={
                    "agent_1_detected_objects": SimpleNamespace(agent_id=0, pixel_threshold=2),
                    "agent_0_detected_objects": SimpleNamespace(agent_id=1, pixel_threshold=2),
                },
                get_observations=_forbidden,
            ),
            actions={},
        )
        return env

    def observations(self, *, visible=True):
        return {
            "agent_1_detected_objects": [142] if visible else [],
            "agent_0_detected_objects": [],
            "agent_1_head_semantic": [[142, 142, 142] if visible else [0, 0, 0]],
            "agent_0_head_semantic": [[142, 142, 0]],
        }

    def test_native_reference_inclusive_boundary_and_real_entity_mapping(self):
        env = self.make_env()
        state = env.get_info(self.observations())["__trajectory_state"]
        self.assertIsNotNone(state)
        self.assertIn("current_goal", state)
        self.assertEqual(state["current_goal"]["any_at"], {
            "any_targets|7": True, "TARGET_any_targets|7": False,
        })
        coverage = state["agents"]["agent_1"]["goal_coverage"]
        self.assertEqual(coverage["reference"], "base_transformation")
        self.assertEqual(coverage["distance_threshold"], 2.0)
        self.assertEqual(coverage["targets"]["any_targets|7"], {"distance": 2.0, "at_goal": True})
        self.assertEqual(state["target_objects"][0]["goal_name"], "any_targets|7")
        self.assertEqual(state["target_objects"][0]["semantic_id"], 142)
        self.assertEqual(state["current_goal"]["env_step"], 4)
        self.assertEqual(state["current_goal"]["timing"], "post_env_step")
        json.dumps(state, allow_nan=False)

    def test_current_detection_loss_ignores_latched_predicate_cache(self):
        env = self.make_env()
        first = env.get_info(self.observations())["__trajectory_state"]
        lost = env.get_info(self.observations(visible=False))["__trajectory_state"]
        self.assertIn("current_goal", first)
        self.assertTrue(first["current_goal"]["is_detected"]["any_targets|7"])
        self.assertFalse(lost["current_goal"]["is_detected"]["any_targets|7"])
        self.assertTrue(env._env.task.pddl_problem._sim_info.pred_truth_cache["is_detected(any_targets|7)"])
        detection = first["agents"]["agent_0"]["detection"]
        self.assertEqual(detection["target_pixel_counts"]["any_targets|7"], {"agent_0_head_semantic": 2})
        self.assertFalse(detection["target_detections"]["any_targets|7"])
        self.assertEqual(detection["pixel_threshold"], 2)
        self.assertEqual(detection["detected_object_ids"], [])
        self.assertEqual(detection["env_step"], 4)

    def test_semantic_count_uses_strict_threshold_per_camera(self):
        env = self.make_env()
        observations = self.observations(visible=False)
        del observations["agent_1_detected_objects"]
        observations["agent_1_head_semantic"] = [[142, 142]]
        observations["agent_1_arm_semantic"] = [[142]]
        state = env.get_info(observations)["__trajectory_state"]
        self.assertIn("current_goal", state)
        self.assertFalse(state["current_goal"]["is_detected"]["any_targets|7"])
        observations["agent_1_head_semantic"] = [[142, 142, 142]]
        state = env.get_info(observations)["__trajectory_state"]
        self.assertTrue(state["current_goal"]["is_detected"]["any_targets|7"])

    def test_missing_observations_mapping_and_transform_are_unknown_with_reasons(self):
        env = self.make_env()
        unknown = env.get_trajectory_state()
        self.assertIn("current_goal", unknown)
        self.assertIsNone(unknown["current_goal"]["is_detected"]["any_targets|7"])
        self.assertTrue(unknown["agents"]["agent_1"]["detection"]["reason"])
        for data in env._env.sim.agents_mgr:
            del data.articulated_agent.base_transformation
        del env._env.sim.habitat_config.object_ids_start
        state = env.get_info(self.observations())["__trajectory_state"]
        self.assertIn("current_goal", state)
        self.assertIsNone(state["current_goal"]["any_at"]["any_targets|7"])
        self.assertIn("current_goal", state)
        self.assertIsNone(state["current_goal"]["is_detected"]["any_targets|7"])
        self.assertTrue(state["agents"]["agent_1"]["goal_coverage"]["targets"]["any_targets|7"]["reason"])
        self.assertTrue(state["agents"]["agent_1"]["detection"]["reason"])

    def test_returned_ids_work_without_images_and_missing_teammate_is_unknown(self):
        env = self.make_env()
        observations = {"agent_1_detected_objects": []}
        state = env.get_info(observations)["__trajectory_state"]
        self.assertIn("current_goal", state)
        self.assertIsNone(state["current_goal"]["is_detected"]["any_targets|7"])
        observations["agent_1_detected_objects"] = [142]
        state = env.get_info(observations)["__trajectory_state"]
        self.assertTrue(state["current_goal"]["is_detected"]["any_targets|7"])
        self.assertIsNone(state["agents"]["agent_0"]["detection"]["target_detections"]["any_targets|7"])

    def test_fresh_navigation_requires_selected_action_and_detaches_commands(self):
        env = self.make_env()
        diagnostic = {"env_step": 4, "timing": "pre_control", "active": True,
                      "available": True, "distance": 2.5,
                      "controller_command": [1.0, 0.0], "skill_done": False}
        env._env.task.actions["agent_1_oracle_nav_action"] = SimpleNamespace(_navigation_diagnostics=diagnostic)
        env._trajectory_executed_action = {"action": ["agent_1_oracle_nav_action"]}
        state = env.get_info(self.observations())["__trajectory_state"]
        self.assertIn("navigation", state["agents"]["agent_1"])
        self.assertTrue(state["agents"]["agent_1"]["navigation"]["active"])
        diagnostic["controller_command"][0] = 99
        self.assertEqual(state["agents"]["agent_1"]["navigation"]["controller_command"], [1.0, 0.0])
        diagnostic["env_step"] = 3
        stale = env.get_info(self.observations())["__trajectory_state"]["agents"]["agent_1"]["navigation"]
        self.assertFalse(stale["active"])
        self.assertIsNone(stale["controller_command"])
        self.assertEqual(stale["distance"], 2.5)
        diagnostic["env_step"] = 4
        env._trajectory_executed_action = {"action": ["agent_1_base_velocity"]}
        inactive = env.get_info(self.observations())["__trajectory_state"]["agents"]["agent_1"]["navigation"]
        self.assertFalse(inactive["active"])
        self.assertTrue(inactive["reason"])


    def test_initial_snapshot_is_unknown_even_with_readable_geometry(self):
        state = self.make_env().get_trajectory_state()
        self.assertIsNone(state["current_goal"]["any_at"]["any_targets|7"])
        self.assertTrue(state["current_goal"]["reasons"]["any_at"]["any_targets|7"])

    def test_root_and_camera_config_are_detached_without_rendering(self):
        env = self.make_env()
        robot = env._env.sim.agents_mgr[0].articulated_agent
        robot.base_transformation.translation = [0, 1, 0]
        camera = SimpleNamespace(type="HabitatSimSemanticSensor", height=20, width=30,
                                 hfov=90, position=UserList([0.1, 1, 0]), orientation=[0, 0, 0])
        env.config.simulator.agents["agent_1"].sim_sensors = {"head_semantic": camera}
        state = env.get_info(self.observations())["__trajectory_state"]
        agent = state["agents"]["agent_1"]
        self.assertEqual(agent["root_position"], [0, 1, 0])
        self.assertIn("camera_sensors", agent)
        self.assertEqual(agent["camera_sensors"]["head_semantic"]["width"], 30)
        camera.position[0] = 99
        self.assertEqual(agent["camera_sensors"]["head_semantic"]["position"], [0.1, 1, 0])

    def test_selected_zero_command_cannot_revive_old_navigation(self):
        env = self.make_env()
        env._env.task.actions["agent_1_oracle_nav_action"] = SimpleNamespace(
            _navigation_diagnostics={"env_step": 4, "timing": "pre_control", "active": True,
                                     "available": True, "distance_to_goal": 2.5,
                                     "control": [1.0, 0.0], "skill_done": False})
        env._trajectory_executed_action = {
            "action": "agent_1_oracle_nav_action",
            "action_args": {"agent_1_oracle_nav_action": [0]},
        }
        nav = env.get_info(self.observations())["__trajectory_state"]["agents"]["agent_1"]["navigation"]
        self.assertFalse(nav["active"])
        self.assertIsNone(nav["control"])
        self.assertEqual(nav["distance_to_goal"], 2.5)

    def test_missing_entity_alias_mapping_never_invents_current_truth(self):
        env = self.make_env()
        env._env.task.pddl_problem._sim_info.obj_ids.clear()
        env._env.task.pddl_problem._sim_info.target_ids.clear()
        env._env.sim.handle_to_object_id.clear()
        state = env.get_info(self.observations())["__trajectory_state"]
        self.assertIsNone(state["current_goal"]["any_at"]["any_targets|0"])
        self.assertIsNone(state["current_goal"]["is_detected"]["any_targets|0"])
        self.assertIsNone(state["target_objects"][0]["semantic_id"])

    def test_pddl_goal_mapping_supplies_alias_when_object_alias_is_unavailable(self):
        env = self.make_env()
        env._env.task.pddl_problem._sim_info.obj_ids.clear()
        env._env.sim.handle_to_object_id.clear()
        state = env.get_info(self.observations())["__trajectory_state"]
        self.assertEqual(state["target_objects"][0]["goal_name"], "any_targets|7")
        self.assertEqual(state["target_objects"][0]["semantic_id"], 142)
        self.assertTrue(state["current_goal"]["any_at"]["any_targets|7"])

    def test_nonfinite_observation_metadata_stays_json_safe(self):
        env = self.make_env()
        env._env.sim.get_scene_pos = lambda: [[0, 0, 0], [0, 0, 0], [float("nan"), 1, 0]]
        state = env.get_info(self.observations())["__trajectory_state"]
        self.assertIsNotNone(state)
        self.assertIsNone(state["target_objects"][0]["position"][0])
        json.dumps(state, allow_nan=False)
        self.assertIsNone(state["current_goal"]["any_at"]["any_targets|7"])


if __name__ == "__main__":
    unittest.main()
