"""Exercise trajectory snapshots without requiring a simulator installation."""

import ast
import json
import importlib.util
import math
import pickle
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def _load_rl_task_env():
    """Compile the production class; replace only unavailable Habitat imports."""
    try:
        from habitat.core.environments import RLTaskEnv

        return RLTaskEnv
    except ImportError:
        pass

    source = (
        Path(__file__).resolve().parents[1]
        / "habitat-lab/habitat/core/environments.py"
    )
    parsed = ast.parse(source.read_text())
    env_class = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "RLTaskEnv"
    )

    class RLEnv:
        @property
        def config(self):
            return self._core_env_config

    habitat = SimpleNamespace(
        RLEnv=RLEnv,
        registry=SimpleNamespace(register_env=lambda **kwargs: lambda cls: cls),
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            env_class,
        ],
        type_ignores=[],
    )
    helper_path = source.with_name("trajectory_diagnostics.py")
    helper_spec = importlib.util.spec_from_file_location("trajectory_diagnostics", helper_path)
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    namespace = {"habitat": habitat, "collect_runtime_diagnostics": helper.collect_runtime_diagnostics}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["RLTaskEnv"]


RLTaskEnv = _load_rl_task_env()


class _Scalar:
    """A non-JSON-native numeric scalar, like the simulator's NumPy values."""

    def __init__(self, value):
        self.value = value

    def __float__(self):
        return float(self.value)

    def __int__(self):
        return int(self.value)


class _AgentData:
    def __init__(self, agent, grasped_id=None, grasp_managers=True):
        self.articulated_agent = agent
        self.grasp_mgrs = (
            [SimpleNamespace(snap_idx=grasped_id)] if grasp_managers else []
        )

    @property
    def grasp_mgr(self):
        if not self.grasp_mgrs:
            raise RuntimeError("Agent data has no grasp manager defined")
        return self.grasp_mgrs[0]


class _Robot:
    def __init__(self, position, rotation, ee_position=None):
        self.base_pos = [_Scalar(value) for value in position]
        self.sim_obj = SimpleNamespace(
            rotation=SimpleNamespace(
                vector=[_Scalar(value) for value in rotation[:3]],
                scalar=_Scalar(rotation[3]),
            )
        )
        self.params = SimpleNamespace(ee_links=[] if ee_position is None else [0])
        self._ee_position = ee_position

    @property
    def base_rot(self):
        # Match ArticulatedAgentBase: angle alone loses the quaternion axis.
        return 2 * math.acos(float(self.sim_obj.rotation.scalar))

    def ee_transform(self):
        if self._ee_position is None:
            raise ValueError("The current manipulator has no end effectors")
        return SimpleNamespace(
            translation=[_Scalar(value) for value in self._ee_position]
        )


class TrajectoryStateTest(unittest.TestCase):
    def make_env(self, *, done=False, success=False, past_limit=False):
        env = RLTaskEnv.__new__(RLTaskEnv)
        env._core_env_config = SimpleNamespace(
            simulator=SimpleNamespace(
                # Deliberately differs from configuration mapping order.
                agents_order=["agent_1", "agent_0"],
                agents={
                    "agent_0": SimpleNamespace(articulated_agent_type="SpotRobot"),
                    "agent_1": SimpleNamespace(articulated_agent_type="DJIDrone"),
                },
            )
        )
        env._end_on_success = True
        env._success_measure_name = "success"
        metrics = {"success": success, "distance": 2.0}
        env._env = SimpleNamespace(
            get_metrics=lambda: metrics,
            current_episode=SimpleNamespace(
                episode_id="episode-7", scene_id="scene/path.glb"
            ),
            _elapsed_steps=_Scalar(4),
            episode_over=done,
            _past_limit=lambda: past_limit,
        )
        env._env.sim = SimpleNamespace(
            agents_mgr=[
                _AgentData(_Robot([1, 2, 3], [0, 0.6, 0, 0.8], [1, 3, 4])),
                _AgentData(
                    _Robot([4, 5, 6], [0, -0.6, 0, 0.8], [4, 6, 7]),
                    _Scalar(42),
                ),
            ],
            get_agent_state=lambda index: self.fail(
                "Articulated trajectories must read robot bases, not cameras"
            ),
        )
        return env

    def test_default_info_preserves_metrics_without_reading_simulator(self):
        env = self.make_env()
        del env._env.sim
        self.assertIs(env.get_info(None), env._env.get_metrics())

    def test_snapshot_failure_keeps_environment_metrics_and_original_execution(self):
        env = self.make_env()
        env.set_trajectory_recording(True)
        with patch.object(env, "get_trajectory_state", side_effect=RuntimeError("unsupported pose getter")):
            info = env.get_info(None)
        self.assertIsNone(info["__trajectory_state"])
        self.assertEqual(info["__trajectory_state_error"], {
            "type": "RuntimeError", "message": "unsupported pose getter",
            "handling": "continue_without_snapshot",
        })
        self.assertEqual(env._env.get_metrics(), {"success": False, "distance": 2.0})

    def test_action_copy_failure_does_not_prevent_original_environment_step(self):
        class UnsupportedAction:
            def tolist(self):
                raise RuntimeError("detached copy unavailable")

        env = self.make_env()
        env.set_trajectory_recording(True)
        action = UnsupportedAction()
        with patch.object(RLTaskEnv.__mro__[1], "step", create=True,
                          return_value=(None, 2.0, False, {})) as parent_step:
            self.assertEqual(env.step(action), (None, 2.0, False, {}))
            parent_step.assert_called_once_with(action)
        state = env.get_trajectory_state()
        self.assertIsNone(state["executed_action"])
        self.assertIn("detached copy unavailable", state["missing_fields"]["executed_action"])

    def test_enabled_info_adds_detached_json_safe_robot_state(self):
        env = self.make_env()
        env.set_trajectory_recording(True)

        info = env.get_info(None)

        self.assertIsNot(info, env._env.get_metrics())
        self.assertNotIn("__trajectory_state", env._env.get_metrics())
        state = info["__trajectory_state"]
        self.assertEqual(
            {key: value for key, value in state.items() if key not in ("agents", "current_goal")},
            {
                "episode_id": "episode-7",
                "scene_id": "scene/path.glb",
                "num_steps": 4,
                "episode_over": False,
                "termination_reason": None,
                "executed_action": None,
                "target_objects": None,
                "task_progress": {"success": False, "distance": 2.0},
                "missing_fields": {
                    "executed_action": "no_recorded_step_since_reset",
                    "target_objects": "simulator_does_not_support_get_targets",
                },
            },
        )
        self.assertEqual(list(state["agents"]), ["agent_1", "agent_0"])
        self.assertEqual(
            {name: {key: value for key, value in agent.items()
                    if key not in ("goal_coverage", "detection", "root_position", "root_position_reason", "camera_sensors")}
             for name, agent in state["agents"].items()},
            {
                "agent_1": {
                    "robot_type": "DJIDrone",
                    "base_position": [1.0, 2.0, 3.0],
                    "base_rotation": [0.0, 0.6, 0.0, 0.8],
                    "ee_position": [1.0, 3.0, 4.0],
                    "grasped_object_id": None,
                },
                "agent_0": {
                    "robot_type": "SpotRobot",
                    "base_position": [4.0, 5.0, 6.0],
                    "base_rotation": [0.0, -0.6, 0.0, 0.8],
                    "ee_position": [4.0, 6.0, 7.0],
                    "grasped_object_id": 42,
                },
            },
        )
        self.assertEqual(json.loads(json.dumps(state, allow_nan=False)), state)
        self.assertEqual(pickle.loads(pickle.dumps(state)), state)
        env._env.sim.agents_mgr[0].articulated_agent.base_pos[0] = _Scalar(99)
        self.assertEqual(state["agents"]["agent_1"]["base_position"], [1, 2, 3])

    def test_targets_use_scene_indices_for_current_positions_and_object_ids(self):
        env = self.make_env()
        positions = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        env._env.sim.get_targets = lambda: ([2, 0], [[10, 11, 12], [13, 14, 15]])
        env._env.sim.get_scene_pos = lambda: positions
        env._env.sim.scene_obj_ids = [41, 42, 43]
        env._env.sim.get_rigid_object_manager = lambda: SimpleNamespace(
            get_object_by_id=lambda object_id: SimpleNamespace(handle=f"obj-{object_id}")
        )
        env._env.get_metrics()["visual"] = {"map": [[0] * 40] * 40}

        state = env.get_trajectory_state()

        self.assertEqual(state["target_objects"], [
            {"scene_index": 2, "object_id": 43, "handle": "obj-43",
             "goal_name": "any_targets|0", "semantic_id": None,
             "position": [7.0, 8.0, 9.0], "goal_position": [10.0, 11.0, 12.0]},
            {"scene_index": 0, "object_id": 41, "handle": "obj-41",
             "goal_name": "any_targets|1", "semantic_id": None,
             "position": [1.0, 2.0, 3.0], "goal_position": [13.0, 14.0, 15.0]},
        ])
        self.assertNotIn("visual", state["task_progress"])
        positions[2][0] = 99
        self.assertEqual(state["target_objects"][0]["position"], [7, 8, 9])

    def test_target_getter_errors_are_explicit_and_do_not_break_snapshot(self):
        env = self.make_env()
        env._env.sim.get_targets = lambda: ([1], [[7, 8, 9]])
        env._env.sim.scene_obj_ids = [3, 4]

        state = env.get_trajectory_state()

        self.assertIsNone(state["target_objects"][0]["position"])
        self.assertIsNone(state["target_objects"][0]["handle"])
        self.assertIn("target_objects.position", state["missing_fields"])
        self.assertIn("target_objects.handle", state["missing_fields"])
        self.assertEqual(json.loads(json.dumps(state)), state)

    def test_step_captures_converted_action_only_while_recording(self):
        env = self.make_env()
        action = {"action": ("agent_0_base_velocity",),
                  "action_args": {"agent_0_base_vel": [0.25, -0.5]}}
        # Replace the unavailable simulator boundary while exercising the real
        # RLTaskEnv capture and the real snapshot implementation.
        with patch.object(RLTaskEnv.__mro__[1], "step", create=True,
                          return_value=(None, 0.0, False, {})) as parent_step:
            env.step(action=action)
            self.assertFalse(hasattr(env, "_trajectory_executed_action"))
            env.set_trajectory_recording(True)
            result = env.step(action=action)
            self.assertEqual(result, (None, 0.0, False, {}))
            parent_step.assert_called_with(action=action)
        action["action_args"]["agent_0_base_vel"][0] = 99
        self.assertEqual(env.get_trajectory_state()["executed_action"], {
            "action": ["agent_0_base_velocity"],
            "action_args": {"agent_0_base_vel": [0.25, -0.5]},
        })

    def test_reset_clears_previous_episode_executed_action(self):
        env = self.make_env()
        env.set_trajectory_recording(True)
        env._trajectory_executed_action = {"action": "old_episode"}
        with patch.object(RLTaskEnv.__mro__[1], "reset", create=True,
                          return_value={"obs": 1}):
            self.assertEqual(env.reset(), {"obs": 1})
        state = env.get_trajectory_state()
        self.assertIsNone(state["executed_action"])
        self.assertEqual(state["missing_fields"]["executed_action"],
                         "no_recorded_step_since_reset")

    def test_recording_can_be_disabled_again(self):
        env = self.make_env()
        env.set_trajectory_recording(True)
        self.assertIn("__trajectory_state", env.get_info(None))
        env.set_trajectory_recording(False)
        del env._env.sim
        self.assertIs(env.get_info(None), env._env.get_metrics())

    def test_direct_snapshot_does_not_require_recording_enabled(self):
        env = self.make_env()
        self.assertEqual(env.get_trajectory_state()["num_steps"], 4)
        self.assertNotIn("__trajectory_state", env.get_info(None))

    def test_agents_without_end_effectors_or_grasps_are_supported(self):
        env = self.make_env()
        env._env.sim.agents_mgr[0] = _AgentData(
            _Robot([1, 2, 3], [0, 0, 0, 1]), grasp_managers=False
        )
        agent = env.get_trajectory_state()["agents"]["agent_1"]
        self.assertIsNone(agent["ee_position"])
        self.assertIsNone(agent["grasped_object_id"])

    def test_non_articulated_simulator_uses_indexed_xyzw_agent_state(self):
        env = self.make_env()
        states = [
            SimpleNamespace(
                position=[_Scalar(7), _Scalar(8), _Scalar(9)],
                rotation=SimpleNamespace(x=0.0, y=0.6, z=0.0, w=0.8),
            ),
            SimpleNamespace(
                position=[_Scalar(-1), _Scalar(-2), _Scalar(-3)],
                rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ]
        env._env.sim = SimpleNamespace(get_agent_state=states.__getitem__)
        for config in env.config.simulator.agents.values():
            config.articulated_agent_type = None

        state = env.get_trajectory_state()

        self.assertEqual(state["agents"]["agent_1"]["base_position"], [7, 8, 9])
        self.assertEqual(
            state["agents"]["agent_1"]["base_rotation"], [0, 0.6, 0, 0.8]
        )
        self.assertEqual(state["agents"]["agent_0"]["base_position"], [-1, -2, -3])
        self.assertEqual(
            state["agents"]["agent_0"]["base_rotation"], [0, 0, 0, 1]
        )
        self.assertIsNone(state["agents"]["agent_0"]["ee_position"])
        self.assertEqual(json.loads(json.dumps(state)), state)

    def test_terminal_reasons_include_success_limit_and_inactive_task(self):
        cases = [
            (False, False, False, False, None),
            (False, True, False, True, "success"),
            (True, True, True, True, "success"),
            (True, False, True, True, "time_limit"),
            (True, False, False, True, "task_inactive"),
        ]
        for done, success, past_limit, episode_over, reason in cases:
            with self.subTest(done=done, success=success, past_limit=past_limit):
                env = self.make_env(
                    done=done, success=success, past_limit=past_limit
                )
                state = env.get_trajectory_state()
                self.assertIs(state["episode_over"], episode_over)
                self.assertEqual(state["termination_reason"], reason)

    def test_success_metric_respects_disabled_end_on_success(self):
        for past_limit, expected_reason in [
            (True, "time_limit"),
            (False, "task_inactive"),
        ]:
            with self.subTest(past_limit=past_limit):
                env = self.make_env(done=True, success=True, past_limit=past_limit)
                env._end_on_success = False

                state = env.get_trajectory_state()

                self.assertIs(state["episode_over"], True)
                self.assertEqual(state["termination_reason"], expected_reason)

    def test_real_magnum_numpy_and_simulator_state_types_when_available(self):
        try:
            import habitat_sim
            import magnum as mn
            import numpy as np
        except ImportError:
            self.skipTest("Habitat-Sim, Magnum, and NumPy are not installed")

        env = self.make_env()
        transform = mn.Matrix4.translation(mn.Vector3(0.5, 1.5, 2.5))
        env._env.sim.agents_mgr[0] = _AgentData(
            SimpleNamespace(
                base_pos=mn.Vector3(1, 2, 3),
                sim_obj=SimpleNamespace(
                    rotation=mn.Quaternion(mn.Vector3(0, 0.6, 0), 0.8)
                ),
                base_rot=np.float32(2 * math.acos(0.8)),
                params=SimpleNamespace(ee_links=[0]),
                ee_transform=lambda: transform,
            ),
            np.int64(13),
        )
        env._env._elapsed_steps = np.int64(5)
        state = env.get_trajectory_state()
        self.assertEqual(state["agents"]["agent_1"]["base_position"], [1, 2, 3])
        self.assertEqual(state["agents"]["agent_1"]["ee_position"], [0.5, 1.5, 2.5])
        self.assertEqual(state["agents"]["agent_1"]["grasped_object_id"], 13)
        self.assertEqual(json.loads(json.dumps(state)), state)

        simulator_state = habitat_sim.AgentState()
        simulator_state.position = np.array([4, 5, 6], dtype=np.float32)
        env._env.sim = SimpleNamespace(get_agent_state=lambda index: simulator_state)
        state = env.get_trajectory_state()
        self.assertEqual(state["agents"]["agent_0"]["base_position"], [4, 5, 6])
        self.assertEqual(state["agents"]["agent_0"]["base_rotation"], [0, 0, 0, 1])
        self.assertEqual(json.loads(json.dumps(state)), state)

    def test_magnum_quaternions_preserve_opposite_yaw_and_roundtrip_rotation(self):
        try:
            import magnum as mn
        except ImportError:
            self.skipTest("Magnum is not installed")

        env = self.make_env()
        robot = _Robot([1, 2, 3], [0, 0, 0, 1])
        env._env.sim.agents_mgr[0] = _AgentData(robot)
        recorded = []
        for yaw, expected_z in [(90, -1), (-90, 1)]:
            with self.subTest(yaw=yaw):
                robot.sim_obj.rotation = mn.Quaternion.rotation(
                    mn.Deg(yaw), mn.Vector3.y_axis()
                )

                state = env.get_trajectory_state()

                rotation = state["agents"]["agent_1"]["base_rotation"]
                self.assertIsInstance(rotation, list)
                recorded.append(rotation)
                self.assertEqual(len(rotation), 4)
                reconstructed = mn.Quaternion(mn.Vector3(rotation[:3]), rotation[3])
                rotated = reconstructed.transform_vector(mn.Vector3.x_axis())
                self.assertAlmostEqual(rotated.x, 0, places=6)
                self.assertAlmostEqual(rotated.y, 0, places=6)
                self.assertAlmostEqual(rotated.z, expected_z, places=6)

        self.assertEqual(len(recorded), 2)
        self.assertGreater(recorded[0][1], 0)
        self.assertLess(recorded[1][1], 0)
        self.assertNotEqual(recorded[0], recorded[1])


if __name__ == "__main__":
    unittest.main()
