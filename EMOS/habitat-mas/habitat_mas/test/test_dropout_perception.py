"""Offline cameras cannot satisfy goals or hide unfinished recovery work."""

import unittest
from types import SimpleNamespace

import numpy as np

from habitat.tasks.rearrange.rearrange_sensors import DetectedObjectsSensor
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.tasks.rearrange.multi_task.pddl_sim_state import PddlSimState, PddlRobotState
from habitat.tasks.rearrange.multi_task.pddl_predicate import Predicate
from habitat.tasks.rearrange.multi_task.rearrange_pddl import PddlEntity, ExprType
from habitat_mas.error_recovery import snapshot_goals, build_recovery_plan
from habitat_mas.test import test_agent_hang_policy, test_dropout_recovery


class DropoutPerceptionTest(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "agent_0_head_semantic": np.full((2, 2), 101),
            "agent_1_head_semantic": np.full((2, 2), 102),
        }
        self.sim = SimpleNamespace(
            offline_agent_ids=set(), scene_obj_ids=[1, 2],
            habitat_config=SimpleNamespace(object_ids_start=100),
            get_sensor_observations=lambda: self.frames,
            sensor_suite=SimpleNamespace(get_observations=lambda obs: obs),
            get_agent_data=lambda i: SimpleNamespace(grasp_mgr=SimpleNamespace(snap_idx=None)),
            agents_mgr=SimpleNamespace(agent_names=["agent_0", "agent_1"]),
        )
        self.sensors = {}
        for i in range(2):
            sensor = DetectedObjectsSensor.__new__(DetectedObjectsSensor)
            sensor._sim, sensor.agent_id, sensor.pixel_threshold = self.sim, i, 0
            self.sensors[f"agent_{i}_detected_objects"] = sensor
        suite = SimpleNamespace(sensors=self.sensors, get=self.sensors.__getitem__)
        self.info = SimpleNamespace(
            sim=self.sim, robot_ids=[0, 1], env=SimpleNamespace(sensor_suite=suite),
            search_for_entity=lambda e: int(e.name.split("|")[-1]) if "|" in e.name else int(e.name[-1]),
            pred_truth_cache={},
        )
        self.goals = []
        for i in range(2):
            entity = PddlEntity(f"any_targets|{i}", ExprType("movable_entity_type", None))
            state = PddlSimState({}, {}, {}, is_detected=entity)
            pred = Predicate("is_detected", state, [entity])
            pred.set_param_values([entity])
            self.goals.append(pred)
        self.problem = SimpleNamespace(
            sim_info=self.info,
            goal=SimpleNamespace(expr_type=SimpleNamespace(value="and"), quantifier=None, sub_exprs=self.goals),
        )
        self.task = RearrangeTask.__new__(RearrangeTask)
        self.task._sim, self.task.pddl_problem = self.sim, self.problem

    def test_offline_sensor_returns_no_detections(self):
        self.sim.offline_agent_ids = {1}
        self.assertEqual(self.sensors["agent_1_detected_objects"].get_observation(self.frames).tolist(), [])
        self.assertEqual(self.sensors["agent_0_detected_objects"].get_observation(self.frames).tolist(), [101])

    def test_aggregate_sensor_excludes_offline_camera(self):
        sensor = self.sensors["agent_0_detected_objects"]
        sensor.agent_id = None
        self.sim.offline_agent_ids = {1}
        self.assertEqual(sensor.get_observation(self.frames).tolist(), [101])
        self.sim.offline_agent_ids = {0, 1}
        self.assertEqual(sensor.get_observation(self.frames).tolist(), [])

    def test_offline_view_cannot_complete_team_goal(self):
        self.sim.offline_agent_ids = {1}
        self.assertFalse(self.goals[1].is_true(self.info))
        self.assertTrue(self.goals[0].is_true(self.info))

    def test_agent_specific_detection_cannot_use_another_agents_camera(self):
        self.sim.offline_agent_ids = {1}
        # Exercise the shared sensor fallback as well as per-agent sensors.
        shared = self.sensors["agent_0_detected_objects"]
        shared.agent_id = None
        self.sensors.clear()
        self.sensors["detected_objects"] = shared
        state = PddlRobotState(detected_object=self.goals[0]._arg_values[0])
        robot = PddlEntity("agent_1", ExprType("robot_entity_type", None))
        self.assertFalse(state.is_true(self.info, robot))

    def test_missing_detection_sensor_does_not_fabricate_success(self):
        self.sensors.clear()
        self.assertFalse(self.goals[0].is_true(self.info))

    def test_dropout_invalidates_cached_truth_before_reassignment(self):
        self.assertTrue(all(p.is_true(self.info) for p in self.goals))
        RearrangeTask.set_offline_agents(self.task, ["agent_1"])
        goals = snapshot_goals(self.problem)
        self.assertEqual([g["complete"] for g in goals], [True, False])
        plan = build_recovery_plan(goals, {"agent_1": "Detect any_targets|1"}, ["agent_0"], ["agent_1"], {})
        self.assertEqual(plan["assignments"]["agent_0"], ["is_detected(any_targets|1)"])
        # The survivor can still complete the same target using its own camera.
        self.frames["agent_0_head_semantic"][:] = 102
        self.assertTrue(self.goals[1].is_true(self.info))

    def test_clearing_offline_state_restores_perception(self):
        RearrangeTask.set_offline_agents(self.task, ["agent_0", "agent_1"])
        self.assertFalse(any(p.is_true(self.info) for p in self.goals))
        RearrangeTask.set_offline_agents(self.task, [])
        self.assertTrue(all(p.is_true(self.info) for p in self.goals))

    def test_episode_reset_clears_offline_state(self):
        self.task._sim_reset = False
        self.task._is_episode_active = True
        self.task.measurements = SimpleNamespace(get_metrics=lambda: {})
        self.sim.offline_agent_ids = {1}
        self.sim.add_perf_timing = lambda *args: None
        self.sim.cur_runtime_perf_scope = []
        self.task.reset(SimpleNamespace(episode_id="next"), fetch_observations=False)
        self.assertTrue(all(p.is_true(self.info) for p in self.goals))

    def test_offline_observations_are_blank_without_hiding_online_camera(self):
        self.sim.offline_agent_ids = {1}
        obs = {**self.frames, "agent_1_head_rgb": np.ones((2, 2, 3), dtype=np.uint8)}
        self.task._mask_offline_observations(obs)
        self.assertFalse(obs["agent_1_head_semantic"].any())
        self.assertFalse(obs["agent_1_head_rgb"].any())
        self.assertTrue(obs["agent_0_head_semantic"].all())


class DropoutPerceptionDispatchTest(unittest.TestCase):
    def test_environment_is_updated_before_survivor_executes(self):
        fixture = test_agent_hang_policy.AgentHangPolicyTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        offline = set()
        original = fixture.policy.act

        def act(*args, **kwargs):
            kwargs["set_offline_agents"] = lambda names: (offline.clear(), offline.update(names))
            return original(*args, **kwargs)

        fixture.policy.act = act
        original_survivor = fixture.agents[0].act.side_effect

        def survivor(*args, **kwargs):
            self.assertEqual(offline, {"agent_1"})
            return original_survivor(*args, **kwargs)

        fixture.agents[0].act.side_effect = survivor
        fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})

    def test_environment_is_updated_before_recovery_reads_goals(self):
        fixture = test_dropout_recovery.RecoveryDispatchTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        offline = set()
        original_act = fixture.fixture.policy.act
        original_recover = fixture.recovery.recover

        def act(*args, **kwargs):
            kwargs["set_offline_agents"] = lambda names: (offline.clear(), offline.update(names))
            return original_act(*args, **kwargs)

        def recover(*args, **kwargs):
            self.assertEqual(offline, {"agent_1"})
            return original_recover(*args, **kwargs)

        fixture.fixture.policy.act = act
        fixture.recovery.recover = recover
        fixture.fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})
        self.assertEqual(fixture.recovery.transferred_tasks, 1)


if __name__ == "__main__":
    unittest.main()
