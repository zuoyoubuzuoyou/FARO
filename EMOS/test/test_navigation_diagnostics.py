"""Run real navigation branches without importing the Habitat simulator stack."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTIONS = ROOT / 'habitat-lab/habitat/tasks/rearrange/actions'


def load_actions():
    class Base:
        def reset(self, *args, **kwargs):
            pass

        def step(self, *args, **kwargs):
            key = 'agent_0_human_joints_trans' if 'agent_0_human_joints_trans' in kwargs else 'agent_0_base_vel'
            self.commands.append(np.asarray(kwargs[key]).tolist())
            return {'command': self.commands[-1]}

    class Cylinder(Base):
        pass

    class NonCylinder(Base):
        pass

    class Human(Base):
        pass

    def angle(a, b):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return 0.0 if denom == 0 else np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))

    ns = dict(np=np, BaseVelAction=Cylinder, BaseVelNonCylinderAction=NonCylinder,
              HumanoidJointAction=Human, get_angle=angle, Vector3=object,
              mn=SimpleNamespace(Vector3=np.asarray), DEBUG_SAVE_PATHFINDER_MAP=False,
              habitat_sim=SimpleNamespace(ShortestPath=lambda: SimpleNamespace()))
    helper = ACTIONS / 'navigation_diagnostics.py'
    if helper.exists():
        spec = importlib.util.spec_from_file_location('navigation_diagnostics_test', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ns['_record_navigation_diagnostics'] = module._record_navigation_diagnostics
    for filename, names in [('oracle_nav_action.py', {'OracleNavAction', 'OracleNavCoordinateAction'}),
                            ('habitat_mas_actions.py', {'OracleNavDiffBaseAction'})]:
        tree = ast.parse((ACTIONS / filename).read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        for node in classes:
            node.decorator_list = []
        body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *classes]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), filename, 'exec'), ns)
    return ns


CLASSES = load_actions()


class NavigationDiagnosticsTests(unittest.TestCase):
    def make_action(self, name='OracleNavAction', *, goal=(2., 0., 0.), obj=(3., 0., 0.),
                    motion='base_velocity', found=True, points=None):
        action = CLASSES[name].__new__(CLASSES[name])
        action._task = SimpleNamespace(_cur_episode_step=6, _episode_id='episode')
        action._action_arg_prefix = 'agent_0_'
        action.motion_type = motion
        action._poss_entities = [SimpleNamespace(name='any_targets|0')]
        action._targets = {0: (np.array(goal), np.array(obj))}
        action._prev_ep_id = 'episode'
        action.skill_done = False
        action.prev_nav_done = False
        action.prev_match_target_id = None
        action.ep_id = 'episode'
        action.commands = []
        action.path_calls = 0
        class Config(SimpleNamespace):
            def __getitem__(self, key):
                return getattr(self, key)
        action._config = Config(dist_thresh=.5, turn_thresh=.2, forward_velocity=.8, turn_velocity=.3, lin_speed=1.)
        action.cur_articulated_agent = SimpleNamespace(base_pos=np.zeros(3),
            base_transformation=SimpleNamespace(transform_vector=lambda v: v))

        def find_path(path):
            action.path_calls += 1
            path.points = [np.zeros(3), np.array(goal)] if points is None else points
            return found

        action.pathfinder = SimpleNamespace(find_path=find_path)
        action._sim = SimpleNamespace(pathfinder=action.pathfinder, ep_info=SimpleNamespace(episode_id='episode'))
        action.humanoid_controller = SimpleNamespace(obj_transform_base=None,
            calculate_turn_pose=lambda *a: None, calculate_walk_pose=lambda *a: None,
            calculate_stop_pose=lambda: None, get_pose=lambda: np.array([.1, .2, .3]))
        action._update_controller_to_navmesh = lambda: None
        action.test_goal = np.array(goal)
        action.test_obj = np.array(obj)
        return action

    def run_action(self, action, *, zero=False):
        if type(action).__name__ == 'OracleNavCoordinateAction':
            return action.step(agent_0_oracle_nav_lookat_action=np.zeros(3) if zero else action.test_obj,
                agent_0_oracle_nav_coord_action=action.test_goal, agent_0_mode=0)
        return action.step(agent_0_oracle_nav_action=np.array([0 if zero else 1]))

    def test_real_controller_commands_and_current_detached_diagnostics(self):
        # Catches changing control vectors, step stamping, paths, and entity/proxy mapping.
        for name in ('OracleNavAction', 'OracleNavCoordinateAction', 'OracleNavDiffBaseAction'):
            for motion in ('base_velocity', 'base_velocity_non_cylinder', 'human_joints'):
                with self.subTest(name=name, motion=motion):
                    action = self.make_action(name, motion=motion)
                    result = self.run_action(action)
                    self.assertEqual(result, {'command': action.commands[-1]} if name == 'OracleNavCoordinateAction' else None)
                    self.assertEqual(action.commands, [[.1, .2, .3] if motion == 'human_joints' else [.8, 0.]])
                    self.assertEqual(action.path_calls, 1)
                    self.assertTrue(hasattr(action, '_navigation_diagnostics'), 'missing navigation evidence')
                    d = action._navigation_diagnostics
                    self.assertEqual(d['env_step'], 7)
                    self.assertEqual(d['timing'], 'pre_control')
                    self.assertTrue(d['active'])
                    self.assertTrue(d['available'])
                    self.assertTrue(d['path_found'])
                    self.assertEqual(d['target_position'], [3., 0., 0.])
                    self.assertEqual(d['proxy_position'], [2., 0., 0.])
                    self.assertEqual(d['distance_to_goal'], 2.)
                    self.assertFalse(d['distance_satisfied'])
                    self.assertTrue(d['orientation_satisfied'])
                    self.assertFalse(d['at_goal'])
                    self.assertFalse(d['skill_done'])
                    self.assertEqual(d['control'], action.commands[-1])
                    if name != 'OracleNavCoordinateAction':
                        self.assertEqual(d['requested_target'], 'any_targets|0')
                        self.assertTrue(d['target_cache_hit'])
                    json.dumps(d, allow_nan=False)
                    action._targets[0][0][:] = 99
                    self.assertEqual(d['proxy_position'], [2., 0., 0.])

    def test_native_target_cache_miss_is_observed_without_extra_target_or_spawn_queries(self):
        counts = {'target': 0, 'spawn': 0}

        def position(entity):
            counts['target'] += 1
            return np.array([3., 0., 0.])

        def place(object_position, minimum_distance, maximum_distance, sim,
                  attempts, physics_stability, agent):
            counts['spawn'] += 1
            return np.array([2., 0., 0.]), None, None

        CLASSES['place_agent_at_dist_from_pos'] = place
        try:
            for name in ('OracleNavAction', 'OracleNavDiffBaseAction'):
                counts.update(target=0, spawn=0)
                action = self.make_action(name)
                action._targets = {}
                action._config.spawn_max_dist_to_obj = 1.
                action._config.num_spawn_attempts = 5
                action._task.pddl_problem = SimpleNamespace(sim_info=SimpleNamespace(get_entity_pos=position))
                self.run_action(action)
                self.assertEqual(action.commands, [[.8, 0]])
                self.assertFalse(action._navigation_diagnostics['target_cache_hit'])
                self.run_action(action)
                self.assertEqual(action.commands, [[.8, 0], [.8, 0]])
                self.assertTrue(action._navigation_diagnostics['target_cache_hit'])
                self.assertEqual(counts, {'target': 1, 'spawn': 1})
                self.assertEqual(action.path_calls, 2)
        finally:
            CLASSES.pop('place_agent_at_dist_from_pos')

    def test_invalid_inputs_keep_original_return_and_error_behavior(self):
        for name in ('OracleNavAction', 'OracleNavDiffBaseAction'):
            action = self.make_action(name)
            self.assertIsNone(action.step(agent_0_oracle_nav_action=np.array([2])))
            self.assertEqual(action.commands, [])
            self.assertFalse(action._navigation_diagnostics['active'])
            with self.assertRaises(ValueError):
                action.step(agent_0_oracle_nav_action=np.array([float('nan')]))
            self.assertIsNone(action._navigation_diagnostics['control'])
        action = self.make_action('OracleNavCoordinateAction')
        with self.assertRaisesRegex(ValueError, 'Mode not provided'):
            action.step()
        self.assertEqual(action.commands, [])
        self.assertFalse(action._navigation_diagnostics['active'])

    def test_single_point_coordinate_path_and_humanoid_arrival_keep_native_results(self):
        action = self.make_action('OracleNavCoordinateAction', goal=(.1, 0., 0.),
                                  points=[np.array([.1, 0., 0.])])
        self.assertEqual(self.run_action(action), {'command': [0, 0]})
        self.assertEqual(action._navigation_diagnostics['next_waypoint'], [.1, 0., 0.])
        action = self.make_action('OracleNavCoordinateAction', goal=(.1, 0., 0.), motion='human_joints')
        self.assertEqual(self.run_action(action), {'command': [.1, .2, .3]})
        self.assertTrue(action._navigation_diagnostics['skill_done'])

    def test_optional_observer_metadata_error_preserves_control_and_execution_exception(self):
        class BrokenName:
            @property
            def name(self):
                raise RuntimeError('optional observer metadata')
        action = self.make_action()
        action._poss_entities = [BrokenName()]
        self.run_action(action)
        self.assertEqual(action.commands, [[.8, 0]])
        self.assertEqual(action._navigation_diagnostics['control'], [.8, 0])
        action.pathfinder.find_path = lambda path: (_ for _ in ()).throw(ValueError('physical error'))
        with self.assertRaisesRegex(ValueError, 'physical error'):
            self.run_action(action)
        self.assertIsNone(action._navigation_diagnostics['control'])

    def test_detector_field_names_retain_real_positions(self):
        action = self.make_action()
        self.run_action(action)
        d = action._navigation_diagnostics
        self.assertEqual(d.get('target'), 'any_targets|0')
        self.assertEqual(d.get('true_object_position'), [3., 0., 0.])
        self.assertEqual(d.get('final_navigation_position'), [2., 0., 0.])
        self.assertEqual(d.get('next_waypoint'), [2., 0., 0.])
        self.assertEqual(d.get('robot_position_before_control'), [0., 0., 0.])

    def test_malformed_step_stamp_cannot_leave_fresh_previous_control(self):
        action = self.make_action()
        self.run_action(action)
        action._task._cur_episode_step = float('nan')
        self.run_action(action, zero=True)
        self.assertFalse(action._navigation_diagnostics['active'])
        self.assertIsNone(action._navigation_diagnostics['env_step'])
        self.assertIsNone(action._navigation_diagnostics['control'])

    def test_arrival_preserves_distinct_coordinate_skill_done_and_shortcut(self):
        for name, expected_done in [('OracleNavAction', True), ('OracleNavCoordinateAction', False), ('OracleNavDiffBaseAction', True)]:
            with self.subTest(name=name):
                action = self.make_action(name, goal=(.1, 0., 0.))
                self.run_action(action)
                self.assertEqual(action.commands, [[0, 0]])
                self.assertEqual(action.skill_done, expected_done)
                self.assertEqual(action._navigation_diagnostics['skill_done'], expected_done)
                self.assertTrue(action._navigation_diagnostics['at_goal'])
        action = self.make_action('OracleNavCoordinateAction', goal=(.01, 0., 0.), obj=(-2., 0., 0.))
        self.run_action(action)
        self.assertEqual(action.commands, [[0, 0]])
        self.assertFalse(action._navigation_diagnostics['orientation_satisfied'])
        self.assertTrue(action._navigation_diagnostics['at_goal'])

    def test_threshold_edge_and_turn_commands(self):
        for goal, obj, expected in [((.5, 0., 0.), (3., 0., 0.), [.8, 0]),
                                    ((.1, 0., 0.), (0., 0., 3.), [0, -.3]),
                                    ((0., 0., 2.), (0., 0., 3.), [0, -.3])]:
            action = self.make_action(goal=goal, obj=obj)
            self.run_action(action)
            self.assertEqual(action.commands, [expected])
            self.assertFalse(action._navigation_diagnostics['at_goal'])

    def test_zero_command_clears_prior_active_diagnostics(self):
        for name in ('OracleNavAction', 'OracleNavCoordinateAction', 'OracleNavDiffBaseAction'):
            action = self.make_action(name)
            self.run_action(action)
            action._task._cur_episode_step += 1
            result = self.run_action(action, zero=True)
            self.assertEqual(result, {} if name == 'OracleNavCoordinateAction' else None)
            self.assertEqual(len(action.commands), 1)
            d = action._navigation_diagnostics
            self.assertFalse(d['active'])
            self.assertFalse(d['available'])
            self.assertEqual(d['env_step'], 8)
            self.assertIsNone(d['control'])
            self.assertIsNone(d['skill_done'])

    def test_path_failure_uses_original_fallback_command(self):
        for name in ('OracleNavAction', 'OracleNavCoordinateAction', 'OracleNavDiffBaseAction'):
            action = self.make_action(name, found=False)
            self.run_action(action)
            self.assertEqual(action.commands, [[.8, 0]])
            self.assertFalse(action._navigation_diagnostics['path_found'])
            self.assertEqual(action.path_calls, 1)

    def test_mas_previous_arrival_suppresses_repeat_without_reusing_old_control(self):
        action = self.make_action('OracleNavDiffBaseAction', goal=(.1, 0., 0.))
        self.run_action(action)
        action._task._cur_episode_step += 1
        self.run_action(action)
        self.assertEqual(action.commands, [[0, 0]])
        self.assertFalse(action._navigation_diagnostics['active'])
        self.assertTrue(action._navigation_diagnostics['prev_nav_done'])
        self.assertIsNone(action._navigation_diagnostics['control'])
        self.assertFalse(action.skill_done)

    def test_reset_clears_navigation_evidence_without_resetting_coordinate_cache(self):
        for name in ('OracleNavAction', 'OracleNavCoordinateAction', 'OracleNavDiffBaseAction'):
            action = self.make_action(name)
            self.run_action(action)
            action.reset()
            self.assertFalse(action._navigation_diagnostics['active'])
            self.assertIsNone(action._navigation_diagnostics['control'])
            self.assertIsNone(action._navigation_diagnostics['env_step'])

    def test_original_exceptions_and_nonfinite_values_remain_observation_safe(self):
        action = self.make_action()
        del action._task._cur_episode_step
        self.run_action(action)
        self.assertEqual(action.commands, [[.8, 0]])
        self.assertIsNone(action._navigation_diagnostics['env_step'])
        action = self.make_action(goal=(float('nan'), 0., 0.))
        self.run_action(action)
        json.dumps(action._navigation_diagnostics, allow_nan=False)
        self.assertIsNone(action._navigation_diagnostics['distance_to_goal'])
        action = self.make_action()
        action.pathfinder.find_path = lambda path: (_ for _ in ()).throw(ValueError('original path error'))
        with self.assertRaisesRegex(ValueError, 'original path error'):
            self.run_action(action)
        self.assertIsNone(action._navigation_diagnostics['control'])


if __name__ == '__main__':
    unittest.main()
