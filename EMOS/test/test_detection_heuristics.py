"""Behavioral counterexamples for causal, team-aware monitoring."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'habitat-mas'))
from habitat_mas.detection.heuristics import RepeatActionDetector, TeamProgressDetector, assess_goals


def frame(step, *, action='wait', position=None, distance=4.0, task='manipulation'):
    return {'step': step, 'task': task,
            'context': {'assignments': {'agent_0': 'Pick any_targets|0 and place at TARGET_any_targets|0.',
                                        'agent_1': 'Pick any_targets|1 and place at TARGET_any_targets|1.'},
                        'goal_positions': {'any_targets|0': [10, 0, 0]}},
            'agents': {'agent_0': {'position': position or [0, 0, 0], 'rotation': 0,
                                   'held_object': None, 'action': {'name': action, 'arguments': {'target_obj': 'any_targets|0'},
                                                                  'id': 'a', 'age': step + 1},
                                   'navigation': None, 'termination': None, 'detection': None},
                       'agent_1': {'position': [10, 0, 0], 'rotation': 0, 'held_object': None,
                                   'action': {'name': 'wait', 'arguments': {}, 'id': 'b', 'age': step + 1},
                                   'navigation': None, 'termination': None, 'detection': None}},
            'objects': {'any_targets|0': {'position': [1, 0, 0], 'object_id': 4},
                        'any_targets|1': {'position': [10, 0, 0], 'object_id': 5}},
            'metrics': {'pddl_success': 0, 'object_to_goal_distance.0': distance,
                        'object_to_goal_distance.1': 0.0}, 'decisions': [], 'messages': []}


class HeuristicTests(unittest.TestCase):
    def feed(self, detector, **kwargs):
        return [v for i in range(0, 501, 50) if (v := detector.observe(frame(i, **kwargs))) is not None]

    def test_repeat_baseline_is_literal_strict_more_than_ten(self):
        d = RepeatActionDetector(threshold=10)
        for i in range(10): self.assertIsNone(d.observe(frame(i, action='nav_to_obj')))
        a = d.observe(frame(10, action='nav_to_obj'))
        self.assertEqual(a['status'], 'system_error')
        self.assertEqual(a['step'], 10)

    def test_normal_navigation_motion_does_not_trigger_team_rule(self):
        d = TeamProgressDetector(window=10, check_every=5)
        for i in range(0, 1001, 50):
            self.assertIsNone(d.observe(frame(i, action='nav_to_obj', position=[i / 100, 0, 0])))

    def test_waiting_completed_robot_while_teammate_moves_is_normal(self):
        d = TeamProgressDetector(window=10, check_every=5)
        for i in range(0, 1001, 50):
            f = frame(i, distance=0)
            f['metrics']['object_to_goal_distance.1'] = 5
            f['agents']['agent_1']['action']['name'] = 'nav_to_obj'
            f['agents']['agent_1']['position'] = [i / 100, 0, 0]
            self.assertIsNone(d.observe(f))

    def test_completed_robot_reset_does_not_imply_team_stall(self):
        d = TeamProgressDetector(window=10, check_every=5)
        for i in range(0, 1001, 50):
            f = frame(i, action='reset_arm', distance=0)
            f['metrics']['object_to_goal_distance.1'] = 5
            f['agents']['agent_1']['action']['name'] = 'nav_to_obj'
            f['agents']['agent_1']['position'] = [i / 100, 0, 0]
            self.assertIsNone(d.observe(f))

    def test_unfinished_responsible_robot_reset_over_budget_is_detected(self):
        alarms = self.feed(TeamProgressDetector(window=10, check_every=5), action='reset_arm')
        self.assertEqual(alarms[0]['reason_code'], 'reset_over_budget')
        self.assertEqual(alarms[0]['agents'], ['agent_0'])
        self.assertIn(450, [e['step'] for e in alarms[0]['evidence']])

    def test_team_wait_requires_unfulfilled_responsibility(self):
        alarms = self.feed(TeamProgressDetector(window=10, check_every=5))
        self.assertEqual(alarms[0]['reason_code'], 'unresolved_team_wait')
        d = TeamProgressDetector(window=10, check_every=5)
        for i in range(0, 1001, 50):
            f = frame(i)
            f['context']['assignments']['agent_0'] = 'Nothing to do.'
            self.assertIsNone(d.observe(f))

    def test_one_frame_no_path_is_not_enough(self):
        d = TeamProgressDetector(window=10, check_every=5)
        f = frame(0, action='nav_to_obj'); f['agents']['agent_0']['navigation'] = {'path_found': False}
        self.assertIsNone(d.observe(f))

    def test_sustained_stationary_navigation_is_detected_without_ik_claim(self):
        alarms = self.feed(TeamProgressDetector(window=10, check_every=5), action='nav_to_obj')
        self.assertEqual(alarms[0]['reason_code'], 'navigation_no_progress')
        self.assertNotIn('机械损坏', alarms[0]['explanation'])

    def test_current_success_suppresses_new_alarm_not_historical_stage(self):
        d = TeamProgressDetector(window=10, check_every=5)
        for i in range(0, 1001, 50):
            f = frame(i); f['metrics']['pddl_success'] = 1
            self.assertIsNone(d.observe(f))
        f = frame(0); f['metrics']['pddl_stage_goals.object_at_receptacle_0_success'] = 1
        self.assertFalse(assess_goals(f)[0]['satisfied'])

    def test_missing_diagnostics_and_positions_are_not_false_values(self):
        f = frame(0, task='mobility')
        f['agents']['agent_0']['position'] = None
        f['agents']['agent_1']['position'] = None
        self.assertTrue(all(r['satisfied'] is None for r in assess_goals(f)))

    def test_perception_detection_loss_is_current_not_latched(self):
        f = frame(0, task='perception')
        for a in f['agents'].values():
            a['detection'] = {'available': True, 'target_detections': {'any_targets|0': True, 'any_targets|1': False}}
        goals = assess_goals(f)
        det = {g['object']:g['satisfied'] for g in goals if g['kind']=='detection'}
        self.assertEqual(det, {'any_targets|0': True, 'any_targets|1': False})

    def test_observed_history_does_not_change_after_caller_mutation(self):
        d = TeamProgressDetector(window=10, check_every=5)
        f=frame(0); original=copy.deepcopy(f); d.observe(f)
        self.assertEqual(f, original)
        f['agents']['agent_0']['position'][0]=99
        self.assertEqual(d.frames[0]['agents']['agent_0']['position'], [0,0,0])

    def test_missing_pose_is_not_stationary_navigation_evidence(self):
        d=TeamProgressDetector()
        for step in range(0, 1001, 50):
            f=frame(step, action='nav_to_obj')
            for a in f['agents'].values(): a['position']=None; a['rotation']=None
            self.assertIsNone(d.observe(f))

    def test_navigation_distance_improvement_is_real_progress(self):
        d=TeamProgressDetector()
        for step in range(0, 1001, 50):
            f=frame(step, action='nav_to_obj')
            f['agents']['agent_0']['navigation']={'distance_to_goal': 20-step/100}
            self.assertIsNone(d.observe(f))

    def test_task_distance_improvement_prevents_no_progress_alarm(self):
        d=TeamProgressDetector()
        for step in range(0, 1001, 50):
            self.assertIsNone(d.observe(frame(step, action='nav_to_obj', distance=20-step/100)))

    def test_unobserved_arm_motion_is_not_team_inactivity(self):
        d=TeamProgressDetector()
        for step in range(0, 1001, 50):
            f=frame(step)
            f['agents']['agent_1']['action'].update(name='pick', id=step)
            self.assertIsNone(d.observe(f))

    def test_unknown_ids_do_not_merge_distinct_action_names(self):
        d=TeamProgressDetector()
        for step in range(0, 451, 50):
            f=frame(step, action='nav_to_obj' if step==450 else 'wait')
            f['agents']['agent_0']['action']['id']=None
            self.assertIsNone(d.observe(f))

    def test_unknown_teammate_pose_makes_uncovered_condition_unknown(self):
        f=frame(0, task='mobility')
        f['agents']['agent_0']['position']=[50,0,0]
        f['agents']['agent_1']['position']=None
        self.assertTrue(all(r['satisfied'] is None for r in assess_goals(f)))

    def test_unknown_teammate_detection_is_not_false(self):
        f=frame(0, task='perception')
        f['agents']['agent_0']['detection']={'available':True,'target_detections':{'any_targets|0':False}}
        f['agents']['agent_1']['detection']=None
        r=next(r for r in assess_goals(f) if r['kind']=='detection' and r['object']=='any_targets|0')
        self.assertIsNone(r['satisfied'])

    def test_scout_is_not_a_placement_owner(self):
        d=TeamProgressDetector()
        for step in range(0,1001,50):
            f=frame(step)
            f['context']['assignments']['agent_0']='Scout any_targets|0 and report its location.'
            self.assertIsNone(d.observe(f))

    def test_ten_raw_frames_window_can_still_detect_a_hundred_step_stall(self):
        d=TeamProgressDetector(window=10,check_every=5,min_stall_steps=100)
        alarms=[a for i in range(201) if (a:=d.observe(frame(i))) is not None]
        self.assertTrue(alarms)
        self.assertGreaterEqual(alarms[0]['step'],100)

    def test_slow_navigation_accumulates_motion_across_short_windows(self):
        d=TeamProgressDetector(window=10,check_every=5,min_stall_steps=100)
        # The historical Manipulation batch has no navigation diagnostics.
        # At 0.005 m/frame, each short window moves <0.05 m but the robot
        # travels a metre over the complete interval.
        for step in range(201):
            self.assertIsNone(d.observe(frame(step,action='nav_to_obj',position=[step*.005,0,0])))

    def test_slow_rotation_accumulates_across_short_windows(self):
        d=TeamProgressDetector(window=10,check_every=5,min_stall_steps=100)
        for step in range(201):
            f=frame(step,action='nav_to_obj')
            f['agents']['agent_0']['rotation']=step*.008
            self.assertIsNone(d.observe(f))

    def test_new_navigation_does_not_inherit_previous_wait_duration(self):
        d=TeamProgressDetector(window=10,check_every=5,min_stall_steps=100)
        for step in range(200):
            f=frame(step,action='wait' if step<100 else 'nav_to_obj')
            f['agents']['agent_0']['action'].update(id='wait-1' if step<100 else 'nav-2',
                age=step+1 if step<100 else step-99)
            alarm=d.observe(f)
            if step>=100:
                self.assertIsNone(alarm,'New navigation has not run for 100 steps')

    def test_repeat_missing_actions_break_consecutive_run(self):
        d=RepeatActionDetector(threshold=10)
        for step in range(12):
            f=frame(step,action='nav_to_obj')
            del f['agents']['agent_1']
            if step==9:f['agents']['agent_0']['action']['name']=None
            self.assertIsNone(d.observe(f))
        d=RepeatActionDetector(threshold=10)
        for step in range(12):
            f=frame(step,action='nav_to_obj')
            del f['agents']['agent_1']
            if step==9: f['agents'].clear()
            self.assertIsNone(d.observe(f))

    def test_short_window_alarm_retains_real_quiet_anchor_evidence(self):
        d=TeamProgressDetector(window=10,check_every=5,min_stall_steps=100)
        alarms=[a for step in range(110) if (a:=d.observe(frame(step,action='nav_to_obj'))) is not None]
        self.assertTrue(alarms)
        details=alarms[0]['evidence'][-1]['observation']
        history=details['quiet_history']
        self.assertEqual(history['anchor_observation']['step'],0)
        self.assertEqual(history['anchor_observation']['agents']['agent_0']['position'],[0,0,0])
        self.assertGreaterEqual(history['duration_env_steps'],100)
        self.assertEqual(history['duration_env_steps'],alarms[0]['step']-history['anchor_observation']['step'])


if __name__ == '__main__': unittest.main()
