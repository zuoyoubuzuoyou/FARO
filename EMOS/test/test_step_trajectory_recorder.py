"""Recorder integration for durable, completed-transition storage."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from test_trajectory_recorder import load_recorder


def storage():
    path = Path(__file__).resolve().parents[1] / 'habitat-mas/habitat_mas/utils/trajectory_storage.py'
    spec = importlib.util.spec_from_file_location('_step_storage_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StepRecorderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recorder = load_recorder()
        self.callbacks = []
        self.trace = self.recorder.EvaluationTrace(self.temp.name, event_callback=self.callbacks.append)
        self.addCleanup(self.trace.close)
        self.trace.start_episode(0, SimpleNamespace(scene_id='scene', episode_id='same'), state={'initial': True})
        self.directory = next((self.trace.run_dir / 'episodes').iterdir())

    def rows(self):
        return [json.loads(line) for line in (self.directory / 'steps.jsonl').read_text().splitlines()]

    def test_two_transitions_group_ordered_callbacks_and_terminal_summary(self):
        self.assertTrue((self.directory / 'steps.jsonl').exists())
        with self.trace.policy_step():
            self.recorder.begin_action('agent_0', 'move', {})
            self.recorder.end_action('agent_0', 'completed', 'returned', defer=True)
            self.recorder.record_event('agent_action', replay_context={'chat_history': [['message']]})
        self.trace.submit_actions(0, [1], [1], ['agent_0'])
        self.trace.record_step(0, [1], 1, False, {'success': False}, {'task_progress': {'success': None}})
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['step'], 1)
        self.assertEqual(self.rows()[0]['schema_version'], 3)
        self.trace.record_step(0, [0], 0, True, {'success': True}, {'terminal': True}, success=True)
        self.assertEqual(len(self.rows()), 2)
        events = list(storage().iter_trace_events(self.directory / 'steps.jsonl'))
        self.assertEqual(events, self.callbacks)
        self.assertEqual(events[-1]['event_type'], 'episode_end')
        names = [e['event_type'] for e in events]
        self.assertLess(names.index('step'), names.index('action_execution'))
        self.assertLess(names.index('action_execution'), names.index('action_end'))
        self.assertEqual(next(e for e in events if e['event_type'] == 'agent_action')['replay_context']['chat_history'], [['message']])
        summary = storage().read_trace_summary(self.directory / 'summary.json')
        self.assertTrue(summary['task_success'])
        self.assertEqual(summary['final_state'], {'terminal': True})
        self.assertEqual(summary['files']['steps'], 'steps.jsonl')
        self.assertFalse((self.trace.run_dir / 'run.json').exists())
        self.callbacks[0]['state']['initial'] = False
        self.assertTrue(list(storage().iter_trace_events(self.directory / 'steps.jsonl'))[0]['state']['initial'])

    def test_error_before_step_has_pending_evidence_and_zero_transitions(self):
        with self.trace.policy_step():
            self.recorder.record_event('llm_request', messages=[{'content': 'request'}])
        pending = list(storage().iter_trace_events(self.directory / 'steps.jsonl'))
        self.assertEqual([e['event_type'] for e in pending], ['episode_start', 'llm_request'])
        self.assertEqual(self.rows(), [])
        self.trace.close(error=RuntimeError('before environment'))
        self.assertEqual(self.rows(), [])
        events = list(storage().iter_trace_events(self.directory / 'steps.jsonl'))
        self.assertIn('llm_request', [e['event_type'] for e in events])
        self.assertIn('exception', [e['event_type'] for e in events])
        summary = storage().read_trace_summary(self.directory / 'summary.json')
        self.assertEqual(summary['num_steps'], 0)
        self.assertIsNone(summary['task_success'])

    def test_error_after_return_keeps_only_completed_transition(self):
        self.trace.record_step(0, [1], 0, False, {}, {'value': 1})
        self.trace.close(error=RuntimeError('next decision'))
        self.assertEqual(len(self.rows()), 1)
        events = list(storage().iter_trace_events(self.directory / 'steps.jsonl'))
        self.assertEqual([e['sequence'] for e in events], list(range(len(events))))
        self.assertEqual([e['event_type'] for e in events][-2:], ['exception', 'episode_end'])

    def test_observer_failure_does_not_poison_later_step_commits(self):
        failed = []

        def observer(event):
            if event['event_type'] == 'action_execution' and not failed:
                failed.append(True)
                raise RuntimeError('observer failed')

        self.trace.event_callback = observer
        self.trace.submit_actions(0, [1], [1], ['agent_0'])
        self.trace.record_step(0, [1], 0, False, {}, {'value': 1})
        self.trace.submit_actions(0, [2], [1], ['agent_0'])
        self.trace.record_step(0, [2], 1, True, {}, {'value': 2}, success=True)
        self.assertEqual([row['step'] for row in self.rows()], [1, 2])
        summary = storage().read_trace_summary(self.directory / 'summary.json')
        self.assertTrue(summary['task_success'])
        self.assertEqual(summary['logging_error_count'], 1)
        events = list(storage().iter_trace_events(self.directory / 'steps.jsonl'))
        self.assertEqual(sum(e['event_type'] == 'action_execution' for e in events), 2)

    def test_timeout_handoff_and_repeated_episode_identity(self):
        with self.trace.policy_step():
            self.recorder.begin_action('agent_0', 'move', {})
            self.recorder.record_event('action_timeout', agent_id='agent_0', handoff=False)
        self.trace.record_step(0, [1], 0, False, {}, {})
        with self.trace.policy_step():
            self.recorder.begin_action('agent_0', 'wait', {})
        self.trace.record_step(0, [0], 0, True, {}, {}, success=False)
        events = list(storage().iter_trace_events(self.directory / 'steps.jsonl'))
        names = [e['event_type'] for e in events]
        self.assertLess(names.index('action_timeout'), names.index('action_end'))
        self.trace.start_episode(0, SimpleNamespace(scene_id='scene', episode_id='same'))
        self.assertEqual(len(list((self.trace.run_dir / 'episodes').iterdir())), 2)

    def test_pause_reindex_keeps_original_environment_identity(self):
        self.trace.start_episode(1, SimpleNamespace(scene_id='other', episode_id='same'))
        second = self.trace._episodes[1].directory
        self.trace.record_step(0, [1], 0, False, {}, {})
        self.trace.pause_envs([0])
        self.trace.record_step(0, [2], 0, True, {}, {}, success=None)
        self.assertEqual(len(self.rows()), 1)
        first_summary = storage().read_trace_summary(self.directory / 'summary.json')
        self.assertEqual(first_summary['status'], 'paused')
        second_events = list(storage().iter_trace_events(second / 'steps.jsonl'))
        self.assertTrue(all(e['env_index'] == 1 for e in second_events))
        self.assertIsNone(storage().read_trace_summary(second / 'summary.json')['task_success'])
