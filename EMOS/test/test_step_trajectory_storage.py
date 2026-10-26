"""Storage contract tests with hand-authored expanded events."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'habitat-mas/habitat_mas/utils/trajectory_storage.py'


class StepStorageTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'Step trajectory storage implementation is missing')
        spec = importlib.util.spec_from_file_location('standalone_storage', MODULE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.identity = dict(run_id='r', scene_id='s', episode_id='78', evaluation_id='e', env_index=0)
        self.store = self.m.StepTraceStore(self.d, self.identity)

    def event(self, seq, kind, **data):
        return dict(self.identity, schema_version=2, sequence=seq, event_id=str(seq), event_type=kind, **data)

    def test_exact_expansion_order_state_and_literal_refs(self):
        state = {'task_progress': {'success': False}, 'executed_action': {'name': 'move'}, 'unknown': None}
        events = [self.event(0, 'episode_start', state={'initial': True}),
                  self.event(1, 'llm_request', messages=[{'content': 'long text ' * 80}], literal={'$trace': ['blob', 'bad']}),
                  self.event(2, 'env_action_submit', action=[1, 0]),
                  self.event(3, 'step', env_step=1, action=[1, 0], reward=0, done=False, metrics={'success': True}, state=state),
                  self.event(4, 'action_end', result=state)]
        for event in events:
            self.store.append(event)
        self.store.commit_step()
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)
        rows = (self.d / 'steps.jsonl').read_text().splitlines()
        self.assertEqual(len(rows), 1)
        row = json.loads(rows[0])
        self.assertEqual(row['step'], 1)
        self.assertEqual(row['schema_version'], 3)
        self.assertEqual(row['state'], state)
        self.assertEqual(rows[0].count('executed_action'), 1)
        summary = dict(self.identity, status='interrupted', task_success=None, final_state=state)
        self.store.write_summary(summary)
        self.assertEqual(self.m.read_trace_summary(self.d), summary)

    def test_nested_replay_refs_expand_through_blob_root_without_redecoding_literals(self):
        literal = {'$trace': ['blob', 'literal-user-data'], 'content': 'literal ' * 100}
        messages = [{'role': 'user', 'content': 'request ' * 100}, literal]
        tools = [{'name': 'move', 'description': 'description ' * 100}]
        context = {'model_memory': {'chat_history': [messages]},
                   'model_config': {'openai_tools': tools}}
        events = [self.event(0, 'llm_request', messages=messages, tools=tools,
                             selected_message=literal),
                  self.event(1, 'agent_action', replay_context=context),
                  self.event(2, 'step', env_step=1, state={})]
        for event in events:
            self.store.append(event)
        self.store.commit_step()
        row = json.loads((self.d / 'steps.jsonl').read_text())
        self.assertEqual(row['replay_context']['$trace'][0], 'blob')
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)

    def test_summary_nested_step_refs_expand_blob_root_and_keep_literal_data(self):
        literal = {'$trace': ['local', ['state'], 'literal-user-hash'],
                   'content': 'literal ' * 100}
        tools = [{'name': 'move', 'description': 'description ' * 100}]
        context = {'model_memory': {'chat_history': [[literal]]},
                   'model_config': {'openai_tools': tools}}
        self.store.append(self.event(0, 'agent_action', replay_context=context))
        self.store.append(self.event(1, 'step', env_step=1, state={}))
        self.store.commit_step()
        summary = {'last_message': literal, 'tools': tools}
        self.store.write_summary(summary)
        encoded = json.loads((self.d / 'summary.json').read_text())
        self.assertEqual(encoded['last_message']['$trace'][0], 'step')
        self.assertEqual(self.m.read_trace_summary(self.d), summary)

    def test_startup_error_and_close_do_not_invent_step(self):
        events = [self.event(0, 'episode_start'), self.event(1, 'llm_error', error='timeout')]
        for e in events:
            self.store.append(e)
        self.store.commit_step()
        self.store.close()
        self.store.close()
        self.assertEqual((self.d / 'steps.jsonl').read_text(), '')
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)
        self.assertEqual(list(self.m.iter_trace_events(self.d, include_pending=False)), events[:1])

    def test_commit_cleanup_crash_does_not_duplicate(self):
        e = self.event(0, 'step', env_step=1, state={})
        self.store.append(e)
        pending = (self.d / 'pending.json').read_bytes()
        self.store.commit_step()
        (self.d / 'pending.json').write_bytes(pending)
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [e])

    def test_partial_final_line_ignored_but_middle_corruption_raises(self):
        e = self.event(0, 'step', env_step=1)
        self.store.append(e)
        self.store.commit_step()
        with (self.d / 'steps.jsonl').open('a') as f:
            f.write('{"schema_version":3')
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [e])
        with (self.d / 'steps.jsonl').open('a') as f:
            f.write('\n{}\n')
        with self.assertRaises(ValueError):
            list(self.m.iter_trace_events(self.d))

    def test_mixed_identity_and_sequence_rejected(self):
        with self.assertRaises(ValueError):
            self.store.append(self.event(0, 'step', episode_id='foreign'))
        self.store.append(self.event(1, 'llm_request'))
        with self.assertRaises(ValueError):
            self.store.append(self.event(1, 'llm_response'))

    def test_payload_hash_and_missing_payload_raise(self):
        self.store.append(self.event(0, 'llm_request', messages=[{'content': 'x' * 2000}]))
        self.store.append(self.event(1, 'step', env_step=1))
        self.store.commit_step()
        paths = list((self.d / 'payloads').glob('*.json'))
        self.assertTrue(paths)
        original = {p: p.read_bytes() for p in paths}
        for p in paths:
            p.write_text('{}')
        with self.assertRaises(ValueError):
            list(self.m.iter_trace_events(self.d))
        for p, data in original.items():
            p.write_bytes(data)
        for p in paths:
            p.unlink()
        with self.assertRaises(ValueError):
            list(self.m.iter_trace_events(self.d))

    def test_summary_literal_control_shape_is_never_interpreted(self):
        e = self.event(0, 'step', state={'pose': [1, 2, 3]})
        self.store.append(e)
        self.store.commit_step()
        summary = {'literal': {'$trace': ['local', ['state'], 'not-a-real-hash']}}
        self.store.write_summary(summary)
        self.assertEqual(self.m.read_trace_summary(self.d), summary)

    def test_tampered_local_reference_and_foreign_row_raise(self):
        self.store.append(self.event(0, 'step', state={'pose': [1, 2, 3]}))
        self.store.commit_step()
        path = self.d / 'steps.jsonl'
        row = json.loads(path.read_text())
        row['state']['pose'][0] = 99
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaises(ValueError):
            list(self.m.iter_trace_events(self.d))
        row['state']['pose'][0] = 1
        row['identity']['evaluation_id'] = 'other'
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaises(ValueError):
            list(self.m.iter_trace_events(self.d))

    def test_payload_reference_cannot_escape_directory(self):
        self.store.append(self.event(0, 'llm_request'))
        pending = json.loads((self.d / 'pending.json').read_text())
        pending['events'] = [{'$trace': ['blob', '../episode']}]
        (self.d / 'pending.json').write_text(json.dumps(pending))
        with self.assertRaises(ValueError):
            list(self.m.iter_trace_events(self.d))

    def test_multiple_steps_preserve_missing_fields_and_message_history(self):
        message = {'role': 'assistant', 'content': 'reasoning ' * 100}
        events = [self.event(0, 'llm_response', message=message, raw_response={'message': message}),
                  self.event(1, 'step', env_step=1, state=None, metrics={'success': False}),
                  self.event(2, 'replay_context', replay_context={'agents': {'a': {'history': [[message], []]}}}),
                  self.event(3, 'step', env_step=2, done=True, state={'success': None})]
        for e in events[:2]:
            self.store.append(e)
        self.store.commit_step()
        for e in events[2:]:
            self.store.append(e)
        self.store.commit_step()
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)
        self.assertEqual(len((self.d / 'steps.jsonl').read_text().splitlines()), 2)
        # Identical long content appears once among content-addressed bodies.
        bodies = ''.join(p.read_text() for p in (self.d / 'payloads').glob('*.json'))
        self.assertEqual(bodies.count('reasoning ' * 100), 1)

    def test_payload_directory_symlink_cannot_escape_episode(self):
        self.store.append(self.event(0, 'llm_request', messages=[{'content': 'x' * 2000}]))
        self.store.append(self.event(1, 'step', env_step=1))
        self.store.commit_step()
        payloads = self.d / 'payloads'
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / 'payloads'
            payloads.rename(target)
            payloads.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                list(self.m.iter_trace_events(self.d))

    def test_legacy_utf8_bom_is_accepted(self):
        path = self.d / 'legacy.jsonl'
        path.write_text('\ufeff{"event":"step","step":1}\n', encoding='utf-8')
        self.assertEqual(list(self.m.iter_trace_events(path)), [{'event': 'step', 'step': 1}])

    def test_pending_and_committed_payloads_have_no_unreachable_files(self):
        state = {'robot_positions': list(range(400)), 'task_progress': {'success': False}}
        events = [self.event(0, 'env_action_submit', state=state),
                  self.event(1, 'llm_request', messages=[{'role': 'user', 'content': 'context ' * 200}]),
                  self.event(2, 'step', env_step=1, state=state),
                  self.event(3, 'action_end', result=state)]
        for e in events:
            self.store.append(e)
        self.store.commit_step()
        reachable = set()
        def visit(value):
            if isinstance(value, dict):
                control = value.get('$trace')
                if isinstance(control, list) and control and control[0] == 'blob':
                    digest = control[1]
                    if digest not in reachable:
                        reachable.add(digest)
                        visit(json.loads((self.d / 'payloads' / (digest + '.json')).read_text()))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(json.loads((self.d / 'episode.json').read_text()))
        for line in (self.d / 'steps.jsonl').read_text().splitlines():
            visit(json.loads(line))
        existing = {p.stem for p in (self.d / 'payloads').glob('*.json')}
        self.assertEqual(existing, reachable, 'Pending must not leave unreachable payload copies')
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)

    def test_compact_repeated_state_fixture_is_smaller_with_inline_event_envelopes(self):
        state = {'positions': [[i, i + 0.25, i + 0.5] for i in range(200)]}
        events = [self.event(i, kind, state=state, env_step=1, timestamp='2026-10-25T00:00:00Z',
                             action_id='action-identifier', agent_id='agent_0',
                             missing_fields={'decision_step': 'not applicable'},
                             elapsed_seconds=0.001)
                  for i, kind in enumerate(['env_action_submit', 'step', 'action_end'])]
        original_size = sum(len(json.dumps(e, separators=(',', ':')).encode()) + 1 for e in events)
        for e in events:
            self.store.append(e)
        self.store.commit_step()
        self.store.write_summary({'status': 'completed', 'final_state': state})
        stored_size = sum(p.stat().st_size for p in self.d.rglob('*') if p.is_file())
        self.assertLess(stored_size, original_size)
        row = json.loads((self.d / 'steps.jsonl').read_text())
        self.assertEqual([e.get('event_type') for e in row['events']],
                         ['env_action_submit', 'step', 'action_end'])
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)

    def test_malformed_step_and_pending_packets_raise_value_error(self):
        self.store.append(self.event(0, 'step', env_step=1))
        self.store.commit_step()
        path = self.d / 'steps.jsonl'
        original = json.loads(path.read_text())
        malformed = [42, dict(original, events=[42]), dict(original, events=None),
                     dict(original, identity=[]), dict(original, step=True)]
        for row in malformed:
            with self.subTest(row=row):
                path.write_text(json.dumps(row) + '\n')
                with self.assertRaises(ValueError):
                    list(self.m.iter_trace_events(self.d))
        path.write_text(json.dumps(original) + '\n')
        for pending in [42, {'schema_version': 3, 'step': 2, 'identity': {}, 'events': [42]},
                        {'schema_version': 3, 'step': 2, 'identity': {}, 'events': None}]:
            with self.subTest(pending=pending):
                (self.d / 'pending.json').write_text(json.dumps(pending))
                with self.assertRaises(ValueError):
                    list(self.m.iter_trace_events(self.d))

    def test_pending_returned_step_must_be_contiguous_and_aliases_agree(self):
        self.store.append(self.event(0, 'step', env_step=1))
        self.store.commit_step()
        malformed = [[self.event(1, 'step', env_step=999)],
                     [self.event(1, 'step', env_step=2, step=999)],
                     [self.event(1, 'step', env_step=2), self.event(2, 'step', env_step=2)]]
        for events in malformed:
            with self.subTest(events=events):
                pending = {'schema_version': 3, 'step': 2, 'identity': self.identity, 'events': events}
                (self.d / 'pending.json').write_text(json.dumps(pending))
                with self.assertRaises(ValueError):
                    list(self.m.iter_trace_events(self.d))

    def test_committed_step_aliases_must_agree(self):
        self.store.append(self.event(0, 'step', env_step=1, step=99))
        with self.assertRaises(ValueError):
            self.store.commit_step()

    def test_interrupted_utf8_final_line_preserves_committed_prefix(self):
        e = self.event(0, 'step', env_step=1)
        self.store.append(e)
        self.store.commit_step()
        with (self.d / 'steps.jsonl').open('ab') as stream:
            stream.write(b'{"message":"\xe4')
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [e])

    def test_legacy_jsonl_is_unchanged(self):
        events = [{'event': 'step', 'step': 1}, self.event(0, 'step', state=None)]
        path = self.d / 'legacy.jsonl'
        path.write_text(''.join(json.dumps(e) + '\n' for e in events))
        self.assertEqual(list(self.m.iter_trace_events(path)), events)


if __name__ == '__main__':
    unittest.main()
