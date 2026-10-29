"""Lossless compact storage contracts; fixtures are expanded JSON evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / 'habitat-mas/habitat_mas/utils/trajectory_storage.py'


class CompactStepStorageTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('compact_test_storage', MODULE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.identity = dict(run_id='run', scene_id='scene', episode_id='61', evaluation_id='eval', env_index=0)
        self.store = self.m.StepTraceStore(self.d, self.identity)

    def event(self, sequence, step, kind='step', **data):
        return dict(self.identity, schema_version=3, event=kind, event_type=kind,
                    sequence=sequence, step=step, env_step=step, agent_id='agent_0',
                    event_id='unique-%s' % sequence, timestamp='time-%s' % sequence, **data)

    def rows(self):
        return [json.loads(line) for line in (self.d / 'steps.jsonl').read_text().splitlines()]

    def test_static_and_unchanged_state_shares_payloads_but_changed_frames_stay_inline(self):
        # Losing subtree sharing bloats rows; sharing unique wrappers explodes file counts.
        camera = {'rgb': {'resolution': [256, 256], 'description': 'static camera ' * 40}}
        navigation = {'active': False, 'diagnostic': 'inactive ' * 60}
        events = []
        for step in range(1, 9):
            state = {'agents': {'agent_0': {'camera_sensors': camera, 'navigation': navigation,
                     'detection': {'env_step': step, 'ids': [step], 'source': 'sensor ' * 80},
                     'goal_coverage': {'env_step': step, 'distance': step / 10}}},
                     'num_steps': step, 'current_value': step}
            event = self.event(step, step, state=state)
            events.append(event)
            self.store.append(event)
            self.store.commit_step()
        rows = self.rows()
        self.assertIn('$trace', rows[1]['state']['agents']['agent_0']['camera_sensors'])
        self.assertIn('$trace', rows[1]['state']['agents']['agent_0']['navigation'])
        for step, row in enumerate(rows, 1):
            self.assertEqual(row['state']['agents']['agent_0']['detection']['env_step'], step)
            self.assertEqual(row['state']['num_steps'], step)
        self.assertLessEqual(len(list((self.d / 'payloads').glob('*.json'))), 4)
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)

    def test_same_row_metrics_reused_and_summary_expands_exactly(self):
        metrics = {'success': False, 'stages': {str(i): i / 10 for i in range(40)}}
        state = {'task_progress': metrics, 'small': {'ok': False}}
        event = self.event(1, 1, state=state, metrics=metrics)
        self.store.append(event)
        self.store.commit_step()
        row = self.rows()[0]
        self.assertEqual(row['state']['task_progress'].get('$trace', [])[:1], ['local'])
        self.assertEqual(row['state']['small'], {'ok': False})
        summary = dict(self.identity, final_state=state, final_metrics=metrics)
        self.store.write_summary(summary)
        self.assertEqual(self.m.read_trace_summary(self.d), summary)
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [event])

    def test_sparse_events_keep_inspectable_fields_and_validate_context(self):
        event = self.event(1, 1, state={'value': 2})
        self.store.append(event)
        self.store.commit_step()
        row = self.rows()[0]
        raw = row['events'][0]
        self.assertNotIn('scene_id', raw)
        self.assertNotIn('schema_version', raw)
        for key in ('event_type', 'agent_id', 'sequence', 'env_step', 'event_id', 'timestamp'):
            self.assertEqual(raw[key], event[key])
        self.assertEqual(self.store.codec.decode(raw, row), event)
        with self.assertRaises(ValueError):
            self.store.codec.decode(raw)
        foreign = copy.deepcopy(row)
        foreign['identity']['episode_id'] = 'foreign'
        with self.assertRaises(ValueError):
            self.store.codec.decode(raw, foreign)
        wrong_step = copy.deepcopy(row)
        wrong_step['step'] = 2
        with self.assertRaises(ValueError):
            self.store.codec.decode(raw, wrong_step)

    def test_nonstandard_envelopes_and_literal_event_markers_remain_exact(self):
        standard = self.event(1, 1, literal={'$trace': ['event'], 'event_type': 'step'})
        missing = self.event(2, 1, 'custom')
        del missing['scene_id']
        alias = self.event(3, 1, 'custom')
        alias['event'] = 'different'
        schema = self.event(4, 1, 'custom')
        schema['schema_version'] = 2
        events = [standard, missing, alias, schema]
        for event in events:
            self.store.append(event)
        self.store.commit_step()
        row = self.rows()[0]
        self.assertNotIn('scene_id', row['events'][0])
        self.assertIn('schema_version', row['events'][1])
        self.assertIn('scene_id', row['events'][2])
        self.assertEqual(row['events'][3]['schema_version'], 2)
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)

    def test_changed_camera_and_repeated_literal_state_expand_without_aliasing(self):
        literal = {'$trace': ['event', 'literal-user-data'], 'note': 'literal state ' * 30}
        events = []
        for number in (1, 2, 3):
            state = {'agents': {'agent_0': {'camera_sensors': {'description': 'camera ' * 40,
                     'width': 256 if number < 3 else 512}}}, 'literal': literal}
            event = self.event(number, number, state=state)
            events.append(copy.deepcopy(event))
            self.store.append(event)
            event['state']['literal']['note'] = 'caller changed the original'
            # Restore shared fixture after proving append takes a snapshot.
            literal['note'] = 'literal state ' * 30
            self.store.commit_step()
        expanded = list(self.m.iter_trace_events(self.d))
        self.assertEqual(expanded, events)
        expanded[0]['state']['literal']['note'] = 'reader mutation'
        self.assertEqual(list(self.m.iter_trace_events(self.d)), events)
        self.assertNotEqual(self.rows()[0]['state']['agents']['agent_0']['camera_sensors'],
                            self.rows()[2]['state']['agents']['agent_0']['camera_sensors'])

    def test_sparse_event_root_cannot_collapse_into_same_row_state_reference(self):
        custom = self.event(0, 1, 'custom', note='event evidence ' * 30)
        sparse = {key: value for key, value in custom.items()
                  if key not in self.identity and key not in ('schema_version', 'step', 'event')}
        step = self.event(1, 1, state=sparse)
        self.store.append(custom)
        self.store.append(step)
        self.store.commit_step()
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [custom, step])
        self.assertEqual(self.rows()[0]['events'][0]['event_type'], 'custom')

    def test_precreated_episode_metadata_advertises_writer_features_without_losing_custom_data(self):
        other = self.d / 'precreated'
        other.mkdir()
        metadata = dict(self.identity, schema_version=3, custom={'keep': 4}, storage_features=['old_feature'])
        (other / 'episode.json').write_text(json.dumps(metadata))
        self.m.StepTraceStore(other, self.identity)
        actual = json.loads((other / 'episode.json').read_text())
        self.assertIn('row_event_defaults_v1', actual['storage_features'])
        self.assertIn('shared_state', actual['storage_features'])
        self.assertIn('old_feature', actual['storage_features'])
        self.assertEqual(actual['custom'], {'keep': 4})

    def test_tiny_duplicate_values_stay_inline_instead_of_growing_references(self):
        event = self.event(1, 1, state={'ok': False}, metrics={'ok': False})
        self.store.append(event)
        self.store.commit_step()
        row = self.rows()[0]
        self.assertEqual(row['events'][0]['state'], {'ok': False})
        self.assertEqual(row['events'][0]['metrics'], {'ok': False})
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [event])

    def test_shared_payload_corruption_and_missing_files_are_checked_on_every_read(self):
        camera = {'rgb': {'description': 'fixed camera ' * 50}}
        event = self.event(1, 1, state={'agents': {'agent_0': {'camera_sensors': camera}}})
        self.store.append(event)
        self.store.commit_step()
        row = self.rows()[0]
        codec = self.m._Codec(self.d)
        ref = row['state']['agents']['agent_0']['camera_sensors']
        self.assertEqual(codec.decode(ref, row), camera)
        payload = self.d / 'payloads' / (ref['$trace'][1] + '.json')
        contents = payload.read_bytes()
        payload.write_text('{}')
        with self.assertRaises(ValueError):
            codec.decode(ref, row)
        payload.write_bytes(contents)
        payload.unlink()
        with self.assertRaises(ValueError):
            codec.decode(ref, row)

    def test_malformed_sparse_events_and_cyclic_locals_are_rejected(self):
        event = self.event(1, 1, state={'description': 'state ' * 60})
        self.store.append(event)
        self.store.commit_step()
        row = self.rows()[0]
        original = row['events'][0]
        for key, value in [('scene_id', 'foreign'), ('schema_version', 3),
                           ('event', 'step'), ('env_step', 2), ('sequence', True)]:
            raw = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.store.codec.decode(raw, row)
        missing = dict(original)
        del missing['event_type']
        with self.assertRaises(ValueError):
            self.store.codec.decode(missing, row)
        reference = {'$trace': ['local', ['state'], '0' * 64]}
        row['state'] = reference
        with self.assertRaisesRegex(ValueError, 'Cyclic'):
            self.store.codec.decode(reference, row)

    def test_summary_step_reference_cache_is_bounded_detached_and_invalidates_on_edit(self):
        events = [self.event(i, i, state={'items': list(range(100)), 'value': i}) for i in (1, 2)]
        for event in events:
            self.store.append(event)
            self.store.commit_step()
        ref = {'$trace': ['step', 2, ['state', 'items'], self.m._hash(list(range(100)))]}
        codec = self.m._Codec(self.d)
        with patch.object(self.m, '_rows', wraps=self.m._rows) as rows:
            first = codec.decode(ref)
            first[0] = -1
            self.assertEqual(codec.decode(ref), list(range(100)))
            self.assertEqual(rows.call_count, 1, 'Repeated summary leaves must share one row scan')
        records = self.rows()
        records[1]['state']['items'][0] = -1
        (self.d / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
        with self.assertRaises(ValueError):
            codec.decode(ref)

    def test_pending_and_truncated_commit_need_no_row_defaults(self):
        event = self.event(1, 1, state={'camera_sensors': {'text': 'camera ' * 80}})
        self.store.append(event)
        pending = (self.d / 'pending.json').read_bytes()
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [event])
        self.store.commit_step()
        raw = self.rows()[0]['events'][0]
        self.assertNotIn('scene_id', raw)
        committed = (self.d / 'steps.jsonl').read_bytes()
        (self.d / 'steps.jsonl').write_bytes(committed[:-1])
        (self.d / 'pending.json').write_bytes(pending)
        self.assertEqual(list(self.m.iter_trace_events(self.d)), [event])


if __name__ == '__main__':
    unittest.main()
