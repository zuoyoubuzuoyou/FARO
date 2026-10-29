"""Offline CLI contract tests using literal, independently authored V3 rows."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / 'scripts/compact_trajectory.py'
STORAGE = ROOT / 'habitat-mas/habitat_mas/utils/trajectory_storage.py'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


class CompactTrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'original/episodes/episode_0061'
        self.source.mkdir(parents=True)
        self.output = self.base / 'compacted/episodes/episode_0061'
        self.identity = dict(run_id='original-run', scene_id='scene', episode_id='61',
                             evaluation_id='evaluation', env_index=0)
        self.events = [
            dict(self.identity, event_type='episode_start', sequence=0, state={'initial': True}, context={'config': 'shared camera metadata ' * 30}),
            dict(self.identity, event_type='native_detection', sequence=1,
                 detections=[{'label': 'cup', 'score': 0.0}], unknowns={'distance': None},
                 literal={'$trace': ['blob', 'literal-not-a-reference']}),
            dict(self.identity, event_type='step', sequence=2, env_step=1,
                 state={'position': [0, 1, 2], 'camera': {'hfov': 90}}, metrics={'success': False}),
            dict(self.identity, event_type='action_end', sequence=3, reason='boundary',
                 actual_observation={'media_path': 'media/frame.png'}),
            dict(self.identity, event_type='step', sequence=4, env_step=2,
                 state={'position': [1, 1, 2], 'camera': {'hfov': 90}}, metrics={'success': True}),
            dict(self.identity, event_type='diagnostic', sequence=5,
                 unknowns=['unobserved'], diagnostics={'raw': [None, False, 0, '']}),
        ]
        # Literal $trace data must be escaped in the stored fixture, independent
        # of the codec under test.
        encoded_detection = dict(self.events[1], literal={'$trace': ['literal',
                                  {'$trace': ['blob', 'literal-not-a-reference']}]})
        rows = [dict(schema_version=3, step=1, identity=self.identity,
                     state=self.events[2]['state'], metrics=self.events[2]['metrics'],
                     events=[encoded_detection, self.events[2], self.events[3]]),
                dict(schema_version=3, step=2, identity=self.identity,
                     state=self.events[4]['state'], metrics=self.events[4]['metrics'],
                     events=self.events[4:])]
        (self.source / 'steps.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        initial = {'camera': {'config': 'shared camera metadata ' * 30}, 'unknown': None,
                   'literal': {'$trace': ['blob', 'plain metadata literal']}}
        blob_bytes = json.dumps(self.events[0]['context'], sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256(blob_bytes).hexdigest()
        (self.source / 'payloads').mkdir()
        (self.source / 'payloads' / (digest + '.json')).write_bytes(blob_bytes)
        (self.source / 'payloads/unused.json').write_text('{"unused":"old orphan"}')
        self.metadata = dict(self.identity, schema_version=2, storage_schema_version=3,
                             files={'steps': 'steps.jsonl', 'summary': 'summary.json',
                                    'config': '../../config.json', 'media': 'media/'},
                             initial_state=initial, initial_event=self.events[0],
                             context={'native': {'unknown': None}, '$trace': ['blob', 'literal context']})
        stored_metadata = dict(self.metadata, initial_event=dict(self.events[0], context={'$trace': ['blob', digest]}))
        (self.source / 'episode.json').write_text(json.dumps(stored_metadata))
        self.summary = dict(self.identity, status='completed', num_steps=2,
                            final_state=self.events[4]['state'], files=self.metadata['files'])
        (self.source / 'summary.json').write_text(json.dumps(self.summary))
        (self.source / 'media').mkdir()
        (self.source / 'media/frame.png').write_bytes(b'fake-png-content\x00\x01')
        (self.source.parents[1] / 'config.json').write_text('{"seed":66,"native":true}')

    def command(self, output=None, extra=()):
        self.assertTrue(COMMAND.exists(), 'Offline compaction command is missing')
        command = load(COMMAND, 'compact_command')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return command.main(['--input', str(self.source), '--output', str(output or self.output), *extra])

    def test_round_trip_preserves_source_row_boundaries_metadata_media_and_provenance(self):
        before = manifest(self.base / 'original')
        self.assertEqual(self.command(), 0)
        storage = load(STORAGE, 'compact_contract_storage')
        self.assertEqual(list(storage.iter_trace_events(self.output)), self.events)
        self.assertEqual(storage.read_trace_summary(self.output), self.summary)
        decoded_metadata = json.loads((self.output / 'episode.json').read_text())
        decoded_metadata['initial_event'] = storage._Codec(self.output).decode(decoded_metadata['initial_event'])
        for key, value in self.metadata.items():
            self.assertEqual(decoded_metadata[key], value, key)
        self.assertEqual([len(json.loads(line)['events']) for line in (self.output / 'steps.jsonl').read_text().splitlines()], [3, 2])
        self.assertEqual((self.output / 'media/frame.png').read_bytes(), b'fake-png-content\x00\x01')
        self.assertEqual((self.output.parents[1] / 'config.json').read_bytes(), (self.source.parents[1] / 'config.json').read_bytes())
        self.assertFalse((self.output / 'payloads/unused.json').exists())
        self.assertEqual(manifest(self.base / 'original'), before)
        report = json.loads((self.output / 'compaction_report.json').read_text())
        self.assertTrue(report['verification']['source_hashes_unchanged'])
        self.assertTrue(report['verification']['events_equal'])
        self.assertTrue(report['verification']['summary_equal'])
        self.assertEqual(report['rows'], 2)
        self.assertEqual(report['events'], 6)
        self.assertEqual(report['input']['steps_bytes'], (self.source / 'steps.jsonl').stat().st_size)
        files = [p for p in self.output.rglob('*') if p.is_file() and p.name != 'compaction_report.json']
        self.assertEqual(report['output']['directory_bytes'], sum(p.stat().st_size for p in files))
        self.assertEqual(report['output']['file_count'], len(files))
        self.assertEqual(report['input']['sha256'], manifest(self.source))

    def test_refuses_existing_same_nested_and_ancestor_outputs_without_writes(self):
        existing = self.base / 'existing'
        existing.mkdir()
        (existing / 'keep').write_text('precious')
        before = manifest(self.base)
        for output in [self.source, existing, self.source / 'copy', self.source.parent]:
            with self.subTest(output=output):
                self.assertNotEqual(self.command(output), 0)
                self.assertEqual(manifest(self.base), before)

    def test_refuses_pending_or_truncated_evidence_without_dropping_it(self):
        pending = self.source / 'pending.json'
        pending.write_text('{"events":[{"event_type":"native_detection","unknown":null}]}')
        before = manifest(self.source)
        self.assertNotEqual(self.command(), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(manifest(self.source), before)
        pending.unlink()
        steps = self.source / 'steps.jsonl'
        steps.write_bytes(steps.read_bytes() + b'{"uncommitted":')
        before = manifest(self.source)
        self.assertNotEqual(self.command(), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(manifest(self.source), before)

    def test_report_cannot_overwrite_source_or_existing_external_file(self):
        before = manifest(self.base)
        for report in [self.source / 'episode.json', self.source / 'new-report.json', self.source.parents[1] / 'config.json']:
            with self.subTest(report=report):
                self.assertNotEqual(self.command(extra=['--report', str(report)]), 0)
                self.assertFalse(self.output.exists())
                self.assertEqual(manifest(self.base), before)

    def test_corrupt_source_cleanup_keeps_preexisting_config_and_source(self):
        (self.source / 'steps.jsonl').write_text('{"schema_version":3,"step":1}\n')
        self.output.parents[1].mkdir(parents=True)
        (self.output.parents[1] / 'config.json').write_text('existing unrelated config')
        before = manifest(self.base)
        self.assertNotEqual(self.command(), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(manifest(self.base), before)

    def test_refuses_dangling_output_or_report_symlinks(self):
        dangling_output = self.base / 'dangling-output'
        dangling_target = self.base / 'not-created'
        dangling_output.symlink_to(dangling_target, target_is_directory=True)
        self.assertNotEqual(self.command(dangling_output), 0)
        self.assertTrue(dangling_output.is_symlink())
        self.assertFalse(dangling_target.exists())
        dangling_report = self.base / 'dangling-report'
        dangling_report.symlink_to(self.base / 'report-target')
        self.assertNotEqual(self.command(extra=['--report', str(dangling_report)]), 0)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.base / 'report-target').exists())

    def test_refuses_symlinked_evidence_without_copying_external_files(self):
        outside = self.base / 'private-file'
        outside.write_bytes(b'external evidence must not be read or copied')
        (self.source / 'media/external').symlink_to(outside)
        self.assertNotEqual(self.command(), 0)
        self.assertFalse(self.output.exists())
        self.assertEqual(outside.read_bytes(), b'external evidence must not be read or copied')

    def test_explicit_report_and_run_provenance_have_exact_byte_totals(self):
        (self.source.parents[1] / 'run.json').write_text('{"original_run_id":"original-run"}')
        report_path = self.base / 'reports/compaction.json'
        self.assertEqual(self.command(extra=['--report', str(report_path)]), 0)
        self.assertTrue(report_path.is_file())
        self.assertFalse((self.output / 'compaction_report.json').exists())
        self.assertEqual((self.output.parents[1] / 'run.json').read_bytes(),
                         (self.source.parents[1] / 'run.json').read_bytes())
        report = json.loads(report_path.read_text())
        for key, directory in [('input', self.source), ('output', self.output)]:
            run_files = [p for p in directory.parents[1].rglob('*') if p.is_file()]
            self.assertEqual(report[key]['run_directory_bytes'], sum(p.stat().st_size for p in run_files))
            self.assertEqual(report[key]['run_file_count'], len(run_files))
        self.assertEqual(len(report['linked_run_files']), 2)

    def test_verification_failure_removes_only_new_output_and_copied_run_files(self):
        # Valid JSON and events, but the canonical row contradicts the step.
        # Repacking gets through copying; the public source reader then rejects.
        rows = [json.loads(line) for line in (self.source / 'steps.jsonl').read_text().splitlines()]
        rows[0]['metrics'] = {'success': 'corrupt canonical metrics'}
        (self.source / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.output.parents[1].mkdir(parents=True)
        unrelated = self.output.parents[1] / 'keep.bin'
        unrelated.write_bytes(b'preexisting')
        before = manifest(self.base)
        self.assertNotEqual(self.command(), 0)
        self.assertFalse(self.output.exists())
        self.assertFalse((self.output.parents[1] / 'config.json').exists())
        self.assertEqual(manifest(self.base), before)

    def test_refuses_canonical_state_only_present_on_row(self):
        rows = [json.loads(line) for line in (self.source / 'steps.jsonl').read_text().splitlines()]
        del rows[0]['events'][1]['state']
        self.assert_row_only_evidence_refused(rows)

    def test_refuses_replay_context_only_present_on_row(self):
        rows = [json.loads(line) for line in (self.source / 'steps.jsonl').read_text().splitlines()]
        rows[0]['replay_context'] = {'model_memory': {'unmirrored': 'retain this evidence'}}
        self.assert_row_only_evidence_refused(rows)

    def test_refuses_unknown_diagnostic_only_present_on_row(self):
        rows = [json.loads(line) for line in (self.source / 'steps.jsonl').read_text().splitlines()]
        rows[0]['native_row_diagnostic'] = {'unknown': None, 'actual_detections': ['cup']}
        self.assert_row_only_evidence_refused(rows)

    def test_refuses_extra_row_identity_evidence(self):
        rows = [json.loads(line) for line in (self.source / 'steps.jsonl').read_text().splitlines()]
        rows[0]['identity']['sensor_capture_id'] = 'native-123'
        self.assert_row_only_evidence_refused(rows)

    def test_refuses_row_identity_type_change(self):
        rows = [json.loads(line) for line in (self.source / 'steps.jsonl').read_text().splitlines()]
        rows[0]['identity']['env_index'] = False
        self.assert_row_only_evidence_refused(rows)

    def assert_row_only_evidence_refused(self, rows):
        (self.source / 'steps.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        # These are valid V3 inputs; the existing reader intentionally accepts
        # extra canonical fields and diagnostics without event mirrors.
        storage = load(STORAGE, 'row_evidence_storage')
        self.assertEqual(len(list(storage.iter_trace_events(self.source))), 6)
        before = manifest(self.base)
        self.assertNotEqual(self.command(), 0, 'Compaction must not silently drop row evidence')
        self.assertFalse(self.output.exists())
        self.assertEqual(manifest(self.base), before)


if __name__ == '__main__':
    unittest.main()
