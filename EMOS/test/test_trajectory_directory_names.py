"""Readable trace directories keep identity and collision safety."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

from test_trajectory_recorder import load_recorder


class FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        value = cls(2026, 10, 25, 20, 5, 6, tzinfo=timezone.utc)
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)


class TrajectoryDirectoryNameTests(unittest.TestCase):
    def setUp(self):
        self.recorder = load_recorder()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.patch = patch.object(self.recorder, 'datetime', FrozenDatetime)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_run_names_use_shanghai_time_and_atomic_collision_suffixes(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            traces = list(executor.map(lambda _: self.recorder.EvaluationTrace(self.temp.name), range(4)))
        for trace in traces:
            self.addCleanup(trace.close)
        self.assertEqual(sorted(t.run_dir.name for t in traces), [
            'run_20261026_040506', 'run_20261026_040506_02',
            'run_20261026_040506_03', 'run_20261026_040506_04'])
        self.assertEqual(len({t.run_id for t in traces}), 4)
        for trace in traces:
            UUID(trace.run_id)
            config = json.loads((trace.run_dir / 'config.json').read_text())
            self.assertEqual(config['run_id'], trace.run_id)

    def test_episode_names_repeats_and_config_paths_keep_unique_json_identity(self):
        with self.recorder.EvaluationTrace(self.temp.name) as trace:
            for index, identifier in enumerate([2, '0002', 'kitchen', 'kitchen', '../a/b', '..', 'a/b', 'a_b']):
                trace.start_episode(index, SimpleNamespace(scene_id='s', episode_id=identifier))
            episodes = trace.config['episodes']
            self.assertEqual([Path(e['directory']).name for e in episodes], [
                'episode_0002', 'episode_0002_attempt_02', 'episode_kitchen',
                'episode_kitchen_attempt_02', 'episode__a_b', 'episode_episode',
                'episode_a_b', 'episode_a_b_attempt_02'])
            self.assertEqual(len({e['evaluation_id'] for e in episodes}), 8)
            for item in episodes:
                directory = trace.run_dir / item['directory']
                self.assertEqual(directory.parent, trace.run_dir / 'episodes')
                snapshot = json.loads((directory / 'episode.json').read_text())
                self.assertEqual(snapshot['evaluation_id'], item['evaluation_id'])
                self.assertEqual(snapshot['run_id'], trace.run_id)
                UUID(snapshot['evaluation_id'])

    def test_simultaneous_same_episode_attempts_across_environments(self):
        with self.recorder.EvaluationTrace(self.temp.name) as trace:
            def create(index):
                return self.recorder._EpisodeWriter(
                    trace, index, SimpleNamespace(scene_id='s', episode_id='2'), None, None)
            with ThreadPoolExecutor(max_workers=4) as executor:
                writers = list(executor.map(create, range(4)))
            try:
                self.assertEqual(sorted(w.directory.name for w in writers), [
                    'episode_0002', 'episode_0002_attempt_02',
                    'episode_0002_attempt_03', 'episode_0002_attempt_04'])
                self.assertEqual({w.identity['env_index'] for w in writers}, set(range(4)))
                self.assertEqual(len({w.identity['evaluation_id'] for w in writers}), 4)
            finally:
                for writer in writers:
                    writer.finish('interrupted')

    def test_long_unsafe_id_is_sanitized_without_escaping_episode_parent(self):
        with self.recorder.EvaluationTrace(self.temp.name) as trace:
            trace.start_episode(0, SimpleNamespace(scene_id='s', episode_id='../' + 'x/' * 90))
            directory = trace._episodes[0].directory
            self.assertEqual(directory.parent, trace.run_dir / 'episodes')
            self.assertTrue(directory.name.startswith('episode_'))
            self.assertLessEqual(len(directory.name), len('episode_') + 80)
            numeric_id = '1' * 300
            trace.start_episode(1, SimpleNamespace(scene_id='s', episode_id=numeric_id))
            numeric_directory = trace._episodes[1].directory
            self.assertLessEqual(len(numeric_directory.name), len('episode_') + 80)
            snapshot = json.loads((numeric_directory / 'episode.json').read_text())
            self.assertEqual(snapshot['episode_id'], numeric_id)
