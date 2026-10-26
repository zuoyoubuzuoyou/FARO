"""Dependency-free access to expanded trajectory evidence in recorder tests."""
import importlib.util
from pathlib import Path


_path = Path(__file__).resolve().parents[1] / 'habitat-mas/habitat_mas/utils/trajectory_storage.py'
_spec = importlib.util.spec_from_file_location('_trajectory_test_storage', _path)
_storage = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_storage)
iter_trace_events = _storage.iter_trace_events
read_trace_summary = _storage.read_trace_summary


def read_episode_events(directory):
    return [list(iter_trace_events(path))
            for path in sorted(Path(directory).rglob('*.jsonl'))]


def read_events(directory):
    return [event for episode in read_episode_events(directory) for event in episode]
