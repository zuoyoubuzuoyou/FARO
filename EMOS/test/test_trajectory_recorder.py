"""Dependency-free tests for evaluation trajectory recording."""

from trajectory_test_utils import read_episode_events

import dataclasses
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "habitat-mas/habitat_mas/utils/trajectory.py"
)


def load_recorder():
    assert MODULE_PATH.is_file(), "The standalone recorder module is missing"
    name = "_standalone_trajectory_recorder"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def read_events(directory):
    return read_episode_events(directory)


def episode(scene="scene.glb", identifier="episode-1"):
    return SimpleNamespace(scene_id=scene, episode_id=identifier)


class TrajectoryRecorderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output_dir = Path(self.temp.name)

    def test_events_are_flushed_and_completed_episode_has_summary(self):
        recorder = load_recorder()
        with recorder.EvaluationTrace(
            self.output_dir, metadata={"experiment": "餐具", "seed": 7}
        ) as trace:
            metadata = json.loads((trace.run_dir / "config.json").read_text())
            self.assertEqual(metadata["metadata"]["experiment"], "餐具")
            self.assertEqual(metadata["schema_version"], 3)
            trace.start_episode(
                0, episode(), context={"instruction": "move"},
                state={"position": [0, 1, 2]},
            )
            initial = read_events(trace.run_dir)[0]
            self.assertEqual(initial[0]["event"], "episode_start")
            self.assertEqual(initial[0]["step"], 0)
            self.assertEqual(initial[0]["context"], {"instruction": "move"})
            self.assertEqual(initial[0]["state"], {"position": [0, 1, 2]})

            recorder.record_event("outside", ignored=object())
            with trace.policy_step():
                recorder.record_event("llm_request", prompt="hello")
                recorder.record_event("decision", action="forward")
            # Read before either the episode or the run is closed.
            events = read_events(trace.run_dir)[0]
            self.assertEqual(len(events), 3)
            self.assertEqual([item["step"] for item in events], [0, 1, 1])
            trace.record_step(
                0, {"action": [1]}, 1.25, False, {"success": False},
                {"position": [1, 1, 2]},
            )
            with trace.policy_step():
                recorder.record_event("decision", action="stop")
            trace.record_step(0, "stop", -0.25, True, {"success": True}, {})
            events = read_events(trace.run_dir)[0]
            self.assertEqual(
                [item["event"] for item in events],
                ["episode_start", "llm_request", "decision", "step",
                 "decision", "step", "episode_end"],
            )
            self.assertEqual(
                [item["step"] for item in events], [0, 1, 1, 1, 2, 2, 2]
            )
            self.assertEqual(
                [item["sequence"] for item in events], list(range(7))
            )
            end = events[-1]
            self.assertEqual(end["status"], "completed")
            self.assertEqual(end["total_reward"], 1.0)
            self.assertEqual(end["num_steps"], 2)
            self.assertEqual(end["metrics"], {"success": True})
            for item in events:
                self.assertEqual(item["run_id"], metadata["run_id"])
                self.assertEqual(item["scene_id"], "scene.glb")
                self.assertEqual(item["episode_id"], "episode-1")
                self.assertEqual(item["env_index"], 0)
                self.assertEqual(item["schema_version"], 3)
                self.assertTrue(item["timestamp"].endswith("+00:00"))
            elapsed = [item["elapsed_seconds"] for item in events]
            self.assertEqual(elapsed, sorted(elapsed))
            self.assertGreaterEqual(elapsed[0], 0)
        self.assertEqual(len(read_events(trace.run_dir)[0]), len(events))

    def test_nested_values_are_serialized_without_optional_dependencies(self):
        recorder = load_recorder()

        @dataclasses.dataclass
        class Pose:
            location: tuple
            path: Path

        class Model:
            def model_dump(self):
                return {"pose": Pose((1, 2), Path("a/b"))}

        class Array:
            def tolist(self):
                return [[1.0, math.nan], [math.inf, -math.inf]]

        class Detached:
            def cpu(self):
                return Array()

        class Tensor:
            def detach(self):
                return Detached()

        class Scalar:
            def item(self):
                return 2.5

        with recorder.EvaluationTrace(self.output_dir) as trace:
            trace.start_episode(0, episode())
            with trace.policy_step():
                recorder.record_event(
                    "payload", model=Model(), tensor=Tensor(), scalar=Scalar(),
                    collection={"tuple": (1, 2), "set": {3, 4}},
                )
            event = read_events(trace.run_dir)[0][-1]
            self.assertEqual(
                event["model"], {"pose": {"location": [1, 2], "path": "a/b"}}
            )
            self.assertEqual(event["tensor"], [[1.0, None], [None, None]])
            self.assertEqual(event["scalar"], 2.5)
            self.assertEqual(event["collection"]["tuple"], [1, 2])
            self.assertEqual(set(event["collection"]["set"]), {3, 4})
            raw = json.dumps(read_events(trace.run_dir), ensure_ascii=False)
            self.assertNotIn("NaN", raw)
            self.assertNotIn("Infinity", raw)

    def test_exceptions_finalize_summary_and_reset_policy_context(self):
        recorder = load_recorder()
        with self.assertRaisesRegex(RuntimeError, "policy failed"):
            with recorder.EvaluationTrace(self.output_dir) as trace:
                trace.start_episode(0, episode())
                trace.record_step(
                    0, "forward", 2.0, False, {"distance": 4}, {}
                )
                with trace.policy_step():
                    recorder.record_event("llm_response", response="ready")
                    raise RuntimeError("policy failed")
        events = read_events(trace.run_dir)[0]
        self.assertEqual(events[-1]["status"], "error")
        self.assertEqual(events[-1]["error"], {
            "type": "RuntimeError", "message": "policy failed",
        })
        self.assertEqual(events[-1]["num_steps"], 1)
        self.assertEqual(events[-1]["total_reward"], 2.0)
        self.assertEqual(events[-1]["metrics"], {"distance": 4})
        recorder.record_event("outside", invalid=object())
        trace.close()
        self.assertEqual(read_events(trace.run_dir)[0], events)

    def test_repeated_episodes_scenes_runs_and_envs_never_overwrite(self):
        recorder = load_recorder()
        all_ids = set()
        for _ in range(2):
            with recorder.EvaluationTrace(self.output_dir) as trace:
                for scene in ("a/scene.glb", "b/scene.glb", "a/scene.glb"):
                    for env_index in (0, 1):
                        trace.start_episode(env_index, episode(scene, "same"))
                        trace.record_step(env_index, "stop", 0, True, {}, {})
                episodes = read_events(trace.run_dir)
                self.assertEqual(len(episodes), 6)
                for events in episodes:
                    evaluation_id = events[0]["evaluation_id"]
                    self.assertNotIn(evaluation_id, all_ids)
                    all_ids.add(evaluation_id)
                    self.assertTrue(all(
                        event["evaluation_id"] == evaluation_id
                        for event in events
                    ))
        self.assertEqual(len(list(self.output_dir.iterdir())), 2)
        self.assertEqual(len(all_ids), 12)

    def test_pause_reindexes_live_envs_and_preserves_original_identity(self):
        recorder = load_recorder()
        with recorder.EvaluationTrace(self.output_dir) as trace:
            for index in range(3):
                trace.start_episode(index, episode(identifier=str(index)))
            with trace.policy_step():
                recorder.record_event("ambiguous", invalid=object())
            trace.pause_envs([1, 0])
            with trace.policy_step():
                recorder.record_event("decision", action="stop")
            trace.record_step(0, "stop", 3, True, {}, {})
            episodes = {
                events[0]["episode_id"]: events
                for events in read_events(trace.run_dir)
            }
            self.assertEqual(episodes["0"][-1]["status"], "paused")
            self.assertEqual(episodes["1"][-1]["status"], "paused")
            self.assertEqual(episodes["2"][-1]["status"], "completed")
            self.assertEqual(episodes["2"][1]["event"], "decision")
            self.assertTrue(all(
                item["env_index"] == 2 for item in episodes["2"]
            ))

    def test_nested_policy_scopes_restore_the_previous_writer(self):
        recorder = load_recorder()
        with recorder.EvaluationTrace(self.output_dir) as outer:
            outer.start_episode(0, episode(identifier="outer"))
            with outer.policy_step():
                recorder.record_event("outer_before")
                with recorder.EvaluationTrace(self.output_dir) as inner:
                    inner.start_episode(0, episode(identifier="inner"))
                    with inner.policy_step():
                        recorder.record_event("inner")
                    recorder.record_event("outer_restored")
                recorder.record_event("outer_after")
            recorder.record_event("outside")
        outer_events = read_events(outer.run_dir)[0]
        inner_events = read_events(inner.run_dir)[0]
        self.assertEqual([item["event"] for item in outer_events], [
            "episode_start", "outer_before", "outer_restored", "outer_after",
            "episode_end",
        ])
        self.assertEqual([item["event"] for item in inner_events], [
            "episode_start", "inner", "episode_end",
        ])
        self.assertEqual(outer_events[-1]["status"], "interrupted")

    def test_invalid_event_data_leaves_no_partial_events(self):
        recorder = load_recorder()
        with recorder.EvaluationTrace(self.output_dir) as trace:
            trace.start_episode(0, episode())
            with trace.policy_step():
                recorder.record_event("unknown", nested={"value": object()})
                for field in ("sequence", "step", "run_id", "event"):
                    with self.subTest(field=field):
                        recorder.record_event("bad", **{field: 100})
                recorder.record_event("valid", message="saved")
            events = read_events(trace.run_dir)[0]
            self.assertEqual([item["sequence"] for item in events], [0, 1, 2])
            self.assertIsNone(events[1]["nested"]["value"])
            self.assertIn("nested.value", events[1]["missing_fields"])
            self.assertEqual(events[-1]["event"], "valid")
            trace.record_step(0, object(), 1, False, {}, {})
            trace.record_step(0, "stop", 0, True, {}, {})
            self.assertEqual(read_events(trace.run_dir)[0][-1]["num_steps"], 2)

    def test_invalid_lifecycle_calls_do_not_destroy_existing_episode(self):
        recorder = load_recorder()
        trace = recorder.EvaluationTrace(self.output_dir)
        self.addCleanup(trace.close)
        trace.start_episode(0, episode())
        with self.assertRaises(ValueError):
            trace.start_episode(0, episode(identifier="replacement"))
        trace.record_step(5, "stop", 0, True, {}, {})
        with self.assertRaises(ValueError):
            trace.pause_envs([-1])
        trace.close()
        with self.assertRaises(RuntimeError):
            trace.start_episode(0, episode())
        self.assertEqual(len(read_events(trace.run_dir)), 1)
        self.assertEqual(
            read_events(trace.run_dir)[0][0]["episode_id"], "episode-1"
        )

    def test_ambiguous_nested_scope_suspends_outer_event_binding(self):
        recorder = load_recorder()
        with recorder.EvaluationTrace(self.output_dir) as outer:
            outer.start_episode(0, episode())
            with outer.policy_step():
                with recorder.EvaluationTrace(self.output_dir) as inner:
                    # A sole episode at a nonzero index is also ambiguous.
                    inner.start_episode(1, episode())
                    with inner.policy_step():
                        recorder.record_event("ambiguous", invalid=object())
                    recorder.record_event("restored")
            self.assertEqual(
                [item["event"] for item in read_events(outer.run_dir)[0]],
                ["episode_start", "restored"],
            )
            self.assertEqual(len(read_events(inner.run_dir)[0]), 2)

    def test_explicit_error_close_finalizes_all_active_episodes(self):
        recorder = load_recorder()
        trace = recorder.EvaluationTrace(self.output_dir)
        for index in (0, 1):
            trace.start_episode(index, episode(identifier=str(index)))
            trace.record_step(index, "forward", index + 1, False, {}, {})
        trace.close(error=ValueError("evaluation stopped"))
        for events in read_events(trace.run_dir):
            self.assertEqual(events[-1]["status"], "error")
            self.assertEqual(events[-1]["num_steps"], 1)
            self.assertEqual(events[-1]["error"], {
                "type": "ValueError", "message": "evaluation stopped",
            })


if __name__ == "__main__":
    unittest.main()
