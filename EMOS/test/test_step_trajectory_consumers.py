"""Step storage must remain causal input to detectors and experiment audits."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from habitat_mas.detection.readers import frames_from_records, iter_frames
import test_episode_stability_runner as runner_tests


ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {"schema_version": 2, "run_id": "run", "scene_id": "scene",
            "episode_id": "64", "evaluation_id": "eval", "env_index": 0}


def event(kind, step, **data):
    return {**IDENTITY, "event_type": kind, "event": kind,
            "env_step": step, "step": step, **data}


class StepConsumerTests(unittest.TestCase):
    def storage(self):
        path = ROOT / "habitat-mas/habitat_mas/utils/trajectory_storage.py"
        self.assertTrue(path.exists(), "The environment-step storage module is missing")
        spec = importlib.util.spec_from_file_location("_step_consumer_storage", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def store_records(self, directory, records, summary=None):
        storage = self.storage()
        identity = {key: records[0][key] for key in IDENTITY if key in records[0]}
        store = storage.StepTraceStore(directory, identity)
        pending_step = None
        for number, record in enumerate(records):
            row = dict(record)
            row.setdefault("sequence", number)
            if row["event_type"] == "step":
                if pending_step is not None:
                    store.commit_step()
                pending_step = row["step"]
            # Non-step events after a returned step can still belong to that step.
            elif pending_step is not None and row.get("env_step", pending_step) > pending_step:
                store.commit_step()
                pending_step = None
            store.append(row)
        if pending_step is not None:
            store.commit_step()
        if summary is not None:
            store.write_summary(summary)
        store.close()
        return Path(directory) / "steps.jsonl"

    def test_detector_reads_step_records_with_same_causal_actions_and_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            records = [
                event("episode_start", 0, context={"pddl_text_goal": "Move object"},
                      state={"agents": {"agent_0": {"base_position": [0, 0, 0]}}}),
                event("action_start", 1, agent_id="agent_0", action_id="nav",
                      action_name="nav_to_obj", parameters={"target_obj": "any_targets|0"}),
                event("communication_send", 1, sender="agent_0", recipient="agent_1",
                      message="Carry it", status="queued"),
                event("step", 1, action=[1], reward=0, done=False,
                      metrics={"pddl_success": False}, state={"agents": {"agent_0": {
                          "base_position": [1, 0, 0], "base_rotation": [0, 0, 0, 1],
                          "grasped_object_id": None}}, "target_objects": []}),
                event("action_end", 1, agent_id="agent_0", action_id="nav",
                      status="timeout", result={"call_high_level": False}),
                event("skill_termination_signal", 2, agent_id="agent_0", action_id="nav",
                      call_high_level=False, over_max_len=True),
                event("step", 2, action=[0], reward=0, done=True,
                      metrics={"pddl_success": False}, state={"agents": {"agent_0": {
                          "base_position": [1, 0, 0], "base_rotation": [0, 0, 0, 1],
                          "grasped_object_id": None}}, "target_objects": []}),
                event("episode_end", 2, status="completed", task_success=False,
                      num_env_steps=2),
            ]
            path = self.store_records(directory, records)
            actual = list(iter_frames(path, "Manipulation"))
        expected = list(frames_from_records(records, "Manipulation"))
        self.assertEqual(len(actual), 2)
        self.assertEqual(actual, expected)
        self.assertEqual(actual[0]["messages"][0]["message"], "Carry it")
        self.assertEqual(actual[1]["agents"]["agent_0"]["action"]["id"], "nav")

    def test_stability_audits_new_step_index_and_original_call_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = runner_tests.StabilityRunnerTests().trace(directory, success=True)
            old = next(Path(trial["trajectory_dir"]).rglob("events.jsonl"))
            records = [json.loads(line) for line in old.read_text().splitlines()]
            for index, row in enumerate(records):
                row.setdefault("env_step", row.get("step", 0 if index == 0 else 1))
                row.setdefault("step", row["env_step"])
            summary = json.loads(old.with_name("summary.json").read_text())
            summary["files"] = {"steps": "steps.jsonl"}
            self.store_records(old.parent, records, summary)
            old.unlink()
            result = runner_tests.runner.inspect_trial(trial, runner_tests.SETTINGS, 0)
        self.assertEqual(result["status"], "completed", result)
        self.assertIs(result["task_success"], True)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["total_tokens"], 15)
        self.assertTrue(result["events_path"].endswith("steps.jsonl"))

    def test_api_error_in_pending_is_not_a_native_failure_or_an_executed_step(self):
        with tempfile.TemporaryDirectory() as directory:
            episode_dir = Path(directory) / "trajectories/run/episodes/64--eval"
            episode_dir.mkdir(parents=True)
            records = [
                event("episode_start", 0, state={"agents": {"agent_0": {}}}),
                event("llm_request", 1, call_id="failed", agent_id="agent_0", phase="execution"),
                event("llm_error", 1, call_id="failed", agent_id="agent_0", phase="execution",
                      error_type="TransportError", error_message="offline"),
                event("episode_end", 1, status="error", task_success=None, num_env_steps=0),
            ]
            summary = {**IDENTITY, "status": "error", "task_success": None,
                       "num_env_steps": 0, "files": {"steps": "steps.jsonl"}}
            path = self.store_records(episode_dir, records, summary)
            self.assertEqual(path.read_text(), "")
            trial = {"task": "Manipulation", "episode_id": "64", "scene_id": "scene",
                     "trial_dir": directory, "trajectory_dir": str(Path(directory) / "trajectories")}
            result = runner_tests.runner.inspect_trial(trial, runner_tests.SETTINGS, 1)
        self.assertEqual(result["status"], "api_error", result)
        self.assertIsNone(result["task_success"])
        self.assertEqual(result["num_env_steps"], 0)

    def test_invalid_event_shape_is_classified_as_recording_error(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = runner_tests.StabilityRunnerTests().trace(directory, success=True)
            old = next(Path(trial["trajectory_dir"]).rglob("events.jsonl"))
            records = list(self.storage().iter_trace_events(old))
            summary = json.loads(old.with_name("summary.json").read_text())
            summary["files"] = {"steps": "steps.jsonl"}
            path = self.store_records(old.parent, records, summary)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["events"] = [42]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = runner_tests.runner.inspect_trial(trial, runner_tests.SETTINGS, 0)
        self.assertEqual(result["status"], "recording_error", result)
        self.assertIsNone(result["task_success"])

    def test_historical_v1_keeps_zero_based_steps_and_false_diagnostics(self):
        records = [{"schema_version": 1, "type": "simulator_step", "simulator_step": 0,
                    "simulator_state": {"agents": {"agent_0": {"position": [1, 0, 0],
                        "rotation": 0.25, "nav_finished": False}}}, "metrics": {"pddl_success": 0}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.jsonl"
            path.write_text(json.dumps(records[0]) + "\n")
            actual = list(iter_frames(path, "Mobility"))
        self.assertEqual(actual[0]["step"], 0)
        self.assertIs(actual[0]["agents"]["agent_0"]["nav_finished"], False)


if __name__ == "__main__":
    unittest.main()
