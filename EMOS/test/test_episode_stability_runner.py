"""Exercise sampling, runtime request controls, and real trace validation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import os
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_episode_stability as runner


SETTINGS = {"model": "gpt-6-sol", "temperature": 0.0, "model_seed": 100}


class StabilityRunnerTests(unittest.TestCase):
    def test_sampling_does_not_depend_on_input_order_or_old_outcomes(self):
        episodes = [{"scene_id": "scene", "episode_id": str(i), "success": i % 2}
                    for i in range(4)]
        first = runner.select_episodes(episodes, count=2, sample_seed=100)
        changed = [{**e, "success": not e["success"]} for e in reversed(episodes)]
        second = runner.select_episodes(changed, count=2, sample_seed=100)
        self.assertEqual([e["episode_id"] for e in first], ["1", "3"])
        self.assertEqual([e["episode_id"] for e in second], ["1", "3"])

    def test_sampling_rejects_duplicate_identities(self):
        episode = {"scene_id": "scene", "episode_id": "1"}
        with self.assertRaises(ValueError):
            runner.select_episodes([episode, episode], count=1)

    def test_runtime_controls_reach_each_phase_and_reject_conflicts(self):
        class Model:
            model = "gpt-6-sol"

            def _create_completion(self, request, phase, **kwargs):
                return {"request": request, "phase": phase, "model": self.model, **kwargs}

        runner.install_decoding_controls(Model, SETTINGS)
        for phase in ("discussion", "planning", "execution"):
            emitted = Model()._create_completion([], phase, tool_choice="required")
            self.assertEqual(emitted["model"], "gpt-6-sol")
            self.assertEqual(emitted["temperature"], 0)
            self.assertEqual(emitted["seed"], 100)
            self.assertEqual(emitted["tool_choice"], "required")
        with self.assertRaises(ValueError):
            Model()._create_completion([], "execution", seed=101)
        different = Model()
        different.model = "other"
        with self.assertRaises(ValueError):
            different._create_completion([], "planning")

    def trace(self, directory, *, success=False, state=True, api_error=False):
        trajectory = Path(directory) / "trajectories"
        episode = trajectory / "run" / "episodes" / "64--eval"
        episode.mkdir(parents=True)
        identity = {"episode_id": "64", "scene_id": "scene", "run_id": "run",
                    "evaluation_id": "eval"}
        events = [
            {"event_type": "episode_start", "state": {"agents": {"agent_0": {}}}
             if state else None, **identity},
            {"event_type": "model_config", "call_id": "call", "phase": "execution", "agent_id": "agent_0",
             "model": "gpt-6-sol", "request_kwargs": {
                 "model": "gpt-6-sol", "temperature": 0, "seed": 100}},
            {"event_type": "llm_request", "call_id": "call", "phase": "execution", "agent_id": "agent_0",
             "request_kwargs": {"model": "gpt-6-sol", "temperature": 0, "seed": 100}},
            {"event_type": "llm_response", "call_id": "call", "phase": "execution", "agent_id": "agent_0", "response_model": "gpt-6-sol",
             "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        ]
        events.extend({"event_type": "step", "done": step == 4, "step": step,
                       "metrics": {"pddl_success": float(success)},
                       "state": {"task_progress": {"pddl_success": success}}}
                      for step in range(1, 5))
        events.append({"event_type": "episode_end", "status": "completed",
                       "task_success": success, "num_env_steps": 4})
        if api_error:
            events.append({"event_type": "llm_error", "error_type": "InternalServerError"})
        (episode / "events.jsonl").write_text("".join(json.dumps({**identity, **e}) + "\n" for e in events))
        (episode / "summary.json").write_text(json.dumps({
            **identity, "status": "error" if api_error else "completed",
            "task_success": success, "metrics": {"pddl_success": float(success)},
            "num_env_steps": 4, "elapsed_seconds": 1.0,
        }))
        return {"task": "Manipulation", "episode_id": "64", "scene_id": "scene",
                "trial_dir": directory, "trajectory_dir": str(trajectory)}

    def test_completed_native_failure_stays_a_valid_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = self.trace(directory)
            result = runner.inspect_trial(trial, SETTINGS, returncode=0)
        self.assertEqual(result["status"], "completed")
        self.assertIs(result["task_success"], False)
        self.assertEqual(result["total_tokens"], 15)
        self.assertTrue(result["initial_state_sha256"])

    def test_missing_state_invalidates_even_a_successful_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = self.trace(directory, success=True, state=False)
            result = runner.inspect_trial(trial, SETTINGS, returncode=0)
        self.assertEqual(result["status"], "recording_error")
        self.assertIsNone(result["task_success"])

    def test_api_error_after_steps_is_not_native_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = self.trace(directory, api_error=True)
            result = runner.inspect_trial(trial, SETTINGS, returncode=1)
        self.assertEqual(result["status"], "api_error")
        self.assertIsNone(result["task_success"])

    def test_wrong_request_seed_invalidates_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = self.trace(directory)
            path = next(Path(trial["trajectory_dir"]).glob("*/episodes/*/events.jsonl"))
            path.write_text(path.read_text().replace('"seed": 100', '"seed": 101'))
            result = runner.inspect_trial(trial, SETTINGS, returncode=0)
        self.assertEqual(result["status"], "recording_error")

    def corrupt_trace(self, change):
        with tempfile.TemporaryDirectory() as directory:
            trial = self.trace(directory)
            path = next(Path(trial["trajectory_dir"]).glob("*/episodes/*/events.jsonl"))
            events = [json.loads(line) for line in path.read_text().splitlines()]
            events = change(events)
            path.write_text("".join(json.dumps(e) + "\n" for e in events))
            return runner.inspect_trial(trial, SETTINGS, returncode=0)

    def test_partial_final_write_and_foreign_terminal_are_rejected(self):
        cases = [
            lambda es: [e for e in es if e["event_type"] != "episode_end"],
            lambda es: [{**e, "run_id": "foreign"} if e.get("done") else e for e in es],
            lambda es: [{**e, "state": {"task_progress": {"pddl_success": True}}}
                        if e.get("done") else e for e in es],
            lambda es: [e for e in es if not (e["event_type"] == "step" and e["step"] == 2)],
        ]
        for change in cases:
            with self.subTest(change=change):
                self.assertEqual(self.corrupt_trace(change)["status"], "recording_error")

    def test_missing_and_duplicated_model_call_ids_are_rejected(self):
        def missing(events):
            for e in events:
                e.pop("call_id", None)
            return events

        def duplicated(events):
            copies = [dict(e) for e in events if e["event_type"] in
                      ("model_config", "llm_request", "llm_response")]
            return events[:4] + copies + events[4:]

        for change in (missing, duplicated):
            with self.subTest(change=change):
                self.assertEqual(self.corrupt_trace(change)["status"], "recording_error")

    def test_existing_attempt_directory_is_never_overwritten_or_started(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = self.trace(directory)
            trial.update(round=1, trial_id="r1_task0", status="pending", task_success=None)
            log = Path(directory) / "worker.log"
            log.write_text("preserve prior evidence")
            manifest = {"status": "prepared", "cohort": [trial], "trials": [trial],
                        "settings": {**SETTINGS, "repetitions": 1, "python": sys.executable},
                        "source_hashes": {}, "input_hashes": {}}
            with patch.object(runner, "check_hashes"), patch.object(runner.subprocess, "Popen") as start:
                with self.assertRaises(FileExistsError):
                    runner.execute(manifest, Path(directory) / "experiment")
                start.assert_not_called()
            self.assertEqual(log.read_text(), "preserve prior evidence")

    def test_asset_inputs_include_scene_robot_resume_and_robot_geometry_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = ["source.json.gz", "robots.json", "scene.json", "single.json.gz",
                        "habitat-mas/habitat_mas/data/robot_resume/FetchRobot.json",
                        "data/robots/fetch/robot.urdf",
                        "data/objects/ycb/configs/bottle.object_config.json"]
            for name in relative:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            cohort = [{"scene_id": "scene.json", "dataset_file": str(root / "single.json.gz")}]
            paths = runner.collect_input_paths(root / "source.json.gz", root / "robots.json", cohort, root=root)
            self.assertTrue({root / p for p in relative}.issubset(set(paths)))

    @unittest.skipUnless(hasattr(os, "fork"), "POSIX process groups required")
    def test_timeout_kills_descendant_even_if_direct_worker_already_exited(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_path = Path(directory) / "child_pid"
            code = (
                "import os,signal,time,pathlib\n"
                "child=os.fork()\n"
                "if child==0:\n"
                " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
                " pathlib.Path(" + repr(str(pid_path)) + ").write_text(str(os.getpid()))\n"
                " time.sleep(60)\n"
                "else:\n time.sleep(60)\n"
            )
            proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(pid_path.exists())
                child_pid = int(pid_path.read_text())
                runner.terminate_process_group(proc, grace_seconds=.05)
                status = Path("/proc/%d/status" % child_pid)
                deadline = time.monotonic() + 2
                while status.exists() and "State:\tZ" not in status.read_text() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(not status.exists() or "State:\tZ" in status.read_text())
            finally:
                try:
                    os.killpg(proc.pid, 9)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
