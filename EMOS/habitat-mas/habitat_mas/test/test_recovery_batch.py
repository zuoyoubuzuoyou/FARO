"""Behavioral coverage for selection, paired execution and recovery statistics."""
import json
import tempfile
import unittest
import sys
import subprocess
import os
from unittest.mock import patch
from pathlib import Path

from habitat_mas.fault_injection import recovery_batch as batch
from habitat_mas.fault_injection import recovery_analysis as analysis
from habitat_mas.fault_injection import recovery_runner as runner


def step(scene, episode, index, done=False, success=0, dropout=False):
    return {
        "type": "simulator_step", "mode": "clean", "scene_id": scene,
        "episode_id": episode, "simulator_step": index, "done": done,
        "metrics": {"pddl_success": success}, "policy_info": {},
        "agent_step_actions": {
            "agent_0": {"action": {"action": "agent_dropout_noop" if dropout else "wait"}},
            "agent_1": {"action": {"action": "wait"}},
        },
        "simulator_state": {"agents": {
            "agent_0": {"dropout_state": {"active": dropout}},
            "agent_1": {"dropout_state": {"active": False}},
        }},
    }


class RecoveryBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_rows(self, rows, name="trace.jsonl"):
        path = self.root / name
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def test_selection_keeps_duplicate_ids_in_distinct_scenes_and_excludes_short(self):
        rows = [step("a", "7", 0), step("a", "7", 1, True, 1),
                step("b", "7", 0, True, 1), step("c", "8", 0, True, 0),
                step("d", "9", 0)]
        selected = batch.select_clean(self.write_rows(rows))
        self.assertEqual(selected["completed"], 3)
        self.assertEqual(selected["successful"], 2)
        self.assertEqual([(x["scene_id"], x["eligible"]) for x in selected["tasks"]],
                         [("a", True), ("b", False)])
        self.assertEqual(selected["tasks"][0]["injection_step"], 1)

    def test_selection_rejects_duplicate_or_missing_steps(self):
        for rows in ([step("a", "1", 0), step("a", "1", 0, True, 1)],
                     [step("a", "1", 1, True, 1)]):
            with self.assertRaises(ValueError):
                batch.select_clean(self.write_rows(rows))

    def test_commands_share_branch_and_preserve_explicit_seeds(self):
        manifest = {"protocol": {"seed": 66, "simulator_seed": 100,
            "max_steps": 750, "config": "multi_rearrange/llm_spot_drone_per.yaml",
            "agents": ["agent_0", "agent_1"], "repeats": 1}}
        task = {"task_id": "abc", "episode_id": "7", "scene_id": "a",
                "dataset": "/tmp/one.json.gz", "trace": "/tmp/one.jsonl",
                "injection_step": 5}
        commands = [batch.build_command(manifest, task, group, "agent_1", 0,
                    self.root / group, "/python") for group in ("dropout", "dropout_recovery")]
        for cmd in commands:
            self.assertIn("habitat.dataset.should_terminate_on_wait=False", cmd)
            self.assertIn("habitat.seed=66", cmd)
            self.assertEqual(cmd[cmd.index("--step") + 1], "5")
            self.assertEqual(cmd[cmd.index("--replay-trajectory") + 1], "/tmp/one.jsonl")
        self.assertIn("habitat_baselines.error_detection.enabled=False", commands[0])
        self.assertIn("habitat_baselines.error_recovery.enabled=True", commands[1])

    def test_wait_count_distinguishes_real_wait_from_offline_noop_and_success(self):
        rows = [step("a", "1", 0), step("a", "1", 1),
                step("a", "1", 2, dropout=True), step("a", "1", 3, dropout=True),
                step("a", "1", 4, done=True, success=1, dropout=True)]
        result = analysis.wait_metrics(rows, threshold=2, start_step=0)
        self.assertEqual(result["double_wait_longest"], 2)
        self.assertEqual(result["offline_wait_longest"], 2)
        self.assertEqual(result["offline_wait_onset"], 2)
        self.assertTrue(result["double_wait_detected"])
        self.assertEqual(analysis.wait_metrics(rows, 2, 2)["double_wait_longest"], 0)

    def test_recovery_event_is_not_terminal_success(self):
        rows = [step("a", "1", 0), {"type": "recovery_completed"}]
        result = analysis.summarize_rows(rows, {"type": "fault_injected", "agent": "agent_0",
            "simulator_step": 1}, "dropout_recovery", "agent_0", 1, 0)
        self.assertFalse(result["valid"])
        self.assertIsNone(result["success"])

    def test_crash_is_valid_failure_only_with_verified_prefix_and_injection(self):
        rows = [step("a", "1", 0), {"type": "replay_prefix_verified",
                "branch_step": 1, "verified_steps": 1, "context_fidelity": "recorded"},
                {"type": "system_crash", "done": True, "simulator_step": 1,
                 "reason": "agent_dropout", "metrics": {"pddl_success": 0}}]
        event = {"type": "fault_injected", "agent": "agent_0", "simulator_step": 1}
        result = analysis.summarize_rows(rows, event, "dropout_recovery", "agent_0", 1, 3)
        self.assertTrue(result["valid"])
        self.assertEqual(result["success"], 0)
        self.assertFalse(analysis.summarize_rows(rows, None, "dropout_recovery", "agent_0", 1, 3)["valid"])
        self.assertFalse(analysis.summarize_rows(rows, event, "dropout_recovery", "agent_0", 1, 1)["valid"])

    def test_pair_statistics_include_harmed_tasks_and_missing_pairs(self):
        pairs = [(0, 1), (0, 1), (1, 0), (1, 1)]
        stats = analysis.paired_statistics(pairs, resamples=200)
        self.assertEqual(stats["rescued"], 2)
        self.assertEqual(stats["harmed"], 1)
        self.assertEqual(stats["rescue_rate"], 1.0)
        self.assertEqual(stats["success_rate_difference"], 0.25)
        self.assertEqual(stats["mcnemar_exact_p"], 1.0)
        self.assertEqual(analysis.paired_statistics([(0, 1)] * 6)["mcnemar_exact_p"], 0.03125)


    def test_prepare_freezes_scene_specific_inputs_and_refuses_overwrite(self):
        trace = self.write_rows([step("a", "7", 0), step("a", "7", 1, True, 1),
                                 step("b", "7", 0), step("b", "7", 1, True, 1)])
        dataset = self.root / "episodes.json"
        dataset.write_text(json.dumps({"episodes": [{"scene_id": s, "episode_id": "7"} for s in ("a", "b")]}))
        output = self.root / "experiment"
        manifest = batch.prepare_experiment(trace, dataset, output)
        self.assertEqual(len(list(batch.jobs(manifest))), 10)
        self.assertNotEqual(manifest["tasks"][0]["task_id"], manifest["tasks"][1]["task_id"])
        batch.validate_manifest(manifest)
        Path(manifest["tasks"][0]["trace"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            batch.validate_manifest(manifest)
        with self.assertRaises(FileExistsError):
            batch.prepare_experiment(trace, dataset, output)

    def test_runner_resumes_valid_result_without_new_process_and_preserves_bad_attempt(self):
        task = {"task_id": "a", "scene_id": "a", "episode_id": "1", "injection_step": 1}
        job = {"task": task, "job_id": "a_clean_r0", "group": "clean", "agent": None, "repeat": 0}
        manifest = {"output": str(self.root), "protocol": {"wait_threshold": 2}}
        def command(_manifest, _task, _group, _agent, _repeat, attempt, _python):
            path = Path(attempt) / "output" / "clean_trajectory.jsonl"
            rows = [step("a", "1", 0), step("a", "1", 1, True, 1)]
            code = "from pathlib import Path; p=Path(" + repr(str(path)) + "); p.parent.mkdir(parents=True); p.write_text(" + repr("".join(json.dumps(r) + "\n" for r in rows)) + ")"
            return [sys.executable, "-c", code]
        with patch.object(runner, "build_command", command):
            first = runner.run_one(manifest, job, sys.executable, 10)
            self.assertEqual(first["state"], "complete")
            second = runner.run_one(manifest, job, sys.executable, 10)
            self.assertEqual(first["attempt"], second["attempt"])
            Path(first["attempt"], "output", "clean_trajectory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in [step("a", "1", 0), step("a", "1", 1, True, 0)]))
            failed = runner.run_one(manifest, job, sys.executable, 10)
            self.assertEqual(failed["state"], "error")
            task["eligible"] = True
            manifest.update(tasks=[task], selection={})
            manifest["protocol"].update(agents=["agent_0"], repeats=1)
            self.assertEqual(analysis.analyze(manifest)["groups"]["clean"]["valid"], 0)
            retried = runner.run_one(manifest, job, sys.executable, 10, retry_errors=True)
            self.assertEqual(retried["state"], "complete")
            self.assertNotEqual(retried["attempt"], first["attempt"])
            report = analysis.analyze(manifest)
            self.assertEqual(report["groups"]["clean"]["valid"], 1)
            self.assertEqual(report["paired_by_agent"]["agent_0"]["missing_pairs"], 1)

    def test_runner_timeout_is_infrastructure_error(self):
        task = {"task_id": "a", "scene_id": "a", "episode_id": "1", "injection_step": 1}
        job = {"task": task, "job_id": "a_clean_r0", "group": "clean", "agent": None, "repeat": 0}
        manifest = {"output": str(self.root), "protocol": {"wait_threshold": 2}}
        with patch.object(runner, "build_command", return_value=[sys.executable, "-c", "import time; time.sleep(10)"]):
            result = runner.run_one(manifest, job, sys.executable, .05)
        self.assertEqual(result["state"], "error")
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["result"]["success"])


    def test_steps_after_terminal_are_invalid(self):
        rows = [step("a", "1", 0, True, 1), step("a", "1", 1)]
        result = analysis.summarize_rows(rows, None, "clean", None, 0, 0)
        self.assertFalse(result["valid"])

    def test_completed_output_wrong_mode_or_replay_source_is_invalid(self):
        task = {"task_id": "a", "scene_id": "a", "episode_id": "1", "injection_step": 1, "trace": "/expected.jsonl"}
        job = {"task": task, "job_id": "a_dropout", "group": "dropout", "agent": "agent_0", "repeat": 0}
        output = self.root / "output"
        output.mkdir()
        trajectory = output / "dropout_trajectory.jsonl"
        rows = [step("a", "1", 0), {"type": "replay_prefix_verified", "scene_id": "a", "episode_id": "1", "branch_step": 1,
                "verified_steps": 1, "context_fidelity": "recorded", "source": "/wrong.jsonl"}, step("a", "1", 1, True, 1, True)]
        for row in rows: row["mode"] = "faulty"
        trajectory.write_text("".join(json.dumps(r) + "\n" for r in rows))
        (output / "dropout_events.jsonl").write_text(json.dumps({"type": "fault_injected", "episode_id": "1", "agent": "agent_0", "simulator_step": 1}) + "\n")
        self.assertFalse(analysis.summarize_attempt(job, self.root, 0)["valid"])
        rows[1]["source"] = "/expected.jsonl"
        trajectory.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertTrue(analysis.summarize_attempt(job, self.root, 0)["valid"])
        for row in rows: row["mode"] = "clean"
        trajectory.write_text("".join(json.dumps(r) + "\n" for r in rows))
        self.assertFalse(analysis.summarize_attempt(job, self.root, 0)["valid"])


    def test_live_stale_worker_blocks_retry(self):
        task = {"task_id": "a", "scene_id": "a", "episode_id": "1", "injection_step": 1}
        job = {"task": task, "job_id": "a_clean_r0", "group": "clean", "agent": None, "repeat": 0}
        manifest = {"output": str(self.root), "protocol": {"wait_threshold": 2}}
        directory = self.root / "runs" / job["job_id"]
        directory.mkdir(parents=True)
        (directory / "status.json").write_text(json.dumps({"state": "running", "pid": os.getpid(), "attempt": str(directory / "attempt_001")}))
        with patch.object(runner, "build_command", return_value=[sys.executable, "-c", "pass"]):
            with self.assertRaisesRegex(ValueError, "still alive"):
                runner.run_one(manifest, job, sys.executable, 1, retry_errors=True)


    def test_validation_only_marker_survives_report_regeneration(self):
        manifest = {"output": str(self.root), "validation_only": True,
                    "tasks": [], "selection": {}, "protocol": {"agents": ["agent_0"], "repeats": 1}}
        report = analysis.analyze(manifest)
        self.assertTrue(report["validation_only"])
        self.assertIn("VALIDATION ONLY", (self.root / "analysis/REPORT.md").read_text())

    def test_normal_cli_refuses_to_resume_fixed_model_experiment(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"validation_only": True}))
        result = subprocess.run([sys.executable, str(batch.ROOT / "scripts/recovery_batch.py"),
                                 "run", "--manifest", str(manifest), "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("validation-only", result.stderr)


    def test_dropout_only_cli_schedules_both_agents_without_clean_or_recovery(self):
        trace = self.write_rows([step("a", "7", 0), step("a", "7", 1, True, 1),
                                step("b", "7", 0), step("b", "7", 1, True, 1)])
        dataset = self.root / "episodes.json"
        dataset.write_text(json.dumps({"episodes": [{"scene_id": scene, "episode_id": "7"} for scene in ("a", "b")]}))
        output = self.root / "experiment"
        batch.prepare_experiment(trace, dataset, output)
        result = subprocess.run([sys.executable, str(batch.ROOT / "scripts/recovery_batch.py"),
                                 "run", "--manifest", str(output / "manifest.json"),
                                 "--groups", "dropout", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = json.loads((output / "commands.json").read_text())
        self.assertEqual(len(commands), 4)
        for row in commands:
            self.assertIn("_dropout_agent_", row["job_id"])
            self.assertNotIn("recovery", row["job_id"])
            self.assertIn("habitat_baselines.error_recovery.enabled=False", row["command"])
        self.assertFalse((output / "runs").exists())
        result = subprocess.run([sys.executable, str(batch.ROOT / "scripts/recovery_batch.py"),
                                 "run", "--manifest", str(output / "manifest.json"),
                                 "--groups", "dropout_recovery", "--limit-tasks", "1", "--dry-run"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        commands = json.loads((output / "commands.json").read_text())
        self.assertEqual(len(commands), 2)
        for row in commands:
            self.assertIn("habitat_baselines.error_recovery.enabled=True", row["command"])


if __name__ == "__main__":
    unittest.main()
