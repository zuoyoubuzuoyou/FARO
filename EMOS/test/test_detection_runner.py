"""End-to-end replay orchestration with a replaced API transport only."""
import copy
import csv
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def load_runner():
    path = ROOT / "habitat-mas/habitat_mas/detection/runner.py"
    assert path.is_file(), "Replay runner is missing"
    return importlib.import_module("habitat_mas.detection.runner")


def frame(step):
    return {"step": step, "task": "manipulation", "context": {
        "assignments": {"agent_0": "move any_targets|0"}},
        "agents": {"agent_0": {"position": [0, 0, 0], "rotation": 0,
            "held_object": None, "action": {"name": "wait", "arguments": {},
            "id": "wait1", "age": step + 1}}},
        "objects": {"any_targets|0": {"position": [2, 0, 0]},
            "any_targets|1": {"position": [3, 0, 0]}},
        "metrics": {"object_to_goal_distance.0": 2,
            "object_to_goal_distance.1": 2},
        "decisions": [{"agent": "agent_0", "tool": "wait", "step": step}],
        "messages": [{"sender": "agent_0", "message": str(step), "step": step}]}


def episode():
    return {"dataset": "batch", "task": "manipulation", "episode_id": "1",
        "scene_id": "scene", "native_success": True, "raw_path": "unused",
        "report_path": "episode.md", "goal_positions": {},
        "source_sha256": "abc"}


class API:
    def __init__(self, status="normal", error=False):
        self.status, self.error, self.requests = status, error, []
        self.chat = SimpleNamespace(completions=self)
        self.base_url = "https://example.invalid"

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.error:
            raise TimeoutError("test timeout")
        recent = json.loads(request["messages"][1]["content"])["recent_frames"]
        value = {"status": self.status, "reason_code": "test",
            "explanation": "观察到窗口内的执行情况", "agents": ["agent_0"],
            "evidence": [{"step": recent[-1]["step"], "observation": "该步等待"}],
            "confidence": 0.8}
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(value), "refusal": None}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5,
                "total_tokens": 10}}


class DetectionRunnerTest(unittest.TestCase):
    def test_sampling_retains_intermediate_events_without_future_labels(self):
        runner = load_runner()
        api = API()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", return_value=iter(frame(i) for i in range(9))):
                rows = runner.run_episode(episode(), {"sample_every": 2,
                    "window": 3, "check_every": 2}, directory, ["llm"], api)
        self.assertEqual(len(api.requests), 2)
        request = json.loads(api.requests[0]["messages"][1]["content"])
        recent = request["recent_frames"]
        self.assertEqual([f["step"] for f in recent], [0, 2, 4])
        self.assertEqual([d["step"] for f in recent for d in f["decisions"]], list(range(5)))
        self.assertNotIn("native_success", json.dumps(request))
        self.assertNotIn("source_sha256", json.dumps(request))
        self.assertIs(rows[0]["detected"], False)
        self.assertIs(rows[0]["complete"], True)

    def test_llm_positive_stops_only_that_method_and_reuses_request_cache(self):
        runner = load_runner()
        api = API("system_error")
        config = {"sample_every": 1, "window": 3, "check_every": 1,
            "repeat_threshold": 1, "min_stall_steps": 1}
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with patch.object(runner, "iter_frames", return_value=iter(frame(i) for i in range(8))):
                    rows = runner.run_episode(episode(), config, directory,
                        ["repeat_action", "team_progress", "llm"], api)
            self.assertEqual(len(api.requests), 1)
            by_method = {r["method"]: r for r in rows}
            self.assertTrue(all(r["detected"] for r in rows))
            self.assertEqual(by_method["llm"]["first_alarm_step"], 2)
            self.assertEqual(by_method["repeat_action"]["first_alarm_step"], 1)
            self.assertEqual(by_method["llm"]["stats"]["cache_hits"], 1)

    def test_short_episode_never_gets_an_extra_terminal_query(self):
        runner = load_runner()
        api = API()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", return_value=iter([frame(0), frame(1)])):
                row = runner.run_episode(episode(), {"window": 10}, directory, ["llm"], api)[0]
        self.assertEqual(api.requests, [])
        self.assertTrue(row["short_window"])
        self.assertTrue(row["complete"])
        self.assertFalse(row["detected"])

    def test_api_error_or_budget_is_partial_never_negative(self):
        runner = load_runner()
        for api, budget in ((API(error=True), None), (API(), 0)):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory:
                with patch.object(runner, "iter_frames", return_value=iter(frame(i) for i in range(4))):
                    row = runner.run_episode(episode(), {"window": 2, "sample_every": 1,
                        "check_every": 1, "max_llm_calls": budget}, directory, ["llm"], api)[0]
                self.assertFalse(row["complete"])
                self.assertIsNone(row["detected"])
                self.assertEqual(row["status"], "partial")

    def test_uncertain_valid_response_counts_as_completed_negative(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", return_value=iter(frame(i) for i in range(4))):
                row = runner.run_episode(episode(), {"window": 2, "sample_every": 1,
                    "check_every": 1}, directory, ["llm"], API("uncertain"))[0]
        self.assertTrue(row["complete"])
        self.assertFalse(row["detected"])
        self.assertEqual(row["stats"]["uncertain"], 3)

    def test_outputs_keep_success_labels_blank_and_do_not_overwrite_answers(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            for _ in range(2):
                with patch.object(runner, "iter_frames", return_value=iter(frame(i) for i in range(4))):
                    result = runner.run_evaluation([episode()], {"window": 2}, directory,
                        ["repeat_action"])
                label = Path(directory) / "manual_labels.csv"
                if _ == 0:
                    self.assertTrue(label.read_bytes().startswith(b"\xef\xbb\xbf"))
                    before = label.read_bytes()
                else:
                    self.assertEqual(label.read_bytes(), before)
            self.assertIsNone(result["metrics"]["batch"]["methods"]["repeat_action"]["precision"])
            self.assertTrue((Path(directory) / "main_table_excel.csv").is_file())

    def test_loader_accepts_rearrange_path_and_gets_outcome_from_last_frame(self):
        runner = load_runner()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manual = root / "analysis_results/batch/manual_review"
            (manual / "raw").mkdir(parents=True)
            raw = manual / "raw/episode_1.jsonl"
            raw.write_text(json.dumps({"type": "episode_start", "scene_id": "scene",
                "episode_id": "1"}) + "\n" + json.dumps({"type": "simulator_step",
                "done": True, "metrics": {"pddl_success": 0}}) + "\n")
            (manual / "manifest.json").write_text(json.dumps({"episodes": [{
                "episode_id": "1", "path": "raw/episode_1.jsonl",
                "report": "episodes/episode_1.md", "sha256": "known"}]}))
            config = root / "config.json"
            config.write_text(json.dumps({"datasets": [{"dataset": "batch",
                "task": "rearrange", "manifest": str((manual / "manifest.json").relative_to(root)),
                "expected_episodes": 1}]}))
            loaded = runner.load_dataset_episodes(config, root)
            self.assertEqual(len(loaded), 1)
            self.assertFalse(loaded[0]["native_success"])
            self.assertEqual(loaded[0]["scene_id"], "scene")
            self.assertEqual(loaded[0]["source_sha256"], "known")

    def test_runtime_share_uses_only_new_human_runtime_labels(self):
        runner = load_runner()
        positive = episode()
        positive["native_success"] = False
        label = {"dataset": "batch", "scene_id": "scene", "episode_id": "1",
            "runtime_system_error": 1, "reviewer": "独立标注者"}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", return_value=iter(frame(i) for i in range(4))):
                runner.run_evaluation([positive], {"sample_every": 1,
                    "repeat_threshold": 1}, directory, ["repeat_action"], [label])
            import csv
            with (Path(directory) / "main_table_excel.csv").open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["runtime_error_share_among_failures"], "1.0")
            self.assertEqual(row["system_failure_share"], "")

    def test_cli_help_runs_without_simulator_or_openai(self):
        script = ROOT / "scripts/evaluate_system_errors.py"
        self.assertTrue(script.is_file(), "Detector CLI missing")
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--labels", result.stdout)

    def test_expanded_run_appends_blank_labels_and_keeps_human_values(self):
        runner = load_runner()
        first = episode()
        second = {**episode(), "episode_id": "2", "native_success": False}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", side_effect=lambda *a: iter([frame(0)])):
                runner.run_evaluation([first], {}, directory, ["repeat_action"])
            path = Path(directory) / "manual_labels.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            rows[0].update(runtime_system_error="1", reviewer=" 人工核查员 ", reason="逐帧人工查看，保留原文")
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            with patch.object(runner, "iter_frames", side_effect=lambda *a: iter([frame(0)])):
                runner.run_evaluation([first, second], {}, directory, ["repeat_action"])
            with path.open(encoding="utf-8-sig", newline="") as handle:
                expanded = list(csv.DictReader(handle))
            self.assertEqual(len(expanded), 2)
            self.assertEqual(expanded[0], rows[0])
            self.assertEqual(expanded[1]["episode_id"], "2")
            self.assertEqual(expanded[1]["runtime_system_error"], "")
            self.assertEqual(expanded[1]["reviewer"], "")

    def test_scoring_reference_audit_records_exact_labels_and_unmatched_rows(self):
        runner = load_runner()
        label = {"dataset": "batch", "scene_id": "scene", "episode_id": "1",
            "runtime_system_error": "1", "reviewer": "人工复核", "onset_step": "12"}
        unmatched = {**label, "episode_id": "outside_selection"}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", return_value=iter([frame(0)])):
                runner.run_evaluation([episode()], {}, directory, ["repeat_action"], [label, unmatched])
            audit = json.loads((Path(directory) / "references_used.json").read_text())
            self.assertEqual(audit["unmatched_annotation_count"], 1)
            self.assertEqual(audit["references"][0]["runtime_system_error"], 1)
            self.assertEqual(audit["references"][0]["reviewer"], "人工复核")
            self.assertEqual(audit["references"][0]["onset_step"], 12)

    def test_threaded_progress_is_reported_for_each_completed_episode(self):
        runner = load_runner()
        observed = []
        episodes = [episode(), {**episode(), "episode_id": "2"}]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(runner, "iter_frames", side_effect=lambda *a: iter([frame(0)])):
                runner.run_evaluation(episodes, {}, directory, ["repeat_action"], workers=2,
                    progress=lambda done, total, ep: observed.append((done, total, ep["episode_id"])))
        self.assertEqual([(done, total) for done, total, _ in observed], [(1, 2), (2, 2)])
        self.assertEqual({identifier for _, _, identifier in observed}, {"1", "2"})

    @unittest.skipUnless(importlib.util.find_spec("dotenv"), "optional dotenv helper unavailable")
    def test_dotenv_loading_preserves_exported_variables(self):
        path = ROOT / "scripts/evaluate_system_errors.py"
        spec = importlib.util.spec_from_file_location("detector_cli_test", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        self.assertTrue(hasattr(cli, "load_environment"), "CLI dotenv loader missing")
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("FARO_DETECTION_TEST_EXISTING=replace\nFARO_DETECTION_TEST_NEW=loaded\n")
            with patch.dict(os.environ, {"FARO_DETECTION_TEST_EXISTING": "keep"}):
                os.environ.pop("FARO_DETECTION_TEST_NEW", None)
                cli.load_environment(env_file)
                self.assertEqual(os.environ["FARO_DETECTION_TEST_EXISTING"], "keep")
                self.assertEqual(os.environ["FARO_DETECTION_TEST_NEW"], "loaded")


if __name__ == "__main__":
    unittest.main()
