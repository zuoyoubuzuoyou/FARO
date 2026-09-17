"""Clean episode selection shares the fault launcher's validated dataset path."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from habitat_mas.fault_injection.cli import build_parser, main, prepare_run
from habitat_mas.fault_injection.launcher import read_dataset


class CleanLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "source.json"
        self.episodes = [
            {"episode_id": "20", "scene_id": "scene_a", "targets": [1, 2]},
            {"episode_id": "28", "scene_id": "scene_b", "targets": [3]},
        ]
        self.dataset.write_text(json.dumps({"config": {"keep": True}, "episodes": self.episodes}))

    def options(self, *options):
        return ["--dataset", str(self.dataset), "--max-episode-steps", "750",
                "--output-dir", str(self.root / "runs"), "--dry-run", *options]

    def plan(self, *options, overrides=None):
        return prepare_run(build_parser(clean=True).parse_args(self.options(*options)), overrides or [])

    def test_selects_id_keeps_episode_data_and_enables_new_clean_recording(self):
        original = self.dataset.read_bytes()
        _, command, plan = self.plan("--episode", "20", "--run-label", "clean_ep20")
        self.assertEqual(read_dataset(Path(plan["subset_dataset"])), {
            "config": {"keep": True}, "episodes": [self.episodes[0]],
        })
        self.assertEqual(self.dataset.read_bytes(), original)
        self.assertEqual(plan["mode"], "clean")
        self.assertIsNone(plan["fault"])
        self.assertIsNone(plan["agent"])
        self.assertIsNone(plan["outputs"]["events"])
        self.assertEqual(plan["episodes"], [{"episode_id": "20", "scene_id": "scene_a"}])
        self.assertTrue(plan["outputs"]["trajectory"].endswith("clean_ep20_trajectory.jsonl"))
        for fault in ("agent_dropout", "agent_hang"):
            self.assertIn(f"habitat_baselines.fault_injection.structure.{fault}.enabled=False", command)
        self.assertIn("habitat_baselines.trajectory.enabled=True", command)
        self.assertIn(f'habitat_baselines.output_dir="{self.root / "runs"}"', command)
        self.assertIn('habitat_baselines.run_label="clean_ep20"', command)
        self.assertIn('habitat_baselines.fault_injection.replay.path=""', command)
        self.assertIn("habitat_baselines.num_environments=1", command)
        self.assertIn("habitat_baselines.test_episode_count=-1", command)
        self.assertFalse(any("simulator_step_schedule=" in arg for arg in command))

    def test_explicit_emos_settings_are_preserved_and_conflicts_rejected(self):
        overrides = ["habitat.seed=42", "habitat_baselines.test_episode_count=1",
                     "habitat.environment.iterator_options.shuffle=True"]
        _, command, plan = self.plan("--episode", "20", overrides=overrides)
        for item in overrides:
            self.assertEqual(command.count(item), 1)
        self.assertEqual(plan["emos_overrides"], overrides)
        for setting in ("habitat_baselines.evaluate=False", "habitat_baselines.num_environments=2",
                        "habitat_baselines.test_episode_count=3"):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.plan("--episode", "20", overrides=[setting])

    def test_multiple_ids_and_unknown_duplicate_ambiguous_ids(self):
        _, _, plan = self.plan("--episode", "28,20")
        self.assertEqual([ep["episode_id"] for ep in plan["episodes"]], ["28", "20"])
        for ids in ("999", "20,20", "20,"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.plan("--episode", ids)
        self.dataset.write_text(json.dumps({"episodes": self.episodes + [{**self.episodes[0], "scene_id": "scene_c"}]}))
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.plan("--episode", "20")
        _, _, plan = self.plan("--episode", "20", "--scene", "scene_c")
        self.assertEqual(plan["episodes"], [{"episode_id": "20", "scene_id": "scene_c"}])

    def test_dry_run_does_not_execute_or_announce_fault_events(self):
        output = io.StringIO()
        with patch("habitat_mas.fault_injection.cli.subprocess.run") as run, contextlib.redirect_stdout(output):
            result = main(self.options("--episode", "20"), clean=True)
        self.assertEqual(result, 0)
        run.assert_not_called()
        self.assertIn("Clean evaluation", output.getvalue())
        self.assertNotIn("Events:", output.getvalue())

    def test_existing_clean_outputs_are_preserved(self):
        output = self.root / "runs" / "clean_preserved_trajectory.jsonl"
        output.parent.mkdir()
        output.write_text("existing clean trace")
        with self.assertRaises(FileExistsError):
            self.plan("--episode", "20", "--run-label", "clean_preserved")
        self.assertEqual(output.read_text(), "existing clean trace")

    def test_fault_options_rejected_by_clean_entry(self):
        for option in ("--fault", "--step", "--replay-trajectory"):
            with self.subTest(option=option):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    main(self.options("--episode", "20", option, "1"), clean=True)

    def test_clean_execution_uses_same_python_and_propagates_exit_code(self):
        settings = (self.dataset, 750, ("*",), ("agent_0", "agent_1"), True, {})
        with patch("habitat_mas.fault_injection.cli.resolve_environment", return_value=settings):
            with patch("habitat_mas.fault_injection.cli.subprocess.run", return_value=SimpleNamespace(returncode=7)) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main(["--episode", "20", "--output-dir", str(self.root / "runs")], clean=True)
        self.assertEqual(result, 7)
        self.assertIn("habitat_baselines.run", run.call_args.args[0])
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
