import contextlib
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from habitat_mas.fault_injection.cli import (
    build_parser, main, prepare_run, resolve_environment, validate_overrides,
    effective_overrides, validate_eval_settings,
)
from habitat_mas.fault_injection.launcher import (
    fault_overrides, make_schedule, normalize_fault, read_dataset,
    read_trace_steps, select_episodes,
)
from habitat_mas.fault_injection.trajectory import AgentDropoutTrajectoryRecorder


def episode(number, scene="scene_a"):
    return {"episode_id": str(number), "scene_id": scene, "metadata": [number]}


class FaultLauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset.json.gz"
        self.episodes = [episode(1), episode(2), episode(3)]
        with gzip.open(self.dataset, "wt") as output:
            json.dump({"config": {"keep": True}, "episodes": self.episodes}, output)
        self.trace = self.root / "clean.jsonl"
        rows = []
        for ep, length in zip(self.episodes, (1, 3, 8)):
            rows.extend({
                "type": "simulator_step", "mode": "clean",
                "episode_id": ep["episode_id"], "scene_id": ep["scene_id"],
                "simulator_step": step, "done": step == length - 1,
            } for step in range(length))
        self.trace.write_text("\n".join(json.dumps(row) for row in rows))

    def args(self, *options):
        return build_parser().parse_args([
            "--fault", "hang", "--dry-run", "--dataset", str(self.dataset),
            "--max-episode-steps", "10", "--output-dir", str(self.root / "runs"),
            *options,
        ])

    def test_fault_aliases(self):
        for alias in ("dropout", "agent dropout", "agent-dropout", "agent_dropout"):
            self.assertEqual(normalize_fault(alias), "agent_dropout")
        for alias in ("hang", "agent hang", "agent-hang", "agent_hang"):
            self.assertEqual(normalize_fault(alias), "agent_hang")
        with self.assertRaises(ValueError):
            normalize_fault("unknown")

    def test_fixed_run_filters_dataset_and_builds_schedule(self):
        args = self.args("--episode", "2", "--step", "4", "--run-label", "fixed")
        path, command, plan = prepare_run(args, [])
        subset = read_dataset(Path(plan["subset_dataset"]))
        self.assertEqual(subset, {"config": {"keep": True}, "episodes": [episode(2)]})
        self.assertEqual(read_dataset(self.dataset)["episodes"], self.episodes)
        self.assertEqual(plan["episodes"][0]["simulator_step"], 4)
        self.assertEqual(json.loads(path.read_text()), plan)
        self.assertFalse(any("habitat_baselines.test_episode_count=" in item for item in command))
        self.assertIn('habitat_baselines.fault_injection.structure.agent_hang.simulator_step_schedule="2@4"', command)
        self.assertFalse(any("agent_hang.trigger_domain" in value for value in command))
        self.assertIn("habitat_baselines.fault_injection.structure.agent_dropout.enabled=False", command)
        with self.assertRaises(FileExistsError):
            prepare_run(args, [])

    def test_fault_outputs_use_clean_layout_and_announced_paths(self):
        for fault in ("dropout", "hang"):
            with self.subTest(fault=fault):
                args = self.args("--fault", fault, "--episode", "2", "--step", "0",
                                 "--run-label", f"agent_{fault}_comparison")
                plan_path, command, plan = prepare_run(args, [])
                prefix = f"habitat_baselines.fault_injection.structure.agent_{fault}"
                values = dict(arg.split("=", 1) for arg in command if arg.startswith(prefix))
                recorder = AgentDropoutTrajectoryRecorder(
                    json.loads(values[f"{prefix}.output_dir"]),
                    json.loads(values[f"{prefix}.run_label"]), "faulty",
                    agent_names=["agent_0"], fault_name=f"agent_{fault}",
                )
                self.assertEqual(recorder.path, self.root / "runs" / f"{fault}_comparison_trajectory.jsonl")
                self.assertEqual(str(recorder.path), plan["outputs"]["trajectory"])
                self.assertEqual(str(recorder.text_path_for_episode("2")), plan["outputs"]["traces"][0])
                self.assertNotEqual(recorder.path.parent, plan_path.parent)
                recorder.record_episode_start(episode_id="2", scene_id="scene_a", text_context={})
                recorder.record_step(
                    episode_id="2", scene_id="scene_a", simulator_step=0,
                    joint_env_action=[0], policy_info={}, reward=0, done=True, metrics={},
                )
                row = json.loads(recorder.path.read_text().splitlines()[-1])
                self.assertEqual(row["type"], "simulator_step")
                self.assertTrue({"agent_step_actions", "simulator_state", "joint_env_action"} <= row.keys())
                self.assertIn("Simulator_Step: 0", recorder.text_path_for_episode("2").read_text())

    def test_existing_flat_outputs_are_not_overwritten(self):
        directory = self.root / "runs"
        directory.mkdir()
        for suffix in ("trajectory.jsonl", "events.jsonl", "trace-episode_1.txt"):
            with self.subTest(suffix=suffix):
                existing = directory / f"hang_preserved_{suffix}"
                existing.write_text("existing output")
                with self.assertRaisesRegex(FileExistsError, "choose another --run-label"):
                    prepare_run(self.args("--episode", "1", "--step", "0", "--run-label", "preserved"), [])
                self.assertEqual(existing.read_text(), "existing output")
                self.assertFalse((directory / "preserved").exists())
                existing.unlink()

    def test_no_trajectory_does_not_announce_nonexistent_trace_files(self):
        _, command, plan = prepare_run(self.args("--episode", "1", "--step", "0", "--no-trajectory"), [])
        self.assertIsNone(plan["outputs"]["trajectory"])
        self.assertEqual(plan["outputs"]["traces"], [])
        self.assertIn("habitat_baselines.fault_injection.structure.agent_hang.record_trajectory=False", command)

    def test_random_plans_reproduce_and_follow_each_clean_length(self):
        results = []
        for label in ("first", "second"):
            _, _, plan = prepare_run(self.args(
                "--random-episodes", "3", "--trajectory", str(self.trace),
                "--seed", "42", "--run-label", label,
            ), [])
            results.append(plan["episodes"])
        self.assertEqual(results[0], results[1])
        lengths = {"1": 1, "2": 3, "3": 8}
        self.assertEqual({row["episode_id"] for row in results[0]}, set(lengths))
        for row in results[0]:
            self.assertGreaterEqual(row["simulator_step"], 0)
            self.assertLess(row["simulator_step"], lengths[row["episode_id"]])

    def test_random_pool_only_includes_matching_complete_trace(self):
        # Same ID in another scene must not match the clean trace.
        data = {"episodes": [episode(1, "other_scene"), episode(2), episode(9)]}
        with gzip.open(self.dataset, "wt") as output:
            json.dump(data, output)
        args = self.args("--random-episodes", "1", "--trajectory", str(self.trace))
        _, _, plan = prepare_run(args, [])
        self.assertEqual([ep["episode_id"] for ep in plan["episodes"]], ["2"])
        with self.assertRaisesRegex(ValueError, "between 1 and 1"):
            prepare_run(self.args("--random-episodes", "2", "--trajectory", str(self.trace)), [])

    def test_episode_disambiguation_and_random_unique_ids(self):
        candidates = [episode(1), episode(1, "scene_b"), episode(2)]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            select_episodes(candidates, episode_ids=["1"], random_count=None, seed=0)
        selected = select_episodes(candidates, episode_ids=["1"], random_count=None, seed=0, scene="scene_b")
        self.assertEqual(selected, [episode(1, "scene_b")])
        selected = select_episodes(candidates, episode_ids=None, random_count=2, seed=0)
        self.assertEqual({ep["episode_id"] for ep in selected}, {"1", "2"})
        with self.assertRaises(ValueError):
            select_episodes(candidates, episode_ids=None, random_count=3, seed=0)

    def test_trace_rejects_faulty_duplicate_and_missing_steps(self):
        original = self.trace.read_text()
        rows = [json.loads(line) for line in original.splitlines()]
        variants = [
            [{**rows[0], "mode": "faulty"}],
            [rows[0], rows[0]],
            [{**rows[0], "simulator_step": 2}],
        ]
        for index, variant in enumerate(variants):
            with self.subTest(index=index):
                self.trace.write_text("\n".join(json.dumps(row) for row in variant))
                with self.assertRaises(ValueError):
                    read_trace_steps(self.trace)
        self.trace.write_text(json.dumps({**rows[0], "done": False}))
        self.assertEqual(read_trace_steps(self.trace), {})

    def test_step_validation_and_no_silent_random_fallback(self):
        for step in (-1, 10):
            with self.assertRaises(ValueError):
                make_schedule(self.episodes, seed=0, max_episode_steps=10, step=step)
        with self.assertRaises(ValueError):
            make_schedule(self.episodes, seed=0, max_episode_steps=10)
        with self.assertRaisesRegex(ValueError, "--trajectory"):
            prepare_run(self.args("--random-episodes", "2"), [])
        with self.assertRaisesRegex(ValueError, "No complete clean trajectory"):
            make_schedule([episode(9)], seed=0, max_episode_steps=10, trace_steps=read_trace_steps(self.trace))

    def test_dropout_configuration_and_safe_string_arguments(self):
        args = fault_overrides("agent_dropout", "agent_1", {"15": 4, "28": 6}, self.root / "space path", "test", True)
        self.assertIn("habitat_baselines.fault_injection.structure.agent_dropout.trigger_domain=simulator_step", args)
        self.assertIn('habitat_baselines.fault_injection.structure.agent_dropout.simulator_step_schedule="15@4;28@6"', args)
        output_arg = next(arg for arg in args if ".output_dir=" in arg)
        self.assertEqual(json.loads(output_arg.split("=", 1)[1]), str(self.root / "space path"))

    def test_reserved_overrides_cannot_invalidate_plan(self):
        for override in (
            "habitat_baselines.fault_injection.structure.agent_hang.enabled=False",
            "habitat_baselines={}", "--multirun",
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_overrides([override])
        validate_overrides(["habitat_baselines.eval.video_option=[]", "habitat.dataset.content_scenes=['scene_a']"])

    def test_cli_dry_run_never_starts_evaluation(self):
        with patch("habitat_mas.fault_injection.cli.subprocess.run") as run:
            with contextlib.redirect_stdout(io.StringIO()):
                result = main([
                    "--fault", "agent dropout", "--episode", "1", "--step", "0",
                    "--dataset", str(self.dataset), "--max-episode-steps", "10",
                    "--output-dir", str(self.root / "out"), "--dry-run",
                ])
            self.assertEqual(result, 0)
            run.assert_not_called()

    def test_cli_executes_argument_vector_and_propagates_failure(self):
        resolved = (self.dataset, 10, ("*",), ("agent_0", "agent_1"), True, {})
        with patch("habitat_mas.fault_injection.cli.resolve_environment", return_value=resolved):
            with patch("habitat_mas.fault_injection.cli.subprocess.run", return_value=SimpleNamespace(returncode=7)) as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = main([
                        "--fault", "hang", "--episode", "1", "--step", "0",
                        "--output-dir", str(self.root / "out"),
                        "--", "habitat_baselines.eval.video_option=[]",
                    ])
        self.assertEqual(result, 7)
        command = run.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertIn("habitat_baselines.eval.video_option=[]", command)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_real_config_resolution_uses_eval_split_and_scene_filter(self):
        args = self.args("--episode", "1", "--step", "0")
        args.dry_run = False
        args.dataset = None
        config = SimpleNamespace(
            habitat=SimpleNamespace(
                dataset=SimpleNamespace(data_path="/data/{split}.json.gz", content_scenes=["scene_a"]),
                environment=SimpleNamespace(max_episode_steps=90, iterator_options=SimpleNamespace(num_episode_sample=-1)),
                simulator=SimpleNamespace(agents={"agent_0": {}}),
            ),
            habitat_baselines=SimpleNamespace(
                evaluate=True, num_environments=1, load_resume_state_config=False,
                test_episode_count=-1, eval=SimpleNamespace(split="val", evals_per_ep=1),
            ),
        )
        with patch.dict("sys.modules", {"habitat_baselines.config.default": SimpleNamespace(get_config=lambda *a, **kw: config)}):
            path, limit, scenes, agents, checked, settings = resolve_environment(args, [])
        self.assertEqual((path, limit, scenes, agents, checked), (Path("/data/val.json.gz"), 90, ("scene_a",), ("agent_0",), True))

    def test_original_emos_arguments_pass_through_without_replacement(self):
        overrides = [
            "habitat_baselines.evaluate=True",
            "habitat_baselines.num_environments=1",
            "habitat_baselines.test_episode_count=1",
            "habitat.environment.iterator_options.shuffle=False",
            "habitat.environment.iterator_options.group_by_scene=True",
            "habitat.seed=123", "habitat.simulator.seed=456",
            "habitat_baselines.eval.should_load_ckpt=True",
            "habitat_baselines.eval.video_option=[]",
        ]
        args = self.args("--episode", "1", "--step", "0", "--seed", "42")
        _, command, plan = prepare_run(args, overrides)
        for item in overrides:
            self.assertEqual(command.count(item), 1)
            key = item.split("=", 1)[0]
            self.assertEqual(sum(arg.startswith(key + "=") for arg in command), 1)
        self.assertEqual(plan["emos_overrides"], overrides)
        self.assertNotIn("habitat.seed=42", command)
        self.assertFalse(any("evaluator._target_=" in item for item in command))

    def test_conflicts_reported_before_writing_plan(self):
        for override in (
            "habitat_baselines.evaluate=False",
            "habitat_baselines.num_environments=2",
            "habitat_baselines.test_episode_count=2",
            "habitat_baselines.load_resume_state_config=True",
        ):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "will not override"):
                prepare_run(self.args("--episode", "1", "--step", "0"), [override])
        self.assertFalse((self.root / "runs").exists())

    def test_shuffle_true_and_all_episode_count_are_preserved(self):
        overrides = ["habitat.environment.iterator_options.shuffle=True", "habitat_baselines.test_episode_count=-1"]
        _, command, _ = prepare_run(self.args("--episode", "1", "--step", "0"), overrides)
        for item in overrides:
            self.assertIn(item, command)
        self.assertNotIn("habitat.environment.iterator_options.shuffle=False", command)

    def test_dataset_and_step_limit_original_spelling_work_offline(self):
        args = self.args("--episode", "1", "--step", "0")
        args.dataset = None
        args.max_episode_steps = None
        overrides = [
            f"habitat.dataset.data_path={self.dataset}",
            "habitat.environment.max_episode_steps=10",
        ]
        _, command, plan = prepare_run(args, overrides)
        self.assertEqual(plan["source_dataset"], str(self.dataset))
        self.assertIn("habitat.environment.max_episode_steps=10", command)
        self.assertEqual(plan["emos_overrides"], overrides)

    def test_config_name_and_interleaved_emos_arguments(self):
        args, overrides = build_parser().parse_known_args([
            "--config-name=multi_rearrange/llm_spot_drone_per.yaml",
            "habitat_baselines.evaluate=True", "--fault", "hang",
            "habitat_baselines.num_environments=1", "--episode", "28",
            "habitat_baselines.test_episode_count=1", "--step", "40",
            "habitat.environment.iterator_options.shuffle=False",
        ])
        self.assertEqual(args.config, "multi_rearrange/llm_spot_drone_per.yaml")
        self.assertEqual(len(overrides), 4)
        validate_overrides(overrides)

    def test_short_alias_cannot_override_original_setting(self):
        args = self.args("--episode", "1", "--step", "0")
        with self.assertRaisesRegex(ValueError, "Conflicting values"):
            effective_overrides(args, ["habitat.environment.max_episode_steps=20"])
        values = effective_overrides(args, ["habitat.environment.max_episode_steps=10"])
        self.assertEqual(values.count("habitat.environment.max_episode_steps=10"), 1)


if __name__ == "__main__":
    unittest.main()
