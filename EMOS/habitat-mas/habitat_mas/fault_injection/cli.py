"""Short command-line interface for EMOS fault evaluation."""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from habitat_mas.fault_injection.launcher import (
    FAULTS,
    fault_overrides,
    make_schedule,
    normalize_fault,
    read_dataset,
    read_trace_steps,
    select_episodes,
)
from habitat_mas.fault_injection.naming import output_run_stem
from habitat_mas.fault_injection.replay import CleanPrefixReplay

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = "multi_rearrange/llm_spot_drone_per.yaml"
OUTPUT_KEYS = (
    "habitat_baselines.trajectory.enabled",
    "habitat_baselines.output_dir",
    "habitat_baselines.run_label",
)


def build_parser(*, clean: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record clean trajectories for selected episodes." if clean else "Inject one fault per selected episode, at a simulator step.",
        allow_abbrev=False,
        epilog="Original EMOS key=value arguments pass through unchanged; -- is optional.",
    )
    parser.set_defaults(clean=clean)
    if clean:
        parser.set_defaults(fault="clean", agent=None, step=None, trajectory=None,
                            replay_trajectory=None, replay_atol=1e-5)
    else:
        parser.add_argument("--fault", required=True, type=normalize_fault,
                            choices=list(FAULTS), help="Registered structure and latent execution faults")
        parser.add_argument("--agent", default="agent_0", help="Target agent, default: agent_0")
        parser.add_argument("--duration-steps", type=int, help="Fault window length; -1 means until episode end")
        parser.add_argument("--object", help="Affected PDDL object, e.g. any_targets|0")
        parser.add_argument("--replacement-object", help="Wrong subgoal target (a different scene entity)")
        parser.add_argument("--delay-steps", type=int, help="Message delivery delay in simulator steps")
        parser.add_argument("--position-offset", type=float, nargs=3, metavar=("DX", "DY", "DZ"), help="Localization bias in world metres (Y is vertical)")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--episode", help="Dataset episode ID, or comma-separated IDs")
    selection.add_argument("--random-episodes", type=int, metavar="N",
                           help="Sample N distinct episode IDs and one scene variant per ID")
    parser.add_argument("--scene", help="Disambiguate episode IDs using a scene path or filename stem")
    if not clean:
        trigger = parser.add_mutually_exclusive_group()
        trigger.add_argument("--step", type=int, help="Fixed zero-based simulator step")
        trigger.add_argument("--trajectory", "--steps-from-trajectory", dest="trajectory", type=Path, metavar="JSONL",
                             help="Sample recorded steps from complete clean episode traces")
        parser.add_argument("--replay-trajectory", type=Path, metavar="CLEAN_JSONL",
                            help="Replay and verify a clean prefix before --step, then run the fault policy live")
        parser.add_argument("--replay-atol", type=float, default=1e-5,
                            help="Absolute tolerance for replay action/state/metric checks (default: 1e-5)")
    parser.add_argument("--seed", type=int, default=0, help="Episode sampling seed only; EMOS seeds remain unchanged (default: 0)" if clean else "Fault sampling seed only; EMOS seeds remain unchanged (default: 0)")
    parser.add_argument("--config", "--config-name", default=DEFAULT_CONFIG,
                        help="Habitat baseline configuration name")
    parser.add_argument("--dataset", type=Path, help="Override the config's episode JSON/JSON.gz")
    parser.add_argument("--max-episode-steps", type=int,
                        help="Override the environment step limit (also enables offline dry-run with --dataset)")
    parser.add_argument("--output-dir", type=Path,
                        help="Shared trajectory/event/crash directory (default: fault_output)")
    parser.add_argument("--run-label", help="Output filename label and plan subdirectory; default: run type + timestamp")
    parser.add_argument("--no-trajectory", action="store_true", help="Disable trajectory recording" if clean else "Only record fault events")
    parser.add_argument("--dry-run", action="store_true", help="Write subset/plan and print command without evaluating")
    return parser


def override_values(overrides: list[str]) -> dict[str, str]:
    return {
        item.partition("=")[0].lstrip("+~"): item.partition("=")[2]
        for item in overrides
    }


def scalar_value(value: str):
    """Read simple explicit values for offline checks, not Hydra composition."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return json.loads(value)
    except ValueError:
        return value.strip("\"'")


def validate_overrides(overrides: list[str]) -> None:
    # Only the fault subtree is owned by this launcher. All other EMOS
    # parameters retain their usual Hydra spelling and precedence.
    prefix = "habitat_baselines.fault_injection"
    for override in overrides:
        key, separator, _ = override.partition("=")
        key = key.lstrip("+~")
        if not separator or not key or override.startswith("-"):
            raise ValueError(f"Expected a Hydra key=value override, got {override!r}")
        if key == prefix or key.startswith(prefix + ".") or prefix.startswith(key + "."):
            raise ValueError(f"{key} conflicts with the short fault-injection options")
        if key == "habitat_baselines.trajectory" or (override.startswith("~") and key in OUTPUT_KEYS):
            raise ValueError(f"Set {OUTPUT_KEYS[0]}, output_dir and run_label individually; cannot replace/delete {key}")


def effective_overrides(args, overrides: list[str]) -> list[str]:
    """Optional aliases supplement, but never overwrite, original arguments."""
    result = list(overrides)
    existing = override_values(overrides)
    aliases = {
        "habitat.dataset.data_path": None if args.dataset is None else str(args.dataset.resolve()),
        "habitat.environment.max_episode_steps": args.max_episode_steps,
        "habitat_baselines.output_dir": None if args.output_dir is None else str(args.output_dir.resolve()),
        "habitat_baselines.run_label": args.run_label,
        "habitat_baselines.trajectory.enabled": False if args.no_trajectory else None,
    }
    for key, value in aliases.items():
        if value is None:
            continue
        if key in existing:
            actual = scalar_value(existing[key])
            if key in ("habitat.dataset.data_path", "habitat_baselines.output_dir"):
                actual = str(Path(actual).resolve())
            if actual != value:
                raise ValueError(f"Conflicting values for {key}; use its EMOS argument or the short alias")
        else:
            result.append(f"{key}={json.dumps(value)}")
    return result


def output_settings(overrides: list[str]) -> tuple[bool, Path, str]:
    """Read shared output options before planning or writing any files."""
    values = override_values(overrides)
    enabled = scalar_value(values[OUTPUT_KEYS[0]])
    directory = scalar_value(values[OUTPUT_KEYS[1]])
    label = scalar_value(values[OUTPUT_KEYS[2]])
    if type(enabled) is not bool:
        raise ValueError(f"{OUTPUT_KEYS[0]} must be True or False")
    if not isinstance(directory, str) or not directory:
        raise ValueError(f"{OUTPUT_KEYS[1]} must be a nonempty directory path")
    if not isinstance(label, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label) is None:
        raise ValueError("--run-label / habitat_baselines.run_label must begin with a letter/digit and use only letters, digits, _, -, .")
    return enabled, Path(directory).resolve(), label


EVAL_KEYS = {
    "habitat_baselines.evaluate": True,
    "habitat_baselines.num_environments": 1,
    "habitat_baselines.eval.evals_per_ep": 1,
    "habitat_baselines.load_resume_state_config": False,
}
COUNT_KEY = "habitat_baselines.test_episode_count"
SAMPLE_KEY = "habitat.environment.iterator_options.num_episode_sample"


def validate_eval_settings(settings: dict, episode_count: int | None = None) -> None:
    for key, expected in EVAL_KEYS.items():
        if key in settings and settings[key] != expected:
            raise ValueError(
                f"Fault injection requires {key}={expected}; "
                f"received {settings[key]!r}. The launcher will not override it."
            )
    if episode_count is not None:
        for key in (COUNT_KEY, SAMPLE_KEY):
            if key in settings and settings[key] not in (-1, episode_count):
                raise ValueError(
                    f"{key}={settings[key]} conflicts with {episode_count} selected episodes; "
                    f"set it to {episode_count} or -1. The launcher will not override it."
                )


def resolve_environment(args, overrides: list[str]):
    # Offline planning validates explicit simple values only. Real runs use
    # the original Hydra composition, with no evaluation/iterator defaults added.
    explicit = {key: scalar_value(value) for key, value in override_values(overrides).items()}
    dataset_key = "habitat.dataset.data_path"
    limit_key = "habitat.environment.max_episode_steps"
    if args.dry_run and dataset_key in explicit and limit_key in explicit:
        if "habitat.dataset.content_scenes" in explicit:
            raise ValueError("Offline dry-run cannot resolve content_scenes; use --scene or an installed Habitat environment")
        dataset_path = str(explicit[dataset_key])
        if "{" in dataset_path:
            raise ValueError("Offline dry-run needs a dataset path without Hydra interpolation or {split}")
        settings = {key: explicit[key] for key in (*EVAL_KEYS, COUNT_KEY, SAMPLE_KEY) if key in explicit}
        validate_eval_settings(settings)
        return Path(dataset_path).resolve(), int(explicit[limit_key]), ("*",), None, False, settings
    for directory in ("habitat-lab", "habitat-baselines"):
        sys.path.insert(0, str(ROOT / directory))
    try:
        from habitat_baselines.config.default import get_config
    except ImportError as error:
        raise ValueError(
            f"Habitat configuration dependencies are unavailable ({error}). "
            "Use the EMOS environment, or --dry-run with habitat.dataset.data_path=FILE "
            "and habitat.environment.max_episode_steps=N for offline planning."
        ) from error
    config = get_config(args.config, overrides=overrides)
    baseline = config.habitat_baselines
    settings = {
        "habitat_baselines.evaluate": baseline.evaluate,
        "habitat_baselines.num_environments": baseline.num_environments,
        "habitat_baselines.eval.evals_per_ep": baseline.eval.evals_per_ep,
        "habitat_baselines.load_resume_state_config": baseline.load_resume_state_config,
        COUNT_KEY: baseline.test_episode_count,
        SAMPLE_KEY: config.habitat.environment.iterator_options.num_episode_sample,
    }
    validate_eval_settings(settings)
    dataset_config = config.habitat.dataset
    dataset_path = Path(dataset_config.data_path.format(split=baseline.eval.split))
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    return (
        dataset_path.resolve(), int(config.habitat.environment.max_episode_steps),
        tuple(dataset_config.content_scenes), tuple(config.habitat.simulator.agents), True,
        settings,
    )


def fault_parameters(args) -> dict:
    applicable = {
        "action_noop": {"duration_steps"},
        "wrong_subgoal": {"object", "replacement_object"},
        "message_delay": {"duration_steps", "delay_steps"},
        "object_localization": {"duration_steps", "object", "position_offset"},
    }
    supplied = {key: getattr(args, key, None) for key in (
        "duration_steps", "object", "replacement_object", "delay_steps", "position_offset")}
    supplied = {key: value for key, value in supplied.items() if value is not None}
    unsupported = set(supplied) - applicable.get(args.fault, set())
    if unsupported:
        raise ValueError(f"Parameters do not apply to {args.fault}: {', '.join(sorted(unsupported))}")
    required = {"wrong_subgoal": {"object", "replacement_object"},
                "object_localization": {"object", "position_offset"},
                "message_delay": {"delay_steps"}}.get(args.fault, set())
    if required - set(supplied):
        raise ValueError(f"{args.fault} requires {', '.join('--' + key.replace('_', '-') for key in sorted(required - set(supplied)))}")
    if args.fault in applicable:
        defaults = FAULTS[args.fault].spec_type()
        supplied = {key: supplied.get(key, getattr(defaults, key)) for key in sorted(applicable[args.fault])}
    return supplied


def prepare_run(args, overrides: list[str]) -> tuple[Path, list[str], dict]:
    validate_overrides(overrides)
    original_overrides = list(overrides)
    parameters = {} if args.clean else fault_parameters(args)
    overrides = effective_overrides(args, overrides)
    # Both convenience entry points record by default; explicit common options
    # or their short aliases take precedence over these launcher defaults.
    existing = override_values(overrides)
    output_defaults = {
        OUTPUT_KEYS[0]: True,
        OUTPUT_KEYS[1]: "fault_output",
        OUTPUT_KEYS[2]: output_run_stem(datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"), args.fault),
    }
    overrides += [f"{key}={json.dumps(value)}" for key, value in output_defaults.items() if key not in existing]
    record_trajectory, output_dir, label = output_settings(overrides)
    if args.clean:
        # The clean convenience entry point supplies evaluation defaults only
        # when the caller has not explicitly provided that setting.
        existing = override_values(overrides)
        overrides += [
            f"{key}={json.dumps(value)}"
            for key, value in {**EVAL_KEYS, COUNT_KEY: -1}.items()
            if key not in existing
        ]
    if not args.clean and re.fullmatch(r"agent_\d+", args.agent) is None:
        raise ValueError("--agent must use agent_N, for example agent_1")
    if args.seed < 0 or args.seed >= 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if not args.clean and args.step is None and args.trajectory is None:
        raise ValueError("Specify --step, or --trajectory CLEAN.jsonl to sample real episode steps")
    if args.replay_trajectory and (args.step is None or args.trajectory is not None):
        raise ValueError("--replay-trajectory requires --step; --trajectory only samples steps")
    if args.replay_trajectory and not record_trajectory:
        raise ValueError("Replay requires trajectory recording; remove --no-trajectory and set habitat_baselines.trajectory.enabled=True")
    replay = CleanPrefixReplay(args.replay_trajectory, atol=args.replay_atol) if args.replay_trajectory else None
    dataset_path, max_steps, scenes, agents, config_checked, eval_settings = resolve_environment(args, overrides)
    if not args.clean and agents is not None and args.agent not in agents:
        raise ValueError(f"Unknown agent {args.agent!r}; configured agents: {', '.join(agents)}")
    data = read_dataset(dataset_path)
    trace_steps = read_trace_steps(args.trajectory) if args.trajectory else None
    candidates = data["episodes"]
    if trace_steps is not None:
        candidates = [ep for ep in candidates if (str(ep["scene_id"]), str(ep["episode_id"])) in trace_steps]
    if replay is not None:
        candidates = [ep for ep in candidates if (str(ep["scene_id"]), str(ep["episode_id"])) in replay.episodes]
    selected = select_episodes(
        candidates,
        episode_ids=None if args.episode is None else [value.strip() for value in args.episode.split(",")],
        random_count=args.random_episodes, seed=args.seed, scene=args.scene,
        content_scenes=scenes,
    )
    validate_eval_settings(eval_settings, len(selected))
    schedule = {} if args.clean else make_schedule(
        selected, seed=args.seed, max_episode_steps=max_steps,
        step=args.step, trace_steps=trace_steps,
    )
    if replay is not None:
        for ep in selected:
            replay.validate_episode(ep["scene_id"], ep["episode_id"], schedule[str(ep["episode_id"])])
    run_dir = output_dir / label
    stem = output_run_stem(label, args.fault)
    trace_episode_ids = [
        re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ep["episode_id"]))
        for ep in selected
    ]
    outputs = {
        "crash_report": str(output_dir / f"{stem}_crash.json"),
        "events": None if args.clean else str(output_dir / f"{stem}_events.jsonl"),
        "summary": str(output_dir / f"{stem}_summary.json") if args.fault in (
            "action_noop", "wrong_subgoal", "message_delay", "object_localization") else None,
        "trajectory": str(output_dir / f"{stem}_trajectory.jsonl") if record_trajectory else None,
        "traces": [
            str(output_dir / f"{stem}_trace-episode_{episode_id}.txt")
            for episode_id in trace_episode_ids
        ] if record_trajectory else [],
    }
    for path in [outputs["events"], outputs["summary"], outputs["trajectory"], outputs["crash_report"], *outputs["traces"]]:
        if path is not None and Path(path).exists():
            raise FileExistsError(f"Output already exists: {path}; choose another --run-label")
    subset_path = run_dir / "episodes.json.gz"
    command = [sys.executable, "-u", "-m", "habitat_baselines.run", f"--config-name={args.config}"]
    # The original data_path identifies the source dataset. Execution uses its
    # selected subset. Shared output arguments are normalized to match the plan;
    # all remaining EMOS arguments pass through verbatim.
    command += [
        item for item in overrides
        if item.partition("=")[0].lstrip("+~") not in ("habitat.dataset.data_path", *OUTPUT_KEYS)
    ]
    command.append(f"habitat.dataset.data_path={json.dumps(str(subset_path))}")
    command += [
        f"habitat_baselines.trajectory.enabled={record_trajectory}",
        f"habitat_baselines.output_dir={json.dumps(str(output_dir))}",
        f"habitat_baselines.run_label={json.dumps(label)}",
    ]
    if args.clean:
        for definition in FAULTS.values():
            command += [f"{definition.config_path}.enabled=False"]
        command.append('habitat_baselines.fault_injection.replay.path=""')
    else:
        command += fault_overrides(args.fault, args.agent, schedule, parameters)
    if replay is not None:
        command += [
            f"habitat_baselines.fault_injection.replay.path={json.dumps(replay.path)}",
            f"habitat_baselines.fault_injection.replay.atol={args.replay_atol}",
        ]
    plan = {
        "schema_version": 1, "mode": "clean" if args.clean else "faulty",
        "fault": None if args.clean else args.fault, "agent": args.agent,
        "fault_parameters": parameters,
        "seed": args.seed, "config": args.config, "config_validated": config_checked,
        "emos_overrides": original_overrides,
        "source_dataset": str(dataset_path), "subset_dataset": str(subset_path),
        "max_episode_steps": max_steps,
        "trajectory": str(args.trajectory.resolve()) if args.trajectory else None,
        "replay": None if replay is None else {"path": replay.path, "atol": replay.atol},
        "episodes": [
            {"episode_id": str(ep["episode_id"]), "scene_id": str(ep["scene_id"]),
             **({} if args.clean else {"simulator_step": schedule[str(ep["episode_id"])]})}
            for ep in selected
        ],
        "command": command, "cwd": str(ROOT),
        "outputs": outputs,
    }
    # Refuse to replace an existing run, including its events and dataset.
    run_dir.mkdir(parents=True, exist_ok=False)
    subset = {**data, "episodes": selected}
    subset.pop("content_scenes_path", None)
    with gzip.open(subset_path, "wt", encoding="utf-8") as output:
        json.dump(subset, output, ensure_ascii=False)
    plan_path = run_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan_path, command, plan


def main(argv: list[str] | None = None, *, clean: bool = False) -> int:
    parser = build_parser(clean=clean)
    args, overrides = parser.parse_known_args(argv)
    if overrides[:1] == ["--"]:
        overrides = overrides[1:]
    try:
        plan_path, command, plan = prepare_run(args, overrides)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Clean evaluation; episode sampling seed: {args.seed}" if clean else f"Fault: {args.fault}; agent: {args.agent}; seed: {args.seed}")
    for episode in plan["episodes"]:
        step_text = "" if clean else f" step={episode['simulator_step']}"
        print(f"  episode={episode['episode_id']}{step_text} scene={episode['scene_id']}")
    print(f"Plan: {plan_path}")
    print(f"Crash report (if detected): {plan['outputs']['crash_report']}")
    if plan["replay"] is not None:
        print(f"Replay clean prefix: {plan['replay']['path']} (atol={plan['replay']['atol']}); abort on divergence")
    if plan["outputs"]["events"] is not None:
        print(f"Events: {plan['outputs']['events']}")
    if plan["outputs"]["summary"] is not None:
        print(f"Fault effect summary: {plan['outputs']['summary']}")
    if plan["outputs"]["trajectory"] is not None:
        print(f"Trajectory JSONL: {plan['outputs']['trajectory']}")
        for trace in plan["outputs"]["traces"]:
            print(f"Episode trace TXT: {trace}")
    if not plan["config_validated"]:
        print("Offline plan: Hydra configuration and agent availability have not been validated.")
    if not clean and args.step is None:
        print("Steps were sampled from complete clean trajectories. A shorter rerun ending before its selected step is reported as a missed injection.")
    if args.dry_run:
        print(f"Run from {ROOT}:")
        print(shlex.join(command))
        return 0
    environment = os.environ.copy()
    paths = [str(ROOT / name) for name in ("habitat-mas", "habitat-lab", "habitat-baselines")]
    environment["PYTHONPATH"] = os.pathsep.join(paths + [environment.get("PYTHONPATH", "")])
    sys.stdout.flush()
    try:
        return subprocess.run(command, cwd=ROOT, env=environment, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
