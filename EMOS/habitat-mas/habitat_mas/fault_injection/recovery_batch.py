"""Frozen clean-success selection and commands for paired dropout experiments.

This module deliberately imports no simulator or model packages.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GROUPS = ("clean", "dropout", "dropout_recovery")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_fingerprint():
    """Include local edits, configs and analysis code, without storing secrets."""
    result = hashlib.sha256()
    for directory in ("habitat-mas/habitat_mas", "habitat-lab/habitat",
                      "habitat-baselines/habitat_baselines", "scripts", "habitat-mas/data/robot_resume",
                      "data/robots/robot_configs"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".yml", ".json"):
                result.update(str(path.relative_to(ROOT)).encode())
                result.update(bytes.fromhex(digest(path)))
    return result.hexdigest()


def select_clean(path, fraction=0.5):
    if not 0 < fraction < 1:
        raise ValueError("injection fraction must be between 0 and 1")
    episodes = {}
    for row in read_rows(path):
        if row.get("type") not in ("episode_start", "simulator_step", "system_crash"):
            continue
        if row.get("mode") != "clean":
            raise ValueError("Selection requires clean trajectories")
        key = (str(row["scene_id"]), str(row["episode_id"]))
        if row["type"] == "episode_start":
            if key in episodes:
                raise ValueError(f"Repeated episode: {key}")
            episodes[key] = {"length": 0, "done": False, "success": False}
            continue
        ep = episodes.setdefault(key, {"length": 0, "done": False, "success": False})
        if row["type"] == "system_crash":
            ep.update(done=True, success=False)
            continue
        if ep["done"] or row["simulator_step"] != ep["length"]:
            raise ValueError(f"Repeated, missing or unordered steps: {key}")
        ep["length"] += 1
        if row.get("done"):
            if "pddl_success" not in row.get("metrics", {}):
                raise ValueError(f"Missing terminal pddl_success: {key}")
            ep.update(done=True, success=row["metrics"]["pddl_success"] == 1)
    tasks = []
    for (scene, episode), ep in sorted(episodes.items()):
        if not ep["done"] or not ep["success"]:
            continue
        eligible = ep["length"] > 1
        tasks.append({
            "task_id": hashlib.sha256(json.dumps([scene, episode]).encode()).hexdigest()[:16],
            "scene_id": scene, "episode_id": episode, "clean_steps": ep["length"],
            "eligible": eligible, "exclusion_reason": None if eligible else "no_nonempty_prefix",
            "injection_step": max(1, min(ep["length"] - 1, math.floor(ep["length"] * fraction))) if eligible else None,
        })
    return {"observed": len(episodes), "completed": sum(e["done"] for e in episodes.values()),
            "successful": len(tasks), "eligible": sum(t["eligible"] for t in tasks), "tasks": tasks}


def prepare_experiment(trace, dataset, output, *, source_config=None, fraction=0.5,
                       seed=66, simulator_seed=100, max_steps=750, repeats=1,
                       agents=("agent_0", "agent_1"), wait_threshold=30):
    trace, dataset, output = (Path(p).resolve() for p in (trace, dataset, output))
    if repeats < 1 or max_steps < 2 or wait_threshold < 1:
        raise ValueError("repeats/threshold must be positive; max_steps must be >= 2")
    if not agents or len(set(agents)) != len(agents) or any(a not in ("agent_0", "agent_1") for a in agents):
        raise ValueError("agents must be a nonempty unique subset of agent_0, agent_1")
    selection = select_clean(trace, fraction)
    opener = gzip.open if dataset.suffix == ".gz" else open
    with opener(dataset, "rt", encoding="utf-8") as source:
        data = json.load(source)
    by_key = {(str(ep["scene_id"]), str(ep["episode_id"])): ep for ep in data["episodes"]}
    if len(by_key) != len(data["episodes"]):
        raise ValueError("Duplicate scene/episode pairs in dataset")
    for task in selection["tasks"]:
        if (task["scene_id"], task["episode_id"]) not in by_key:
            raise ValueError("Selected task is missing from dataset")
        if task["eligible"] and task["injection_step"] >= max_steps:
            raise ValueError("Injection step exceeds experimental step budget")
    if not selection["eligible"]:
        raise ValueError("No eligible successful clean episodes")
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    paths = {}
    selected_dataset = {k: v for k, v in data.items() if k not in ("episodes", "content_scenes_path")}
    selected_dataset["episodes"] = [by_key[(t["scene_id"], t["episode_id"])] for t in selection["tasks"]]
    with gzip.open(output / "clean_success_subset.json.gz", "wt", encoding="utf-8") as target:
        json.dump(selected_dataset, target)
    files[str(output / "clean_success_subset.json.gz")] = digest(output / "clean_success_subset.json.gz")
    for task in selection["tasks"]:
        directory = output / "inputs" / task["task_id"]
        directory.mkdir(parents=True)
        dataset_path, trace_path = directory / "episode.json.gz", directory / "clean.jsonl"
        task.update(dataset=str(dataset_path), trace=str(trace_path))
        with gzip.open(dataset_path, "wt", encoding="utf-8") as target:
            json.dump({**selected_dataset, "episodes": [by_key[(task["scene_id"], task["episode_id"])]]}, target)
        files[str(dataset_path)] = digest(dataset_path)
        paths[(task["scene_id"], task["episode_id"])] = trace_path
    handles = {}
    try:
        handles = {key: p.open("w", encoding="utf-8") for key, p in paths.items()}
        for row in read_rows(trace):
            key = (str(row.get("scene_id")), str(row.get("episode_id")))
            if key in handles:
                handles[key].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    for path in paths.values():
        files[str(path)] = digest(path)
    config_info = None
    if source_config:
        source_config = Path(source_config).resolve()
        # Source config can contain service credentials: retain only its digest.
        config_info = {"path": str(source_config), "sha256": digest(source_config)}
    manifest = {
        "schema_version": 1, "root": str(ROOT), "output": str(output),
        "source": {"trace": str(trace), "trace_sha256": digest(trace),
                   "dataset": str(dataset), "dataset_sha256": digest(dataset),
                   "config": config_info},
        "code_sha256": source_fingerprint(), "input_sha256": files,
        "selection": {k: v for k, v in selection.items() if k != "tasks"},
        "protocol": {"config": "multi_rearrange/llm_spot_drone_per.yaml", "seed": seed,
                     "simulator_seed": simulator_seed, "max_steps": max_steps,
                     "fraction": fraction, "repeats": repeats, "agents": list(agents),
                     "wait_threshold": wait_threshold, "terminate_on_wait": False,
                     "primary_comparison": "dropout_recovery versus dropout by fault agent",
                     "selection_policy": "fixed original clean-success cohort; never reselect on rerun",
                     "replay_source": "original selection trace", "context_fidelity_required": "recorded"},
        "tasks": selection["tasks"],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_manifest(manifest):
    if manifest["code_sha256"] != source_fingerprint():
        raise ValueError("Source/config files changed since preparation; prepare a new experiment directory")
    for path, expected in manifest["input_sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def jobs(manifest, limit_tasks=None):
    tasks = [t for t in manifest["tasks"] if t["eligible"]]
    if limit_tasks is not None:
        if limit_tasks < 1:
            raise ValueError("limit-tasks must be positive")
        tasks = tasks[:limit_tasks]
    for task in tasks:
        for repeat in range(manifest["protocol"]["repeats"]):
            yield {"task": task, "group": "clean", "agent": None, "repeat": repeat,
                   "job_id": f"{task['task_id']}_clean_r{repeat}"}
            for agent in manifest["protocol"]["agents"]:
                for group in GROUPS[1:]:
                    yield {"task": task, "group": group, "agent": agent, "repeat": repeat,
                           "job_id": f"{task['task_id']}_{group}_{agent}_r{repeat}"}


def build_command(manifest, task, group, agent, repeat, attempt, python):
    protocol = manifest["protocol"]
    attempt = Path(attempt)
    recovery = group == "dropout_recovery"
    command = [str(python), str(ROOT / "scripts" / ("run_clean.py" if group == "clean" else "inject_fault.py")),
               "--episode", task["episode_id"], "--dataset", task["dataset"],
               "--config", protocol["config"], "--output-dir", str(attempt / "output"),
               "--run-label", group, "--max-episode-steps", str(protocol["max_steps"])]
    if group != "clean":
        command += ["--fault", "dropout", "--agent", agent, "--step", str(task["injection_step"]),
                    "--replay-trajectory", task["trace"]]
    command += [
        "habitat_baselines.evaluate=True", "habitat_baselines.num_environments=1",
        "habitat_baselines.test_episode_count=1", "habitat_baselines.eval.evals_per_ep=1",
        "habitat_baselines.load_resume_state_config=False", "habitat_baselines.eval.should_load_ckpt=False",
        "habitat.environment.iterator_options.num_episode_sample=-1",
        "habitat.environment.iterator_options.shuffle=False",
        "habitat.environment.iterator_options.group_by_scene=False",
        "habitat.dataset.should_terminate_on_wait=False",
        f"habitat.seed={protocol['seed']}", f"habitat.simulator.seed={protocol['simulator_seed']}",
        f"habitat_baselines.error_detection.enabled={recovery}",
        f"habitat_baselines.error_recovery.enabled={recovery}",
        "habitat_baselines.eval.video_option=[]",
        "habitat_baselines.tensorboard_dir=" + json.dumps(str(attempt / "tb")),
        "habitat_baselines.video_dir=" + json.dumps(str(attempt / "video")),
        "habitat_baselines.image_dir=" + json.dumps(str(attempt / "images")),
        "hydra.run.dir=" + json.dumps(str(attempt / "hydra")),
    ]
    return command
