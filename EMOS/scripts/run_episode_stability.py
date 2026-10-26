#!/usr/bin/env python3
"""Run a fixed cohort from fresh processes and audit native S/F outcomes.

Use the Habitat Python interpreter. No production source is modified. Prepare
and summarize modes use only the standard library; real workers need GPU/API.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import wraps
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from episode_stability_report import write_reports

DEFAULT_SETTINGS = {
    "task": "Manipulation", "repetitions": 3, "count": 10,
    "sample_seed": 100, "environment_seed": 100, "simulator_seed": 100,
    "model": "gpt-6-sol", "temperature": 0.0, "model_seed": 100,
    "gpu": "1", "max_episode_steps": 4000, "wall_timeout_seconds": 1800,
    "config": "multi_rearrange/llm_height_man.yaml",
}
DATASET = ROOT / "data/datasets/hssd/0/hssd_height_man.json.gz"
ROBOTS = ROOT / "fault_output/manipulation_hssd/height_man_fetch_stretch_robots.json"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__).resolve(), ROOT / "scripts/episode_stability_report.py"]
    for package in ("habitat-lab", "habitat-baselines", "habitat-mas"):
        paths.extend(p for p in (ROOT / package).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".yaml", ".yml", ".json")
                     and "__pycache__" not in p.parts and ".git" not in p.parts)
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(set(paths))}


ASSET_ROOTS = ("data/robots", "data/scene_datasets/hssd-hab", "data/objects/ycb",
               "data/replica_cad", "habitat-mas/habitat_mas/data")


def collect_input_paths(dataset_path, robot_path, cohort, root=ROOT):
    """Hash configuration inputs, including scene descriptions and prompt data."""
    paths = {Path(dataset_path), Path(robot_path)}
    for episode in cohort:
        paths.update((Path(episode["dataset_file"]), root / episode["scene_id"]))
    for relative in ASSET_ROOTS:
        paths.update(p for p in (root / relative).rglob("*")
                     if p.is_file() and p.suffix in (".json", ".urdf", ".yaml", ".yml"))
    return sorted(paths)


def asset_metadata(root=ROOT):
    """Monitor binary asset metadata without repeatedly hashing huge meshes."""
    result = {}
    for relative in ASSET_ROOTS:
        for path in (root / relative).rglob("*"):
            if path.is_file():
                stat = path.stat()
                result[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return result


def terminate_process_group(process, grace_seconds=10):
    """Kill leftover descendants even when the group leader exits first."""
    def send(sig):
        try:
            os.killpg(process.pid, sig)
            return True
        except ProcessLookupError:
            return False

    if not send(signal.SIGTERM):
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(min(.05, max(0, deadline - time.monotonic())))
    send(signal.SIGKILL)
    process.wait()


def select_episodes(episodes, count=10, sample_seed=100):
    ordered = sorted(episodes, key=lambda e: (e["scene_id"], str(e["episode_id"])))
    identities = [(e["scene_id"], str(e["episode_id"])) for e in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate dataset identities")
    if type(count) is not int or not 1 <= count <= len(ordered):
        raise ValueError("Sample count outside dataset size")
    selected = random.Random(sample_seed).sample(ordered, count)
    return sorted(selected, key=lambda e: (int(e["episode_id"]), e["scene_id"]))


def install_decoding_controls(model_class, settings):
    """Wrap the existing request builder only in this experiment process."""
    original = model_class._create_completion

    @wraps(original)
    def controlled(self, request, phase, **kwargs):
        if self.model != settings["model"]:
            raise ValueError("Model differs from the experiment manifest")
        for name, expected in (("temperature", settings["temperature"]),
                               ("seed", settings["model_seed"])):
            if name in kwargs and kwargs[name] != expected:
                raise ValueError("Conflicting experiment parameter: " + name)
            kwargs[name] = expected
        return original(self, request, phase, **kwargs)

    model_class._create_completion = controlled


def inspect_trial(trial, settings, returncode, timed_out=False):
    """Validate actual trace evidence; partial native metrics never imply F."""
    # Prepare/help still work without an installed MAS package. The storage
    # reader and the package modules imported here use only the standard library.
    sys.path.insert(0, str(ROOT / "habitat-mas"))
    from habitat_mas.utils.trajectory_storage import iter_trace_events, read_trace_summary

    result = {"status": "recording_error", "task_success": None,
              "returncode": returncode, "finished_at": utc_now()}
    summaries = list(Path(trial["trajectory_dir"]).glob("*/episodes/*/summary.json"))
    events = []
    summary = None
    try:
        if len(summaries) != 1:
            raise ValueError("Expected exactly one recorded episode summary")
        summary_path = summaries[0]
        summary = read_trace_summary(summary_path)
        files = summary.get("files") or {}
        filename = files.get("steps", files.get("events"))
        if filename is None:
            filename = "steps.jsonl" if summary_path.with_name("steps.jsonl").exists() else "events.jsonl"
        event_path = (summary_path.parent / filename).resolve()
        if event_path.parent != summary_path.parent.resolve():
            raise ValueError("Episode trace file must remain inside its episode directory")
        events = list(iter_trace_events(event_path))
        result.update(summary_path=str(summary_path), events_path=str(event_path),
                      run_id=summary.get("run_id"), evaluation_id=summary.get("evaluation_id"),
                      num_env_steps=summary.get("num_env_steps"),
                      elapsed_seconds=summary.get("elapsed_seconds"),
                      native_observed_success=summary.get("task_success"),
                      error=summary.get("error"))
        responses = [e for e in events if e.get("event_type") == "llm_response"]
        result["response_models"] = sorted({e["response_model"] for e in responses
                                            if isinstance(e.get("response_model"), str)})
        result["response_fingerprints"] = sorted({
            e["raw_response"]["system_fingerprint"] for e in responses
            if isinstance(e.get("raw_response"), dict)
            and isinstance(e["raw_response"].get("system_fingerprint"), str)})
        result["model_calls"] = len([e for e in events if e.get("event_type") == "llm_request"])
        result["total_tokens"] = sum((e.get("usage") or {}).get("total_tokens", 0) or 0
                                     for e in responses)
        result["model_phases"] = sorted({e["phase"] for e in events
                                          if e.get("event_type") == "model_config"
                                          and isinstance(e.get("phase"), str)})
        starts = [e for e in events if e.get("event_type") == "episode_start"]
        if len(starts) == 1 and isinstance(starts[0].get("state"), dict) and starts[0]["state"]:
            state_bytes = json.dumps(starts[0]["state"], sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()
            result["initial_state_sha256"] = hashlib.sha256(state_bytes).hexdigest()
        if timed_out:
            result["status"] = "timeout"
            return result
        if any(e.get("event_type") == "llm_error" for e in events):
            result["status"] = "api_error"
            result["model_errors"] = [{k: e.get(k) for k in ("error_type", "error_message", "phase")}
                                      for e in events if e.get("event_type") == "llm_error"]
            return result
        if returncode != 0 or summary.get("status") != "completed":
            result["status"] = "execution_error"
            return result
        identity_fields = ("scene_id", "episode_id", "run_id", "evaluation_id")
        if any(not isinstance(summary.get(k), str) or not summary[k] for k in identity_fields):
            raise ValueError("Recorded identity is incomplete")
        for record in (summary, *events):
            if any(str(record.get(k)) != str(trial[k]) for k in ("scene_id", "episode_id")):
                raise ValueError("Recorded episode identity differs from planned identity")
            if any(record.get(k) != summary[k] for k in ("run_id", "evaluation_id")):
                raise ValueError("Mixed run/evaluation identities in trace")
        if not result.get("initial_state_sha256"):
            raise ValueError("Initial state is missing")
        configs = [e for e in events if e.get("event_type") == "model_config"]
        requests = [e for e in events if e.get("event_type") == "llm_request"]
        if not configs or len(configs) != len(requests) or len(requests) != len(responses):
            raise ValueError("Model config/request/response records are incomplete")
        for event in configs + requests:
            kwargs = event.get("request_kwargs") or {}
            if (kwargs.get("model") != settings["model"]
                    or kwargs.get("temperature") != settings["temperature"]
                    or kwargs.get("seed") != settings["model_seed"]):
                raise ValueError("Actual model request differs from fixed parameters")
        indexed_calls = []
        for records in (configs, requests, responses):
            ids = [e.get("call_id") for e in records]
            if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
                raise ValueError("Missing or duplicated model call IDs")
            indexed_calls.append(dict(zip(ids, records)))
        if not (indexed_calls[0].keys() == indexed_calls[1].keys() == indexed_calls[2].keys()):
            raise ValueError("Model config/request/response call IDs differ")
        for call_id in indexed_calls[0]:
            records = [index[call_id] for index in indexed_calls]
            owners = [(e.get("phase"), e.get("agent_id") or e.get("agent_name")) for e in records]
            if len(set(owners)) != 1 or any(not isinstance(v, str) or not v for v in owners[0]):
                raise ValueError("Model call phase or agent is inconsistent")
        success = summary.get("task_success")
        if type(success) is not bool:
            raise ValueError("Native terminal success metric is missing")
        steps = [e for e in events if e.get("event_type") == "step"]
        expected_steps = summary.get("num_env_steps")
        if type(expected_steps) is not int or expected_steps < 1 or [e.get("step") for e in steps] != list(range(1, expected_steps + 1)):
            raise ValueError("Recorded steps are missing, duplicated or non-contiguous")
        terminals = [e for e in events if e.get("event_type") == "step" and e.get("done") is True]
        ends = [e for e in events if e.get("event_type") == "episode_end"]
        if len(terminals) != 1 or terminals[0] is not steps[-1] or len(ends) != 1 or events[-1] is not ends[0]:
            raise ValueError("Native termination evidence is missing or inconsistent")
        terminal_native = ((terminals[0].get("state") or {}).get("task_progress") or {}).get("pddl_success")
        if terminal_native is None or terminal_native not in (True, False, 0, 1) or bool(terminal_native) != success:
            raise ValueError("Terminal native state and original evaluator success disagree")
        end = ends[0]
        if end.get("status") != "completed" or end.get("task_success") is not success or end.get("num_env_steps") != expected_steps:
            raise ValueError("Final episode_end differs from episode summary")
        result.update(status="completed", task_success=success)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["validation_error"] = str(error)
        if timed_out:
            result["status"] = "timeout"
        elif returncode != 0 and summary is None:
            result["status"] = "process_error"
    return result


def prepare(output, python):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing experiment: " + str(output))
    with gzip.open(DATASET, "rt") as stream:
        dataset = json.load(stream)
    robots = json.loads(ROBOTS.read_text())
    selected = select_episodes(dataset["episodes"])
    for episode in selected:
        if not (ROOT / episode["scene_id"]).is_file():
            raise FileNotFoundError(episode["scene_id"])
        if len(robots.get(str(episode["episode_id"]), {}).get("agents", [])) != 2:
            raise ValueError("Missing robot start configuration")
    output.mkdir(parents=True)
    settings = {**DEFAULT_SETTINGS, "python": str(Path(python).resolve()),
                "robot_config": str(ROBOTS), "source_dataset": str(DATASET)}
    cohort = []
    for index, episode in enumerate(selected):
        target = output / "datasets" / ("task_%03d.json.gz" % index)
        target.parent.mkdir(exist_ok=True)
        with gzip.open(target, "wt") as stream:
            json.dump({**dataset, "episodes": [episode]}, stream)
        cohort.append({"task": settings["task"], "scene_id": episode["scene_id"],
                       "episode_id": str(episode["episode_id"]), "dataset_file": str(target)})
    trials = []
    for round_number in range(1, settings["repetitions"] + 1):
        for index, episode in enumerate(cohort):
            directory = output / ("round_%d" % round_number) / ("task_%03d" % index)
            trials.append({**episode, "round": round_number,
                           "trial_id": "r%d_task%03d" % (round_number, index),
                           "trial_dir": str(directory),
                           "trajectory_dir": str(directory / "trajectories"),
                           "status": "pending", "task_success": None})
    inputs = collect_input_paths(DATASET, ROBOTS, cohort)
    manifest = {"schema_version": "emos-episode-stability-v1", "created_at": utc_now(),
                "status": "prepared", "settings": settings, "cohort": cohort,
                "trials": trials, "source_hashes": source_hashes(),
                "input_hashes": {str(p): file_hash(p) for p in inputs},
                "asset_metadata": asset_metadata(),
                "asset_validation": "SHA256 for scene/object/robot/prompt configurations; size and mtime_ns for binary assets (not a content hash of meshes)."}
    save_json(output / "manifest.json", manifest)
    write_reports(manifest, output)
    return manifest


def check_hashes(manifest):
    if source_hashes() != manifest["source_hashes"]:
        raise ValueError("Execution source/config files changed")
    for name, expected in manifest["input_hashes"].items():
        if file_hash(name) != expected:
            raise ValueError("Experiment input changed: " + name)
    if asset_metadata() != manifest.get("asset_metadata", {}):
        raise ValueError("Asset file inventory, size or modification time changed")


def worker(spec_path, config_only=False):
    spec = json.loads(Path(spec_path).resolve().read_text())
    settings, trial = spec["settings"], spec["trial"]
    os.environ["CUDA_VISIBLE_DEVICES"] = settings["gpu"]
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    workdir = Path(trial["trial_dir"]) / "workspace"
    workdir.mkdir(parents=True, exist_ok=True)
    for name in ("data", "res", "configs", "habitat-lab", "habitat-baselines", "habitat-mas"):
        link = workdir / name
        if not link.exists():
            link.symlink_to(ROOT / name, target_is_directory=True)
    os.chdir(workdir)
    for package in ("habitat-lab", "habitat-baselines", "habitat-mas"):
        sys.path.insert(0, str(ROOT / package))
    dependencies = Path("/tmp/emos_detection_deps")
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    os.environ.setdefault("MPLCONFIGDIR", str(workdir / "matplotlib"))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from habitat_mas.utils.models import OpenAIModel
    install_decoding_controls(OpenAIModel, settings)
    directory = Path(trial["trial_dir"])
    overrides = [
        "habitat_baselines.llm_model=" + settings["model"],
        "habitat_baselines.llm_temperature=" + str(settings["temperature"]),
        "habitat_baselines.llm_seed=" + str(settings["model_seed"]),
        "habitat.seed=" + str(settings["environment_seed"]),
        "habitat.simulator.seed=" + str(settings["simulator_seed"]),
        "habitat.dataset.data_path=" + trial["dataset_file"],
        "habitat.dataset.robot_config=" + settings["robot_config"],
        "habitat.simulator.json_path=" + settings["robot_config"],
        "habitat.dataset.randomize_agent_start=0",
        "habitat.dataset.should_terminate_on_wait=False",
        "habitat.simulator.w2j=False",
        "habitat.simulator.additional_object_paths=[data/objects/ycb/configs/,data/replica_cad/configs/objects/]",
        "habitat.simulator.agents.agent_0.joint_start_noise=0.0",
        "habitat.simulator.agents.agent_1.joint_start_noise=0.0",
        "habitat.task.force_regenerate=True", "habitat.task.should_save_to_cache=False",
        "habitat.environment.max_episode_steps=" + str(settings["max_episode_steps"]),
        "habitat.environment.iterator_options.shuffle=False",
        "habitat_baselines.evaluate=True", "habitat_baselines.num_environments=1",
        "habitat_baselines.test_episode_count=1", "habitat_baselines.eval.evals_per_ep=1",
        "habitat_baselines.eval.should_load_ckpt=False",
        "habitat_baselines.load_resume_state_config=False",
        "habitat_baselines.eval.video_option=[]", "habitat_baselines.eval.image_option=[]",
        "habitat_baselines.eval.recovery.enabled=False",
        "habitat_baselines.eval.save_trajectories=True",
        "habitat_baselines.eval.trajectory_dir=" + trial["trajectory_dir"],
        "habitat_baselines.tensorboard_dir=" + str(directory / "tensorboard"),
        "habitat_baselines.log_file=" + str(directory / "habitat.log"),
        "hydra.run.dir=" + str(directory / "hydra"), "hydra.job.chdir=False",
    ]
    save_json(directory / "invocation.json", {"settings": settings, "overrides": overrides,
                                               "working_directory": str(workdir)})
    sys.argv = ["habitat_baselines.run", "--config-name=" + settings["config"]] + overrides
    if config_only:
        sys.argv += ["--cfg", "job"]
    runpy.run_module("habitat_baselines.run", run_name="__main__")


def execute(manifest, output):
    output = Path(output).resolve()
    if manifest["status"] != "prepared" or any(t["status"] != "pending" for t in manifest["trials"]):
        raise ValueError("Only a fresh prepared manifest can run; existing attempts are never replaced")
    for trial in manifest["trials"]:
        if Path(trial["trial_dir"]).exists():
            raise FileExistsError("Trial directory already exists; preserve prior evidence: " + trial["trial_dir"])
    manifest.update(status="running", started_at=utc_now())
    try:
        for trial in manifest["trials"]:
            try:
                check_hashes(manifest)
            except (OSError, ValueError) as error:
                trial.update(status="configuration_changed", error=str(error))
                break
            directory = Path(trial["trial_dir"])
            directory.mkdir(parents=True, exist_ok=False)
            spec_path = directory / "worker_spec.json"
            save_json(spec_path, {"settings": manifest["settings"], "trial": trial})
            command = [manifest["settings"]["python"], "-u", str(Path(__file__).resolve()),
                       "--worker-spec", str(spec_path)]
            trial.update(status="running", started_at=utc_now(), command=command,
                         log_path=str(directory / "worker.log"))
            save_json(output / "manifest.json", manifest)
            print(json.dumps({"event": "trial_start", "round": trial["round"],
                              "episode_id": trial["episode_id"]}), flush=True)
            started = time.monotonic()
            timed_out = False
            with (directory / "worker.log").open("x") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                try:
                    while True:
                        remaining = manifest["settings"]["wall_timeout_seconds"] - (time.monotonic() - started)
                        if remaining <= 0:
                            timed_out = True
                            terminate_process_group(process)
                            break
                        try:
                            process.wait(timeout=min(30, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            print(json.dumps({"event": "heartbeat", "round": trial["round"],
                                              "episode_id": trial["episode_id"],
                                              "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
                except BaseException:
                    terminate_process_group(process)
                    raise
            trial.update(inspect_trial(trial, manifest["settings"], process.returncode, timed_out))
            trial["wall_elapsed_seconds"] = time.monotonic() - started
            try:
                check_hashes(manifest)
            except (OSError, ValueError) as error:
                trial.update(status="configuration_changed", task_success=None, error=str(error))
            save_json(output / "manifest.json", manifest)
            save_json(directory / "result.json", trial)
            write_reports(manifest, output)
            print(json.dumps({"event": "trial_end", **{k: trial.get(k) for k in (
                "round", "episode_id", "status", "task_success", "num_env_steps", "total_tokens")}}), flush=True)
            if trial["status"] != "completed":
                break
    finally:
        manifest["status"] = "completed" if all(t["status"] == "completed" for t in manifest["trials"]) else "stopped"
        manifest["finished_at"] = utc_now()
        save_json(output / "manifest.json", manifest)
        write_reports(manifest, output)
    return 0 if manifest["status"] == "completed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default="/home/dyc/.conda/envs/habitat/bin/python")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--run-prepared", action="store_true")
    modes.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_spec:
        worker(args.worker_spec, args.config_only)
        return 0
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if args.run_prepared or args.summarize_only:
        manifest = json.loads((output / "manifest.json").read_text())
    else:
        manifest = prepare(output, args.python)
    if args.prepare_only or args.summarize_only:
        report = write_reports(manifest, output)
        print(json.dumps({"status": manifest["status"], "planned_trials": len(manifest["trials"]),
                          "output": str(output)}), flush=True)
        return 0
    return execute(manifest, output)


if __name__ == "__main__":
    raise SystemExit(main())
