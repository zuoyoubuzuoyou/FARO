#!/usr/bin/env python3
"""Run fixed-model native Habitat evidence checks for all four latent faults."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name in ("habitat-mas", "habitat-lab", "habitat-baselines"):
    sys.path.insert(0, str(ROOT / name))

from habitat_mas.fault_injection.cli import build_parser, prepare_run


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def run_case(output, label, options, common, clean=False, abort=False, episodes=1):
    args = build_parser(clean=clean).parse_args([
        "--episode", "295", "--scene", "v3_sc3_staging_07.scene_instance", "--output-dir", str(output),
        "--run-label", label, *options,
    ])
    launcher_common = common
    if episodes > 1:
        # The single-run CLI deliberately requires one evaluation per episode.
        # Exercise repeated reset through native evaluator overrides instead.
        launcher_common = [value for value in common if not value.startswith((
            "habitat_baselines.eval.evals_per_ep=", "habitat_baselines.test_episode_count="))]
        launcher_common += ["habitat_baselines.eval.evals_per_ep=1", "habitat_baselines.test_episode_count=1"]
    _, command, plan = prepare_run(args, launcher_common)
    if episodes > 1:
        command = [value for value in command if not value.startswith((
            "habitat_baselines.eval.evals_per_ep=", "habitat_baselines.test_episode_count="))]
        command += [f"habitat_baselines.eval.evals_per_ep={episodes}", "habitat_baselines.test_episode_count=-1"]
    command[command.index("habitat_baselines.run")] = "habitat_mas.test.native_latent_fault_smoke"
    command.insert(4, "--smoke-fault=" + ("clean" if clean else args.fault))
    (output / label / "native_command.json").write_text(json.dumps(command, indent=2) + "\n")
    log = output / label / "native.log"
    print(f"Running {label}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(ROOT / name) for name in ("habitat-mas", "habitat-lab", "habitat-baselines"))
    environment["MPLCONFIGDIR"] = str(output / "matplotlib")
    environment["EMOS_NATIVE_MODEL_EVIDENCE"] = str(output / label / "model_observations.jsonl")
    if abort:
        environment["EMOS_NATIVE_ABORT"] = "1"
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if abort:
        assert result.returncode != 0 and "native smoke deliberate abort" in log.read_text()
        events = read_jsonl(plan["outputs"]["events"])
        summary = json.loads(Path(plan["outputs"]["summary"]).read_text())["episodes"][-1]
        assert summary["termination_reason"] == "exception" and summary["effective"]
        assert any(event["type"] == "message_dropped" for event in events)
        return plan, [], events
    if result.returncode:
        raise RuntimeError(f"{label} exited {result.returncode}; log: {log}")
    rows = read_jsonl(plan["outputs"]["trajectory"])
    steps = [row for row in rows if row["type"] == "simulator_step"]
    assert steps and steps[-1]["done"], label
    if clean:
        return plan, steps, []
    events = read_jsonl(plan["outputs"]["events"])
    summaries = [event for event in events if event["type"] == "fault_episode_summary"]
    assert len(summaries) == episodes and all(row["triggered"] for row in summaries), label
    return plan, steps, events


def verify_case(label, fault, target, steps, events, require_effect=True):
    applications = [event for event in events if event["type"] == "fault_applied"]
    if require_effect:
        assert applications, f"{label}: fault did not affect execution"
    for event in applications:
        assert event["agent"] == target
        assert event["logical_agent_responsive"] and event["logical_agent_active"]
    if fault == "action_noop" and applications:
        for event in applications:
            assert event["original_action"] != event["effective_action"]
            assert 2 <= event["simulator_step"] < 7
        window = [row for row in steps if 2 <= row["simulator_step"] < 7]
        previous = steps[1]["simulator_state"]["agents"][target]
        for row in window:
            state = row["simulator_state"]["agents"][target]
            assert max(abs(a-b) for a,b in zip(state["position"], previous["position"])) < 1e-3
            assert abs(state["rotation"] - previous["rotation"]) < 1e-3
        assert any(event["type"] == "fault_expired" and event["simulator_step"] == 7 for event in events)
        assert any(row["simulator_step"] > 7 and row["simulator_state"]["agents"][target]["position"] != previous["position"]
                   for row in steps), f"{label}: navigation did not resume"
    elif fault == "wrong_subgoal" and applications:
        event = applications[0]
        assert event["original_subtask"] != event["effective_subtask"]
        assert any(row["policy_info"].get(target + "_high_level_decision", {}).get("action") == "nav_to_obj"
                   and row["policy_info"][target + "_high_level_decision"]["arguments"]["target_obj"] == "any_targets|1"
                   for row in steps if row["simulator_step"] >= event["simulator_step"])
    elif fault == "message_delay" and applications:
        queued = {event["message_id"]: event for event in events if event["type"] == "message_queued"}
        delivered = [event for event in events if event["type"] == "message_delivered"]
        assert delivered, f"{label}: no message delivered"
        for event in delivered:
            sent = queued[event["message_id"]]
            assert event["actual_delivery_step"] == sent["delivery_step"] > sent["send_step"]
            assert event["request"] == sent["request"]
    elif fault == "object_localization" and applications:
        assert any(event["simulator_step"] == 2 for event in applications), f"{label}: cached target not invalidated"
        for event in applications:
            assert event["estimated_position"] != event["true_position"]
            assert all(abs(estimate-true-bias) < 1e-6 for estimate, true, bias in zip(
                event["estimated_position"], event["true_position"], event["position_offset"]))
        assert any(event["type"] == "fault_expired" and event["simulator_step"] == 7 for event in events)
        restored = [event for event in events if event["type"] == "localization_target_restored"]
        assert restored and restored[0]["simulator_step"] == 7, f"{label}: native target not restored"
        assert restored[0]["estimated_position"] == restored[0]["true_position"]
    return {"case": label, "fault": fault, "agent": target, "steps": len(steps),
            "applications": len(applications), "effective": bool(applications)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case", help="Run only a named case (clean, noop_0/1, planning, delay, localization_0/1)")
    parser.add_argument("--replay", action="store_true", help="Also verify all four faults against a new clean prefix")
    settings = parser.parse_args()
    output = settings.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = [
        "habitat_baselines.evaluate=True", "habitat_baselines.num_environments=1",
        "habitat_baselines.test_episode_count=1", "habitat_baselines.eval.evals_per_ep=1",
        "habitat_baselines.load_resume_state_config=False", "habitat_baselines.eval.should_load_ckpt=False",
        "habitat.environment.iterator_options.num_episode_sample=-1",
        "habitat.environment.iterator_options.shuffle=False", "habitat.environment.iterator_options.group_by_scene=False",
        "habitat.dataset.should_terminate_on_wait=False", "habitat.seed=66", "habitat.simulator.seed=100",
        "habitat_baselines.error_detection.enabled=False", "habitat_baselines.error_recovery.enabled=False",
        "habitat_baselines.eval.video_option=[]", "habitat.environment.max_episode_steps=60",
        "habitat_baselines.tensorboard_dir=" + json.dumps(str(output / "tb")),
        "hydra.run.dir=" + json.dumps(str(output / "hydra")),
    ]
    cases = [
        ("noop_0", "action_noop", "agent_0", ["--duration-steps", "5"], 2),
        ("noop_1", "action_noop", "agent_1", ["--duration-steps", "5"], 2),
        ("planning", "wrong_subgoal", "agent_0", ["--object", "any_targets|0", "--replacement-object", "any_targets|1"], 2),
        ("delay", "message_delay", "agent_1", ["--duration-steps", "3", "--delay-steps", "4"], 0),
        ("localization_0", "object_localization", "agent_0", ["--object", "any_targets|0", "--position-offset", "1", "0", "0", "--duration-steps", "5"], 2),
        ("localization_1", "object_localization", "agent_1", ["--object", "any_targets|1", "--position-offset", "-1", "0", "0", "--duration-steps", "5"], 2),
    ]
    if settings.case and settings.case not in {"clean", *(case[0] for case in cases)}:
        parser.error("Unknown --case")
    results = []
    clean_plan, clean_steps = None, None
    if not settings.case or settings.case == "clean" or settings.replay:
        clean_plan, clean_steps, _ = run_case(output, "clean", [], common, clean=True)
        results.append({"case": "clean", "steps": len(clean_steps)})
    for label, fault, target, options, step in cases:
        if settings.case and settings.case != label:
            continue
        plan, steps, events = run_case(output, label, ["--fault", fault, "--agent", target, "--step", str(step), *options], common)
        results.append(verify_case(label, fault, target, steps, events))
        if clean_steps is not None and fault == "object_localization":
            assert steps[0]["simulator_state"]["objects"] == clean_steps[0]["simulator_state"]["objects"]
            assert steps[-1]["simulator_state"]["objects"] == clean_steps[-1]["simulator_state"]["objects"]
            peer = "agent_1" if target == "agent_0" else "agent_0"
            for row in steps[:8]:
                expected = clean_steps[row["simulator_step"]]["simulator_state"]["agents"][peer]
                assert row["simulator_state"]["agents"][peer] == expected, f"{label}: peer affected"
        if fault == "object_localization":
            observations = [row for row in read_jsonl(output / label / "model_observations.jsonl")
                            if row["agent"] == target and not row["planning"] and not row["discussion"]]
            biased = next(event["estimated_position"] for event in events if event["type"] == "fault_applied")
            restored = next(event["true_position"] for event in events if event["type"] == "localization_target_restored")
            fmt = lambda pos: "[" + ", ".join(f"{value:.3f}" for value in pos) + "]"
            assert any(fmt(biased) in row["prompt"] for row in observations if row["step"] == 2)
            assert any(fmt(restored) in row["prompt"] for row in observations if row["step"] == 7)
        if fault == "message_delay":
            observations = [row for row in read_jsonl(output / label / "model_observations.jsonl")
                            if row["agent"] == target and not row["planning"] and not row["discussion"]]
            request = next(event["request"] for event in events if event["type"] == "message_queued")
            assert observations and not any(request in row["prompt"] for row in observations if row["step"] < 4)
            assert any(request in row["prompt"] for row in observations if row["step"] >= 4), "recipient model did not observe delivery"
        print(json.dumps(results[-1]), flush=True)
    if settings.replay:
        for label, fault, target, options, step in (cases[0], cases[2], cases[3], cases[4]):
            replay_label = "replay_" + label
            # Replay two completed clean steps; message delay may have no new send in this live suffix.
            options = ["--fault", fault, "--agent", target, "--step", "2", *options,
                       "--replay-trajectory", clean_plan["outputs"]["trajectory"]]
            plan, steps, events = run_case(output, replay_label, options, common)
            rows = read_jsonl(plan["outputs"]["trajectory"])
            assert any(row["type"] == "replay_prefix_verified" for row in rows), replay_label
            assert not any(event["simulator_step"] < 2 for event in events)
            result = verify_case(replay_label, fault, target, steps, events, require_effect=fault != "message_delay")
            result["prefix_replay_verified"] = True
            results.append(result)
            print(json.dumps(result), flush=True)
    if not settings.case:
        repeat_common = [value for value in common if not value.startswith((
            "habitat_baselines.eval.evals_per_ep=", "habitat_baselines.test_episode_count=", "habitat.environment.max_episode_steps="))]
        repeat_common += ["habitat_baselines.test_episode_count=-1", "habitat_baselines.eval.evals_per_ep=2", "habitat.environment.max_episode_steps=2"]
        plan, steps, events = run_case(output, "repeat_delay", ["--fault", "message_delay", "--agent", "agent_1",
            "--step", "0", "--delay-steps", "20"], repeat_common, episodes=2)
        summaries = [event for event in events if event["type"] == "fault_episode_summary"]
        queued = [event for event in events if event["type"] == "message_queued"]
        dropped = [event for event in events if event["type"] == "message_dropped"]
        assert len(summaries) == len(queued) == len(dropped) == 2
        assert all(event["applications"] == 1 for event in summaries)
        assert all(event["message_id"] == 0 and event["send_step"] == 0 for event in queued)
        results.append({"case": "repeat_delay", "episodes": 2, "reset_verified": True})
        repeat_pf = [value for value in repeat_common if not value.startswith("habitat.environment.max_episode_steps=")]
        repeat_pf += ["habitat.environment.max_episode_steps=8"]
        _, steps, events = run_case(output, "repeat_localization", ["--fault", "object_localization", "--agent", "agent_0",
            "--step", "2", "--duration-steps", "5", "--object", "any_targets|0", "--position-offset", "1", "0", "0"],
            repeat_pf, episodes=2)
        assert len([event for event in events if event["type"] == "fault_applied"]) == 2
        assert len([event for event in events if event["type"] == "localization_target_restored"]) == 2
        observations = [row for row in read_jsonl(output / "repeat_localization" / "model_observations.jsonl")
                        if row["agent"] == "agent_0" and row["step"] == 0 and not row["planning"] and not row["discussion"]]
        biased = next(event["estimated_position"] for event in events if event["type"] == "fault_applied")
        biased_xyz = "[" + ", ".join(f"{value:.3f}" for value in biased) + "]"
        assert len(observations) == 2 and all(biased_xyz not in row["prompt"] for row in observations)
        results.append({"case": "repeat_localization", "episodes": 2, "reset_verified": True})
        run_case(output, "abort_delay", ["--fault", "message_delay", "--agent", "agent_1", "--step", "0",
            "--delay-steps", "20"], common, abort=True)
        results.append({"case": "abort_delay", "abort_evidence_verified": True})
    (output / "verification.json").write_text(json.dumps({"verified": True, "cases": results}, indent=2) + "\n")
    print(f"Verified native cases: {len(results)}; evidence: {output / 'verification.json'}", flush=True)


if __name__ == "__main__":
    main()
