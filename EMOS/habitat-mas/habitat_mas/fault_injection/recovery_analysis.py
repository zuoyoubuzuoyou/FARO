"""Outcome validation, effective waiting metrics and paired statistics."""
from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

from habitat_mas.fault_injection.recovery_batch import digest, jobs, read_rows, write_json


def wait_metrics(rows, threshold=30, start_step=0):
    result = {}
    for label in ("double_wait", "offline_wait"):
        streak = longest = total = 0
        onset = sustained_onset = None
        previous = None
        for row in rows:
            if row.get("type") != "simulator_step" or row["simulator_step"] < start_step:
                continue
            index = row["simulator_step"]
            agents = row.get("simulator_state", {}).get("agents", {})
            actions = row.get("agent_step_actions", {})
            online = [name for name, state in agents.items() if not state.get("dropout_state", {}).get("active", False)]
            offline = len(agents) - len(online)
            waiting = [name for name in online if actions.get(name, {}).get("action", {}).get("action") == "wait"
                       and not agents[name].get("hang_state", {}).get("active", False)]
            unfinished = row.get("metrics", {}).get("pddl_success") == 0
            active = unfinished and len(agents) == 2 and len(waiting) == len(online) and bool(online)
            active = active and (offline == 0 if label == "double_wait" else offline == 1)
            if previous is not None and index != previous + 1:
                streak = 0
            previous = index
            streak = streak + 1 if active else 0
            if active:
                total += 1
                if onset is None:
                    onset = index
                if streak == threshold and sustained_onset is None:
                    sustained_onset = index - threshold + 1
            longest = max(longest, streak)
        result.update({f"{label}_longest": longest, f"{label}_steps": total,
                       f"{label}_onset": onset, f"{label}_sustained_onset": sustained_onset,
                       f"{label}_detected": longest >= threshold})
    return result


def summarize_rows(rows, injection, group, agent, branch_step, returncode, threshold=30):
    steps = [r for r in rows if r.get("type") == "simulator_step"]
    terminal = [r for r in rows if r.get("type") == "system_crash" or
                (r.get("type") == "simulator_step" and r.get("done"))]
    replay = [r for r in rows if r.get("type") == "replay_prefix_verified"]
    verified = any(r.get("branch_step") == branch_step and r.get("verified_steps") == branch_step
                   and r.get("context_fidelity") == "recorded" for r in replay)
    injected = bool(injection and injection.get("type") == "fault_injected"
                    and injection.get("agent") == agent and injection.get("simulator_step") == branch_step)
    reasons = []
    if returncode not in (0, 3):
        reasons.append("process_error")
    if not terminal:
        reasons.append("missing_terminal")
    terminal_seen = False
    for row in rows:
        if terminal_seen and row.get("type") == "simulator_step":
            reasons.append("steps_after_terminal")
            break
        if row.get("type") == "system_crash" or (row.get("type") == "simulator_step" and row.get("done")):
            terminal_seen = True
    if len([r for r in terminal if r.get("type") == "simulator_step"]) > 1:
        reasons.append("multiple_terminals")
    if any(r.get("simulator_step") != i for i, r in enumerate(steps)):
        reasons.append("unordered_steps")
    if group != "clean":
        if not verified:
            reasons.append("prefix_not_verified_with_recorded_context")
        if not injected:
            reasons.append("injection_not_applied_at_expected_step")
    if any(r.get("type") == "replay_failed" for r in rows):
        reasons.append("replay_failed")
    last = terminal[-1] if terminal else {}
    success = last.get("metrics", {}).get("pddl_success")
    if terminal and success not in (0, 1):
        reasons.append("missing_success_metric")
    if returncode == 3 and last.get("type") != "system_crash":
        reasons.append("missing_crash_report")
    if last.get("type") == "system_crash" and (returncode != 3 or success != 0):
        reasons.append("inconsistent_crash_result")
    valid = not reasons
    return {"valid": valid, "invalid_reason": ";".join(reasons),
            "success": int(success) if valid else None, "replay_verified": verified,
            "injection_applied": injected, "final_steps": len(steps),
            "termination_reason": last.get("reason", "success" if success == 1 else "environment_done") if terminal else "incomplete",
            "recovery_attempts": sum(r.get("type") == "failure_detected" for r in rows),
            "recovery_completed": sum(r.get("type") == "recovery_completed" for r in rows),
            "transferred_tasks": sum(len(r.get("transfers", [])) for r in rows if r.get("type") == "tasks_reassigned"),
            **wait_metrics(steps, threshold, branch_step or 0)}


def summarize_attempt(job, attempt, returncode, threshold=30):
    from habitat_mas.fault_injection.naming import output_run_stem
    group, task = job["group"], job["task"]
    stem = output_run_stem(group, "clean" if group == "clean" else "dropout")
    path = Path(attempt) / "output" / (stem + "_trajectory.jsonl")
    event_path = path.with_name(stem + "_events.jsonl")
    try:
        rows = list(read_rows(path)) if path.exists() else []
        events = list(read_rows(event_path)) if event_path.exists() else []
        # Fault events omit scene_id; one process contains exactly one frozen task.
        injections = [e for e in events if e.get("type") == "fault_injected"
                      and str(e.get("episode_id")) == task["episode_id"]]
        result = summarize_rows(rows, injections[0] if len(injections) == 1 else None,
                                group, job["agent"], task["injection_step"], returncode, threshold)
        expected = (task["scene_id"], task["episode_id"])
        wrong = any((str(r.get("scene_id")), str(r.get("episode_id"))) != expected
                    for r in rows if r.get("type") in ("simulator_step", "system_crash", "episode_start"))
        problems = [result["invalid_reason"]] if result["invalid_reason"] else []
        if wrong:
            problems.append("wrong_scene_or_episode")
        expected_mode = "clean" if group == "clean" else "faulty"
        if any(r.get("mode") != expected_mode for r in rows if r.get("type") in ("simulator_step", "episode_start")):
            problems.append("wrong_run_mode")
        if group != "clean":
            proofs = [r for r in rows if r.get("type") == "replay_prefix_verified"]
            if any(str(Path(r.get("source", "")).resolve()) != str(Path(task["trace"]).resolve())
                   or (str(r.get("scene_id")), str(r.get("episode_id"))) != expected for r in proofs):
                problems.append("wrong_replay_source_or_episode")
        if problems:
            result.update(valid=False, success=None, invalid_reason=";".join(problems))
        return result
    except (ValueError, KeyError, TypeError) as error:
        return {"valid": False, "success": None, "invalid_reason": f"malformed_output: {error}"}


def paired_statistics(pairs, resamples=10000):
    n = len(pairs)
    if not n:
        return {"n": 0}
    rescued = sum(d == 0 and r == 1 for d, r in pairs)
    harmed = sum(d == 1 and r == 0 for d, r in pairs)
    failures = sum(d == 0 for d, _ in pairs)
    discordant = rescued + harmed
    p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(rescued, harmed) + 1)) / (2 ** discordant)) if discordant else 1.0
    deltas = [r - d for d, r in pairs]
    rng = random.Random(20260925)
    samples = sorted(sum(rng.choices(deltas, k=n)) / n for _ in range(resamples))
    return {"n": n, "dropout_success_rate": sum(d for d, _ in pairs) / n,
            "recovery_success_rate": sum(r for _, r in pairs) / n,
            "success_rate_difference": sum(deltas) / n,
            "difference_ci95_percentile": [samples[int(.025 * resamples)], samples[min(resamples - 1, int(.975 * resamples))]],
            "rescued": rescued, "harmed": harmed, "dropout_failures": failures,
            "rescue_rate": rescued / failures if failures else None,
            "harm_rate": harmed / (n - failures) if n > failures else None, "mcnemar_exact_p": p}



def artifact_hashes(attempt):
    attempt = Path(attempt)
    paths = [attempt / "command.json"]
    paths += sorted((attempt / "output").rglob("*.jsonl"))
    paths += sorted((attempt / "output").rglob("*.json"))
    return {str(p): digest(p) for p in paths if p.is_file()}


def artifacts_match(status):
    expected = status.get("artifact_sha256")
    return bool(expected) and expected == artifact_hashes(status["attempt"])


def analyze(manifest):
    output = Path(manifest["output"])
    manifest_path = output / "manifest.json"
    guard = output / "manifest.sha256"
    if guard.exists() and digest(manifest_path) != guard.read_text().strip():
        raise ValueError("Manifest changed after first run")
    rows, indexed = [], {}
    for job in jobs(manifest):
        path = output / "runs" / job["job_id"] / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {"state": "pending"}
        result = {}
        if status.get("attempt") and "returncode" in status:
            result = summarize_attempt(job, status["attempt"], status["returncode"], manifest["protocol"]["wait_threshold"])
            if not artifacts_match(status):
                result.update(valid=False, success=None, invalid_reason="completed_artifacts_changed")
            if status["state"] != "complete":
                result.update(valid=False, success=None, invalid_reason=result.get("invalid_reason") or "attempt_not_complete")
        row = {"task_id": job["task"]["task_id"], "scene_id": job["task"]["scene_id"],
               "episode_id": job["task"]["episode_id"], "group": job["group"], "agent": job["agent"],
               "repeat": job["repeat"], "injection_step": job["task"]["injection_step"],
               "state": status["state"], "valid": False, "success": None, **result}
        if row["state"] == "complete" and not row["valid"]:
            row["state"] = "error"
        rows.append(row)
        indexed[(row["task_id"], row["group"], row["agent"], row["repeat"])] = row
    strata, paired_rows = {}, []
    tasks = [t for t in manifest["tasks"] if t["eligible"]]
    for agent in manifest["protocol"]["agents"]:
        pairs = []
        for task in tasks:
            for repeat in range(manifest["protocol"]["repeats"]):
                d = indexed[(task["task_id"], "dropout", agent, repeat)]
                r = indexed[(task["task_id"], "dropout_recovery", agent, repeat)]
                c = indexed[(task["task_id"], "clean", None, repeat)]
                complete = d["valid"] and r["valid"]
                paired_rows.append({"task_id": task["task_id"], "scene_id": task["scene_id"],
                    "episode_id": task["episode_id"], "agent": agent, "repeat": repeat,
                    "complete_pair": complete, "clean_success": c["success"],
                    "dropout_success": d["success"], "recovery_success": r["success"]})
                # A single prespecified repeat per task for exact McNemar; never pool correlated repeats.
                if repeat == 0 and complete:
                    pairs.append((d["success"], r["success"]))
        strata[agent] = {**paired_statistics(pairs), "expected_pairs": len(tasks),
                         "missing_pairs": len(tasks) - len(pairs), "inferential_repeat": 0,
                         "complete": len(pairs) == len(tasks)}
    ordered = sorted((value["mcnemar_exact_p"], agent) for agent, value in strata.items() if value["n"])
    previous = 0
    for i, (p, agent) in enumerate(ordered):
        previous = max(previous, min(1.0, p * (len(ordered) - i)))
        strata[agent]["mcnemar_holm_p"] = previous
    group_summary = {}
    for group in ("clean", "dropout", "dropout_recovery"):
        selected = [r for r in rows if r["group"] == group]
        valid = [r for r in selected if r["valid"]]
        group_summary[group] = {"expected": len(selected), "valid": len(valid),
            "successes": sum(r["success"] for r in valid),
            "success_rate": sum(r["success"] for r in valid) / len(valid) if valid else None,
            "double_wait_count": sum(r.get("double_wait_detected", False) for r in valid),
            "offline_wait_count": sum(r.get("offline_wait_detected", False) for r in valid),
            "termination_reasons": dict(Counter(r.get("termination_reason") for r in valid))}
    summary = {"validation_only": bool(manifest.get("validation_only", False)),
               "selection": manifest["selection"], "groups": group_summary,
               "paired_by_agent": strata, "states": dict(Counter(r["state"] for r in rows)),
               "inference_scope": "Original clean-success cohort; repeat 0 only; separate fault agents; Holm across agents. Incomplete pairs are provisional, never imputed as failures.",
               "waiting_window": "simulator_step >= original injection_step for all groups; terminal success steps excluded",
               "ci_method": "paired task-level percentile bootstrap, 10000 resamples; degenerate intervals need caution"}
    report = output / "analysis"
    report.mkdir(exist_ok=True)
    for name, records in (("episodes.csv", rows), ("pairs.csv", paired_rows)):
        fields = list(dict.fromkeys(k for row in records for k in row))
        with (report / name).open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    write_json(report / "summary.json", summary)
    lines = ["# Paired recovery experiment", "", "Selection: " + json.dumps(manifest["selection"]), "",
             "| Group | Valid / planned | Success | Double wait | Offline + wait |",
             "|---|---:|---:|---:|---:|"]
    for group, value in group_summary.items():
        lines.append(f"| {group} | {value['valid']}/{value['expected']} | {value['successes']} | {value['double_wait_count']} | {value['offline_wait_count']} |")
    lines += ["", "Only final pddl_success counts as success. recovery_completed is a control-flow event.", "",
              "Partial results are provisional. Missing/API/replay errors are not task failures.", "",
              "Inference uses repeat 0, paired by task, stratified by fault agent, with Holm correction.", ""]
    for agent, stats in strata.items():
        lines += [f"## {agent}", "", "```json", json.dumps(stats, indent=2), "```", ""]
    if summary["validation_only"]:
        lines = ["**VALIDATION ONLY — FixedModel; not empirical model performance.**", ""] + lines
    (report / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
