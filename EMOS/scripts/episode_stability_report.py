"""Summarize a fixed repeated-episode cohort without simulation dependencies.

Rates are fractions in [0, 1], except ``success_rate_range_pp`` in percentage
points. Only completed trials with a literal bool are native S/F outcomes.
State agreement describes recorded hashes, not all simulator/physics state.
"""

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


TRIAL_STATUSES = frozenset({
    "pending", "running", "completed", "api_error", "execution_error",
    "process_error", "timeout", "recording_error", "configuration_changed",
})


def _identity(record):
    fields = ("task", "scene_id", "episode_id")
    if not isinstance(record, dict) or any(
        not isinstance(record.get(field), str) or not record[field]
        for field in fields
    ):
        raise ValueError("Each identity requires nonempty task, scene_id and string episode_id")
    return tuple(record[field] for field in fields)


def _outcome(trial):
    if trial is None:
        return "missing"
    if trial["status"] == "completed":
        if type(trial.get("task_success")) is bool:
            return "S" if trial["task_success"] else "F"
        return "invalid_completed"
    return trial["status"]


def _agreement(hashes):
    recorded = [value for value in hashes if value]
    if len(set(recorded)) > 1:
        return False
    if len(recorded) == len(hashes) and recorded:
        return True
    return None


def summarize(manifest: dict) -> dict:
    """Validate task/round identities and report native outcomes and coverage.

    Missing planned trial records remain explicit ``missing`` cells. Invalid
    completed metrics remain ``invalid_completed`` cells instead of becoming F.
    Duplicate/foreign identities and out-of-range rounds raise ValueError.
    """
    repetitions = manifest.get("settings", {}).get("repetitions")
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError("settings.repetitions must be a positive integer")
    cohort = manifest.get("cohort")
    trials = manifest.get("trials")
    if not isinstance(cohort, list) or not cohort or not isinstance(trials, list):
        raise ValueError("A nonempty cohort and trials list are required")
    identities = [_identity(record) for record in cohort]
    identity_set = set(identities)
    if len(identity_set) != len(identities):
        raise ValueError("Duplicate cohort identity")
    indexed = {}
    trial_ids = set()
    for trial in trials:
        identity = _identity(trial)
        if identity not in identity_set:
            raise ValueError("Foreign trial identity")
        round_number = trial.get("round")
        if type(round_number) is not int or not 1 <= round_number <= repetitions:
            raise ValueError("Trial round outside planned repetitions")
        key = (identity, round_number)
        if key in indexed:
            raise ValueError("Duplicate task/round identity")
        trial_id = trial.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id or trial_id in trial_ids:
            raise ValueError("Each trial requires a distinct nonempty trial_id")
        if not isinstance(trial.get("status"), str) or trial["status"] not in TRIAL_STATUSES:
            raise ValueError("Each trial requires a recognized status")
        trial_ids.add(trial_id)
        indexed[key] = trial

    task_outcomes = []
    for record, identity in zip(cohort, identities):
        task_trials = [indexed.get((identity, number)) for number in range(1, repetitions + 1)]
        hashes = [trial.get("initial_state_sha256") if trial else None for trial in task_trials]
        task_outcomes.append({
            "task": record["task"], "scene_id": record["scene_id"],
            "episode_id": record["episode_id"],
            "outcomes": [_outcome(trial) for trial in task_trials],
            "statuses": [trial["status"] if trial else "missing" for trial in task_trials],
            "initial_state_hashes": hashes,
            "initial_state_agreement": _agreement(hashes),
        })

    planned = len(cohort)
    rounds = []
    for index in range(repetitions):
        counts = Counter(task["outcomes"][index] for task in task_outcomes)
        completed = counts["S"] + counts["F"]
        rounds.append({
            "round": index + 1, "planned": planned, "completed": completed,
            "success": counts["S"], "failure": counts["F"],
            "other": planned - completed,
            "status_counts": dict(sorted(Counter(task["statuses"][index] for task in task_outcomes).items())),
            "completion_coverage": completed / planned,
            "success_rate": counts["S"] / planned if completed == planned else None,
        })

    pairwise = []
    for first, second in combinations(range(repetitions), 2):
        valid = [(task["outcomes"][first], task["outcomes"][second]) for task in task_outcomes
                 if task["outcomes"][first] in ("S", "F") and task["outcomes"][second] in ("S", "F")]
        denominator = len(valid)
        success_to_failure = sum(a == "S" and b == "F" for a, b in valid)
        failure_to_success = sum(a == "F" and b == "S" for a, b in valid)
        flips = success_to_failure + failure_to_success
        pairwise.append({
            "round_a": first + 1, "round_b": second + 1,
            "common_valid": denominator, "flips": flips,
            "flip_rate": flips / denominator if denominator else None,
            "success_to_failure": success_to_failure,
            "success_to_failure_rate": success_to_failure / denominator if denominator else None,
            "failure_to_success": failure_to_success,
            "failure_to_success_rate": failure_to_success / denominator if denominator else None,
        })

    complete = [task for task in task_outcomes if all(value in ("S", "F") for value in task["outcomes"])]
    changed = sum(len(set(task["outcomes"])) > 1 for task in complete)
    rates = [record["success_rate"] for record in rounds]
    agreements = [task["initial_state_agreement"] for task in task_outcomes]
    overall_agreement = False if False in agreements else (True if all(value is True for value in agreements) else None)
    return {
        "cohort_size": planned, "repetitions": repetitions,
        "planned_trials": planned * repetitions,
        "completed_trials": sum(record["completed"] for record in rounds),
        "rounds": rounds,
        "success_rate_range_pp": (max(rates) - min(rates)) * 100 if all(rate is not None for rate in rates) else None,
        "pairwise": pairwise,
        "any_change": {"common_valid": len(complete), "changed": changed,
                       "rate": changed / len(complete) if complete else None},
        "task_outcomes": task_outcomes,
        "initial_state_agreement": overall_agreement,
        "interpretation": "Descriptive repeated-cohort results only; a small pilot does not establish statistical stability. Initial-state agreement compares recorded hashes only.",
    }


def write_reports(manifest: dict, output: Path) -> dict:
    """Write summary.json and spreadsheet-compatible UTF-8 BOM CSV reports."""
    summary = summarize(manifest)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "rounds.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary["rounds"][0]))
        writer.writeheader()
        for row in summary["rounds"]:
            writer.writerow(dict(row, status_counts=json.dumps(row["status_counts"], sort_keys=True)))
    round_fields = [f"round_{number}" for number in range(1, summary["repetitions"] + 1)]
    with (output / "task_outcomes.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["task", "scene_id", "episode_id", *round_fields, "initial_state_agreement", "initial_state_hashes"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in summary["task_outcomes"]:
            row = {key: task[key] for key in ("task", "scene_id", "episode_id", "initial_state_agreement")}
            row.update(zip(round_fields, task["outcomes"]))
            row["initial_state_hashes"] = json.dumps(task["initial_state_hashes"])
            writer.writerow(row)
    return summary
