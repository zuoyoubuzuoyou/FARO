"""Validated clean-prefix replay. No Habitat or model imports are needed here.

Run the live skill policies with recorded high-level decisions, and reject any
action/state divergence before branching. This preserves skill/RNN progression
instead of teleporting poses or restarting a policy at the injection boundary.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
import json
import math
from pathlib import Path


class ReplayMismatchError(ValueError):
    """The run no longer reproduces the recorded clean prefix."""


def compare_recorded(expected, actual, *, path="value", atol=1e-5):
    """Compare JSON data with absolute numeric tolerance; fail on missing data."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise ReplayMismatchError(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, (int, float)):
        if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                or not math.isfinite(expected) or not math.isfinite(actual)
                or abs(expected - actual) > atol):
            raise ReplayMismatchError(f"{path}: expected {expected!r}, got {actual!r} (atol={atol})")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ReplayMismatchError(f"{path}: list length/type mismatch")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare_recorded(left, right, path=f"{path}[{i}]", atol=atol)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ReplayMismatchError(f"{path}: expected mapping")
        for key, value in expected.items():
            if key not in actual:
                raise ReplayMismatchError(f"{path}.{key}: missing")
            compare_recorded(value, actual[key], path=f"{path}.{key}", atol=atol)
    else:
        raise ReplayMismatchError(f"{path}: unsupported recorded value")


def physical_state(state):
    """Compare all recorded physical fields, excluding trace annotations."""
    if not isinstance(state, dict):
        raise ValueError("Replay requires a simulator_state mapping")
    agents = state.get("agents", {})
    if not isinstance(agents, dict) or not agents or not isinstance(state.get("objects"), list) or state.get("timing") != "post_env_step":
        raise ValueError("Replay requires post_env_step agent/object snapshots in JSONL")
    fields = ("position", "rotation", "held_object")
    result = {"agents": {}, "objects": state["objects"]}
    for name, agent in agents.items():
        if not isinstance(agent, dict) or any(key not in agent for key in fields):
            raise ValueError(f"Replay requires position, rotation and held_object for {name}")
        result["agents"][name] = {key: agent[key] for key in fields}
    return result


class CleanPrefixReplay:
    def __init__(self, path, *, atol=1e-5):
        if not math.isfinite(atol) or atol < 0:
            raise ValueError("Replay tolerance must be finite and nonnegative")
        self.path = str(Path(path).resolve())
        self.atol = atol
        self.episodes = {}
        self.contexts = {}
        completed = set()
        with open(self.path, encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Replay JSONL rows must be objects")
                if row.get("type") not in ("episode_start", "simulator_step"):
                    continue
                if row.get("mode") != "clean":
                    raise ValueError("Replay requires a clean JSONL trajectory")
                if not {"scene_id", "episode_id"} <= row.keys():
                    raise ValueError("Replay row is missing scene_id/episode_id")
                key = (str(row["scene_id"]), str(row["episode_id"]))
                if row["type"] == "episode_start":
                    if key in self.contexts or key in self.episodes:
                        raise ValueError(f"Repeated replay episode: {key}")
                    self.contexts[key] = row.get("text_context", {})
                    continue
                rows = self.episodes.setdefault(key, [])
                step = row.get("simulator_step")
                if key in completed or type(step) is not int or step != len(rows):
                    raise ValueError(f"Missing, duplicate or unordered replay step for {key}")
                rows.append(row)
                if row.get("done") is True:
                    completed.add(key)
        self.episodes = {key: rows for key, rows in self.episodes.items() if key in completed}
        self.key = None
        self.branch_step = None
        self.verified_steps = 0

    def validate_episode(self, scene_id, episode_id, branch_step):
        key = (str(scene_id), str(episode_id))
        rows = self.episodes.get(key)
        if not rows:
            raise ValueError(f"No complete clean replay trajectory for {key}")
        # Replaying at least one decision is necessary to establish policy state.
        if not 1 <= branch_step < len(rows):
            raise ValueError(f"Replay branch step for {key} must be in [1, {len(rows) - 1}]")
        names = set(physical_state(rows[0].get("simulator_state", {}))["agents"])
        for step, row in enumerate(rows[:branch_step]):
            if not isinstance(row.get("joint_env_action"), list) or not row["joint_env_action"]:
                raise ValueError(f"Replay step {step} is missing joint_env_action")
            if not isinstance(row.get("policy_info"), dict) or not isinstance(row.get("metrics"), dict):
                raise ValueError(f"Replay step {step} is missing policy_info/metrics")
            if "reward" not in row or row.get("done") is not False:
                raise ValueError(f"Replay step {step} has invalid reward/done")
            if set(physical_state(row.get("simulator_state", {}))["agents"]) != names:
                raise ValueError("Replay agent set changed within episode")
            for name in names:
                decision = row["policy_info"].get(f"{name}_high_level_decision")
                if step == 0 and decision is None:
                    raise ValueError(f"Replay needs initial high-level decision for {name}")
                if decision is not None and (not isinstance(decision, dict) or not {"action", "arguments"} <= decision.keys()):
                    raise ValueError(f"Incomplete replay decision for {name} at {step}")
        return rows

    def start_episode(self, scene_id, episode_id, branch_step, agent_names):
        rows = self.validate_episode(scene_id, episode_id, branch_step)
        if set(rows[0]["simulator_state"]["agents"]) != set(agent_names):
            raise ValueError("Configured agents differ from clean replay")
        self.key = (str(scene_id), str(episode_id))
        self.branch_step = branch_step
        self.verified_steps = 0
        decisions = [value for row in rows[:branch_step]
                     for key, value in row["policy_info"].items()
                     if key.endswith("_high_level_decision")]
        self.context_fidelity = (
            "recorded" if all("replay_context" in d for d in decisions)
            else "decision_history_only"
        )
        return self.report("replay_start")

    def report(self, event):
        return {"type": event, "source": self.path,
                "scene_id": self.key[0], "episode_id": self.key[1],
                "branch_step": self.branch_step, "verified_steps": self.verified_steps,
                "atol": self.atol, "context_fidelity": self.context_fidelity}

    @contextmanager
    def record_failures(self, recorder):
        try:
            yield
        except ValueError as error:
            recorder.record({**self.report("replay_failed"), "error": str(error)})
            raise

    def prefix_row(self, step):
        if step >= self.branch_step:
            if self.verified_steps != self.branch_step:
                raise ReplayMismatchError("Cannot inject before the complete prefix is verified")
            return None
        if step != self.verified_steps:
            raise ReplayMismatchError(f"Expected replay step {self.verified_steps}, got {step}")
        return self.episodes[self.key][step]

    def initial_decision(self, agent_name):
        return copy.deepcopy(self.episodes[self.key][0]["policy_info"][f"{agent_name}_high_level_decision"])

    def verify_action(self, row, action, policy_info):
        step = row["simulator_step"]
        compare_recorded(row["joint_env_action"], action, path=f"step {step}.joint_env_action", atol=self.atol)
        expected = {k: v for k, v in row["policy_info"].items() if k.endswith("_high_level_decision")}
        actual = {k: v for k, v in (policy_info or {}).items() if k.endswith("_high_level_decision")}
        if expected.keys() != actual.keys():
            raise ReplayMismatchError(f"step {step}: high-level decision timing diverged")
        for key, decision in expected.items():
            for field in ("action", "arguments"):
                compare_recorded(decision[field], actual[key][field], path=f"step {step}.{key}.{field}", atol=0)

    def verify_result(self, row, state, reward, done, metrics):
        step = row["simulator_step"]
        expected_state = physical_state(row["simulator_state"])
        actual_state = physical_state(state or {})
        if expected_state["agents"].keys() != actual_state["agents"].keys():
            raise ReplayMismatchError(f"step {step}: agent set differs")
        compare_recorded(expected_state, actual_state, path=f"step {step}.state", atol=self.atol)
        compare_recorded(row["done"], bool(done), path=f"step {step}.done", atol=0)
        compare_recorded(row["reward"], reward, path=f"step {step}.reward", atol=self.atol)
        # Policy diagnostic keys may change with recording versions. Task metrics
        # must still match, including detection history and collision counters.
        task_metrics = {k: v for k, v in row["metrics"].items() if not k.startswith("agent_")}
        compare_recorded(task_metrics, metrics, path=f"step {step}.metrics", atol=self.atol)
        self.verified_steps += 1
