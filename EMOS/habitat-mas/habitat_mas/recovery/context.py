"""Fixed prompt positions and explicit repository source inputs."""
from copy import deepcopy
import hashlib
from pathlib import Path

from ..detection.llm import _data, _frame, _plain
from ..utils.trajectory import redact


SOURCE_FILES = {
    "simulation": "habitat-lab/habitat/core/environments.py",
    "execution_loop": "habitat-baselines/habitat_baselines/rl/multi_agent/habitat_mas_evaluator.py",
    "control": "habitat-baselines/habitat_baselines/rl/hrl/hierarchical_policy.py",
    "skills": "habitat-baselines/habitat_baselines/rl/hrl/skills/skill.py",
    "policy": "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py",
    "agent": "habitat-mas/habitat_mas/agents/crab_agent.py",
    "model_calls": "habitat-mas/habitat_mas/utils/models.py",
}

AVAILABLE_API = {
    "entrypoint": "recover(context, memory)",
    "return": {"commands": "list of commands, at most one per agent", "memory": "JSON object persisted within this episode"},
    "commands": {
        "resume": {"type": "resume", "agent": "actual agent ID", "reason": "why to interrupt and select a fresh skill"},
        "replan": {"type": "replan", "agent": "actual agent ID", "instruction": "corrective planning instruction"},
        "action": {"type": "action", "agent": "actual agent ID", "name": "configured skill", "arguments": "argument object matching that agent's available_tools"},
    },
    "execution": "Branches, assignments, bounded for loops and JSON data operations only; no imports, files, network, globals or simulator objects.",
    "builtins": ["len", "min", "max", "abs", "sum", "range", "enumerate", "zip", "list", "dict", "str", "int", "float", "bool", "round", "sorted", "any", "all"],
    "methods": ["get", "items", "keys", "values", "append", "count", "index", "copy", "update", "pop", "lower", "upper", "split", "startswith", "endswith", "strip"],
}


def source_snapshot():
    root = Path(__file__).resolve().parents[3]
    result = {}
    for role, relative in SOURCE_FILES.items():
        path = root / relative
        try:
            content = path.read_text(encoding="utf-8")
            result[role] = {"path": relative, "sha256": hashlib.sha256(content.encode()).hexdigest(),
                            "content": content[:64000], "truncated": len(content) > 64000}
        except OSError as error:
            result[role] = {"path": relative, "content": None, "missing_reason": type(error).__name__}
    return redact(result)


def build_context(frame, recent_frames, error_report, policy, config, versions, memory, sources=None):
    runtime = _data(_plain(policy.get_recovery_context(history_turns=config.history_turns)))
    agents = runtime.get("agents") or {}
    context = frame.get("context") or {}
    value = {
        "schema_version": "emos-code-recovery-v1",
        "modification_instruction": config.modification_instruction,
        "error_report": _data(deepcopy(error_report)),
        "task": {"type": frame.get("task"), "description": context.get("task_description"),
                 "scene_description": context.get("scene_description"),
                 "assignments": context.get("assignments"), "goal_positions": context.get("goal_positions"),
                 "robot_capabilities": context.get("robot_capabilities")},
        "planner_plan": {name: agent.get("planner_plan") for name, agent in agents.items()},
        "recent_history": {name: (agent.get("recent_history") or [])[-config.history_turns:] for name, agent in agents.items()},
        "runtime": {"current_frame": _frame(frame), "recent_frames": [_frame(f) for f in recent_frames],
                    "agents": agents},
        "code": deepcopy(sources if sources is not None else source_snapshot()),
        "recovery": {"iteration": len(versions) + 1, "previous_versions": deepcopy(versions),
                     "memory": deepcopy(memory)},
        "available_api": deepcopy(AVAILABLE_API),
    }
    value["task"] = _data(value["task"])
    return redact(value)
