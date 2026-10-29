"""Remove exact runtime duplicates with a small, readable input declaration.

This is an input-only transformation. Sampled frames, events, dynamic goals,
agent poses and action ages are retained. ``expand_runtime_context`` provides
a local check that the original runtime can be recovered without information
loss; source paths in the declaration are documentation, not an arbitrary
reference language.
"""
from copy import deepcopy
import json


_SHARED_FRAME_FIELDS = ("assignments", "objects", "camera_sensors")
_SHARED_AGENT_FIELDS = ("camera_sensors", "robot_type")
_AGENT_FIELDS = ("task_description", "subtask_description", "planner_plan", "recent_history")


def _equal(left, right):
    """JSON equality that does not equate false, zero, or different types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _agent_sources(context, agent):
    """The four supported source locations; missing and null stay unknown."""
    task = context.get("task")
    task = task if isinstance(task, dict) else {}
    sources = (
        ("task_description", task, "description", "$.task.description"),
        ("subtask_description", task.get("assignments"), agent,
         "$.task.assignments[" + json.dumps(agent, ensure_ascii=False) + "]"),
        ("planner_plan", context.get("planner_plan"), agent,
         "$.planner_plan[" + json.dumps(agent, ensure_ascii=False) + "]"),
        ("recent_history", context.get("recent_history"), agent,
         "$.recent_history[" + json.dumps(agent, ensure_ascii=False) + "]"),
    )
    return {field: (values[key], path) for field, values, key, path in sources
            if isinstance(values, dict) and key in values and values[key] is not None}


def _retained_frames(runtime):
    frames = runtime.get("recent_frames")
    result = [("$.runtime.recent_frames[" + str(index) + "]", frame)
              for index, frame in enumerate(frames)] if isinstance(frames, list) else []
    if "current_frame" in runtime:
        result.append(("$.runtime.current_frame", runtime["current_frame"]))
    return result


def _shared_agent_fields(frames):
    """Extract only the fixed per-agent sensor and robot metadata fields."""
    first_agents = frames[0][1].get("agents")
    shared, mappings = {}, {}
    if not isinstance(first_agents, dict):
        return shared, mappings
    for agent, first_fields in first_agents.items():
        if not isinstance(first_fields, dict):
            continue
        agent_path = ".agents[" + json.dumps(agent, ensure_ascii=False) + "]"
        agent_frames = [(path, frame.get("agents", {}).get(agent)) for path, frame in frames
                        if isinstance(frame.get("agents"), dict)]
        if len(agent_frames) != len(frames) or not all(isinstance(fields, dict) for _, fields in agent_frames):
            continue
        for field in _SHARED_AGENT_FIELDS:
            first = first_fields.get(field)
            if first is None or not all(field in fields and _equal(fields[field], first) for _, fields in agent_frames):
                continue
            shared.setdefault(agent, {})[field] = deepcopy(first)
            mappings.setdefault(agent, {})[field] = {
                "source_path": "$.runtime.shared_frame_fields" + agent_path + "." + field,
                "omitted_paths": [path + agent_path + "." + field for path, _ in agent_frames],
                "steps": [frame.get("step") for _, frame in frames],
            }
    # Collect every omission before editing: Python inputs can alias agent maps.
    for _, frame in frames:
        for agent, fields in shared.items():
            for field in fields:
                frame["agents"][agent].pop(field, None)
    return shared, mappings


def compact_runtime_context(context: dict) -> dict:
    """Deep-copy a context and explicitly omit only exact runtime duplicates.

    Null fields are kept. Shared frame fields apply to every retained frame
    only when all those frames explicitly supply the same non-null value.
    Existing reserved compaction data is left alone, making this idempotent.
    """
    result = deepcopy(context)
    runtime = result.get("runtime")
    layout = result.get("input_layout", {})
    if not isinstance(runtime, dict) or not isinstance(layout, dict) or "runtime_compaction" in layout:
        return result
    declaration = {}
    recent = runtime.get("recent_frames")
    current = runtime.get("current_frame")
    if isinstance(current, dict) and isinstance(recent, list):
        for index in reversed(range(len(recent))):
            if _equal(current, recent[index]):
                declaration["current_frame"] = {
                    "omitted_path": "$.runtime.current_frame",
                    "source_path": "$.runtime.recent_frames[" + str(index) + "]",
                    "recent_frames_index": index, "step": current.get("step"),
                }
                del runtime["current_frame"]
                break

    agents = runtime.get("agents")
    if isinstance(agents, dict):
        mappings = []
        for agent, fields in agents.items():
            if not isinstance(fields, dict):
                continue
            # Python callers may reuse one dictionary for several agents.
            fields = deepcopy(fields)
            agents[agent] = fields
            for field, (value, source_path) in _agent_sources(result, agent).items():
                if field in fields and fields[field] is not None and _equal(fields[field], value):
                    del fields[field]
                    mappings.append({"agent": agent, "field": field,
                        "omitted_path": "$.runtime.agents[" + json.dumps(agent, ensure_ascii=False) + "]." + field,
                        "source_path": source_path})
        if mappings:
            declaration["agent_fields"] = mappings

    frames = _retained_frames(runtime)
    if ("shared_frame_fields" not in runtime and len(frames) >= 2
            and all(isinstance(frame, dict) for _, frame in frames)):
        shared, mappings = {}, {}
        for field in _SHARED_FRAME_FIELDS:
            first = frames[0][1].get(field)
            if first is None or not all(field in frame and _equal(frame[field], first) for _, frame in frames):
                continue
            shared[field] = deepcopy(first)
            mappings[field] = {
                "source_path": "$.runtime.shared_frame_fields." + field,
                "omitted_paths": [path + "." + field for path, _ in frames],
                "steps": [frame.get("step") for _, frame in frames],
            }
            for _, frame in frames:
                frame.pop(field, None)
        shared_agents, agent_mappings = _shared_agent_fields(frames)
        if shared_agents:
            shared["agents"] = shared_agents
            mappings["agents"] = agent_mappings
        if shared:
            runtime["shared_frame_fields"] = shared
            declaration["shared_frame_fields"] = mappings

    if declaration:
        declaration["schema_version"] = 1
        declaration["rules"] = (
            "Only listed fields are omitted; all other missing or null fields remain unknown. "
            "current_frame is the indicated recent_frames entry at the stated step. "
            "Agent fields equal their stated top-level source. "
            "Each shared_frame_fields value applies exactly to every retained recent_frames/current_frame "
            "entry listed by omitted_paths and steps; its agents map shares only camera_sensors/robot_type. "
            "Sampled steps and dynamic fields are unchanged."
        )
        result.setdefault("input_layout", {})["runtime_compaction"] = declaration
    return result


def expand_runtime_context(context: dict) -> dict:
    """Deep-copy and restore this module's fixed runtime omissions for checks."""
    result = deepcopy(context)
    layout, runtime = result.get("input_layout"), result.get("runtime")
    declaration = layout.get("runtime_compaction") if isinstance(layout, dict) else None
    if not isinstance(runtime, dict) or not isinstance(declaration, dict) or declaration.get("schema_version") != 1:
        return result

    shared = runtime.get("shared_frame_fields")
    shared_mappings = declaration.get("shared_frame_fields", {})
    if isinstance(shared, dict) and isinstance(shared_mappings, dict):
        for field in _SHARED_FRAME_FIELDS:
            if field in shared_mappings and field in shared:
                for _, frame in _retained_frames(runtime):
                    if isinstance(frame, dict) and field not in frame:
                        frame[field] = deepcopy(shared[field])
        shared_agents, agent_mappings = shared.get("agents"), shared_mappings.get("agents")
        if isinstance(shared_agents, dict) and isinstance(agent_mappings, dict):
            for agent, fields in agent_mappings.items():
                values = shared_agents.get(agent)
                if not isinstance(fields, dict) or not isinstance(values, dict):
                    continue
                for field in _SHARED_AGENT_FIELDS:
                    if field not in fields or field not in values:
                        continue
                    for _, frame in _retained_frames(runtime):
                        agents = frame.get("agents") if isinstance(frame, dict) else None
                        target = agents.get(agent) if isinstance(agents, dict) else None
                        if isinstance(target, dict) and field not in target:
                            target[field] = deepcopy(values[field])
        if shared_mappings:
            del runtime["shared_frame_fields"]

    agents = runtime.get("agents")
    if isinstance(agents, dict):
        for entry in declaration.get("agent_fields", []):
            agent, field = entry["agent"], entry["field"]
            fields, sources = agents.get(agent), _agent_sources(result, agent)
            if isinstance(fields, dict) and field in _AGENT_FIELDS and field in sources and field not in fields:
                fields[field] = deepcopy(sources[field][0])

    current, recent = declaration.get("current_frame"), runtime.get("recent_frames")
    if isinstance(current, dict) and isinstance(recent, list) and "current_frame" not in runtime:
        runtime["current_frame"] = deepcopy(recent[current["recent_frames_index"]])
    del layout["runtime_compaction"]
    if not layout:
        del result["input_layout"]
    return result
