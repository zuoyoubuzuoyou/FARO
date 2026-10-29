"""Read-only runtime evidence, independent of Habitat and simulator imports.

Only already returned observations and existing object/configuration metadata are
consumed. In particular this module never evaluates a sensor or PDDL predicate.
"""

import math
from collections.abc import Mapping, Sequence


def _value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _value(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_value(item) for item in value]
    for method in ("tolist", "item"):
        if callable(getattr(value, method, None)):
            return _value(getattr(value, method)())
    return None


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _any_known(values):
    values = list(values)
    if any(value is True for value in values):
        return True
    if not values or any(value is None for value in values):
        return None
    return False


def _targets(simulator, sim_info, target_objects):
    """Join scene indices to PDDL aliases without searching/evaluating PDDL."""
    obj_ids = _mapping(getattr(sim_info, "obj_ids", None))
    if not obj_ids:
        obj_ids = _mapping(getattr(simulator, "handle_to_object_id", None))
    goal_ids = _mapping(getattr(sim_info, "target_ids", None))
    entities = _mapping(getattr(sim_info, "all_entities", None))
    offset = _number(getattr(getattr(simulator, "habitat_config", None), "object_ids_start", None))
    evidence = {}
    rows = []
    for ordinal, original in enumerate(target_objects or []):
        row = dict(original)
        index = row.get("scene_index")
        names = [str(name) for name, value in obj_ids.items() if value == index]
        preferred = [name for name in names if name in entities] if entities else names
        aliases = [name for name in preferred if name.startswith("any_targets|")]
        names = aliases or preferred
        if not names:
            names = [str(name)[len("TARGET_"):]
                     for name, value in goal_ids.items()
                     if value == index and str(name).startswith("TARGET_")]
        mapped = bool(names)
        if not names:
            names = [str(row.get("goal_name") or f"any_targets|{ordinal}")]
        semantic_id = None
        if mapped and row.get("object_id") is not None and offset is not None:
            semantic_id = int(row["object_id"]) + int(offset)
        row.update(goal_name=names[0], semantic_id=semantic_id)
        for name in names:
            evidence[name] = {
                "position": row.get("position"), "semantic_id": semantic_id,
                "mapping_available": mapped,
            }
            goals = [str(key) for key, value in goal_ids.items() if value == index]
            goal_name = f"TARGET_{name}"
            # The simulator's target list establishes the goal counterpart.
            if not goals or goal_name in goals:
                evidence[goal_name] = {
                    "position": row.get("goal_position"), "semantic_id": None,
                    "mapping_available": mapped,
                }
        rows.append(row)
    return rows, evidence


def _coverage(robot, targets, threshold, observations, env_step):
    coverage = {"env_step": env_step, "timing": "post_env_step",
                "reference": "base_transformation", "distance_threshold": threshold,
                "targets": {}}
    for name, target in targets.items():
        distance = None
        reason = None
        if observations is None:
            reason = "no_returned_observations"
        elif not target["mapping_available"]:
            reason = "target_entity_mapping_unavailable"
        elif target["position"] is None:
            reason = "target_position_unavailable"
        elif threshold is None:
            reason = "robot_at_threshold_unavailable"
        else:
            try:
                local = robot.base_transformation.inverted().transform_point(target["position"])
                distance = _number(math.sqrt(sum(float(value) ** 2 for value in local)))
                if distance is None:
                    reason = "nonfinite_native_distance"
            except Exception as error:
                reason = f"base_transformation_unavailable: {type(error).__name__}: {error}"
        result = {"distance": distance, "at_goal": distance <= threshold if distance is not None else None}
        if reason:
            result["reason"] = reason
        coverage["targets"][name] = result
    return coverage


def _pixel_counts(image):
    """Use vectorized unique for NumPy images, with no mandatory dependency."""
    if type(image).__module__.startswith("numpy"):
        import numpy as np
        ids, counts = np.unique(image, return_counts=True)
        return {int(key): int(count) for key, count in zip(ids, counts)}
    counts = {}
    def visit(value):
        if isinstance(value, Sequence):
            for child in value:
                visit(child)
        else:
            key = int(value)
            counts[key] = counts.get(key, 0) + 1
    visit(image)
    return counts


def _detection(name, task, observations, targets, env_step):
    sensors = _mapping(getattr(getattr(task, "sensor_suite", None), "sensors", None))
    sensor_name = f"{name}_detected_objects"
    if sensor_name not in sensors and sensor_name not in _mapping(observations):
        if "detected_objects" in sensors or "detected_objects" in _mapping(observations):
            sensor_name = "detected_objects"
    sensor = sensors.get(sensor_name)
    threshold = _number(getattr(sensor, "pixel_threshold", None))
    generic = sensor_name == "detected_objects"
    obs = _mapping(observations)
    semantic_keys = [str(key) for key in obs if "semantic" in str(key)
                     and (generic or str(key).startswith(name + "_"))]
    per_sensor = {}
    reason = None
    try:
        per_sensor = {key: _pixel_counts(obs[key]) for key in semantic_keys}
    except Exception as error:
        reason = f"semantic_observation_unavailable: {type(error).__name__}: {error}"
    detected_ids = None
    source = None
    if sensor_name in obs and obs[sensor_name] is not None:
        try:
            detected_ids = [int(value) for value in obs[sensor_name]]
            source = "returned_detected_objects"
        except Exception as error:
            reason = f"detected_objects_unavailable: {type(error).__name__}: {error}"
    elif per_sensor and threshold is not None:
        detected_ids = sorted({key for counts in per_sensor.values()
                               for key, count in counts.items() if count > threshold})
        source = "returned_semantic_pixel_counts"
    if detected_ids is None and reason is None:
        if observations is None:
            reason = "no_returned_observations"
        elif semantic_keys and threshold is None:
            reason = "pixel_threshold_unavailable"
        else:
            reason = "detected_objects_and_semantic_observations_unavailable"
    target_results = {}
    target_counts = {}
    target_reasons = {}
    for entity, target in targets.items():
        if entity.startswith("TARGET_"):
            continue
        semantic_id = target["semantic_id"]
        target_counts[entity] = {key: counts.get(semantic_id, 0) if semantic_id is not None else None
                                 for key, counts in per_sensor.items()}
        if semantic_id is None:
            target_results[entity] = None
            target_reasons[entity] = "target_semantic_id_mapping_unavailable"
        elif detected_ids is None:
            target_results[entity] = None
            target_reasons[entity] = reason
        else:
            target_results[entity] = bool(semantic_id in detected_ids)
    if target_reasons and reason is None:
        reason = "target_semantic_id_mapping_unavailable"
    return {
        "env_step": env_step, "timing": "post_env_step", "source": source,
        "available": detected_ids is not None, "eligible": sensor is not None or sensor_name in obs,
        "sensor": sensor_name if sensor is not None or sensor_name in obs else None,
        "pixel_threshold": threshold, "detected_object_ids": detected_ids,
        "target_detections": target_results, "target_pixel_counts": target_counts,
        "target_semantic_ids": {
            key: target["semantic_id"] for key, target in targets.items()
            if not key.startswith("TARGET_")
        },
        "semantic_sensors": semantic_keys,
        "target_pixel_counts_by_sensor": {
            key: {
                entity: counts.get(target["semantic_id"], 0)
                if target["semantic_id"] is not None else None
                for entity, target in targets.items()
                if not entity.startswith("TARGET_")
            }
            for key, counts in per_sensor.items()
        },
        "reason": reason, "target_reasons": target_reasons,
    }


def _navigation(name, task, executed_action, env_step, observations):
    selected = _mapping(executed_action).get("action", [])
    if isinstance(selected, str):
        selected = [selected]
    selected = selected if isinstance(selected, (list, tuple)) else []
    candidates = []
    for action_name, action in _mapping(getattr(task, "actions", None)).items():
        if not str(action_name).startswith(name + "_"):
            continue
        diagnostic = getattr(action, "_navigation_diagnostics", None)
        if isinstance(diagnostic, Mapping):
            candidates.append((action_name, diagnostic))
    if not candidates:
        return None
    action_name, diagnostic = max(
        candidates,
        key=lambda item: (item[0] in selected, item[1].get("env_step") == env_step),
    )
    result = _value(diagnostic)
    result["action"] = str(action_name)
    reason = None
    if observations is None:
        reason = "no_returned_observations"
    elif result.get("env_step") != env_step:
        reason = "stale_navigation_diagnostics"
    elif result.get("timing") != "pre_control":
        reason = "invalid_navigation_timing"
    elif action_name not in selected:
        reason = "navigation_action_not_selected"
    elif _inactive_command(name, action_name, executed_action):
        reason = "navigation_command_inactive"
    elif result.get("active") is not True:
        reason = result.get("reason") or "navigation_action_inactive"
    if reason:
        result.update(active=False, available=False, control=None, controller_command=None,
                      skill_done=None, reason=reason)
    return result


def _inactive_command(name, action_name, executed_action):
    arguments = _mapping(_mapping(executed_action).get("action_args"))
    # Indexed navigation uses zero as its inactive command. Coordinates use a
    # zero look-at vector. Read the recorded argument, never rerun the action.
    if action_name == name + "_oracle_nav_action" and action_name in arguments:
        command = arguments[action_name]
        if isinstance(command, (list, tuple)):
            command = command[0] if command else None
        number = _number(command)
        return number is None or number <= 0
    lookat = arguments.get(name + "_oracle_nav_lookat_action")
    if isinstance(lookat, (list, tuple)) and lookat:
        return all(_number(value) == 0 for value in lookat)
    return False


def _camera_metadata(agent_config):
    result = {}
    for name, config in _mapping(getattr(agent_config, "sim_sensors", None)).items():
        metadata = {}
        for field in ("type", "uuid", "height", "width", "hfov", "position", "orientation"):
            value = getattr(config, field, None)
            if value is not None:
                metadata[field] = _value(value)
        result[str(name)] = metadata
    return result


def collect_runtime_diagnostics(*, simulator, task, observations, agent_names,
                                target_objects, env_step, executed_action, agent_configs=None):
    """Collect finite, detached evidence; all live execution calls stay outside."""
    problem = getattr(task, "pddl_problem", None)
    sim_info = getattr(problem, "_sim_info", None)
    threshold = _number(getattr(sim_info, "robot_at_thresh", getattr(problem, "_robot_at_thresh", None)))
    rows, targets = _targets(simulator, sim_info, target_objects)
    agents = {}
    managers = getattr(simulator, "agents_mgr", None)
    for index, name in enumerate(agent_names):
        robot = managers[index].articulated_agent if managers is not None else None
        coverage = _coverage(robot, targets, threshold, observations, env_step)
        detection = _detection(str(name), task, observations, targets, env_step)
        agent_config = _mapping(agent_configs).get(name)
        agent = {"goal_coverage": coverage, "detection": detection,
                 "root_position": None, "root_position_reason": None,
                 "camera_sensors": _camera_metadata(agent_config)}
        try:
            root = robot.base_transformation.translation
            agent["root_position"] = [_number(value) for value in root]
        except Exception:
            agent["root_position_reason"] = "base_transformation_translation_unavailable"
        navigation = _navigation(str(name), task, executed_action, env_step, observations)
        if navigation is not None:
            agent["navigation"] = navigation
        agents[str(name)] = agent
    current = {"env_step": env_step, "timing": "post_env_step",
               "source": "native_base_transformation_and_returned_observations",
               "any_at": {}, "is_detected": {}, "reasons": {"any_at": {}, "is_detected": {}}}
    for entity in targets:
        at = _any_known(agent["goal_coverage"]["targets"][entity]["at_goal"] for agent in agents.values())
        current["any_at"][entity] = at
        if at is None:
            current["reasons"]["any_at"][entity] = "native_coverage_incomplete"
        if not entity.startswith("TARGET_"):
            detected = _any_known(
                agent["detection"]["target_detections"][entity]
                for agent in agents.values()
            )
            current["is_detected"][entity] = detected
            if detected is None:
                current["reasons"]["is_detected"][entity] = "current_detection_evidence_incomplete"
    return _value({
        "agents": agents, "current_goal": current,
        "target_objects": rows if target_objects is not None else None,
    })
