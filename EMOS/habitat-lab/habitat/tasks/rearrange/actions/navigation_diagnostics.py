"""Fail-open observations of navigation calculations, without simulator queries."""

import math


def _detach(value):
    """Copy only JSON values; invalid numeric evidence stays unknown."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _detach(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_detach(item) for item in value]
    if hasattr(value, 'tolist'):
        return _detach(value.tolist())
    return None


def _record_navigation_diagnostics(action, values=None, phase='begin'):
    """Publish existing local values only; an observer failure cannot affect control.

    Calls live at the original calculation sites. In particular, path failure is
    recorded before the controller's unchanged straight-line fallback, and the
    final command is captured before invoking the physical control action.
    """
    try:
        values = values or {}
        if phase in ('begin', 'reset'):
            # Invalidate old evidence before touching optional task metadata.
            action._navigation_diagnostics = dict(
                env_step=None, timing='pre_control', available=False, active=False,
                control=None, skill_done=None, reason='observation_unavailable',
            )
            try:
                step = getattr(getattr(action, '_task', None), '_cur_episode_step', None)
                step = None if step is None or phase == 'reset' else int(step) + 1
            except Exception:
                step = None
            data = dict.fromkeys((
                'requested_target', 'target_index', 'target_position',
                'proxy_position', 'waypoint_position', 'robot_position',
                'path_found', 'distance_to_goal', 'distance_threshold',
                'angle_to_object', 'angle_to_waypoint', 'angle_threshold',
                'distance_satisfied', 'orientation_satisfied', 'at_goal',
                'control', 'skill_done', 'target_cache_hit',
            ))
            data.update(
                env_step=step,
                timing='pre_control', source=type(action).__name__,
                available=False, active=False,
                motion_type=getattr(action, 'motion_type', None),
                prev_nav_done=getattr(action, 'prev_nav_done', None),
                reason='reset' if phase == 'reset' else 'not_evaluated',
            )
        else:
            data = dict(getattr(action, '_navigation_diagnostics', {}))
        if phase == 'input':
            if 'nav_to_target_idx' in values:
                data['requested_target'] = values['nav_to_target_idx']
            else:
                data.update(requested_target=values.get('nav_position_coord'),
                            requested_look_at=values.get('nav_to_target_coord'),
                            mode=values.get('mode'))
            data['reason'] = 'invalid_or_inactive_command'
        elif phase in ('cache_hit', 'cache_miss'):
            data['target_cache_hit'] = phase == 'cache_hit'
        elif phase == 'target':
            data.update(active=True, reason='control_not_computed',
                        proxy_position=values['final_nav_targ'],
                        target_position=values['obj_targ_pos'])
            index = values.get('nav_to_target_idx')
            if index is not None:
                data['target_index'] = index
                data['requested_target'] = getattr(action._poss_entities[index], 'name', None)
        elif phase == 'path':
            data['path_found'] = bool(values['found_path'])
        elif phase == 'geometry':
            distance = values['dist_to_final_nav_targ']
            angle = values['angle_to_obj']
            data.update(
                robot_position=values['robot_pos'],
                waypoint_position=values['cur_nav_targ'],
                distance_to_goal=distance, distance_threshold=action._config.dist_thresh,
                angle_to_object=angle, angle_to_waypoint=values['angle_to_target'],
                angle_threshold=action._config.turn_thresh,
                distance_satisfied=bool(distance < action._config.dist_thresh) if math.isfinite(distance) else None,
                orientation_satisfied=bool(angle < action._config.turn_thresh) if math.isfinite(angle) else None,
                at_goal=bool(values['at_goal']),
            )
            if 'distance_close_no_distance' in values:
                data['distance_without_orientation_threshold'] = values['distance_close_no_distance']
        elif phase == 'control':
            data.update(available=True, active=True, reason=None,
                        control=values.get('base_action', values.get('vel')),
                        skill_done=action.skill_done,
                        prev_nav_done=getattr(action, 'prev_nav_done', None))
        # Keep the names consumed by existing trajectory readers and detectors.
        for name, source in (
            ('target', 'requested_target'),
            ('true_object_position', 'target_position'),
            ('final_navigation_position', 'proxy_position'),
            ('proxy_point', 'proxy_position'),
            ('next_waypoint', 'waypoint_position'),
            ('robot_position_before_control', 'robot_position'),
        ):
            data[name] = data.get(source)
        action._navigation_diagnostics = _detach(data)
    except Exception:
        # Observation must neither replace the physical result nor mask its error.
        pass
