"""Causal baselines; an action name alone is intentionally a weak baseline."""
from collections import deque
from copy import deepcopy
import math
import re
import json


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return tuple(float(v) for v in value) if all(math.isfinite(float(v)) for v in value) else None
    except (ValueError, TypeError):
        return None


def _distance(a, b):
    a, b = _vector(a), _vector(b)
    return math.dist(a, b) if a is not None and b is not None else None


def _target_position(frame, name):
    positions = frame.get('context', {}).get('goal_positions') or {}
    return positions.get(name) or positions.get(name.removeprefix('TARGET_'))


def assess_goals(frame):
    """Current requirements only; coverage distances are conservative proxies."""
    task = frame['task'].lower()
    objects = frame.get('objects') or {}
    metrics = frame.get('metrics') or {}
    result = []
    if task in ('rearrange', 'manipulation'):
        for i in range(2):
            obj = 'any_targets|' + str(i)
            distance = metrics.get('object_to_goal_distance.' + str(i))
            if distance is None:
                distance = _distance(objects.get(obj, {}).get('position'), _target_position(frame, obj))
            result.append({'kind': 'placement', 'object': obj,
                           'satisfied': None if distance is None else distance < 0.3,
                           'distance': distance, 'threshold': 0.3, 'approximate': False})
    elif task in ('mobility', 'perception'):
        names = ['any_targets|0', 'TARGET_any_targets|0'] if task == 'mobility' else ['any_targets|0', 'any_targets|1']
        threshold = 2.0 if task == 'mobility' else 1.0
        for name in names:
            position = _target_position(frame, name) if name.startswith('TARGET_') else objects.get(name, {}).get('position')
            distances = [_distance(a.get('position'), position) for a in frame['agents'].values()]
            available = [v for v in distances if v is not None]
            distance = min(available) if available else None
            # Logged base references differ from native predicate transforms.
            satisfied = None if distance is None else True if distance <= threshold else False if len(available) == len(distances) and distance > threshold + 0.5 else None
            result.append({'kind': 'coverage', 'object': name, 'satisfied': satisfied,
                           'distance': distance, 'threshold': threshold, 'approximate': True})
        if task == 'perception':
            for name in ['any_targets|0', 'any_targets|1']:
                flags = []; unknown = False
                for a in frame['agents'].values():
                    detection = a.get('detection') or {}
                    if detection.get('eligible') is False:
                        continue
                    value = (detection.get('target_detections') or {}).get(name)
                    if detection.get('available') is not False and isinstance(value, bool): flags.append(value)
                    else: unknown = True
                result.append({'kind': 'detection', 'object': name,
                               'satisfied': True if any(flags) else False if flags and not unknown else None,
                               'threshold': 'semantic_pixels>10', 'approximate': False})
    else:
        raise ValueError('Unsupported detection task: ' + task)
    if metrics.get('pddl_success') == 1:
        for r in result: r['satisfied'] = True
    return result


def _action(agent):
    return agent.get('action') or {}


def _owns(frame, agent, requirement):
    text = (frame.get('context', {}).get('assignments') or {}).get(agent, '')
    if re.search(r'no feasible|nothing to do|no task|cannot be completed', text, re.I):
        return False
    name = requirement['object']
    if text and frame['task'].lower() in ('rearrange', 'manipulation'):
        if not re.search(r'\b(?:pick|grasp|place|carry|transport|retrieve|receive|handle|manipulate|move|deliver)\b', text, re.I):
            return False
        if re.search(r'\bdo not\b[^.;]*' + re.escape(name), text, re.I):
            return False
    identifiers = re.findall(r'(?:TARGET_)?any_targets\|\d+', text)
    if not text:
        arguments = _action(frame['agents'][agent]).get('arguments') or {}
        if isinstance(arguments, dict): identifiers = [str(v) for v in arguments.values()]
    if frame['task'].lower() in ('rearrange', 'manipulation'):
        return name in [v.removeprefix('TARGET_') for v in identifiers]
    return name in identifiers


def _motion(frames, agent=None, tolerance=0.05):
    """Any observed movement counts, including an out-and-back trajectory."""
    points = {}
    rotations = []
    holding = []
    for f in frames:
        if agent is not None:
            a = f['agents'].get(agent, {})
            points.setdefault(agent, []).append(a.get('position'))
            rotations.append(a.get('rotation'))
            holding.append(a.get('held_object'))
        else:
            for key, obj in f.get('objects', {}).items():
                points.setdefault(key, []).append(obj.get('position'))
    observed = False
    for positions in points.values():
        known = [p for p in positions if _vector(p) is not None]
        if known and any(_distance(known[0], p) > tolerance for p in known[1:]): return True
        if len(known) == len(frames) and len(known) >= 2: observed = True
    if agent is not None:
        if any(h != holding[0] for h in holding[1:]): return True
        scalar = [r for r in rotations if isinstance(r, (int, float)) and math.isfinite(r)]
        if scalar and any(abs(math.atan2(math.sin(r-scalar[0]), math.cos(r-scalar[0]))) > 0.08 for r in scalar[1:]): return True
        quats = [r for r in rotations if isinstance(r, (list, tuple)) and len(r) == 4]
        if quats:
            initial = quats[0]
            for q in quats[1:]:
                norm = math.sqrt(sum(v*v for v in initial) * sum(v*v for v in q))
                if norm and 2*math.acos(min(1.0, abs(sum(a*b for a,b in zip(initial,q)))/norm)) > 0.08: return True
        if len(scalar) != len(frames) and len(quats) != len(frames): return None
    return False if observed else None


def _progress(frames, agent=None):
    """Distances and fresh detections can prove progress without base motion."""
    if len(frames) < 2: return False
    first = frames[0]
    for current in list(frames)[1:]:
        for key, old in first.get('metrics', {}).items():
            new = current.get('metrics', {}).get(key)
            if key.startswith('object_to_goal_distance.') and isinstance(old,(int,float)) and isinstance(new,(int,float)) and old-new > 0.02:
                return True
        names = [agent] if agent is not None else current['agents']
        for ag in names:
            old = first['agents'].get(ag,{})
            new = current['agents'].get(ag,{})
            old_distance=(old.get('navigation') or {}).get('distance_to_goal')
            new_distance=(new.get('navigation') or {}).get('distance_to_goal')
            if isinstance(old_distance,(int,float)) and isinstance(new_distance,(int,float)) and old_distance-new_distance > 0.02: return True
            old_flags=(old.get('detection') or {}).get('target_detections') or {}
            new_flags=(new.get('detection') or {}).get('target_detections') or {}
            if any(value is True and old_flags.get(key) is False for key,value in new_flags.items()): return True
    return False


def _scope(agent):
    action=_action(agent)
    return (action.get('name'), action.get('id'), json.dumps(action.get('arguments'), sort_keys=True))


def _same_quiet_scope(previous, current, agent):
    """Reissued waits can continue one idle interval; a new skill cannot."""
    old = previous['agents'].get(agent, {})
    new = current['agents'].get(agent, {})
    old_assignment = (previous.get('context', {}).get('assignments') or {}).get(agent)
    new_assignment = (current.get('context', {}).get('assignments') or {}).get(agent)
    if old_assignment != new_assignment:
        return False
    if _action(old).get('name') == _action(new).get('name') == 'wait':
        return True
    return _scope(old) == _scope(new)


def _quiet_history(anchor, current, agents, observations):
    """Retain the actual older observation that established a quiet interval.

    A ten-frame display window may be shorter than min_stall_steps. Its first
    frame is therefore not evidence for the entire duration; this summary
    identifies the retained historical anchor and the number of observations.
    """
    states = {}
    for agent in agents:
        state = anchor['agents'].get(agent, {})
        states[agent] = {key: deepcopy(state.get(key)) for key in
                         ('position', 'rotation', 'held_object', 'action', 'navigation')}
    return {'duration_env_steps': current['step'] - anchor['step'],
            'observations': observations,
            'anchor_observation': {'step': anchor['step'], 'agents': states,
                                   'objects': deepcopy(anchor.get('objects', {})),
                                   'metrics': deepcopy(anchor.get('metrics', {}))}}


def _alarm(frame, detector, code, explanation, agents, first_step, evidence):
    return {'step': frame['step'], 'detector': detector, 'status': 'system_error',
            'reason_code': code, 'explanation': explanation, 'agents': agents,
            'confidence': None, 'evidence': [{'step': first_step, 'observation': '观察窗口开始'},
                                           {'step': frame['step'], 'observation': evidence}]}


class RepeatActionDetector:
    """Literal consecutive-action-name baseline; no team reasoning."""
    name = 'repeat_action'

    def __init__(self, threshold=10):
        if threshold < 1: raise ValueError('threshold must be positive')
        self.threshold = threshold
        self.last = {}; self.counts = {}; self.starts = {}

    def observe(self, frame):
        repeated = []
        # Unknown or absent observations cannot bridge a consecutive run.
        for ag in set(self.last) - set(frame['agents']):
            self.last.pop(ag, None); self.counts.pop(ag, None); self.starts.pop(ag, None)
        for ag, a in frame['agents'].items():
            name = _action(a).get('name')
            if name is None:
                self.last.pop(ag, None); self.counts.pop(ag, None); self.starts.pop(ag, None)
                continue
            if self.last.get(ag) == name:
                self.counts[ag] += 1
            else:
                self.counts[ag] = 1; self.starts[ag] = frame['step']
            self.last[ag] = name
            if self.counts[ag] > self.threshold: repeated.append(ag)
        if repeated:
            return _alarm(frame, self.name, 'repeated_action_name',
                          '动作名连续超过阈值；这是未经团队目标校验的简单基线，可能误报正常长动作。',
                          repeated, min(self.starts[a] for a in repeated), {'consecutive_observations': {a:self.counts[a] for a in repeated}})
        return None


class TeamProgressDetector:
    name = 'team_progress'

    def __init__(self, window=10, check_every=5, min_stall_steps=100, reset_budget=100, motion_tolerance=0.05):
        if window < 2 or check_every < 1 or min_stall_steps < 1: raise ValueError('Invalid monitoring window/cadence')
        self.frames = deque(maxlen=window); self.window = window; self.check_every = check_every
        self.min_stall_steps = min_stall_steps; self.reset_budget = reset_budget
        self.motion_tolerance = motion_tolerance; self.count = 0
        self.still_since = {}; self.team_still_since = None
        self.quiet_anchors = {}; self.quiet_observations = {}
        self.team_quiet_anchor = None; self.team_quiet_observations = 0

    def observe(self, frame):
        previous = self.frames[-1] if self.frames else None
        # Compare to a retained anchor, not only the previous sample: many
        # small normal translations or rotations must accumulate into motion.
        for ag in frame['agents']:
            anchor = self.quiet_anchors.get(ag)
            unchanged = (previous is not None and anchor is not None
                         and _same_quiet_scope(previous, frame, ag)
                         and _motion([anchor, frame], ag, self.motion_tolerance) is False
                         and not _progress([anchor, frame], ag))
            if unchanged:
                self.quiet_observations[ag] += 1
            else:
                self.quiet_anchors[ag] = deepcopy(frame)
                self.quiet_observations[ag] = 1
            self.still_since[ag] = self.quiet_anchors[ag]['step']
        for ag in set(self.quiet_anchors) - set(frame['agents']):
            self.quiet_anchors.pop(ag, None); self.quiet_observations.pop(ag, None)
            self.still_since.pop(ag, None)
        anchor = self.team_quiet_anchor
        same_team = previous is not None and set(previous['agents']) == set(frame['agents'])
        team_unchanged = (same_team and anchor is not None
                          and all(_same_quiet_scope(previous, frame, ag) for ag in frame['agents'])
                          and not _progress([anchor, frame])
                          and _motion([anchor, frame], tolerance=self.motion_tolerance) is False
                          and all(_motion([anchor, frame], ag, self.motion_tolerance) is False
                                  for ag in frame['agents']))
        if team_unchanged:
            self.team_quiet_observations += 1
        else:
            self.team_quiet_anchor = deepcopy(frame)
            self.team_quiet_observations = 1
        self.team_still_since = self.team_quiet_anchor['step']
        self.frames.append(deepcopy(frame)); self.count += 1
        if len(self.frames) < self.window or (self.count-self.window) % self.check_every: return None
        if (frame.get('metrics') or {}).get('pddl_success') == 1: return None
        first = self.frames[0]['step']; duration = frame['step'] - first
        unmet = [r for r in assess_goals(frame) if r['satisfied'] is False]
        if not unmet: return None
        owners = {ag:[r for r in unmet if _owns(frame,ag,r)] for ag in frame['agents']}
        stalled = {ag:reqs for ag,reqs in owners.items() if reqs
                   and _motion(self.frames,ag,self.motion_tolerance) is False
                   and not _progress(self.frames,ag)
                   and frame['step']-self.still_since.get(ag,frame['step']) >= self.min_stall_steps}
        for ag, reqs in stalled.items():
            a = frame['agents'][ag]; action = _action(a); name = action.get('name')
            same = all(_scope(f['agents'].get(ag,{})) == _scope(a) for f in self.frames)
            if name == 'reset_arm' and same and (action.get('age') or 0) > self.reset_budget:
                return _alarm(frame,self.name,'reset_over_budget',
                              '仍有未完成职责的机器人持续停留在复位，超过已知技能预算且未见进度；不据此推断机械原因。',
                              [ag],first,{'action_age':action.get('age'),'reset_budget':self.reset_budget,'unmet':reqs,
                                          'quiet_history':_quiet_history(self.quiet_anchors[ag],frame,[ag],self.quiet_observations[ag])})
            if name == 'nav_to_obj' and same and (action.get('age') or 0) >= self.min_stall_steps:
                return _alarm(frame,self.name,'navigation_no_progress',
                              '承担未完成目标的机器人持续执行同一导航，窗口内位置、朝向及持物均无有效变化。',
                              [ag],first,{'action_age':action.get('age'),'unmet':reqs,'navigation':a.get('navigation'),'termination':a.get('termination'),
                                          'quiet_history':_quiet_history(self.quiet_anchors[ag],frame,[ag],self.quiet_observations[ag])})
        if _motion(self.frames, tolerance=self.motion_tolerance) is not False or _progress(self.frames): return None
        if any(_motion(self.frames,ag,self.motion_tolerance) is not False for ag in frame['agents']): return None
        if frame['step']-self.team_still_since < self.min_stall_steps: return None
        # Arm joint motion is unavailable in historical traces. An active pick
        # or place cannot be declared idle just because the base stays still.
        if any(_action(f['agents'].get(ag,{})).get('name') not in ('wait','reset_arm')
               for f in self.frames for ag in frame['agents']): return None
        waiting = [ag for ag in stalled if all(_action(f['agents'].get(ag,{})).get('name') in ('wait','reset_arm') for f in self.frames)]
        if waiting:
            return _alarm(frame,self.name,'unresolved_team_wait',
                          '团队当前仍有未完成且已分配的职责，责任人持续等待／复位，全队窗口内未见实际推进。',
                          waiting,first,{'unmet':unmet,'window_env_steps':duration,
                                         'quiet_history':_quiet_history(self.team_quiet_anchor,frame,list(frame['agents']),self.team_quiet_observations)})
        return None
