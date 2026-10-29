"""Narrow AST regions for execution repairs, retaining the original method shell.

This is a finite syntax profile, not a Python sandbox or a proof of task
correctness. It trusts the host's original source and runtime tensor types.
Calls, geometry, thresholds, observations, diagnostics and motion code outside
the declared regions must retain their exact original AST and control position.
"""
import ast
from copy import deepcopy
import math


_SKILLS = "habitat-baselines/habitat_baselines/rl/hrl/skills/"
_NAV = "habitat-lab/habitat/tasks/rearrange/actions/oracle_nav_action.py"
_DIFF_NAV = "habitat-lab/habitat/tasks/rearrange/actions/habitat_mas_actions.py"
EXECUTION_METHODS = {
    (_SKILLS + "wait.py", "WaitSkillPolicy._is_skill_done"):
        {"process": "parent", "region": "wait"},
    (_SKILLS + "skill.py", "SkillPolicy.should_terminate"):
        {"process": "parent", "region": "interrupt"},
    (_SKILLS + "oracle_nav.py", "OracleNavPolicy._is_skill_done"):
        {"process": "parent", "region": "sensor"},
    (_NAV, "OracleNavAction.step"):
        {"process": "worker", "region": "navigation"},
    (_NAV, "OracleNavCoordinateAction.step"):
        {"process": "worker", "region": "navigation"},
    (_DIFF_NAV, "OracleNavDiffBaseAction.step"):
        {"process": "worker", "region": "navigation"},
}


def execution_target(relative, qualified):
    """Return an exact declared execution target, or None for other methods."""
    return EXECUTION_METHODS.get((relative, qualified))


def _reject(detail):
    raise ValueError("Restricted execution profile: " + detail)


def _expression(code):
    return ast.parse(code, mode="eval").body


def _equal(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _matches(node, code):
    return _equal(node, _expression(code))


def _number(node, minimum=0):
    return (isinstance(node, ast.Constant) and type(node.value) in {int, float}
            and math.isfinite(node.value) and node.value >= minimum)


def _slot(node):
    node.value = ast.Constant(value="<execution-expression-slot>")


def _same_shell(candidate, original):
    if not _equal(candidate, original):
        _reject("edits outside the declared execution region are read-only")


def _single(nodes, label):
    if len(nodes) != 1:
        _reject("expected one original " + label + " region")
    return nodes[0]


def _assignments(method, name):
    return [node for node in ast.walk(method) if isinstance(node, ast.Assign)
            and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name]


def _conversions(node):
    """Unwrap only bounded, non-mutating CPU/boolean tensor conversions."""
    count = 0
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        call = node
        if call.func.attr in {"cpu", "bool"} and not call.args and not call.keywords:
            pass
        elif call.func.attr == "to" and (
                (len(call.args) == 1 and not call.keywords
                 and _matches(call.args[0], "torch.bool"))
                or (not call.args and len(call.keywords) == 1
                    and call.keywords[0].arg == "dtype"
                    and _matches(call.keywords[0].value, "torch.bool"))):
            pass
        else:
            break
        count += 1
        if count > 4:
            _reject("tensor conversion chains must be bounded")
        node = call.func.value
    return node


def _predicate(node, atom, tensor=False):
    """Require the declared bound along every OR path of a finite predicate.

An AND keeps a mandatory bound if one child has it; an OR keeps it only
when every child has it. All leaves are validated, including other AND
children. No constants, negation, computed calls or arbitrary arithmetic.
"""
    if tensor:
        node = _conversions(node)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitAnd, ast.BitOr)) and tensor:
        proofs = [_predicate(node.left, atom, tensor), _predicate(node.right, atom, tensor)]
        return any(proofs) if isinstance(node.op, ast.BitAnd) else all(proofs)
    if isinstance(node, ast.BoolOp) and not tensor:
        proofs = [_predicate(value, atom, tensor) for value in node.values]
        return any(proofs) if isinstance(node.op, ast.And) else all(proofs)
    return atom(node)


def _step_comparison(node, indexed, duration=False):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Gt, ast.GtE)) and len(node.comparators) == 1
            and _matches(node.left, "self._cur_skill_step[batch_idx]" if indexed
                         else "self._cur_skill_step")):
        _reject("timing predicates require the original elapsed steps and batch indices")
    bound = node.comparators[0]
    if duration and _matches(bound, "self._wait_time"):
        return True
    if _number(bound):
        return False
    _reject("timing predicates cannot shorten or replace the requested wait duration")


def _wait(candidate, original):
    returns = []
    for method in (candidate, original):
        result = _single([node for node in ast.walk(method) if isinstance(node, ast.Return)], "wait return")
        value = _conversions(result.value)
        outer_index = isinstance(value, ast.Subscript) and _matches(value.slice, "batch_idx")
        if outer_index:
            value = value.value
        if not _predicate(value, lambda node: _step_comparison(node, not outer_index, duration=True), tensor=True):
            _reject("every wait completion path must retain the requested duration bound")
        returns.append(result)
    for result in returns:
        _slot(result)
    _same_shell(candidate, original)


def _interrupt(candidate, original):
    for method in (candidate, original):
        combination = _single([node for node in ast.walk(method)
            if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
            and node.target.id == "is_skill_done" and isinstance(node.op, ast.BitOr)], "interrupt combination")
        if not _matches(_conversions(combination.value), "hl_wants_skill_term"):
            _reject("interrupt combination must use the actual high-level interrupt signal")
        _slot(combination)
    _same_shell(candidate, original)


def _sensor_and(node, source):
    node = _conversions(node)
    if _equal(node, source):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        proofs = [_sensor_and(node.left, source), _sensor_and(node.right, source)]
        return any(proofs)
    _step_comparison(node, indexed=True)
    return False


def _sensor(candidate, original):
    original_values = _assignments(original, "ret")
    candidate_values = _assignments(candidate, "ret")
    if len(original_values) != 2 or len(candidate_values) != 2:
        _reject("navigation sensor repairs retain the original result assignments")
    source = original_values[-1].value
    if not _matches(source, "finish_oracle_nav.to(torch.bool)[:, 0]"):
        _reject("the original navigation completion sensor layout is unsupported")
    for method, values in ((candidate, candidate_values), (original, original_values)):
        if not _sensor_and(values[-1].value, source):
            _reject("navigation completion must retain the real sensor signal")
        _slot(values[-1])
        result = _single([node for node in ast.walk(method) if isinstance(node, ast.Return)], "sensor return")
        if not _sensor_and(result.value, _expression("ret")):
            _reject("navigation completion cannot replace the actual sensor result")
        _slot(result)
    _same_shell(candidate, original)


def _threshold(node, field, aliases):
    if _matches(node, "self._config." + field):
        return True
    if isinstance(node, ast.Name) and node.id in aliases:
        return field == "dist_thresh"
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            and _matches(node.left, "self._config." + field) and _number(node.right, 1))


def _navigation_atom(node, aliases):
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.Lt, ast.LtE)) and len(node.comparators) == 1):
        _reject("navigation completion requires original distance/angle threshold comparisons")
    if _matches(node.left, "dist_to_final_nav_targ") and _threshold(node.comparators[0], "dist_thresh", aliases):
        return True
    if _matches(node.left, "angle_to_obj") and _threshold(node.comparators[0], "turn_thresh", set()):
        return False
    _reject("navigation completion cannot replace raw geometry or inflate thresholds")


_CACHED_COMPLETION = ast.parse(
    "if self.prev_nav_done and int(nav_to_target_idx[0]) - 1 == self.prev_match_target_id:\n"
    "    nav_to_target_idx = 0\n").body[0]
_COMPLETION_WRITE = ast.parse("self.skill_done = True").body[0]


def _at_goal_branches(method):
    return [node for node in ast.walk(method) if isinstance(node, ast.If)
            and _matches(node.test, "not at_goal")]


def _navigation(candidate, original, clear_cache=False):
    aliases = set()
    for node in _assignments(original, "distance_close_no_distance"):
        if _threshold(node.value, "dist_thresh", set()):
            aliases.add("distance_close_no_distance")
    for method in (candidate, original):
        predicate = _single(_assignments(method, "at_goal"), "navigation completion predicate")
        if not _predicate(predicate.value, lambda node: _navigation_atom(node, aliases)):
            _reject("every navigation completion OR branch must retain a real distance bound")
        _slot(predicate)

    branches, originals = _at_goal_branches(candidate), _at_goal_branches(original)
    if not originals or len(branches) != len(originals):
        _reject("navigation motion branches and control positions must be retained")
    for branch, before in zip(branches, originals):
        existing = any(_equal(node, _COMPLETION_WRITE) for node in before.orelse)
        if not existing and branch.orelse and _equal(branch.orelse[-1], _COMPLETION_WRITE):
            # Only append a missing signal to an existing true-at_goal branch.
            # Existing completion signals cannot be removed or relocated.
            branch.orelse.pop()

    if clear_cache:
        _single([node for node in original.body if _equal(node, _CACHED_COMPLETION)], "cached-completion suppression")
        for method in (candidate, original):
            matches = [node for node in method.body if _equal(node, _CACHED_COMPLETION)]
            if len(matches) > 1:
                _reject("cached-completion suppression cannot be duplicated")
            method.body = [node for node in method.body if not _equal(node, _CACHED_COMPLETION)]
    _same_shell(candidate, original)


def validate_execution_method(relative, qualified, method, original):
    """Validate a declared region at prepare/install; return False if unhandled.

Never execute original or proposed methods. A recognized target requires an
existing ordinary method; all syntax outside its narrowly validated regions,
including signature/decorators, remains exactly equal to the original AST.
"""
    declaration = execution_target(relative, qualified)
    if declaration is None:
        return False
    if not isinstance(method, ast.FunctionDef) or not isinstance(original, ast.FunctionDef):
        _reject("execution repairs require an existing ordinary method")
    if method.name != qualified.split(".")[1] or original.name != method.name:
        _reject("execution method name must match its declared target")
    candidate, before = deepcopy(method), deepcopy(original)
    region = declaration["region"]
    if region == "navigation":
        _navigation(candidate, before, clear_cache=qualified == "OracleNavDiffBaseAction.step")
    else:
        {"wait": _wait, "interrupt": _interrupt, "sensor": _sensor}[region](candidate, before)
    return True
