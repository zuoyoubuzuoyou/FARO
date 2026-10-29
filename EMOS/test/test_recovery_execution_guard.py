"""Execution edits retain actual timing, sensor input and original motion code."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

try:
    from habitat_mas.recovery import source_execution_guard as guard
except ImportError:
    guard = None


ROOT = Path(__file__).resolve().parents[1]
SKILLS = "habitat-baselines/habitat_baselines/rl/hrl/skills/"
NAV = "habitat-lab/habitat/tasks/rearrange/actions/oracle_nav_action.py"
DIFF_NAV = "habitat-lab/habitat/tasks/rearrange/actions/habitat_mas_actions.py"
WAIT_TARGET = (SKILLS + "wait.py", "WaitSkillPolicy._is_skill_done")
INTERRUPT_TARGET = (SKILLS + "skill.py", "SkillPolicy.should_terminate")
SENSOR_TARGET = (SKILLS + "oracle_nav.py", "OracleNavPolicy._is_skill_done")
NAV_TARGETS = [(NAV, "OracleNavAction.step"),
               (NAV, "OracleNavCoordinateAction.step"),
               (DIFF_NAV, "OracleNavDiffBaseAction.step")]


def production_method(target):
    relative, qualified = target
    class_name, method_name = qualified.split(".")
    owner = next(node for node in ast.parse((ROOT / relative).read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == class_name)
    return next(node for node in owner.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name)


def expression(code):
    return ast.parse(code, mode="eval").body


def assignment(method, name):
    return next(node for node in ast.walk(method) if isinstance(node, ast.Assign)
                and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == name)


def completion_branches(method):
    return [node for node in ast.walk(method) if isinstance(node, ast.If)
            and ast.dump(node.test) == ast.dump(expression("not at_goal"))]


class ExecutionMethodGuardTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(guard, "execution-region validation has not been implemented")

    def validate(self, target, candidate, original=None):
        return guard.validate_execution_method(*target, candidate,
            production_method(target) if original is None else original)

    def reject(self, target, candidate, original=None):
        with self.assertRaisesRegex(ValueError, "execution|Execution"):
            self.validate(target, candidate, original)

    def test_exact_target_registry_preserves_process_boundaries(self):
        for target in [WAIT_TARGET, INTERRUPT_TARGET, SENSOR_TARGET] + NAV_TARGETS:
            process = "worker" if target in NAV_TARGETS else "parent"
            self.assertEqual(guard.execution_target(*target)["process"], process)
            self.assertEqual(guard.EXECUTION_METHODS[target]["process"], process)
            method = production_method(target)
            self.assertTrue(self.validate(target, deepcopy(method), method))
        self.assertIsNone(guard.execution_target(NAV, "OracleNavAction.reset"))
        self.assertIsNone(guard.execution_target("other.py", "OracleNavAction.step"))
        method = production_method(WAIT_TARGET)
        self.assertFalse(guard.validate_execution_method("other.py", "Other.act", method, method))

    def test_wait_can_only_complete_after_the_requested_duration(self):
        original = production_method(WAIT_TARGET)
        candidate = deepcopy(original)
        result = next(node for node in candidate.body if isinstance(node, ast.Return))
        result.value = expression("(self._cur_skill_step > self._wait_time)[batch_idx]")
        self.assertTrue(self.validate(WAIT_TARGET, candidate, original))
        candidate = deepcopy(original)
        next(node for node in candidate.body if isinstance(node, ast.Return)).value = expression(
            "(self._cur_skill_step[batch_idx] >= self._wait_time) & "
            "(self._cur_skill_step[batch_idx] > 2)")
        self.assertTrue(self.validate(WAIT_TARGET, candidate, original))
        namespace = {"torch": torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[])),
                     "<wait-predicate>", "exec"), namespace)
        policy = SimpleNamespace(_cur_skill_step=torch.tensor([1., 2., 3., 4.]), _wait_time=3)
        result = namespace[candidate.name](policy, None, None, None, None, [0, 1, 2, 3])
        self.assertEqual(result.tolist(), [False, False, True, True])

    def test_wait_rejects_false_completion_and_state_changes(self):
        original = production_method(WAIT_TARGET)
        for code in ["True", "torch.ones(len(batch_idx), dtype=torch.bool)",
                     "(self._cur_skill_step >= 0)[batch_idx]",
                     "(self._cur_skill_step >= self._wait_time - 100)[batch_idx]",
                     "(self._cur_skill_step >= self._wait_time)[batch_idx] | True",
                     "(-self._cur_skill_step <= self._wait_time)[batch_idx]",
                     "(self._cur_skill_step <= self._wait_time)[batch_idx]",
                     "(self._cur_skill_step >= self._wait_time)[0]"]:
            candidate = deepcopy(original)
            next(node for node in candidate.body if isinstance(node, ast.Return)).value = expression(code)
            with self.subTest(code=code):
                self.reject(WAIT_TARGET, candidate, original)
        candidate = deepcopy(original)
        candidate.body.insert(0, ast.parse("self._wait_time = 1").body[0])
        self.reject(WAIT_TARGET, candidate, original)

    def test_interrupt_conversion_preserves_postconditions_logging_and_actions(self):
        original = production_method(INTERRUPT_TARGET)
        candidate = deepcopy(original)
        combination = next(node for node in ast.walk(candidate)
                           if isinstance(node, ast.AugAssign)
                           and isinstance(node.target, ast.Name) and node.target.id == "is_skill_done")
        combination.value = expression("hl_wants_skill_term.cpu().to(torch.bool)")
        self.assertTrue(self.validate(INTERRUPT_TARGET, candidate, original))
        for code in ["True", "torch.ones_like(is_skill_done)", "hl_wants_skill_term.fill_(True)",
                     "hl_wants_skill_term.to(torch.float32)"]:
            invalid = deepcopy(candidate)
            next(node for node in ast.walk(invalid) if isinstance(node, ast.AugAssign)
                 and isinstance(node.target, ast.Name) and node.target.id == "is_skill_done").value = expression(code)
            with self.subTest(code=code):
                self.reject(INTERRUPT_TARGET, invalid, original)
        for code in ["self._apply_postconds = False", "actions[:] = 0", "self._max_skill_steps = 1"]:
            invalid = deepcopy(candidate)
            invalid.body.insert(0, ast.parse(code).body[0])
            self.reject(INTERRUPT_TARGET, invalid, original)

    def test_sensor_completion_can_be_gated_without_creating_positive_signals(self):
        original = production_method(SENSOR_TARGET)
        candidate = deepcopy(original)
        ret = [node for node in candidate.body if isinstance(node, ast.Assign)
               and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "ret"][-1]
        ret.value = expression("finish_oracle_nav.to(torch.bool)[:, 0] & (self._cur_skill_step[batch_idx] > 0)")
        self.assertTrue(self.validate(SENSOR_TARGET, candidate, original))
        namespace = {"torch": torch, "HasFinishedOracleNavSensor": SimpleNamespace(cls_uuid="finished")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[])),
                     "<sensor-predicate>", "exec"), namespace)
        policy = SimpleNamespace(_cur_skill_step=torch.tensor([0., 1., 2., 3.]))
        result = namespace[candidate.name](policy, {"finished": torch.tensor([[1], [0], [1], [0]])},
            None, None, torch.ones(4, 1), [0, 1, 2, 3])
        self.assertEqual(result.tolist(), [False, False, True, False])
        candidate = deepcopy(original)
        next(node for node in candidate.body if isinstance(node, ast.Return)).value = expression(
            "ret & (self._cur_skill_step[batch_idx] >= 1)")
        self.assertTrue(self.validate(SENSOR_TARGET, candidate, original))

    def test_sensor_rejects_constant_success_forged_input_and_or_gates(self):
        original = production_method(SENSOR_TARGET)
        for code in ["True", "ret | (self._cur_skill_step[batch_idx] > 0)",
                     "self._cur_skill_step[batch_idx] > 0", "ret | True", "ret.fill_(True)"]:
            candidate = deepcopy(original)
            next(node for node in candidate.body if isinstance(node, ast.Return)).value = expression(code)
            self.reject(SENSOR_TARGET, candidate, original)
        candidate = deepcopy(original)
        assignment(candidate, "finish_oracle_nav").value = expression("torch.ones_like(masks)")
        self.reject(SENSOR_TARGET, candidate, original)

    def test_navigation_accepts_bounded_geometry_predicates(self):
        for target in NAV_TARGETS:
            original = production_method(target)
            for code in ["dist_to_final_nav_targ < self._config.dist_thresh",
                         "(dist_to_final_nav_targ < self._config.dist_thresh / 10.0) or "
                         "(dist_to_final_nav_targ < self._config.dist_thresh and angle_to_obj < self._config.turn_thresh)"]:
                candidate = deepcopy(original)
                assignment(candidate, "at_goal").value = expression(code)
                with self.subTest(target=target, code=code):
                    self.assertTrue(self.validate(target, candidate, original))

    def test_navigation_can_fill_missing_completion_and_recheck_cached_targets(self):
        target = NAV_TARGETS[1]
        original = production_method(target)
        candidate = deepcopy(original)
        additions = 0
        for branch in completion_branches(candidate):
            if not any(isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute)
                       and node.targets[0].attr == "skill_done" for node in branch.orelse):
                branch.orelse.append(ast.parse("self.skill_done = True").body[0])
                additions += 1
        self.assertEqual(additions, 2)
        self.assertTrue(self.validate(target, candidate, original))
        target = NAV_TARGETS[2]
        original = production_method(target)
        candidate = deepcopy(original)
        suppression = next(node for node in candidate.body if isinstance(node, ast.If)
                           and "prev_nav_done" in ast.unparse(node.test))
        candidate.body.remove(suppression)
        self.assertTrue(self.validate(target, candidate, original))

    def test_navigation_rejects_constant_success_negative_distance_and_threshold_inflation(self):
        for target in NAV_TARGETS:
            original = production_method(target)
            for code in ["True", "dist_to_final_nav_targ < 1000000",
                         "-dist_to_final_nav_targ < self._config.dist_thresh",
                         "dist_to_final_nav_targ < self._config.dist_thresh * 100",
                         "dist_to_final_nav_targ < self._config.dist_thresh / 0.01",
                         "dist_to_final_nav_targ < self._config.dist_thresh or angle_to_obj < self._config.turn_thresh",
                         "dist_to_final_nav_targ == dist_to_final_nav_targ",
                         "dist_to_final_nav_targ < self._config.dist_thresh or True"]:
                candidate = deepcopy(original)
                assignment(candidate, "at_goal").value = expression(code)
                with self.subTest(target=target, code=code):
                    self.reject(target, candidate, original)

    def test_navigation_freezes_motion_geometry_task_state_and_log_control(self):
        original = production_method(NAV_TARGETS[0])
        for code in ["self.cur_articulated_agent.base_pos = [0, 0, 0]",
                     "self._task.goal = None", "self._task.measurements = {}",
                     "self._config.dist_thresh = 1000000", "self._sim.step({})",
                     "self._sim.reset()", "self.skill_done = True"]:
            candidate = deepcopy(original)
            candidate.body.insert(0, ast.parse(code).body[0])
            with self.subTest(code=code):
                self.reject(NAV_TARGETS[0], candidate, original)
        candidate = deepcopy(original)
        assignment(candidate, "dist_to_final_nav_targ").value = expression("-1")
        self.reject(NAV_TARGETS[0], candidate, original)
        candidate = deepcopy(original)
        diagnostic = next(node for node in ast.walk(candidate) if isinstance(node, ast.Call)
                          and isinstance(node.func, ast.Name) and node.func.id == "_record_navigation_diagnostics")
        diagnostic.func.id = "ignore_navigation_diagnostics"
        self.reject(NAV_TARGETS[0], candidate, original)
        candidate = deepcopy(original)
        motion_branch = next(node for node in ast.walk(candidate) if isinstance(node, ast.If)
                             and "base_velocity" in ast.unparse(node.test))
        motion = next(node for node in motion_branch.body if isinstance(node, ast.Expr)
                      and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                      and node.value.func.attr == "step")
        motion_branch.body.remove(motion)
        candidate.body.append(motion)
        self.reject(NAV_TARGETS[0], candidate, original)

    def test_navigation_completion_cannot_move_outside_original_at_goal_branches(self):
        original = production_method(NAV_TARGETS[1])
        candidate = deepcopy(original)
        completion_branches(candidate)[0].body.append(ast.parse("self.skill_done = True").body[0])
        self.reject(NAV_TARGETS[1], candidate, original)
        candidate = deepcopy(original)
        completion_branches(candidate)[0].orelse.append(ast.parse("self.skill_done = False").body[0])
        self.reject(NAV_TARGETS[1], candidate, original)

    def test_allowlisted_execution_methods_cannot_be_new_or_change_signature(self):
        original = production_method(WAIT_TARGET)
        with self.assertRaises(ValueError):
            guard.validate_execution_method(*WAIT_TARGET, original, None)
        candidate = deepcopy(original)
        candidate.args.args.append(ast.arg(arg="extra"))
        self.reject(WAIT_TARGET, candidate, original)


if __name__ == "__main__":
    unittest.main()
