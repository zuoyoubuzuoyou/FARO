"""Reviewed method edits and live installation on existing EMOS class objects.

The syntax checks are a guard against accidental capabilities, not a Python
security sandbox. Generated methods run in the same process as their owner.
"""
import ast
from collections import Counter
from copy import deepcopy
import difflib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import types
from uuid import uuid4

from .source_execution_guard import execution_target, validate_execution_method


SOURCE_TARGETS = {
    "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py": {
        "module": "habitat_baselines.rl.multi_agent.multi_llm_policy",
        "classes": ["MultiLLMPolicy"], "process": "parent"},
    "habitat-mas/habitat_mas/agents/crab_agent.py": {
        "module": "habitat_mas.agents.crab_agent", "classes": ["CrabAgent"], "process": "parent"},
    "habitat-mas/habitat_mas/utils/models.py": {
        "module": "habitat_mas.utils.models", "classes": ["OpenAIModel"], "process": "parent"},
    "habitat-baselines/habitat_baselines/rl/hrl/hierarchical_policy.py": {
        "module": "habitat_baselines.rl.hrl.hierarchical_policy",
        "classes": ["HierarchicalPolicy"], "process": "parent"},
    "habitat-baselines/habitat_baselines/rl/hrl/hl/llm_policy.py": {
        "module": "habitat_baselines.rl.hrl.hl.llm_policy",
        "classes": ["LLMHighLevelPolicy"], "process": "parent"},
    "habitat-baselines/habitat_baselines/rl/hrl/skills/skill.py": {
        "module": "habitat_baselines.rl.hrl.skills.skill", "classes": ["SkillPolicy"], "process": "parent"},
    "habitat-baselines/habitat_baselines/rl/hrl/skills/oracle_nav.py": {
        "module": "habitat_baselines.rl.hrl.skills.oracle_nav",
        "classes": ["OracleNavPolicy", "OracleNavCoordPolicy", "OracleNavHumanPolicy"], "process": "parent"},
    "habitat-baselines/habitat_baselines/rl/hrl/skills/oracle_arm_policies.py": {
        "module": "habitat_baselines.rl.hrl.skills.oracle_arm_policies",
        "classes": ["OraclePickPolicy", "OraclePlacePolicy"], "process": "parent"},
    "habitat-baselines/habitat_baselines/rl/hrl/skills/wait.py": {
        "module": "habitat_baselines.rl.hrl.skills.wait", "classes": ["WaitSkillPolicy"], "process": "parent"},
    "habitat-lab/habitat/tasks/rearrange/actions/oracle_nav_action.py": {
        "module": "habitat.tasks.rearrange.actions.oracle_nav_action",
        "classes": ["OracleNavAction", "OracleNavCoordinateAction"], "process": "worker"},
    "habitat-lab/habitat/tasks/rearrange/actions/habitat_mas_actions.py": {
        "module": "habitat.tasks.rearrange.actions.habitat_mas_actions",
        "classes": ["OracleNavDiffBaseAction"], "process": "worker"},
    "habitat-lab/habitat/core/env.py": {
        "module": "habitat.core.env", "classes": ["Env", "RLEnv"], "process": "worker"},
    "habitat-lab/habitat/core/environments.py": {
        "module": "habitat.core.environments", "classes": ["RLTaskEnv"], "process": "worker"},
    "habitat-lab/habitat/tasks/rearrange/rearrange_sim.py": {
        "module": "habitat.tasks.rearrange.rearrange_sim", "classes": ["RearrangeSim"], "process": "worker"},
    "habitat-lab/habitat/tasks/rearrange/rearrange_task.py": {
        "module": "habitat.tasks.rearrange.rearrange_task", "classes": ["RearrangeTask"], "process": "worker"},
}

MAX_SOURCE_BYTES = 128 * 1024
_MISSING = object()
_FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "input", "breakpoint",
    "globals", "locals", "getattr", "setattr", "delattr", "vars",
}
_FORBIDDEN_ROOTS = {
    "os", "sys", "subprocess", "socket", "requests", "urllib", "httpx", "http",
    "pathlib", "shutil", "pickle", "marshal", "importlib", "multiprocessing",
}
_FORBIDDEN_CALLS = {
    "open", "read", "write", "read_text", "write_text", "read_bytes", "write_bytes",
    "unlink", "mkdir", "makedirs", "rmdir", "remove", "rename", "replace", "chmod",
    "system", "popen", "Popen", "spawn", "fork", "connect", "recv", "send",
    "urlopen", "request", "get", "post", "put", "delete", "reset", "close",
}
_NETWORK_PARTS = {"client", "socket", "session", "connection", "completions"}

RESTRICTED_SOURCE_POLICY = {
    "profile": "emos-execution-source-v2",
    "automatic_application": "Only declared parser, control-state and execution edit regions are accepted in parent and navigation worker methods.",
    "editable": {
        "MultiLLMPolicy": ["Resume* (signature self, context)"],
        "OpenAIModel": ["_parse_tool_arguments", "_parse_response_message"],
        "WaitSkillPolicy": ["_is_skill_done (elapsed-step/requested-duration predicate only)"],
        "SkillPolicy": ["should_terminate (high-level interrupt combination only)"],
        "OracleNavPolicy": ["_is_skill_done (gate the actual completion sensor with control timing only)"],
        "OracleNavAction": ["step (completion predicate and at-goal completion flag only)"],
        "OracleNavCoordinateAction": ["step (completion predicate and at-goal completion flag only)"],
        "OracleNavDiffBaseAction": ["step (completion predicate, at-goal flag and stale-cache suppression only)"],
    },
    "control_writes": ["child._cur_call_high_level[0] = True while iterating self._active_policies", "self.source_resume_count = a nonnegative literal integer",
                       "Inside Resume, iterate child._skills.values(): skill._cur_skill_step[:] = 0 and skill._was_running_on_prev_step = False; preserve arguments, selected targets and all other state."],
    "parser_operations": "Return JSON derived from actual call.function.arguments, without overwriting parsed keys; appending up to four closing delimiters/whitespace characters is supported. Conditions, exceptions and one literal range retry loop (at most 32 iterations) are supported. Existing parser logging calls, order and control branches must remain unchanged.",
    "execution_operations": "Preserve the complete original method AST outside the declared edit regions. Wait completion must still depend on actual elapsed steps and the original requested duration. Navigation completion must retain real distance bounds using original or stricter configuration thresholds; never constant true. Sensor completion may only be gated, never synthesized. Preserve geometry, motion writes/calls, timeouts, postconditions and diagnostics.",
    "read_only": ["environment/task/simulator source", "navigation action code outside the declared completion regions", "physics and held objects", "observations and task goals",
                  "success/reward/metrics", "model identity, capabilities and conversation history"],
    "unsupported": "Arbitrary rewrites outside declared execution regions, changed motion generation or thresholds, new external calls, live-object aliases, computed calls and unbounded loops are rejected. This static profile is not an arbitrary-Python security sandbox or a proof of repair correctness.",
}
_RESTRICTED_PROFILE = RESTRICTED_SOURCE_POLICY["profile"]
_RESTRICTED_BUILTINS = {"len", "min", "max", "abs", "sum", "range", "enumerate", "zip",
                        "list", "dict", "tuple", "str", "int", "float", "bool", "round",
                        "sorted", "any", "all", "isinstance"}
_RESTRICTED_EXCEPTIONS = {"Exception", "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError"}
_RESTRICTED_DATA_METHODS = {"get", "items", "keys", "values", "count", "index", "copy",
                            "lower", "upper", "split", "startswith", "endswith", "strip", "replace"}
_RESTRICTED_MUTATORS = {"append", "extend", "update", "pop", "clear"}


def _restricted_target(relative, qualified, registry):
    """Readable original source is deliberately broader than editable source."""
    class_name, name = _target(relative, qualified, registry)
    specification = execution_target(relative, qualified)
    if specification is not None:
        if registry[relative]["process"] != specification["process"]:
            raise ValueError("Execution method process does not match its declared source target")
        return False
    if registry[relative]["process"] != "parent":
        raise ValueError("Restricted source profile keeps worker/environment/simulator source read-only")
    production = relative in SOURCE_TARGETS
    allowed = RESTRICTED_SOURCE_POLICY["editable"].get(class_name, [])
    # Hosts may explicitly supply small test/application classes. They receive
    # the same expression/write checks, never unrestricted object access.
    if not production:
        allowed = list(allowed) + ["Resume", "act", "parse", "parse_arguments", "_parse_arguments"]
    resume = class_name == "MultiLLMPolicy" and name.startswith("Resume")
    if (not resume and name not in allowed) or class_name in {"Env", "RLEnv", "RLTaskEnv", "RearrangeSim", "RearrangeTask"}:
        raise ValueError("Method is read-only in the restricted source profile: " + qualified)
    return not production


class _RestrictedMethod:
    """Fail-closed syntax/capability profile for trusted host data and classes.

    This is intentionally small: it does not infer the effects of arbitrary
    Python helpers or expose module globals as callable values. Only parser
    descriptors and the one declared control flag may reference live objects.
    """

    def __init__(self, method, original, fixture=False):
        self.names = {arg.arg: "data" for arg in method.args.args + method.args.posonlyargs + method.args.kwonlyargs
                      if arg.arg != "self"}
        self.fixture = fixture
        self.resume = method.name.startswith("Resume")
        self.parser = method.name if method.name in {"_parse_tool_arguments", "_parse_response_message"} else None
        self.inputs = set(self.names) if self.parser else set()
        for name in {"call", "response"} & self.names.keys():
            self.names[name] = "descriptor"
        self.original_calls = Counter(ast.dump(node, include_attributes=False)
            for node in ast.walk(original) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == "record_event") if original else Counter()
        self.loop_count = 0
        if self.logging_layout(method, bool(self.parser)) != self.logging_layout(original, bool(self.parser)):
            self.reject("existing parser logging must retain its calls, order and control branches")
        self.block(method.body)

    @staticmethod
    def logging_layout(method, include_returns=False):
        result = []

        def visit(node, control=()):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "record_event":
                result.append((ast.dump(node, include_attributes=False), control))
            if include_returns and isinstance(node, ast.Return):
                result.append(("return", control))
            for field, value in ast.iter_fields(node):
                branch = control
                if isinstance(node, (ast.If, ast.For, ast.Try, ast.ExceptHandler)):
                    branch += (type(node).__name__ + "." + field,)
                    if isinstance(node, ast.ExceptHandler):
                        branch += (ast.dump(node.type, include_attributes=False) if node.type else "bare", node.name)
                for child in value if isinstance(value, list) else [value]:
                    if isinstance(child, ast.AST):
                        visit(child, branch)

        if method is not None:
            visit(method)
        return result

    def reject(self, detail):
        raise ValueError("Restricted source profile: " + detail)

    def merge(self, branches):
        self.names = {name: values[0] if all(value == values[0] for value in values) else "data"
                      for name in set().union(*branches)
                      for values in [[branch[name] for branch in branches if name in branch]]}

    def branch(self, statements, initial):
        self.names = dict(initial)
        self.block(statements)
        return dict(self.names)

    def block(self, statements):
        for node in statements:
            if isinstance(node, (ast.Return, ast.Expr)):
                kind = "data"
                if node.value is not None:
                    kind = self.expr(node.value)
                if isinstance(node, ast.Return) and self.parser:
                    required = "parsed" if self.parser == "_parse_tool_arguments" else "message"
                    if kind != required:
                        self.reject("parser returns must come from the actual model response/arguments")
            elif isinstance(node, ast.Assign):
                kind = self.expr(node.value)
                # A chained assignment can otherwise share a mutable object
                # between a declared live slot and a local alias.
                if len(node.targets) != 1:
                    self.reject("chained assignments are unsupported")
                self.store(node.targets[0], kind, node.value)
            elif isinstance(node, ast.AugAssign):
                if not (isinstance(node.op, ast.Add) and isinstance(node.value, ast.Constant)
                        and type(node.value.value) is int and abs(node.value.value) <= 1000000):
                    self.reject("augmented assignments are limited to small counter increments")
                self.expr(node.value)
                self.store(node.target, "data", node.value, augmented=True)
            elif isinstance(node, ast.If):
                self.expr(node.test)
                initial = dict(self.names)
                self.merge([self.branch(node.body, initial), self.branch(node.orelse, initial)])
            elif isinstance(node, ast.For):
                self.loop_count += 1
                if self.loop_count > (2 if self.resume else 1) or not isinstance(node.target, ast.Name) or node.target.id in self.names:
                    self.reject("only bounded control iteration with fresh local bindings is supported")
                initial = dict(self.names)
                if _parts(node.iter) == ["self", "_active_policies"]:
                    self.bind(node.target.id, "child")
                elif (self.resume and isinstance(node.iter, ast.Call)
                      and isinstance(node.iter.func, ast.Attribute)
                      and not node.iter.args and not node.iter.keywords
                      and len(_parts(node.iter.func)) == 3
                      and _parts(node.iter.func)[1:] == ["_skills", "values"]
                      and self.names.get(_parts(node.iter.func)[0]) == "child"):
                    self.bind(node.target.id, "skill")
                elif (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                      and node.iter.func.id == "range" and not node.iter.keywords):
                    if not self.parser:
                        self.reject("literal retry loops are limited to parser repairs")
                    self.literal_range(node.iter)
                    self.bind(node.target.id, "data")
                else:
                    self.reject("only literal retry ranges and declared policy/skill iteration are supported")
                self.block(node.body)
                after = dict(self.names)
                after.pop(node.target.id, None)
                self.merge([initial, after])
                self.block(node.orelse)
            elif isinstance(node, ast.Try):
                initial = dict(self.names)
                body = self.branch(node.body, initial)
                self.merge([initial, body])
                interrupted = dict(self.names)
                branches = [self.branch(node.orelse, body)]
                for handler in node.handlers:
                    if handler.type is not None:
                        self.exception_type(handler.type)
                    self.names = dict(interrupted)
                    if handler.name:
                        self.bind(handler.name, "data")
                    self.block(handler.body)
                    if handler.name:
                        self.names.pop(handler.name, None)
                    branches.append(dict(self.names))
                self.merge(branches)
                self.block(node.finalbody)
            elif isinstance(node, ast.Raise):
                if node.exc is not None:
                    self.expr(node.exc)
                if node.cause is not None:
                    self.expr(node.cause)
            elif isinstance(node, ast.Assert):
                self.expr(node.test)
                if node.msg is not None:
                    self.expr(node.msg)
            elif not isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
                self.reject("unsupported recovery syntax " + type(node).__name__)

    def bind(self, name, kind):
        if (name == "self" or name in self.inputs or name.startswith("__") or name in _FORBIDDEN_NAMES | _FORBIDDEN_ROOTS
                or name in _RESTRICTED_BUILTINS | _RESTRICTED_EXCEPTIONS | {"json", "record_event", "type"}
                or self.names.get(name) in {"child", "skill"}):
            self.reject("binding would alias or replace a trusted capability")
        self.names[name] = kind

    def literal_range(self, node):
        try:
            values = [ast.literal_eval(value) for value in node.args]
            if node.keywords or not 1 <= len(values) <= 3 or any(type(value) is not int for value in values):
                raise ValueError()
            if len(range(*values)) > 32:
                raise ValueError()
        except (ValueError, TypeError, OverflowError):
            self.reject("retry ranges must have literal bounds and at most 32 iterations")

    def store(self, target, kind, value, augmented=False):
        if isinstance(target, ast.Name):
            self.bind(target.id, kind)
            return
        parts = _parts(target)
        if (self.resume and len(parts) == 2 and self.names.get(parts[0]) == "skill"
                and parts[1] == "_was_running_on_prev_step"
                and isinstance(value, ast.Constant) and value.value is False and not augmented):
            return
        if parts == ["self", "source_resume_count"]:
            if not (isinstance(value, ast.Constant) and type(value.value) is int and 0 <= value.value <= 1000000 and not augmented):
                self.reject("source resume counters must be small nonnegative literal integers")
            return
        fixture_fields = {"active", "resume_count", "source_resume_count", "steps", "mode"}
        if (self.fixture and len(parts) == 2 and parts[0] == "self" and parts[1] in fixture_fields):
            if parts[1] == "active" and not (isinstance(value, ast.Constant) and type(value.value) is bool and not augmented):
                self.reject("activation flags must be literal booleans")
            return
        if isinstance(target, ast.Subscript):
            flag = _parts(target.value)
            if (self.resume and len(flag) == 2 and self.names.get(flag[0]) == "skill"
                    and flag[1] == "_cur_skill_step" and isinstance(target.slice, ast.Slice)
                    and all(item is None for item in (target.slice.lower, target.slice.upper, target.slice.step))
                    and isinstance(value, ast.Constant) and type(value.value) is int
                    and value.value == 0 and not augmented):
                return
            if (len(flag) == 2 and flag[1] == "_cur_call_high_level"
                    and (flag[0] == "self" or self.names.get(flag[0]) == "child")
                    and isinstance(target.slice, ast.Constant) and type(target.slice.value) is int and target.slice.value == 0
                    and isinstance(value, ast.Constant) and value.value is True and not augmented):
                return
            if self.expr(target.value) == "owned":
                self.expr(target.slice)
                return
        self.reject("writes are limited to local data and declared resume control flags")

    def exception_type(self, node):
        if isinstance(node, ast.Name) and node.id in _RESTRICTED_EXCEPTIONS:
            return
        if _parts(node) == ["json", "JSONDecodeError"]:
            return
        if isinstance(node, ast.Tuple):
            for item in node.elts:
                self.exception_type(item)
            return
        self.reject("unknown exception type")

    def expr(self, node):
        if isinstance(node, ast.Constant):
            return "data"
        if isinstance(node, ast.Name):
            kind = self.names.get(node.id)
            if kind not in {"data", "owned", "arguments", "parsed", "message"}:
                self.reject("unknown, live-object or callable reference " + node.id)
            return kind
        if isinstance(node, ast.Attribute):
            parts = _parts(node)
            if (self.parser == "_parse_tool_arguments" and self.names.get("call") == "descriptor"
                    and parts in [["call", "function", "arguments"], ["call", "function", "name"], ["call", "id"]]):
                return "arguments" if parts[-1] == "arguments" else "data"
            if (len(parts) == 2 and parts[0] == "self" and parts[1] in
                    ({"agent_name", "_trace_call_id", "source_resume_count"} | ({"active", "resume_count", "steps", "mode", "history"} if self.fixture else set()))):
                return "data"
            if (node.attr == "__name__" and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "type"
                    and len(node.value.args) == 1 and not node.value.keywords
                    and isinstance(node.value.args[0], ast.Name) and self.names.get(node.value.args[0].id) == "data"):
                self.expr(node.value.args[0])
                return "data"
            if (self.parser == "_parse_response_message" and self.names.get("response") == "descriptor"
                    and node.attr == "message" and isinstance(node.value, ast.Subscript)
                    and _parts(node.value.value) == ["response", "choices"]
                    and isinstance(node.value.slice, ast.Constant) and node.value.slice.value == 0):
                return "message"
            self.reject("attribute access is outside declared parser/control data")
        if isinstance(node, ast.Subscript):
            self.expr(node.value)
            self.expr(node.slice)
            # Shallow copies/containers can still contain borrowed objects.
            return "data"
        if isinstance(node, ast.Slice):
            for value in (node.lower, node.upper, node.step):
                if value is not None:
                    self.expr(value)
            return "data"
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for item in node.elts:
                if self.expr(item) == "owned" and not isinstance(item, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    self.reject("shared local containers are outside the bounded data profile")
            return "owned"
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                self.reject("dictionary unpacking is unsupported")
            for item in node.keys + node.values:
                if self.expr(item) == "owned" and not isinstance(item, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
                    self.reject("shared local containers are outside the bounded data profile")
            return "owned"
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Pow, ast.LShift, ast.RShift, ast.Mult, ast.MatMult, ast.Mod)):
                self.reject("unbounded integer expansion is unsupported")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                left, right = self.expr(node.left), self.expr(node.right)
                literal = node.right
                if (left == "arguments" and isinstance(literal, ast.Constant)
                        and isinstance(literal.value, str) and len(literal.value) <= 4
                        and set(literal.value) <= set('}] \n\t')):
                    return "arguments"
                self.reject("dynamic concatenation is limited to closing argument delimiters")
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.expr(child)
            return "data"
        self.reject("unsupported expression " + type(node).__name__)

    def call(self, node):
        if not isinstance(node.func, (ast.Name, ast.Attribute)) or any(keyword.arg is None for keyword in node.keywords):
            self.reject("computed calls and argument unpacking are unsupported")
        parts = _parts(node.func)
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.names:
                self.reject("local/input bindings cannot be used as trusted callable names")
            if name == "record_event":
                key = ast.dump(node, include_attributes=False)
                if self.original_calls[key] <= 0:
                    self.reject("new or changed observation/logging producers are unsupported")
                self.original_calls[key] -= 1
            elif name not in _RESTRICTED_BUILTINS | _RESTRICTED_EXCEPTIONS:
                self.reject("only declared data calls are supported")
            if name == "range":
                self.literal_range(node)
            if name == "isinstance":
                if len(node.args) != 2 or not isinstance(node.args[1], ast.Name) or node.args[1].id not in {"dict", "list", "str", "int", "float", "bool", "tuple"}:
                    self.reject("isinstance requires a declared data type")
                self.expr(node.args[0])
                return "data"
        elif parts == ["json", "loads"]:
            if any(keyword.arg not in {"ensure_ascii", "allow_nan", "sort_keys"} for keyword in node.keywords):
                self.reject("JSON callbacks/options outside the data profile are unsupported")
            if self.parser == "_parse_tool_arguments" and parts[-1] == "loads":
                if len(node.args) != 1 or self.expr(node.args[0]) != "arguments":
                    self.reject("argument parsing must use the actual model arguments")
        elif (self.fixture and parts == ["self", "history", "append"]
              and len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
              and not node.keywords):
            return "data"
        elif node.func.attr in _RESTRICTED_DATA_METHODS | _RESTRICTED_MUTATORS:
            kind = self.expr(node.func.value)
            if node.func.attr in _RESTRICTED_MUTATORS:
                self.reject("container mutators are outside the bounded parser/control profile")
        else:
            self.reject("new live-object, filesystem, network or control calls are unsupported")
        for argument in node.args + [keyword.value for keyword in node.keywords]:
            self.expr(argument)
        if self.parser == "_parse_tool_arguments" and parts == ["json", "loads"]:
            return "parsed"
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "strip"
                and not node.args and not node.keywords and self.expr(node.func.value) == "arguments"):
            return "arguments"
        if parts == ["json", "loads"] or parts in [["dict"], ["list"], ["tuple"]] or (isinstance(node.func, ast.Attribute) and node.func.attr == "copy"):
            return "owned"
        return "data"


def _hash(content):
    return hashlib.sha256(content).hexdigest()


def _root(root):
    return Path(root).resolve()


def _path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Source path must be a canonical repository-relative path")
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or str(parsed) != relative or any(p in {".", ".."} for p in parsed.parts):
        raise ValueError("Source path must be a canonical repository-relative path")
    current = root
    for part in parsed.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Source paths cannot traverse symlinks")
    if not current.is_file() or current.suffix != ".py":
        raise ValueError("Source target must be an existing Python file")
    return current


def _read(root, relative):
    content = _path(root, relative).read_bytes()
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError("Source file exceeds the 128KB patch limit")
    try:
        return content, content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Source patches require UTF-8 Python files") from error


def _registry(targets):
    registry = deepcopy(SOURCE_TARGETS if targets is None else targets)
    for value in registry.values():
        if (not isinstance(value, dict) or not isinstance(value.get("module"), str)
                or not isinstance(value.get("classes"), (list, tuple))
                or value.get("process") not in {"parent", "worker"}):
            raise ValueError("Invalid source target registry")
    return registry


def source_snapshot(root=None, targets=None):
    """Return complete current source and exact byte hashes for allowed files."""
    root = _root(root if root is not None else Path(__file__).resolve().parents[3])
    result = {}
    for relative, target in _registry(targets).items():
        content, source = _read(root, relative)
        available = {node.name for node in _parse(source, relative).body if isinstance(node, ast.ClassDef)}
        metadata = deepcopy(target)
        metadata["classes"] = [name for name in target["classes"] if name in available]
        result[relative] = {**metadata, "sha256": _hash(content), "content": source}
    return result


def _parse(source, filename):
    try:
        tree = ast.parse(source, filename=filename)
        compile(tree, filename, "exec", dont_inherit=True)
        return tree
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError("Invalid Python source for " + filename + ": " + str(error)) from error


def _class(tree, name):
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original class " + name)
    return matches[0]


def _own_method(owner, name):
    matches = [node for node in owner.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
    if len(matches) > 1:
        raise ValueError("Cannot patch an overloaded or multiply defined method")
    return matches[0] if matches else None


def _target(relative, qualified, registry):
    if relative not in registry:
        raise ValueError("Source file is not allowlisted: " + str(relative))
    if not isinstance(qualified, str) or qualified.count(".") != 1:
        raise ValueError("Target must be Class.method")
    class_name, method_name = qualified.split(".")
    if not class_name.isidentifier() or not method_name.isidentifier():
        raise ValueError("Target must be Class.method")
    if class_name not in registry[relative]["classes"]:
        raise ValueError("Source class is not allowlisted: " + class_name)
    lower = method_name.lower()
    if (method_name.startswith("__") or any(word in lower for word in ("success", "metric", "reward"))
            or lower.startswith(("_recovery_", "_trajectory_", "_runtime_source_"))
            or lower in {"reset", "close", "seed", "reconfigure", "get_done", "get_info",
                         "evaluate_agent", "_evaluate_agent", "apply_runtime_source_patch",
                         "rollback_runtime_source_patch", "check_runtime_source_rollback",
                         "set_trajectory_recording", "get_trajectory_state",
                         "get_recovery_context", "validate_recovery_commands", "apply_recovery_commands"}):
        raise ValueError("Lifecycle, evaluation loops and success/metric methods cannot be patched")
    return class_name, method_name


def _signature(method):
    return (ast.dump(method.args, include_attributes=False),
            tuple(ast.dump(node, include_attributes=False) for node in method.decorator_list),
            ast.dump(method.returns, include_attributes=False) if method.returns else None,
            tuple(ast.dump(node, include_attributes=False) for node in getattr(method, "type_params", [])))


def _parts(node):
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        parts = _parts(node.value)
        return parts + [node.attr] if parts else []
    return []


def _hazards(method):
    hazards = []
    for node in ast.walk(method):
        dangerous = isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal))
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            dangerous = True
        if isinstance(node, ast.Call):
            parts = _parts(node.func)
            if parts:
                dangerous |= parts[0] in _FORBIDDEN_NAMES or parts[0] in _FORBIDDEN_ROOTS
                if len(parts) > 1:
                    # Ordinary data get/update operations are useful; get/post
                    # become network operations when performed through a client.
                    dangerous |= parts[-1] in (_FORBIDDEN_CALLS - {"get", "post", "put", "delete"})
                    dangerous |= bool(set(parts[:-1]) & _NETWORK_PARTS)
        if dangerous:
            hazards.append(ast.dump(node, include_attributes=False))
    return Counter(hazards)


def _method(code, method_name, original=None):
    if not isinstance(code, str) or not code.strip() or len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Expected complete method source of at most 128KB")
    tree = _parse(code, "<recovery-method>")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Patch code must define exactly one ordinary method")
    method = tree.body[0]
    if method.name != method_name:
        raise ValueError("Patch method name must match its target")
    if original is not None:
        if not isinstance(original, ast.FunctionDef) or _signature(method) != _signature(original):
            raise ValueError("Replacement must preserve the exact original method signature and decorators")
        decorators = [ast.unparse(node) for node in original.decorator_list]
        if any(value not in {"staticmethod", "classmethod"} for value in decorators):
            raise ValueError("Only ordinary, static and class methods can be replaced")
    else:
        if not method.args.args or method.args.args[0].arg != "self":
            raise ValueError("New recovery methods require self as the first argument")
        if method.decorator_list:
            raise ValueError("New recovery methods cannot introduce decorators")
        for value in method.args.defaults + [x for x in method.args.kw_defaults if x is not None]:
            try:
                ast.literal_eval(value)
            except (ValueError, TypeError) as error:
                raise ValueError("New method defaults must be inert literals") from error
        annotations = [a.annotation for a in method.args.args + method.args.posonlyargs + method.args.kwonlyargs]
        annotations += [method.returns]
        if method.args.vararg:
            annotations.append(method.args.vararg.annotation)
        if method.args.kwarg:
            annotations.append(method.args.kwarg.annotation)
        if any(value is not None and not (isinstance(value, ast.Constant) and isinstance(value.value, str))
               for value in annotations):
            raise ValueError("New method annotations must be inert strings")
    if len(list(ast.walk(method))) > 16000:
        raise ValueError("Method exceeds the recovery AST limit")
    if _hazards(method) - (_hazards(original) if original is not None else Counter()):
        raise ValueError("Generated code introduces an import, lifecycle, dynamic execution, filesystem or network capability")
    for node in ast.walk(method):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Yield, ast.YieldFrom, ast.Await)):
            raise ValueError("Recovery methods cannot add classes, async execution or suspended generators")
        if isinstance(node, ast.FunctionDef) and node is not method:
            original_nested = Counter(ast.dump(value, include_attributes=False)
                for value in ast.walk(original) if isinstance(value, ast.FunctionDef) and value is not original) if original else Counter()
            if ast.dump(node, include_attributes=False) not in original_nested:
                raise ValueError("Only unchanged original nested helper functions may be preserved")
    return method


def _entrypoint(entrypoint, files, registry):
    if not isinstance(entrypoint, dict) or set(entrypoint) != {"path", "target"}:
        raise ValueError("Entrypoint requires path and target")
    relative, qualified = entrypoint["path"], entrypoint["target"]
    class_name, method_name = _target(relative, qualified, registry)
    if registry[relative]["process"] != "parent":
        raise ValueError("Resume entrypoint must belong to a parent policy")
    if registry == SOURCE_TARGETS and class_name != "MultiLLMPolicy":
        raise ValueError("Resume entrypoint must belong to MultiLLMPolicy")
    file = next((item for item in files if item["path"] == relative), None)
    if file is None or not any(change["target"] == qualified for change in file["changes"]):
        raise ValueError("Entrypoint must be a method changed by this transaction")
    method = _own_method(_class(ast.parse(file["after_source"]), class_name), method_name)
    args = method.args
    if (method.decorator_list or [a.arg for a in args.args] != ["self", "context"]
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults):
        raise ValueError("Resume entrypoint signature must be (self, context)")


def _atomic_write(path, content, mode):
    descriptor, temporary = tempfile.mkstemp(prefix=".emos-source-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SourceTransaction:
    def __init__(self, root, files, entrypoint, patches, restricted=False):
        self.id = str(uuid4())
        self.root = root
        self.files = deepcopy(files)
        self.entrypoint = deepcopy(entrypoint)
        self.patches = deepcopy(patches)
        self.restricted = bool(restricted)
        self._committed = False

    def manifest(self, process=None):
        if process not in {None, "parent", "worker"}:
            raise ValueError("Unknown source process")
        result = {"transaction_id": self.id, "files": deepcopy([
            item for item in self.files if process is None or item["process"] == process])}
        if self.restricted:
            result["validation_profile"] = _RESTRICTED_PROFILE
        return result

    def _write(self, reverse=False):
        expected, replacement = (("after", "before") if reverse else ("before", "after"))
        # Validate every base before touching any member of the transaction.
        paths = []
        for file in self.files:
            path = _path(self.root, file["path"])
            if _hash(path.read_bytes()) != file[expected + "_sha256"]:
                raise ValueError("Source hash conflict: " + file["path"])
            paths.append((file, path, stat.S_IMODE(path.stat().st_mode)))
        written = []
        try:
            for file, path, mode in paths:
                _atomic_write(path, file[replacement + "_source"].encode("utf-8"), mode)
                written.append((file, path, mode))
        except BaseException:
            for file, path, mode in reversed(written):
                if _hash(path.read_bytes()) == file[replacement + "_sha256"]:
                    _atomic_write(path, file[expected + "_source"].encode("utf-8"), mode)
            raise

    def commit_files(self):
        if self._committed:
            raise ValueError("Source transaction is already committed")
        self._write()
        self._committed = True
        return {"status": "committed", "transaction_id": self.id}

    def rollback_files(self):
        if not self._committed:
            return {"status": "not_committed", "transaction_id": self.id}
        self._write(reverse=True)
        self._committed = False
        return {"status": "rolled_back", "transaction_id": self.id}

    def check_rollback_files(self):
        """Check all installed file hashes without changing any source bytes."""
        if not self._committed:
            return {"status": "not_committed", "transaction_id": self.id}
        for file in self.files:
            path = _path(self.root, file["path"])
            if _hash(path.read_bytes()) != file["after_sha256"]:
                raise ValueError("Source hash conflict: " + file["path"])
        return {"status": "checked", "transaction_id": self.id}

    def save(self, artifact_dir):
        directory = Path(artifact_dir) / self.id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "manifest.json").write_text(json.dumps({**self.manifest(),
            "entrypoint": self.entrypoint}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        differences = []
        for file in self.files:
            for kind in ("before", "after"):
                path = directory / kind / file["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(file[kind + "_source"].encode("utf-8"))
            differences.extend(difflib.unified_diff(file["before_source"].splitlines(True),
                file["after_source"].splitlines(True), fromfile="a/" + file["path"], tofile="b/" + file["path"]))
        (directory / "changes.diff").write_text("".join(differences), encoding="utf-8")
        return directory


class SourcePatchManager:
    def __init__(self, root, artifact_dir, targets=None, restricted=False):
        self.root = _root(root)
        self.artifact_dir = Path(artifact_dir)
        self.targets = _registry(targets)
        self.restricted = bool(restricted)
        self._current = None

    def snapshot(self):
        return source_snapshot(self.root, self.targets)

    def current_manifest(self, process=None):
        return self._current.manifest(process) if self._current else None

    def prepare(self, proposal):
        if not isinstance(proposal, dict) or not isinstance(proposal.get("explanation"), str):
            raise ValueError("Source proposal requires an explanation")
        patches = proposal.get("patches")
        if not isinstance(patches, list) or not patches or len(patches) > 32:
            raise ValueError("Expected between one and 32 source method patches")
        grouped, seen = {}, set()
        for patch in patches:
            if not isinstance(patch, dict) or set(patch) != {"path", "target", "base_sha256", "code"}:
                raise ValueError("Patch requires exactly path, target, base_sha256 and code")
            relative = patch["path"]
            _path(self.root, relative)
            class_name, method_name = _target(relative, patch["target"], self.targets)
            fixture = _restricted_target(relative, patch["target"], self.targets) if self.restricted else False
            identity = (relative, patch["target"])
            if identity in seen:
                raise ValueError("Duplicate source target")
            seen.add(identity)
            if relative not in grouped:
                content, source = _read(self.root, relative)
                grouped[relative] = {**deepcopy(self.targets[relative]), "path": relative,
                    "before_source": source, "before_sha256": _hash(content),
                    "changes": [], "tree": _parse(source, relative), "edits": []}
            file = grouped[relative]
            if patch["base_sha256"] != file["before_sha256"]:
                raise ValueError("Source base hash does not match current file: " + relative)
            owner = _class(file["tree"], class_name)
            original = _own_method(owner, method_name)
            if original is None and any(isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(isinstance(item, ast.Name) and item.id == method_name for item in ast.walk(node))
                    for node in owner.body):
                raise ValueError("New method would replace an existing nonmethod class attribute")
            method = _method(patch["code"], method_name, original)
            if self.restricted and not validate_execution_method(relative, patch["target"], method, original):
                _RestrictedMethod(method, original, fixture=fixture)
            if original is None:
                start = end = owner.end_lineno
                insertion = "\n" + "\n".join(" " * (owner.col_offset + 4) + line
                    if line else "" for line in patch["code"].strip().splitlines()) + "\n"
            else:
                start = min([original.lineno] + [node.lineno for node in original.decorator_list]) - 1
                end = original.end_lineno
                insertion = "\n".join(" " * original.col_offset + line if line else ""
                    for line in patch["code"].strip().splitlines()) + "\n"
            file["edits"].append((start, end, insertion))
            file["changes"].append({"target": patch["target"],
                "operation": "add" if original is None else "replace", "code": patch["code"]})
        files = []
        for file in grouped.values():
            lines = file["before_source"].splitlines(True)
            ordered = sorted(enumerate(file.pop("edits")), key=lambda item: (item[1][0], item[0]), reverse=True)
            for _, (start, end, insertion) in ordered:
                if start == end and start and not lines[start - 1].endswith(("\n", "\r")):
                    insertion = "\n" + insertion
                lines[start:end] = [insertion]
            source = "".join(lines)
            _parse(source, file["path"])
            file.update(after_source=source, after_sha256=_hash(source.encode("utf-8")))
            file.pop("tree")
            files.append(file)
        _entrypoint(proposal.get("entrypoint"), files, self.targets)
        transaction = SourceTransaction(self.root, files, proposal["entrypoint"], patches, restricted=self.restricted)
        transaction.save(self.artifact_dir)
        self._current = transaction
        return transaction


def _cell(value):
    return (lambda: value).__closure__[0]


def _compiled_method(method, owner, module, previous, filename):
    candidate = deepcopy(method)
    candidate.decorator_list = []
    candidate.returns = None
    candidate.args.defaults = []
    candidate.args.kw_defaults = [None] * len(candidate.args.kwonlyargs)
    for argument in candidate.args.posonlyargs + candidate.args.args + candidate.args.kwonlyargs:
        argument.annotation = None
    if candidate.args.vararg:
        candidate.args.vararg.annotation = None
    if candidate.args.kwarg:
        candidate.args.kwarg.annotation = None
    if hasattr(candidate, "type_params"):
        candidate.type_params = []
    # Compile a class scope so Python produces the correct __class__ freevar
    # for super(); replace its cell with the existing owner before installation.
    wrapper = ast.ClassDef(name=owner.__name__, bases=[], keywords=[],
        body=[candidate], decorator_list=[])
    tree = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    code = compile(tree, filename, "exec", dont_inherit=True)
    local = {}
    exec(code, module.__dict__, local)
    raw = local[owner.__name__].__dict__[method.name]
    closure = tuple(_cell(owner) if name == "__class__" else cell
        for name, cell in zip(raw.__code__.co_freevars, raw.__closure__ or ()))
    function = types.FunctionType(raw.__code__, module.__dict__, method.name,
        closure=closure or None)
    function.__module__ = module.__name__
    function.__qualname__ = owner.__qualname__ + "." + method.name
    original = previous.__func__ if isinstance(previous, (staticmethod, classmethod)) else previous
    if original is not _MISSING:
        if not isinstance(original, types.FunctionType):
            raise ValueError("Live target is not an ordinary Python method")
        function.__defaults__ = original.__defaults__
        function.__kwdefaults__ = dict(original.__kwdefaults__) if original.__kwdefaults__ is not None else None
        function.__annotations__ = dict(original.__annotations__)
    else:
        function.__defaults__ = tuple(ast.literal_eval(value) for value in method.args.defaults) or None
        function.__kwdefaults__ = {arg.arg: ast.literal_eval(value)
            for arg, value in zip(method.args.kwonlyargs, method.args.kw_defaults) if value is not None} or None
        function.__annotations__ = {arg.arg: arg.annotation.value
            for arg in method.args.posonlyargs + method.args.args + method.args.kwonlyargs
            if arg.annotation is not None}
        if method.returns is not None:
            function.__annotations__["return"] = method.returns.value
    if isinstance(previous, staticmethod):
        return staticmethod(function)
    if isinstance(previous, classmethod):
        return classmethod(function)
    return function


class RuntimeSourcePatcher:
    """Install methods only in modules already loaded by this process."""
    def __init__(self, root, targets=None, restricted=False):
        self.root = _root(root)
        self.targets = _registry(targets)
        self.restricted = bool(restricted)
        self._stack = []

    def install(self, manifest):
        if (not isinstance(manifest, dict)
                or set(manifest) not in ({"transaction_id", "files"}, {"transaction_id", "files", "validation_profile"})):
            raise ValueError("Invalid source transaction manifest")
        profile = manifest.get("validation_profile")
        if "validation_profile" in manifest and profile != _RESTRICTED_PROFILE:
            raise ValueError("Unknown or downgraded source validation profile")
        restricted = self.restricted or profile == _RESTRICTED_PROFILE
        if restricted and profile != _RESTRICTED_PROFILE:
            raise ValueError("Restricted source validation profile is required by this host")
        identity = manifest["transaction_id"]
        if not isinstance(identity, str) or not identity or any(item["id"] == identity for item in self._stack):
            raise ValueError("Source transaction ID is missing or already installed")
        if not isinstance(manifest["files"], list):
            raise ValueError("Source manifest files must be a list")
        changes, pending, seen = [], [], set()
        for file in manifest["files"]:
            relative = file["path"]
            path = _path(self.root, relative)
            if relative not in self.targets or file["module"] != self.targets[relative]["module"]:
                raise ValueError("Manifest module is not allowlisted")
            if restricted and file.get("process") != self.targets[relative]["process"]:
                raise ValueError("Restricted source manifest process does not match its declared source target")
            for kind in ("before", "after"):
                if _hash(file[kind + "_source"].encode("utf-8")) != file[kind + "_sha256"]:
                    raise ValueError("Manifest source hash is invalid")
            current_hash = _hash(path.read_bytes())
            if restricted and current_hash != file["before_sha256"]:
                raise ValueError("Restricted source install requires the current original source before file commit")
            if current_hash not in {file["before_sha256"], file["after_sha256"]}:
                raise ValueError("Runtime source hash conflict: " + relative)
            before = _parse(file["before_source"], relative)
            after = _parse(file["after_source"], relative)
            expected = deepcopy(before)
            module = sys.modules.get(file["module"])
            if module is None:
                pending.append(file["module"])
            elif Path(getattr(module, "__file__", "")).resolve() != path:
                raise ValueError("Loaded module does not match its source target: " + relative)
            for change in file["changes"]:
                class_name, method_name = _target(relative, change["target"], self.targets)
                fixture = _restricted_target(relative, change["target"], self.targets) if restricted else False
                key = (relative, change["target"])
                if key in seen:
                    raise ValueError("Duplicate runtime source target")
                seen.add(key)
                original_ast = _own_method(_class(before, class_name), method_name)
                method = _method(change["code"], method_name, original_ast)
                if restricted and not validate_execution_method(relative, change["target"], method, original_ast):
                    _RestrictedMethod(method, original_ast, fixture=fixture)
                installed_ast = _own_method(_class(after, class_name), method_name)
                if installed_ast is None or ast.dump(installed_ast, include_attributes=False) != ast.dump(method, include_attributes=False):
                    raise ValueError("Manifest method does not match complete edited source")
                operation = "add" if original_ast is None else "replace"
                if change["operation"] != operation:
                    raise ValueError("Manifest operation disagrees with original source")
                expected_owner = _class(expected, class_name)
                expected_original = _own_method(expected_owner, method_name)
                if expected_original is None:
                    expected_owner.body.append(deepcopy(method))
                else:
                    expected_owner.body[expected_owner.body.index(expected_original)] = deepcopy(method)
                if module is None:
                    continue
                owner = vars(module).get(class_name)
                if not isinstance(owner, type):
                    raise ValueError("Live original class is unavailable: " + class_name)
                previous = vars(owner).get(method_name, _MISSING)
                if (operation == "add") != (previous is _MISSING):
                    raise ValueError("Live class own method disagrees with original source")
                function = _compiled_method(method, owner, module, previous, str(path))
                changes.append((owner, method_name, previous, function))
            if ast.dump(expected, include_attributes=False) != ast.dump(after, include_attributes=False):
                raise ValueError("Manifest contains edits outside its declared structured methods")
        applied = []
        try:
            for owner, name, previous, function in changes:
                setattr(owner, name, function)
                applied.append((owner, name, previous, function))
        except BaseException:
            for owner, name, previous, function in reversed(applied):
                if previous is _MISSING:
                    delattr(owner, name)
                else:
                    setattr(owner, name, previous)
            raise
        self._stack.append({"id": identity, "changes": applied})
        self.restricted = restricted
        return {"status": "installed", "transaction_id": identity,
            "pending_modules": sorted(set(pending)), "installed_methods": len(applied)}

    def check_rollback(self, transaction_id):
        """Validate rollback order and live descriptors without changing them."""
        if not self._stack or self._stack[-1]["id"] != transaction_id:
            raise ValueError("Source rollback must follow strict LIFO installation order")
        transaction = self._stack[-1]
        for owner, name, previous, installed in transaction["changes"]:
            if vars(owner).get(name, _MISSING) is not installed:
                raise ValueError("Live method changed after source install; refusing rollback conflict")
        return {"status": "checked", "transaction_id": transaction_id}

    def rollback(self, transaction_id):
        self.check_rollback(transaction_id)
        transaction = self._stack[-1]
        for owner, name, previous, installed in reversed(transaction["changes"]):
            if previous is _MISSING:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)
        self._stack.pop()
        return {"status": "rolled_back", "transaction_id": transaction_id}
