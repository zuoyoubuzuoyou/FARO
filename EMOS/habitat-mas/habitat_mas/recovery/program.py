"""Restricted, data-only Python policies executed outside the simulator process.

This deliberately supports a small Python subset; it is not a general-purpose
sandbox for arbitrary Python, imports, third-party packages or simulator objects.
"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile


class ProgramError(ValueError):
    pass


BUILTINS = ("len", "min", "max", "abs", "sum", "range", "enumerate", "zip", "list",
            "dict", "str", "int", "float", "bool", "round", "sorted", "any", "all")
METHODS = {"get", "items", "keys", "values", "append", "count", "index", "copy", "update",
           "pop", "lower", "upper", "split", "startswith", "endswith", "strip"}
_NODES = {
    "Module", "FunctionDef", "arguments", "arg", "Return", "Assign", "AugAssign", "Expr",
    "If", "For", "Break", "Continue", "Pass", "Name", "Load", "Store", "Constant", "List",
    "Tuple", "Dict", "Subscript", "Slice", "Attribute", "Call", "keyword", "BinOp", "UnaryOp",
    "BoolOp", "Compare", "IfExp", "ListComp", "DictComp", "comprehension",
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "USub", "UAdd", "Not", "And", "Or",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
}


def validate_code(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 32000:
        raise ProgramError("Expected nonempty recovery code of at most 32000 characters")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ProgramError("Recovery code is not valid Python: " + str(error)) from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ProgramError("Define exactly one recover(context, memory) function")
    function = tree.body[0]
    args = function.args
    if (function.name != "recover" or function.decorator_list or function.returns
            or [a.arg for a in args.args] != ["context", "memory"]
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults
            or any(a.annotation for a in args.args)):
        raise ProgramError("Expected undecorated recover(context, memory) without annotations/defaults")
    nodes = list(ast.walk(tree))
    if len(nodes) > 4000:
        raise ProgramError("Recovery AST exceeds 4000 nodes")
    for node in nodes:
        if type(node).__name__ not in _NODES:
            raise ProgramError("Unsupported recovery syntax: " + type(node).__name__)
        if isinstance(node, ast.FunctionDef) and node is not function:
            raise ProgramError("Nested functions are unsupported")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id == "recover"):
            raise ProgramError("Private names and recursive recovery calls are unsupported")
        if isinstance(node, ast.Attribute) and node.attr not in METHODS:
            raise ProgramError("Unsupported data method: " + node.attr)
        if isinstance(node, ast.Call):
            if not ((isinstance(node.func, ast.Name) and node.func.id in BUILTINS)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr in METHODS)):
                raise ProgramError("Only the documented data builtins/methods may be called")
        if isinstance(node, ast.comprehension) and node.is_async:
            raise ProgramError("Async comprehensions are unsupported")
    return tree


def validate_result(value):
    if not isinstance(value, dict) or set(value) != {"commands", "memory"}:
        raise ProgramError("Return exactly commands and memory")
    if not isinstance(value["memory"], dict) or not isinstance(value["commands"], list):
        raise ProgramError("commands must be a list and memory must be an object")
    if len(value["commands"]) > 16:
        raise ProgramError("At most 16 recovery commands per observation")
    agents = set()
    for command in value["commands"]:
        if not isinstance(command, dict) or command.get("type") not in {"resume", "replan", "action"}:
            raise ProgramError("Unknown recovery command")
        agent = command.get("agent")
        if not isinstance(agent, str) or not agent or agent in agents:
            raise ProgramError("Each command must target a different actual agent")
        agents.add(agent)
        allowed = {"type", "agent", "reason", "instruction"} if command["type"] == "resume" else {"type", "agent", "instruction"}
        if command["type"] == "action":
            allowed = {"type", "agent", "name", "arguments"}
            if not isinstance(command.get("name"), str) or not isinstance(command.get("arguments"), dict):
                raise ProgramError("Action requires a skill name and argument object")
        if set(command) - allowed:
            raise ProgramError("Unknown command fields")
        if command["type"] == "replan" and (not isinstance(command.get("instruction"), str) or not command["instruction"].strip()):
            raise ProgramError("Replan requires an instruction")
        for key in ("reason", "instruction"):
            if key in command and (not isinstance(command[key], str) or len(command[key]) > 8000):
                raise ProgramError(key + " must be text of at most 8000 characters")
    return value


class RecoveryProgram:
    def __init__(self, code):
        validate_code(code)
        self.code = code

    def run(self, context, memory, timeout=1.0):
        request = json.dumps({"code": self.code, "context": context, "memory": memory,
                              "cpu_seconds": max(1, int(timeout) + 1)}, allow_nan=False)
        if len(request.encode("utf-8")) > 2_000_000:
            raise ProgramError("Recovery input exceeds 2MB")
        worker = Path(__file__).with_name("worker.py")
        try:
            with tempfile.TemporaryDirectory(prefix="emos-recovery-") as directory:
                result = subprocess.run(
                    [sys.executable, "-I", "-S", str(worker)], input=request, text=True,
                    capture_output=True, timeout=timeout, cwd=directory, env={}, close_fds=True,
                )
        except subprocess.TimeoutExpired as error:
            raise ProgramError("Recovery code exceeded its wall clock budget") from error
        if result.returncode:
            raise ProgramError("Recovery worker failed: " + result.stderr.strip()[:1000])
        try:
            value = json.loads(result.stdout)
        except (ValueError, RecursionError) as error:
            raise ProgramError("Recovery worker returned invalid JSON") from error
        return validate_result(value)
