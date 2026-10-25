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
        return _parts(node.value) + [node.attr]
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
    def __init__(self, root, files, entrypoint, patches):
        self.id = str(uuid4())
        self.root = root
        self.files = deepcopy(files)
        self.entrypoint = deepcopy(entrypoint)
        self.patches = deepcopy(patches)
        self._committed = False

    def manifest(self, process=None):
        if process not in {None, "parent", "worker"}:
            raise ValueError("Unknown source process")
        return {"transaction_id": self.id, "files": deepcopy([
            item for item in self.files if process is None or item["process"] == process])}

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
    def __init__(self, root, artifact_dir, targets=None):
        self.root = _root(root)
        self.artifact_dir = Path(artifact_dir)
        self.targets = _registry(targets)
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
            _method(patch["code"], method_name, original)
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
        transaction = SourceTransaction(self.root, files, proposal["entrypoint"], patches)
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
    def __init__(self, root, targets=None):
        self.root = _root(root)
        self.targets = _registry(targets)
        self._stack = []

    def install(self, manifest):
        if not isinstance(manifest, dict) or set(manifest) != {"transaction_id", "files"}:
            raise ValueError("Invalid source transaction manifest")
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
            for kind in ("before", "after"):
                if _hash(file[kind + "_source"].encode("utf-8")) != file[kind + "_sha256"]:
                    raise ValueError("Manifest source hash is invalid")
            if _hash(path.read_bytes()) not in {file["before_sha256"], file["after_sha256"]}:
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
                key = (relative, change["target"])
                if key in seen:
                    raise ValueError("Duplicate runtime source target")
                seen.add(key)
                original_ast = _own_method(_class(before, class_name), method_name)
                method = _method(change["code"], method_name, original_ast)
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
