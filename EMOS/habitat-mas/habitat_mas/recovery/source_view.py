"""AST-selected, complete method bodies with original full-file patch hashes.

These are diagnostic views, not compilable replacement files. Selection uses
declared edit permissions, structural runtime evidence and explicit adapter
rules; source contents never execute and no filesystem or runtime imports occur.
"""
import ast
from copy import deepcopy
from fnmatch import fnmatchcase
import hashlib
import io
import re
import tokenize


# Domain adapter declarations. The extractor below is independent of these
# particular class names; hosts can override individual rule groups in context.
DEFAULT_SOURCE_VIEW_RULES = {
    "entry_methods": {
        "MultiLLMPolicy": ["act", "get_recovery_context"],
        "HierarchicalPolicy": ["act", "_update_skills", "_get_terminations"],
        "LLMHighLevelPolicy": ["get_next_skill"],
        "OpenAIModel": ["_chat", "_parse*"],
        "CrabAgent": ["chat"],
    },
    "role_methods": {
        "policy": ["act", "get_recovery_context"],
        "control": ["act", "_update_skills", "_get_terminations"],
        "high_level": ["get_next_skill"], "agent": ["chat"],
        "model": ["_chat", "_parse*"],
    },
    "diagnostic_methods": ["step", "act", "should_terminate", "_is_skill_done",
                           "_internal_act", "_parse_skill_arg", "on_enter"],
    "diagnostic_aliases": {
        "wait": ["WaitSkillPolicy"],
        "nav_to_obj": ["OracleNavPolicy"],
        "nav_to_goal": ["OracleNavPolicy"],
        "nav_to_point": ["OracleNavCoordPolicy"],
        "pick": ["OraclePickPolicy"], "place": ["OraclePlacePolicy"],
    },
}

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_ASSIGNMENTS = (ast.Assign, ast.AnnAssign, ast.AugAssign)


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _start(node):
    return min([node.lineno] + [item.lineno for item in getattr(node, "decorator_list", [])])


class _Source:
    def __init__(self, path, metadata):
        self.path, self.metadata = path, metadata
        self.text = metadata.get("content")
        if not isinstance(self.text, str) or metadata.get("truncated") or metadata.get("view") == "methods-v1":
            raise ValueError("Complete source is required for " + path)
        try:
            self.tree = ast.parse(self.text, filename=path)
            compile(self.tree, path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError, RecursionError) as error:
            raise ValueError("Invalid Python source for " + path + ": " + str(error)) from error
        self.lines = self.text.splitlines(keepends=True)
        self.parents = {child: parent for parent in ast.walk(self.tree) for child in ast.iter_child_nodes(parent)}
        self.methods, self.classes = {}, {}
        self.imports = [node for node in self.tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.aliases = {}
        for node in self.imports:
            for name in node.names:
                self.aliases[name.asname or name.name.split(".")[0]] = (
                    ((node.module or "") + "." if isinstance(node, ast.ImportFrom) else "") + name.name)
        self._index(self.tree.body)

    def _index(self, body, prefix=""):
        for node in body:
            target = prefix + node.name if isinstance(node, (ast.ClassDef,) + _FUNCTIONS) else None
            if isinstance(node, ast.ClassDef):
                self.classes[target] = node
                self._index(node.body, target + ".")
            elif isinstance(node, _FUNCTIONS):
                descriptor = next((item.attr for item in node.decorator_list
                    if isinstance(item, ast.Attribute) and item.attr in {"setter", "deleter"}
                    and isinstance(item.value, ast.Name) and item.value.id == node.name), None)
                if descriptor:
                    target += "@" + descriptor
                if target in self.methods:
                    # Repeated definitions (e.g. property setters) must never
                    # silently choose one incompatible replacement body.
                    raise ValueError("Ambiguous method " + target + " in " + self.path)
                self.methods[target] = node

    def code(self, node):
        return "".join(self.lines[_start(node) - 1:node.end_lineno])

    def header(self, node):
        """Preserve the original multiline signature, annotations and spacing."""
        text = "".join(self.lines[node.lineno - 1:node.end_lineno])
        depth = 0
        # Tokenizing an indented declaration produces INDENT but retains its
        # original columns, unlike ast.unparse (which rewrites signatures).
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.OP:
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth -= 1
                elif token.string == ":" and depth == 0:
                    row, column = token.end
                    lines = text.splitlines(keepends=True)
                    return "".join(lines[:row - 1]) + lines[row - 1][:column]
        raise ValueError("Missing declaration boundary in " + self.path)


class _Index:
    def __init__(self, sources):
        self.files = {path: _Source(path, info) for path, info in sources.items()}
        self.classes = [(file, name) for file in self.files.values() for name in file.classes]
        self.selected = {}
        self.dispatch_classes = {}

    def find_classes(self, name, module=None):
        result = []
        for file, target in self.classes:
            qualified = file.metadata.get("module", "") + "." + target
            if name in {target, target.rsplit(".", 1)[-1], qualified} and (
                    module is None or file.metadata.get("module") == module):
                result.append((file, target))
        return result

    def bases(self, file, owner):
        for base in file.classes[owner].bases:
            name = _dotted(base)
            if not name:
                continue
            root, *rest = name.split(".")
            resolved = file.aliases.get(root, root) + ("." + ".".join(rest) if rest else "")
            matches = self.find_classes(resolved)
            if not matches:
                matches = [(f, c) for f, c in self.find_classes(name) if f is file]
            if not matches:
                matches = self.find_classes(name)
            yield from matches

    def hierarchy(self, file, owner):
        """C3 order over the classes present in the supplied source snapshot."""
        def linearize(candidate, cls, active):
            key = (candidate.path, cls)
            if key in active:
                raise ValueError("Cyclic source inheritance in " + candidate.path)
            parents = list(self.bases(candidate, cls))
            sequences = [linearize(parent, name, active | {key}) for parent, name in parents]
            sequences.append([(parent.path, name) for parent, name in parents])
            result = [key]
            while any(sequences):
                sequences = [sequence for sequence in sequences if sequence]
                head = next((sequence[0] for sequence in sequences
                             if not any(sequence[0] in other[1:] for other in sequences)), None)
                if head is None:
                    raise ValueError("Inconsistent source inheritance in " + candidate.path)
                result.append(head)
                for sequence in sequences:
                    if sequence[0] == head:
                        sequence.pop(0)
            return result
        for path, cls in linearize(file, owner, set()):
            yield self.files[path], cls

    def select(self, file, target, reason):
        if target in file.methods:
            self.selected.setdefault((file.path, target), set()).add(reason)

    def select_class(self, file, owner, patterns, reason):
        found = set()
        for candidate, cls in self.hierarchy(file, owner):
            for target in candidate.methods:
                name = target.rsplit(".", 1)[-1]
                if target.rsplit(".", 1)[0] == cls and name not in found:
                    found.add(name)
                    if any(fnmatchcase(name, pattern) for pattern in patterns):
                        self.select(candidate, target, reason)
                        self.dispatch_classes.setdefault((candidate.path, target), set()).add((file.path, owner))

    def method(self, file, owner, name):
        for candidate, cls in self.hierarchy(file, owner):
            target = cls + "." + name
            if target in candidate.methods:
                return candidate, target
        return None


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root = _dotted(node.value)
        return root + "." + node.attr if root else None
    return None


def _pattern(specification):
    return specification.split(" ", 1)[0].split("(", 1)[0]


def _walk_data(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_data(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_data(item)


def _select_runtime(index, context, rules):
    runtime = context.get("runtime") or {}
    for binding in runtime.get("class_bindings") or []:
        role = re.sub(r"^agent_\d+\.", "", binding.get("role", ""))
        for file, owner in index.find_classes(binding.get("class"), binding.get("module")):
            index.select_class(file, owner, rules["role_methods"].get(role, []), "runtime_binding:" + role)
    frames = [runtime.get("current_frame")] + list(runtime.get("recent_frames") or [])
    diagnostic_names = set()
    for frame in frames:
        for item in _walk_data(frame):
            # Only structural diagnostic identifiers participate. Explanations,
            # task prose and model output cannot steer source selection.
            for field in ("action", "termination", "navigation", "detection", "diagnostic", "diagnostics"):
                value = item.get(field)
                if isinstance(value, str):
                    diagnostic_names.add(value)
                for detail in _walk_data(value):
                    for key in ("source", "class", "skill", "skill_name", "action_name", "name"):
                        name = detail.get(key)
                        if isinstance(name, str):
                            diagnostic_names.add(name)
    for name in diagnostic_names:
        names = [name] + list(rules["diagnostic_aliases"].get(name, []))
        for candidate in names:
            for file, owner in index.find_classes(candidate):
                index.select_class(file, owner, rules["diagnostic_methods"], "runtime_diagnostic:" + name)
            for file in index.files.values():
                for target in file.methods:
                    if candidate in {target, file.metadata.get("module", "") + "." + target}:
                        index.select(file, target, "runtime_diagnostic:" + name)
    for item in _walk_data(context.get("error_report")):
        path = item.get("path") or item.get("filename")
        line = item.get("line") or item.get("lineno")
        function = item.get("function") or item.get("name")
        if not isinstance(path, str) or not function:
            continue
        for file in index.files.values():
            if path != file.path and not path.endswith("/" + file.path):
                continue
            for target, node in file.methods.items():
                if function in {target, node.name} and (not isinstance(line, int) or _start(node) <= line <= node.end_lineno):
                    index.select(file, target, "execution_traceback")


def _select_helpers(index):
    # Snapshot the seed set: helpers of helpers are intentionally not expanded.
    for path, target in list(index.selected):
        file = index.files[path]
        owner = target.rsplit(".", 1)[0] if "." in target else None
        for node in ast.walk(file.methods[target]):
            if not isinstance(node, ast.Call):
                continue
            function = _dotted(node.func)
            if function and function.startswith(("self.", "cls.")) and function.count(".") == 1 and owner:
                dispatch = index.dispatch_classes.get((path, target), {(path, owner)})
                for dispatch_path, dispatch_owner in dispatch:
                    found = index.method(index.files[dispatch_path], dispatch_owner, function.split(".")[1])
                    if found:
                        index.select(*found, "one_hop_helper:" + target)
                        index.dispatch_classes.setdefault((found[0].path, found[1]), set()).add((dispatch_path, dispatch_owner))
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Name) and node.func.value.func.id == "super" and owner:
                dispatch = index.dispatch_classes.get((path, target), {(path, owner)})
                for dispatch_path, dispatch_owner in dispatch:
                    order = list(index.hierarchy(index.files[dispatch_path], dispatch_owner))
                    position = next(i for i, (parent, cls) in enumerate(order) if parent is file and cls == owner)
                    for parent, cls in order[position + 1:]:
                        helper = cls + "." + node.func.attr
                        if helper in parent.methods and node.func.attr != "__init__":
                            index.select(parent, helper, "one_hop_super:" + target)
                            index.dispatch_classes.setdefault((parent.path, helper), set()).add((dispatch_path, dispatch_owner))
                            break
            elif function:
                root, *parts = function.split(".")
                resolved = file.aliases.get(root, root) + ("." + ".".join(parts) if parts else "")
                for candidate in index.files.values():
                    for helper in candidate.methods:
                        qualified = candidate.metadata.get("module", "") + "." + helper
                        if (candidate is file and helper == function and "." not in helper) or qualified == resolved:
                            index.select(candidate, helper, "one_hop_helper:" + target)


def _names(node):
    return {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}


def _assigned(node):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return {item.id for target in targets for item in ast.walk(target)
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}


def _scope_nodes(root):
    """Traverse statement blocks without crossing nested function/class scopes."""
    yield root
    for child in ast.iter_child_nodes(root):
        if not isinstance(child, (ast.ClassDef,) + _FUNCTIONS):
            yield from _scope_nodes(child)


def _member_reads(node):
    return {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
            and isinstance(item.ctx, ast.Load) and isinstance(item.value, ast.Name)
            and item.value.id in {"self", "cls"}}


def _support(index):
    support = {path: {} for path in index.files}

    def add(file, node, kind, target=None, code=None):
        key = (_start(node), node.end_lineno, kind)
        end = node.end_lineno if code is None else _start(node) + len(code.splitlines()) - 1
        support[file.path][key] = {"kind": kind, "target": target,
            "start_line": _start(node), "end_line": end,
            "code": file.code(node) if code is None else code}

    required_names = {path: set() for path in index.files}

    def control_context(file, node, emit=False):
        names = set()
        parent = file.parents.get(node)
        while parent is not None and not isinstance(parent, (ast.Module, ast.ClassDef) + _FUNCTIONS):
            if isinstance(parent, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try, ast.ExceptHandler)):
                if emit:
                    add(file, parent, "initialization_condition", code=file.header(parent) + "\n")
                    for number in range(parent.lineno, min(node.lineno, parent.end_lineno)):
                        line = file.lines[number - 1]
                        if len(line) - len(line.lstrip()) == parent.col_offset and re.match(r"(?:else|elif|except|finally)\b", line.lstrip()):
                            support[file.path][(number, number, "branch")] = {"kind": "branch", "target": None,
                                "start_line": number, "end_line": number, "code": line}
                for field, value in ast.iter_fields(parent):
                    if field not in {"body", "orelse", "finalbody", "handlers"}:
                        for item in value if isinstance(value, list) else [value]:
                            if isinstance(item, ast.AST):
                                names.update(_names(item))
            parent = file.parents.get(parent)
        return names

    referenced_members = {}
    for path, target in index.selected:
        file, node = index.files[path], index.files[path].methods[target]
        required_names[path].update(_names(node))
        if "." not in target:
            continue
        owner = target.rsplit(".", 1)[0]
        attrs = {item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
                 and isinstance(item.value, ast.Name) and item.value.id in {"self", "cls"}}
        dispatch = index.dispatch_classes.get((path, target), {(path, owner)})
        hierarchy = {(parent.path, cls) for dispatch_path, dispatch_owner in dispatch
                     for parent, cls in index.hierarchy(index.files[dispatch_path], dispatch_owner)}
        for parent_path, cls in hierarchy:
            parent = index.files[parent_path]
            class_node = parent.classes[cls]
            header = parent.header(class_node)
            decorators = "".join(parent.lines[_start(class_node) - 1:class_node.lineno - 1])
            add(parent, class_node, "class_declaration", cls, decorators + header + "\n")
            for base in class_node.bases + class_node.decorator_list + [keyword.value for keyword in class_node.keywords]:
                required_names[parent.path].update(_names(base))
            referenced_members.setdefault((parent.path, cls), set()).update(attrs)
    for (path, owner), attrs in referenced_members.items():
        file = index.files[path]
        constructor = file.methods.get(owner + ".__init__")
        candidates = list(_scope_nodes(constructor)) if constructor else []
        for statement in file.classes[owner].body:
            if not isinstance(statement, (ast.ClassDef,) + _FUNCTIONS):
                candidates.extend(_scope_nodes(statement))
        assignments = [node for node in candidates if isinstance(node, _ASSIGNMENTS)]
        needed_locals = set()
        chosen = set()
        while True:
            previous = len(chosen)
            for node in assignments:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                members = {item.attr for value in targets for item in ast.walk(value)
                           if isinstance(item, ast.Attribute) and isinstance(item.value, ast.Name)
                           and item.value.id in {"self", "cls"}}
                local_names = _assigned(node)
                class_assignment = node in file.classes[owner].body and bool(local_names & attrs)
                if members & attrs or local_names & needed_locals or class_assignment:
                    chosen.add(node)
                    needed_locals.update(_names(node))
                    needed_locals.update(control_context(file, node))
                    attrs.update(_member_reads(node))
            if len(chosen) == previous:
                break
        for node in chosen:
            add(file, node, "member_initialization", owner)
            required_names[path].update(_names(node))
            required_names[path].update(control_context(file, node, emit=True))
    for path, file in index.files.items():
        # Include module constants referenced by methods or their initializers,
        # and the imports binding names those snippets actually use.
        assignments = [node for node in _scope_nodes(file.tree) if isinstance(node, _ASSIGNMENTS)]
        chosen = set()
        while True:
            previous = len(chosen)
            for node in assignments:
                if _assigned(node) & required_names[path]:
                    chosen.add(node)
                    required_names[path].update(_names(node))
                    required_names[path].update(control_context(file, node))
            if len(chosen) == previous:
                break
        for node in chosen:
            add(file, node, "module_assignment")
            control_context(file, node, emit=True)
        for node in file.imports:
            imported = {name.asname or name.name.split(".")[0] for name in node.names}
            if imported & required_names[path] or any(name.name == "*" for name in node.names):
                add(file, node, "import")
    return support


def extract_method_sources(sources, context):
    """Return per-path method views without mutating complete input snapshots.

    ``sha256`` retains the input full-file byte identity used by base_sha256.
    ``content_sha256`` and method ``snippet_sha256`` identify only view text.
    No size-based slicing is allowed: unavailable, partial or invalid input fails
    explicitly instead of presenting a silently incomplete replacement method.
    """
    index = _Index(sources)
    rules = deepcopy(DEFAULT_SOURCE_VIEW_RULES)
    for group, value in ((context.get("code") or {}).get("source_view_rules") or {}).items():
        rules[group] = deepcopy(value)
    editable = ((context.get("code") or {}).get("edit_policy") or {}).get("editable") or {}
    declarations = {path: [] for path in sources}
    for owner, specifications in editable.items():
        for file, cls in index.find_classes(owner):
            for specification in specifications:
                pattern = _pattern(specification)
                matches = [target for target in file.methods if target.rsplit(".", 1)[0] == cls
                           and fnmatchcase(target.rsplit(".", 1)[-1], pattern)]
                declaration = {"target_pattern": cls + "." + pattern, "declaration": specification,
                               "existing_targets": matches}
                if "signature" in specification:
                    declaration["signature"] = specification.split("signature", 1)[1].strip(" ()")
                declarations[file.path].append(declaration)
                for target in matches:
                    index.select(file, target, "declared_editable")
    for owner, patterns in rules["entry_methods"].items():
        for file, cls in index.find_classes(owner):
            index.select_class(file, cls, patterns, "adapter_entry")
    _select_runtime(index, context, rules)
    _select_helpers(index)
    support = _support(index)
    result = {}
    for path, file in index.files.items():
        methods = []
        for target, node in sorted(file.methods.items(), key=lambda item: _start(item[1])):
            if (path, target) in index.selected:
                code = file.code(node)
                methods.append({"target": target, "start_line": _start(node), "end_line": node.end_lineno,
                    "code": code, "snippet_sha256": _digest(code),
                    "reason": ",".join(sorted({reason.split(":", 1)[0] for reason in index.selected[(path, target)]}))})
        snippets = sorted(support[path].values(), key=lambda item: (item["start_line"], item["end_line"]))
        content = ""
        for item in sorted(snippets + methods, key=lambda part: (part["start_line"], part["end_line"])):
            code = item.pop("code")
            content += ("\n" if content else "") + "# %s:%s-%s\n" % (path, item["start_line"], item["end_line"])
            item["content_start"] = len(content)
            content += code
            item["content_end"] = len(content)
            item["snippet_sha256"] = _digest(code)
        symbols = {}
        for target, node in sorted({**file.classes, **file.methods}.items(), key=lambda item: _start(item[1])):
            if (path, target) not in index.selected:
                symbols[target] = [_start(node), node.end_lineno]
        value = deepcopy(file.metadata)
        value.update(view="methods-v1", hash_source="full_file", content=content,
            content_sha256=_digest(content), methods=methods,
            body_status="selected_methods" if methods else "not_sent",
            source_note="Diagnostic view; sha256 identifies the full file. symbol_index lists unsent bodies.",
            editable_targets=declarations[path])
        if "symbol_index" not in value:
            value["symbol_index"] = symbols
        else:
            value["available_symbols"] = symbols
        result[path] = value
    return result
