"""Method views preserve replacement bodies and full-file version identities."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def extract(sources, context):
    from habitat_mas.recovery.source_view import extract_method_sources
    return extract_method_sources(sources, context)


def source(text, classes, module="example", **metadata):
    return {"content": text, "classes": classes, "module": module,
            "process": "parent", "sha256": hashlib.sha256(text.encode()).hexdigest(), **metadata}


def context(editable=None, **runtime):
    return {"code": {"edit_policy": {"editable": editable or {}}}, "runtime": runtime}


class MethodSourceTests(unittest.TestCase):
    def test_editable_method_keeps_decorators_multiline_signature_and_complete_body(self):
        body = ('    @classmethod\n    @audit("x")\n'
                '    def parse(\n        cls, value: str,\n        *, strict=True,\n    ) -> dict:\n'
                '        result = {"value": value}\n        return result\n')
        text = 'import json\nfrom decorators import audit\n\nclass Parser:\n' + body + (
            '    def unused(self):\n        return "NOT_SENT"\n')
        sources = {"parser.py": source(text, ["Parser"], custom={"keep": True})}
        original = deepcopy(sources)
        value = extract(sources, context({"Parser": ["parse (parser region)"]}))["parser.py"]
        method = value["methods"][0]
        self.assertEqual(value["content"][method["content_start"]:method["content_end"]], body)
        self.assertNotIn("code", method)
        self.assertEqual((method["start_line"], method["end_line"]), (5, 12))
        self.assertIn('*, strict=True,', value["content"][method["content_start"]:method["content_end"]])
        self.assertEqual(method["snippet_sha256"], hashlib.sha256(body.encode()).hexdigest())
        self.assertIn(body, value["content"])
        self.assertIn('from decorators import audit', value["content"])
        self.assertNotIn('NOT_SENT', value["content"])
        self.assertEqual(value["sha256"], original["parser.py"]["sha256"])
        self.assertEqual(value["hash_source"], "full_file")
        self.assertNotEqual(value["content_sha256"], value["sha256"])
        self.assertEqual(value["custom"], {"keep": True})
        self.assertEqual(sources, original)

    def test_unselected_read_only_file_keeps_index_without_claiming_empty_full_source(self):
        text = 'class Env:\n    def reset(self):\n        return "DO_NOT_SEND_BODY"\n'
        value = extract({"env.py": source(text, ["Env"])}, context())["env.py"]
        self.assertEqual(value["content"], "")
        self.assertEqual(value["methods"], [])
        self.assertEqual(value["body_status"], "not_sent")
        self.assertEqual(value["symbol_index"]["Env.reset"], [2, 3])
        self.assertEqual(value["sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_resume_wildcard_is_reextracted_from_each_current_snapshot(self):
        old = 'class MultiLLMPolicy:\n    def act(self):\n        return 1\n'
        new = old + '    def ResumeLatest(self, context):\n        return {"revision": 2}\n'
        cfg = context({"MultiLLMPolicy": ["Resume* (signature self, context)"]})
        first = extract({"p.py": source(old, ["MultiLLMPolicy"])}, cfg)["p.py"]
        second = extract({"p.py": source(new, ["MultiLLMPolicy"])}, cfg)["p.py"]
        self.assertEqual(first["editable_targets"][0]["existing_targets"], [])
        self.assertIn("self, context", first["editable_targets"][0]["signature"])
        self.assertNotIn("def Resume", first["content"])
        self.assertIn('return {"revision": 2}', second["content"])
        self.assertIn("MultiLLMPolicy.ResumeLatest", [item["target"] for item in second["methods"]])
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_one_hop_helpers_inheritance_and_member_initializers_are_available(self):
        base = ('class Base:\n    def __init__(self):\n        self.limit = 4\n'
                '        self.unrelated = "HIDE_INIT"\n    def helper(self, value):\n'
                '        return value < self.limit\n')
        child = ('from base import Base\n\ndef normalize(value):\n    return value.strip()\n\n'
                 'class Child(Base):\n    def __init__(self):\n        self.value = "test"\n'
                 '        self.unused = "HIDE_CHILD_INIT"\n    def run(self):\n'
                 '        return self.helper(normalize(self.value))\n')
        values = extract({"base.py": source(base, ["Base"], "base"),
                          "child.py": source(child, ["Child"], "child")}, context({"Child": ["run"]}))
        self.assertIn("Base.helper", [item["target"] for item in values["base.py"]["methods"]])
        self.assertIn("normalize", [item["target"] for item in values["child.py"]["methods"]])
        self.assertIn("self.limit = 4", values["base.py"]["content"])
        self.assertIn('self.value = "test"', values["child.py"]["content"])
        self.assertIn("class Child(Base):", values["child.py"]["content"])
        for value in values.values():
            self.assertNotIn("HIDE_", value["content"])
            self.assertNotIn("def __init__", value["content"])

    def test_actual_runtime_source_and_traceback_select_read_only_diagnostic_methods(self):
        text = ('class Motor:\n    def step(self):\n        return self.measure()\n'
                '    def measure(self):\n        return 7\n    def reset(self):\n'
                '        return "NO_RESET"\n\nclass Sensor:\n    def read(self):\n        return 3\n')
        cfg = context(current_frame={"agents": {"a": {"navigation": {"source": "Motor"}}}})
        cfg["error_report"] = {"exception": {"traceback": [{"path": "system.py", "line": 11, "function": "read"}]}}
        value = extract({"system.py": source(text, ["Motor", "Sensor"], process="worker")}, cfg)["system.py"]
        targets = {item["target"] for item in value["methods"]}
        self.assertTrue({"Motor.step", "Motor.measure", "Sensor.read"} <= targets)
        self.assertNotIn("NO_RESET", value["content"])

    def test_runtime_binding_role_selects_bound_entry_methods_without_all_methods(self):
        text = ('class CustomPolicy:\n    def act(self):\n        return 1\n'
                '    def other(self):\n        return "UNRELATED"\n')
        cfg = context(class_bindings=[{"role": "policy", "module": "custom", "class": "CustomPolicy"}])
        value = extract({"custom.py": source(text, ["CustomPolicy"], "custom")}, cfg)["custom.py"]
        self.assertEqual([item["target"] for item in value["methods"]], ["CustomPolicy.act"])
        self.assertNotIn("UNRELATED", value["content"])

    def test_bound_subclass_uses_inherited_entry_method(self):
        text = ('class Base:\n    def act(self):\n        return self.limit\n\nclass LivePolicy(Base):\n'
                '    def __init__(self):\n        self.limit = 7\n')
        cfg = context(class_bindings=[{"role": "policy", "module": "custom", "class": "LivePolicy"}])
        value = extract({"custom.py": source(text, ["Base", "LivePolicy"], "custom")}, cfg)["custom.py"]
        self.assertIn("Base.act", [item["target"] for item in value["methods"]])
        self.assertIn("class LivePolicy(Base):", value["content"])
        self.assertIn("self.limit = 7", value["content"])

    def test_inherited_entry_uses_runtime_subclass_helper_override(self):
        text = ('class Base:\n    def act(self):\n        return self.helper()\n'
                '    def helper(self):\n        return "BASE_HELPER"\n\n'
                'class LivePolicy(Base):\n    def helper(self):\n        return "LIVE_HELPER"\n')
        cfg = context(class_bindings=[{"role": "policy", "module": "custom", "class": "LivePolicy"}])
        value = extract({"custom.py": source(text, ["Base", "LivePolicy"], "custom")}, cfg)["custom.py"]
        self.assertIn("LIVE_HELPER", value["content"])
        self.assertNotIn("BASE_HELPER", value["content"])

    def test_multiple_inheritance_uses_python_method_resolution_order(self):
        text = ('class Root:\n    def act(self):\n        return "ROOT_BODY"\n'
                'class Left(Root):\n    pass\nclass Right(Root):\n'
                '    def act(self):\n        return "RIGHT_BODY"\n'
                'class LivePolicy(Left, Right):\n    pass\n')
        cfg = context(class_bindings=[{"role": "policy", "module": "custom", "class": "LivePolicy"}])
        value = extract({"custom.py": source(text, ["Root", "Left", "Right", "LivePolicy"], "custom")}, cfg)["custom.py"]
        self.assertIn("RIGHT_BODY", value["content"])
        self.assertNotIn("ROOT_BODY", value["content"])

    def test_super_helper_uses_next_runtime_mro_class(self):
        text = ('class Root:\n    def act(self):\n        return "ROOT_BODY"\n'
                'class Left(Root):\n    def act(self):\n        return super().act()\n'
                'class Right(Root):\n    def act(self):\n        return "RIGHT_BODY"\n'
                'class LivePolicy(Left, Right):\n    pass\n')
        cfg = context(class_bindings=[{"role": "policy", "module": "custom", "class": "LivePolicy"}])
        value = extract({"custom.py": source(text, ["Root", "Left", "Right", "LivePolicy"], "custom")}, cfg)["custom.py"]
        self.assertIn("RIGHT_BODY", value["content"])
        self.assertNotIn("ROOT_BODY", value["content"])

    def test_related_field_initializers_keep_rhs_dependencies_and_conditions(self):
        text = ('class Worker:\n    def __init__(self, cfg):\n        self.base = cfg.base\n'
                '        if cfg.strict:\n            self.limit = self.base + 1\n'
                '        else:\n            self.limit = self.base + 20\n'
                '        self.unrelated = "HIDE_ME"\n    def run(self):\n        return self.limit\n')
        value = extract({"w.py": source(text, ["Worker"])}, context({"Worker": ["run"]}))["w.py"]
        self.assertIn("self.base = cfg.base", value["content"])
        self.assertIn("if cfg.strict:", value["content"])
        self.assertIn("else:", value["content"])
        self.assertNotIn("HIDE_ME", value["content"])
        self.assertNotIn("def __init__", value["content"])

    def test_module_conditional_constant_and_metaclass_import_remain_available(self):
        text = ('import helpers\nfrom types import Meta\nif helpers.ENABLED:\n    LIMIT = helpers.VALUE\n'
                'class Worker(metaclass=Meta):\n    def run(self):\n        return LIMIT\n')
        value = extract({"w.py": source(text, ["Worker"])}, context({"Worker": ["run"]}))["w.py"]
        self.assertIn("from types import Meta", value["content"])
        self.assertIn("if helpers.ENABLED:", value["content"])
        self.assertIn("LIMIT = helpers.VALUE", value["content"])

    def test_missing_invalid_and_pretruncated_sources_are_rejected_explicitly(self):
        for data in ({"content": None, "missing_reason": "FileNotFoundError"},
                     source('class Broken:\n    def run(', ["Broken"]),
                     source('class Cut:\n    pass\n', ["Cut"], truncated=True)):
            with self.subTest(data=data), self.assertRaisesRegex(ValueError, "bad.py"):
                extract({"bad.py": data}, context({"Broken": ["run"]}))

    def test_method_longer_than_old_preview_limit_is_never_truncated(self):
        body = '    def run(self):\n' + ''.join('        value_%d = %r\n' % (i, 'x' * 100) for i in range(600)) + '        return "TAIL_PRESENT"\n'
        text = 'class Long:\n' + body
        value = extract({"long.py": source(text, ["Long"])}, context({"Long": ["run"]}))["long.py"]
        method = value["methods"][0]
        self.assertEqual(value["content"][method["content_start"]:method["content_end"]], body)
        self.assertIn('return "TAIL_PRESENT"', value["content"])

    def test_read_only_property_getter_and_setter_both_remain_in_symbol_index(self):
        text = ('class Env:\n    @property\n    def episode(self):\n        return self._episode\n'
                '    @episode.setter\n    def episode(self, value):\n        self._episode = value\n')
        value = extract({"env.py": source(text, ["Env"])}, context())["env.py"]
        self.assertEqual(value["symbol_index"]["Env.episode"], [2, 4])
        self.assertEqual(value["symbol_index"]["Env.episode@setter"], [5, 7])
        self.assertEqual(value["content"], "")

    def test_helper_expansion_stops_after_one_hop(self):
        text = ('class Worker:\n    def run(self):\n        return self.first()\n'
                '    def first(self):\n        return self.second()\n'
                '    def second(self):\n        return "SECOND_HOP_NOT_SENT"\n')
        value = extract({"w.py": source(text, ["Worker"])}, context({"Worker": ["run"]}))["w.py"]
        self.assertEqual([item["target"] for item in value["methods"]], ["Worker.run", "Worker.first"])
        self.assertNotIn("SECOND_HOP_NOT_SENT", value["content"])

    def test_module_alias_helper_and_class_constant_are_included_as_dependencies(self):
        helper = 'def clean(value):\n    return value.strip()\n'
        text = ('import helpers as h\n\nclass Parser:\n    limit = 4\n'
                '    other = "HIDDEN_CONSTANT"\n    def parse(self, value):\n'
                '        return h.clean(value)[:self.limit]\n')
        values = extract({"helpers.py": source(helper, [], "helpers"),
                          "p.py": source(text, ["Parser"])}, context({"Parser": ["parse"]}))
        self.assertIn("clean", [item["target"] for item in values["helpers.py"]["methods"]])
        self.assertIn("limit = 4", values["p.py"]["content"])
        self.assertNotIn("HIDDEN_CONSTANT", values["p.py"]["content"])

    def test_view_metadata_keeps_unselected_index_compact_and_does_not_repeat_code(self):
        text = ('class Parser:\n    def parse(self, value):\n        return value\n'
                '    def unused(self, long_signature_argument):\n        return "HIDDEN_BODY"\n')
        value = extract({"p.py": source(text, ["Parser"])}, context({"Parser": ["parse"]}))["p.py"]
        self.assertNotIn("Parser.parse", value["symbol_index"])
        self.assertEqual(value["symbol_index"]["Parser.unused"], [4, 5])
        self.assertNotIn("signature", value["methods"][0])
        self.assertNotIn("snippets", value)
        self.assertEqual(value["content"].count("def parse("), 1)

    def test_current_repository_snapshot_keeps_each_selected_method_exact_and_reduces_payload(self):
        from habitat_mas.recovery.source_patch import source_snapshot, RESTRICTED_SOURCE_POLICY
        sources = source_snapshot()
        values = extract(sources, {"code": {"edit_policy": RESTRICTED_SOURCE_POLICY}})
        self.assertLess(len(json.dumps(values)), len(json.dumps(sources)) * 0.65)
        required = {"OpenAIModel._parse_tool_arguments", "OpenAIModel._parse_response_message",
                    "WaitSkillPolicy._is_skill_done", "SkillPolicy.should_terminate",
                    "OracleNavPolicy._is_skill_done", "OracleNavAction.step",
                    "OracleNavCoordinateAction.step", "OracleNavDiffBaseAction.step"}
        self.assertTrue(required <= {method["target"] for value in values.values() for method in value["methods"]})
        for path, value in values.items():
            original = sources[path]
            self.assertEqual(value["sha256"], original["sha256"])
            self.assertEqual(value["module"], original["module"])
            self.assertEqual(value["classes"], original["classes"])
            self.assertEqual(value["process"], original["process"])
            lines = original["content"].splitlines(keepends=True)
            for method in value["methods"]:
                with self.subTest(path=path, target=method["target"]):
                    body = value["content"][method["content_start"]:method["content_end"]]
                    self.assertEqual(body, ''.join(lines[method["start_line"] - 1:method["end_line"]]))
                    self.assertEqual(method["snippet_sha256"], hashlib.sha256(body.encode()).hexdigest())
                    self.assertFalse(method["target"].endswith(".__init__"))
            tree = ast.parse(original["content"])
            indexed = set(value["symbol_index"]) | {method["target"] for method in value["methods"]}
            for owner in tree.body:
                if not isinstance(owner, ast.ClassDef):
                    continue
                for node in owner.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self.assertIn(owner.name + "." + node.name, indexed)


if __name__ == "__main__":
    unittest.main()
