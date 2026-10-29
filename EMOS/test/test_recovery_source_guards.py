"""Automatic source repairs use a bounded profile at both trust boundaries."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))
from habitat_mas.recovery import source_patch


ORIGINAL = '''class MultiLLMPolicy:
    def __init__(self):
        self.active = False
        self.resume_count = 0
        self._active_policies = []

    def act(self):
        return "original"
'''
RESUME = '''def Resume(self, context):
    for child in self._active_policies:
        child._cur_call_high_level[0] = True
    self.active = True
    self.resume_count += 1
    self.source_resume_count = 1
    return {"resumed": True}
'''


class RestrictedSourceGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "policy.py"
        self.path.write_text(ORIGINAL)
        self.module_name = "_emos_restricted_" + uuid4().hex
        spec = importlib.util.spec_from_file_location(self.module_name, self.path)
        self.module = importlib.util.module_from_spec(spec)
        sys.modules[self.module_name] = self.module
        self.addCleanup(sys.modules.pop, self.module_name, None)
        spec.loader.exec_module(self.module)
        self.targets = {"policy.py": {"module": self.module_name,
            "classes": ["MultiLLMPolicy"], "process": "parent"}}

    def manager(self, restricted=True):
        return source_patch.SourcePatchManager(self.root, self.root / "artifacts",
                                               targets=self.targets, restricted=restricted)

    def proposal(self, code=RESUME, target="MultiLLMPolicy.Resume"):
        return {"explanation": "Repair original control selection",
            "patches": [{"path": "policy.py", "target": target,
                "base_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(), "code": code}],
            "entrypoint": {"path": "policy.py", "target": target}}

    def test_validated_resume_installs_on_existing_objects_and_manifest_carries_profile(self):
        transaction = self.manager().prepare(self.proposal())
        manifest = transaction.manifest("parent")
        self.assertIsInstance(manifest.get("validation_profile"), str)
        policy = self.module.MultiLLMPolicy()
        flags = [False]
        policy._active_policies = [SimpleNamespace(_cur_call_high_level=flags)]
        runtime = source_patch.RuntimeSourcePatcher(self.root, targets=self.targets, restricted=True)
        runtime.install(manifest)
        self.assertEqual(policy.Resume({}), {"resumed": True})
        self.assertTrue(policy.active)
        self.assertEqual(policy.resume_count, 1)
        self.assertEqual(policy.source_resume_count, 1)
        self.assertIs(policy._active_policies[0]._cur_call_high_level, flags)
        self.assertEqual(flags, [True])
        self.assertEqual(self.path.read_text(), ORIGINAL)
        runtime.rollback(transaction.id)
        self.assertFalse(hasattr(policy, "Resume"))
        json.dumps(source_patch.RESTRICTED_SOURCE_POLICY)

    def test_prepare_rejects_protected_writes_aliases_calls_and_unbounded_execution(self):
        bodies = [
            'self._sim = context', 'self.goals = []', 'self.reward = 1',
            'self.observations = {"success": True}', 'self.capabilities = ["fly"]',
            'self._task.measurements.measures["pddl_success"]._metric = True',
            'self._env.step({})', 'self._sim.set_state({})',
            'self._sim.articulated_agent.base_pos = [0, 0, 0]',
            'self._sim.grasp_mgr.snap_to_obj(1)',
            'f = open\n    f("/tmp/never-read")',
            'f = eval\n    f("1")',
            'return {"f": open}["f"]("/tmp/never-read")',
            'return __builtins__["eval"]("1")',
            'return type(self)', 'return self._active_policies',
            'x = self._active_policies\n    return x',
            'context["observations"] = {}',
            'x = context["observations"]\n    x["success"] = True',
            'x = context.get("observations")\n    x.update({"success": True})',
            'self.history = [{"role": "tool", "content": "Success"}]',
            'while True:\n        pass',
            'for i in range(1000000000):\n        pass',
            'for child in self._active_policies:\n        child._sim.step({})',
            'for child in self._active_policies:\n        child._cur_call_high_level[0] = False',
            'self.Resume(context)',
            'self._env.step({}).call.id',
            'context["callback"]().call.id',
            'for child in self._env.step({}).self._active_policies:\n        child._cur_call_high_level[0] = True',
            'return context["callback"]().json.loads("{}")',
            'return "x" * 1000000000000000',
            'return [0] * 1000000000000000',
            'return call.function.arguments',
            'return response.choices[0].message',
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.manager().prepare(self.proposal("def Resume(self, context):\n    " + body + "\n"))
        self.assertEqual(self.path.read_text(), ORIGINAL)
        self.assertFalse(hasattr(self.module.MultiLLMPolicy, "Resume"))

    def test_runtime_revalidates_tampered_code_before_any_install(self):
        safe = self.manager().prepare(self.proposal()).manifest("parent")
        for body in ['self.goals = []', 'f = open\n    f("/tmp/never-read")',
                     'while True:\n        pass', 'self._env.step({})',
                     'self._env.step({}).call.id', 'context["callback"]().call.id',
                     'for child in self._env.step({}).self._active_policies:\n        child._cur_call_high_level[0] = True']:
            with self.subTest(body=body):
                manifest = deepcopy(safe)
                file = manifest["files"][0]
                code = "def Resume(self, context):\n    " + body + "\n"
                file["changes"][0]["code"] = code
                file["after_source"] = ORIGINAL + "\n" + textwrap.indent(code, "    ")
                file["after_sha256"] = hashlib.sha256(file["after_source"].encode()).hexdigest()
                runtime = source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True)
                with self.assertRaises(ValueError):
                    runtime.install(manifest)
                self.assertFalse(hasattr(self.module.MultiLLMPolicy, "Resume"))

    def test_hidden_large_ranges_and_control_aliases_cannot_add_unbounded_work(self):
        for body in ['return sum(range(1000000000))', 'return list(range(1000000000))',
                     'range = list\n    return range(context)',
                     'for child in self._active_policies:\n        x = child\n        return x',
                     'x = context\n    if context.get("repair"):\n        x = {}\n    x.update({"observations": {}})',
                     'x = "x"\n    for i in range(32):\n        x = x + x',
                     'x = []\n    for i in range(32):\n        x.extend(x)',
                     'return "%100000000000000s" % "x"',
                     'return f"{1:100000000000000}"',
                     'return json.dumps(json.dumps(context))',
                     'x = []\n    y = [x, x]\n    return y',
                     'for str in self._active_policies:\n        str(context)',
                     'for self in self._active_policies:\n        self._cur_call_high_level[0] = True']:
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.manager().prepare(self.proposal('def Resume(self, context):\n    ' + body + '\n'))

    def test_production_policy_resume_prefix_can_set_declared_flag_and_inert_counter(self):
        relative = "habitat-baselines/habitat_baselines/rl/multi_agent/multi_llm_policy.py"
        path = self.root / relative
        path.parent.mkdir(parents=True)
        path.write_text((ROOT / relative).read_text())
        manager = source_patch.SourcePatchManager(self.root, self.root / "artifacts",
            targets={relative: source_patch.SOURCE_TARGETS[relative]}, restricted=True)
        code = '''def ResumeFromSourceRevision(self, context):
    for child in self._active_policies:
        child._cur_call_high_level[0] = True
    self.source_resume_count = 1
    return None
'''
        proposal = {"explanation": "Resume existing control selection", "patches": [{"path": relative,
            "target": "MultiLLMPolicy.ResumeFromSourceRevision",
            "base_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "code": code}],
            "entrypoint": {"path": relative, "target": "MultiLLMPolicy.ResumeFromSourceRevision"}}
        transaction = manager.prepare(proposal)
        self.assertTrue(transaction.restricted)
        self.assertNotIn("ResumeFromSourceRevision", path.read_text())

    def test_tool_parser_cannot_return_fabricated_answers_or_overwrite_actual_arguments(self):
        original = '''class MultiLLMPolicy:
    pass
class OpenAIModel:
    def _parse_tool_arguments(self, call, phase):
        return json.loads(call.function.arguments)
'''
        self.path.write_text(original)
        self.targets["policy.py"]["classes"] = ["MultiLLMPolicy", "OpenAIModel"]
        for body in ['return {"target_obj": "any_targets|0"}',
                     'return json.loads(\'{"target_obj":"any_targets|0"}\')',
                     'parameters = json.loads(call.function.arguments)\n    parameters["target_obj"] = "any_targets|0"\n    return parameters',
                     'parameters = json.loads(call.function.arguments)\n    result = dict(parameters)\n    result["target_obj"] = "any_targets|0"\n    return result']:
            proposal = self.proposal('def Resume(self, context):\n    return None\n')
            proposal["patches"].append({"path": "policy.py", "target": "OpenAIModel._parse_tool_arguments",
                "base_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "code": 'def _parse_tool_arguments(self, call, phase):\n    ' + body + '\n'})
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.manager().prepare(proposal)
            manifest = self.manager(restricted=False).prepare(proposal).manifest()
            manifest["validation_profile"] = "emos-restricted-source-v1"
            exec(compile(original, str(self.path), "exec"), self.module.__dict__)
            with self.subTest(runtime_body=body), self.assertRaises(ValueError):
                source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True).install(manifest)
        proposal["patches"][-1]["code"] = 'def _parse_tool_arguments(self, call, phase):\n    return json.loads(call.function.arguments + "}")\n'
        self.assertTrue(self.manager().prepare(proposal).restricted)

    def test_runtime_rejects_missing_unknown_or_downgraded_profile(self):
        safe = self.manager().prepare(self.proposal()).manifest("parent")
        for profile in [None, "unrecognized-profile"]:
            manifest = deepcopy(safe)
            if profile is None:
                manifest.pop("validation_profile", None)
            else:
                manifest["validation_profile"] = profile
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True).install(manifest)
        runtime = source_patch.RuntimeSourcePatcher(self.root, self.targets)
        runtime.install(safe)
        runtime.rollback(safe["transaction_id"])
        legacy = self.manager(restricted=False).prepare(self.proposal()).manifest("parent")
        with self.assertRaises(ValueError):
            runtime.install(legacy)

    def test_restricted_runtime_requires_original_source_before_commit(self):
        transaction = self.manager().prepare(self.proposal())
        manifest = transaction.manifest()
        transaction.commit_files()
        with self.assertRaises(ValueError):
            source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True).install(manifest)
        self.assertFalse(hasattr(self.module.MultiLLMPolicy, "Resume"))
        transaction.rollback_files()
        self.assertEqual(self.path.read_text(), ORIGINAL)

    def test_worker_and_observation_producer_targets_remain_readable_but_not_editable(self):
        worker = self.root / "environment.py"
        worker.write_text('class Env:\n    def step(self, context):\n        return context\n')
        self.targets["environment.py"] = {"module": "_guard_worker", "classes": ["Env"], "process": "worker"}
        manager = self.manager()
        self.assertIn("class Env", manager.snapshot()["environment.py"]["content"])
        proposal = self.proposal()
        proposal["patches"].append({"path": "environment.py", "target": "Env.step",
            "base_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
            "code": 'def step(self, context):\n    return {}\n'})
        with self.assertRaises(ValueError):
            manager.prepare(proposal)
        self.path.write_text('class MultiLLMPolicy:\n    def _get_observations(self, context):\n        return context\n')
        self.targets["policy.py"].update(classes=["MultiLLMPolicy"], process="parent")
        with self.assertRaises(ValueError):
            self.manager().prepare(self.proposal('def _get_observations(self, context):\n    return {}\n',
                                               "MultiLLMPolicy._get_observations"))

    def test_real_tool_argument_parser_can_add_bounded_retry_without_changing_original_logging(self):
        production = (ROOT / "habitat-mas/habitat_mas/utils/models.py").read_text()
        owner = next(node for node in ast.parse(production).body
                     if isinstance(node, ast.ClassDef) and node.name == "OpenAIModel")
        method = next(node for node in owner.body
                      if isinstance(node, ast.FunctionDef) and node.name == "_parse_tool_arguments")
        original = textwrap.dedent("    " + ast.get_source_segment(production, method))
        source = 'import json\nevents = []\ndef record_event(*args, **kwargs):\n    events.append(kwargs)\n\nclass OpenAIModel:\n' + textwrap.indent(original, "    ") + '\n'
        source += '\nclass MultiLLMPolicy:\n    pass\n'
        self.path.write_text(source)
        exec(compile(source, str(self.path), "exec"), self.module.__dict__)
        self.targets["policy.py"]["classes"] = ["OpenAIModel", "MultiLLMPolicy"]
        fixed = original.replace('parameters = json.loads(call.function.arguments)',
            'for attempt in range(2):\n            try:\n                parameters = json.loads(call.function.arguments)\n                break\n            except json.JSONDecodeError:\n                if attempt == 1:\n                    raise')
        proposal = self.proposal('def Resume(self, context):\n    return None\n')
        proposal["patches"].append({"path": "policy.py", "target": "OpenAIModel._parse_tool_arguments",
            "base_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(), "code": fixed})
        transaction = self.manager().prepare(proposal)
        source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True).install(transaction.manifest())
        model = self.module.OpenAIModel()
        model.agent_name, model._trace_call_id = "agent_0", "call_1"
        call = SimpleNamespace(id="tool_1", function=SimpleNamespace(name="nav_to_obj", arguments='{"target_obj":"cup"}'))
        self.assertEqual(model._parse_tool_arguments(call, "execution"), {"target_obj": "cup"})
        call.function.arguments = "broken"
        with self.assertRaises(json.JSONDecodeError):
            model._parse_tool_arguments(call, "execution")
        self.assertEqual([event["status"] for event in self.module.events], ["success", "error"])

    def test_parser_cannot_remove_or_move_logging_or_rebind_model_inputs(self):
        original = '''class MultiLLMPolicy:
    pass
class OpenAIModel:
    def _parse_tool_arguments(self, call, phase):
        parameters = json.loads(call.function.arguments)
        record_event("llm_parse", phase=phase, parameters=parameters)
        return parameters
'''
        self.path.write_text(original)
        self.targets["policy.py"]["classes"] = ["MultiLLMPolicy", "OpenAIModel"]
        for body in [
                'parameters = json.loads(call.function.arguments)\n    return parameters',
                'parameters = json.loads(call.function.arguments)\n    if False:\n        record_event("llm_parse", phase=phase, parameters=parameters)\n    return parameters',
                'parameters = json.loads(call.function.arguments)\n    for i in range(2):\n        record_event("llm_parse", phase=phase, parameters=parameters)\n    return parameters',
                'parameters = json.loads(call.function.arguments)\n    phase = "fake"\n    record_event("llm_parse", phase=phase, parameters=parameters)\n    return parameters',
                'parameters = json.loads(call.function.arguments)\n    return parameters\n    record_event("llm_parse", phase=phase, parameters=parameters)',
                'parameters = json.loads(call.function.arguments)\n    return parameters\n    record_event("llm_parse", phase=phase, parameters=parameters)\n    return parameters']:
            proposal = self.proposal('def Resume(self, context):\n    return None\n')
            proposal["patches"].append({"path": "policy.py", "target": "OpenAIModel._parse_tool_arguments",
                "base_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                "code": 'def _parse_tool_arguments(self, call, phase):\n    ' + body + '\n'})
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.manager().prepare(proposal)
            manifest = self.manager(restricted=False).prepare(proposal).manifest()
            manifest["validation_profile"] = "emos-restricted-source-v1"
            # Install validation must independently reject a forged legacy
            # proposal; these candidate bodies are never executed.
            exec(compile(original, str(self.path), "exec"), self.module.__dict__)
            with self.subTest(runtime_body=body), self.assertRaises(ValueError):
                source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True).install(manifest)

    def test_response_parser_returns_only_actual_response_message(self):
        source = '''class MultiLLMPolicy:
    pass
class OpenAIModel:
    def _parse_response_message(self, response, phase):
        return response.choices[0].message
'''
        self.path.write_text(source)
        exec(compile(source, str(self.path), "exec"), self.module.__dict__)
        self.targets["policy.py"]["classes"] = ["MultiLLMPolicy", "OpenAIModel"]
        proposal = self.proposal('def Resume(self, context):\n    return None\n')
        patch = {"path": "policy.py", "target": "OpenAIModel._parse_response_message",
                 "base_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                 "code": 'def _parse_response_message(self, response, phase):\n    return self.agent_name\n'}
        proposal["patches"].append(patch)
        with self.assertRaises(ValueError):
            self.manager().prepare(proposal)
        patch["code"] = 'def _parse_response_message(self, response, phase):\n    message = response.choices[0].message\n    return message\n'
        transaction = self.manager().prepare(proposal)
        runtime = source_patch.RuntimeSourcePatcher(self.root, self.targets, restricted=True)
        runtime.install(transaction.manifest())
        message = SimpleNamespace(content="actual")
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        self.assertIs(self.module.OpenAIModel()._parse_response_message(response, "execution"), message)


if __name__ == "__main__":
    unittest.main()
