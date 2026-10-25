"""Recovery changes live decisions; all model transports are local test data."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.recovery.config import RecoveryConfig
from habitat_mas.recovery.context import build_context
from habitat_mas.recovery.controller import RecoveryController
from habitat_mas.recovery.generator import RecoveryGenerator
from habitat_mas.recovery.program import ProgramError, RecoveryProgram


RESUME = '''def recover(context, memory):
    return {"commands": [{"type": "resume", "agent": "agent_0", "reason": "stalled"}], "memory": {"resumed": True}}
'''
MOVE = '''def recover(context, memory):
    commands = []
    if not memory.get("sent", False):
        commands.append({"type": "action", "agent": "agent_0", "name": "nav_to_obj", "arguments": {"target_obj": "target", "robot": "agent_0"}})
    return {"commands": commands, "memory": {"sent": True}}
'''


def frame(step, position=0):
    return {"step": step, "task": "mobility", "context": {
        "task_description": "Cover target together", "scene_description": "room",
        "assignments": {"agent_0": "target"}}, "agents": {"agent_0": {
        "position": [position, 0, 0], "rotation": [0, 0, 0, 1],
        "held_object": "cup", "action": {"name": "wait", "arguments": {}, "age": step + 1}}},
        "objects": {}, "metrics": {"distance_to_goal": 10 - position},
        "decisions": [], "messages": []}


class Policy:
    def __init__(self):
        self.history = [[{"role": "user", "content": str(i)}] for i in range(15)]
        self.commands = []
        self.mode = "wait"

    def get_recovery_context(self, history_turns=10):
        return {"agents": {"agent_0": {
            "task_description": "Cover target together", "subtask_description": "target",
            "planner_plan": "navigate then wait", "recent_history": self.history[-history_turns:],
            "available_tools": [{"name": "nav_to_obj", "parameters": {"type": "object"}}],
            "current_skill": "wait"}}}

    def validate_recovery_commands(self, commands):
        if not commands:
            raise ValueError("commands must be nonempty")
        for command in commands:
            if command["agent"] != "agent_0":
                raise ValueError("unknown agent")
            if command["type"] == "action" and command["name"] != "nav_to_obj":
                raise ValueError("unknown skill")

    def apply_recovery_commands(self, commands):
        self.validate_recovery_commands(commands)
        self.commands.extend(copy.deepcopy(commands))
        self.mode = "move"
        return {"applied": commands, "simulator_reset": False}


class Detector:
    def __init__(self, alarm=True):
        self.frames = []
        self.alarm = alarm

    def observe(self, value):
        self.frames.append(copy.deepcopy(value))
        return {"status": "system_error" if self.alarm else "normal", "reason_code": "stall",
                "explanation": "no progress", "agents": ["agent_0"],
                "evidence": [{"step": value["step"], "observation": "waiting"}], "confidence": .9}


class Generator:
    def __init__(self, *codes):
        self.codes = iter(codes)
        self.contexts = []

    def generate(self, context):
        self.contexts.append(copy.deepcopy(context))
        return {"code": next(self.codes), "explanation": "restore progress"}


class ProgramTests(unittest.TestCase):
    def test_generated_condition_and_memory_determine_commands(self):
        program = RecoveryProgram(MOVE)
        first = program.run({}, {}, timeout=1)
        second = program.run({}, first["memory"], timeout=1)
        self.assertEqual(first["commands"][0]["name"], "nav_to_obj")
        self.assertEqual(second["commands"], [])

    def test_forbidden_code_cannot_read_files_or_escape_data_api(self):
        for body in ["import os\n    return {}", "return open('/etc/passwd').read()",
                     "return context.__class__", "return getattr(context, '__class__')",
                     "return eval('1')", "while True:\n        pass"]:
            with self.subTest(body=body), self.assertRaises(ProgramError):
                RecoveryProgram("def recover(context, memory):\n    " + body + "\n")

    def test_expensive_generated_loop_has_a_wall_clock_limit(self):
        program = RecoveryProgram('''def recover(context, memory):
    for i in range(1000000000):
        memory["x"] = i
    return {"commands": [], "memory": memory}
''')
        with self.assertRaises(ProgramError):
            program.run({}, {}, timeout=.1)

    def test_invalid_output_and_duplicate_agent_commands_are_rejected(self):
        for result in ["None", '{"commands": [], "memory": []}',
                       '{"commands": [{"type":"shell","agent":"agent_0"}], "memory":{}}',
                       '{"commands": [{"type":"resume","agent":"agent_0"}, {"type":"resume","agent":"agent_0"}], "memory":{}}']:
            with self.subTest(result=result), self.assertRaises(ProgramError):
                RecoveryProgram("def recover(context, memory):\n    return " + result).run({}, {})


class ContextTests(unittest.TestCase):
    def test_fixed_context_includes_actual_plan_history_sources_and_no_labels(self):
        value = frame(50)
        value["native_success"] = "HIDDEN_OUTCOME"
        value["agents"]["agent_0"]["hang_state"] = "HIDDEN_FAULT"
        context = build_context(value, [value], {"status": "system_error"}, Policy(),
                                RecoveryConfig(), [], {}, sources={"control": {"content": "def act(): pass"}})
        self.assertEqual(set(context), {"schema_version", "modification_instruction", "error_report",
            "task", "planner_plan", "recent_history", "runtime", "code", "recovery", "available_api"})
        self.assertEqual(context["planner_plan"]["agent_0"], "navigate then wait")
        self.assertEqual(len(context["recent_history"]["agent_0"]), 10)
        self.assertEqual(context["recent_history"]["agent_0"][0][0]["content"], "5")
        self.assertIn("def act()", json.dumps(context))
        self.assertNotIn("HIDDEN_", json.dumps(context))

    def test_generator_requires_complete_json_code_response(self):
        requests = []
        def create(**request):
            requests.append(request)
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps({"code": RESUME, "explanation": "resume"})}}]}
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = RecoveryGenerator(client=client).generate({"task": "example"})
        self.assertEqual(result["code"], RESUME)
        self.assertEqual(json.loads(requests[0]["messages"][1]["content"]), {"task": "example"})
        self.assertIn("recover(context, memory)", requests[0]["messages"][0]["content"])

    def test_failed_model_request_retains_the_exact_prompt_for_audit(self):
        def create(**request):
            raise TimeoutError("transport timeout")
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with self.assertRaises(ValueError) as caught:
            RecoveryGenerator(client=client).generate({"task": "example"})
        self.assertEqual(json.loads(caught.exception.request["messages"][1]["content"]), {"task": "example"})
        self.assertIsNone(caught.exception.response)


class ControllerTests(unittest.TestCase):
    def setup_controller(self, directory, codes=(RESUME,), **changes):
        config = RecoveryConfig(sample_every=1, window=2, check_every=1,
            cooldown_steps=2, verify_after_steps=2, **changes)
        detector, generator = Detector(), Generator(*codes)
        return RecoveryController(config, directory, detector=detector, generator=generator), detector, generator

    def test_generated_code_resumes_same_environment_without_losing_held_object(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _, generator = self.setup_controller(directory, codes=(MOVE,))
            policy, position, resets = Policy(), 0, 0
            first = monitor.observe(frame(0, position), policy)
            position += int(policy.mode == "move")
            monitor.observe(frame(1, position), policy)
            self.assertEqual(first["status"], "applied")
            self.assertEqual(position, 1)
            self.assertEqual(resets, 0)
            self.assertEqual(len(policy.history), 15)
            self.assertEqual(generator.contexts[0]["runtime"]["current_frame"]["agents"]["agent_0"]["held_object"], "cup")
            self.assertTrue(list(Path(directory).glob("versions/*/recovery.py")))

    def test_next_generation_receives_previous_code_and_progress_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _, generator = self.setup_controller(directory, codes=(RESUME, MOVE))
            policy = Policy()
            monitor.observe(frame(0), policy)
            monitor.observe(frame(1), policy)
            monitor.observe(frame(2, 1), policy)
            text = json.dumps(generator.contexts[1], ensure_ascii=False)
            self.assertIn(RESUME, text.replace('\\n', '\n').replace('\\"', '"'))
            self.assertIn("progress_observed", text)
            self.assertEqual(len(policy.commands), 2)

    def test_bad_revision_does_not_replace_previous_code(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _, _ = self.setup_controller(directory, codes=(RESUME, "import os"))
            policy = Policy()
            monitor.observe(frame(0), policy)
            monitor.observe(frame(1), policy)
            result = monitor.observe(frame(2), policy)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(monitor.active_code, RESUME)

    def test_empty_command_branch_keeps_code_and_updates_memory(self):
        code = '''def recover(context, memory):
    return {"commands": [], "memory": {"observations": memory.get("observations", 0) + 1}}
'''
        with tempfile.TemporaryDirectory() as directory:
            monitor, detector, _ = self.setup_controller(directory, codes=(code,))
            result = monitor.observe(frame(0), Policy())
            self.assertEqual(result["status"], "active_no_command")
            detector.alarm = False
            monitor.observe(frame(1), Policy())
            self.assertEqual(monitor.memory, {"observations": 2})
            self.assertEqual(monitor.active_code, code)

    def test_later_execution_error_rolls_back_to_previous_valid_version(self):
        bad_later = '''def recover(context, memory):
    if context["runtime"]["current_frame"]["step"] >= 4:
        return 1 / 0
    return {"commands": [], "memory": {"revision": 2}}
'''
        with tempfile.TemporaryDirectory() as directory:
            monitor, detector, _ = self.setup_controller(directory, codes=(RESUME, bad_later))
            policy = Policy()
            monitor.observe(frame(0), policy)
            monitor.observe(frame(2), policy)
            detector.alarm = False
            result = monitor.observe(frame(4), policy)
            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(monitor.active_code, RESUME)
            self.assertEqual(monitor.memory, {"resumed": True})

    def test_terminal_frame_never_calls_generator_or_mutates_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _, generator = self.setup_controller(directory)
            policy = Policy()
            monitor.observe(frame(0), policy, terminal=True)
            self.assertEqual(generator.contexts, [])
            self.assertEqual(policy.commands, [])

    def test_cooldown_and_attempt_budget_bound_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            monitor, _, generator = self.setup_controller(directory, codes=(RESUME, RESUME), max_iterations=2)
            policy = Policy()
            for step in range(10):
                monitor.observe(frame(step), policy)
            self.assertEqual(len(generator.contexts), 2)
            self.assertEqual([c["runtime"]["current_frame"]["step"] for c in generator.contexts], [0, 2])

    def test_sampling_keeps_intermediate_messages_and_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            config = RecoveryConfig(sample_every=2)
            detector = Detector(alarm=False)
            monitor = RecoveryController(config, directory, detector=detector, generator=Generator())
            for step in range(3):
                value = frame(step)
                value["messages"] = [{"step": step, "message": str(step)}]
                monitor.observe(value, Policy())
            self.assertEqual([f["step"] for f in detector.frames], [0, 2])
            self.assertEqual([m["step"] for m in detector.frames[1]["messages"]], [1, 2])

    def test_unknown_tool_does_not_apply_any_command(self):
        code = '''def recover(context, memory):
    return {"commands":[{"type":"action","agent":"agent_0","name":"unknown","arguments":{}}],"memory":{}}
'''
        with tempfile.TemporaryDirectory() as directory:
            monitor, _, _ = self.setup_controller(directory, codes=(code,))
            policy = Policy()
            result = monitor.observe(frame(0), policy)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(policy.commands, [])
            self.assertIsNone(monitor.active_code)


if __name__ == "__main__":
    unittest.main()
