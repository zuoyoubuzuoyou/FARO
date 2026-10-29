"""Explicit runtime deduplication preserves the sampled observations exactly."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))


def compaction_module(test):
    name = "habitat_mas.recovery.context_compaction"
    test.assertIsNotNone(importlib.util.find_spec(name), "runtime compaction is required")
    return __import__(name, fromlist=["compact_runtime_context"])


class ContextCompactionTests(unittest.TestCase):
    def test_current_frame_is_explicitly_identified_with_its_sampled_index_and_step(self):
        module = compaction_module(self)
        original = {"runtime": {"current_frame": {"step": 20, "metrics": {"success": False}},
                    "recent_frames": [{"step": 5}, {"step": 20, "metrics": {"success": False}}]}}
        compact = module.compact_runtime_context(original)
        self.assertNotIn("current_frame", compact["runtime"])
        current = compact["input_layout"]["runtime_compaction"]["current_frame"]
        self.assertEqual(current["recent_frames_index"], 1)
        self.assertEqual(current["step"], 20)
        self.assertEqual(current["source_path"], "$.runtime.recent_frames[1]")
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], original["runtime"])

    def test_only_matching_live_agent_task_plan_and_history_fields_are_omitted(self):
        module = compaction_module(self)
        original = {"task": {"description": "Reach goal", "assignments": {"agent_0": "Scout"}},
                    "planner_plan": {"agent_0": "inspect then move", "agent_1": "wait"},
                    "recent_history": {"agent_0": [{"role": "user", "content": "go"}]},
                    "runtime": {"agents": {
                        "agent_0": {"task_description": "Reach goal", "subtask_description": "Scout",
                                    "planner_plan": "inspect then move",
                                    "recent_history": [{"role": "user", "content": "go"}],
                                    "current_skill": "move"},
                        "agent_1": {"task_description": "A different goal", "subtask_description": "Wait",
                                    "planner_plan": "another plan", "recent_history": None}}}}
        compact = module.compact_runtime_context(original)
        self.assertEqual(compact["runtime"]["agents"]["agent_0"], {"current_skill": "move"})
        self.assertEqual(compact["runtime"]["agents"]["agent_1"], original["runtime"]["agents"]["agent_1"])
        mappings = compact["input_layout"]["runtime_compaction"]["agent_fields"]
        self.assertEqual({entry["source_path"] for entry in mappings}, {
            "$.task.description", '$.task.assignments["agent_0"]',
            '$.planner_plan["agent_0"]', '$.recent_history["agent_0"]'})
        self.assertTrue(all(entry["omitted_path"].startswith('$.runtime.agents["agent_0"]')
                            for entry in mappings))
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], original["runtime"])

    def test_shared_frame_fields_preserve_dynamic_goals_positions_action_age_and_events(self):
        module = compaction_module(self)
        original = {"runtime": {"recent_frames": [
            {"step": 4, "assignments": {"agent_0": "Scout"},
             "objects": {"cup": {"position": [1, 2, 3]}},
             "camera_sensors": ["rgb", "semantic"],
             "agents": {"agent_0": {"position": [0, 0, 0], "action": {"name": "move", "age": 2}}},
             "current_goal": {"any_at": {"cup": False}, "is_detected": {"cup": None}},
             "decisions": [{"step": 4, "tool": "move"}], "messages": []},
            {"step": 19, "assignments": {"agent_0": "Scout"},
             "objects": {"cup": {"position": [1, 2, 3]}},
             "camera_sensors": ["rgb", "semantic"],
             "agents": {"agent_0": {"position": [0.01, 0, 0], "action": {"name": "move", "age": 17}}},
             "current_goal": {"any_at": {"cup": None}, "is_detected": {"cup": False}},
             "decisions": [], "messages": [{"step": 19, "content": "arrived"}]}]}}
        compact = module.compact_runtime_context(original)
        self.assertEqual(compact["runtime"]["shared_frame_fields"], {
            "assignments": {"agent_0": "Scout"}, "objects": {"cup": {"position": [1, 2, 3]}},
            "camera_sensors": ["rgb", "semantic"]})
        frames = compact["runtime"]["recent_frames"]
        self.assertEqual([frame["step"] for frame in frames], [4, 19])
        self.assertEqual([frame["agents"]["agent_0"]["action"]["age"] for frame in frames], [2, 17])
        self.assertEqual(frames[0]["current_goal"]["any_at"]["cup"], False)
        self.assertIsNone(frames[1]["current_goal"]["any_at"]["cup"])
        self.assertEqual(frames[1]["agents"]["agent_0"]["position"], [0.01, 0, 0])
        self.assertEqual(frames[1]["messages"], [{"step": 19, "content": "arrived"}])
        shared = compact["input_layout"]["runtime_compaction"]["shared_frame_fields"]
        self.assertEqual(shared["objects"]["steps"], [4, 19])
        self.assertEqual(shared["objects"]["omitted_paths"], [
            "$.runtime.recent_frames[0].objects", "$.runtime.recent_frames[1].objects"])
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], original["runtime"])

    def test_missing_null_false_and_different_numeric_types_are_never_conflated(self):
        module = compaction_module(self)
        original = {"task": {"description": None, "assignments": {"agent_0": False}},
                    "planner_plan": {"agent_0": None}, "recent_history": {"agent_0": []},
                    "runtime": {"current_frame": {"step": 8, "objects": {"flag": False}},
                                "recent_frames": [{"step": 8, "objects": {"flag": 0}},
                                                  {"step": 10, "objects": {"flag": None}},
                                                  {"step": 12}],
                                "agents": {"agent_0": {"task_description": None,
                                            "subtask_description": 0, "planner_plan": None,
                                            "recent_history": None}, "agent_1": {}}}}
        self.assertEqual(module.compact_runtime_context(original), original)
        numbers = {"runtime": {"current_frame": {"step": 1, "objects": {"x": 1.0}},
                               "recent_frames": [{"step": 1, "objects": {"x": 1}}]}}
        self.assertEqual(module.compact_runtime_context(numbers), numbers)

    def test_distinct_current_frame_and_old_schemas_are_preserved_without_mutation(self):
        module = compaction_module(self)
        for original in ({}, {"runtime": {}}, {"runtime": {"agents": None, "recent_frames": None}},
                         {"runtime": {"current_frame": None, "recent_frames": [None]}},
                         {"runtime": {"current_frame": {"step": 9}, "recent_frames": [{"step": 8}]}}):
            before = deepcopy(original)
            compact = module.compact_runtime_context(original)
            self.assertEqual(original, before)
            self.assertEqual(compact, original)
            self.assertIsNot(compact, original)
            self.assertEqual(module.expand_runtime_context(compact), original)

    def test_both_directions_deepcopy_and_preserve_existing_layout(self):
        module = compaction_module(self)
        original = {"input_layout": {"instructions": ["retain source ordering"]},
                    "runtime": {"current_frame": {"step": 7, "objects": {"cup": {"position": [1, 2, 3]}}},
                                "recent_frames": [{"step": 7, "objects": {"cup": {"position": [1, 2, 3]}}}]}}
        before = deepcopy(original)
        compact = module.compact_runtime_context(original)
        compact_before = deepcopy(compact)
        expanded = module.expand_runtime_context(compact)
        self.assertEqual(original, before)
        self.assertEqual(compact, compact_before)
        self.assertEqual(expanded["runtime"], before["runtime"])
        expanded["runtime"]["current_frame"]["objects"]["cup"]["position"][0] = 99
        compact["input_layout"]["instructions"].append("separate copy")
        self.assertEqual(original, before)
        self.assertEqual(compact["runtime"]["recent_frames"][0]["objects"]["cup"]["position"], [1, 2, 3])
        self.assertEqual(module.compact_runtime_context(compact), compact)
        self.assertEqual(json.loads(json.dumps(module.expand_runtime_context(compact)))["runtime"],
                         before["runtime"])

    def test_nonfinal_current_frame_and_shared_values_round_trip_together(self):
        module = compaction_module(self)
        first = {"step": 3, "objects": {"cup": {"position": [1, 2, 3]}},
                 "assignments": {"agent_0": "Scout"}, "current_goal": {"detected": False}}
        last = {"step": 11, "objects": {"cup": {"position": [1, 2, 3]}},
                "assignments": {"agent_0": "Scout"}, "current_goal": {"detected": None}}
        original = {"runtime": {"current_frame": deepcopy(first), "recent_frames": [first, last]}}
        compact = module.compact_runtime_context(original)
        self.assertEqual(compact["input_layout"]["runtime_compaction"]["current_frame"]["recent_frames_index"], 0)
        self.assertNotIn("objects", compact["runtime"]["recent_frames"][0])
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], original["runtime"])

    def test_shared_fields_include_a_distinct_current_frame_but_never_fill_missing_fields(self):
        module = compaction_module(self)
        original = {"runtime": {
            "current_frame": {"step": 8, "assignments": {"agent_0": "Scout"}, "objects": None},
            "recent_frames": [{"step": 1, "assignments": {"agent_0": "Scout"}, "objects": None},
                              {"step": 5, "assignments": {"agent_0": "Scout"}}]}}
        compact = module.compact_runtime_context(original)
        self.assertEqual(compact["runtime"]["shared_frame_fields"], {"assignments": {"agent_0": "Scout"}})
        self.assertNotIn("objects", compact["runtime"]["recent_frames"][1])
        self.assertEqual(compact["input_layout"]["runtime_compaction"]["shared_frame_fields"]["assignments"]["steps"], [1, 5, 8])
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], original["runtime"])

    def test_shared_python_frame_aliases_keep_every_sample_and_do_not_mutate_input(self):
        module = compaction_module(self)
        frame = {"step": 5, "objects": {"cup": {"position": [1, 2, 3]}}}
        original = {"runtime": {"recent_frames": [frame, frame]}}
        before = deepcopy(original)
        compact = module.compact_runtime_context(original)
        self.assertEqual([sample["step"] for sample in compact["runtime"]["recent_frames"]], [5, 5])
        self.assertEqual(original, before)
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], before["runtime"])

    def test_aliased_agent_dictionaries_still_declare_every_omitted_field(self):
        module = compaction_module(self)
        fields = {"task_description": "Reach goal", "current_skill": "move"}
        original = {"task": {"description": "Reach goal"},
                    "runtime": {"agents": {"agent_0": fields, "agent_1": fields}}}
        compact = module.compact_runtime_context(original)
        self.assertEqual({entry["agent"] for entry in compact["input_layout"]["runtime_compaction"]["agent_fields"]},
                         {"agent_0", "agent_1"})
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], original["runtime"])

    def test_repeated_agent_camera_and_robot_metadata_share_fixed_paths_only(self):
        module = compaction_module(self)
        sensors = {"semantic": {"uuid": "agent_0_semantic", "resolution": [256, 256], "available": True,
                                "orientation": None, "normalize_depth": False}}
        frames = [{"step": step, "agents": {"agent_0": {
            "camera_sensors": deepcopy(sensors), "robot_type": "FetchRobot",
            "position": [step / 1000, 0, 0], "action": {"name": "move", "age": step},
            "stop": False, "detection": None}}, "current_goal": {"detected": False},
            "decisions": [], "messages": []} for step in (5, 20)]
        original = {"runtime": {"current_frame": deepcopy(frames[-1]), "recent_frames": frames}}
        before = deepcopy(original)
        compact = module.compact_runtime_context(original)
        self.assertEqual(compact["runtime"]["shared_frame_fields"], {
            "agents": {"agent_0": {"camera_sensors": sensors, "robot_type": "FetchRobot"}}})
        for frame in compact["runtime"]["recent_frames"]:
            self.assertNotIn("camera_sensors", frame["agents"]["agent_0"])
            self.assertNotIn("robot_type", frame["agents"]["agent_0"])
            self.assertEqual(frame["agents"]["agent_0"]["action"]["age"], frame["step"])
            self.assertIs(frame["agents"]["agent_0"]["stop"], False)
            self.assertIsNone(frame["agents"]["agent_0"]["detection"])
            self.assertIs(frame["current_goal"]["detected"], False)
        mappings = compact["input_layout"]["runtime_compaction"]["shared_frame_fields"]["agents"]["agent_0"]
        self.assertEqual(mappings["camera_sensors"]["steps"], [5, 20])
        self.assertEqual(mappings["camera_sensors"]["source_path"],
                         '$.runtime.shared_frame_fields.agents["agent_0"].camera_sensors')
        self.assertEqual(mappings["camera_sensors"]["omitted_paths"], [
            '$.runtime.recent_frames[0].agents["agent_0"].camera_sensors',
            '$.runtime.recent_frames[1].agents["agent_0"].camera_sensors'])
        self.assertEqual(original, before)
        self.assertEqual(module.expand_runtime_context(compact)["runtime"], before["runtime"])

    def test_camera_changes_missing_robot_type_and_null_agents_remain_explicit(self):
        module = compaction_module(self)
        original = {"runtime": {"recent_frames": [
            {"step": 1, "agents": {"agent_0": {"camera_sensors": {"camera": {"rotation": 0.0}},
                                              "robot_type": "FetchRobot"},
                                    "agent_1": {"camera_sensors": None}}},
            {"step": 8, "agents": {"agent_0": {"camera_sensors": {"camera": {"rotation": 0.000001}}},
                                    "agent_1": None}},
            {"step": 19, "agents": {"agent_0": {"camera_sensors": {"camera": {"rotation": 0.0}},
                                               "robot_type": None}}}]}}
        self.assertEqual(module.compact_runtime_context(original), original)


if __name__ == "__main__":
    unittest.main()
