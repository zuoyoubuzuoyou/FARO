"""Causal replay adapters, without Habitat, assets, or a model service."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from habitat_mas.detection.readers import frames_from_records, iter_frames


def old_start():
    return {"schema_version": 1, "type": "episode_start", "episode_id": "7",
            "text_context": {"pddl_text_goal": "Both objects at their targets",
                             "scene_description": "Two objects", "robot_resume": '{"agent_0":{"robot_type":"FetchRobot"}}'}}


def old_frame(step, name="nav_to_obj", decision=0, context=None):
    action = {"action": name, "arguments": {"target_obj": "any_targets|0"},
              "decision_step": decision, "simulator_step": decision}
    row = {"schema_version": 1, "type": "simulator_step", "episode_id": "7",
           "simulator_step": step, "done": True, "fault_events": [{"secret": "oracle"}],
           "metrics": {"pddl_success": 0, "object_to_goal_distance.0": 2,
                       "pddl_stage_goals.robot_at_object_0_success": 1,
                       "error_detection.system_crash": 1, "num_steps": step+1},
           "simulator_state": {"agents": {"agent_0": {"position": [step, 0, 0],
               "rotation": 0.3, "held_object": None,
               "current_high_level_action": action,
               "navigation": {"agent_0_oracle_nav_action": {"available": True, "active": True,
                   "path_found": False, "distance_to_goal": 3},
                   "agent_0_oracle_nav_coord_action": {"available": False}},
               "hang_state": {"active": True}}},
               "objects": [{"goal_name": "any_targets|0", "position": [1, 2, 3], "object_id": 23}]},
           "agent_step_actions": {"agent_0": {"action": action}}, "policy_info": {}}
    if context is not None:
        row["policy_info"]["agent_0_high_level_decision"] = {
            "action": name, "arguments": action["arguments"], "replay_context": context}
    return row


def event(kind, step, **values):
    return {"schema_version": 2, "event_type": kind, "episode_id": "7",
            "evaluation_id": "attempt-7", "env_step": step, **values}


class ReaderTests(unittest.TestCase):
    def test_old_normalizes_current_observations_and_excludes_oracles(self):
        frame, = frames_from_records([old_start(), old_frame(0)], "Mobility")
        self.assertEqual(frame["step"], 0)
        self.assertEqual(frame["task"], "Mobility")
        self.assertEqual(frame["metrics"], {"pddl_success": 0, "object_to_goal_distance.0": 2})
        self.assertEqual(frame["objects"]["any_targets|0"], {"position": [1, 2, 3], "object_id": 23})
        self.assertIs(frame["agents"]["agent_0"]["navigation"]["path_found"], False)
        text = json.dumps(frame)
        for forbidden in ["fault_events", "hang_state", "error_detection", "stage_goals", "done", "oracle"]:
            self.assertNotIn(forbidden, text)

    def test_context_and_goal_coordinates_are_detached_and_causal(self):
        goals = {"any_targets|0": [3, 4, 5]}
        records = [old_start(), old_frame(0), old_frame(1, context={
            "task_description": "Updated goal", "subtask_description": "Move object 0"})]
        stream = frames_from_records(records, "Rearrange", goals)
        first = next(stream)
        self.assertEqual(first["context"]["assignments"], {})
        self.assertEqual(first["context"]["robot_capabilities"]["agent_0"]["robot_type"], "FetchRobot")
        second = next(stream)
        self.assertEqual(second["context"]["assignments"], {"agent_0": "Move object 0"})
        self.assertEqual(first["context"]["assignments"], {})
        second["context"]["goal_positions"]["any_targets|0"][0] = 99
        self.assertEqual(goals["any_targets|0"][0], 3)
        self.assertEqual(first["context"]["goal_positions"]["any_targets|0"][0], 3)

    def test_action_identity_and_age_distinguish_reissued_same_skill(self):
        result = list(frames_from_records([old_start(), old_frame(0), old_frame(1),
            old_frame(2, decision=1), old_frame(3, "pick", decision=2)], "Manipulation"))
        actions = [r["agents"]["agent_0"]["action"] for r in result]
        self.assertEqual([a["age"] for a in actions], [1, 2, 1, 1])
        self.assertEqual([a["id"] for a in actions], [0, 0, 1, 2])
        self.assertEqual(actions[-1]["name"], "pick")

    def test_absent_diagnostics_and_unknown_action_remain_null(self):
        record = old_frame(0)
        record["simulator_state"]["agents"] = {"agent_0": {"nav_finished": False}}
        record["agent_step_actions"] = {}
        frame, = frames_from_records([old_start(), record], "Manipulation")
        agent = frame["agents"]["agent_0"]
        for key in ["navigation", "termination", "detection", "stop", "position", "rotation", "held_object"]:
            self.assertIsNone(agent[key], key)
        self.assertIs(agent["nav_finished"], False)
        self.assertIsNone(agent["action"]["name"])

    def test_original_tool_send_request_and_actual_receipt_not_mapped_wait(self):
        message = '"agent_1" agent sent you requests: "Please carry object 0".'
        context = {"subtask_description": "Ask for help", "sent_messages": [
            {"source": "agent_0", "target": "agent_1", "request": "Need help"}],
            "chat_history": [[{"role": "user", "content": message},
                {"role": "assistant", "tool_calls": [{"id": "call-1", "function": {
                    "name": "send_request", "arguments": '{"target_agent":"agent_1","request":"Need help"}'}}]},
                {"role": "tool", "content": "Success", "tool_call_id": "call-1"}]]}
        first, second = frames_from_records([old_start(), old_frame(0, "wait", context=context),
                                             old_frame(1, "wait", context=context)], "Rearrange")
        self.assertEqual(first["agents"]["agent_0"]["action"]["name"], "wait")
        self.assertEqual(first["decisions"][0]["tool"], "send_request")
        self.assertTrue(first["decisions"][0]["feedback"][-1]["placeholder"])
        self.assertEqual({m["kind"] for m in first["messages"]}, {"send", "receive"})
        self.assertEqual(second["messages"], [])

    def test_malformed_tool_arguments_preserved_without_execution(self):
        context = {"chat_history": [[{"role": "assistant", "tool_calls": [{"function": {
            "name": "pick", "arguments": "not valid json"}}]}]]}
        row, = frames_from_records([old_start(), old_frame(0, context=context)], "Manipulation")
        self.assertEqual(row["decisions"][0]["arguments"], "not valid json")

    def test_schema_two_active_actions_and_real_recorded_state_fields(self):
        records = [event("episode_start", 0, context={"pddl_text_goal": "Place 0"}),
            event("task_assignment", 1, agent_id="agent_0", subtask_description="Move 0"),
            event("action_start", 1, agent_id="agent_0", action_id="move-1",
                  action_name="nav_to_obj", parameters={"target_obj": "any_targets|0"}, started_step=1),
            event("step", 1, done=False, metrics={"pddl_success": 0, "error_detection.enabled": 1}, state={
                "agents": {"agent_0": {"base_position": [1, 2, 3], "base_rotation": [0, 0, 0, 1], "grasped_object_id": 9}},
                "target_objects": [{"position": [2, 3, 4], "object_id": 9, "goal_position": [5, 6, 7]}]}),
            event("step", 2, state={"agents": {"agent_0": {"base_position": [2, 2, 3]}}}, metrics={}),
            event("action_end", 2, agent_id="agent_0", action_id="move-1", status="timeout", reason="max_skill_steps"),
            event("action_start", 3, agent_id="agent_0", action_id="pick-2", action_name="pick", parameters=[]),
            event("step", 3, state={"agents": {"agent_0": {}}}, metrics={}),
            event("episode_end", 3, task_success=False, final_state={"future": "forbidden"})]
        frames = list(frames_from_records(records, "Manipulation"))
        self.assertEqual(len(frames), 3)
        self.assertEqual([f["step"] for f in frames], [1, 2, 3])
        self.assertEqual(frames[0]["agents"]["agent_0"]["position"], [1, 2, 3])
        self.assertEqual(frames[0]["agents"]["agent_0"]["held_object"], 9)
        self.assertEqual(frames[0]["agents"]["agent_0"]["rotation"], [0, 0, 0, 1])
        self.assertEqual(frames[1]["agents"]["agent_0"]["action"]["age"], 2)
        self.assertEqual(frames[2]["agents"]["agent_0"]["action"]["id"], "pick-2")
        self.assertEqual(frames[0]["objects"]["any_targets|0"]["object_id"], 9)
        self.assertEqual(frames[0]["context"]["goal_positions"]["any_targets|0"], [5, 6, 7])
        self.assertNotIn("future", json.dumps(frames))

    def test_schema_two_message_events_are_flushed_only_at_next_environment_frame(self):
        records = [event("episode_start", 0, context={}),
            event("agent_tool_decision", 1, agent_id="agent_0", action_name="send_request", parameters={"request": "Help"}),
            event("communication_send", 1, sender="agent_0", recipient="agent_1", message="Help", status="queued"),
            event("communication_receive", 1, sender="agent_0", recipient="agent_1", message="Help", status="consumed"),
            event("step", 1, state={"agents": {}}, metrics={}),
            event("communication_send", 2, sender="agent_0", recipient="agent_0", message="Ignored", status="suppressed"),
            event("step", 2, state={"agents": {}}, metrics={})]
        one, two = frames_from_records(records, "Rearrange")
        self.assertEqual([m["kind"] for m in one["messages"]], ["send", "receive"])
        self.assertEqual(one["decisions"][0]["tool"], "send_request")
        self.assertEqual(two["messages"], [])

    def test_schema_two_real_model_event_order_preserves_placeholder_feedback(self):
        # models.py writes placeholders before tool_selection; CrabAgent then
        # writes agent_tool_decision. None of those merges may erase feedback.
        records = [event("episode_start", 0, context={}),
            event("tool_response_placeholder", 1, agent_id="agent_0", phase="execution",
                  action_name="pick", tool_call_id="call-pick", content="Success"),
            event("tool_response_placeholder", 1, agent_id="agent_1", phase="execution",
                  action_name="wait", tool_call_id="call-wait", content="Success"),
            event("tool_selection", 1, agent_id="agent_0", phase="execution",
                  action_name="pick", parameters={"target_obj": "any_targets|0"}),
            event("agent_tool_decision", 1, agent_id="agent_0", phase="execution",
                  action_name="pick", parameters={"target_obj": "any_targets|0"}),
            event("tool_selection", 1, agent_id="agent_1", phase="execution",
                  action_name="wait", parameters={}),
            event("agent_tool_decision", 1, agent_id="agent_1", phase="execution",
                  action_name="wait", parameters={}),
            event("step", 1, state={"agents": {}}, metrics={}),
            event("step", 2, state={"agents": {}}, metrics={})]
        first, second = frames_from_records(records, "Manipulation")
        decisions = {d["agent"]:d for d in first["decisions"]}
        self.assertEqual(decisions["agent_0"]["tool"], "pick")
        self.assertEqual(decisions["agent_0"]["arguments"], {"target_obj":"any_targets|0"})
        self.assertEqual(decisions["agent_0"]["feedback"], [{"role":"tool", "content":"Success",
            "tool_call_id":"call-pick", "placeholder":True}])
        self.assertEqual(decisions["agent_1"]["feedback"][0]["tool_call_id"], "call-wait")
        self.assertEqual(second["decisions"], [])

    def test_schema_two_late_agent_decision_keeps_existing_tool_feedback(self):
        records = [event("episode_start", 0, context={}),
            event("tool_selection", 1, agent_id="agent_0", action_name="pick", parameters={}),
            event("tool_response_placeholder", 1, agent_id="agent_0", content="Success", tool_call_id="first"),
            event("tool_response_placeholder", 1, agent_id="agent_0", content="Didn't execute because only the first tool call is executed", tool_call_id="second"),
            event("agent_tool_decision", 1, agent_id="agent_0", action_name="pick", parameters={}),
            event("step", 1, state={"agents": {}}, metrics={})]
        result, = frames_from_records(records, "Manipulation")
        self.assertEqual([f["tool_call_id"] for f in result["decisions"][0]["feedback"]], ["first", "second"])
        self.assertTrue(all(f["placeholder"] for f in result["decisions"][0]["feedback"]))

    def test_schema_two_feedback_does_not_wait_for_future_selection(self):
        records = [event("episode_start", 0, context={}),
            event("tool_response_placeholder", 1, agent_id="agent_0", action_name="pick", content="Success", tool_call_id="old"),
            event("step", 1, state={"agents": {}}, metrics={}),
            event("tool_selection", 2, agent_id="agent_0", action_name="wait", parameters={}),
            event("step", 2, state={"agents": {}}, metrics={})]
        first, second = frames_from_records(records, "Manipulation")
        self.assertEqual(first["decisions"][0]["feedback"][0]["tool_call_id"], "old")
        self.assertEqual(second["decisions"][0]["feedback"], [])
        self.assertEqual(second["decisions"][0]["tool"], "wait")

    def test_schema_two_lower_priority_selection_cannot_replace_agent_decision(self):
        records = [event("episode_start", 0, context={}),
            event("agent_tool_decision", 1, agent_id="agent_0", action_name="pick",
                  parameters={"target_obj":"any_targets|0"}),
            event("tool_response_placeholder", 1, agent_id="agent_0", content="Success", tool_call_id="pick"),
            event("tool_selection", 1, agent_id="agent_0", action_name="pick", parameters="unparsed"),
            event("step", 1, state={"agents": {}}, metrics={})]
        result, = frames_from_records(records, "Manipulation")
        self.assertEqual(result["decisions"][0]["arguments"], {"target_obj":"any_targets|0"})
        self.assertEqual(result["decisions"][0]["feedback"][0]["tool_call_id"], "pick")

    def test_episode_reset_clears_action_and_context_before_next_frame(self):
        start = old_start()
        second_start = copy.deepcopy(start)
        second_start.update(episode_id="8", text_context={"scene_description": "Other scene"})
        second = old_frame(0)
        second["episode_id"] = "8"
        a, b = frames_from_records([start, old_frame(0, context={"subtask_description": "First only"}),
                                    second_start, second], "Mobility")
        self.assertEqual(b["context"]["assignments"], {})
        self.assertEqual(b["context"]["scene_description"], "Other scene")
        self.assertIsNone(b["context"]["task_description"])

    def test_prefix_invariance_and_file_streaming_do_not_read_future(self):
        def source():
            yield old_start()
            yield old_frame(0)
            raise RuntimeError("reader advanced past requested observation")
        first = next(frames_from_records(source(), "Perception"))
        self.assertEqual(first["step"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"episode.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in [old_start(), old_frame(0)]))
            self.assertEqual(list(iter_frames(path, "Perception")), [first])

    def test_bad_json_reports_source_path_and_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"bad.jsonl"
            path.write_text(json.dumps(old_start())+"\nnot json\n")
            with self.assertRaisesRegex(ValueError, "bad.jsonl.*2"):
                list(iter_frames(path, "Mobility"))


if __name__ == "__main__":
    unittest.main()
