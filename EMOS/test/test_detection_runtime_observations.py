"""Native evidence stays current, causal and distinct from base-pose proxies."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "habitat-mas"))
from habitat_mas.detection.heuristics import assess_goals, TeamProgressDetector
from habitat_mas.detection.llm import LLMDetector
from habitat_mas.detection.readers import frames_from_records


def goal(step, covered=False, detected=False):
    return {"env_step": step, "timing": "post_env_step", "source": "runtime_observation",
            "any_at": {"any_targets|0": covered, "any_targets|1": True},
            "is_detected": {"any_targets|0": detected, "any_targets|1": True}}


def event(kind, step, **fields):
    return {"event_type": kind, "env_step": step, **fields}


def state(step, **updates):
    value = {"current_goal": goal(step), "agents": {"agent_0": {
        "base_position": [0, 0, 0], "base_rotation": [0, 0, 0, 1],
        "robot_type": "Fetch", "ee_position": [0, 1, 0],
        "root_position": [0, 0.5, 0], "base_reference_position": [0, 0.5, 0],
        "goal_coverage": {"env_step": step, "timing": "post_env_step",
            "reference": "base_transformation", "distance_threshold": 1.0,
            "targets": {"any_targets|0": {"distance": 1.1, "at_goal": False}}},
        "detection": {"env_step": step, "timing": "post_env_step", "available": True,
            "eligible": True, "source": "returned_observations", "reason": None,
            "target_detections": {"any_targets|0": True, "any_targets|1": True}},
        "navigation": {"env_step": step, "timing": "pre_control", "active": True,
                       "available": True, "distance_to_goal": 1.1}}},
        "objects": {"any_targets|0": {"position": [0, 0, 0]},
                    "any_targets|1": {"position": [0, 0, 0]}}}
    value.update(updates)
    return value


def normalized(step=1, raw=None):
    return next(frames_from_records([
        event("task_assignment", 0, agent_id="agent_0",
              subtask_description="Cover and detect any_targets|0 and any_targets|1."),
        event("action_start", 0, agent_id="agent_0", action_id="wait-1", action_name="wait"),
        event("step", step, state=raw or state(step), metrics={})], "Perception"))


def satisfied(frame, kind="coverage"):
    return next(r for r in assess_goals(frame)
                if r["kind"] == kind and r["object"] == "any_targets|0")


class RuntimeObservationTests(unittest.TestCase):
    def test_reader_preserves_native_metrics_and_reference_metadata(self):
        frame = normalized()
        self.assertIs(frame["metrics"].get("current_goal.any_at.any_targets|0"), False)
        self.assertEqual(frame.get("current_goal"), goal(1))
        agent = frame["agents"]["agent_0"]
        self.assertEqual(agent.get("robot_type"), "Fetch")
        self.assertEqual(agent.get("ee_position"), [0, 1, 0])
        self.assertEqual(agent.get("root_position"), [0, 0.5, 0])
        self.assertEqual(agent.get("base_reference_position"), [0, 0.5, 0])
        self.assertEqual(agent.get("goal_coverage"), state(1)["agents"]["agent_0"]["goal_coverage"])
        self.assertEqual(agent["navigation"].get("env_step"), 1)

    def test_native_bool_or_null_takes_precedence_over_base_and_detection_proxy(self):
        for value in (False, True, None):
            with self.subTest(value=value):
                raw = state(1, current_goal=goal(1, value, value))
                # Both old approximations disagree with native false/null.
                frame = normalized(raw=raw)
                self.assertIs(satisfied(frame)["satisfied"], value)
                self.assertIs(satisfied(frame, "detection")["satisfied"], value)
                self.assertFalse(satisfied(frame)["approximate"])
                self.assertEqual(satisfied(frame)["distance"], 1.1)

    def test_native_true_overrides_distant_base_proxy(self):
        raw = state(1, current_goal=goal(1, True, True))
        raw["agents"]["agent_0"]["base_position"] = [100, 0, 0]
        self.assertIs(satisfied(normalized(raw=raw))["satisfied"], True)

    def test_current_detection_loss_never_reuses_previous_true(self):
        frames = list(frames_from_records([
            event("step", 1, state=state(1, current_goal=goal(1, True, True))),
            event("step", 2, state=state(2, current_goal=goal(2, True, False)))], "Perception"))
        self.assertIs(satisfied(frames[0], "detection")["satisfied"], True)
        self.assertIs(satisfied(frames[1], "detection")["satisfied"], False)

    def test_invalid_native_timestamp_timing_source_or_value_is_unknown(self):
        for update in ({"env_step": 0}, {"env_step": 2}, {"env_step": True},
                       {"timing": "pre_control"}, {"source": None},
                       {"any_at": {"any_targets|0": 1}}):
            with self.subTest(update=update):
                native = goal(1, True, True)
                native.update(update)
                frame = normalized(raw=state(1, current_goal=native))
                self.assertIsNone(satisfied(frame)["satisfied"])

    def test_stale_navigation_and_detection_are_unavailable(self):
        for recorded_step in (0, 2):
            raw = state(1)
            raw.pop("current_goal")
            raw["agents"]["agent_0"]["navigation"]["env_step"] = recorded_step
            raw["agents"]["agent_0"]["detection"]["env_step"] = recorded_step
            frame = normalized(raw=raw)
            self.assertIsNot(frame["agents"]["agent_0"]["navigation"].get("active"), True)
            self.assertIsNone(satisfied(frame, "detection")["satisfied"])

    def test_native_false_drives_end_to_end_wait_alarm(self):
        records = [event("task_assignment", 0, agent_id="agent_0",
                         subtask_description="Cover and detect any_targets|0."),
                   event("action_start", 0, agent_id="agent_0", action_id="w", action_name="wait")]
        records += [event("step", i, state=state(i)) for i in (1, 51, 101)]
        detector = TeamProgressDetector(window=3, check_every=1, min_stall_steps=100)
        alarms = [alarm for frame in frames_from_records(records, "Perception")
                  if (alarm := detector.observe(frame)) is not None]
        self.assertEqual([a["reason_code"] for a in alarms], ["unresolved_team_wait"])

    def test_llm_request_keeps_native_evidence_and_filters_forbidden_nested_fields(self):
        frame = normalized()
        frame["current_goal"] = {**goal(1), "labels": {"secret_label": "forbidden"}}
        frame["agents"]["agent_0"]["goal_coverage"] = {
            **state(1)["agents"]["agent_0"]["goal_coverage"], "fault": "forbidden"}
        frame["metrics"]["pddl_stage_goals.current_goal.any_at.any_targets|0"] = True
        detector = LLMDetector(window=1, max_calls=0)
        detector.observe(frame)
        payload = json.loads(detector._request()["messages"][1]["content"])["recent_frames"][0]
        self.assertEqual(payload.get("current_goal"), goal(1))
        agent = payload["agents"]["agent_0"]
        self.assertEqual(agent["goal_coverage"]["reference"], "base_transformation")
        self.assertEqual(agent["detection"]["env_step"], 1)
        self.assertEqual(agent["detection"]["source"], "returned_observations")
        self.assertEqual(agent["navigation"]["env_step"], 1)
        self.assertNotIn("forbidden", json.dumps(payload))
        self.assertNotIn("pddl_stage_goals", json.dumps(payload))

    def test_stale_coverage_distance_cannot_reach_llm_as_current_evidence(self):
        raw = state(1)
        raw["agents"]["agent_0"]["goal_coverage"]["env_step"] = 2
        frame = normalized(raw=raw)
        detector = LLMDetector(window=1, max_calls=0)
        detector.observe(frame)
        payload = json.loads(detector._request()["messages"][1]["content"])["recent_frames"][0]
        target = payload["agents"]["agent_0"]["goal_coverage"]["targets"]["any_targets|0"]
        self.assertIsNone(target["distance"])
        self.assertIsNone(target["at_goal"])

    def test_structured_outcome_and_real_request_feedback_are_causal_and_sanitized(self):
        end_state = state(1)
        end_state["task_progress"] = {"pddl_stage_goals.detected": True}
        end_state["agents"]["agent_0"]["fault"] = "hidden_fault"
        records = [event("step", 1, state=state(1)),
            event("action_outcome", 2, agent_id="agent_0", action_id="nav-1",
                action_name="nav_to_obj", status="completed", reason="controller_finished",
                controller_result={"is_skill_done": True, "label": "hidden_label",
                                   "pddl_stage_goals": {"latched_detection": True}},
                termination_signal_step=2, observation_env_step=1,
                actual_execution_observed=True, is_placeholder=False, state=end_state,
                metrics={"pddl_success": 0, "pddl_stage_goals.detected": True},
                task_success=False),
            event("execution_feedback", 2, agent_id="agent_0", phase="execution",
                call_id="query-2", preceding_action_id="nav-1", observation_env_step=1,
                outcome_env_step=2, actual_execution_observed=True,
                feedback_source="model_request_messages",
                model_feedback={"tool_messages": [{"tool_call_id": "call-1", "name": "nav_to_obj",
                    "content": "Success", "is_placeholder": True}], "user_message": "Continue"},
                preceding_outcome={"action_id": "nav-1", "observation_env_step": 1,
                    "status": "completed", "reason": "controller_finished", "actual_execution_observed": True}),
            event("step", 2, state=state(2))]
        first, second = frames_from_records(records, "Perception")
        self.assertEqual(first["decisions"], [])
        feedback = [item for decision in second["decisions"] for item in decision["feedback"]]
        self.assertEqual([item.get("kind") for item in feedback], ["action_outcome", "execution_feedback"])
        self.assertEqual(feedback[0]["action_id"], "nav-1")
        self.assertEqual(feedback[0]["observation_env_step"], 1)
        self.assertEqual(feedback[0]["state"]["current_goal"], goal(1))
        self.assertEqual(feedback[1]["preceding_action_id"], "nav-1")
        self.assertTrue(feedback[1]["model_feedback"]["tool_messages"][0]["is_placeholder"])
        detector = LLMDetector(window=1, max_calls=0)
        detector.observe(second)
        payload = json.loads(detector._request()["messages"][1]["content"])["recent_frames"][0]
        self.assertEqual(len(payload["decisions"]), 2)
        serialized = json.dumps(payload)
        for forbidden in ("hidden_fault", "hidden_label", "pddl_stage_goals", "task_success"):
            self.assertNotIn(forbidden, serialized)

    def test_future_feedback_waits_until_its_observation_and_event_steps(self):
        records = [event("action_outcome", 3, agent_id="agent_0", action_id="future",
                         observation_env_step=3, state=state(3), actual_execution_observed=True),
                   event("step", 1, state=state(1)), event("step", 2, state=state(2)),
                   event("step", 3, state=state(3))]
        frames = list(frames_from_records(records, "Perception"))
        self.assertEqual([len(f["decisions"]) for f in frames], [0, 0, 1])
        self.assertEqual(frames[2]["decisions"][0]["feedback"][0]["action_id"], "future")

    def test_native_reference_distance_progress_prevents_stationary_base_alarm(self):
        detector = TeamProgressDetector(window=3, check_every=1, min_stall_steps=100)
        for step, distance in ((1, 5.0), (51, 4.0), (101, 3.0)):
            raw = state(step)
            raw["agents"]["agent_0"]["goal_coverage"]["targets"]["any_targets|0"]["distance"] = distance
            self.assertIsNone(detector.observe(normalized(step, raw)))

    def test_superseded_detection_proxy_does_not_hide_native_stall(self):
        detector = TeamProgressDetector(window=3, check_every=1, min_stall_steps=100)
        alarms = []
        for step, proxy in ((1, False), (51, True), (101, True)):
            raw = state(step)
            raw["agents"]["agent_0"]["detection"]["target_detections"]["any_targets|0"] = proxy
            alarm = detector.observe(normalized(step, raw))
            if alarm:
                alarms.append(alarm)
        self.assertEqual([a["reason_code"] for a in alarms], ["unresolved_team_wait"])

    def test_sensor_and_missing_reason_metadata_survives_reader_to_llm(self):
        raw = state(1)
        raw["current_goal"]["reasons"] = {"is_detected": {"any_targets|0": "sensor_missing"}}
        agent = raw["agents"]["agent_0"]
        agent["root_position_reason"] = "read_only_articulated_root"
        agent["navigation"].update(source="oracle_nav_coordinate", action="agent_0_oracle_nav_coord_action")
        agent["camera_sensors"] = {"head_panoptic": {"uuid": "head_panoptic", "height": 256, "width": 256}}
        agent["detection"].update(target_reasons={"any_targets|0": "not_visible"},
            semantic_sensors=["head_panoptic"], target_semantic_ids={"any_targets|0": 101},
            target_pixel_counts={"any_targets|0": {"head_panoptic": 0}},
            target_pixel_counts_by_sensor={"head_panoptic": {"any_targets|0": 0}})
        frame = normalized(raw=raw)
        detector = LLMDetector(window=1, max_calls=0)
        detector.observe(frame)
        payload = json.loads(detector._request()["messages"][1]["content"])["recent_frames"][0]
        self.assertEqual(payload["current_goal"].get("reasons"), {"is_detected": {"any_targets|0": "sensor_missing"}})
        actual = payload["agents"]["agent_0"]
        self.assertEqual(actual.get("camera_sensors"), agent["camera_sensors"])
        self.assertEqual(actual.get("root_position_reason"), "read_only_articulated_root")
        self.assertEqual(actual["navigation"].get("source"), "oracle_nav_coordinate")
        self.assertEqual(actual["navigation"].get("action"), "agent_0_oracle_nav_coord_action")
        for key in ("target_reasons", "semantic_sensors", "target_semantic_ids",
                    "target_pixel_counts", "target_pixel_counts_by_sensor"):
            self.assertEqual(actual["detection"].get(key), agent["detection"][key])

    def test_invalid_diagnostic_stamp_clears_every_dynamic_field_in_llm_payload(self):
        for stamp in (0, 2):
            with self.subTest(stamp=stamp):
                raw = state(1)
                nav = raw["agents"]["agent_0"]["navigation"]
                nav.update(env_step=stamp, control=[0.4, 0.2], skill_done=True, path_found=True,
                    target=[9, 8, 7], final_navigation_position=[9, 8, 7], proxy_point=[9, 8, 7],
                    robot_position_before_control=[6, 5, 4], angle_to_object=0.3)
                detection = raw["agents"]["agent_0"]["detection"]
                detection.update(env_step=stamp, detected_object_ids=[101],
                    target_pixel_counts_by_sensor={"head_panoptic": {"any_targets|0": 500}},
                    target_pixel_counts={"any_targets|0": {"head_panoptic": 500}},
                    all_targets_detected=True)
                frame = normalized(raw=raw)
                detector = LLMDetector(window=1, max_calls=0)
                detector.observe(frame)
                payload = json.loads(detector._request()["messages"][1]["content"])["recent_frames"][0]
                agent = payload["agents"]["agent_0"]
                for key in ("control", "skill_done", "path_found", "target", "final_navigation_position",
                            "proxy_point", "robot_position_before_control", "angle_to_object"):
                    self.assertIsNone(agent["navigation"].get(key), key)
                for key in ("detected_object_ids", "target_pixel_counts_by_sensor", "all_targets_detected"):
                    self.assertIsNone(agent["detection"].get(key), key)
                self.assertEqual(agent["detection"]["target_pixel_counts"], {"any_targets|0": None})

    def test_aggregate_success_never_overrides_explicit_native_false_or_null(self):
        for value in (False, None):
            with self.subTest(value=value):
                frame = normalized(raw=state(1, current_goal=goal(1, value, value)))
                frame["metrics"]["pddl_success"] = 1
                self.assertIs(satisfied(frame)["satisfied"], value)
                self.assertIs(satisfied(frame, "detection")["satisfied"], value)

    def test_native_false_wait_stall_is_not_suppressed_by_aggregate_success(self):
        detector = TeamProgressDetector(window=3, check_every=1, min_stall_steps=100)
        alarms = []
        for step in (1, 51, 101):
            frame = normalized(step)
            frame["metrics"]["pddl_success"] = 1
            if (alarm := detector.observe(frame)) is not None:
                alarms.append(alarm)
        self.assertEqual([a["reason_code"] for a in alarms], ["unresolved_team_wait"])

    def test_direct_llm_indexed_navigation_also_rejects_future_evidence(self):
        frame = normalized()
        frame["agents"]["agent_0"]["navigation"] = {"agent_0_oracle_nav_action": {
            "env_step": 2, "timing": "pre_control", "active": True, "available": True,
            "control": [0.4, 0.2], "skill_done": True, "distance_to_goal": 0.1}}
        detector = LLMDetector(window=1, max_calls=0)
        detector.observe(frame)
        payload = json.loads(detector._request()["messages"][1]["content"])["recent_frames"][0]
        navigation = payload["agents"]["agent_0"]["navigation"]["agent_0_oracle_nav_action"]
        self.assertIsNone(navigation.get("control"))
        self.assertIsNone(navigation.get("skill_done"))
        self.assertIsNone(navigation.get("distance_to_goal"))


if __name__ == "__main__":
    unittest.main()
