"""Recovery contracts, independent of a simulator or a live LLM service."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from habitat_mas.error_detection import DetectedSystemCrash, ErrorDetector
from habitat_mas.error_recovery import DropoutRecovery, RecoveryError, build_recovery_plan, snapshot_goals
from habitat_mas.agents.message_bus import AgentMessageBus
from habitat_mas.test import test_agent_hang_policy


def goal(index, complete=False):
    return dict(id=f"is_detected(any_targets|{index})", entities=[f"any_targets|{index}"],
                complete=complete)


class RecoveryPlanTest(unittest.TestCase):
    def test_transfer_only_unfinished_goals_and_keep_survivor_work(self):
        plan = build_recovery_plan(
            [goal(0), goal(1, True), goal(2)],
            {"agent_0": "Detect any_targets|0", "agent_1": "Detect any_targets|1 and any_targets|2"},
            ["agent_0"], ["agent_1"], {},
        )
        self.assertEqual(plan["assignments"], {
            "agent_0": ["is_detected(any_targets|0)", "is_detected(any_targets|2)"]})
        self.assertEqual(plan["unfinished_tasks"], ["is_detected(any_targets|2)"])
        self.assertEqual(plan["completed_tasks"], ["is_detected(any_targets|1)"])

    def test_exact_entity_matching_and_deterministic_load_balancing(self):
        plan = build_recovery_plan(
            [goal(1), goal(10), goal(2)],
            {"agent_0": "Detect any_targets|10", "agent_1": "Nothing to do",
             "agent_2": "Detect any_targets|1 and any_targets|2"},
            ["agent_0", "agent_1"], ["agent_2"], {},
        )
        self.assertEqual(plan["assignments"], {
            "agent_0": ["is_detected(any_targets|10)", "is_detected(any_targets|2)"],
            "agent_1": ["is_detected(any_targets|1)"]})

    def test_missing_initial_assignment_is_explicit_and_does_not_lose_goals(self):
        plan = build_recovery_plan([goal(0)], {}, ["agent_1"], ["agent_0"], {})
        self.assertEqual(plan["unassigned_tasks"], ["is_detected(any_targets|0)"])
        self.assertEqual(plan["assignments"], {"agent_1": ["is_detected(any_targets|0)"]})

    def test_ambiguous_free_text_ownership_is_explicit(self):
        plan = build_recovery_plan([goal(0)],
            {"agent_0": "Detect any_targets|0", "agent_1": "Help with any_targets|0"},
            ["agent_0"], ["agent_1"], {})
        self.assertEqual(plan["unassigned_tasks"], ["is_detected(any_targets|0)"])
        self.assertEqual(plan["previous_owners"]["is_detected(any_targets|0)"], ["agent_0", "agent_1"])

    def test_agent_bound_goal_cannot_be_silently_rewritten(self):
        bound = dict(id="detected_object(any_targets|0,agent_1)",
                     entities=["any_targets|0", "agent_1"], complete=False)
        with self.assertRaisesRegex(RecoveryError, "agent_1"):
            build_recovery_plan([bound], {}, ["agent_0"], ["agent_1"], {})

    def test_surviving_agent_bound_goal_is_assigned_to_required_agent(self):
        bound = dict(id="detected_object(any_targets|0,agent_1)",
                     entities=["any_targets|0", "agent_1"], complete=False)
        plan = build_recovery_plan([bound], {}, ["agent_0", "agent_1"], ["agent_2"], {})
        self.assertEqual(plan["assignments"]["agent_1"], [bound["id"]])

    def test_frozen_held_object_is_reported_as_unrecoverable(self):
        state = {"agents": {"agent_1": {"held_object": {"goal_name": "any_targets|0"}}}}
        with self.assertRaisesRegex(RecoveryError, "held"):
            build_recovery_plan([goal(0)], {}, ["agent_0"], ["agent_1"], state)

    def test_no_survivors_fails(self):
        with self.assertRaisesRegex(RecoveryError, "online"):
            build_recovery_plan([goal(0)], {}, [], ["agent_0"], {})

    def test_empty_pending_set_does_not_reassign_completed_work(self):
        plan = build_recovery_plan([goal(0, True)], {}, ["agent_0"], ["agent_1"], {})
        self.assertEqual(plan["assignments"], {"agent_0": []})

    def test_balance_accounts_for_goals_survivor_must_keep_satisfied(self):
        goals = [dict(id=f"any_at(any_targets|{i})", entities=[f"any_targets|{i}"], complete=i == 0)
                 for i in range(2)]
        plan = build_recovery_plan(goals,
            {"agent_0": "Stay at any_targets|0", "agent_1": "Go to any_targets|1"},
            ["agent_0", "agent_2"], ["agent_1"], {})
        self.assertEqual(plan["assignments"], {"agent_0": [], "agent_2": ["any_at(any_targets|1)"]})
        self.assertEqual(plan["preserved_tasks"]["agent_0"], ["any_at(any_targets|0)"])


class GoalSnapshotTest(unittest.TestCase):
    def test_reads_each_predicate_even_after_an_unsatisfied_first_goal(self):
        predicates = [SimpleNamespace(compact_str=f"is_detected(any_targets|{i})",
                         _arg_values=[SimpleNamespace(name=f"any_targets|{i}")],
                         is_true=lambda info, value=value: value)
                      for i, value in enumerate([False, True])]
        problem = SimpleNamespace(goal=SimpleNamespace(expr_type=SimpleNamespace(value="and"),
                                  sub_exprs=predicates, quantifier=None), sim_info=object())
        self.assertEqual(snapshot_goals(problem), [goal(0), goal(1, True)])

    def test_disjunction_is_not_flattened_into_mandatory_tasks(self):
        problem = SimpleNamespace(goal=SimpleNamespace(expr_type=SimpleNamespace(value="or"),
                                  sub_exprs=[], quantifier=None), sim_info=object())
        with self.assertRaisesRegex(RecoveryError, "conjunction"):
            snapshot_goals(problem)


class RecoveryWatchdogTest(unittest.TestCase):
    def test_acknowledged_dropout_does_not_retrigger_or_become_a_hang(self):
        detector = ErrorDetector()
        detector.reset("scene", "20", ["agent_0", "agent_1"])
        health = {"agent_0": {"online": True, "responsive": True},
                  "agent_1": {"online": False, "responsive": False}}
        with self.assertRaises(DetectedSystemCrash) as caught:
            detector.before_actions(0, health)
        detector.acknowledge_dropout(["agent_1"])
        for step in range(12):
            detector.before_actions(step, health)
            detector.after_step(step)
        self.assertIsNone(detector.crash)
        self.assertFalse(detector.health["agent_1"]["online"])
        self.assertEqual(caught.exception.event["reason"], "agent_dropout")
        detector.reset("scene", "20", health)
        with self.assertRaises(DetectedSystemCrash):
            detector.before_actions(0, health)

    def test_survivor_hang_is_still_fatal_after_dropout_recovery(self):
        detector = ErrorDetector()
        detector.reset("scene", "20", ["agent_0", "agent_1"])
        health = {"agent_0": {"online": True, "responsive": False},
                  "agent_1": {"online": False, "responsive": False}}
        with self.assertRaises(DetectedSystemCrash):
            detector.before_actions(0, health)
        detector.acknowledge_dropout(["agent_1"])
        with self.assertRaises(DetectedSystemCrash) as caught:
            for step in range(10):
                detector.before_actions(step, health)
                detector.after_step(step)
        self.assertEqual(caught.exception.event["reason"], "agent_hang")
        self.assertEqual(caught.exception.event["agent"], "agent_0")

    def test_cannot_acknowledge_a_different_fault(self):
        detector = ErrorDetector()
        with self.assertRaises(ValueError):
            detector.acknowledge_dropout(["agent_1"])


class RecoveryDispatchTest(unittest.TestCase):
    def setUp(self):
        import torch
        from habitat_mas.agents.crab_agent import CrabAgent
        from habitat_mas.utils.models import OpenAIModel

        self.fixture = test_agent_hang_policy.AgentHangPolicyTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.namespace["DetectedSystemCrash"] = DetectedSystemCrash
        self.detector = ErrorDetector()
        self.detector.reset("scene", "15", ["agent_0", "agent_1"])
        self.events = []
        self.recovery = DropoutRecovery(record_event=self.events.append)
        self.context = {"goals": [goal(0, True), goal(1)], "scene_description": "current scene",
                        "simulator_state": {"agents": {"agent_0": {"held_object": None}}}}
        self.plan_prompts = []

        class Model(OpenAIModel):
            def __init__(model, system_message, **kwargs):
                model.set_system_message(system_message)
                model.chat_history = []

            def chat(model, prompt, **kwargs):
                self.plan_prompts.append(prompt)
                model.chat_history.append(prompt)
                return "Navigate to the pending target."

        self.addCleanup(patch.stopall)
        patch("habitat_mas.agents.crab_agent.OpenAIModel", Model).start()
        for i, policy in enumerate(self.fixture.agents):
            llm = CrabAgent(f"agent_{i}", [])
            llm.init_agent("test", "Detect both", f"Detect any_targets|{i}", skip_planning=True)
            policy._high_level_policy.llm_agent = llm
            policy._cur_call_high_level = torch.tensor([False])
        original = self.fixture.policy.act

        def act(*args, **kwargs):
            kwargs["error_detector"] = self.detector
            kwargs["dropout_recovery"] = self.recovery
            kwargs["get_recovery_context"] = lambda: self.context
            kwargs["envs_text_context"][0]["robot_resume"] = json.dumps({
                "agent_0": {"robot_type": "test"}, "agent_1": {"robot_type": "test"}})
            return original(*args, **kwargs)

        self.fixture.policy.act = act

    def test_dropout_replans_survivor_in_same_step_and_continues(self):
        actions = self.fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})
        survivor = self.fixture.agents[0]
        failed = self.fixture.agents[1]
        self.assertEqual(survivor.act.call_count, 1)
        failed.act.assert_not_called()
        self.assertTrue(survivor._cur_call_high_level.item())
        llm = survivor._high_level_policy.llm_agent
        self.assertIn("is_detected(any_targets|1)", llm.subtask_description)
        self.assertNotIn("any_targets|0", llm.subtask_description)
        self.assertEqual(llm.llm_model.system_message["role"], "system")
        self.assertIn("any_targets|1", llm.llm_model.system_message["content"])
        # Satisfied predicates may become false after movement. They remain
        # execution constraints, rather than being erased from the mission.
        self.assertIn("is_detected(any_targets|0)", llm.llm_model.system_message["content"])
        self.assertIn("current scene", self.plan_prompts[-1])
        self.assertIn("interrupted", self.plan_prompts[-1])
        self.assertTrue(actions[1].policy_info[0]["dropout_active"])
        self.detector.after_step(0)
        for step in range(1, 12):
            self.fixture.act(step, {})
            self.detector.after_step(step)
        self.assertEqual(survivor.act.call_count, 12)
        failed.act.assert_not_called()
        self.assertEqual(self.recovery.recoveries, 1)
        self.assertEqual([event["type"] for event in self.events],
                         ["failure_detected", "tasks_reassigned", "agent_replanned", "recovery_completed"])

    def test_membership_loss_without_injection_still_freezes_failed_policy(self):
        AgentMessageBus.drop_agent("agent_1")
        self.fixture.act(0, {})
        self.fixture.agents[1].act.assert_not_called()
        self.assertEqual(self.recovery.recoveries, 1)

    def test_step_zero_uses_remaining_team_without_initial_discussion(self):
        self.fixture.act(0, {}, starting=True, dropout={"enabled": True, "agent": "agent_1"})
        self.fixture.namespace["group_discussion"].assert_not_called()
        self.fixture.agents[1].act.assert_not_called()
        self.assertTrue(self.fixture.agents[0]._high_level_policy.llm_agent.initialized)
        self.assertEqual(self.recovery.recoveries, 1)

    def test_repeated_episode_id_gets_a_fresh_recovery(self):
        for _ in range(2):
            # These are the episode-boundary calls made by the evaluator.
            self.detector.reset("scene", "15", ["agent_0", "agent_1"])
            self.recovery.reset()
            self.fixture.act(0, {}, starting=True, dropout={"enabled": True, "agent": "agent_1"})
            self.assertEqual(self.recovery.recoveries, 1)
            self.assertEqual(self.detector.recovered_dropouts, {"agent_1"})
        self.assertEqual(len([event for event in self.events if event["type"] == "recovery_completed"]), 2)

    def test_loss_of_whole_team_keeps_crash_and_does_not_run_policies(self):
        AgentMessageBus.drop_agent("agent_0")
        AgentMessageBus.drop_agent("agent_1")
        with self.assertRaises(DetectedSystemCrash) as caught:
            self.fixture.act(0, {})
        self.assertIn("No online agents", caught.exception.event["recovery_failure"])
        for agent in self.fixture.agents:
            agent.act.assert_not_called()

    def test_replanned_goal_reaches_real_high_level_action_selection(self):
        import torch
        from habitat_baselines.rl.hrl.hl.llm_policy import LLMHighLevelPolicy

        self.fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})
        llm = self.fixture.agents[0]._high_level_policy.llm_agent
        llm.llm_model.token_usage = 0
        observations = []

        def select_action(observation):
            observations.append(observation)
            return "nav_to_obj", {"target_obj": "any_targets|1"}

        llm.llm_model.chat = select_action
        hl = LLMHighLevelPolicy.__new__(LLMHighLevelPolicy)
        torch.nn.Module.__init__(hl)
        hl.llm_agent = llm
        hl._skill_name_to_idx = {"nav_to_obj": 7, "wait": 0}
        skill, args, _, info = hl.get_next_skill(
            {}, None, None, torch.ones(1, 1), torch.ones(1), False, [{}],
            envs_text_context=[{"scene_description": "fresh current scene"}], agent_arguments=[],
        )
        self.assertEqual(skill.tolist(), [7])
        self.assertEqual(args, [{"target_obj": "any_targets|1", "robot": "agent_0"}])
        self.assertEqual(info.policy_info[0]["high_level_decision"]["action"], "nav_to_obj")
        self.assertIn("fresh current scene", observations[0])
        self.assertNotIn("completed your previous action", observations[0])

    def test_recovery_events_are_visible_in_readable_trajectory(self):
        from habitat_mas.fault_injection.trajectory import AgentDropoutTrajectoryRecorder
        with tempfile.TemporaryDirectory() as directory:
            recorder = AgentDropoutTrajectoryRecorder(directory, "recovery", "faulty", ["agent_0", "agent_1"])
            recorder.record_episode_start(episode_id="15", scene_id="scene", text_context={})
            self.recovery = DropoutRecovery(record_event=recorder.record_recovery_event)
            self.fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})
            trace = recorder.text_path_for_episode("15").read_text()
            self.assertIn("tasks_reassigned", trace)
            self.assertIn("is_detected(any_targets|1)", trace)
            self.assertIn("recovery_completed", trace)

    def test_unrecoverable_goal_retains_crash_and_executes_no_agent(self):
        self.context["unsupported_reason"] = "Recovery requires conjunctions"
        with self.assertRaises(DetectedSystemCrash) as caught:
            self.fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})
        for agent in self.fixture.agents:
            agent.act.assert_not_called()
        self.assertIn("conjunctions", caught.exception.event["recovery_failure"])
        self.assertEqual(self.events[-1]["type"], "recovery_failed")

    def test_audit_survives_disabled_trajectory_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recovery.jsonl"
            self.recovery = DropoutRecovery(event_path=path)
            self.fixture.act(0, {}, dropout={"enabled": True, "agent": "agent_1"})
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[-1]["type"], "recovery_completed")
            self.recovery.reset()
            self.assertEqual(self.recovery.recoveries, 0)


if __name__ == "__main__":
    unittest.main()
