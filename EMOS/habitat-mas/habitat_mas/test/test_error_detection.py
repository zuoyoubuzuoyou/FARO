import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from habitat_mas.error_detection import ErrorDetector, ErrorDetectionSpec, DetectedSystemCrash
from habitat_mas.agents.message_bus import AgentMessageBus
from habitat_mas.fault_injection.trajectory import AgentDropoutTrajectoryRecorder
from habitat_mas.test import test_agent_hang_policy


ONLINE = {name: {"online": True, "responsive": True} for name in ("agent_0", "agent_1")}


class ErrorDetectorTest(unittest.TestCase):
    def detector(self, **kwargs):
        detector = ErrorDetector(ErrorDetectionSpec(**kwargs))
        detector.reset("scene", "20", ONLINE)
        return detector

    def test_dropout_immediate_and_latched_until_reset(self):
        detector = self.detector()
        health = copy.deepcopy(ONLINE)
        health["agent_1"]["online"] = False
        with self.assertRaises(DetectedSystemCrash) as caught:
            detector.before_actions(40, health)
        self.assertEqual(caught.exception.event["phase"], "PRE_ACTION")
        self.assertEqual(caught.exception.event["simulator_step"], 40)
        with self.assertRaises(DetectedSystemCrash): detector.before_actions(41, ONLINE)
        with self.assertRaises(DetectedSystemCrash): detector.check_messages(41, [])
        detector.reset("scene", "20", ONLINE)
        detector.before_actions(0, ONLINE)
        self.assertEqual(detector.metrics()["error_detection.system_crash"], 0)

    def test_hang_after_exactly_ten_or_twenty_completed_steps(self):
        for limit in (10, 20):
            with self.subTest(limit=limit):
                detector = self.detector(hang_steps=limit)
                health = copy.deepcopy(ONLINE)
                for step in range(40 + limit):
                    health["agent_1"]["responsive"] = step < 40
                    detector.before_actions(step, copy.deepcopy(health))
                    if step == 40 + limit - 1:
                        with self.assertRaises(DetectedSystemCrash) as caught:
                            detector.after_step(step)
                        self.assertEqual(caught.exception.event["consecutive_count"], limit)
                        self.assertEqual(caught.exception.event["reason"], "agent_hang")
                        self.assertEqual(caught.exception.event["phase"], "POST_ENV_STEP")
                    else:
                        detector.after_step(step)
                self.assertEqual(detector.metrics()["error_detection.all_agents_online"], 1)
                self.assertEqual(detector.metrics()["error_detection.all_agents_responsive"], 0)

    def test_normal_wait_is_responsive_even_for_750_steps(self):
        detector = self.detector()
        for step in range(750):
            detector.before_actions(step, ONLINE)
            detector.check_messages(step, [], progress="unchanged pose")
            detector.after_step(step)
        self.assertEqual(detector.metrics()["error_detection.system_crash"], 0)

    def test_recovery_resets_consecutive_unresponsive_counter(self):
        detector = self.detector(hang_steps=3)
        for step, responsive in enumerate([False, False, True, False, False]):
            health = copy.deepcopy(ONLINE)
            health["agent_1"]["responsive"] = responsive
            detector.before_actions(step, health)
            detector.after_step(step)
        self.assertEqual(detector.unresponsive["agent_1"], 2)

    def test_only_actual_duplicate_messages_count_and_progress_breaks_loop(self):
        detector = self.detector(repeated_message_limit=3)
        message = {"source": "agent_0", "target": "agent_1", "request": "please help"}
        detector.check_messages(0, [message], progress="position A")
        for step in range(1, 50): detector.check_messages(step, [], progress="position A")
        detector.check_messages(50, [message], progress="position A")
        detector.check_messages(51, [message], progress="position B")
        detector.check_messages(52, [{**message, "request": "please   help"}], progress="position B")
        with self.assertRaises(DetectedSystemCrash) as caught:
            detector.check_messages(53, [message], progress="position B")
        self.assertEqual(caught.exception.event["reason"], "message_loop")
        self.assertEqual(caught.exception.event["consecutive_count"], 3)
        self.assertEqual(detector.metrics()["error_detection.no_message_loop"], 0)

    def test_new_content_or_new_recipient_breaks_message_sequence(self):
        detector = self.detector(repeated_message_limit=2)
        for step in range(10):
            detector.check_messages(step, [{"source": "agent_0", "target": "agent_1", "request": str(step)}])
        detector.check_messages(10, [{"source": "agent_0", "target": "agent_2", "request": "9"}])

    def test_disabled_does_not_stop_injected_faults(self):
        detector = self.detector(enabled=False)
        for step in range(100):
            detector.before_actions(step, {"agent_1": {"online": False, "responsive": False}})
            detector.after_step(step)
        self.assertIsNone(detector.crash)

    def test_invalid_configuration_rejected(self):
        for values in ({"hang_steps": 0}, {"hang_steps": True}, {"hang_steps": 1.5},
                       {"repeated_message_limit": -1}, {"enabled": "false"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ErrorDetectionSpec.from_config(values)

    def test_no_missing_status_or_double_counting(self):
        detector = self.detector()
        with self.assertRaises(ValueError): detector.before_actions(0, {})
        detector.before_actions(0, ONLINE)
        detector.after_step(0)
        with self.assertRaises(ValueError): detector.after_step(0)

    def test_real_dispatch_blocks_all_policies_on_dropout(self):
        fixture = test_agent_hang_policy.AgentHangPolicyTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.namespace["DetectedSystemCrash"] = DetectedSystemCrash
        detector = self.detector()
        original = fixture.policy.act
        fixture.policy.act = lambda *args, **kwargs: original(*args, error_detector=detector, **kwargs)
        fixture.act(0, {}, starting=True)
        calls = [agent.act.call_count for agent in fixture.agents]
        with self.assertRaises(DetectedSystemCrash) as caught:
            fixture.act(40, {}, dropout={"enabled": True, "agent": "agent_1", "simulator_step": 40})
        self.assertEqual([agent.act.call_count for agent in fixture.agents], calls)
        self.assertEqual(caught.exception.fault_events[0]["fault_subtype"], "AgentDropout")

    def test_step_zero_dropout_blocks_initial_llm_discussion(self):
        fixture = test_agent_hang_policy.AgentHangPolicyTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.namespace["DetectedSystemCrash"] = DetectedSystemCrash
        detector = self.detector()
        original = fixture.policy.act
        fixture.policy.act = lambda *args, **kwargs: original(*args, error_detector=detector, **kwargs)
        with self.assertRaises(DetectedSystemCrash):
            fixture.act(0, {}, starting=True, dropout={"enabled": True, "agent": "agent_1", "simulator_step": 0})
        fixture.namespace["group_discussion"].assert_not_called()
        for agent in fixture.agents:
            agent.act.assert_not_called()

    def test_membership_loss_is_detected_without_an_injection_schedule(self):
        fixture = test_agent_hang_policy.AgentHangPolicyTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.namespace["DetectedSystemCrash"] = DetectedSystemCrash
        detector = self.detector()
        original = fixture.policy.act
        fixture.policy.act = lambda *args, **kwargs: original(*args, error_detector=detector, **kwargs)
        fixture.act(0, {}, starting=True)
        AgentMessageBus.drop_agent("agent_1")
        with self.assertRaises(DetectedSystemCrash) as caught:
            fixture.act(1, {})
        self.assertEqual(caught.exception.event["reason"], "agent_dropout")
        self.assertEqual(caught.exception.fault_events, [])
        for agent in fixture.agents:
            self.assertEqual(agent.act.call_count, 1)


class CrashTerminationTest(unittest.TestCase):
    def test_post_step_hang_saves_actual_terminal_step_and_optional_trajectory(self):
        from habitat_baselines.rl.multi_agent.habitat_mas_evaluator import HabitatMASEvaluator
        for record_trajectory in (True, False):
            with self.subTest(record_trajectory=record_trajectory), tempfile.TemporaryDirectory() as directory:
                evaluator = HabitatMASEvaluator()
                detector = ErrorDetector(ErrorDetectionSpec(hang_steps=2))
                detector.reset("scene", "20", ONLINE)
                health = copy.deepcopy(ONLINE)
                health["agent_1"]["responsive"] = False
                detector.before_actions(0, health)
                detector.after_step(0)
                detector.before_actions(1, health)
                with self.assertRaises(DetectedSystemCrash) as caught:
                    detector.after_step(1)
                recorder = (AgentDropoutTrajectoryRecorder(directory, "hang_test", "faulty", ONLINE)
                            if record_trajectory else None)
                path = Path(directory) / "crash.json"
                evaluator._crash_context = dict(
                    detector=detector, metrics={"pddl_success": 0.0}, completed_steps=2,
                    reward_total=-0.02, last_state={}, report_path=path,
                    trajectory_recorder=recorder, fault_recorder=None, pbar=MagicMock(),
                    transition=dict(episode_id="20", scene_id="scene", simulator_step=1,
                                    joint_env_action=[0, 0], policy_info={}, reward=-0.01,
                                    simulator_state={}),
                )
                evaluator._evaluate_agent = MagicMock(side_effect=caught.exception)
                envs = MagicMock()
                with self.assertRaises(SystemExit) as exit_code:
                    evaluator.evaluate_agent(None, envs, None, 0, 0, MagicMock(), None, None, None, None)
                self.assertEqual(exit_code.exception.code, 3)
                envs.close.assert_called_once()
                envs.step.assert_not_called()
                report = json.loads(path.read_text())
                self.assertTrue(report["env_step_executed"])
                self.assertEqual(report["completed_steps"], 2)
                if recorder is not None:
                    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
                    steps = [row for row in rows if row["type"] == "simulator_step"]
                    self.assertEqual(len(steps), 1)
                    self.assertTrue(steps[0]["done"])
                    self.assertEqual(steps[0]["simulator_step"], 1)
                    self.assertEqual(rows[-1]["type"], "system_crash")

    def test_unrelated_exceptions_keep_their_original_error(self):
        from habitat_baselines.rl.multi_agent.habitat_mas_evaluator import HabitatMASEvaluator
        evaluator = HabitatMASEvaluator()
        evaluator._evaluate_agent = MagicMock(side_effect=ValueError("replay mismatch"))
        evaluator._record_system_crash = MagicMock()
        with self.assertRaisesRegex(ValueError, "replay mismatch"):
            evaluator.evaluate_agent(None, MagicMock(), None, 0, 0, None, None, None, None, None)
        evaluator._record_system_crash.assert_not_called()

    def test_real_evaluator_catches_crash_and_closes_environment(self):
        from habitat_baselines.rl.multi_agent.habitat_mas_evaluator import HabitatMASEvaluator
        evaluator = HabitatMASEvaluator()
        detector = ErrorDetector()
        detector.reset("scene", "20", ONLINE)
        health = copy.deepcopy(ONLINE)
        health["agent_1"].update(online=False, responsive=False)
        with self.assertRaises(DetectedSystemCrash) as caught:
            detector.before_actions(40, health)
        with tempfile.TemporaryDirectory() as directory:
            recorder = AgentDropoutTrajectoryRecorder(directory, "dropout_test", "faulty", ONLINE)
            recorder.record_episode_start(episode_id="20", scene_id="scene", text_context={})
            path = Path(directory) / "crash.json"
            fault_recorder = MagicMock()
            caught.exception.fault_events = [{"type": "fault_injected"}]
            evaluator._crash_context = dict(
                detector=detector, metrics={"pddl_success": 0.0}, completed_steps=40,
                reward_total=-0.02, transition=None, last_state={}, report_path=path,
                trajectory_recorder=recorder, fault_recorder=fault_recorder, pbar=MagicMock(),
            )
            evaluator._evaluate_agent = MagicMock(side_effect=caught.exception)
            envs, writer = MagicMock(), MagicMock()
            with self.assertRaises(SystemExit) as exit_code:
                evaluator.evaluate_agent(None, envs, None, 0, 0, writer, None, None, None, None)
            self.assertEqual(exit_code.exception.code, 3)
            envs.close.assert_called_once()
            envs.step.assert_not_called()
            report = json.loads(path.read_text())
            self.assertTrue(report["done"])
            self.assertFalse(report["env_step_executed"])
            self.assertEqual(report["completed_steps"], 40)
            self.assertEqual(report["metrics"]["error_detection.system_crash"], 1)
            self.assertEqual(fault_recorder.record.call_count, 1)
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            self.assertFalse(any(row["type"] == "simulator_step" for row in rows))
            self.assertIn("System_Crash:", recorder.text_path_for_episode("20").read_text())


if __name__ == "__main__":
    unittest.main()
