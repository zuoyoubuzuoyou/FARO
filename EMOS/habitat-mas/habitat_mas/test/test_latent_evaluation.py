import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from habitat_mas.fault_injection.latent import MessageDelayFaultSpec
from habitat_mas.fault_injection.latent_runtime import LatentFaultRuntime
from habitat_mas.fault_injection.trajectory import AgentDropoutTrajectoryRecorder
from habitat_mas.agents.message_bus import AgentMessageBus
from habitat_mas.fault_injection.agent_dropout import AgentDropoutEventRecorder


class LatentEvaluationTest(unittest.TestCase):
    def test_policy_exception_keeps_original_error_and_saves_pending_fault_evidence(self):
        from habitat_baselines.rl.multi_agent.habitat_mas_evaluator import HabitatMASEvaluator
        AgentMessageBus.reset()
        self.addCleanup(AgentMessageBus.reset)
        runtime = LatentFaultRuntime(MessageDelayFaultSpec(enabled=True, delay_steps=10), "scene", "15")
        runtime.begin_step(0)
        AgentMessageBus.send_request("agent_1", "agent_0", "delayed before API failure")
        error = RuntimeError("original API failure")
        with tempfile.TemporaryDirectory() as directory:
            evaluator = HabitatMASEvaluator()
            events = AgentDropoutEventRecorder(directory, "abort", fault_name="message_delay")
            trajectory = AgentDropoutTrajectoryRecorder(directory, "abort", "faulty", ["agent_0"], fault_name="message_delay")
            summary_path = Path(directory) / "summary.json"
            evaluator._crash_context = dict(latent_runtime=runtime, latent_summaries=[],
                latent_summary_path=summary_path, fault_recorder=events, trajectory_recorder=trajectory)
            evaluator._evaluate_agent = MagicMock(side_effect=error)
            with self.assertRaises(RuntimeError) as caught:
                evaluator.evaluate_agent(None, MagicMock(), None, 0, 0, None, None, None, None, None)
            self.assertIs(caught.exception, error)
            rows = [json.loads(line) for line in events.path.read_text().splitlines()]
            self.assertEqual([row["type"] for row in rows], ["fault_injected", "message_queued", "fault_applied", "message_dropped", "fault_episode_summary"])
            self.assertEqual(rows[-1]["termination_reason"], "exception")
            self.assertEqual(rows[-1]["error_type"], "RuntimeError")
            self.assertTrue(json.loads(summary_path.read_text())["episodes"][0]["effective"])
            self.assertFalse(AgentMessageBus.pending_messages)
            trajectory_rows = [json.loads(line) for line in trajectory.path.read_text().splitlines()]
            self.assertEqual(trajectory_rows[-1]["type"], "fault_episode_summary")

    def test_generic_events_are_written_and_rendered_with_evidence(self):
        event = {"type": "fault_applied", "fault_type": "ActionFault", "fault_subtype": "ActionNoop",
                 "agent": "agent_0", "simulator_step": 4, "original_action": [3], "effective_action": [0]}
        info = {"agent_0_fault_events": [event]}
        self.assertTrue(AgentDropoutTrajectoryRecorder.is_high_level_trace_event(info))
        with tempfile.TemporaryDirectory() as directory:
            recorder = AgentDropoutTrajectoryRecorder(directory, "case", "faulty", ["agent_0"], fault_name="action_noop")
            recorder.record_episode_start(episode_id="15", scene_id="scene", text_context={})
            recorder.record_step(episode_id="15", scene_id="scene", simulator_step=4, joint_env_action=[0],
                                 policy_info=info, reward=0, done=False, metrics={})
            row = json.loads(recorder.path.read_text().splitlines()[-1])
            self.assertEqual(row["fault_events"], [event])
            text = recorder.text_path_for_episode("15").read_text()
            self.assertIn("ActionFault/ActionNoop", text)
            self.assertIn("original_action", text)
            self.assertIn("effective_action", text)

    def test_message_summary_counts_only_actual_queue_interception(self):
        AgentMessageBus.reset()
        self.addCleanup(AgentMessageBus.reset)
        runtime = LatentFaultRuntime(MessageDelayFaultSpec(enabled=True, simulator_step=4, delay_steps=2), "scene", "15")
        runtime.begin_step(3)
        AgentMessageBus.send_request("agent_1", "agent_0", "before")
        runtime.collect_messages()
        self.assertFalse(runtime.summary()["triggered"])
        runtime.begin_step(4)
        AgentMessageBus.send_request("agent_1", "agent_0", "delayed")
        runtime.collect_messages()
        self.assertEqual(runtime.summary()["applications"], 1)
        runtime.begin_step(6)
        runtime.collect_messages()
        self.assertEqual(runtime.summary()["applications"], 1)
        summary = runtime.finish_episode()
        self.assertTrue(summary["effective"])
        self.assertEqual(summary["type"], "fault_episode_summary")


if __name__ == "__main__":
    unittest.main()
