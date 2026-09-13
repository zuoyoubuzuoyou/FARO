import json
import unittest
from tempfile import TemporaryDirectory

from habitat_mas.fault_injection.agent_dropout import FaultInjectionError
from habitat_mas.fault_injection.agent_hang import (
    AgentHangEventRecorder,
    AgentHangFaultSpec,
    AgentHangState,
)
from habitat_mas.fault_injection.trajectory import AgentDropoutTrajectoryRecorder


class AgentHangFaultTest(unittest.TestCase):
    def advance(self, state, spec, step, **kwargs):
        return state.advance(
            spec,
            episode_id=kwargs.get("episode_id", "15"),
            agent=kwargs.get("agent", "agent_1"),
            simulator_step=step,
            decision_step=kwargs.get("decision_step", 2),
        )

    def test_permanent_hang_emits_once_and_remains_logically_alive(self):
        spec = AgentHangFaultSpec(enabled=True, agent="agent_1", simulator_step=40)
        state = AgentHangState()
        self.assertIsNone(self.advance(state, spec, 39))
        event = self.advance(state, spec, 40)
        self.assertEqual(event["fault_subtype"], "AgentHang")
        self.assertEqual(event["duration"], "until_episode_end")
        self.assertEqual(event["phase"], "PRE_ACTION")
        self.assertEqual(event["trigger_domain"], "simulator_step")
        self.assertNotIn("configured_decision_step", event)
        self.assertTrue(event["logical_agent_active"])
        self.assertFalse(event["logical_agent_responsive"])
        self.assertFalse(event["simulator_agent_removed"])
        self.assertTrue(event["silent"])
        self.assertEqual(event["effects"]["team_membership"], "retained")
        self.assertEqual(event["effects"]["incoming_messages"], "queued_not_processed")
        self.assertEqual(event["effects"]["queued_messages"], "retained")
        self.assertEqual(event["effects"]["policy_state"], "frozen")
        for step in (40, 41, 750, 10000):
            self.assertIsNone(self.advance(state, spec, step))
            self.assertIs(state.active_event, event)

    def test_rejects_decision_trigger_configuration(self):
        for config in (
            {"trigger_domain": "high_level_decision"},
            {"decision_step": 2},
        ):
            with self.subTest(config=config):
                with self.assertRaisesRegex(FaultInjectionError, "simulator_step"):
                    AgentHangFaultSpec.from_config({"enabled": True, **config})

    def test_selection_and_schedule(self):
        spec = AgentHangFaultSpec(
            enabled=True, agent="agent_1", simulator_step_schedule="15@2;16@0"
        )
        for episode, agent, step in (("17", "agent_1", 2), ("15", "agent_0", 2), ("15", "agent_1", 1)):
            self.assertIsNone(self.advance(AgentHangState(), spec, step, episode_id=episode, agent=agent))
        for episode, step in (("15", 2), ("16", 0)):
            event = self.advance(AgentHangState(), spec, step, episode_id=episode)
            self.assertEqual(event["configured_simulator_step"], step)

    def test_disabled_and_invalid_configuration(self):
        self.assertIsNone(self.advance(AgentHangState(), AgentHangFaultSpec(), 0))
        self.assertEqual(AgentHangFaultSpec.from_config(None).fault_id, "SF_agent_hang_001")
        for config in ({"agent": "drone"}, {"simulator_step": -1}, {"trigger_domain": "wall_clock"}, {"simulator_step_schedule": "15@x"}):
            with self.subTest(config=config), self.assertRaisesRegex(FaultInjectionError, "agent_hang"):
                AgentHangFaultSpec.from_config({"enabled": True, **config}).validate()

    def test_hang_event_and_trajectory_outputs(self):
        spec = AgentHangFaultSpec(enabled=True, agent="agent_1", simulator_step=1)
        state = AgentHangState()
        event = self.advance(state, spec, 1)
        with TemporaryDirectory() as directory:
            events = AgentHangEventRecorder(directory, "hang test")
            events.record(event)
            self.assertEqual(events.path.name, "hang_test_events.jsonl")
            self.assertEqual(events.episode_ids, {"15"})
            self.assertEqual(json.loads(events.path.read_text()), event)
            recorder = AgentDropoutTrajectoryRecorder(
                directory, "hang test", "faulty", ["agent_1"], fault_name="agent_hang"
            )
            recorder.record_episode_start(episode_id="15", scene_id="scene", text_context={})
            for step in (1, 2):
                info = {"agent_1_hang_active": True, "agent_1_hang_since_simulator_step": 1}
                if step == 1:
                    info["agent_1_hang_event"] = event
                recorder.record_step(
                    episode_id="15", scene_id="scene", simulator_step=step,
                    joint_env_action=[0], policy_info=info, reward=0, done=False, metrics={}
                )
            rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
            self.assertEqual(recorder.path.name, "hang_test_trajectory.jsonl")
            for row in rows[-2:]:
                self.assertEqual(row["agent_step_actions"]["agent_1"]["status"], "hang_noop")
                self.assertTrue(row["simulator_state"]["agents"]["agent_1"]["hang_state"]["active"])
            trace = recorder.text_path_for_episode("15").read_text()
            self.assertEqual(trace.count("StructureFault/AgentHang fault_injected"), 1)
            self.assertEqual(trace.count("AgentHangNoOp[]"), 4)
            self.assertTrue(recorder.is_high_level_trace_event({"agent_1_hang_event": event}))


if __name__ == "__main__":
    unittest.main()
