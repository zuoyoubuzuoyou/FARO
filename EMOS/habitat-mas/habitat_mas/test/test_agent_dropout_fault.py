import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from habitat_mas.fault_injection.agent_dropout import (
    AgentDropoutEventRecorder,
    AgentDropoutFaultSpec,
    FaultInjectionError,
    build_agent_dropout_event,
)


class AgentDropoutFaultTest(unittest.TestCase):
    def test_dropout_activates_at_selected_simulator_step(self):
        spec = AgentDropoutFaultSpec(
            enabled=True,
            episode_id="15",
            agent="agent_1",
            simulator_step=40,
        )

        self.assertFalse(
            spec.is_active(
                episode_id="15", agent="agent_1", simulator_step=39
            )
        )
        self.assertTrue(
            spec.is_trigger(
                episode_id="15", agent="agent_1", simulator_step=40
            )
        )
        self.assertTrue(
            spec.is_active(
                episode_id="15", agent="agent_1", simulator_step=749
            )
        )

    def test_dropout_only_matches_selected_episode_and_agent(self):
        spec = AgentDropoutFaultSpec(
            enabled=True,
            episode_id="15",
            agent="agent_1",
            simulator_step=0,
        )

        self.assertFalse(
            spec.is_active(
                episode_id="14", agent="agent_1", simulator_step=0
            )
        )
        self.assertFalse(
            spec.is_active(
                episode_id="15", agent="agent_0", simulator_step=0
            )
        )

    def test_per_episode_simulator_step_schedule(self):
        spec = AgentDropoutFaultSpec(
            enabled=True,
            agent="agent_1",
            simulator_step_schedule="233@12;96@0;77@31",
        )

        self.assertEqual(
            spec.simulator_step_schedule_map(),
            {"233": 12, "96": 0, "77": 31},
        )
        self.assertFalse(
            spec.is_trigger(
                episode_id="233", agent="agent_1", simulator_step=11
            )
        )
        self.assertTrue(
            spec.is_trigger(
                episode_id="233", agent="agent_1", simulator_step=12
            )
        )
        self.assertTrue(
            spec.is_trigger(
                episode_id="96", agent="agent_1", simulator_step=0
            )
        )
        self.assertFalse(
            spec.is_active(
                episode_id="15", agent="agent_1", simulator_step=100
            )
        )

        event = build_agent_dropout_event(
            spec, episode_id="77", simulator_step=31
        )
        self.assertEqual(event["configured_simulator_step"], 31)

    def test_invalid_simulator_step_schedule_fails_loudly(self):
        invalid_schedules = (
            "233",
            "233@x",
            "233@-1",
            "233@1;233@2",
        )
        for schedule in invalid_schedules:
            with self.subTest(schedule=schedule):
                with self.assertRaises(FaultInjectionError):
                    AgentDropoutFaultSpec(
                        enabled=True,
                        simulator_step_schedule=schedule,
                    ).validate()

        with self.assertRaisesRegex(
            FaultInjectionError, "mutually exclusive"
        ):
            AgentDropoutFaultSpec(
                enabled=True,
                episode_id="233",
                simulator_step_schedule="233@1",
            ).validate()
        with self.assertRaisesRegex(
            FaultInjectionError, "trigger_domain=simulator_step"
        ):
            AgentDropoutFaultSpec(
                enabled=True,
                trigger_domain="high_level_decision",
                simulator_step_schedule="233@1",
            ).validate()

    def test_dropout_triggers_at_selected_high_level_decision(self):
        spec = AgentDropoutFaultSpec(
            enabled=True,
            episode_id="15",
            agent="agent_1",
            trigger_domain="high_level_decision",
            decision_step=2,
        )

        self.assertFalse(
            spec.is_trigger(
                episode_id="15",
                agent="agent_1",
                simulator_step=40,
                decision_step=2,
                is_high_level_decision=False,
            )
        )
        self.assertFalse(
            spec.is_trigger(
                episode_id="15",
                agent="agent_1",
                simulator_step=86,
                decision_step=1,
                is_high_level_decision=True,
            )
        )
        self.assertTrue(
            spec.is_trigger(
                episode_id="15",
                agent="agent_1",
                simulator_step=87,
                decision_step=2,
                is_high_level_decision=True,
            )
        )

        event = build_agent_dropout_event(
            spec,
            episode_id="15",
            simulator_step=87,
            decision_step=2,
        )
        self.assertEqual(event["trigger_domain"], "high_level_decision")
        self.assertEqual(event["phase"], "PRE_HIGH_LEVEL_DECISION")
        self.assertEqual(event["decision_step"], 2)
        self.assertEqual(event["simulator_step"], 87)

    def test_event_describes_logical_silent_dropout(self):
        spec = AgentDropoutFaultSpec(
            enabled=True,
            fault_id="SF_test_001",
            episode_id="15",
            agent="agent_1",
            simulator_step=12,
        )

        event = build_agent_dropout_event(spec, episode_id="15")

        self.assertEqual(event["fault_type"], "StructureFault")
        self.assertEqual(event["fault_subtype"], "AgentDropout")
        self.assertEqual(event["phase"], "PRE_ACTION")
        self.assertEqual(event["simulator_step"], 12)
        self.assertTrue(event["silent"])
        self.assertFalse(event["logical_agent_active"])
        self.assertFalse(event["simulator_agent_removed"])
        self.assertEqual(
            event["effects"]["environment_action"], "zero_noop"
        )

    def test_invalid_configuration_fails_loudly(self):
        with self.assertRaisesRegex(FaultInjectionError, "agent_N"):
            AgentDropoutFaultSpec(
                enabled=True, agent="drone", simulator_step=0
            ).validate()
        with self.assertRaisesRegex(FaultInjectionError, ">= 0"):
            AgentDropoutFaultSpec(
                enabled=True, agent="agent_1", simulator_step=-1
            ).validate()
        with self.assertRaisesRegex(FaultInjectionError, "trigger_domain"):
            AgentDropoutFaultSpec(
                enabled=True, trigger_domain="wall_clock"
            ).validate()
        with self.assertRaisesRegex(FaultInjectionError, "decision_step"):
            AgentDropoutFaultSpec(
                enabled=True,
                trigger_domain="high_level_decision",
                decision_step=-1,
            ).validate()

    def test_event_recorder_writes_one_jsonl_event(self):
        with TemporaryDirectory() as temp_dir:
            recorder = AgentDropoutEventRecorder(temp_dir, "fault run")
            event = {"type": "fault_injected", "agent": "agent_1"}

            recorder.record(event)

            self.assertEqual(recorder.count, 1)
            self.assertEqual(recorder.episode_ids, set())
            self.assertEqual(
                recorder.path.name,
                "dropout_fault_run_events.jsonl",
            )
            lines = Path(recorder.path).read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual([json.loads(line) for line in lines], [event])

    def test_event_recorder_tracks_triggered_episode_ids(self):
        with TemporaryDirectory() as temp_dir:
            recorder = AgentDropoutEventRecorder(temp_dir, "scheduled")

            recorder.record({"episode_id": "233"})
            recorder.record({"episode_id": 96})

            self.assertEqual(recorder.episode_ids, {"233", "96"})


if __name__ == "__main__":
    unittest.main()
