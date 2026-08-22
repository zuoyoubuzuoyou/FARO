import json
import unittest

from habitat_mas.fault_injection.object_localization import (
    FaultEventRecorder,
    FaultInjectionError,
    ObjectLocalizationFaultSpec,
    inject_object_localization_error,
)


class ObjectLocalizationFaultTest(unittest.TestCase):
    def test_mp3d_region_fault_changes_only_target_object(self):
        scene = (
            'The object "cup_0" is located in kitchen_1 on 0 floor. '
            'The height of object "cup_0" from the floor is 0.8.\n'
            'The object "bowl_0" is located in dining_room_2 on 0 floor.'
        )
        spec = ObjectLocalizationFaultSpec(
            enabled=True,
            episode_id="272",
            agent="agent_1",
            decision_step=2,
            object_name="cup_0",
            replacement_region="bedroom_3",
            replacement_floor="1",
        )

        mutated, event = inject_object_localization_error(
            scene,
            spec,
            episode_id="272",
            agent="agent_1",
            decision_step=2,
            simulator_step=41,
        )

        self.assertIn(
            'The object "cup_0" is located in bedroom_3 on 1 floor.',
            mutated,
        )
        self.assertIn(
            'The object "bowl_0" is located in dining_room_2 on 0 floor.',
            mutated,
        )
        self.assertIn("kitchen_1", scene)
        self.assertEqual(event["phase"], "POST_PERCEPTION")
        self.assertEqual(event["decision_step"], 2)
        self.assertEqual(event["simulator_step"], 41)
        self.assertTrue(event["ground_truth_preserved"])
        self.assertEqual(
            [change["field"] for change in event["changes"]],
            ["region", "floor"],
        )

    def test_hssd_geometric_fault_changes_height_and_distance(self):
        scene = (
            'The height of "any_targets|0" is 0.7. '
            'The horizontal distance of "any_targets|0" to the nearest '
            "navigable point is 0.2."
        )
        spec = ObjectLocalizationFaultSpec(
            enabled=True,
            agent="agent_0",
            decision_step=0,
            object_name="any_targets|0",
            replacement_height="1.9",
            replacement_horizontal_distance="3.4",
        )

        mutated, event = inject_object_localization_error(
            scene,
            spec,
            episode_id="1",
            agent="agent_0",
            decision_step=0,
        )

        self.assertIn('The height of "any_targets|0" is 1.9.', mutated)
        self.assertIn("nearest navigable point is 3.4.", mutated)
        self.assertEqual(
            [change["field"] for change in event["changes"]],
            ["height", "horizontal_distance"],
        )

    def test_fault_spec_matches_exact_trigger_only(self):
        spec = ObjectLocalizationFaultSpec(
            enabled=True,
            episode_id="9",
            agent="agent_1",
            decision_step=3,
            object_name="cup",
            replacement_region="hall_0",
        )

        self.assertTrue(
            spec.matches(episode_id=9, agent="agent_1", decision_step=3)
        )
        self.assertFalse(
            spec.matches(episode_id=8, agent="agent_1", decision_step=3)
        )
        self.assertFalse(
            spec.matches(episode_id=9, agent="agent_0", decision_step=3)
        )
        self.assertFalse(
            spec.matches(episode_id=9, agent="agent_1", decision_step=2)
        )

    def test_missing_location_fact_fails_loudly(self):
        spec = ObjectLocalizationFaultSpec(
            enabled=True,
            agent="agent_0",
            decision_step=0,
            object_name="missing_cup",
            replacement_region="hall_0",
        )

        with self.assertRaisesRegex(
            FaultInjectionError, "no MP3D region fact"
        ):
            inject_object_localization_error(
                'The object "cup" is located in kitchen_0 on 0 floor.',
                spec,
                episode_id="1",
                agent="agent_0",
                decision_step=0,
            )

    def test_duplicate_object_location_is_rejected_as_ambiguous(self):
        spec = ObjectLocalizationFaultSpec(
            enabled=True,
            agent="agent_0",
            decision_step=0,
            object_name="cup",
            replacement_region="hall_0",
        )
        scene = (
            'The object "cup" is located in kitchen_0 on 0 floor. '
            'The object "cup" is located in kitchen_1 on 0 floor.'
        )

        with self.assertRaisesRegex(
            FaultInjectionError, "multiple MP3D region facts"
        ):
            inject_object_localization_error(
                scene,
                spec,
                episode_id="1",
                agent="agent_0",
                decision_step=0,
            )

    def test_fault_event_recorder_writes_jsonl(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = FaultEventRecorder(temp_dir, "fault run 1")
            event = {"type": "fault_injected", "object": "杯子"}

            recorder.record(event)

            self.assertEqual(
                recorder.path.name, "fault_run_1_fault_events.jsonl"
            )
            self.assertEqual(recorder.count, 1)
            lines = recorder.path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0]), event)


if __name__ == "__main__":
    unittest.main()
