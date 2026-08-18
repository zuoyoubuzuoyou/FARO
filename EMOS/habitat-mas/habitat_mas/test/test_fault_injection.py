import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from habitat_mas.fault_injection import TextObservationFaultInjector
from habitat_mas.fault_injection.trajectory import (
    TrajectoryRecorder,
    compare_trajectories,
)


def _fault(**overrides):
    fault = {
        "fault_id": "PF_wrong_location_001",
        "fault_type": "PerceptionFault",
        "fault_subtype": "WrongObjectLocation",
        "episode_id": "2",
        "injected_at_step": 0,
        "faulty_agent": "agent_1",
        "affected_object": "031_spoon_:0000",
        "affected_subtask": "pick_up_spoon",
        "observed_location": "inside the cabinet",
        "severity": "medium",
        "expected_recovery": "refresh_observation_or_ask_teammate",
    }
    fault.update(overrides)
    return fault


def _phase_fault(subtype, fault_type, **overrides):
    fault = {
        "fault_id": f"test_{subtype}",
        "fault_type": fault_type,
        "fault_subtype": subtype,
        "episode_id": "21",
        "injected_at_step": 0,
        "faulty_agent": "agent_1",
    }
    fault.update(overrides)
    return fault


class TextObservationFaultInjectorTest(unittest.TestCase):
    def test_all_mvp_example_schedules_load(self):
        schedule_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "config", "fault_injection"
        )
        schedules = {
            "object_localization_episode_21.json": "WrongObjectLocation",
            "f2_object_recognition_episode_21.json": "ObjectRecognitionError",
            "f3_wrong_subgoal_episode_21.json": "WrongSubgoal",
            "f4_missing_subgoal_episode_21.json": "MissingSubgoal",
            "f5_stale_message_episode_21.json": "MessageDelayOrStale",
            "f6_duplicate_assignment_episode_21.json": "DuplicateAssignment",
            "f7_action_noop_episode_21.json": "ActionNoOpOrFalseSuccess",
            "f8_false_verification_episode_21.json": "MissingOrFalseVerification",
        }
        for filename, expected_subtype in schedules.items():
            with self.subTest(schedule=filename):
                with tempfile.TemporaryDirectory() as temp_dir:
                    injector = TextObservationFaultInjector.from_config(
                        {
                            "enabled": True,
                            "schedule_path": os.path.join(
                                schedule_dir, filename
                            ),
                            "log_dir": temp_dir,
                        }
                    )
                self.assertEqual(
                    injector.faults[0]["fault_subtype"], expected_subtype
                )

    def test_f2_recognition_and_f5_stale_message(self):
        faults = [
            _phase_fault(
                "ObjectRecognitionError",
                "PerceptionFault",
                affected_object="cup_0",
                observed_object="bowl_0",
            ),
            _phase_fault(
                "MessageDelayOrStale",
                "CommunicationFault",
                fault_id="test_stale_message",
                stale_message="The cup is still on the table.",
            ),
        ]
        scene = 'The object "cup_0" is located in kitchen_1.\n'
        with tempfile.TemporaryDirectory() as temp_dir:
            result = TextObservationFaultInjector(faults, temp_dir).inject(
                scene, episode_id="21", step=0, observer="agent_1"
            )

            self.assertNotIn('"cup_0"', result.scene_description)
            self.assertIn('"bowl_0"', result.scene_description)
            self.assertIn("Teammate message", result.scene_description)
            self.assertNotIn("stale message", result.scene_description.lower())
            self.assertEqual(len(result.records), 2)

    def test_f3_f4_f6_assignment_faults(self):
        assignments = {
            "agent_0": "Open the cabinet, then pick up the cup",
            "agent_1": "Detect the bowl",
        }
        cases = [
            (
                _phase_fault(
                    "WrongSubgoal",
                    "PlanningFault",
                    wrong_subgoal="Pick up the plate",
                ),
                "agent_1",
                "Pick up the plate",
            ),
            (
                _phase_fault(
                    "MissingSubgoal",
                    "PlanningFault",
                    faulty_agent="agent_0",
                    missing_subgoal="Open the cabinet, then ",
                ),
                "agent_0",
                "pick up the cup",
            ),
            (
                _phase_fault(
                    "DuplicateAssignment",
                    "CoordinationFault",
                    faulty_agent="agent_0",
                    source_agent="agent_1",
                ),
                "agent_0",
                "Detect the bowl",
            ),
        ]
        for fault, target_agent, expected in cases:
            with self.subTest(fault=fault["fault_subtype"]):
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = TextObservationFaultInjector(
                        [fault], temp_dir
                    ).inject_assignments(
                        assignments, episode_id="21", step=0
                    )
                self.assertEqual(result.assignments[target_agent], expected)
                self.assertTrue(result.changed)

    def test_f7_and_f8_control_faults(self):
        faults = [
            _phase_fault(
                "ActionNoOpOrFalseSuccess",
                "ActionFault",
                false_success=True,
            ),
            _phase_fault(
                "MissingOrFalseVerification",
                "VerificationFault",
                fault_id="test_false_verification",
                verification_mode="false_positive",
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = TextObservationFaultInjector(
                faults, temp_dir
            ).inject_control(
                episode_id="21",
                step=0,
                observer="agent_1",
                requested_action="pick",
            )

            self.assertTrue(result.force_noop)
            self.assertTrue(result.false_success)
            self.assertTrue(result.verification_result)
            self.assertEqual(len(result.records), 2)
            self.assertIn(
                "action_success=false", result.records[0]["ground_truth_state"]
            )
            self.assertEqual(
                result.records[1]["ground_truth_state"],
                "verification_result=false",
            )

    def test_nonzero_observation_fault_is_rejected(self):
        fault = _phase_fault(
            "ObjectRecognitionError",
            "PerceptionFault",
            affected_object="cup",
            observed_object="bowl",
            injected_at_step=3,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "must use injected_at_step=0"):
                TextObservationFaultInjector([fault], temp_dir)

    def test_episode_21_example_schedule_matches_real_scene_format(self):
        schedule_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "config",
            "fault_injection",
            "object_localization_episode_21.json",
        )
        scene = (
            "Here are the descriptions of the scene:\n"
            'The height of "002_master_chef_can_:0000" is 0.6. '
            'The horizontal distance of "002_master_chef_can_:0000" to the '
            "nearest navigable point is 0.5.\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            injector = TextObservationFaultInjector.from_config(
                {
                    "enabled": True,
                    "schedule_path": schedule_path,
                    "log_dir": temp_dir,
                    "strict": True,
                }
            )
            result = injector.inject(
                scene, episode_id="21", step=0, observer="agent_1"
            )

            self.assertTrue(result.changed)
            self.assertIn(
                'located at "inside the cabinet"', result.scene_description
            )
            self.assertTrue(
                (
                    Path(temp_dir)
                    / "21"
                    / "PF_wrong_location_001.json"
                ).exists()
            )

    def test_hssd_fault_is_agent_specific_and_logged(self):
        scene = (
            "There are 2 objects in the scene.\n"
            'The height of "031_spoon_:0000" is 1.5. '
            'The horizontal distance of "031_spoon_:0000" to the nearest '
            "navigable point is 0.2.\n"
            'The height of "051_large_clamp_:0000" is 0.6.\n'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            injector = TextObservationFaultInjector([_fault()], temp_dir)

            healthy = injector.inject(
                scene, episode_id="2", step=0, observer="agent_0"
            )
            faulty = injector.inject(
                scene, episode_id="2", step=0, observer="agent_1"
            )

            self.assertEqual(healthy.scene_description, scene)
            self.assertFalse(healthy.changed)
            self.assertTrue(faulty.changed)
            self.assertIn(
                'located at "inside the cabinet"', faulty.scene_description
            )
            self.assertNotIn(
                '051_large_clamp_:0000" is located',
                faulty.scene_description,
            )

            record_path = (
                Path(temp_dir) / "2" / "PF_wrong_location_001.json"
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertTrue(record["ground_truth_state"].startswith("The height of"))
            self.assertEqual(
                record["agent_observed_state"],
                'The object "031_spoon_:0000" is located at '
                '"inside the cabinet".',
            )
            self.assertEqual(record["status"], "injected")

    def test_mp3d_fault_replaces_region_but_preserves_height(self):
        scene = (
            'The object "cup_0" is located in kitchen_1 on 0 floor. '
            'The height of object "cup_0" from the floor is 0.8. \n'
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            injector = TextObservationFaultInjector(
                [_fault(affected_object="cup_0")], temp_dir
            )

            result = injector.inject(
                scene, episode_id="2", step=0, observer="agent_1"
            )

            self.assertNotIn("kitchen_1", result.scene_description)
            self.assertIn(
                'located at "inside the cabinet"', result.scene_description
            )
            self.assertIn("from the floor is 0.8", result.scene_description)

    def test_strict_schedule_rejects_missing_object(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            injector = TextObservationFaultInjector([_fault()], temp_dir)

            with self.assertRaisesRegex(ValueError, "was not found"):
                injector.inject(
                    "There are no objects.",
                    episode_id="2",
                    step=0,
                    observer="agent_1",
                )

    def test_records_and_compares_complete_step_trajectories(self):
        def record_run(output_dir, positions):
            recorder = TrajectoryRecorder(output_dir)
            recorder.start_episode("21", "test_scene")
            for step, (agent_0_position, agent_1_position) in enumerate(
                positions
            ):
                batch = {
                    "agent_0_localization_sensor": np.array(
                        [[*agent_0_position, 0.1]], dtype=np.float32
                    ),
                    "agent_1_localization_sensor": np.array(
                        [[*agent_1_position, 0.2]], dtype=np.float32
                    ),
                }
                action_data = SimpleNamespace(
                    env_actions=np.array([[1.0, 0.0, 0.0, 1.0]]),
                    length_take_actions=[2, 2],
                    length_actions=[2, 2],
                    policy_info=[
                        {
                            "agent_0_cur_skill": "nav_to_obj",
                            "agent_1_cur_skill": "wait",
                        }
                    ],
                )
                record = recorder.make_step(
                    episode_id="21",
                    step=step,
                    batch=batch,
                    env_index=0,
                    action_data=action_data,
                )
                recorder.finish_step(
                    record,
                    episode_id="21",
                    reward=1.0,
                    done=step == len(positions) - 1,
                    info={"pddl_success": step == len(positions) - 1},
                )
            return Path(output_dir) / "episode_21.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_path = record_run(
                os.path.join(temp_dir, "baseline"),
                [
                    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    ((1.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
                ],
            )
            fault_path = record_run(
                os.path.join(temp_dir, "fault"),
                [
                    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    ((1.0, 0.0, 0.0), (0.0, 0.0, 3.0)),
                ],
            )
            comparison_dir = os.path.join(temp_dir, "comparison")
            comparison = compare_trajectories(
                str(baseline_path), str(fault_path), comparison_dir
            )

            self.assertEqual(comparison["baseline_num_steps"], 2)
            self.assertEqual(comparison["fault_num_steps"], 2)
            self.assertIsNone(
                comparison["agent_summary"]["agent_0"][
                    "first_divergent_step"
                ]
            )
            self.assertEqual(
                comparison["agent_summary"]["agent_1"][
                    "first_divergent_step"
                ],
                1,
            )
            self.assertAlmostEqual(
                comparison["agent_summary"]["agent_1"][
                    "max_position_distance"
                ],
                1.0,
            )
            self.assertTrue(
                (Path(comparison_dir) / "trajectory_comparison.json").exists()
            )
            self.assertTrue(
                (Path(comparison_dir) / "trajectory_comparison.csv").exists()
            )


if __name__ == "__main__":
    unittest.main()
