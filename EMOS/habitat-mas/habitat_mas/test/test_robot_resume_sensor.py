"""Robot resumes must cover every configured agent before task assignment."""

import json
import tempfile
import unittest
from pathlib import Path

import habitat  # Initialize Habitat before importing its MAS sensor dependency.
from habitat_mas.tasks.habitat_mas_sensors import RobotResumeSensor


class RobotResumeSensorTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.sensor = RobotResumeSensor.__new__(RobotResumeSensor)
        self.sensor.robot_resume_dir = str(self.root)
        (self.root / "SpotRobot_head_jaw.json").write_text(json.dumps({
            "robot_id": "SpotRobot_head_jaw", "robot_type": "SpotRobot",
            "perception": {"cameras_info": {"head_camera": {"height": 0.483}}},
        }))
        (self.root / "DJIDrone_default.json").write_text(json.dumps({
            "robot_id": "DJIDrone_default", "robot_type": "DJIDrone",
        }))

    def test_loads_both_agents_with_their_configured_ids(self):
        result = json.loads(self.sensor.get_observation([
            {"agent_idx": 1, "agent_type": "DJIDrone_default"},
            {"agent_idx": 0, "agent_type": "SpotRobot_head_jaw"},
        ]))
        self.assertEqual(set(result), {"agent_0", "agent_1"})
        self.assertEqual(result["agent_0"]["robot_type"], "SpotRobot")
        self.assertEqual(result["agent_1"]["robot_type"], "DJIDrone")
        self.assertEqual(
            result["agent_0"]["perception"]["cameras_info"]["head_camera"]["height"],
            0.48,
        )

    def test_missing_resume_identifies_agent_type_and_file(self):
        with self.assertRaises(FileNotFoundError) as caught:
            self.sensor.get_observation([
                {"agent_idx": 0, "agent_type": "SpotRobot_head_arm"},
                {"agent_idx": 1, "agent_type": "DJIDrone_default"},
            ])
        message = str(caught.exception)
        self.assertIn("agent_0", message)
        self.assertIn("SpotRobot_head_arm", message)
        self.assertIn(str(self.root / "SpotRobot_head_arm.json"), message)

    def test_missing_second_resume_does_not_return_a_partial_team(self):
        (self.root / "DJIDrone_default.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "agent_1"):
            self.sensor.get_observation([
                {"agent_idx": 0, "agent_type": "SpotRobot_head_jaw"},
                {"agent_idx": 1, "agent_type": "DJIDrone_default"},
            ])
