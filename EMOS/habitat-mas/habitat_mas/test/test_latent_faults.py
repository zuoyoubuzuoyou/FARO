import json
import tempfile
import unittest
from pathlib import Path

from habitat_mas.fault_injection.agent_dropout import FaultInjectionError
from habitat_mas.fault_injection.latent import (
    ActionNoopFaultSpec, WrongSubgoalFaultSpec, MessageDelayFaultSpec,
    ObjectLocalizationFaultSpec, FaultWindow, read_latent_spec,
)
from habitat_mas.fault_injection.launcher import normalize_fault
from habitat_mas.fault_injection.cli import build_parser, prepare_run


class LatentFaultSpecTest(unittest.TestCase):
    def test_half_open_window_and_repeated_episode_reset(self):
        spec = ActionNoopFaultSpec(enabled=True, simulator_step=4, duration_steps=3)
        window = FaultWindow(spec, "scene", "15")
        self.assertEqual(window.advance(3), [])
        self.assertFalse(window.active)
        self.assertEqual(window.advance(4)[0]["type"], "fault_injected")
        self.assertTrue(window.active)
        self.assertEqual(window.advance(6), [])
        self.assertTrue(window.active)
        self.assertEqual(window.advance(7)[0]["type"], "fault_expired")
        self.assertFalse(window.active)
        self.assertEqual(window.advance(8), [])
        self.assertEqual(FaultWindow(spec, "scene", "15").advance(4)[0]["type"], "fault_injected")

    def test_schedule_and_permanent_window(self):
        spec = ObjectLocalizationFaultSpec.from_config({
            "enabled": True, "simulator_step_schedule": "15@4;16@2",
            "position_offset": [2, 0, -1],
        })
        self.assertEqual(spec.target_simulator_step("16"), 2)
        self.assertEqual(FaultWindow(spec, "scene", "17").advance(4), [])
        window = FaultWindow(spec, "scene", "15")
        window.advance(4)
        window.advance(1000)
        self.assertTrue(window.active)

    def test_invalid_parameters(self):
        cases = [
            (ActionNoopFaultSpec, {"duration_steps": 0}),
            (ActionNoopFaultSpec, {"duration_steps": -2}),
            (WrongSubgoalFaultSpec, {"object": "cup"}),
            (WrongSubgoalFaultSpec, {"object": "any_targets|1", "replacement_object": "any_targets|1"}),
            (MessageDelayFaultSpec, {"delay_steps": 0}),
            (ObjectLocalizationFaultSpec, {"position_offset": [0, 0, 0]}),
            (ObjectLocalizationFaultSpec, {"position_offset": [1, float("nan"), 0]}),
            (ObjectLocalizationFaultSpec, {"position_offset": [1, 0]}),
            (ActionNoopFaultSpec, {"agent": "drone"}),
        ]
        for cls, values in cases:
            with self.subTest(cls=cls, values=values), self.assertRaises(FaultInjectionError):
                cls.from_config({"enabled": True, **values}).validate()

    def test_one_enabled_latent_spec(self):
        config = {"action": {"action_noop": {"enabled": True}}}
        self.assertIsInstance(read_latent_spec(config), ActionNoopFaultSpec)
        config["communication"] = {"message_delay": {"enabled": True}}
        with self.assertRaisesRegex(FaultInjectionError, "one"):
            read_latent_spec(config)

    def test_aliases_and_cli_records_parameters(self):
        for name in ("action_noop", "wrong_subgoal", "message_delay", "object_localization"):
            self.assertEqual(normalize_fault(name.replace("_", "-")), name)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "episodes.json"
            dataset.write_text(json.dumps({"episodes": [{"scene_id": "scene", "episode_id": "15"}]}))
            args = build_parser().parse_args([
                "--fault", "object_localization", "--episode", "15", "--step", "4",
                "--duration-steps", "3", "--object", "any_targets|0",
                "--position-offset", "2", "0", "-1", "--dry-run",
                "--dataset", str(dataset), "--max-episode-steps", "10",
                "--output-dir", directory, "--run-label", "localization",
            ])
            _, command, plan = prepare_run(args, [])
            self.assertEqual(plan["fault_parameters"]["position_offset"], [2.0, 0.0, -1.0])
            self.assertEqual(plan["fault_parameters"]["duration_steps"], 3)
            self.assertTrue(any("perception.object_localization.position_offset=" in item for item in command))


if __name__ == "__main__":
    unittest.main()
