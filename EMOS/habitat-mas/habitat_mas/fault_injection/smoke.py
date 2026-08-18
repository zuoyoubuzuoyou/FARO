"""Offline runnable smoke test for all eight EMOS MVP fault types."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from habitat_mas.fault_injection.text_observation import (
    FAULT_DEFINITIONS,
    TextObservationFaultInjector,
)


SCHEDULES = (
    "object_localization_episode_21.json",
    "f2_object_recognition_episode_21.json",
    "f3_wrong_subgoal_episode_21.json",
    "f4_missing_subgoal_episode_21.json",
    "f5_stale_message_episode_21.json",
    "f6_duplicate_assignment_episode_21.json",
    "f7_action_noop_episode_21.json",
    "f8_false_verification_episode_21.json",
)

SCENE_DESCRIPTION = (
    "Here are the descriptions of the scene:\n"
    "There are 4 objects in the scene.\n"
    'The height of "002_master_chef_can_:0000" is 0.6. '
    'The horizontal distance of "002_master_chef_can_:0000" to the nearest '
    "navigable point is 0.5.\n"
    'The height of "072-d_toy_airplane_:0000" is 1.4. '
    'The horizontal distance of "072-d_toy_airplane_:0000" to the nearest '
    "navigable point is 0.1.\n"
)

ASSIGNMENTS = {
    "agent_0": "Navigate to the target objects",
    "agent_1": "Detect target objects",
}


def run_smoke(output_dir: Path) -> list[dict]:
    schedule_dir = Path(__file__).resolve().parents[2] / "config" / "fault_injection"
    summary = []
    for schedule_name in SCHEDULES:
        injector = TextObservationFaultInjector.from_config(
            {
                "enabled": True,
                "schedule_path": str(schedule_dir / schedule_name),
                "log_dir": str(output_dir / Path(schedule_name).stem),
                "strict": True,
            }
        )
        fault = injector.faults[0]
        phase = FAULT_DEFINITIONS[fault["fault_subtype"]]["phase"]
        if phase == "observation":
            result = injector.inject(
                SCENE_DESCRIPTION,
                episode_id="21",
                step=0,
                observer=fault["faulty_agent"],
            )
        elif phase == "assignment":
            result = injector.inject_assignments(
                ASSIGNMENTS,
                episode_id="21",
                step=0,
            )
        else:
            result = injector.inject_control(
                episode_id="21",
                step=0,
                observer=fault["faulty_agent"],
                requested_action="nav_to_obj",
            )
        summary.append(
            {
                "fault_id": fault["fault_id"],
                "fault_subtype": fault["fault_subtype"],
                "phase": phase,
                "status": "injected" if result.changed else "not_injected",
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("fault_injection_smoke_output"),
    )
    args = parser.parse_args()
    summary = run_smoke(args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "injected" for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
