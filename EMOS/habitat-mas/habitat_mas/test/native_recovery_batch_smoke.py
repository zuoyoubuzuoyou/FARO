"""Native end-to-end batch validation with FixedModel, never model-quality data.

Example (from EMOS, in the habitat environment):
  PYTHONPATH=habitat-mas python -m habitat_mas.test.native_recovery_batch_smoke \
    --source-manifest fault_output/recovery_clean_success_seed66/manifest.json \
    --episode 295 --output fault_output/recovery_batch_native_check

Runs one clean and four paired fault conditions in real Habitat, using the
existing deterministic smoke model. Outputs are explicitly validation-only.
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

from habitat_mas.fault_injection.cli import build_parser, prepare_run
from habitat_mas.fault_injection.recovery_batch import prepare_experiment, write_json
from habitat_mas.fault_injection import recovery_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=60)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    matches = [t for t in source["tasks"] if t["eligible"] and t["episode_id"] == args.episode
               and (args.scene is None or t["scene_id"] == args.scene)]
    if len(matches) != 1:
        parser.error("Choose exactly one eligible episode; use --scene for duplicate IDs")
    task = matches[0]
    manifest = prepare_experiment(task["trace"], task["dataset"], args.output,
                                  max_steps=args.max_steps, seed=source["protocol"]["seed"],
                                  simulator_seed=source["protocol"]["simulator_seed"])
    manifest["validation_only"] = True
    manifest["model"] = "FixedModel: deterministic test double; no API calls"
    manifest_path = Path(manifest["output"]) / "manifest.json"
    write_json(manifest_path, manifest)
    (Path(manifest["output"]) / "VALIDATION_ONLY.txt").write_text(
        "Deterministic test-model results. Do not use for recovery performance claims.\n")
    original_builder = recovery_runner.build_command

    def native_command(manifest, task, group, agent, repeat, attempt, python):
        command = original_builder(manifest, task, group, agent, repeat, attempt, python)
        launch_args, overrides = build_parser(clean=group == "clean").parse_known_args(command[2:])
        _, evaluator_command, _ = prepare_run(launch_args, overrides)
        evaluator_command[3] = "habitat_mas.test.native_dropout_recovery_smoke"
        return evaluator_command

    with patch.object(recovery_runner, "build_command", native_command):
        code = recovery_runner.run_batch(manifest_path, sys.executable, timeout=180)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
