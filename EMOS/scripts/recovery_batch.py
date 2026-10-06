#!/usr/bin/env python3
"""Prepare, run and analyze paired clean/dropout/recovery experiments."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.fault_injection.recovery_batch import GROUPS, prepare_experiment
from habitat_mas.fault_injection.recovery_runner import run_batch
from habitat_mas.fault_injection.recovery_analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("prepare", help="Freeze the successful clean cohort and paired protocol")
    prepare.add_argument("--trace", required=True, type=Path)
    prepare.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/replica_cad/two_agent_perception_eval.json.gz")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--source-config", type=Path)
    prepare.add_argument("--fraction", type=float, default=.5)
    prepare.add_argument("--seed", type=int, default=66)
    prepare.add_argument("--simulator-seed", type=int, default=100)
    prepare.add_argument("--max-steps", type=int, default=750)
    prepare.add_argument("--repeats", type=int, default=1)
    prepare.add_argument("--agents", nargs="+", choices=("agent_0", "agent_1"), default=["agent_0", "agent_1"])
    prepare.add_argument("--wait-threshold", type=int, default=30)
    run = commands.add_parser("run", help="Resume valid results; stop on the first infrastructure error")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--groups", nargs="+", choices=GROUPS, help="Run only these groups; default: all three")
    run.add_argument("--python", default=sys.executable)
    run.add_argument("--limit-tasks", type=int, help="Pilot on the first N fixed tasks (selected groups included)")
    run.add_argument("--timeout", type=float, default=1800, help="Wall-clock seconds per process")
    run.add_argument("--retry-errors", action="store_true", help="Keep failed attempts and create a new attempt")
    run.add_argument("--dry-run", action="store_true", help="Write commands.json only; does not reserve attempt paths")
    report = commands.add_parser("analyze", help="Write episodes.csv, pairs.csv, summary.json and REPORT.md")
    report.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.action == "prepare":
            values = vars(args).copy()
            values.pop("action")
            manifest = prepare_experiment(**values)
            print(json.dumps(manifest["selection"], ensure_ascii=False))
            print(Path(manifest["output"]) / "manifest.json")
        elif args.action == "run":
            if json.loads(args.manifest.read_text()).get("validation_only"):
                raise ValueError("Cannot run a validation-only experiment with the real-model CLI; use a new formal manifest")
            return run_batch(args.manifest, args.python, limit_tasks=args.limit_tasks,
                             timeout=args.timeout, retry_errors=args.retry_errors, dry_run=args.dry_run, groups=args.groups)
        else:
            manifest = json.loads(args.manifest.read_text())
            summary = analyze(manifest)
            print(json.dumps(summary["states"], ensure_ascii=False))
            print(Path(manifest["output"]) / "analysis/REPORT.md")
        return 0
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
