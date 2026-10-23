#!/usr/bin/env python3
"""Replay the four EMOS tasks and evaluate against independent runtime labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "habitat-mas"))

from habitat_mas.detection.evaluation import load_annotations
from habitat_mas.detection.runner import load_dataset_episodes, run_evaluation


def load_environment(path=None):
    """Read local dotenv settings while preserving explicitly exported values."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Environment-only operation remains valid without this optional helper.
        return False
    return load_dotenv(path or ROOT / ".env", override=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/detection/emos.json")
    parser.add_argument("--output", type=Path, default=ROOT / "analysis_results/detection/emos")
    parser.add_argument("--methods", nargs="+", choices=("repeat_action", "team_progress", "llm"),
        default=["repeat_action", "team_progress"])
    parser.add_argument("--dataset", nargs="+", help="Task name or dataset batch identifier")
    parser.add_argument("--limit", type=int, help="Smoke test: first N episodes per dataset")
    parser.add_argument("--labels", type=Path, help="Independent human runtime annotation CSV")
    parser.add_argument("--sample-every", type=int, default=50)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--repeat-threshold", type=int, default=10)
    parser.add_argument("--min-stall-steps", type=int, default=100)
    parser.add_argument("--reset-budget", type=int, default=100)
    parser.add_argument("--model", default="gpt-6-sol")
    parser.add_argument("--max-llm-calls", type=int, help="Per-episode uncached API request budget")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_llm_calls is not None and args.max_llm_calls < 0:
        parser.error("--max-llm-calls cannot be negative")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if "llm" in args.methods:
        load_environment()
    config = {key: getattr(args, key) for key in (
        "sample_every", "window", "check_every", "repeat_threshold",
        "min_stall_steps", "reset_budget", "model", "max_llm_calls", "timeout")}
    episodes = load_dataset_episodes(args.config, ROOT)
    if args.dataset:
        episodes = [ep for ep in episodes if ep["dataset"] in args.dataset or ep["task"] in args.dataset]
    if args.limit:
        counts, selected = {}, []
        for ep in episodes:
            count = counts.get(ep["dataset"], 0)
            if count < args.limit:
                selected.append(ep)
                counts[ep["dataset"]] = count + 1
        episodes = selected
    if not episodes:
        parser.error("No episodes matched selection")
    labels = load_annotations(args.labels) if args.labels else None
    print(f"Replaying {len(episodes)} episodes with {', '.join(args.methods)}", flush=True)
    def progress(done, total, episode):
        print(f"[{done}/{total}] {episode['dataset']} episode={episode['episode_id']}",
            file=sys.stderr, flush=True)
    result = run_evaluation(episodes, config, args.output, args.methods,
        annotations=labels, workers=args.workers, progress=progress)
    print(json.dumps({"episodes": result["episode_count"], "output": str(args.output),
        "methods": result["overall"]["methods"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
