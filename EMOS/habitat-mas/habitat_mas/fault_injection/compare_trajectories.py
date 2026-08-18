"""Command-line entry point for comparing two EMOS trajectory files."""

import argparse
import json

from .trajectory import compare_trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("fault")
    parser.add_argument("--output-dir", default="trajectory_comparison_output")
    args = parser.parse_args()
    result = compare_trajectories(args.baseline, args.fault, args.output_dir)
    print(json.dumps({
        "episode_id": result["episode_id"],
        "baseline_num_steps": result["baseline_num_steps"],
        "fault_num_steps": result["fault_num_steps"],
        "agent_summary": result["agent_summary"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
