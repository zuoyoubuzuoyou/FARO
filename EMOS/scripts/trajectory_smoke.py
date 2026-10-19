#!/usr/bin/env python3
"""Record real, short Habitat navigation runs without an LLM or API key.

Run with the Habitat environment's Python interpreter from the repository root.
The success case uses Habitat's geodesic follower; the failure case reaches the
real environment time limit; the error case submits an explicitly invalid test
action and records the original exception. No policy configuration is changed.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for package in ("habitat-lab", "habitat-baselines", "habitat-mas"):
    sys.path.insert(0, str(ROOT / package))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/trajectory-smoke")
    parser.add_argument("--scene", default="data/scene_datasets/habitat-test-scenes/van-gogh-room.glb")
    arguments = parser.parse_args()

    import habitat
    import habitat_sim
    import numpy as np
    from habitat.core.environments import RLTaskEnv
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
    from habitat_mas.utils.trajectory import (
        EvaluationTrace, begin_action, decision_scope, end_action, record_event,
    )
    from habitat_mas.utils.run_config import build_run_metadata

    scene = (ROOT / arguments.scene).resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"Real Habitat smoke scene is missing: {scene}")
    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(str(scene.with_suffix(".navmesh"))):
        raise RuntimeError(f"Cannot load real scene navmesh for {scene}")
    pathfinder.seed(7)
    for _ in range(100):
        start = np.asarray(pathfinder.get_random_navigable_point())
        goal = np.asarray(pathfinder.snap_point(start + np.array([0, 0, -0.75])))
        if not np.isfinite(goal).all():
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = goal
        if pathfinder.find_path(path) and 0.5 < path.geodesic_distance < 1.0:
            break
    else:
        raise RuntimeError("Cannot find a short navigable path in the real scene")

    summaries = []
    for case in ("success", "failure", "error"):
        config = habitat.get_config("benchmark/nav/pointnav/pointnav_habitat_test.yaml")
        with habitat.config.read_write(config):
            config.habitat.simulator.scene = str(scene)
            config.habitat.simulator.create_renderer = False
            config.habitat.simulator.requires_textures = False
            config.habitat.simulator.agents.main_agent.sim_sensors = {}
            config.habitat.task.success_measure = "success"
            config.habitat.environment.max_episode_steps = 2 if case == "failure" else 40
            config.habitat.environment.iterator_options.shuffle = False
        dataset = PointNavDatasetV1()
        dataset.episodes = [NavigationEpisode(
            episode_id=f"smoke-{case}", scene_id=str(scene),
            start_position=start.tolist(), start_rotation=[0, 0, 0, 1],
            goals=[NavigationGoal(position=goal.tolist())],
            info={"geodesic_distance": path.geodesic_distance},
        )]
        trace = EvaluationTrace(arguments.output, metadata={
            **build_run_metadata(config),
            "smoke_case": case, "simulator": "real Habitat-Sim",
            "controller": "ShortestPathFollower" if case == "success" else "diagnostic",
            "model_calls": 0,
        })
        caught = None
        with RLTaskEnv(config, dataset=dataset) as env:
            env.set_trajectory_recording(True)
            env.reset()
            trace.start_episode(
                0, env.current_episode(),
                context={"smoke_case": case, "goal_position": goal.tolist()},
                state=env.get_trajectory_state(),
            )
            follower = ShortestPathFollower(
                env.habitat_env.sim, goal_radius=0.1,
                return_one_hot=False, stop_on_error=False,
            )
            try:
                with trace:
                    for index in range(config.habitat.environment.max_episode_steps):
                        with trace.policy_step():
                            if index == 0:
                                with decision_scope("main_agent", phase="execution"):
                                    record_event("controller_decision", controller="ShortestPathFollower"
                                                 if case == "success" else "diagnostic", goal_position=goal)
                                    begin_action("main_agent", "navigate" if case == "success" else case,
                                                 {"goal_position": goal})
                            if case == "success":
                                action = int(follower.get_next_action(goal))
                            elif case == "failure":
                                action = "turn_left"
                            else:
                                action = "invalid_action_for_trajectory_error_smoke"
                            record_event("controller_action", agent_id="main_agent", action=action)
                            trace.submit_actions(0, {"action": action}, agent_ids=["main_agent"])
                            try:
                                _, reward, done, info = env.step(action={"action": action})
                            except Exception as error:
                                record_event(
                                    "execution_error", agent_id="main_agent", stage="simulator_step",
                                    attempted_action={"action": action},
                                    error={"type": type(error).__name__, "message": str(error)},
                                )
                                raise
                            if done:
                                end_action(
                                    "main_agent", "controller_terminated" if case == "success" else "timeout",
                                    reason="stop_requested" if case == "success" else "environment_time_limit",
                                    result={"success": bool(info["success"])},
                                )
                        state = info.pop("__trajectory_state")
                        trace.record_step(0, {"action": action}, reward, done, info, state,
                                          success=bool(info["success"]),
                                          termination_reason=state["termination_reason"])
                        if done:
                            break
            except Exception as error:
                caught = {"type": type(error).__name__, "message": str(error)}
                if case != "error":
                    raise
        events = [json.loads(line) for file in trace.run_dir.rglob("events.jsonl")
                  for line in file.read_text().splitlines()]
        summary = {
            "case": case, "run_dir": str(trace.run_dir),
            "steps": sum(event["event"] == "step" for event in events),
            "terminal": events[-1], "caught_exception": caught,
        }
        summaries.append(summary)
        if case == "success" and not any(event.get("metrics", {}).get("success") for event in events):
            raise AssertionError("Real navigation controller did not reach success")
        if case == "failure" and not any(event.get("state", {}).get("termination_reason") == "time_limit"
                                         for event in events):
            raise AssertionError("Real failure case did not reach the environment time limit")
        if case == "error" and caught is None:
            raise AssertionError("Real invalid simulator action did not raise")
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "smoke-summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps([
        {"case": item["case"], "steps": item["steps"], "run_dir": item["run_dir"],
         "status": item["terminal"]["status"], "caught_exception": item["caught_exception"]}
        for item in summaries
    ], indent=2))


if __name__ == "__main__":
    main()
