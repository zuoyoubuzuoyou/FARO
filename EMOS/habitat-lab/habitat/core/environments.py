#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@habitat.registry.register_env(name="myEnv")` for reusability
"""

import importlib
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Type, Union

import gym
import numpy as np

import habitat
from habitat import Dataset
from habitat.core.trajectory_diagnostics import collect_runtime_diagnostics
from habitat.gym.gym_wrapper import HabGymWrapper

if TYPE_CHECKING:
    from omegaconf import DictConfig


RLTaskEnvObsType = Union[np.ndarray, Dict[str, np.ndarray]]


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return habitat.registry.get_env(env_name)

@habitat.registry.register_env(name="RLTaskEnv")
class RLTaskEnv(habitat.RLEnv):
    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        super().__init__(config, dataset)
        self._reward_measure_name = self.config.task.reward_measure
        self._success_measure_name = self.config.task.success_measure
        self._slack_reward = self.config.task.slack_reward
        self._success_reward = self.config.task.success_reward
        self._end_on_success = self.config.task.end_on_success
        assert (
            self._reward_measure_name is not None
        ), "The key task.reward_measure cannot be None"
        assert (
            self._success_measure_name is not None
        ), "The key task.success_measure cannot be None"

    def reset(
        self, *args, return_info: bool = False, **kwargs
    ) -> Union[RLTaskEnvObsType, Tuple[RLTaskEnvObsType, Dict]]:
        if getattr(self, "_trajectory_recording_enabled", False):
            self._trajectory_executed_action = None
            self.__dict__.pop("_trajectory_executed_action_error", None)
        return super().reset(*args, return_info=return_info, **kwargs)

    def step(
        self, *args, **kwargs
    ) -> Tuple[RLTaskEnvObsType, float, bool, dict]:
        if getattr(self, "_trajectory_recording_enabled", False):
            # HabGymWrapper has already converted the policy vector into the
            # Habitat action dictionary at this boundary. Detach before the
            # simulator consumes it, since actions can contain mutable arrays.
            action = args[0] if args else kwargs.get("action")
            self.__dict__.pop("_trajectory_executed_action_error", None)
            try:
                self._trajectory_executed_action = self._trajectory_value(action)
            except Exception as error:
                self._trajectory_executed_action = None
                self._trajectory_executed_action_error = (
                    f"action_copy_failed: {type(error).__name__}: {error}"
                )
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        # We don't know what the reward measure is bounded by
        return (-np.inf, np.inf)

    def get_reward(self, observations):
        current_measure = self._env.get_metrics()[self._reward_measure_name]
        reward = self._slack_reward

        reward += current_measure

        if self._episode_success():
            reward += self._success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        if self._end_on_success and self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        metrics = self._env.get_metrics()
        if not getattr(self, "_trajectory_recording_enabled", False):
            return metrics
        info = dict(metrics)
        try:
            info["__trajectory_state"] = self.get_trajectory_state(observations)
        except Exception as error:
            # Observation-only failures must not interrupt the real step or
            # replace its reward, done flag, metrics, or original exception.
            info["__trajectory_state"] = None
            info["__trajectory_state_error"] = {
                "type": type(error).__name__, "message": str(error),
                "handling": "continue_without_snapshot",
            }
        return info

    def set_trajectory_recording(self, enabled: bool) -> None:
        """Opt in to simulator snapshots in step info, before vector auto-reset."""
        self._trajectory_recording_enabled = bool(enabled)
        if not enabled:
            self.__dict__.pop("_trajectory_executed_action", None)
            self.__dict__.pop("_trajectory_executed_action_error", None)

    @staticmethod
    def _trajectory_value(value):
        """Detach action values without retaining simulator or tensor objects."""
        import math
        from collections.abc import Mapping

        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, Mapping):
            return {str(key): RLTaskEnv._trajectory_value(item)
                    for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RLTaskEnv._trajectory_value(item) for item in value]
        for method_name in ("detach", "cpu"):
            method = getattr(value, method_name, None)
            if callable(method):
                value = method()
        for method_name in ("tolist", "item"):
            method = getattr(value, method_name, None)
            if callable(method):
                return RLTaskEnv._trajectory_value(method())
        return None

    def _trajectory_targets(self, simulator, missing):
        """Read rearrange targets using their relative scene-object indices."""
        get_targets = getattr(simulator, "get_targets", None)
        if not callable(get_targets):
            missing["target_objects"] = "simulator_does_not_support_get_targets"
            return None
        try:
            indices, goals = get_targets()
        except Exception as error:
            missing["target_objects"] = f"get_targets_failed: {type(error).__name__}: {error}"
            return None
        positions = None
        get_positions = getattr(simulator, "get_scene_pos", None)
        if callable(get_positions):
            try:
                positions = get_positions()
            except Exception as error:
                missing["target_objects.position"] = f"get_scene_pos_failed: {type(error).__name__}: {error}"
        else:
            missing["target_objects.position"] = "simulator_does_not_support_get_scene_pos"
        object_ids = getattr(simulator, "scene_obj_ids", None)
        if object_ids is None:
            missing["target_objects.object_id"] = "simulator_does_not_expose_scene_obj_ids"
        manager = None
        get_manager = getattr(simulator, "get_rigid_object_manager", None)
        if callable(get_manager):
            try:
                manager = get_manager()
            except Exception as error:
                missing["target_objects.handle"] = f"get_rigid_object_manager_failed: {type(error).__name__}: {error}"
        else:
            missing["target_objects.handle"] = "simulator_does_not_support_get_rigid_object_manager"
        targets = []
        for target_index, scene_index in enumerate(indices):
            scene_index = int(scene_index)
            target = {
                "scene_index": scene_index,
                "object_id": None,
                "handle": None,
                "position": None,
                "goal_position": [float(value) for value in goals[target_index]],
            }
            try:
                if positions is not None:
                    target["position"] = [float(value) for value in positions[scene_index]]
                if object_ids is not None:
                    target["object_id"] = int(object_ids[scene_index])
                if manager is not None and target["object_id"] is not None:
                    obj = manager.get_object_by_id(target["object_id"])
                    handle = getattr(obj, "handle", None)
                    if handle is not None:
                        target["handle"] = str(handle)
                    else:
                        missing[f"target_objects.{target_index}.handle"] = "rigid_object_has_no_handle"
            except Exception as error:
                missing[f"target_objects.{target_index}"] = f"object_state_failed: {type(error).__name__}: {error}"
            targets.append(target)
        return targets

    def _trajectory_progress(self, missing):
        """Keep small numeric task metrics, omitting map and image payloads."""
        def copy_metric(value, path):
            if isinstance(value, dict):
                result = {}
                for key, item in value.items():
                    copied = copy_metric(item, f"{path}.{key}")
                    if copied is not omit:
                        result[str(key)] = copied
                return result if result or not value else omit
            shape = getattr(value, "shape", None)
            if shape is not None and len(shape) > 0:
                missing[path] = "nonscalar_metric_omitted"
                return omit
            if isinstance(value, (list, tuple)):
                missing[path] = "nonscalar_metric_omitted"
                return omit
            copied = self._trajectory_value(value)
            if copied is None and value is not None:
                missing[path] = "unsupported_or_nonfinite_metric"
            return copied

        omit = object()
        return copy_metric(self._env.get_metrics(), "task_progress")

    def get_trajectory_state(self, observations=None) -> dict:
        """Return a detached, JSON-safe snapshot of the current simulator state.

        Articulated poses come from robot bases rather than sensor agents.
        All base rotations are quaternions in ``[x, y, z, w]`` order,
        preserving the full orientation and the direction of rotation.
        End-effector positions and grasped object IDs are null when absent.
        """
        simulator = self._env.sim
        simulator_config = self.config.simulator
        agents_manager = getattr(simulator, "agents_mgr", None)
        missing = {}
        agents = {}
        for index, name in enumerate(simulator_config.agents_order):
            agent_config = simulator_config.agents[name]
            agent_state = {
                "ee_position": None,
                "grasped_object_id": None,
            }
            if agents_manager is not None:
                agent_data = agents_manager[index]
                robot = agent_data.articulated_agent
                rotation = robot.sim_obj.rotation
                agent_state.update(
                    robot_type=str(
                        agent_config.articulated_agent_type
                        or type(robot).__name__
                    ),
                    base_position=[float(value) for value in robot.base_pos],
                    base_rotation=[float(value) for value in rotation.vector]
                    + [float(rotation.scalar)],
                )
                if robot.params.ee_links:
                    agent_state["ee_position"] = [
                        float(value)
                        for value in robot.ee_transform().translation
                    ]
                else:
                    missing[f"agents.{name}.ee_position"] = "robot_has_no_end_effector"
                if agent_data.grasp_mgrs:
                    object_id = agent_data.grasp_mgr.snap_idx
                    if object_id is not None:
                        agent_state["grasped_object_id"] = int(object_id)
                else:
                    missing[f"agents.{name}.grasped_object_id"] = "agent_has_no_grasp_manager"
            else:
                simulator_state = simulator.get_agent_state(index)
                rotation = simulator_state.rotation
                agent_state.update(
                    robot_type=str(
                        agent_config.articulated_agent_type
                        or type(simulator).__name__
                    ),
                    base_position=[
                        float(value) for value in simulator_state.position
                    ],
                    base_rotation=[
                        float(rotation.x),
                        float(rotation.y),
                        float(rotation.z),
                        float(rotation.w),
                    ],
                )
                missing[f"agents.{name}.ee_position"] = "non_articulated_agent"
                missing[f"agents.{name}.grasped_object_id"] = "non_articulated_agent"
            agents[str(name)] = agent_state

        episode_over = bool(self.get_done(None))
        termination_reason = None
        if episode_over:
            if self._end_on_success and self._episode_success():
                termination_reason = "success"
            elif self._env._past_limit():
                termination_reason = "time_limit"
            else:
                termination_reason = "task_inactive"
        episode = self._env.current_episode
        executed_action = getattr(self, "_trajectory_executed_action", None)
        if executed_action is None:
            missing["executed_action"] = getattr(
                self, "_trajectory_executed_action_error",
                "no_recorded_step_since_reset",
            )
        target_objects = self._trajectory_targets(simulator, missing)
        task_progress = self._trajectory_progress(missing)
        runtime = collect_runtime_diagnostics(
            simulator=simulator, task=getattr(self._env, "task", None),
            observations=observations, agent_names=list(agents),
            target_objects=target_objects, env_step=int(self._env._elapsed_steps),
            executed_action=executed_action, agent_configs=simulator_config.agents,
        )
        for name, diagnostics in runtime["agents"].items():
            agents[name].update(diagnostics)
        target_objects = runtime["target_objects"]
        return {
            "episode_id": str(episode.episode_id),
            "scene_id": str(episode.scene_id),
            "num_steps": int(self._env._elapsed_steps),
            "agents": agents,
            "episode_over": episode_over,
            "termination_reason": termination_reason,
            "executed_action": self._trajectory_value(executed_action),
            "target_objects": target_objects,
            "task_progress": task_progress,
            "current_goal": runtime["current_goal"],
            "missing_fields": missing,
        }

    def get_task_text_context(self) -> dict:
        """
        Get the text context of the task.
        """
        assert hasattr(self._env.task, "get_task_text_context"), (
            "The task does not have a get_task_text_context method. "
            "Please implement this method in the task class."
        )
        return self._env.task.get_task_text_context()

    def apply_runtime_source_patch(self, manifest: dict) -> dict:
        """Install methods on existing worker objects without recreating the env.

        The VectorEnv RPC dispatcher propagates uncaught exceptions out of its
        worker loop, so patch failures must cross this boundary as data.
        """
        try:
            if not hasattr(self, "_runtime_source_patcher"):
                from habitat_mas.recovery.source_patch import RuntimeSourcePatcher
                from habitat_mas.recovery.source_runtime import REPOSITORY_ROOT
                self._runtime_source_patcher = RuntimeSourcePatcher(REPOSITORY_ROOT)
            return self._runtime_source_patcher.install(manifest)
        except Exception as error:
            from habitat_mas.utils.trajectory import redact
            return {"status": "error", "error": redact(str(error)),
                    "error_type": type(error).__name__}

    def check_runtime_source_rollback(self, transaction_id: str) -> dict:
        """Validate worker rollback before the parent changes any component."""
        try:
            if not hasattr(self, "_runtime_source_patcher"):
                raise ValueError("No runtime source patch has been installed")
            return self._runtime_source_patcher.check_rollback(transaction_id)
        except Exception as error:
            from habitat_mas.utils.trajectory import redact
            return {"status": "error", "error": redact(str(error)),
                    "error_type": type(error).__name__}

    def rollback_runtime_source_patch(self, transaction_id: str) -> dict:
        """Revert a worker revision through the existing public RPC path."""
        try:
            if not hasattr(self, "_runtime_source_patcher"):
                raise ValueError("No runtime source patch has been installed")
            return self._runtime_source_patcher.rollback(transaction_id)
        except Exception as error:
            from habitat_mas.utils.trajectory import redact
            return {"status": "error", "error": redact(str(error)),
                    "error_type": type(error).__name__}


@habitat.registry.register_env(name="GymRegistryEnv")
class GymRegistryEnv(gym.Wrapper):
    """
    A registered environment that wraps a gym environment to be
    used with habitat-baselines
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        for dependency in config["env_task_gym_dependencies"]:
            importlib.import_module(dependency)
        env_name = config["env_task_gym_id"]
        gym_env = gym.make(env_name)
        super().__init__(gym_env)


@habitat.registry.register_env(name="GymHabitatEnv")
class GymHabitatEnv(gym.Wrapper):
    """
    A registered environment that wraps a RLTaskEnv with the HabGymWrapper
    to use the default gym API.
    """

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset] = None
    ):
        base_env = RLTaskEnv(config=config, dataset=dataset)
        env = HabGymWrapper(env=base_env)
        super().__init__(env)
