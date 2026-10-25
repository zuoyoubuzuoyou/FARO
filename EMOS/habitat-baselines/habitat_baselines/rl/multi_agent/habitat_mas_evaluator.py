import os
import sys
from copy import deepcopy
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat_mas.utils.trajectory import EvaluationTrace, redact
from habitat_mas.utils.run_config import build_run_metadata

from habitat import logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (
    observations_to_image,
    overlay_frame,
)
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import extract_scalars_from_info

class HabitatMASEvaluator(Evaluator):
    """
    Evaluator for Habitat environments.
    """

    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        args = (
            agent, envs, config, checkpoint_index, step_id, writer, device,
            obs_transforms, env_spec, rank0_keys,
        )
        eval_config = config.habitat_baselines.eval
        recovery_config = getattr(eval_config, "recovery", None)
        recovery_enabled = bool(getattr(recovery_config, "enabled", False))
        if recovery_enabled and envs.num_envs != 1:
            raise ValueError("Online recovery currently requires a single environment")
        if not eval_config.save_trajectories and not recovery_enabled:
            return self._evaluate_agent(*args)

        try:
            metadata = build_run_metadata(config, checkpoint_index, step_id)
        except Exception as error:
            logger.warning("Could not capture trajectory run metadata: %s", redact(str(error)))
            metadata = {"missing_fields": {"run_metadata": type(error).__name__}}
        try:
            trajectory = EvaluationTrace(eval_config.trajectory_dir, metadata)
        except Exception as error:
            if recovery_enabled:
                raise RuntimeError("Online recovery requires trajectory recording") from error
            logger.warning("Could not initialize trajectory recording: %s", redact(str(error)))
            return self._evaluate_agent(*args)
        with trajectory:
            recovery = None
            if recovery_enabled:
                from habitat_mas.recovery.online import OnlineRecovery

                task_hint = getattr(recovery_config, "task", "auto")
                if task_hint == "auto":
                    task_hint = getattr(config.habitat.task, "task_spec", "auto") or "auto"
                recovery = OnlineRecovery(
                    recovery_config, trajectory.run_dir / "recovery", task=task_hint,
                    envs=envs,
                )
                trajectory.event_callback = recovery.on_event
            logger.info("Saving structured trajectories to %s", trajectory.run_dir)
            try:
                try:
                    envs.call(
                        ["set_trajectory_recording"] * envs.num_envs,
                        [{"enabled": True} for _ in range(envs.num_envs)],
                    )
                except Exception as error:
                    if recovery_enabled:
                        raise RuntimeError("Online recovery requires state capture setup") from error
                    logger.warning("Could not enable trajectory state capture: %s", redact(str(error)))
                return self._evaluate_agent(*args, trajectory=trajectory, recovery=recovery)
            finally:
                # Cleanup must not replace an exception from the evaluated task.
                had_error = sys.exc_info()[0] is not None
                cleanup_error = None
                try:
                    if recovery is not None:
                        recovery.close()
                except Exception as error:
                    cleanup_error = error
                    logger.error("Could not restore recovery source revisions: %s", redact(str(error)))
                try:
                    if envs.num_envs:
                        envs.call(
                            ["set_trajectory_recording"] * envs.num_envs,
                            [{"enabled": False} for _ in range(envs.num_envs)],
                        )
                except Exception as error:
                    logger.warning("Could not disable trajectory recording: %s", redact(str(error)))
                if cleanup_error is not None and not had_error:
                    raise RuntimeError("Could not restore recovery source revisions") from cleanup_error

    def _evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
        trajectory=None,
        recovery=None,
    ):
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                *agent.actor_critic.hidden_state_shape,
            ),
            device=device,
        )

        hidden_state_lens = agent.actor_critic.hidden_state_shape_lens
        action_space_lens = agent.actor_critic.policy_action_space_shape_lens

        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            *agent.masks_shape,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        if len(config.habitat_baselines.eval.image_option) > 0:
            os.makedirs(config.habitat_baselines.image_dir, exist_ok=True)

        if len(config.habitat_baselines.eval.video_option) > 0:
            # Add the first frame of the episode to the video.
            rgb_frames: List[List[np.ndarray]] = [
                [
                    observations_to_image(
                        {k: v[env_idx] for k, v in batch.items() if
                             k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"}, {}, config,
                        0,
                    )
                ]
                for env_idx in range(config.habitat_baselines.num_environments)
            ]
            if config.habitat_baselines.eval.generate_fourth_rgb:
                rgb_frames_fourth: List[List[np.ndarray]] = [
                    [
                        observations_to_image(
                            {k: v[env_idx] for k, v in batch.items()if
                                 k == "agent_0_fourth_rgb"}, {}, config, 0,
                        )
                    ]
                    for env_idx in range(config.habitat_baselines.num_environments)
                ]
        else:
            rgb_frames_fourth = None
            rgb_frames = None

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"
        envs_text_context = {}
        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        cur_ep_id = -1
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            # If all prev_actions are zero, meaning this is the start of an episode
            # Then collect the context of the episode
            if current_episodes_info[0].episode_id != cur_ep_id:
                cur_ep_id = current_episodes_info[0].episode_id
                print("===============================================================================")
                print("=================================Episode ID====================================")
                print("Current Episode ID: ", cur_ep_id)
                print("=================================Episode ID====================================")
                print("===============================================================================")
                envs_text_context = envs.call(["get_task_text_context"] * envs.num_envs)
                if 'pddl_text_goal' in batch:
                    envs_pddl_text_goal_np = batch['pddl_text_goal'].cpu().numpy()
                    for i in range(envs.num_envs):
                        pddl_text_goal_np = envs_pddl_text_goal_np[i, ...]
                        envs_text_context[i]['pddl_text_goal'] = ''.join(str(pddl_text_goal_np, encoding='UTF-8'))

                for i in range(envs.num_envs):
                    # also add the debug/ logging info to the text context for convenience
                    envs_text_context[i]['episode_id'] = current_episodes_info[i].episode_id

            if trajectory is not None:
                # Logger lifecycle is independent of the policy's context cache.
                for i, episode in enumerate(current_episodes_info):
                    if trajectory.has_episode(i):
                        continue
                    context = {"episode_id": episode.episode_id,
                               "scene_id": episode.scene_id}
                    try:
                        context.update(deepcopy(envs.call_at(i, "get_task_text_context")))
                        if "pddl_text_goal" in batch:
                            goal = batch["pddl_text_goal"][i].cpu().numpy()
                            context["pddl_text_goal"] = str(goal, encoding="UTF-8")
                    except Exception as error:
                        context["missing_fields"] = {"task_context": type(error).__name__}
                        logger.warning("Could not capture trajectory task context: %s", error)
                    context["episode_id"] = episode.episode_id
                    context["scene_id"] = episode.scene_id
                    state = None
                    try:
                        state = envs.call_at(i, "get_trajectory_state")
                    except Exception as error:
                        if recovery is not None:
                            raise RuntimeError("Online recovery requires initial simulator state") from error
                        logger.warning("Could not capture initial trajectory state: %s", error)
                    if recovery is not None and not isinstance(state, dict):
                        raise RuntimeError("Online recovery requires initial simulator state")
                    try:
                        trajectory.start_episode(i, episode, context=context, state=state)
                    except Exception as error:
                        if recovery is not None:
                            raise RuntimeError("Online recovery could not start episode recording") from error
                        logger.warning("Could not start trajectory episode: %s", error)

            space_lengths = {}
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with inference_mode(), (
                trajectory.policy_step(observations=batch)
                if trajectory is not None else nullcontext()
            ):
                def policy_attempt():
                    return agent.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                        envs_text_context=envs_text_context,
                        **space_lengths,
                    )

                action_data = (
                    recovery.run_policy(policy_attempt, agent.actor_critic,
                        tensors=(test_recurrent_hidden_states, prev_actions, not_done_masks))
                    if recovery is not None else policy_attempt()
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = (
                        action_data.rnn_hidden_states
                    )
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    agent.actor_critic.update_hidden_state(
                        test_recurrent_hidden_states, prev_actions, action_data
                    )

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            if trajectory is not None:
                agent_ids = trajectory.metadata.get("agents_order")
                for i, action in enumerate(step_data):
                    if not trajectory.has_episode(i):
                        continue
                    trajectory.submit_actions(
                        i, action, action_lengths=action_space_lens,
                        agent_ids=agent_ids,
                    )
            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            # Captured inside the worker before VectorEnv's automatic reset.
            trajectory_states = [
                info.pop("__trajectory_state", None) for info in infos
            ]
            state_errors = [
                info.pop("__trajectory_state_error", None) for info in infos
            ]
            if recovery is not None and (
                any(error is not None for error in state_errors)
                or any(not isinstance(state, dict) for state in trajectory_states)
            ):
                raise RuntimeError("Online recovery requires returned simulator state for every step")
            if trajectory is not None:
                for i, error in enumerate(state_errors):
                    if error is not None:
                        trajectory.record_environment_event(
                            i, "state_capture_error", error=error,
                            error_origin="logging", handling="continue_without_snapshot",
                        )
            # Read the configured original measure before policy extras can
            # overwrite an identically named metric. Do not infer success.
            original_successes = [None] * len(infos)
            if trajectory is not None:
                task = getattr(config.habitat, "task", None)
                try:
                    success_measure = getattr(task, "success_measure", None)
                except Exception as error:
                    success_measure = None
                    logger.warning("Could not read configured success measure: %s", error)
                if isinstance(success_measure, str) and success_measure:
                    original_successes = [info.get(success_measure) for info in infos]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            if trajectory is not None:
                for i in range(len(infos)):
                    if not trajectory.has_episode(i):
                        continue
                    trajectory.record_step(
                        i, action=step_data[i], reward=rewards_l[i],
                        done=dones[i], metrics=infos[i], state=trajectory_states[i],
                        success=original_successes[i],
                        termination_reason=(trajectory_states[i] or {}).get("termination_reason"),
                    )
            if recovery is not None:
                # The simulator remains live while diagnosis/generation runs.
                # Terminal frames are audited; VectorEnv has already reset them.
                recovery.advance(agent.actor_critic, dones)

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            ).repeat(1, *agent.masks_shape)

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v for k, v in infos[i].items() if k not in rank0_keys
                }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()if
                         k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"}, disp_info,
                        config, len(rgb_frames[0]),
                        episode_id=current_episodes_info[i].episode_id,
                    )
                    if config.habitat_baselines.eval.generate_fourth_rgb:
                        frame_fourth = observations_to_image(
                            {k: v[i] for k, v in batch.items() if
                             k == "agent_0_fourth_rgb"}, infos[i],
                            config, len(rgb_frames_fourth[0]),
                            episode_id=current_episodes_info[i].episode_id,
                        )
                    if not not_done_masks[i].any().item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()if
                             k != "agent_0_fourth_rgb" and k != "agent_1_fourth_rgb"},
                            disp_info, config,
                            frame_id=len(rgb_frames[0]),
                            episode_id=current_episodes_info[i].episode_id,
                        )
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            final_frame_fourth = observations_to_image(
                                {k: v[i] for k, v in batch.items() if
                                 k == "agent_0_fourth_rgb"}, infos[i],
                                config, len(rgb_frames_fourth[0]),
                                episode_id=current_episodes_info[i].episode_id,
                            )
                        final_frame = overlay_frame(final_frame, disp_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            final_frame_fourth = overlay_frame(final_frame_fourth, infos[i])
                            rgb_frames_fourth[i].append(final_frame_fourth)
                            rgb_frames_fourth[i].append(frame_fourth)
                    else:
                        frame = overlay_frame(frame, disp_info)
                        rgb_frames[i].append(frame)
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            frame_fourth = overlay_frame(frame_fourth, infos[i])
                            rgb_frames_fourth[i].append(frame_fourth)

                # episode ended
                if not not_done_masks[i].any().item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    # clear the prev_actions and recurrent_hidden_states
                    prev_actions[i] = 0
                    test_recurrent_hidden_states[i] = 0

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            generate_video(
                                video_option=config.habitat_baselines.eval.video_option,
                                video_dir=config.habitat_baselines.video_dir,
                                images=rgb_frames_fourth[i][:-1],
                                episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}_fourth",
                                checkpoint_idx=checkpoint_index,
                                metrics=extract_scalars_from_info(disp_info),
                                fps=config.habitat_baselines.video_fps,
                                tb_writer=writer,
                                keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                            )
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        if config.habitat_baselines.eval.generate_fourth_rgb:
                            rgb_frames_fourth[i] = rgb_frames_fourth[i][-1:]
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            if trajectory is not None:
                trajectory.pause_envs(envs_to_pause)
            if recovery is not None and envs_to_pause:
                # Source rollback needs the worker's active RPC connection.
                recovery.close()
            not_done_masks = not_done_masks.to(device=device)
            (
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = pause_envs(
                envs_to_pause,
                envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

            # We pause the statefull parameters in the policy.
            # We only do this if there are envs to pause to reduce the overhead.
            # In addition, HRL policy requires the solution_actions to be non-empty, and
            # empty list of envs_to_pause will raise an error.
            if any(envs_to_pause):
                agent.actor_critic.on_envs_pause(envs_to_pause)

        pbar.close()
        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())
        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)
