import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm

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
from habitat_mas.fault_injection.agent_dropout import (
    AgentDropoutEventRecorder,
    AgentDropoutFaultSpec,
    FaultInjectionError,
)
from habitat_mas.fault_injection.agent_hang import (
    AgentHangEventRecorder,
    AgentHangFaultSpec,
)
from habitat_mas.fault_injection.trajectory import (
    AgentDropoutTrajectoryRecorder,
)
from habitat_mas.fault_injection.replay import CleanPrefixReplay

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
        agent_dropout_fault = (
            config.habitat_baselines.fault_injection.structure.agent_dropout
        )
        agent_hang_fault = (
            config.habitat_baselines.fault_injection.structure.agent_hang
        )
        dropout_spec = AgentDropoutFaultSpec.from_config(agent_dropout_fault)
        hang_spec = AgentHangFaultSpec.from_config(agent_hang_fault)
        dropout_spec.validate()
        hang_spec.validate()
        if dropout_spec.enabled and hang_spec.enabled:
            raise FaultInjectionError(
                "agent_dropout and agent_hang cannot both be enabled"
            )
        use_hang = hang_spec.enabled or (
            agent_hang_fault.record_trajectory and not dropout_spec.enabled
        )
        fault_spec = hang_spec if use_hang else dropout_spec
        fault_config = agent_hang_fault if use_hang else agent_dropout_fault
        fault_name = "agent_hang" if use_hang else "agent_dropout"
        replay_config = config.habitat_baselines.fault_injection.replay
        replay = CleanPrefixReplay(replay_config.path, atol=replay_config.atol) if replay_config.path else None
        if replay is not None:
            if (not fault_spec.enabled or not fault_config.record_trajectory
                    or config.habitat_baselines.num_environments != 1
                    or config.habitat_baselines.eval.evals_per_ep != 1
                    or getattr(fault_spec, "trigger_domain", "simulator_step") != "simulator_step"):
                raise ValueError("Replay requires a simulator-step fault, trajectory recording, one environment and one evaluation per episode")
            if not is_continuous_action_space(env_spec.action_space):
                raise ValueError("Clean prefix replay currently requires a continuous joint action space")
        if (
            fault_spec.enabled
            and fault_spec.agent not in config.habitat.simulator.agents
        ):
            raise FaultInjectionError(
                f"unknown {fault_name}.agent: {fault_spec.agent}"
            )
        recorder_type = (
            AgentHangEventRecorder if use_hang else AgentDropoutEventRecorder
        )
        fault_recorder = (
            recorder_type(fault_config.output_dir, fault_config.run_label)
            if fault_spec.enabled
            else None
        )
        trajectory_recorder = (
            AgentDropoutTrajectoryRecorder(
                fault_config.output_dir,
                fault_config.run_label,
                mode="faulty" if fault_spec.enabled else "clean",
                agent_names=config.habitat.simulator.agents.keys(),
                fault_name=fault_name,
            )
            if fault_config.record_trajectory
            else None
        )
        if trajectory_recorder is not None:
            envs.call(
                ["set_task_dynamic_state_capture"] * envs.num_envs,
                [{"enabled": True} for _ in range(envs.num_envs)],
            )
        simulator_steps = [0 for _ in range(envs.num_envs)]
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

            # Refresh context for every episode, including repeat evaluations
            # of the same dataset episode ID.
            if (
                current_episodes_info[0].episode_id != cur_ep_id
                or not not_done_masks.any().item()
            ):
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
                    if replay is not None:
                        ep = current_episodes_info[i]
                        schedule = fault_spec.simulator_step_schedule_map()
                        if ((schedule and str(ep.episode_id) not in schedule)
                                or (fault_spec.episode_id and str(ep.episode_id) != fault_spec.episode_id)):
                            raise ValueError(f"Replay episode {ep.episode_id} has no scheduled injection")
                        branch_step = schedule.get(str(ep.episode_id), fault_spec.simulator_step)
                        event = replay.start_episode(ep.scene_id, ep.episode_id, branch_step, config.habitat.simulator.agents.keys())
                        trajectory_recorder.record(event)
                        logger.info(f"Clean prefix replay: {event}")
                    if trajectory_recorder is not None:
                        trajectory_recorder.record_episode_start(
                            episode_id=current_episodes_info[i].episode_id,
                            scene_id=current_episodes_info[i].scene_id,
                            text_context=envs_text_context[i],
                        )

            space_lengths = {}
            replay_row = replay.prefix_row(simulator_steps[0]) if replay is not None else None
            n_agents = len(config.habitat.simulator.agents)
            if n_agents > 1:
                space_lengths = {
                    "index_len_recurrent_hidden_states": hidden_state_lens,
                    "index_len_prev_actions": action_space_lens,
                }
            with (replay.record_failures(trajectory_recorder) if replay_row is not None else nullcontext()), inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                    envs_text_context=envs_text_context,
                    agent_dropout_fault=agent_dropout_fault,
                    agent_hang_fault=agent_hang_fault,
                    simulator_steps=simulator_steps,
                    trajectory_replay=replay,
                    replay_row=replay_row,
                    record_replay_context=trajectory_recorder is not None,
                    **space_lengths,
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

            if (
                fault_recorder is not None
                and action_data.policy_info is not None
            ):
                for policy_info in action_data.policy_info:
                    for key, value in policy_info.items():
                        if key.endswith(("_dropout_event", "_hang_event")):
                            fault_recorder.record(value)

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

            if replay_row is not None:
                try:
                    replay.verify_action(
                        replay_row, step_data[0].tolist(),
                        action_data.policy_info[0] if action_data.policy_info is not None else {},
                    )
                except ValueError as error:
                    trajectory_recorder.record({**replay.report("replay_failed"), "error": str(error)})
                    raise

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            dynamic_state_snapshots = [
                info.pop("_dynamic_state_snapshot", None) for info in infos
            ]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(
                action_data, infos, dones
            )
            for i in range(len(policy_infos)):
                # Model memory belongs in the JSONL, not scalar metrics or
                # video overlays. Do not mutate action_data.policy_info.
                infos[i].update({
                    key: {field: item for field, item in value.items() if field != "replay_context"}
                    if key.endswith("_high_level_decision") and isinstance(value, dict) else value
                    for key, value in policy_infos[i].items()
                })

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
            trajectory_text_contexts = [None for _ in range(len(infos))]
            if trajectory_recorder is not None:
                refresh_text_context = [
                    not dones[i]
                    and trajectory_recorder.is_high_level_trace_event(
                        None
                        if action_data.policy_info is None
                        else action_data.policy_info[i]
                    )
                    for i in range(len(infos))
                ]
                if any(refresh_text_context):
                    latest_text_contexts = envs.call(
                        ["get_task_text_context"] * envs.num_envs
                    )
                    for i, should_refresh in enumerate(refresh_text_context):
                        if should_refresh:
                            trajectory_text_contexts[i] = (
                                latest_text_contexts[i]
                            )
            if trajectory_recorder is not None:
                for i in range(len(infos)):
                    trajectory_recorder.record_step(
                        episode_id=current_episodes_info[i].episode_id,
                        scene_id=current_episodes_info[i].scene_id,
                        simulator_step=simulator_steps[i],
                        joint_env_action=step_data[i],
                        policy_info=(
                            None
                            if action_data.policy_info is None
                            else action_data.policy_info[i]
                        ),
                        reward=rewards_l[i],
                        done=dones[i],
                        metrics=extract_scalars_from_info(infos[i]),
                        text_context=trajectory_text_contexts[i],
                        simulator_state=dynamic_state_snapshots[i],
                    )
            if replay_row is not None:
                try:
                    replay.verify_result(
                        replay_row, dynamic_state_snapshots[0], rewards_l[0], dones[0],
                        extract_scalars_from_info(infos[0]),
                    )
                except ValueError as error:
                    trajectory_recorder.record({**replay.report("replay_failed"), "error": str(error)})
                    raise
                if replay.verified_steps == replay.branch_step:
                    event = replay.report("replay_prefix_verified")
                    trajectory_recorder.record(event)
                    logger.info(f"Clean prefix verified; next step injects the fault: {event}")
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
                    simulator_steps[i] = 0

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
                else:
                    simulator_steps[i] += 1

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
            if envs_to_pause:
                agent.actor_critic.on_envs_pause(envs_to_pause)
                paused = set(envs_to_pause)
                simulator_steps = [
                    step
                    for env_i, step in enumerate(simulator_steps)
                    if env_i not in paused
                ]

        pbar.close()
        if fault_recorder is not None:
            scheduled_episode_ids = set(
                fault_spec.simulator_step_schedule_map()
            )
            if fault_recorder.count == 0:
                trigger_setting = (
                    "simulator_step" if use_hang
                    else "simulator_step / decision_step"
                )
                raise FaultInjectionError(
                    "evaluation completed without triggering the configured "
                    f"{fault_name}; check episode_id, agent, and "
                    f"{trigger_setting}"
                )
            missing_episode_ids = (
                scheduled_episode_ids - fault_recorder.episode_ids
            )
            if missing_episode_ids:
                raise FaultInjectionError(
                    f"evaluation did not trigger scheduled {fault_name} for "
                    "episode(s) "
                    f"{', '.join(sorted(missing_episode_ids))}; their "
                    "simulator steps were not reached"
                )
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
