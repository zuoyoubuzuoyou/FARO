""" 
Comment by Junting: 
The logic behind the design of use OraclePixelXXXPolicy as a dummy policy to pass the arguments to PixelXXXAction classes:
Due to the multi-processing nature of habitat framework, the process/thread running policies/ agents are different from the process/thread running the simulator.
The thread running the simulator has the full access to GT information of agents and actions. 
Thus we put all the data processing logics into the PixelXXXAction classes, which have full access to simulator GT information.
"""
from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import torch
from habitat_baselines.rl.hrl.skills.nn_skill import NnSkillPolicy
from habitat_baselines.rl.hrl.skills.oracle_arm_policies import (
    OraclePickPolicy, 
    OraclePlacePolicy
)
from habitat_baselines.rl.hrl.skills.oracle_nav import OracleNavPolicy
from habitat_baselines.rl.hrl.utils import find_action_range
from habitat_baselines.rl.ppo.policy import PolicyActionData

import numpy as np

RELEASE_ID = 0
PICK_ID = 1
PLACE_ID = 2

class OraclePixelPickPolicy(OraclePickPolicy):
    """
    Skill to generate a picking motion based on a pixel position.
    Moves the arm to the 3D position corresponding to the pixel on the RGB image.
    """

    @dataclass
    class OraclePixelPickActionArgs:
        """
        :property position: The (x, y) pixel position on the RGB image
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        position: List  # [x, y]
        grab_release: int

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        """
        Parses the pixel position we should be picking at.
        :param skill_arg: a dictionary specifying the 'position' as [x, y]
        """
        if isinstance(skill_arg, dict) and "position" in skill_arg:
            position = skill_arg["position"]
        else:
            raise ValueError(f"Skill argument must include 'position', got {skill_arg}")

        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"'position' must be a list of [x, y], got {position}")

        return OraclePixelPickPolicy.OraclePixelPickActionArgs(position=position, grab_release=PICK_ID)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        positions = torch.FloatTensor(
            [self._cur_skill_args[i].position for i in cur_batch_idx]
        )
        full_action[:, self._pick_srt_idx:self._pick_srt_idx + 2] = positions
        full_action[:, self._pick_end_idx-1] = torch.FloatTensor([PICK_ID] * masks.shape[0])
        full_action[:, self._grip_ac_idx] = torch.FloatTensor([PICK_ID] * masks.shape[0])
        
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
        
class OraclePixelPlacePolicy(OraclePlacePolicy):
    """
    Skill to generate a placing motion based on a pixel position.
    Moves the arm to the 3D position corresponding to the pixel on the RGB image.
    """

    @dataclass
    class OraclePixelPlaceActionArgs:
        """
        :property position: The (x, y) pixel position on the RGB image
        :property grab_release: Whether we want to grab (1) or drop an object (0)
        """
        position: list  # [x, y]
        grab_release: int

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        """
        Parses the pixel position we should be placing at.
        :param skill_arg: a dictionary specifying the 'position' as [x, y]
        """
        if isinstance(skill_arg, dict) and "position" in skill_arg:
            position = skill_arg["position"]
        else:
            raise ValueError(f"Skill argument must include 'position', got {skill_arg}")

        if not isinstance(position, list) or len(position) != 2:
            raise ValueError(f"'position' must be a list of [x, y], got {position}")

        return OraclePixelPlacePolicy.OraclePixelPlaceActionArgs(position=position, grab_release=RELEASE_ID)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        positions = torch.FloatTensor(
            [self._cur_skill_args[i].position for i in cur_batch_idx]
        )
        
        full_action[:, self._place_srt_idx:self._place_srt_idx + 2] = positions
        full_action[:, self._place_end_idx-1] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
        full_action[:, self._grip_ac_idx] = torch.FloatTensor([PLACE_ID] * masks.shape[0])
        # if self._is_skill_done(observations, rnn_hidden_states, prev_actions, masks, cur_batch_idx):
        #     full_action[0][self._place_end_idx-1] = -1.0
        
        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
        
        
class OraclePixelNavPolicy(OracleNavPolicy):
    """
    Skill to pass navigation pixel target to PixelNavAction. 
    Move to the 3D position corresponding to the pixel on the RGB image.
    """
    @dataclass
    class OraclePixelNavActionArgs:
        """
        :property target_position: (2, ) The target position in pixel coordinates
        """
        target_position: List[float]
        

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config,
    ):
        NnSkillPolicy.__init__(
            self,
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )

        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, "pixel_nav_action"
        )

    def _parse_skill_arg(self, skill_name: str, skill_arg):
        # if skill arg is a dictionary
        if isinstance(skill_arg, dict):
            target_position = skill_arg["target_position"]
            
        else:
            raise ValueError(
                f"Unexpected number of skill arguments in {skill_arg}"
            )

        return OraclePixelNavPolicy.OraclePixelNavActionArgs(target_position)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        deterministic=False,
    ):
        full_action = torch.zeros(
            (masks.shape[0], self._full_ac_size), device=masks.device
        )
        target_positions = torch.FloatTensor(
            [self._cur_skill_args[i].target_position for i in cur_batch_idx]
        )

        full_action[:, self._oracle_nav_ac_idx: self._oracle_nav_ac_idx + 2] = target_positions

        return PolicyActionData(
            actions=full_action, rnn_hidden_states=rnn_hidden_states
        )
