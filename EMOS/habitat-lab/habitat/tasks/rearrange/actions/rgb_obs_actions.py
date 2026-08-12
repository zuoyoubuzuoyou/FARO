'''
This file is used to define the rearrange actions with pixel on rgb observations as input.
'''
from habitat.tasks.rearrange.actions.actions import (
    ArticulatedAgentAction, 
    ArmEEAction
)
from habitat.tasks.rearrange.social_nav.oracle_social_nav_actions import OracleNavCoordAction
from habitat.tasks.rearrange.rearrange_sim import RearrangeSim
from habitat.core.registry import registry
from gym import spaces
import numpy as np
import cv2


def unproject_pixel_to_point(sim: RearrangeSim, sensor_name: str, depth_map: np.ndarray, pixel: tuple) -> np.ndarray:
    """
    Unprojects a pixel from the depth map to a 3D point in space.

    :param sim: RearrangeSim instance
    :param sensor_name: Name of the sensor
    :param depth_map: Depth map from the sensor
    :param pixel: (x, y) pixel coordinates
    :return: 3D point in space
    """
    depth_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    hfov = float(sim._sensors[sensor_name]._sensor_object.hfov) * np.pi / 180.
    W, H = depth_camera.viewport[0], depth_camera.viewport[1]

    K = np.array([
        [1 / np.tan(hfov / 2.), 0., 0., 0.],
        [0., 1 / np.tan(hfov / 2.), 0., 0.],
        [0., 0., 1, 0],
        [0., 0., 0, 1]
    ])

    x, y = pixel
    depth = depth_map[y, x]

    xs = (x / W) * 2 - 1
    ys = (y / H) * 2 - 1

    xys = np.array([xs * depth, ys * depth, -depth, 1])
    xy_c = np.matmul(np.linalg.inv(K), xys)

    depth_rotation = np.array(depth_camera.camera_matrix.rotation())
    depth_translation = np.array(depth_camera.camera_matrix.translation)

    T_world_camera = np.eye(4)
    T_world_camera[0:3, 0:3] = depth_rotation
    T_world_camera[0:3, 3] = depth_translation

    T_camera_world = np.linalg.inv(T_world_camera)
    point_world = np.matmul(T_camera_world, xy_c)

    return point_world[:3] / point_world[3]

@registry.register_task_action
class PixelArmAction(ArmEEAction):
    """
    Pick/Place action for the articulated_agent given the (x, y) pixel on the RGB image.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """

    def __init__(self, *args, sim: RearrangeSim, task, **kwargs):
        ArmEEAction.__init__(self, *args, sim=sim, **kwargs)
        self._task = task
        self._prev_ep_id = None
        self._hand_pose_iter = 0
        self.is_reset = False
        
        # camera parameters
        self.camera_type = kwargs.get("camera_type", "head")
        self.depth_camera_name = self._action_arg_prefix + f"{self.camera_type}_depth"

    def reset(self, *args, **kwargs):
        super().reset()

    @property
    def action_space(self):
        return spaces.Box(
            shape=(3,),
            low=0,
            high=10000,
            dtype=np.float32,
        )

    def step(self, arm_action, **kwargs):
        pixel_coord = arm_action[:2]
        action_type = arm_action[2]
        
        # if no action is specified, return the current end-effector position
        if action_type == 0:
            return self.ee_target

        depth_obs = self._sim._prev_sim_obs[self.depth_camera_name].squeeze()

        object_coord = unproject_pixel_to_point(self._sim, self.depth_camera_name, depth_obs, pixel_coord)
        cur_ee_pos = self.cur_articulated_agent.ee_transform().translation
        translation = object_coord - cur_ee_pos

        translation_base = self.cur_articulated_agent.base_transformation.inverted().transform_vector(translation)

        translation_base = np.clip(translation_base, -1, 1)
        self.set_desired_ee_pos(translation_base)

        if self._render_ee_target:
            global_pos = self.cur_articulated_agent.base_transformation.transform_point(
                self.ee_target
            )
            self._sim.viz_ids["ee_target"] = self._sim.visualize_position(
                global_pos, self._sim.viz_ids["ee_target"]
            )

        return self.ee_target
    
@registry.register_task_action
class PixelNavAction(OracleNavCoordAction):
    """
    Navigate action for the articulated_agent given the (x, y) pixel on the RGB image.
    Uses inverse kinematics (requires pybullet) to apply end-effector position control for the articulated_agent's arm.
    """
    
    def __init__(self, *args, task, **kwargs):
        super().__init__(*args, task=task, **kwargs)
        
        # camera parameters
        self.camera_type = kwargs.get("camera_type", "head")
        self.depth_camera_name = self._action_arg_prefix + f"{self.camera_type}_depth"
    
    @property
    def action_space(self):
        return spaces.Dict(
            {
                self._action_arg_prefix
                + "pixel_nav_action": spaces.Box(
                    shape=(2,),
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    dtype=np.float32,
                )
            }
        )
    
    def step(self, pixel_nav_action, **kwargs):
        
        # if all pixels are zero, return
        if np.all(pixel_nav_action == 0):
            return
        
        depth_obs = self._sim._prev_sim_obs[self.depth_camera_name].squeeze()
        target_coord = unproject_pixel_to_point(self._sim, self.depth_camera_name, depth_obs, pixel_nav_action)

        kwargs[self._action_arg_prefix + "oracle_nav_coord_action"] = target_coord
        return super().step(**kwargs)
