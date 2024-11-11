# Main script for hand interractions
import os
import time
import cv2
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
from gymnasium import spaces
from openteach.robot.allegro.allegro import AllegroHand
from openteach.robot.kinova import KinovaArm
from openteach.utils.network import ZMQCameraSubscriber
from openteach_api.api import DeployAPI

from PIL import Image as im
from scipy.spatial.transform import Rotation

from object_rewards.kinematics.fingertip_ik_full_robot_solver import (
    FingertipIKFullRobotSolver,
)
from object_rewards.utils.constants import EEF_TO_END
from object_rewards.utils.files import get_root_path
from object_rewards.utils.augmentations import crop_transform


class DexterityEnv(gym.Env):

    def __init__(
        self,
        host_address="172.24.71.240",
        camera_port=10005,
        camera_num=0,
        height=480,
        width=480,
        robot_initial_pose=None,  # NOTE: the defualt value should change dependent on robots
        max_episode_steps=200,
        **kwargs,
    ):
        self.width = width
        self.height = height
        self.view_num = camera_num

        self.deploy_api = DeployAPI(
            host_address=host_address,
            required_data={"rgb_idxs": [camera_num], "depth_idxs": []},
        )
        # NOTE: Here we pass the initialized end effector position to self.home_state
        kinova_initial_pose = robot_initial_pose
        self.set_home_state(
            kinova_initial_pose
        )  # This should be implemented by all the rest of the tasks
        self.arm = KinovaArm()
        self.hand = AllegroHand()
        self.ik_solver = FingertipIKFullRobotSolver(
            urdf_path=f"{get_root_path()}/models/hand_and_arm_real_world.urdf",  # NOTE: I have updated the hand_and_arm_realworld urdf to the one with 2
            desired_finger_types=["index", "middle", "ring", "thumb"],
            compute_type="position",
        )

        # Get the action and observation spaces (related to gym)
        action_dim = 12
        self.action_space = spaces.Box(
            low=np.array([-1] * action_dim, dtype=np.float32),  # Actions are 12 + 7
            high=np.array([1] * action_dim, dtype=np.float32),
            dtype=np.float32,
        )
        obs_space_dict = self._build_observation_space_dict()
        self.observation_space = spaces.Dict(obs_space_dict)

        self.image_subscriber = ZMQCameraSubscriber(
            host=host_address, port=camera_port + self.view_num, topic_type="RGB"
        )
        self.host_address = host_address
        self.image_transform = T.Compose(
            [
                T.Resize((480, 640)),
                T.Lambda(
                    lambda image: crop_transform(image, camera_view=self.camera_id)
                ),
            ]
        )
        self.step_count = 0
        self.max_episode_steps = max_episode_steps

    def _build_observation_space_dict(self):
        obs_space_dict = dict()
        features_dim = 12
        obs_space_dict["pixels"] = spaces.Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([255, 255], dtype=np.float32),
            dtype=np.float32,
        )
        obs_space_dict["features"] = spaces.Box(
            low=np.array([-1] * features_dim, dtype=np.float32),
            high=np.array([1] * features_dim, dtype=np.float32),
            dtype=np.float32,
        )
        return obs_space_dict

    def set_home_state(self):
        raise NotImplementedError  # This method should be implemented by every class that inherits this

    def _get_curr_image(self):
        image, _ = self.image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image, "RGB")
        img = self.image_transform(image)
        img = np.asarray(img)

        return img  # NOTE: This is for environment

    def _get_curr_joint_positions(self):

        arm_joints = self.arm.get_joint_position()[:6]
        arm_joints = [float(element) for element in arm_joints]
        hand_joints = self.hand.get_joint_position()
        hand_joints = [float(element) for element in hand_joints]
        current_joint_positions = np.concatenate([arm_joints, hand_joints], axis=0)

        return current_joint_positions

    def _turn_arm_joint_action_to_cartesian(self, arm_hand_action):

        rotation_end_to_base = self.ik_solver.get_endeff_pose(arm_hand_action)
        rotation_eef_to_base = np.dot(rotation_end_to_base, EEF_TO_END)

        eef_quat = Rotation.from_matrix(rotation_eef_to_base[:3, :3]).as_quat()
        eef_tvec = rotation_eef_to_base[:3, 3].reshape(1, 3)[0]
        arm_cartesian_action = np.concatenate([eef_tvec, eef_quat], axis=0)

        return arm_cartesian_action

    # If the action type is tip states - it will turn it to joint states using the ik solver
    def step(self, action):

        current_joint_positions = self._get_curr_joint_positions()

        solver_action, _ = self.ik_solver.inverse_kinematics(
            desired_poses=action[:4],
            current_joint_positions=current_joint_positions,
            desired_orientation_poses=None,
        )

        robot_action_dict = dict(
            allegro=solver_action[6:],
            kinova=self._turn_arm_joint_action_to_cartesian(
                arm_hand_action=solver_action
            ),  # We send the cartesian for the arm
        )

        self.deploy_api.send_robot_action(robot_action_dict)
        # Get the observations
        obs = self._get_obs()
        reward = self.get_reward()  # This will be rewritten by children classes

        infos = {"is_success": False}
        self.step_count += 1
        if self.max_episode_steps is not None:
            done = truncated = self.step_count >= self.max_episode_steps
        else:
            done = truncated = False

        return obs, reward, done, truncated, infos

    def get_reward(self):
        return 0.0

    def render(self, mode="rbg_array", width=0, height=0):
        return self._get_curr_image()

    def reset(self):
        self._reset_state()
        obs = self._get_obs()
        return obs

    def _reset_state(self):
        self.deploy_api.send_robot_action(self.home_state)
        self.step_count = 0
        time.sleep(3)
        input("Press Enter to continue... after resetting env")

    def _get_obs(self):
        obs = {}

        obs["pixels"] = self._get_curr_image()

        current_joint_positions = self._get_curr_joint_positions()
        fingertip_poses = self.ik_solver.forward_kinematics(
            current_joint_positions=current_joint_positions
        )

        obs["features"] = np.concatenate(
            [fingertip_pose[:3, 3] for fingertip_pose in fingertip_poses], axis=-1
        )

        return obs
