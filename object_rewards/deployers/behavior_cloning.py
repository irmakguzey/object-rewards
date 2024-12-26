import os
import time
from pathlib import Path

import cv2
import einops
import hydra
import numpy as np
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from openteach.constants import DEPTH_PORT_OFFSET
from openteach.robot.allegro.allegro import AllegroHand
from openteach.robot.kinova import KinovaArm
from openteach.utils.network import ZMQCameraSubscriber
from openteach_api.api import DeployAPI
from PIL import Image as im

from object_rewards.calibration import CalibrateBase
from object_rewards.kinematics import FingertipIKFullRobotSolver
from object_rewards.offline_policies.initialize_learner import init_learner
from object_rewards.point_tracking import CoTrackerLangSam
from object_rewards.utils import (
    VideoRecorder,
    apply_residuals,
    crop_transform,
    flatten_homo_position,
    get_initial_kinova_position,
    get_point_cloud,
    get_root_path,
    homo_flattened_position,
    wrist_rotation_to_kinova_position,
)


class BCH2R:

    def __init__(
        self,
        host,
        camera_port,
        camera_id,
        object_camera_port,
        object_camera_id,
        marker_size,
        calibration_pics_dir,
        training_dir,
        save_deployment,
        device,
        cotracker_grid_size,
        object_text_prompt,
        average_actions,
        deterministic,
        deployment_num=0,
        demo_residuals=None,
        load_dataset_info=True,
        deployment_save_dir="/data/irmak/third_person_manipulation/deployments",
        initial_kinova_position=None,
        **kwargs,
    ):

        # Load the config and initialize the learner
        for dirs in os.walk(training_dir):
            if ".hydra" in dirs[0]:
                vq_vae_cfg_dir = dirs[0]

        cfg = OmegaConf.load(os.path.join(vq_vae_cfg_dir, "config.yaml"))
        self.cfg = cfg
        model_path = Path(training_dir) / "models"
        self.device = device

        self.learner = init_learner(cfg=cfg, device=self.device, load=True)
        self.learner.load(model_path, model_type="best")
        self.learner.to(self.device)
        self.deterministic = deterministic
        self.traj_len = cfg.traj_len  # Will stack the observations
        self.action_chunking_len = cfg.action_chunking_len
        self.obs_dim = cfg.obs_dim
        self.act_dim = cfg.action_dim
        self.feature_type = cfg.feature_type
        self.normalize_features = cfg.normalize_features
        self.delta_actions = cfg.delta_actions
        self.average_actions = average_actions
        self.demo_residuals = demo_residuals
        self.load_dataset_info = load_dataset_info
        self.initial_kinova_position = initial_kinova_position

        # Get the features mean and std
        dataset = hydra.utils.instantiate(cfg.dataset, cotracker_device=device)
        self.dataset = dataset
        if cfg.normalize_features:
            self.observations_std, self.observations_mean = (
                dataset.observations_std,
                dataset.observations_mean,
            )
            self.actions_std, self.actions_mean = (
                dataset.actions_std,
                dataset.actions_mean,
            )

        # Object detection parts
        self.object_image_subscriber = ZMQCameraSubscriber(
            host=host, port=object_camera_port + object_camera_id, topic_type="RGB"
        )
        self.cotracker_langsam = CoTrackerLangSam(
            device=self.device,
            is_online=True,
            frame_by_frame=True,
            grid_size=cotracker_grid_size,
        )
        self.object_text_prompt = object_text_prompt
        self.image_transform = T.Compose(
            [
                T.Resize((480, 640)),
                T.Lambda(lambda image: crop_transform(image, camera_view=camera_id)),
            ]
        )
        self.object_camera_id = object_camera_id

        # Calibrate the camera to the base
        calibrate_base = CalibrateBase(
            host=host,
            calibration_pics_dir=calibration_pics_dir,
            cam_idx=camera_id,
            marker_size=marker_size,
        )
        self.H_B_C = calibrate_base.load_base_to_camera()

        # Initialize the deploy api
        self.deploy_api = DeployAPI(
            host_address=host, required_data={"rgb_idxs": [0], "depth_idxs": [0]}
        )
        self.host = host

        # Start the IK solver and the robot drivers to be able to get the fingertips
        self.ik_solver = FingertipIKFullRobotSolver(
            urdf_path=f"{get_root_path()}/models/hand_and_arm_real_world.urdf",  # NOTE: I have updated the hand_and_arm_realworld urdf to the one with 2
            desired_finger_types=["index", "middle", "ring", "thumb"],
        )

        # Robot initialization
        self.arm = KinovaArm()
        self.hand = AllegroHand()

        if save_deployment:
            task_name = f"{cfg.learner.type}_{cfg.task}"
            deployment_save_dir = os.path.join(deployment_save_dir, f"{task_name}")
            if not os.path.exists(deployment_save_dir):
                os.makedirs(deployment_save_dir)

            self.video_recorder = VideoRecorder(
                save_dir=Path(deployment_save_dir) / str(deployment_num),
                resize_and_transpose=False,
            )
            self.image_subscriber = ZMQCameraSubscriber(
                host=host, port=camera_port + camera_id, topic_type="RGB"
            )

    def _init_cotracker(self):
        print("** INITIALIZING COTRACKER **")

        # Capturing the image
        image = self._get_curr_image()

        self.window_frames = [
            torch.from_numpy(image).float().to(self.device)
            for _ in range(self.cotracker_langsam.cotracker.step * 2)
        ]

        # Get the initial queries from text prompt
        queries = self.cotracker_langsam.get_queries(
            frame=image, text_prompt=self.object_text_prompt, segm_mask=None
        )
        self.cotracker_langsam.cotracker(
            video_chunk=self._get_video_chunk(), is_first_step=True, queries=queries
        )

    def _get_video_chunk(self):
        # Create the video_chunk
        return (
            torch.tensor(
                torch.stack(
                    self.window_frames[-self.cotracker_langsam.cotracker.step * 2 :]
                ),
                device=self.window_frames[0].device,
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

    def _get_rot_and_trans(
        self, tracks
    ):  # Should add this to the state as well, since this is the reward

        if self.is_first_step:
            self.total_translation, self.total_rotation = (
                np.zeros(2, dtype=np.float32),
                0.0,
            )
            self.prev_points = tracks[-1, :, :].detach().cpu().numpy()

        else:
            curr_points = tracks[-1, :, :].detach().cpu().numpy()

            # Translation
            # Calculate the translation difference
            curr_mean = np.mean(curr_points, axis=0)
            prev_mean = np.mean(self.prev_points, axis=0)
            diff_mean = curr_mean - prev_mean

            if (diff_mean == 0.0).all():
                diff_mean = np.ones(2) * 1.0e-12

            # Add it to the total translation
            self.total_translation[0] += diff_mean[0]
            self.total_translation[1] += diff_mean[1]

            # Rotation
            # Bring all the points to the same space
            curr_feat_norm = curr_points - curr_mean
            prev_feat_norm = self.prev_points - prev_mean

            # Calculate the rotation
            n = np.cross(prev_feat_norm, curr_feat_norm)
            if (n == 0.0).all():
                average_rot = 1e-12
            else:
                rot = n / np.linalg.norm(n)
                average_rot = np.mean(rot)

            self.total_rotation += (
                average_rot  # NOTE: If the KL divergence doesn't work well
            )

            self.prev_points = curr_points.copy()

        rot_and_trans = torch.FloatTensor(
            [
                self.total_translation[0],
                self.total_translation[1],
                self.total_rotation,
            ]
        )
        self.is_first_step = False
        return rot_and_trans

    def _get_obj_position(self, tracks):

        obj_position = torch.mean(tracks[-1, :, :], dim=0).detach().cpu()
        obj_position[0]
        obj_position[1]
        return obj_position

    def _get_curr_image(self):
        image, _ = self.object_image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = im.fromarray(image, "RGB")
        image = np.asarray(self.image_transform(image))
        return image

    def _get_tracks(self):
        image = torch.from_numpy(self._get_curr_image()).float().to(self.device)
        self.window_frames.append(image)

        pred_tracks, _ = self.cotracker_langsam.cotracker(
            video_chunk=self._get_video_chunk(), is_first_step=False, queries=None
        )

        return pred_tracks[0, :, :, :]

    def _get_obj_and_rot_trans(self):
        tracks = self._get_tracks()
        rot_and_trans = self._get_rot_and_trans(tracks=tracks)
        mean_position = self._get_obj_position(tracks=tracks)

        final_obj_pose = torch.concat([rot_and_trans, mean_position], dim=0)

        return final_obj_pose

    def _get_current_joint_positions(self):
        arm_joints = self.arm.get_joint_position()[:6]
        arm_joints = [float(element) for element in arm_joints]
        hand_joints = self.hand.get_joint_position()
        hand_joints = [float(element) for element in hand_joints]
        current_joint_positions = np.concatenate([arm_joints, hand_joints], axis=0)
        return current_joint_positions

    def _get_frames_in_camera(self, poses_in_base):

        poses_in_camera = []
        for pose_in_base in poses_in_base:
            pose_in_camera = self.H_B_C @ pose_in_base
            poses_in_camera.append(pose_in_camera)
        poses_in_camera = np.stack(poses_in_camera, axis=0)

        return poses_in_camera

    def _get_fingertip_position(self):
        current_joint_positions = self._get_current_joint_positions()
        robot_fingertips_in_base = self.ik_solver.forward_kinematics(
            current_joint_positions=current_joint_positions
        )

        robot_fingertips_in_base = apply_residuals(
            self.demo_residuals, robot_fingertips_in_base, inverse=True
        )

        flattened_fingertips = flatten_homo_position(position=robot_fingertips_in_base)

        return torch.FloatTensor(flattened_fingertips)

    def get_current_observation(self):

        obs = torch.concat(
            [
                self._get_fingertip_position(),
                self._get_obj_and_rot_trans(),
            ],
            dim=-1,
        )

        if self.normalize_features:
            obs = (obs - self.observations_mean) / self.observations_std

        return obs

    def initialize_robot_position(self, wrist_extend_length):

        wrist_to_base, finger_roots_to_base = (
            self.dataset.get_initialization_wrist_and_finger_roots()
        )

        if self.initial_kinova_position is None:
            position = get_initial_kinova_position(
                wrist_to_base,
                finger_roots_to_base,
                wrist_extend_length,
            )
        else:
            position = self.initial_kinova_position

        return {"kinova": position}

    def get_actions(self, stacked_obs=None):
        current_obs = self.get_current_observation()
        if stacked_obs is None:
            stacked_obs = torch.stack([current_obs] * self.traj_len)
        else:
            stacked_obs = torch.roll(stacked_obs, shifts=-1, dims=0)
            stacked_obs[-1] = current_obs
        obs = stacked_obs.unsqueeze(0).to(self.device)

        actions = (
            self.learner.predict(obs=obs, deterministic=self.deterministic)
            .detach()
            .cpu()
        )

        return actions, stacked_obs

    def deploy(self):

        kinova_command = self.initialize_robot_position(wrist_extend_length=0.05)
        _ = input("Make sure if the kinova command is good or not!")
        self.deploy_api.send_robot_action(kinova_command)
        print(
            "************************wrist successfully retargeted!***************************"
        )
        time.sleep(5)

        self._init_cotracker()
        self.is_first_step = True

        try:

            step_id = 0
            stacked_obs = None
            stacked_actions = torch.zeros(
                (
                    self.action_chunking_len,
                    self.action_chunking_len,
                    self.act_dim,
                )
            )
            while True:

                actions, stacked_obs = self.get_actions(stacked_obs=stacked_obs)

                if self.average_actions:
                    stacked_actions.roll(shifts=-1, dims=0)
                    stacked_actions[-1] = actions
                    step_actions = einops.rearrange(
                        stacked_actions.flip(dims=[1]).diagonal(dim1=0, dim2=1),
                        "A W -> W A",
                    )

                    actions_populated = torch.any(step_actions != 0, dim=[1])
                    step_actions = step_actions[actions_populated]
                    k = 0.01
                    exp_weights = torch.exp(-k * torch.arange(len(step_actions)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = exp_weights.unsqueeze(1)
                    pred_act = (
                        (step_actions * exp_weights).sum(dim=0, keepdim=True).squeeze()
                    )
                else:
                    pred_act = actions[0]

                # Denormalize fingertips
                if self.normalize_features:
                    pred_act = pred_act * self.actions_std + self.actions_mean

                if self.delta_actions:
                    curr_fingertip_positions = self._get_fingertip_position()
                    pred_act = curr_fingertip_positions + pred_act
                pred_act = pred_act.numpy()

                # Send the predicted action to the robot
                fingertips_to_base = homo_flattened_position(position=pred_act)
                fingertips_to_base = apply_residuals(
                    self.demo_residuals, fingertips_to_base
                )
                current_joint_positions = self._get_current_joint_positions()

                robot_command = {}
                action, _ = self.ik_solver.inverse_kinematics(
                    desired_poses=fingertips_to_base[:4],
                    current_joint_positions=current_joint_positions,
                    desired_orientation_poses=None,
                )
                robot_command["allegro"] = action[6:]

                # Get the cartesian command to apply in kinova
                rotation_end_to_base = self.ik_solver.get_endeff_pose(
                    action
                )  # This is the wrist position
                # We get the end effector with respect to the wrist:
                kinova_position = wrist_rotation_to_kinova_position(
                    rotation_end_to_base,
                )
                robot_command["kinova"] = kinova_position

                self.deploy_api.send_robot_action(robot_command)

                # Save the image
                image, _ = self.image_subscriber.recv_rgb_image()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = np.transpose(image, (2, 0, 1))
                if step_id == 0:
                    self.video_recorder.init(image)
                else:
                    self.video_recorder.record(image)

                step_id += 1

        except KeyboardInterrupt:

            self.video_recorder.save(f"deployment_{self.cfg.learner.type}.mp4")

    def run(self):
        self.deploy()


class BCH2RwPCD(BCH2R):  # Point cloud BC baseline
    def __init__(self, depth_camera_port, depth_camera_id, **kwargs):
        super().__init__(**kwargs)
        self.depth_image_subscriber = ZMQCameraSubscriber(
            host=self.host,
            port=depth_camera_port + depth_camera_id + DEPTH_PORT_OFFSET,
            topic_type="Depth",
        )

    def _get_pcd(self):
        depth_img, _ = self.depth_image_subscriber.recv_depth_image()

        pcd = get_point_cloud(
            color_img=None,
            depth_img=depth_img,
            filter_outliners=False,
            x_bounds=(-0.15, 0.15),
            y_bounds=(-0.1, 0.2),
        )

        # Sample the pcd
        pc_idx = np.random.choice(range(np.asarray(pcd.points).shape[0]), 3000)
        sampled_pcd = torch.FloatTensor(pcd.points)[pc_idx, :]
        sampled_pcd = torch.permute(sampled_pcd, (1, 0))

        return sampled_pcd

    def get_current_observation(self):

        obs = torch.concat(
            [
                self._get_fingertip_position(),
                self._get_obj_and_rot_trans(),
            ],
            dim=-1,
        )
        if self.normalize_features:
            obs = (obs - self.observations_mean) / self.observations_std

        pcd = self._get_pcd()

        return pcd, obs

    def get_actions(self, stacked_obs=None):  # For this class stacked_obs is a tuple
        current_pcd, current_obs = self.get_current_observation()
        if stacked_obs is None:
            stacked_obs = [
                torch.stack([current_pcd] * self.traj_len),
                torch.stack([current_obs] * self.traj_len),
            ]
        else:
            stacked_obs[0] = torch.roll(stacked_obs[0], shifts=-1, dims=0)
            stacked_obs[1] = torch.roll(stacked_obs[1], shifts=-1, dims=0)

            stacked_obs[0][-1] = current_pcd
            stacked_obs[1][-1] = current_obs

        pcd = stacked_obs[0].unsqueeze(0).to(self.device)
        ft_obj = stacked_obs[1].unsqueeze(0).to(self.device)

        actions = (
            self.learner.predict(
                pcd=pcd, ft_obj=ft_obj, deterministic=self.deterministic
            )
            .detach()
            .cpu()
        )

        return actions, stacked_obs
