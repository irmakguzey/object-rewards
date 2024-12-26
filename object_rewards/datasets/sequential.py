# Dataset to be used for training VQ-Bet training
import glob
import os
import pickle

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader as loader
from tqdm import tqdm

from object_rewards.calibration import CalibrateBase, CalibrateFingertips
from object_rewards.point_tracking.co_tracker import CoTrackerLangSam
from object_rewards.utils.augmentations import crop_transform
from object_rewards.utils.data import get_all_demo_frame_ids, get_demo_action_ids
from object_rewards.utils.trajectory import get_rot_and_trans
from object_rewards.utils.transforms import flatten_homo_action


class SeqH2RDataset(
    data.Dataset
):  # NOTE: Change this name to SeqVQVAE or smth like that

    def __init__(
        self,
        data_path,
        host,
        camera_id,
        camera_port,
        object_detection_camera_id,
        marker_size,
        calibration_pics_dir,
        traj_len,
        action_chunking_len,
        delta_actions,
        normalize_features,
        ts_step,
        object_detection_camera_type,  # realsense or fisheye
        object_text_prompt,
        cotracker_grid_size,
        cotracker_device,
        demo_residuals=None,
    ):

        self.is_fish_eye = object_detection_camera_type == "fisheye"
        self.object_text_prompt = object_text_prompt

        # Initialize the cotracker
        # Initialize models
        self.predictor = CoTrackerLangSam(
            device=cotracker_device,
            is_online=True,
            frame_by_frame=False,
            grid_size=cotracker_grid_size,
        )

        self.image_transform = T.Compose(
            [
                T.Resize((480, 640)),
                T.Lambda(
                    lambda image: crop_transform(image, camera_view=self.camera_id)
                ),
            ]
        )

        # data_representations: [image, features], features is fingertip positions wrt the base
        self.traj_len = traj_len
        self.action_chunking_len = action_chunking_len
        self.data_path = data_path
        self.roots = sorted(glob.glob(f"{data_path}/demonstration_*"))
        self.camera_id = camera_id
        self.object_camera_id = object_detection_camera_id
        self.host = host

        # Get the base calibrations
        self.calibrate_base = CalibrateBase(
            host=host,
            calibration_pics_dir=calibration_pics_dir,
            cam_idx=camera_id,
            marker_size=marker_size,
        )
        self.H_B_C = self.calibrate_base.load_base_to_camera()
        self.H_A_C = None  # We hold this so that we won't need to calibrate the camera to the aruco for every demo

        self.camera_port = camera_port
        self.marker_size = marker_size
        self.demo_frame_ids = get_all_demo_frame_ids()
        self.action_dim = 12
        self.delta_actions = delta_actions
        self.normalize_features = normalize_features

        self.demo_residuals = demo_residuals
        self.ts_step = ts_step
        self.get_demo_info()

    def __len__(self):
        return self._observations.shape[0]  # If it's the final frame we don't

    def _load_image(self, demo_path, frame_id):

        dir_name = (
            f"cam_{self.object_camera_id}_fish_eye_images"
            if self.is_fish_eye
            else f"cam_{self.object_camera_id}_rgb_images"
        )

        image_path = os.path.join(
            demo_path,
            "{}/frame_{}.png".format(dir_name, str(frame_id).zfill(5)),
        )
        img = self.image_transform(loader(image_path))  # This loads images as PIL image
        return img  # width: 640, height: 480

    def _get_pred_tracks_for_demo(self, demo_path, demo_action_ids):
        # Use view_num and the collected frames to get the predicted tracks
        demo_num = demo_path.split("/")[-1].split("_")[-1]

        pkl_file_name = (
            f"image_indices_fish_eye_cam_{self.object_camera_id}.pkl"
            if self.is_fish_eye
            else f"image_indices_cam_{self.object_camera_id}.pkl"
        )
        object_image_indices_path = os.path.join(demo_path, pkl_file_name)

        with open(object_image_indices_path, "rb") as file:
            demo_object_image_indices = pickle.load(file)

        demo_imgs = []
        pbar = tqdm(total=demo_action_ids[1] - demo_action_ids[0])
        for action_id in range(demo_action_ids[0], demo_action_ids[1]):
            img_frame_id = demo_object_image_indices[action_id][1]
            img = self._load_image(demo_path=demo_path, frame_id=img_frame_id)
            demo_imgs.append(img)
            pbar.update(1)
            pbar.set_description(f"Getting Demo {demo_num} Tracks")

        demo_imgs = np.stack(demo_imgs, axis=0)

        demo_tracks = self.predictor.get_segmented_tracks_by_batch(
            frames=demo_imgs, text_prompt=self.object_text_prompt
        )

        pbar.close()

        print(f"Demo: {demo_num} - demo_tracks.shape: {demo_tracks.shape}")
        return demo_tracks

    def get_initialization_wrist_and_finger_roots(self):
        wrist_to_base = np.mean(self.init_demo_wrist_to_base, axis=0)
        finger_roots_to_base = np.mean(self.init_demo_finger_roots_to_bases, axis=0)

        return wrist_to_base, finger_roots_to_base

    def get_demo_info(self):
        # Method to traverse through the roots and get the action ids that starts the demos
        # It will return image indices as [demo_id, image_frame_id] and all the fingertip positions

        print("** GETTING DATASET INFO **")
        fingertips = []
        index_to_demo_indexes = []
        demo_start_index = 0

        object_position_observations = []
        init_demo_wrist_to_base = []
        init_demo_finger_roots_to_bases = []

        for demo_path in self.roots:  # Iterate through all the demos
            demo_num = demo_path.split("/")[-1].split("_")[-1]

            # Get the demo action ids
            demo_action_ids = get_demo_action_ids(
                data_path=self.data_path, view_num=self.camera_id, demo_num=demo_num
            )

            demo_calibrate_fingertips = CalibrateFingertips(
                demo_path=demo_path,
                realsense_view_num=self.camera_id,
                marker_size=self.marker_size,
                host=self.host,
                camera_port=self.camera_port,
                H_A_C=self.H_A_C,
            )

            if self.H_A_C is None:
                self.H_A_C = (
                    demo_calibrate_fingertips.H_A_C
                )  # This is done so that it won't need to get this everytime

            # Get the demo tracks, rotation and translation and mean position of the object
            demo_tracks = self._get_pred_tracks_for_demo(
                demo_path=demo_path, demo_action_ids=demo_action_ids
            )
            # Get rot and trans
            rot_and_trans = torch.FloatTensor(
                get_rot_and_trans(tracks=demo_tracks, delta=False)
            )
            mean_pos = torch.FloatTensor(np.mean(demo_tracks, axis=1))

            demo_len = demo_action_ids[1] - demo_action_ids[0]
            for i, frame_id in enumerate(range(demo_action_ids[0], demo_action_ids[1])):

                if i == 0:
                    wrist_to_base = demo_calibrate_fingertips.get_wrist_to_base(
                        frame_id=frame_id, base_to_camera=self.H_B_C
                    )
                    finger_roots_to_base = (
                        demo_calibrate_fingertips.get_finger_roots_to_base(
                            frame_id=frame_id, base_to_camera=self.H_B_C
                        )
                    )

                    init_demo_wrist_to_base.append(wrist_to_base)
                    init_demo_finger_roots_to_bases.append(finger_roots_to_base)

                # Get the fingertip positions
                fingertips_to_base = demo_calibrate_fingertips.get_fingertips_to_base(
                    frame_id=frame_id, base_to_camera=self.H_B_C
                )  # (8,4,4)
                if self.demo_residuals is not None:
                    for j in range(len(self.demo_residuals)):
                        fingertips_to_base[:, j, 3] += self.demo_residuals[j]

                flattened_fingertips = flatten_homo_action(action=fingertips_to_base)
                fingertips.append(flattened_fingertips)
                index_to_demo_indexes.append(
                    (demo_start_index, demo_start_index + demo_len)
                )

                object_position_obs = torch.concat(
                    [rot_and_trans[i], mean_pos[i]], dim=0
                )
                object_position_observations.append(object_position_obs)

            demo_start_index += demo_len

        self.init_demo_wrist_to_base = np.stack(init_demo_wrist_to_base, axis=0)
        self.init_demo_finger_roots_to_bases = np.stack(
            init_demo_finger_roots_to_bases, axis=0
        )

        # Normalize and get the features
        fingertips = torch.stack(fingertips, dim=0)
        object_position_observations = torch.stack(object_position_observations, dim=0)
        observations = torch.concat([fingertips, object_position_observations], dim=-1)

        if self.normalize_features:
            self.observations_std, self.observations_mean = torch.std_mean(
                observations, dim=0
            )
            self.actions_std, self.actions_mean = torch.std_mean(fingertips, dim=0)
            observations = (
                observations - self.observations_mean
            ) / self.observations_std
            fingertips = (fingertips - self.actions_mean) / self.actions_std
        self._observations = observations

        self._fingertips = fingertips
        self._index_to_demo_indexes = index_to_demo_indexes

    def __getitem__(self, idx):
        demo_start_index, demo_end_index = self._index_to_demo_indexes[idx]
        demo_observation, demo_fingertips = (
            self._observations[demo_start_index:demo_end_index],
            self._fingertips[demo_start_index:demo_end_index],
        )
        idx_in_demo = idx - demo_start_index

        obs, act = [], [None] * (self.traj_len + self.action_chunking_len - 1)
        for i_past in range(self.traj_len - 1, -1, -1):
            current_ts_idx = idx_in_demo - i_past * self.ts_step
            obs.append(demo_observation[max(0, current_ts_idx)])
            act_chunk = []
            for i_future in range(self.action_chunking_len):
                current_act_idx = current_ts_idx + (i_future + 1) * self.ts_step
                current_act = demo_fingertips[
                    max(0, min(len(demo_fingertips) - 1, current_act_idx))
                ]
                if self.delta_actions:
                    prev_act_idx = current_ts_idx + i_future * self.ts_step
                    prev_act = demo_fingertips[
                        max(0, min(len(demo_fingertips) - 1, prev_act_idx))
                    ]
                    act_chunk.append(current_act - prev_act)
                else:
                    act_chunk.append(current_act)
            act[
                self.traj_len
                - i_past
                - 1 : self.traj_len
                - i_past
                - 1
                + self.action_chunking_len
            ] = act_chunk
        obs, act = torch.stack(obs), torch.stack(act)
        return obs, act
