import os
import pickle

import h5py
import numpy as np
import torch
from tqdm import tqdm

from object_rewards.calibration import CalibrateFingertips
from object_rewards.datasets.sequential import SeqH2RDataset
from object_rewards.utils.point_cloud import get_point_cloud
from object_rewards.utils.trajectory import get_rot_and_trans


class SeqH2RwCoTrackerandDepthDataset(SeqH2RDataset):
    def __init__(self, depth_camera_id, x_bounds, y_bounds, **kwargs):
        # This method will return point clouds as a part of the observations
        self.depth_camera_id = depth_camera_id
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        super().__init__(**kwargs)

    # Will return the point cloud for whole demo - we'll only use depth for now
    def _get_pcd_per_demo(self, demo_path, demo_action_ids):

        depth_file_path = os.path.join(
            demo_path, f"cam_{self.depth_camera_id}_depth.h5"
        )
        with h5py.File(depth_file_path, "r") as f:
            depth_images = f["depth_images"][()]

        # Use view_num and the collected frames to get the predicted tracks
        demo_num = demo_path.split("/")[-1].split("_")[-1]
        pkl_file_name = (
            f"image_indices_fish_eye_cam_{self.object_camera_id}.pkl"
            if self.is_fish_eye
            else f"image_indices_cam_{self.object_camera_id}.pkl"
        )
        # NOTE: For now we're not including colors to the point cloud
        rgb_image_indices_path = os.path.join(demo_path, pkl_file_name)
        with open(rgb_image_indices_path, "rb") as file:
            rgb_image_indices = pickle.load(file)

        demo_points = []
        pbar = tqdm(total=demo_action_ids[1] - demo_action_ids[0])
        for action_id in range(demo_action_ids[0], demo_action_ids[1]):
            img_frame_id = rgb_image_indices[action_id][1]
            depth_img = depth_images[img_frame_id]
            pcd = get_point_cloud(
                color_img=None,  # NOTE: For now we're not using color image
                depth_img=depth_img,
                filter_outliners=False,
                x_bounds=self.x_bounds,
                y_bounds=self.y_bounds,
            )

            pc_idx = np.random.choice(range(np.asarray(pcd.points).shape[0]), 3000)
            demo_points.append(np.asarray(pcd.points)[pc_idx, :])

            pbar.update(1)
            pbar.set_description(f"Getting Demo {demo_num} Point Cloud")

        demo_points = torch.FloatTensor(np.stack(demo_points, axis=0))
        return demo_points

    def get_demo_info(self):
        # Method to traverse through the roots and get the action ids that starts the demos
        # It will return image indices as [demo_id, image_frame_id] and all the fingertip positions

        print("** GETTING DATASET INFO **")
        fingertips = []
        index_to_demo_indexes = []
        demo_start_index = 0

        object_position_observations = []
        pcds = []

        init_demo_wrist_to_base = []
        init_demo_finger_roots_to_bases = []

        for demo_path in self.roots:  # Iterate through all the demos
            demo_num = demo_path.split("/")[-1].split("_")[-1]

            # Get the demo action ids
            demo_action_ids = self._get_demo_action_id(
                demo_path=demo_path, image_frame_ids=self.demo_frame_ids[demo_num]
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

            demo_pcd = self._get_pcd_per_demo(
                demo_path=demo_path, demo_action_ids=demo_action_ids
            )
            pcds.append(torch.permute(demo_pcd, (0, 2, 1)))

            # Get rot and trans
            rot_and_trans = torch.FloatTensor(
                get_rot_and_trans(tracks=demo_tracks, delta=False)
            )
            mean_pos = torch.FloatTensor(np.mean(demo_tracks, axis=1))

            demo_len = demo_action_ids[1] - demo_action_ids[0]
            # print(f"demo_len: {demo_len}")
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

                flattened_fingertips = self.flatten_homo_action(
                    action=fingertips_to_base
                )
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
        if self.feature_type == "fingertips_only":
            observations = fingertips
        else:
            object_position_observations = torch.stack(
                object_position_observations, dim=0
            )
            if self.feature_type == "object_only":
                observations = object_position_observations
            elif self.feature_type == "fingertips_and_object":
                observations = torch.concat(
                    [fingertips, object_position_observations], dim=-1
                )
            else:
                raise ValueError("Invalid feature type: {}".format(self.feature_type))

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
        self._pcds = torch.concat(pcds, dim=0)

    def __getitem__(self, idx):
        demo_start_index, demo_end_index = self._index_to_demo_indexes[idx]
        demo_pcd, demo_observation, demo_fingertips = (
            self._pcds[demo_start_index:demo_end_index],
            self._observations[demo_start_index:demo_end_index],
            self._fingertips[demo_start_index:demo_end_index],
        )
        idx_in_demo = idx - demo_start_index

        pcds, obs, act = [], [], [None] * (self.traj_len + self.action_chunking_len - 1)
        for i_past in range(self.traj_len - 1, -1, -1):
            current_ts_idx = idx_in_demo - i_past * self.ts_step
            obs.append(demo_observation[max(0, current_ts_idx)])
            pcds.append(demo_pcd[max(0, current_ts_idx)])

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
        pcds, obs, act = torch.stack(pcds), torch.stack(obs), torch.stack(act)
        return pcds, obs, act
