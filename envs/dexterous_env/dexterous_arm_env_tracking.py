import time

import cv2
import numpy as np
import torch
from openteach.utils.network import ZMQCameraSubscriber

from object_rewards.point_tracking.co_tracker import CoTrackerLangSam
from object_rewards.utils.constants import ALLEGRO_HOME_POSITION

from .dexterous_arm_env import DexterityEnv


# TODO: Finish this
class DexterityEnvTracking(DexterityEnv):
    def __init__(
        self,
        object_camera_port,
        object_camera_id,
        text_prompt,
        cotracker_grid_size,
        device,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.object_image_subscriber = ZMQCameraSubscriber(
            host=self.host_address,
            port=object_camera_port + object_camera_id,
            topic_type="RGB",
        )
        self.device = device

        self.cotracker_langsam = CoTrackerLangSam(
            device=self.device,
            frame_by_frame=True,
            grid_size=cotracker_grid_size,
        )
        self.text_prompt = text_prompt
        self._init_cotracker(text_prompt)

    def set_home_state(self, kinova_initial_pose=None):
        if kinova_initial_pose is None:
            kinova_home_pose = np.array([0.14, -0.44, 0.16, -0.59, 0.61, -0.36, -0.4])
        else:
            kinova_home_pose = kinova_initial_pose

        self.home_state = dict(kinova=kinova_home_pose, allegro=ALLEGRO_HOME_POSITION)

    def _init_cotracker(self, text_prompt):

        # Move the arm to the special reset pose
        image, _ = self.object_image_subscriber.recv_rgb_image()
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.img_shape = image.shape[:2]  # (H,W)

        self.window_frames = [
            torch.from_numpy(image).float().to(self.device)
            for _ in range(self.cotracker_langsam.cotracker.step * 2)
        ]

        # Get the initial queries from text prompt
        queries = self.cotracker_langsam.get_queries(
            frame=image, text_prompt=text_prompt, segm_mask=None
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
                # print("average_rot = 0")
            else:
                rot = n / np.linalg.norm(n)
                average_rot = np.mean(rot)

            self.total_rotation += (
                average_rot  # NOTE: If the KL divergence doesn't work well
            )

            self.prev_points = curr_points.copy()

        rot_and_trans = np.asarray(
            [
                self.total_translation[0] / self.img_shape[1],
                self.total_translation[1] / self.img_shape[0],
                self.total_rotation,
            ]
        )
        print(f"rot_and_trans: {rot_and_trans}")
        return rot_and_trans

    def _get_obj_position(self, tracks):

        # pred_tracks = self._get_all_tracks()
        print(f"obj position: {torch.mean(tracks[-1,:,:], dim=0)}")
        obj_position = torch.mean(tracks[-1, :, :], dim=0).detach().cpu().numpy()
        obj_position[0] /= self.img_shape[1]
        obj_position[1] /= self.img_shape[0]
        return obj_position

    def _get_tracks(self):
        image, _ = self.object_image_subscriber.recv_rgb_image()
        image = (
            torch.from_numpy(np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            .float()
            .to(self.device)
        )
        self.window_frames.append(image)

        pred_tracks, _ = self.cotracker_langsam.cotracker(
            video_chunk=self._get_video_chunk(), is_first_step=False, queries=None
        )
        print(f"pred_tracks.shape: {pred_tracks.shape}")

        return pred_tracks[0, :, :, :]

    def _get_obs(self):
        obs = super()._get_obs()

        tracks = self._get_tracks()
        object_position = self._get_obj_position(tracks)
        rot_and_trans = self._get_rot_and_trans(tracks)
        self.is_first_step = False
        obs["features"] = np.concatenate(
            [obs["features"], object_position, rot_and_trans], axis=-1
        )
        return obs

    def reset(self):
        self._reset_state()
        self._init_cotracker(text_prompt=self.text_prompt)
        self.is_first_step = True
        obs = self._get_obs()

        return obs
