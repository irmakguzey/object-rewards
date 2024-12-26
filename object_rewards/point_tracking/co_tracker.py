import os
import pickle
from base64 import b64encode
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.models.core.model_utils import get_points_on_a_grid
from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from PIL import Image as im
from tqdm import tqdm

from object_rewards.point_tracking.lang_sam import LangSAM
from object_rewards.utils.video_recorder import VideoRecorder
from object_rewards.utils.visualization import (
    plot_rotation_and_translation,
    vis_sam_mask,
)


class CoTrackerLangSam:
    def __init__(
        self,
        device=0,
        checkpoint_path="../checkpoints/cotracker2.pth",  # TODO: co-tracker checkpoint path
        is_online=True,
        frame_by_frame=False,
        grid_size=50,
    ):
        self.device = device
        self.is_online = is_online
        self.langsam = LangSAM(device=device)
        if is_online:

            self.cotracker = CoTrackerOnlinePredictor(
                checkpoint=checkpoint_path, single_frame=frame_by_frame
            ).to(
                device
            )  # Try this if doesn't work it should work like above
        else:
            self.cotracker = CoTrackerPredictor(checkpoint=checkpoint_path).to(device)

        self.total_translation, self.total_rotation = (
            np.zeros(2, dtype=np.float32),
            0.0,
        )
        self.frame_by_frame = frame_by_frame
        self.grid_size = grid_size

    def _calculate_translation(self, curr_features, prev_features):
        # Calculate the translation difference
        curr_mean = np.mean(curr_features, axis=0)
        prev_mean = np.mean(prev_features, axis=0)
        diff_mean = curr_mean - prev_mean

        # Add it to the total translation
        self.total_translation[0] += diff_mean[0]
        self.total_translation[1] += diff_mean[1]

    def _calculate_rotation(self, curr_features, prev_features):
        curr_mean = np.mean(curr_features, axis=0)
        prev_mean = np.mean(prev_features, axis=0)

        # Bring all the points to the same space
        curr_feat_norm = curr_features - curr_mean
        prev_feat_norm = prev_features - prev_mean

        # Calculate the rotation
        n = np.cross(prev_feat_norm, curr_feat_norm)
        if (n == 0.0).all():
            average_rot = 0
        else:
            rot = n / np.linalg.norm(n)
            average_rot = np.mean(rot)

        self.total_rotation += average_rot

    def get_segmented_tracks_from_video(
        self, video_path, text_prompt, frame_matches=-1
    ):
        if frame_matches != -1:
            frames = read_video_from_path(video_path)[-frame_matches:, :]
        else:
            frames = read_video_from_path(video_path)[:, :]
        print("frames.shape: {}".format(frames.shape))

        if self.is_online:
            if self.frame_by_frame:
                return (
                    self.get_segmented_tracks_by_frames(
                        frames=frames, text_prompt=text_prompt
                    ),
                    frames,
                )
            else:
                return (
                    self.get_segmented_tracks_by_batch(
                        frames=frames, text_prompt=text_prompt
                    ),
                    frames,
                )

        return (
            self.get_segmented_tracks_from_frames_offline(
                frames=frames, text_prompt=text_prompt
            ),
            frames,
        )

    def get_segmented_tracks_from_frames_offline(self, frames, text_prompt):
        segm_mask = self.get_segmentation(frame=frames[0], text_prompt=text_prompt)

        video = (
            torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        )

        pred_tracks, _ = self.cotracker(
            video,
            grid_size=self.grid_size,
            segm_mask=segm_mask[None, None].type(torch.uint8),
        )

        return pred_tracks[0, :].detach().cpu().numpy()

    def get_segmented_tracks_by_batch(self, frames, text_prompt=None, queries=None):

        print("** GETTING TRACKS BY BATCH **")

        if queries is None:
            assert not (
                text_prompt is None
            ), "text_prompt cannot be None if queries is None"
            segm_mask = self.get_segmentation(frame=frames[0], text_prompt=text_prompt)
            segm_mask = segm_mask[None, None].type(torch.uint8)

            queries = self.get_queries(
                frame=torch.from_numpy(frames[0]).float().to(self.device),
                segm_mask=segm_mask,
            )

        video = (
            torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        )

        self.total_translation, self.total_rotation = (
            np.zeros(2, dtype=np.float32),
            0.0,
        )

        # Start the tracking online
        self.cotracker(
            video_chunk=video,
            is_first_step=True,
            queries=queries,
        )
        # batch_size = 64
        pbar = tqdm(
            total=len(
                range(0, video.shape[1] - self.cotracker.step, self.cotracker.step)
            )
        )
        for ind in range(0, video.shape[1] - self.cotracker.step, self.cotracker.step):
            pred_tracks, pred_visibility = self.cotracker(
                video_chunk=video[:, ind : ind + (self.cotracker.step * 2)],
                is_first_step=False,
            )
            pbar.update(1)
            pbar.set_description(
                "ind: {}, pred_tracks.shape: {}".format(ind, pred_tracks.shape)
            )

        pbar.close()

        return pred_tracks[0, :].detach().cpu().numpy()

    def get_segmented_tracks_by_frames(self, frames, text_prompt):

        self.total_translation, self.total_rotation = (
            np.zeros(2, dtype=np.float32),
            0.0,
        )

        segm_mask = self.get_segmentation(frame=frames[0], text_prompt=text_prompt)
        segm_mask = segm_mask[None, None].type(torch.uint8)

        frames = torch.from_numpy(frames).float().to(self.device)

        window_frames = [
            frames[0] for _ in range(self.cotracker.step * 2)
        ]  # For the beginning
        is_first_frame = True
        pbar = tqdm(total=len(frames))
        all_pred_tracks = []
        print(f"self.cotracker.step: {self.cotracker.step}")
        for frame_id, frame in enumerate(frames):

            pred_tracks, _ = self.single_frame_track(
                segm_mask=segm_mask,
                is_first_step=is_first_frame,
                window_frames=window_frames,
            )
            is_first_frame = False

            window_frames.append(frame)
            pbar.update(1)
            if frame_id != 0:
                pbar.set_description(f"pred_tracks: {pred_tracks.shape}")

        pbar.close()

        return pred_tracks[0, (self.cotracker.step - 1) * 2 :].detach().cpu().numpy()

    def get_queries(self, frame, segm_mask=None, text_prompt=None):
        if segm_mask is None:  # text_prompt cannot be none
            segm_mask = self.get_segmentation(frame=frame, text_prompt=text_prompt)
            segm_mask = segm_mask[None, None].type(torch.uint8)
            frame = torch.from_numpy(frame).float().to(self.device)

        # Set the queries first
        grid_pts = get_points_on_a_grid(
            self.grid_size, self.cotracker.interp_shape, device=frame.device
        )
        segm_mask = F.interpolate(
            segm_mask, tuple(self.cotracker.interp_shape), mode="nearest"
        )
        point_mask = segm_mask[0, 0][
            (grid_pts[0, :, 1]).round().long().cpu(),
            (grid_pts[0, :, 0]).round().long().cpu(),
        ].bool()
        # print(f"point_mask.shape: {point_mask.shape}")
        grid_pts = grid_pts[:, point_mask]
        grid_query_frame = 0  # setting to the default value
        queries = torch.cat(
            [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
            dim=2,
        ).repeat(1, 1, 1)

        H, W, _ = frame.shape
        queries[
            :, :, 1:
        ] /= queries.new_tensor(  # NOTE: This gets multiplied in the online code...
            [
                (self.cotracker.interp_shape[1] - 1) / (W - 1),
                (self.cotracker.interp_shape[0] - 1) / (H - 1),
            ]
        )

        return queries

    def single_frame_track(self, segm_mask, is_first_step, window_frames):
        if is_first_step:
            queries = self.get_queries(frame=window_frames[0], segm_mask=segm_mask)

        # Create the video_chunk
        video_chunk = (
            torch.tensor(
                torch.stack(window_frames[-self.cotracker.step * 2 :]),
                device=window_frames[0].device,
            )  # NOTE: If this doesn't work, then will include all the current frames
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)

        # Pass it through the model
        return self.cotracker(
            video_chunk=video_chunk,
            is_first_step=is_first_step,
            queries=queries if is_first_step else None,
        )

    def visualize_tracks(
        self,
        pred_tracks,
        frames,
        video_name,
        dump_dir="./videos",
    ):

        video_recorder = VideoRecorder(
            save_dir=Path(dump_dir), resize_and_transpose=False
        )

        for i in range(len(pred_tracks)):
            curr_features = pred_tracks[i, :]
            if i == 0:
                prev_features = pred_tracks[0, :]
            else:
                prev_features = pred_tracks[i - 1, :]
            self._calculate_translation(curr_features, prev_features)
            self._calculate_rotation(curr_features, prev_features)

            frame_img = frames[i, :]
            frame_img = plot_rotation_and_translation(
                frame_img,
                translation=self.total_translation,
                rotation=self.total_rotation,
                translation_pos=(470, 300),  # (1500, 600),
                ellipse_pos=(400, 300),  # (1200, 600),
            )

            for feat in curr_features:
                x, y = np.int32(feat.ravel())
                frame_img = cv2.circle(frame_img, (x, y), 5, (0, 0, 255), -1)

            if frame_img.shape[0] != 3:
                frame_img = np.transpose(
                    frame_img, (2, 0, 1)
                )  # NOTE: I have not idea how we didn't have this problem previously

            if i == 0:
                video_recorder.init(obs=frame_img)
            else:
                video_recorder.record(obs=frame_img)

        video_recorder.save(f"{video_name}.mp4")

    def get_segmentation(self, frame, text_prompt):
        image_pil = im.fromarray(frame)

        # Initialize langsam
        masks, _, _, _, _ = self.langsam.predict(
            image_pil=image_pil, text_prompt=text_prompt
        )

        _, ax = plt.subplots(1, 1)
        ax = vis_sam_mask(ax, mask=masks[0])
        plt.savefig("segm_mask.png")

        return masks[0]  # Return the mask with the highest confidence
