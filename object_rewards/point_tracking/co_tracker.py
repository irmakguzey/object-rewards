import os
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
import pickle

from object_rewards.point_tracking.lang_sam import LangSAM
from object_rewards.utils.video_recorder import VideoRecorder
from object_rewards.utils.visualization import (
    plot_rotation_and_translation,
    vis_sam_mask,
)
from object_rewards.utils.data import get_demo_action_ids
from object_rewards.utils.augmentations import crop_transform
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader as loader


def get_segmented_video_from_human(
    data_path, view_num, demo_num, is_fish_eye, object_prompt, segm_video_name
):
    demo_action_ids = get_demo_action_ids(
        data_path=data_path, view_num=view_num, demo_num=demo_num
    )

    demo_path = f"{data_path}/demonstration_{demo_num}"

    image_transform = T.Compose(
        [
            T.Resize((480, 640)),
            T.Lambda(lambda image: crop_transform(image, camera_view=view_num)),
        ]
    )

    def _load_image(frame_id):

        dir_name = (
            f"cam_{view_num}_fish_eye_images"
            if is_fish_eye
            else f"cam_{view_num}_rgb_images"
        )

        image_path = os.path.join(
            demo_path,
            "{}/frame_{}.png".format(dir_name, str(frame_id).zfill(5)),
        )
        img = image_transform(loader(image_path))  # This loads images as PIL image
        return img

    indices_file_name = (
        f"image_indices_fish_eye_cam_{view_num}.pkl"
        if is_fish_eye
        else f"image_indices_cam_{view_num}.pkl"
    )
    image_indices_path = os.path.join(demo_path, indices_file_name)
    with open(image_indices_path, "rb") as file:
        image_indices = pickle.load(file)

    # Stack all the image representations
    expert_imgs = []
    pbar = tqdm(total=demo_action_ids[1] - demo_action_ids[0])
    for action_id in range(demo_action_ids[0], demo_action_ids[1]):
        img_frame_id = image_indices[action_id][1]
        img = _load_image(frame_id=img_frame_id)
        expert_imgs.append(img)
        pbar.update(1)
        pbar.set_description("Getting the expert images")

    expert_imgs = np.stack(expert_imgs, axis=0)
    print(
        "in rewarder - _set_expr_imgs -> self.expert_imgs.shape: {}".format(
            expert_imgs.shape
        )
    )

    video = expert_imgs
    print(f"video.shape: {video.shape}")
    # Get the image from the frames
    first_frame = video[0]
    # Rotate it to a PIL image to pass to langsam
    image_pil = im.fromarray(first_frame)

    # Initialize langsam
    langsam = LangSAM()
    masks, boxes, phrases, logits, embeddings = langsam.predict(
        image_pil=image_pil, text_prompt=object_prompt
    )

    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            "/home/irmak/Workspace/co-tracker/checkpoints/cotracker2.pth"
        )
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(
        video, grid_size=50, segm_mask=masks[0][None, None].type(torch.uint8)
    )

    print(f"pred_tracks.shape: {pred_tracks.shape}")

    vis = Visualizer(
        save_dir="./videos",
        pad_value=0,
        linewidth=2,
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=segm_video_name,
    )


def get_segmented_video(video_path, object_prompt, segm_video_name):
    video = read_video_from_path(video_path)[:, :]

    print(f"video.shape: {video.shape}")
    # Get the image from the frames
    first_frame = video[0]
    # Rotate it to a PIL image to pass to langsam
    image_pil = im.fromarray(first_frame)

    # Initialize langsam
    langsam = LangSAM()
    masks, boxes, phrases, logits, embeddings = langsam.predict(
        image_pil=image_pil, text_prompt=object_prompt
    )

    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            "/home/irmak/Workspace/co-tracker/checkpoints/cotracker2.pth"
        )
    )
    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(
        video, grid_size=50, segm_mask=masks[0][None, None].type(torch.uint8)
    )

    print(f"pred_tracks.shape: {pred_tracks.shape}")

    vis = Visualizer(
        save_dir="./videos",
        pad_value=0,
        linewidth=2,
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename=segm_video_name,
    )


class CoTrackerLangSam:
    def __init__(
        self,
        device,
        checkpoint_path="/home/irmak/Workspace/co-tracker/checkpoints/cotracker2.pth", # TODO: co-tracker checkpoint path
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

        # self.langsam = LangSAM()
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


if __name__ == "__main__":

    get_segmented_video(
        video_path="/data_ssd/irmak/third-person-manipulation-online-trainings/online_training_outs/eval_video/videos/2024.10.16T10-34_h2r_bread_pic_larger_gen_langsam_cotracker_trans_mean_sqr/h2r_bread_pic_larger_gen_eval_0_8_r-3.3004539489746096.mp4",
        object_prompt="orange bread",
        segm_video_name="robot_bread_picking_larger_cotracker_vis_8",
    )
