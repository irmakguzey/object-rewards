# Given a video this script will return the segmented video with the tracks added
import os
import pickle

import numpy as np
import torch
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from PIL import Image as im
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader as loader
from tqdm import tqdm

from object_rewards.point_tracking.lang_sam import LangSAM
from object_rewards.utils.augmentations import crop_transform
from object_rewards.utils.data import get_demo_action_ids


# This method will give segmented video from processed human demonstrations
# for this to work one needs to process human demonstrations saved using HuDOR
# then run this method with desired arguments
def get_segmented_video_from_human_demonstrations(
    data_path, view_num, demo_num, use_rgb_camera, object_prompt, segm_video_name
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
            if use_rgb_camera
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
        if use_rgb_camera
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
    video = expert_imgs

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
    model = CoTrackerPredictor()

    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(
        video, grid_size=50, segm_mask=masks[0][None, None].type(torch.uint8)
    )

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


# This will return the segmented video directly from a given video
def get_segmented_video(video_path, object_prompt, segm_video_name):
    video = read_video_from_path(video_path)[:, :]

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
    model = CoTrackerPredictor()

    if torch.cuda.is_available():
        model = model.cuda()
        video = video.cuda()
    pred_tracks, pred_visibility = model(
        video, grid_size=50, segm_mask=masks[0][None, None].type(torch.uint8)
    )

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


if __name__ == "__main__":

    # Example usage
    get_segmented_video(
        video_path="human_video_without_mask.mp4",
        object_prompt="orange card",
        segm_video_name="human_segmented_video",
    )
    get_segmented_video(
        video_path="robot_video_without_mask.mp4",
        object_prompt="orange card",
        segm_video_name="robot_segmented_video",
    )

    # Example usage
    # get_segmented_video_from_human_demonstrations(
    #     data_path = '/data/card_sliding',
    #     view_num = 0,
    #     demo_num = 3,
    #     is_fish_eye = False,
    #     object_prompt = 'orange card',
    #     segm_video_name='human_card_sliding_cotracker_vis'
    # )
