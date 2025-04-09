import pickle

import cv2
from object_rewards.point_tracking import *
from object_rewards.utils import crop_transform, get_demo_action_ids
from PIL import Image as im
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader as loader
from tqdm import tqdm


class ObjectRewarder:
    def __init__(
        self,
        method_type,
        input_type,
        data_path,
        human_demo_num,
        view_num,
        device,
        human_prompt,
        robot_prompt,
        expert_frame_matches,
        episode_frame_matches,
        sinkhorn_rew_scale,
        auto_rew_scale_factor,
        experiment_name,
        work_dir,
        cotracker_grid_size,
        normalize_traj,
        normalize_importance,
        use_rgb_camera=False,
        **kwargs,
    ):

        self.__dict__.update(**kwargs)

        # In this code piece, do OT matching on
        # TODO: Test different trajectory similarity algorithms
        self.data_path = data_path
        self.demo_path = os.path.join(data_path, f"demonstration_{human_demo_num}")
        self.human_demo_num = human_demo_num
        self.view_num = view_num
        self.device = device
        self.human_prompt = human_prompt
        self.robot_prompt = robot_prompt
        self.sinkhorn_rew_scale = sinkhorn_rew_scale
        self.auto_rew_scale_factor = auto_rew_scale_factor
        self.expert_frame_matches = expert_frame_matches
        self.episode_frame_matches = episode_frame_matches
        self.experiment_name = experiment_name
        self.use_rgb_camera = use_rgb_camera
        self.normalize_traj = normalize_traj

        self.dump_path = os.path.join(
            f"{work_dir}/online_training_outs/langsam_cotracker_trajsim_outs/{self.experiment_name}"
        )
        if not os.path.exists:
            os.makedirs(self.dump_path)

        self.image_transform = T.Compose(
            [
                T.Resize((480, 640)),
                T.Lambda(lambda image: crop_transform(image, camera_view=view_num)),
            ]
        )

        # Initialize models
        self.predictor = CoTrackerLangSam(
            device=device,
            grid_size=cotracker_grid_size,
        )

        # Initialize the trajectory matching
        self.traj_matcher = TrajSimilarity(
            method=method_type,
            input=input_type,
            normalize_traj=normalize_traj,
            normalize_importance=normalize_importance,
            img_shape=(640.0, 480.0),  # width / height wise
            curr_traj_frame_matches=episode_frame_matches,
            ref_traj_frame_matches=expert_frame_matches,
        )

        # Get the expert images' point tracking
        self.demo_action_ids = get_demo_action_ids(
            data_path=data_path, view_num=view_num, demo_num=human_demo_num
        )
        self._get_human_tracks(visualize=True)

    def _load_image(self, frame_id):

        dir_name = (
            f"cam_{self.view_num}_fish_eye_images"
            if self.use_rgb_camera
            else f"cam_{self.view_num}_rgb_images"
        )

        image_path = os.path.join(
            self.demo_path,
            "{}/frame_{}.png".format(dir_name, str(frame_id).zfill(5)),
        )
        img = self.image_transform(loader(image_path))  # This loads images as PIL image
        return img

    def _get_human_tracks(self, visualize=False):
        # Get all the images
        indices_file_name = (
            f"image_indices_fish_eye_cam_{self.view_num}.pkl"
            if self.use_rgb_camera
            else f"image_indices_cam_{self.view_num}.pkl"
        )
        image_indices_path = os.path.join(self.demo_path, indices_file_name)
        with open(image_indices_path, "rb") as file:
            image_indices = pickle.load(file)

        # Stack all the image representations
        expert_imgs = []
        pbar = tqdm(total=self.demo_action_ids[1] - self.demo_action_ids[0])
        for action_id in range(self.demo_action_ids[0], self.demo_action_ids[1]):
            img_frame_id = image_indices[action_id][1]
            img = self._load_image(frame_id=img_frame_id)
            expert_imgs.append(img)
            pbar.update(1)
            pbar.set_description("Getting the expert images")

        self.expert_imgs = np.stack(expert_imgs, axis=0)

        if os.path.exists(f"{self.dump_path}/expert_tracks.npy"):
            with open(f"{self.dump_path}/expert_tracks.npy", "rb") as f:
                self.expert_pred_tracks = np.load(f)
        else:
            self.expert_pred_tracks = self._get_tracks(
                frames=self.expert_imgs,
                prompt=self.human_prompt,
                visualize=visualize,
                is_human=True,
            )
            with open(f"{self.dump_path}/expert_tracks.npy", "wb") as f:
                np.save(f, self.expert_pred_tracks)

    def _get_tracks(self, frames, prompt, visualize=False, is_human=True, episode_id=0):

        # Make sure you get the number of frames desired
        frame_matches = (
            self.expert_frame_matches if is_human else self.episode_frame_matches
        )
        if frame_matches != -1:
            frames = frames[-frame_matches:]

        tracks = self.predictor.get_segmented_tracks_by_batch(
            frames=frames, text_prompt=prompt
        )

        if visualize:
            self.predictor.visualize_tracks(
                pred_tracks=tracks,
                frames=frames,
                video_name=(
                    f"rewarder_tracks_human_demo_{self.human_demo_num}"
                    if is_human
                    else f"rewarder_tracks_robot_episode_{episode_id}"
                ),
                dump_dir=f"{self.dump_path}/point_tracks",
            )

        return tracks

    def _get_episode_tracks(self, obs, visualize=False, episode_id=0):

        episode_frames = np.stack(
            [np.array(pil_image) for pil_image in obs["pil_image_obs"]], axis=0
        )

        if os.path.exists(f"{self.dump_path}/episode_{episode_id}_tracks.npy"):
            with open(f"{self.dump_path}/episode_{episode_id}_tracks.npy", "rb") as f:
                episode_tracks = np.load(f)
        else:
            episode_tracks = self._get_tracks(
                frames=episode_frames,
                prompt=self.robot_prompt,
                visualize=visualize,
                is_human=False,
                episode_id=episode_id,
            )
            with open(f"{self.dump_path}/episode_{episode_id}_tracks.npy", "wb") as f:
                np.save(f, episode_tracks)

        return episode_tracks

    def get(self, obs, visualize, episode_id, **kwargs):
        episode_tracks = self._get_episode_tracks(
            obs=obs, visualize=True, episode_id=episode_id
        )

        similarities = self.traj_matcher.get(
            curr_tracks=episode_tracks,
            ref_tracks=self.expert_pred_tracks,
            sinkhorn_rew_scale=self.sinkhorn_rew_scale,
        )
        sim_dump_dir = f"{self.dump_path}/traj_sims"
        if not os.path.exists(sim_dump_dir):
            os.makedirs(sim_dump_dir)

        if visualize:
            self.traj_matcher.visualize(
                similarities,
                plot_name=f"{sim_dump_dir}/episode_{episode_id}_reward_{np.sum(similarities)}",
            )

        return similarities

    def update_scale(self, current_rewards):
        sum_rewards = np.sum(current_rewards)
        self.sinkhorn_rew_scale = (
            self.sinkhorn_rew_scale
            * self.auto_rew_scale_factor
            / float(np.abs(sum_rewards))
        )
