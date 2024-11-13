import glob
import numpy as np
import torch

from PIL import Image

def load_one_episode(fn):
    with open(fn, "rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
    return episode


def load_root_episodes(root_path):
    episodes = []
    fns = sorted(glob.glob(f"{root_path}/*.npz"))
    for i, fn in enumerate(fns):
        episode = load_one_episode(fn)
        episodes.append(episode)

    return episodes


def load_all_episodes(root_path=None, roots=None):
    if roots is None:
        roots = glob.glob(f"{root_path}/*")
    all_episodes = []
    for root in roots:
        root_episodes = load_root_episodes(root)
        all_episodes += root_episodes

    return all_episodes


# Given a list of episodes concatenate the given key features
def merge_all_episodes(all_episodes, included_keys):
    concat_episodes = dict()
    for key in included_keys:
        all_key_values = []
        for curr_episode in all_episodes:
            curr_value = curr_episode[key]
            all_key_values.append(curr_value)
        concat_episodes[key] = np.concatenate(all_key_values, 0)
    return concat_episodes


# This image transform will have the totensor and normalization only
def load_episode_demos(all_episodes, image_transform):
    episode_demos = []
    for episode in all_episodes:
        transformed_image_obs = []
        for image_obs in episode["pixels"]:
            pil_image = Image.fromarray(np.transpose(image_obs, (1, 2, 0)), "RGB")
            transformed_image_obs.append(image_transform(pil_image))
        episode_demos.append(
            dict(
                image_obs=torch.stack(transformed_image_obs, 0),
                tactile_repr=torch.FloatTensor(episode["tactile"]),
            )
        )

    return episode_demos