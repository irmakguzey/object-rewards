# Script to calculate the similarities between two trajectories
import glob
import os
from copy import deepcopy as copy
from enum import Enum, IntEnum

import matplotlib.pyplot as plt
import numpy as np
from dtw import *
from matplotlib import cm
from object_rewards.utils.metrics import cosine_distance, optimal_transport_plan
from object_rewards.utils.trajectory import get_rot_and_trans
from scipy.special import kl_div, rel_entr


class TrajSimMethod(IntEnum):
    KL_DIVERGENCE = 1
    OPTIMAL_TRANSPORT = 2
    DYNAMIC_TIME_WARPING = 3
    RELATIVE_ENTR = 4
    MEAN_SQR = 5


class TrajSimInputs(IntEnum):
    PRED_TRACKS = 1
    ROTATION_AND_TRANSLATION = 2
    MEAN_POSITION = 3
    TRANSLATION = 4


class TrajSimilarity:
    def __init__(
        self,
        method=TrajSimMethod.MEAN_SQR,
        input=TrajSimInputs.TRANSLATION,
        normalize_traj=True,
        curr_traj_frame_matches=-1,  # The last number of frames to include for the tracking
        ref_traj_frame_matches=-1,
        normalize_importance=False,
        img_shape=(640.0, 480.0),
    ):
        self.method_type = method
        self.input_type = input
        self.normalize_traj = normalize_traj
        self.normalize_importance = normalize_importance
        self.curr_traj_frame_matches = curr_traj_frame_matches
        self.ref_traj_frame_matches = ref_traj_frame_matches
        self.img_shape = img_shape

    def get_trajectories(self, curr_tracks, ref_tracks):
        # Make sure the size of trajectories are the same
        min_traj_len = min(
            curr_tracks.shape[0], ref_tracks.shape[0]
        )  # NOTE: this will not be a problem when we actually do this on the robot

        min_num_points = min(curr_tracks.shape[1], ref_tracks.shape[1])
        if self.curr_traj_frame_matches == -1:
            curr_tracks = curr_tracks[:min_traj_len, :min_num_points, :]
        else:
            curr_tracks = curr_tracks[
                min_traj_len - self.curr_traj_frame_matches : min_traj_len,
                :min_num_points,
                :,
            ]
        if self.ref_traj_frame_matches == -1:
            ref_tracks = ref_tracks[:min_traj_len, :min_num_points, :]
        else:
            ref_tracks = ref_tracks[
                min_traj_len - self.ref_traj_frame_matches : min_traj_len,
                :min_num_points,
                :,
            ]

        curr_traj = self._process_input(curr_tracks, self.normalize_traj)
        ref_traj = self._process_input(ref_tracks, self.normalize_traj)

        return curr_traj, ref_traj

    def get(self, curr_tracks, ref_tracks, **kwargs):
        # Make sure the size of trajectories are the same
        min_traj_len = min(
            curr_tracks.shape[0], ref_tracks.shape[0]
        )  # NOTE: this will not be a problem when we actually do this on the robot

        min_num_points = min(curr_tracks.shape[1], ref_tracks.shape[1])
        if self.curr_traj_frame_matches == -1:
            curr_tracks = curr_tracks[:min_traj_len, :min_num_points, :]
        else:
            curr_tracks = curr_tracks[
                min_traj_len - self.curr_traj_frame_matches : min_traj_len,
                :min_num_points,
                :,
            ]
        if self.ref_traj_frame_matches == -1:
            ref_tracks = ref_tracks[:min_traj_len, :min_num_points, :]
        else:
            ref_tracks = ref_tracks[
                min_traj_len - self.ref_traj_frame_matches : min_traj_len,
                :min_num_points,
                :,
            ]

        self.curr_tracks = curr_tracks
        self.ref_tracks = ref_tracks
        curr_traj = self._process_input(curr_tracks, self.normalize_traj)
        ref_traj = self._process_input(ref_tracks, self.normalize_traj)
        self.curr_traj = curr_traj
        self.ref_traj = ref_traj

        if self.normalize_importance:
            curr_dim_stds = np.std(self.ref_traj, axis=0)
            curr_dim_order = np.argsort(
                curr_dim_stds
            )  # Will return indices of this std descendingly

            curr_traj[:, curr_dim_order] *= np.asarray(range(len(curr_dim_order)))
            ref_traj[:, curr_dim_order] *= np.asarray(range(len(curr_dim_order)))

        if self.method_type == TrajSimMethod.KL_DIVERGENCE:
            return self._get_kl_divergence(curr_traj, ref_traj, **kwargs)

        if self.method_type == TrajSimMethod.RELATIVE_ENTR:
            return self._get_relative_entr(curr_traj, ref_traj, **kwargs)

        if self.method_type == TrajSimMethod.OPTIMAL_TRANSPORT:
            return self._get_optimal_transport(curr_traj, ref_traj, **kwargs)

        if self.method_type == TrajSimMethod.DYNAMIC_TIME_WARPING:
            return self._get_dynamic_time_warping(curr_traj, ref_traj, **kwargs)

        if self.method_type == TrajSimMethod.MEAN_SQR:
            return self._get_mean_sqr(curr_traj, ref_traj, **kwargs)

    def get_print_str(self):
        print_str = ""
        if self.method_type == TrajSimMethod.KL_DIVERGENCE:
            print_str += "kl_"
        elif self.method_type == TrajSimMethod.OPTIMAL_TRANSPORT:
            print_str += "ot_"
        elif self.method_type == TrajSimMethod.MEAN_SQR:
            print_str += "mean_sqr_"
        elif self.method_type == TrajSimMethod.DYNAMIC_TIME_WARPING:
            print_str += "dtw_"

        if self.input_type == TrajSimInputs.PRED_TRACKS:
            print_str += "pred_tracks"
        elif self.input_type == TrajSimInputs.ROTATION_AND_TRANSLATION:
            print_str += "rot_trans"
        elif self.input_type == TrajSimInputs.MEAN_POSITION:
            print_str += "mean_pos"
        elif self.input_type == TrajSimInputs.TRANSLATION:
            print_str += "trans"

        return print_str

    def visualize(self, similarity, plot_name):

        if self.input_type == TrajSimInputs.PRED_TRACKS:

            plot_ref_x = np.mean(self.ref_tracks, axis=1)[:, 0]
            plot_cur_x = np.mean(self.curr_tracks, axis=1)[:, 0]
            plot_ref_y = np.mean(self.ref_tracks, axis=1)[:, 1]
            plot_cur_y = np.mean(self.curr_tracks, axis=1)[:, 1]

        else:

            plot_ref_x = self.ref_traj[:, 0]
            plot_cur_x = self.curr_traj[:, 0]
            plot_ref_y = self.ref_traj[:, 1]
            plot_cur_y = self.curr_traj[:, 1]

        if self.input_type == TrajSimInputs.ROTATION_AND_TRANSLATION:
            ncols = 4
            plot_ref_rot = self.ref_traj[:, 2]
            plot_curr_rot = self.curr_traj[:, 2]
        else:
            ncols = 3

        _, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(3 * 6, 6))

        axs[0].plot(plot_ref_x, label="X of Reference Traj")
        axs[0].plot(plot_cur_x, label="X of Current Traj")
        axs[0].legend()
        axs[0].set_title("X Trajectories")
        axs[1].plot(plot_ref_y, label="Y of Reference Traj")
        axs[1].plot(plot_cur_y, label="Y of Current Traj")
        axs[1].set_title("Y Trajectories")
        axs[1].legend()

        if self.input_type == TrajSimInputs.ROTATION_AND_TRANSLATION:
            axs[2].plot(plot_ref_rot, label="Rotation of Reference Traj")
            axs[2].plot(plot_curr_rot, label="Rotation of Current Traj")
            axs[2].legend()
            axs[2].set_title("Rotation Trajectories")

        axs[ncols - 1].plot(similarity)
        axs[ncols - 1].set_title(f"Similarities for: {self.get_print_str()}")
        axs[ncols - 1].set_xlabel("Time")
        axs[ncols - 1].set_ylabel("Similarity")

        plt.savefig(f"{plot_name}.png")
        plt.close()

    def _get_kl_divergence(self, curr_traj, ref_traj, sinkhorn_rew_scale=1, **kwargs):
        # Take the abs of them
        curr_traj_abs = np.abs(curr_traj)
        ref_traj_abs = np.abs(ref_traj)

        # Compute KL divergence
        kl_div_result = kl_div(
            curr_traj_abs, ref_traj_abs
        )  # KL divergence is for how much a trajectory diverges from the other

        return (
            -sinkhorn_rew_scale
            * np.sum(  # It is always good to have a base to multiply things with
                kl_div_result, axis=-1
            )
        )  # For timewise similarity

    def _get_relative_entr(self, curr_traj, ref_traj, **kwargs):
        rel_entr_result = rel_entr(curr_traj, ref_traj)

        return -np.sum(rel_entr_result, axis=-1)

    def _get_optimal_transport(
        self, curr_traj, ref_traj, sinkhorn_rew_scale=1, **kwargs
    ):

        cost_matrix = cosine_distance(curr_traj, ref_traj, as_torch=False)
        # cost_matrix = np.abs(curr_traj - ref_traj)

        transport_plan = optimal_transport_plan(
            curr_traj,
            ref_traj,
            cost_matrix,
            method="sinkhorn",
            niter=100,
            exponential_weight_init=False,
            as_torch=False,
        )
        return -sinkhorn_rew_scale * np.diag(np.dot(transport_plan, cost_matrix.T))

    def _get_mean_sqr(self, curr_traj, ref_traj, sinkhorn_rew_scale=1, **kwargs):
        diff = ref_traj - curr_traj
        traj_dist = np.linalg.norm(diff, axis=1)

        return -sinkhorn_rew_scale * traj_dist

    def _get_dynamic_time_warping(self, curr_traj, ref_traj, sinkhorn_rew_scale=1):
        # Get alignment for each dimension of the trajectory, and find the difference
        # on the alignment of the arrays
        dim_alignment = dtw(curr_traj, ref_traj, keep_internals=True)

        dim_reward = -sinkhorn_rew_scale * np.abs(
            curr_traj[dim_alignment.index1] - ref_traj[dim_alignment.index2]
        )

        return dim_reward

    def _process_input(self, tracks, normalize=False):
        if self.input_type == TrajSimInputs.PRED_TRACKS:
            trajectory = tracks.copy()

            if normalize:
                trajectory[:, :, 0] /= self.img_shape[0]  # 1280.0
                trajectory[:, :, 1] /= self.img_shape[1]  # 720.0

            trajectory = np.reshape(trajectory, (trajectory.shape[0], -1))

        elif self.input_type == TrajSimInputs.MEAN_POSITION:
            trajectory = self._get_mean_pos(tracks=tracks)
            if normalize:
                trajectory[:, 0] /= self.img_shape[0]  # 1280.0
                trajectory[:, 1] /= self.img_shape[1]  # 720.0

        elif (
            self.input_type == TrajSimInputs.ROTATION_AND_TRANSLATION
            or self.input_type == TrajSimInputs.TRANSLATION
        ):
            trajectory = self._get_rot_and_trans(tracks=tracks, delta=False)
            # print(f"trajectory.shape: {trajectory.shape}")
            if normalize:
                trajectory[:, 0] /= self.img_shape[0]  # 1280.0
                trajectory[:, 1] /= self.img_shape[1]  # 720.0
                trajectory[:, 2] /= 2 * np.pi

            if self.input_type == TrajSimInputs.TRANSLATION:
                trajectory = trajectory[:, :2]

        return trajectory

    def _get_mean_pos(self, tracks):
        mean_pos = np.mean(tracks, axis=1)
        return mean_pos

    def _get_rot_and_trans(self, tracks, delta=False):

        return get_rot_and_trans(tracks=tracks, delta=delta)


# This method will load bunch of tracks numpy files, then will try
# bunch of similarty method and plot all of theirs' similarities through the trajectory
def test_similarities(expert_track_path, episode_track_paths):
    with open(expert_track_path, "rb") as f:
        expert_tracks = np.load(f)

    all_episode_tracks = []
    for episode_track_path in episode_track_paths:
        with open(episode_track_path, "rb") as f:
            all_episode_tracks.append(np.load(f))

    print(f"expert_tracks.shape: {expert_tracks.shape}")

    sinkhorn_rew_scale = 1.0
    auto_rew_scale_factor = 10.0
    methods_list = [
        TrajSimMethod.KL_DIVERGENCE,
        # TrajSimMethod.OPTIMAL_TRANSPORT,
        TrajSimMethod.MEAN_SQR,
        # TrajSimMethod.DYNAMIC_TIME_WARPING,
    ]
    inputs_list = [
        TrajSimInputs.PRED_TRACKS,
        TrajSimInputs.ROTATION_AND_TRANSLATION,
        TrajSimInputs.MEAN_POSITION,
        TrajSimInputs.TRANSLATION,
    ]

    fig, axs = plt.subplots(
        nrows=len(all_episode_tracks),
        ncols=len(methods_list) * len(inputs_list),
        figsize=(
            5 * (len(methods_list) * len(inputs_list)),
            6 * len(all_episode_tracks),
        ),
    )
    col_id = 0
    for sim_method in methods_list:
        for sim_input in inputs_list:
            sinkhorn_rew_scale = 1.0

            sim = TrajSimilarity(
                method=sim_method,
                input=sim_input,
                normalize_traj=True,
                normalize_importance=False,
            )

            for episode_id in range(len(all_episode_tracks)):

                print(f"episode_tracks: {all_episode_tracks[episode_id].shape}")

                episode_num = (
                    episode_track_paths[episode_id].split("/")[-1].split("_")[1]
                )

                similarities = sim.get(
                    curr_tracks=all_episode_tracks[episode_id],
                    ref_tracks=expert_tracks,
                    sinkhorn_rew_scale=sinkhorn_rew_scale,
                )

                if episode_id == 0:  # Update the scale and get the similarities again
                    sum_rewards = np.sum(similarities)

                    sinkhorn_rew_scale = (
                        sinkhorn_rew_scale
                        * auto_rew_scale_factor
                        / float(np.abs(sum_rewards))
                    )
                    similarities = sim.get(
                        curr_tracks=all_episode_tracks[episode_id],
                        ref_tracks=expert_tracks,
                        sinkhorn_rew_scale=sinkhorn_rew_scale,
                    )

                print(
                    f"episode_id: {episode_id}, col_id: {col_id} shapes: {sim.ref_traj[:, 0].shape}, {sim.curr_traj[:, 0].shape}"
                )

                if sim_input == TrajSimInputs.PRED_TRACKS:
                    plotted_ref_x = np.mean(
                        np.reshape(sim.ref_traj, (sim.ref_traj.shape[0], -1, 2)), axis=1
                    )[:, 0]
                    plotted_curr_x = np.mean(
                        np.reshape(sim.curr_traj, (sim.curr_traj.shape[0], -1, 2)),
                        axis=1,
                    )[:, 0]
                    plotted_ref_y = np.mean(
                        np.reshape(sim.ref_traj, (sim.ref_traj.shape[0], -1, 2)), axis=1
                    )[:, 1]
                    plotted_curr_y = np.mean(
                        np.reshape(sim.curr_traj, (sim.curr_traj.shape[0], -1, 2)),
                        axis=1,
                    )[:, 1]
                else:
                    plotted_ref_x = sim.ref_traj[:, 0]
                    plotted_curr_x = sim.curr_traj[:, 0]
                    plotted_ref_y = sim.ref_traj[:, 1]
                    plotted_curr_y = sim.curr_traj[:, 1]

                # axs[episode_id, col_id].plot(similarities)
                axs[episode_id, col_id].plot(
                    range(len(plotted_ref_x)), plotted_ref_x, label="ref_x"
                )
                axs[episode_id, col_id].plot(
                    range(len(plotted_curr_x)), plotted_curr_x, label="curr_x"
                )
                axs[episode_id, col_id].plot(
                    range(len(plotted_ref_y)), plotted_ref_y, label="ref_y"
                )
                axs[episode_id, col_id].plot(
                    range(len(plotted_curr_y)), plotted_curr_y, label="curr_y"
                )
                axs[episode_id, col_id].legend()
                method_str = sim.get_print_str()
                axs[episode_id, col_id].set_title(
                    f"{method_str}_episode_{episode_num} "
                )

                sum = np.sum(similarities)
                print(f"{sim.get_print_str()} - reward: {sum}, episode: {episode_num}")

                axs[episode_id, col_id].set_xlabel(
                    f"Reward Sum: {np.around(sum, 2)}, len(similarities): {len(similarities)}"
                )

            col_id += 1

    plt.savefig("trajectory_similarities_test.png", bbox_inches="tight")


def plot_trajectories(expert_track_path, episode_track_paths):
    # Will load all the tracks and plot translation and rotation of the trajectories
    with open(expert_track_path, "rb") as f:
        expert_tracks = np.load(f)

    all_episode_tracks = []
    episode_nums = []
    for episode_track_path in episode_track_paths:
        with open(episode_track_path, "rb") as f:
            all_episode_tracks.append(np.load(f))
            episode_num = episode_track_path.split("/")[-1].split("_")[1]
            episode_nums.append(int(episode_num))
    episode_nums = np.asarray(episode_nums)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(3 * 5, 5))

    nlines = len(all_episode_tracks[:])

    cmap = cm.get_cmap("RdYlGn", nlines)
    red_to_green_cmap = cm.get_cmap("RdYlGn", nlines)

    sorted_episode_ids = np.argsort(episode_nums)

    for i, episode_id in enumerate(sorted_episode_ids[:]):

        episode_tracks = all_episode_tracks[episode_id]

        sim = TrajSimilarity(
            method=TrajSimMethod.MEAN_SQR,
            input=TrajSimInputs.TRANSLATION,
            normalize_traj=True,
            normalize_importance=True,
            img_shape=(640, 480),
        )

        similarities = sim.get(
            curr_tracks=episode_tracks,
            ref_tracks=expert_tracks,
            sinkhorn_rew_scale=10,
        )

        episode_num = episode_nums[episode_id]

        if i == 0:

            axs[0].plot(sim.ref_traj[:, 0], label="Human Traj", color="orange")
            axs[1].plot(sim.ref_traj[:, 1], label="Human Traj", color="orange")

            axs[0].plot(
                sim.curr_traj[:, 0],
                label=f"Episode {episode_num}",
                color=cmap(i),
                alpha=1,
            )
            axs[1].plot(
                sim.curr_traj[:, 1],
                label=f"Episode {episode_num}",
                color=cmap(i),
                alpha=1,
            )

            axs[2].plot(
                similarities,
                label=f"Episode {episode_num}",
                color=red_to_green_cmap(i),
                alpha=1,
            ),

        else:
            axs[0].plot(
                sim.curr_traj[:, 0],
                label=(
                    f"Episode {episode_num}"
                    if episode_id == sorted_episode_ids[-1]
                    else None
                ),
                color=cmap(i),
                alpha=1 if episode_id == sorted_episode_ids[-1] else 1 / 3,
            )
            axs[1].plot(
                sim.curr_traj[:, 1],
                label=(
                    f"Episode {episode_num}"
                    if episode_id == sorted_episode_ids[-1]
                    else None
                ),
                color=cmap(i),
                alpha=1 if episode_id == sorted_episode_ids[-1] else 1 / 3,
            )

            axs[2].plot(
                similarities,
                label=(
                    f"Episode {episode_num}"
                    if episode_id == sorted_episode_ids[-1]
                    else None
                ),
                color=red_to_green_cmap(i),
                alpha=1 if episode_id == sorted_episode_ids[-1] else 1 / 3,
            )

        print(f"{nlines} - {i}")

        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        axs[0].set_xlabel("Episode Timesteps")
        axs[1].set_xlabel("Episode Timesteps")
        axs[2].set_xlabel("Episode Timesteps")

        axs[0].set_title("X Trajectories of Episodes and Human")
        axs[1].set_title("Y Trajectories of Episodes and Human")
        axs[2].set_title("Similarities of Trajectories")
