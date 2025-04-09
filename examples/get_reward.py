# Given two videos this example will give rewards of how similar these two videos are by looking at the objects' trajectories
from object_rewards.point_tracking import *
CHECKPOINT_PATH = "/home/irmak/Workspace/robot-hand-project/submodules/object-rewards/submodules/co-tracker/checkpoints/cotracker2.pth"

def get_object_rewards(
    current_video_path, reference_video_path, text_prompt, visualize=False
):

    # Initialize predictor and trajectory similarity modules
    predictor = CoTrackerLangSam(
        checkpoint_path=CHECKPOINT_PATH,
        grid_size=100  # This value can change wrt how small / large the object is
    )
    traj_matcher = TrajSimilarity()

    # Get the predicted points for both videos given text prompt
    current_tracks = predictor.get_segmented_tracks_from_video(
        video_path=current_video_path, text_prompt=text_prompt
    )[0]
    reference_tracks = predictor.get_segmented_tracks_from_video(
        video_path=reference_video_path, text_prompt=text_prompt
    )[0]

    similarities = traj_matcher.get(
        curr_tracks=current_tracks,
        ref_tracks=reference_tracks,
    )

    if visualize:
        traj_matcher.visualize(
            similarities,
            plot_name=f"trajectory_similarities",
        )

    return similarities


if __name__ == "__main__":
    similarities = get_object_rewards(
        current_video_path="human_video_without_mask.mp4",
        reference_video_path="robot_video_without_mask.mp4",
        text_prompt="orange card",
        visualize=True,
    )

    print(f"SUM OF THE REWARD: {np.sum(similarities)}")
