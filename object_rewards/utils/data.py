import glob
import os
import pickle
import re
from pathlib import Path

import h5py
import numpy as np


# This function takes in a directory containing demos after preprocessing
# output a dictionary, that takes the form of:
# {
#  demo_num : {frame_idx: [start_frame_idx, end_frame_idx], demo_idx: [start_demo_idx, end_demo_idx]}
#  }
def crop_demo(data_path, view_num, demo_num=None):
    # NOTE: The demo number here does not taking into account of the calibration steps!!!
    demo_dict = dict()
    roots = sorted(glob.glob(f"{data_path}/demonstration_*"))
    if demo_num is None:
        for demo_path in roots:
            pattern = r"\d+$"
            demo_num = int(re.findall(pattern, demo_path)[0])

            try:
                demo_dict[demo_num] = get_cropped_demo_dict(demo_path, view_num)
            except:
                continue  # If there is an error continue to the next demonstration - this error occurs due to having missing keypoints file

        assert (
            len(demo_dict.keys()) > 0
        ), "None of the demos during demo cropping got processed"

    else:

        demo_path = f"{data_path}/demonstration_{demo_num}"
        demo_dict[demo_num] = get_cropped_demo_dict(demo_path, view_num)

    return demo_dict


def get_cropped_demo_dict(demo_path, view_num):
    # Get keypoint timestamps and record_status
    timestep_path = Path(demo_path) / "keypoints.h5"
    index_path = Path(demo_path) / "keypoint_indices.pkl"
    with h5py.File(timestep_path, "r") as file:
        keypoint_timestamps = list(file["timestamps"])
        keypoint_status = list(file["keypoint_status"])
    with open(index_path, "rb") as file:
        indices = pickle.load(file)
    demo_start_index, demo_end_index = get_indices_from_timestampps(
        keypoint_timestamps, keypoint_status, indices
    )

    # Now get frame_start_index, frame_end_index
    frame_index_path = Path(demo_path) / "image_indices_cam_{}.pkl".format(view_num)
    with open(frame_index_path, "rb") as file:
        frame_indices = pickle.load(file)
    frame_start_index, frame_end_index = (
        frame_indices[demo_start_index][1],
        frame_indices[demo_end_index][1],
    )

    # Build the dictionary
    demo_dict = dict(
        frame_idx=[frame_start_index, frame_end_index],
        demo_idx=[demo_start_index, demo_end_index],
    )

    return demo_dict


# This function find the indices of start and end point of keypoints
def get_indices_from_timestampps(keypoint_timestamps, keypoint_status, indices):
    start_and_end = []
    sampled_timestamps = [keypoint_timestamps[idx[1]] for idx in indices]
    for i in keypoint_status:
        closest_index = find_closest_timestamp_index(sampled_timestamps, i)
        start_and_end.append(closest_index)
    demo_start_index, demo_end_index = start_and_end
    return demo_start_index, demo_end_index


def find_closest_timestamp_index(keypoint_timestamps, sampled_timestamp):
    distance_list = []
    for timestamp in keypoint_timestamps:
        distance_list.append(abs(timestamp - sampled_timestamp))
    distance_list = np.array(distance_list)
    min_index = np.argmin(distance_list)
    return min_index


def get_all_demo_frame_ids(data_path, camera_id):
    all_demos_dict = crop_demo(data_path=data_path, camera_id=camera_id)
    all_demos_frame_ids = convert_to_demo_frame_ids(demo_dict=all_demos_dict)
    return all_demos_frame_ids


def get_single_demo_frame_ids(data_path, demo_num, camera_id):
    demo_dict = crop_demo(data_path=data_path, camera_id=camera_id, demo_num=demo_num)
    demo_frame_ids = convert_to_demo_frame_ids(demo_dict=demo_dict)
    return np.asarray(demo_frame_ids[str(demo_num)])


def convert_to_demo_frame_ids(demo_dict):
    # Just returns the image frame ids of the demo dictionary
    demo_frame_ids = dict()
    for demo_num in demo_dict.keys():
        frame_ids = demo_dict[demo_num]["frame_idx"]
        demo_frame_ids[str(demo_num)] = frame_ids
    return demo_frame_ids


def get_demo_action_ids(data_path, view_num, demo_num):
    demo_frame_ids = get_single_demo_frame_ids(
        data_path=data_path, camera_id=view_num, demo_num=demo_num
    )
    return get_demo_action_ids_from_frame_ids(
        root=data_path,
        image_frame_ids=demo_frame_ids,
        demo_num=demo_num,
        view_num=view_num,
    )


def get_demo_action_ids_from_frame_ids(root, image_frame_ids, demo_num, view_num):
    action_ids = []
    # Will traverse through image indices and return the idx that have the image_frame_ids
    image_indices_path = os.path.join(
        root,
        "demonstration_{}".format(demo_num),
        "image_indices_cam_{}.pkl".format(view_num),
    )
    with open(image_indices_path, "rb") as file:
        image_indices = pickle.load(file)
    i = 0
    for action_id, (demo_id, image_id) in enumerate(image_indices):
        if image_id == image_frame_ids[i]:
            action_ids.append(action_id)
            i += 1

        if i == 2 or i == len(image_frame_ids):
            break

    return action_ids
