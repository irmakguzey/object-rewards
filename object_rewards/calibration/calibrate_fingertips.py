# Class to return fingertip positions wrt to the aruco marker

import os
import pickle
from tqdm import tqdm
import cv2
import h5py
import numpy as np
from openteach.utils.network import ZMQCameraSubscriber


from object_rewards.utils import (
    turn_frames_to_homo,
    aruco_in_world, 
    get_markers,
    get_average_aruco_in_frame
)
from object_rewards.utils.constants import TABLE_ARUCO_ID, HAND_ARUCO_SIZE


class CalibrateFingertips:

    def __init__(
        self, demo_path, realsense_view_num, host, camera_port, marker_size=HAND_ARUCO_SIZE, H_A_C=None
    ):

        self.demo_path = demo_path
        self.realsense_view_num = realsense_view_num
        self._load_indices()  # NOTE: This script should be called after the preprocessing

        # Set the necessary parameters
        self.translation_ratio = 1
        self.translation_camera_to_eye = np.array([0, 0.01, 0])
        self.demo_start_frame = 20
        self.calibration_start_frame = 5
        self.marker_size = marker_size

        # Check if demo_in_aruco has been dumped yet
        demo_in_aruco_path = os.path.join(self.demo_path, "demo_in_aruco.pkl")
        print(demo_in_aruco_path)
        if not os.path.exists(demo_in_aruco_path):
            self._set_average_aruco_in_world()
            self.demo_in_aruco = self._dump_demo_in_aruco()
        else:
            with open(demo_in_aruco_path, "rb") as f:
                self.demo_in_aruco = pickle.load(f)

        self.image_subscriber = ZMQCameraSubscriber(
            host=host, port=camera_port + realsense_view_num, topic_type="RGB"
        )
        if H_A_C is None:
            self._set_average_aruco_in_camera()
        else:
            self.H_A_C = H_A_C

    def _load_indices(self):

        # Get the preprocessed data paths
        oculus_indices_path = os.path.join(self.demo_path, "oculus_indices.pkl")
        keypoint_indices_path = os.path.join(self.demo_path, "keypoint_indices.pkl")
        image_indices_path = os.path.join(
            self.demo_path, f"image_indices_cam_{self.realsense_view_num}.pkl"
        )

        # Get keypoints
        keypoint_path = os.path.join(self.demo_path, "keypoints.h5")
        with h5py.File(keypoint_path, "r") as keypoint_file:
            self.oculus_movement = keypoint_file["oculus_camera"][()]
            self.keypoints = keypoint_file["original_keypoints"][()]
            self.keypoint_timestamps = keypoint_file["timestamps"][()]

        with open(image_indices_path, "rb") as file:
            self.image_indices = pickle.load(file)
        with open(oculus_indices_path, "rb") as file:
            self.oculus_indices = pickle.load(file)
        with open(keypoint_indices_path, "rb") as file:
            self.keypoint_indices = pickle.load(file)

    def _set_average_aruco_in_world(self):

        print("** Setting Average ARUCO in Oculus World ** ")
        oculus_path = os.path.join(self.demo_path, "oculus_0_images")

        pbar = tqdm(total=self.demo_start_frame - self.calibration_start_frame)
        past_Rs = []
        for idx in range(self.calibration_start_frame, self.demo_start_frame):

            keypoint_index = self.keypoint_indices[idx][1]
            eye_position = np.array(self.oculus_movement[keypoint_index][:3])
            eye_quat = np.array(self.oculus_movement[keypoint_index][3:])
            oculus_index = self.oculus_indices[idx][1]
            oculus_image_path = os.path.join(
                oculus_path, f"frame_{oculus_index:05d}.png"
            )
            frame = cv2.imread(oculus_image_path)
            markers, frame, rvec, tvec = get_markers(  # NOTE: Clean this code
                frame,
                True,
                aruco_id=TABLE_ARUCO_ID,
                camera="oculus",
                marker_size=self.marker_size,
            )
            pbar.update(1)
            if len(markers) != 0:
                R_aruco_in_world = aruco_in_world(
                    rvec,
                    tvec,
                    self.translation_camera_to_eye,
                    eye_position,
                    eye_quat,
                    self.translation_ratio,
                )
                past_Rs.append(R_aruco_in_world)

        pbar.close()

        # Get average R_aruco_in_world
        self.R_aruco_in_world = np.mean(past_Rs, axis=0)

    def _get_aruco_in_world_at_frame(self, frame_id):
        oculus_path = os.path.join(self.demo_path, "oculus_0_images")

        keypoint_index = self.keypoint_indices[frame_id - 1][1]
        eye_position = np.array(self.oculus_movement[keypoint_index][:3])
        eye_quat = np.array(self.oculus_movement[keypoint_index][3:])
        oculus_index = self.oculus_indices[frame_id][1]
        oculus_image_path = os.path.join(oculus_path, f"frame_{oculus_index:05d}.png")
        oculus_frame = cv2.imread(oculus_image_path)
        markers, _, rvec, tvec = get_markers(
            oculus_frame, True, camera="oculus", marker_size=self.marker_size
        )

        if len(markers) != 0:
            aruco_in_world_at_frame = aruco_in_world(
                rvec,
                tvec,
                self.translation_camera_to_eye,
                eye_position,
                eye_quat,
                self.translation_ratio,
            )

            return aruco_in_world_at_frame

        return None

    def _set_average_aruco_in_camera(self):

        calibration_len_frames = 50
        print(
            f"*** Calibrating ARUCO in CAMERA for {calibration_len_frames} frames ***"
        )

        pbar = tqdm(total=calibration_len_frames)
        img_frames = []
        for _ in range(calibration_len_frames):
            image, _ = self.image_subscriber.recv_rgb_image()
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_frames.append(img_gray)
            pbar.update(1)

        pbar.close()

        H_A_C = get_average_aruco_in_frame(
            frames=img_frames,
            intrinsics_matrix=REALSENSE_INTRINSICS,
            distortion_matrix=REALSENSE_DISTORTION,
            aruco_id=TABLE_ARUCO_ID,
            marker_size=self.marker_size,
        )

        self.H_A_C = H_A_C

    def _dump_demo_in_aruco(self):

        print("** Dumping Demo in Aruco **")

        # Save fingertips wrt Aruco
        demo_in_aruco = []
        pbar = tqdm(total=len(self.oculus_indices) - self.demo_start_frame)

        # print("self.keypoint_indices: {}".format(self.keypoint_indices))

        for idx in range(self.demo_start_frame, len(self.oculus_indices)):

            # Also get fingertip keypoints
            keypoint_index = self.keypoint_indices[idx][1]
            keypoint = self.keypoints[keypoint_index]
            fingertips = convert_raw_keypoints(keypoint)

            fingertips_to_aruco = fingertips_from_world_to_aruco(
                self.R_aruco_in_world, fingertips, self.translation_ratio
            )
            demo_in_aruco.append(fingertips_to_aruco)

            pbar.update(1)

        pbar.close()
        with open(os.path.join(self.demo_path, "demo_in_aruco.pkl"), "wb") as file:
            pickle.dump(demo_in_aruco, file)

        return demo_in_aruco

    def fingertip_reshape(self, fingertips, finger_to_change=[], ratio=1):
        # Given finegrtips in the base frame
        # We want to change the length of the link to human demos morphology closer to robot hand
        for finger in finger_to_change:
            position = fingertips[finger]
            orientation = fingertips[finger + "_1"]
            v = orientation - position
            new_orientation = position + v * ratio
            fingertips[finger + "_1"] = new_orientation
        return fingertips

    # We will read from the pickle file if it doesn't exist
    def get_fingertips_to_aruco(self, frame_id):

        ft_index = frame_id - self.demo_start_frame
        fingertip_to_aruco = self.demo_in_aruco[ft_index]
        fingertip_to_aruco = self.fingertip_reshape(
            fingertip_to_aruco, ["thumb"], ratio=2
        )

        # fingertip_tvecs = []
        H_F_A = []
        finger_types = [
            "index",
            "middle",
            "ring",
            "thumb",
            "index_1",
            "middle_1",
            "ring_1",
            "thumb_1",
        ]
        for finger_type in finger_types:

            fingertip_tvec = fingertip_to_aruco[finger_type].squeeze()
            fingertip_rot_mtx = np.eye(3)

            homo_fingertip_to_aruco = turn_frames_to_homo(
                rvec=fingertip_rot_mtx, tvec=fingertip_tvec
            )

            H_F_A.append(homo_fingertip_to_aruco)

        H_F_A = np.stack(H_F_A)

        return H_F_A

    def fingertip_reshape(self, fingertips, finger_to_change=[], ratio=1):
        # Given finegrtips in the base frame
        # We want to change the length of the link to human demos morphology closer to robot hand
        # print("fingertips: {} being reshaped!".format(finger_to_change))
        for finger in finger_to_change:
            position = fingertips[finger]
            orientation = fingertips[finger + "_1"]
            v = orientation - position
            new_orientation = position + v * ratio
            fingertips[finger + "_1"] = new_orientation
        return fingertips

    def get_fingertips_to_base(
        self, frame_id, base_to_camera=None
    ):  # base_to_camera will be received by CalibrateBase class

        # Get the average aruco_to_camera
        H_A_C = self.H_A_C

        # Get fingertips to aruco
        H_F_A = self.get_fingertips_to_aruco(frame_id=frame_id)

        # Get the base_to_camera
        H_B_C = base_to_camera

        # Get the fingertips to camera
        H_F_C = H_A_C @ H_F_A

        # Get fingertips to base
        H_F_B = np.linalg.pinv(H_B_C) @ H_F_C

        return H_F_B

    def get_wrist_to_aruco(self, frame_id):

        ft_index = frame_id - self.demo_start_frame
        fingertip_to_aruco = self.demo_in_aruco[ft_index]

        # fingertip_tvecs = []
        H_F_A = []
        finger_types = ["wrist"]
        for finger_type in finger_types:

            fingertip_tvec = fingertip_to_aruco[finger_type].squeeze()
            fingertip_rot_mtx = np.eye(3)

            homo_fingertip_to_aruco = turn_frames_to_homo(
                rvec=fingertip_rot_mtx, tvec=fingertip_tvec
            )

            H_F_A.append(homo_fingertip_to_aruco)

        H_F_A = np.stack(H_F_A)

        return H_F_A

    def get_wrist_to_base(
        self, frame_id, base_to_camera=None
    ):  # base_to_camera will be received by CalibrateBase class

        # Get the average aruco_to_camera
        H_A_C = self.H_A_C
        # Get fingertips to aruco
        H_F_A = self.get_wrist_to_aruco(frame_id=frame_id)

        # Get the base_to_camera
        H_B_C = base_to_camera

        # Get the fingertips to camera
        H_F_C = H_A_C @ H_F_A

        # Get fingertips to base
        H_F_B = np.linalg.pinv(H_B_C) @ H_F_C

        return H_F_B

    def get_finger_roots_to_aruco(self, frame_id):

        ft_index = frame_id - self.demo_start_frame
        fingertip_to_aruco = self.demo_in_aruco[ft_index]

        # fingertip_tvecs = []
        H_F_A = []
        finger_types = ["index_3", "middle_3", "ring_3", "thumb_3"]
        for finger_type in finger_types:

            fingertip_tvec = fingertip_to_aruco[finger_type].squeeze()
            fingertip_rot_mtx = np.eye(3)

            homo_fingertip_to_aruco = turn_frames_to_homo(
                rvec=fingertip_rot_mtx, tvec=fingertip_tvec
            )

            H_F_A.append(homo_fingertip_to_aruco)

        H_F_A = np.stack(H_F_A)

        return H_F_A

    def get_finger_roots_to_base(
        self, frame_id, base_to_camera=None
    ):  # base_to_camera will be received by CalibrateBase class

        # Get the average aruco_to_camera
        H_A_C = self.H_A_C
        # Get fingertips to aruco
        H_F_A = self.get_finger_roots_to_aruco(frame_id=frame_id)

        # Get the base_to_camera
        H_B_C = base_to_camera

        # Get the fingertips to camera
        H_F_C = H_A_C @ H_F_A

        # Get fingertips to base
        H_F_B = np.linalg.pinv(H_B_C) @ H_F_C
        # print('H_F_B[0] finger_in_base: {}'.format(H_F_B[0]))

        # return None

        return H_F_B
