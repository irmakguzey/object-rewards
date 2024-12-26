# Script to move the arm to different positions and take images through realsense

import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from cv2 import aruco
from openteach.robot.allegro.allegro import AllegroHand
from openteach.robot.kinova import KinovaArm
from openteach.utils.network import ZMQCameraSubscriber
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from object_rewards.utils import transforms
from object_rewards.utils.visualization import plot_axes, project_axes, project_poses
from object_rewards.utils.constants import *


# Class to move the arm around, wait until it goes to that position and take
# pics with realsense
class CalibrateBase:

    def __init__(
        self, host, calibration_pics_dir, cam_idx, marker_size=HAND_ARUCO_SIZE
    ):

        # Create the directory if it doesn't exist
        os.makedirs(calibration_pics_dir, exist_ok=True)
        self.calibration_pics_dir = calibration_pics_dir
        self.marker_size = marker_size
        self.host = host

        self._init_camera_subscribers(cam_idx=cam_idx)

    def _init_robots(self):
        self.hand = AllegroHand()
        self.arm = KinovaArm()

        time.sleep(1)

    def _init_camera_subscribers(self, cam_idx):

        self.image_subscriber = ZMQCameraSubscriber(
            host=self.host, port=10005 + cam_idx, topic_type="RGB"
        )

    def home_hand(self):
        self.hand.move(
            [  # Will move the hand to have a fist position for safety
                0.03165045,
                1.639166,
                1.7101783,
                0.61326075,
                -0.10495461,
                1.7395127,
                1.5695766,
                0.7708849,
                0.03578715,
                1.7371392,
                1.6312336,
                0.7145495,
                1.3683549,
                0.22154835,
                0.7254645,
                0.9895618,
            ]
        )

    def get_image(self):
        image, timestamp = self.image_subscriber.recv_rgb_image()
        return image

    def get_arm_pose(self):
        return self.arm.get_cartesian_position()

    def is_arm_there(self, des_pose):
        curr_pose = self.get_arm_pose()

        # Tvec difference
        tvec_diff = np.linalg.norm(des_pose[:3] - curr_pose[:3])

        # Get the rotation difference
        curr_quat = curr_pose[3:]
        des_quat = des_pose[3:]
        quat_diff = transforms.quat_multiply(
            curr_quat, transforms.quat_inverse(des_quat)
        )
        angle_diff = 180 * np.linalg.norm(transforms.quat2axisangle(quat_diff)) / np.pi

        print(f"Angle diff: {angle_diff} Tvec diff: {tvec_diff}")
        if angle_diff < 10 and tvec_diff < 1e-1:
            return True

        return False

    def move_arm_to_pose(self, frame_id):
        des_pose = self.kinova_poses[frame_id]

        while not self.is_arm_there(des_pose=des_pose):
            self.arm.move_coords(self.kinova_poses[frame_id])
            time.sleep(0.2)

    def save_img(self, frame_id):
        img = self.get_image()
        img_path = os.path.join(
            self.calibration_pics_dir, f"frame_{str(frame_id).zfill(5)}.png"
        )
        cv2.imwrite(img_path, img)

    def save_img(self, frame_id):
        img = self.get_image()
        img_path = os.path.join(
            self.calibration_pics_dir, f"frame_{str(frame_id).zfill(5)}.png"
        )
        cv2.imwrite(img_path, img)

    def save_poses(self):

        print("** SAVING POSES **")
        self._init_robots()

        arm_poses = []
        frame_id = 0
        while frame_id < 50:

            try:
                x = input("Press Enter to save arm and image")
                pose = self.get_arm_pose()
                arm_poses.append(pose)
                self.save_img(frame_id)
                frame_id += 1

            except KeyboardInterrupt:
                break

        print("Saving the poses!")
        poses = np.stack(arm_poses, axis=0)

        with open(os.path.join(self.calibration_pics_dir, "arm_poses.npy"), "wb") as f:
            np.save(f, poses)

    def load_poses(self):
        with open(os.path.join(self.calibration_pics_dir, "arm_poses.npy"), "rb") as f:
            self.arm_poses = np.load(f)

        return self.arm_poses

    def get_aruco_corners_in_2d(self, frame_id):

        # Load the image
        img_path = os.path.join(
            self.calibration_pics_dir, f"frame_{str(frame_id).zfill(5)}.png"
        )
        img = np.asarray(cv2.imread(img_path))
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        aruco_detector = aruco.ArucoDetector(
            dictionary=aruco_dict, detectorParams=parameters
        )

        corners, ids, _ = aruco_detector.detectMarkers(img_gray)

        hand_corners = None
        for i in range(len(corners)):
            if ids[i][0] == HAND_ARUCO_ID:
                hand_corners = corners[i]
                break

        if hand_corners is None:
            return None
        return hand_corners.squeeze()

    def get_aruco_corners_in_3d(self, arm_pose):
        # position of three points in the end effector frame
        p1 = np.array(
            [0.034, self.marker_size / 2, -0.07 - self.marker_size / 2, 1]
        )  # Upper left clockwise points
        p2 = np.array([0.034, -self.marker_size / 2, -0.07 - self.marker_size / 2, 1])
        p3 = np.array([0.034, -self.marker_size / 2, -0.07 + self.marker_size / 2, 1])
        p4 = np.array([0.034, self.marker_size / 2, -0.07 + self.marker_size / 2, 1])

        # transfer the cartesian position into rotation matrix
        end_trans = np.array(arm_pose[0:3])
        end_quat = np.array(arm_pose[3:])
        rotation_end_to_base = Rotation.from_quat(end_quat).as_matrix()
        rotation_end_to_base = np.hstack(
            (rotation_end_to_base, end_trans.reshape(3, 1))
        )
        rotation_end_to_base = np.vstack((rotation_end_to_base, np.array([0, 0, 0, 1])))

        corner_1 = np.dot(rotation_end_to_base, p1)[:3]
        corner_2 = np.dot(rotation_end_to_base, p2)[:3]
        corner_3 = np.dot(rotation_end_to_base, p3)[:3]
        corner_4 = np.dot(rotation_end_to_base, p4)[:3]

        corners = np.stack((corner_1, corner_2, corner_3, corner_4))

        return corners

    def get_base_to_camera(self):

        # Find the length of kinova_poses
        arm_poses = self.load_poses()
        print(arm_poses.shape)
        len_frames = arm_poses.shape[0]

        pts_3d = []
        pts_2d = []
        num_of_frames_missing = 0
        for frame_id in range(len_frames):
            curr_2d = self.get_aruco_corners_in_2d(frame_id)  # (4,2) - A_C
            if curr_2d is None:
                num_of_frames_missing += 1
                continue
            arm_pose = arm_poses[frame_id]
            curr_3d = self.get_aruco_corners_in_3d(arm_pose)  # (4,3) - A_B

            pts_3d.append(curr_3d)
            pts_2d.append(curr_2d)

        pts_2d = np.concatenate(pts_2d, axis=0)
        pts_3d = np.concatenate(pts_3d, axis=0)

        # Solve this equation
        _, rvec, tvec = cv2.solvePnP(
            pts_3d,
            pts_2d,
            REALSENSE_INTRINSICS,
            REALSENSE_DISTORTION,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        rot_mtx = Rotation.from_rotvec(rvec.squeeze()).as_matrix()
        homo_base_to_cam = transforms.turn_frames_to_homo(
            rvec=rot_mtx, tvec=tvec.squeeze()
        )

        return homo_base_to_cam  # H_B_C

    def get_aruco_corners_in_2d_from_robot(self, arm_pose, base_to_camera):
        # First get the corners in 3d from the robot
        corners_to_base_tvecs = self.get_aruco_corners_in_3d(
            arm_pose=arm_pose
        )  # (4,3) H_A_B
        # Turn them into matrices just to project tha axes properly
        corners_to_base = np.stack(
            len(corners_to_base_tvecs) * [np.eye(4)], axis=0
        )  # (4,4,4)
        corners_to_base[:, :3, 3] = corners_to_base_tvecs[:, :3]

        # Take these corners to the camera
        corners_to_camera = base_to_camera @ corners_to_base

        # Project these corners
        projected_corners_in_camera = project_poses(
            poses=corners_to_camera,
            intrinsic_matrix=REALSENSE_INTRINSICS,
            distortion=REALSENSE_DISTORTION,
            scale=0,
        )[:, 3, :, :].squeeze()

        return projected_corners_in_camera

    def get_calibration_error_in_2d(self):

        arm_poses = self.load_poses()
        print(arm_poses.shape)
        len_frames = arm_poses.shape[0]

        corner_errors = []

        base_to_camera = self.get_base_to_camera()

        for frame_id in range(len_frames):

            corners_in_2d_from_camera = self.get_aruco_corners_in_2d(
                frame_id=frame_id
            )  # (4,2)
            if corners_in_2d_from_camera is None:
                continue
            corners_in_2d_from_robot = self.get_aruco_corners_in_2d_from_robot(
                arm_pose=arm_poses[frame_id], base_to_camera=base_to_camera
            )  # (4,2)

            corner_diffs = np.linalg.norm(
                corners_in_2d_from_camera - corners_in_2d_from_robot, axis=-1
            )  # (4,)

            corner_errors.append(corner_diffs)

        pixel_error_sep = np.mean(corner_errors, axis=0)  # (4,)

        pixel_error_all = np.mean(pixel_error_sep)  # (1,)

        print(
            "** CALIBRATION ERROR IN 2D - SEP CORNERS: {}, ALL CORNERS: {} **".format(
                pixel_error_sep, pixel_error_all
            )
        )

    def calibrate(self, save_transform, save_img):

        homo_base_to_camera = self.get_base_to_camera()
        print(
            f"** IN CALIBRATION - base-to-camera tranform ***:\n{homo_base_to_camera}"
        )

        if save_transform:
            np.save(
                f"{self.calibration_pics_dir}/homo_base_to_cam_{self.cam_idx}.npy",
                homo_base_to_camera,
            )

        if save_img:
            rotation_matrix = homo_base_to_camera[:3, :3]
            tvec = homo_base_to_camera[:3, 3]

            # Plotting as RGB
            img = cv2.imread(
                f"{self.calibration_pics_dir}/frame_00000.png", cv2.IMREAD_COLOR
            )
            img = np.multiply(np.ones(img.shape, np.uint8), img)
            # project homo_cam_to_base
            projected_H_B_C = project_axes(
                rvec=rotation_matrix,
                tvec=tvec,
                intrinsic_matrix=self.intrinsics.K,
                scale=0.03,
            )
            img = plot_axes(axes=[projected_H_B_C], img=img)
            filename = (
                f"{self.calibration_pics_dir}/base_test_rgb_{self.marker_size}.png"
            )

            # Plot the corners of the aruco
            aruco_corners = self.get_aruco_corners_in_2d(0)
            aruco_corners = [tuple(pt.ravel().astype(int)) for pt in aruco_corners]
            img = cv2.circle(
                img, aruco_corners[0], radius=6, color=(255, 0, 0), thickness=-1
            )  # Top left
            img = cv2.circle(
                img, aruco_corners[1], radius=6, color=(0, 255, 0), thickness=-1
            )  # Top right
            img = cv2.circle(
                img, aruco_corners[2], radius=6, color=(0, 0, 255), thickness=-1
            )  # Bottom right
            img = cv2.circle(
                img, aruco_corners[3], radius=6, color=(200, 200, 200), thickness=-1
            )  # Bottom left

            cv2.imwrite(filename, img)

        return homo_base_to_camera

    def get_images_from_poses(
        self,
    ):  # Method that will move the arm to loaded kinova poses and take pictures
        print("** IN MOVING + SAVING IMAGES **")

        self.load_poses()
        self._init_robots()
        self.home_hand()
        x = input("Press Enter to make sure everything is okay! ")

        pbar = tqdm(total=len(self.arm_poses))
        for frame_id in range(len(self.arm_poses)):

            self.move_arm_to_pose(frame_id=frame_id)

            print(" ** TAKING PICTURE **")
            time.sleep(1)

            self.save_img(frame_id=frame_id)

            pbar.update(1)

        pbar.close()

    def load_base_to_camera(self):
        H_B_C = np.load(
            f"{self.calibration_pics_dir}/homo_base_to_cam_{self.cam_idx}.npy"
        )
        return H_B_C
