import os

import cv2
import numpy as np
from cv2 import aruco
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from object_rewards.utils.constants import (
    OCULUS_DISTORTION,
    OCULUS_INTRINSICS,
    REALSENSE_DISTORTION,
    REALSENSE_INTRINSICS,
)


def estimate_pose_single_markers(
    corners, marker_size, mtx, distortion
):  # NOTE: This is implemented since aruco doesn't have that method anymore
    """
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    """
    marker_points = np.array(
        [
            [-marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, marker_size / 2, 0],
            [marker_size / 2, -marker_size / 2, 0],
            [-marker_size / 2, -marker_size / 2, 0],
        ],
        dtype=np.float32,
    )
    trash = []
    rvecs = []
    tvecs = []

    # print('corners: {}'.format(corners))
    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE
        )
        # print('R: {}, t: {}'.format(R, t))
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return np.asarray(rvecs), np.asarray(tvecs), trash


# Plotting method to add markers - NOTE: Add this to utils or smth
def get_markers(
    image, plot_marker_axis, marker_id, camera="realsense", marker_size=0.05
):
    markers = []
    if camera == "realsense":
        camera_intrinsics = REALSENSE_INTRINSICS
        distortion_coefficients = REALSENSE_DISTORTION
    elif camera == "oculus":
        camera_intrinsics = OCULUS_INTRINSICS
        distortion_coefficients = OCULUS_DISTORTION

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    aruco_detector = aruco.ArucoDetector(
        dictionary=aruco_dict, detectorParams=parameters
    )
    corners, ids, rejectedImgPoints = aruco_detector.detectMarkers(gray)

    frame_markers = aruco.drawDetectedMarkers(img.copy(), corners)
    frame_axis = frame_markers.copy()
    frame_markers = cv2.cvtColor(frame_markers, cv2.COLOR_BGR2RGB)

    for i in range(len(corners)):
        if ids[i][0] == marker_id:
            rvec, tvec, _ = estimate_pose_single_markers(
                corners[i], marker_size, camera_intrinsics, np.zeros(5)
            )

            markers.append([rvec, tvec])

            if plot_marker_axis == True:
                if i == 0:
                    frame_axis = cv2.drawFrameAxes(
                        frame_markers.copy(),
                        camera_intrinsics,
                        distortion_coefficients,
                        rvec,
                        tvec,
                        0.01,
                    )
                else:
                    frame_axis = cv2.drawFrameAxes(
                        frame_axis.copy(),
                        camera_intrinsics,
                        distortion_coefficients,
                        rvec,
                        tvec,
                        0.01,
                    )

    if len(markers) > 0:
        return markers, frame_axis, markers[0][0], markers[0][1]
    else:
        return markers, frame_axis, None, None


def convert_raw_keypoints(
    keypoints,
):
    fingertips = {}
    for idx, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):

        num = idx - 6
        if finger == "pinky":
            for joint_num in range(3):
                joint_name = finger + "_" + str(joint_num + 1)
                fingertips[joint_name] = keypoints[
                    num * 9 - joint_num * 3 : num * 9 - joint_num * 3 + 3
                ]
        else:
            for joint_num in range(3):
                joint_name = finger + "_" + str(joint_num + 1)
                fingertips[joint_name] = keypoints[
                    num * 9 - joint_num * 3 - 3 : num * 9 - joint_num * 3
                ]
        tip_num = 3 * (idx - 5)
        if finger == "pinky":
            fingertips[finger] = keypoints[tip_num:]
        else:
            fingertips[finger] = keypoints[tip_num : tip_num + 3]

    fingertips["wrist"] = keypoints[-66:-63]
    return fingertips


def aruco_in_world(
    rvec, tvec, translation_camera_to_eye, eye_position, eye_quat, translation_ratio
):
    R_aruco_in_world = []
    rotation_matrix_marker, _ = cv2.Rodrigues(rvec)
    R_aruco_in_camera = np.hstack(
        (rotation_matrix_marker, np.array(tvec.squeeze().reshape(3, 1)))
    )
    R_aruco_in_camera = np.vstack((R_aruco_in_camera, np.array([0, 0, 0, 1])))

    translation = -np.array(
        [0.02, 0.015, -0.0]
    )  # -x: right, -y: up, -z: forward (with respect to the camera)
    theta = 5 * np.pi / 180
    rotation_around_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    phi = 4 * np.pi / 180
    rotation_matrix_y = np.array(
        [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
    )
    alpha = -3 * np.pi / 180
    rotation_matrix_z = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    rotation = np.eye(3)
    rotation = np.dot(np.dot(rotation_around_x, rotation_matrix_y), rotation_matrix_z)
    actual_eye_in_camera = np.eye(4)
    actual_eye_in_camera[:3, :3] = rotation
    actual_eye_in_camera[:3, 3] = translation.squeeze().reshape(
        3,
    )
    R_actual_aruco_in_camera = np.dot(
        np.linalg.inv(actual_eye_in_camera), R_aruco_in_camera
    )
    R_actual_aruco_in_camera[:3, 3] = (
        R_actual_aruco_in_camera[:3, 3] / translation_ratio
    )

    left_to_right = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    r = Rotation.from_quat(eye_quat)
    R = r.as_matrix()
    R_eye_in_world = np.eye(4)
    R_eye_in_world[:3, :3] = R  # fill the rotation part
    R_eye_in_world[:3, 3] = eye_position.squeeze().reshape(
        3,
    )
    R_camera_in_world = R_eye_in_world
    R_camera_in_world[:3, 3] = R_camera_in_world[
        :3, 3
    ] + translation_camera_to_eye.squeeze().reshape(
        3,
    )
    R_aruco_in_world = np.dot(
        np.dot(R_camera_in_world, left_to_right), R_actual_aruco_in_camera
    )

    return R_aruco_in_world  # Need to bear in mind that here the auroco marker has flipped y


# Find the position of finegrtips with respect to Aruco
# Known: fingertips_to_world, Auroco_to_world, NEED: fingertips_to_auroco
def fingertips_from_world_to_aruco(R_aruco_in_world, fingertips, translation_ratio):
    homog_fingertips = {}
    for finger in fingertips.keys():
        fingertips_in_camera = np.array(fingertips[finger])
        homog_fingertips[finger] = np.hstack((fingertips_in_camera, 1))

    # Apply transformation
    fingertips_in_aruco = {}
    for finger in homog_fingertips.keys():
        fingertips = homog_fingertips[finger]
        fingertips = np.dot(np.linalg.inv(R_aruco_in_world), fingertips.reshape(4, 1))
        # Need to do the translation ratio, and flip the y axis as well
        fingertips = fingertips[:3] / fingertips[3]
        fingertips = fingertips * translation_ratio
        fingertips_in_aruco[finger] = fingertips
    return fingertips_in_aruco


# Now we have camera_to_world every frame, we can get rvec, tvec without aruco in view
def aruco_in_camera(
    R_aruco_in_world,
    translation_camera_to_eye,
    eye_position,
    eye_quat,
    translation_ratio,
):
    # Known: camera_in_world, aruco_in_world, Want: rvec, tvec of aruco in camera
    r = Rotation.from_quat(eye_quat)
    R = r.as_matrix()
    R_eye_in_world = np.eye(4)
    R_eye_in_world[:3, :3] = R  # fill the rotation part
    R_eye_in_world[:3, 3] = eye_position.squeeze().reshape(
        3,
    )

    R_camera_in_world = R_eye_in_world
    R_camera_in_world[:3, 3] = R_camera_in_world[
        :3, 3
    ] + translation_camera_to_eye.squeeze().reshape(
        3,
    )
    left_to_right = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    R_aruco_in_camera = np.dot(
        np.linalg.inv(np.dot(R_camera_in_world, left_to_right)), R_aruco_in_world
    )
    # we need to apply transaltion ratio here since we want translation in world frame
    R_aruco_in_camera[:3, 3] = R_aruco_in_camera[:3, 3] * translation_ratio
    translation = -np.array([0.02, 0.015, -0.0])
    theta = 5 * np.pi / 180
    rotation_around_x = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )
    phi = 4 * np.pi / 180
    rotation_matrix_y = np.array(
        [[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]]
    )
    alpha = -3 * np.pi / 180
    rotation_matrix_z = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
    )
    rotation = np.eye(3)
    rotation = np.dot(np.dot(rotation_around_x, rotation_matrix_y), rotation_matrix_z)
    actual_camera_in_eye = np.eye(4)
    actual_camera_in_eye[:3, :3] = rotation
    actual_camera_in_eye[:3, 3] = translation.squeeze().reshape(
        3,
    )
    R_actual_aruco_in_camera = np.dot(actual_camera_in_eye, R_aruco_in_camera)

    rvec = Rotation.from_matrix(R_actual_aruco_in_camera[:3, :3]).as_rotvec()
    tvec = R_actual_aruco_in_camera[:3, 3]

    return rvec, tvec
