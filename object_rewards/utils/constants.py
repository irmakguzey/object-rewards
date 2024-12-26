import numpy as np

# Constans for camera images
VISION_IMAGE_MEANS = [0.4191, 0.4445, 0.4409]
VISION_IMAGE_STDS = [0.2108, 0.1882, 0.1835]

REALSENSE_ALL_INTRINSICS = {
    0: np.array(  # These intrinsics are for the camera id 1 - 141722071999
        [
            [900.57477747, 0.0, 638.12883091],
            [0.0, 900.57477747, 366.73743594],
            [0.0, 0.0, 1.0],
        ]
    ),
    1: np.array(  # These intrinsics are for the camera id 2 - 023422073116
        [
            [
                919.230285644531,
                0.0,
                643.210754394531,
            ],  # NOTE: These are still old intrinsics , to be modified!
            [0.0, 917.224609375, 371.862487792969],
            [0.0, 0.0, 1.0],
        ]
    ),
}
REALSENSE_ALL_DISTORTION = {
    0: np.array(  # Camera 1 distortion
        [
            -1.51647215e00,
            2.75404119e01,
            -2.54301263e-03,
            7.46352349e-04,
            9.56617500e01,
            -1.51968157e00,
            2.66225571e01,
            9.66198953e01,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
            0.00000000e00,
        ]
    ),
    1: np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
}
REALSENSE_INTRINSICS = REALSENSE_ALL_INTRINSICS[0]
REALSENSE_DISTORTION = REALSENSE_ALL_DISTORTION[0]

REALSENSE_SIDE_CAMERA_INTRINSICS = REALSENSE_ALL_INTRINSICS[
    1
]  # NOTE: Camera 2 - to be modified
REALSENSE_SIDE_CAMERA_DISTORTION = REALSENSE_ALL_DISTORTION[1]

OCULUS_INTRINSICS = np.array(
    [
        [804.50982384, 0.0, 493.67921725],
        [0.0, 804.50982384, 676.26613348],
        [0.0, 0.0, 1.0],
    ]
)

OCULUS_DISTORTION = np.array(
    [
        -7.10867181e-01,
        5.38216808e00,
        -5.05950088e-03,
        7.02100603e-03,
        9.87236639e-01,
        -2.65290700e-01,
        4.89742506e00,
        3.51895939e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]
)

# Aruco marker IDs
HAND_ARUCO_ID = 181
TABLE_ARUCO_ID = 189
REWARD_ARUCO_ID = 180
HAND_ARUCO_SIZE = 0.05

ALLEGRO_HOME_POSITION = np.array(
    [
        -0.0124863,
        -0.10063279,
        0.7970152,
        0.7542225,
        -0.01191735,
        -0.10746645,
        0.78338414,
        0.7421494,
        0.06945032,
        -0.02277208,
        0.8780185,
        0.76349473,
        1.0707821,
        0.424525,
        0.30425942,
        0.79608095,
    ]
)

# Transformation between the kinova end effector to allegro_mount link on the urdf
EEF_TO_END = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0.12], [0, 0, 0, 1]]
