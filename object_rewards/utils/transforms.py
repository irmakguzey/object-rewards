import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def get_diff_in_homo(matrix_a, matrix_b):

    rotmax_a, tvec_a = turn_homo_to_frames(matrix_a)
    rotmax_b, tvec_b = turn_homo_to_frames(matrix_b)

    # Get the translational error
    translational_dist = np.linalg.norm(tvec_a - tvec_b)

    # Get the rotational error
    quat_a = Rotation.from_matrix(rotmax_a).as_quat()
    quat_b = Rotation.from_matrix(rotmax_b).as_quat()
    quat_diff = quat_multiply(quat_a, quat_inverse(quat_b))
    angle_diff = 180 * np.linalg.norm(quat2axisangle(quat_diff)) / np.pi

    return translational_dist, angle_diff


def turn_homo_to_frames(matrix):
    # matrix: 4x4 homogenous matrix
    rvec = matrix[:3, :3]
    tvec = matrix[:3, 3]

    return rvec, tvec


def turn_frames_to_homo(rvec, tvec):
    homo_mat = np.zeros((4, 4))
    homo_mat[:3, :3] = rvec
    homo_mat[:3, 3] = tvec
    homo_mat[3, 3] = 1
    return homo_mat


def flatten_homo_position(position):
    flattened_action = []
    for homo_action in position:
        _, action_tvec = turn_homo_to_frames(matrix=homo_action)
        flattened_action.append(torch.FloatTensor(action_tvec))

    flattened_action = torch.concat(flattened_action, axis=0)
    # return flattened_action[: self.action_dim]
    return flattened_action


def homo_flattened_position(position):  # position: (12,) or (24,) dimensional tvecs
    homo_position = []
    for idx in range(0, len(position), 3):
        curr_tvec = position[idx : idx + 3]

        curr_homo = turn_frames_to_homo(rvec=np.identity(3), tvec=curr_tvec)
        homo_position.append(curr_homo)

    homo_position = np.stack(homo_position, axis=0)

    return homo_position


def apply_residuals(demo_residuals, action, inverse=False):
    if demo_residuals is not None:
        for j in range(len(demo_residuals)):
            if inverse:
                action[:, j, 3] -= demo_residuals[j]
            else:
                action[:, j, 3] += demo_residuals[j]
    return action


# Taken from https://github.com/UT-Austin-RPL/deoxys_control
def quat_multiply(quaternion1, quaternion0):
    """
    Return multiplication of two quaternions (q1 * q0).

    E.g.:
    >>> q = quat_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> np.allclose(q, [-44, -14, 48, 28])
    True

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) multiplied quaternion
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array(
        (
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        ),
        dtype=np.float32,
    )


def quat2axisangle(quat):
    """
    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    # print('den: {}, quat[:3]: {}, math.acos(quat[3]): {}'.format(
    #     den, quat[:3], math.acos(quat[3])
    # ))
    # print(f'2.0 * math.acos(quat[3]): {2.0 * math.acos(quat[3])}')
    # print(f'quat[:3] * 2.0 * math.acos(quat[3]): {quat[:3] * 2.0 * math.acos(quat[3])}')

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def quat_inverse(quaternion):
    """
    Return inverse of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_inverse(q0)
    >>> np.allclose(quat_multiply(q0, q1), [0, 0, 0, 1])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion inverse
    """
    return quat_conjugate(quaternion) / np.dot(quaternion, quaternion)


def quat_distance(quaternion1, quaternion0):
    """
    Returns distance between two quaternions, such that distance * quaternion0 = quaternion1

    Args:
        quaternion1 (np.array): (x,y,z,w) quaternion
        quaternion0 (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    return quat_multiply(quaternion1, quat_inverse(quaternion0))


def quat_conjugate(quaternion):
    """
    Return conjugate of quaternion.

    E.g.:
    >>> q0 = random_quaternion()
    >>> q1 = quat_conjugate(q0)
    >>> q1[3] == q0[3] and all(q1[:3] == -q0[:3])
    True

    Args:
        quaternion (np.array): (x,y,z,w) quaternion

    Returns:
        np.array: (x,y,z,w) quaternion conjugate
    """
    return np.array(
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
        dtype=np.float32,
    )
