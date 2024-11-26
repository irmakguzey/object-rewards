import numpy as np
from scipy.spatial.transform import Rotation

from object_rewards.utils.constants import EEF_TO_END


def get_initial_kinova_position(
    wrist_to_base, fingertips_to_base, wrist_extend_length=0.05, wrist_raise=0.05
):

    # we extend on the vector from middle_3 to wrist
    middle_3_position = fingertips_to_base[1][:3, 3]
    wrist_position = wrist_to_base[0][:3, 3]
    # For more natural composition, lift wrist a little bit?
    wrist_position[2] = wrist_position[2] + 0.02
    v_middle_to_wrist = wrist_position - middle_3_position
    v_unit = v_middle_to_wrist / np.linalg.norm(v_middle_to_wrist)
    extended_wrist_position = wrist_position + wrist_extend_length * (v_unit)

    # Then get x-axis and y-axis
    extended_wrist_z = -v_unit
    approx_x = fingertips_to_base[1][:3, 3] - fingertips_to_base[0][:3, 3]
    approx_x = approx_x / np.linalg.norm(approx_x)
    projection = np.dot(approx_x, v_unit) * v_unit
    extended_wrist_x = approx_x - projection
    extended_wrist_x = extended_wrist_x / np.linalg.norm(extended_wrist_x)
    extended_wrist_y = np.cross(extended_wrist_z, extended_wrist_x)

    extended_wrist_orientation = np.hstack(
        (
            extended_wrist_x.reshape(3, 1),
            extended_wrist_y.reshape(3, 1),
            extended_wrist_z.reshape(3, 1),
        )
    )
    rotation_wrist_to_base = np.hstack(
        (extended_wrist_orientation, extended_wrist_position.reshape(3, 1))
    )
    rotation_wrist_to_base = np.vstack((rotation_wrist_to_base, np.array([0, 0, 0, 1])))

    return wrist_rotation_to_kinova_position(rotation_wrist_to_base, wrist_raise)


def wrist_rotation_to_kinova_position(rotation_wrist_to_base, wrist_raise=0.0):
    rotation_eef_to_base = np.dot(rotation_wrist_to_base, EEF_TO_END)
    eef_quat = Rotation.from_matrix(rotation_eef_to_base[:3, :3]).as_quat()
    eef_tvec = rotation_eef_to_base[:3, 3].reshape(1, 3)[0] + [
        0,
        0,
        wrist_raise,
    ]  # Raise the wrist a little bit higher
    return np.concatenate((eef_tvec, eef_quat), axis=0)
