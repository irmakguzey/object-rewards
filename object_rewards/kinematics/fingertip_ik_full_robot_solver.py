import os
from copy import deepcopy as copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from klampt import IKSolver, WorldModel
from klampt.math import se3
from klampt.model import ik
from klampt.model.subrobot import SubRobotModel
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class FingertipChain:
    def __init__(
        self,
        robot_model,
        base_link_name,
        tip_link_name,
    ):
        self.full_robot = robot_model
        self.base_link = self.full_robot.link(base_link_name)
        self.tip_link_name = tip_link_name

        self.robot = SubRobotModel(
            robot=self.full_robot,
            links=self.get_subrobot_joint_indices(
                robot=robot_model,
                base_link_name=base_link_name,
                tip_link_name=tip_link_name,
            ),
        )
        self.tip_link = self.robot.link(tip_link_name)
        self.num_links = len(self.robot.getConfig())

    def get_current_position(self):
        _, current_tvec = self.tip_link.getTransform()
        return current_tvec

    def get_current_pose(self):
        from third_person_man.utils import turn_frames_to_homo

        current_rvec, current_tvec = self.tip_link.getTransform()
        current_rvec = np.asarray(current_rvec).reshape(3, 3).T
        current_pose = turn_frames_to_homo(rvec=current_rvec, tvec=current_tvec)

        return current_pose

    def get_current_orientation(self):
        # NOTE: getTransform returns rotation column major wise
        current_rvec, _ = self.tip_link.getTransform()
        return np.asarray(current_rvec).reshape(3, 3).T

    def get_current_state(self, compute_type="position"):
        if compute_type == "position":
            return self.get_current_position()
        if compute_type == "orientation":
            return self.get_current_orientation()
        if compute_type == "all":
            return self.get_current_pose()

    def get_jacobian(self, compute_type="position"):
        # Get the whole jacobian of the tip link\
        # tip_jacobian: 6xn (n: number of DOFs) jacobian matrix
        if compute_type == "position":
            return self.get_position_jacobian()
        if compute_type == "orientation":
            return self.get_orientation_jacobian()
        if compute_type == "all":
            return self.get_pose_jacobian()

    def get_pose_jacobian(self):  # Returns a 6 dimensional jacobian
        return self.tip_link.getJacobian([0, 0, 0])  # plocal is 0 wrt the link

    def get_position_jacobian(self):  # 3 dimenisional jacobian
        # tip_jacobian: 3xn (n: number of DOFs) position jacobian matrix
        return self.tip_link.getPositionJacobian([0, 0, 0])

    def get_orientation_jacobian(self):  # 3 dimensional jacobian
        # tip_jacobian: 3xn (n: number of DOFs) orientation jacobian matrix
        return self.tip_link.getOrientationJacobian()

    # Returns error in position vectors and axis angles
    def get_current_error(self, desired, error_type="position"):
        if error_type == "position":
            current_position = self.get_current_position()
            transform = desired - current_position
            return transform

        if error_type == "orientation":
            current_orientation = self.get_current_orientation()  # 3,3 rotation matrix

            # Turn everything to quaternion
            desired_quat = Rotation.from_matrix(desired).as_quat()
            current_quat = Rotation.from_matrix(current_orientation).as_quat()

            # New error:
            if np.dot(desired_quat, desired_quat) < 0.0:
                current_quat = -current_quat
            from third_person_man.utils import quat2axisangle, quat_distance

            error_quat = quat_distance(desired_quat, current_quat)
            error_axis_angle = quat2axisangle(error_quat)

            return error_axis_angle

        if error_type == "all":
            current_pose = self.get_current_pose()
            transform = np.linalg.pinv(current_pose) @ desired

            from third_person_man.utils import quat2axisangle, turn_homo_to_frames

            rvec, tvec = turn_homo_to_frames(transform)
            # rvec_axis_angle = quat2axisangle(Rotation.from_matrix(rvec).as_quat())
            rvec_axis_angle = Rotation.from_matrix(rvec).as_euler("zyx")
            error = np.concatenate(
                [rvec_axis_angle, tvec], axis=0
            )  # Jacobian returns the orientation first and then position
            return error

    def get_fingertip_pose(self, joint_positions):
        # old_config = self.robot.getConfig()

        cur_config = self.robot.getConfig()
        cur_config[6:] = joint_positions[:]
        self.robot.setConfig(cur_config)

        ft_rvec, ft_tvec = self.tip_link.getTransform()

        from third_person_man.utils import turn_frames_to_homo

        fingertip_pose = turn_frames_to_homo(
            rvec=np.asarray(ft_rvec).reshape(3, 3).T, tvec=np.asarray(ft_tvec)
        )

        return fingertip_pose

    def get_endeff_pose(self, joint_positions):

        old_config = self.robot.getConfig()

        self.robot.setConfig(joint_positions)

        endeff_link = "allegro_mount"
        endeff_rvec, endeff_tvec = self.robot.link(endeff_link).getTransform()

        from third_person_man.utils import turn_frames_to_homo

        endeff_pose = turn_frames_to_homo(
            rvec=np.asarray(endeff_rvec).reshape(3, 3).T, tvec=np.asarray(endeff_tvec)
        )

        self.robot.setConfig(old_config)

        return endeff_pose

    def get_arm_link_pose(self, joint_positions, link_name):

        old_config = self.robot.getConfig()

        self.robot.setConfig(joint_positions)

        link_rvec, link_tvec = self.robot.link(link_name).getTransform()

        from third_person_man.utils import turn_frames_to_homo

        endeff_pose = turn_frames_to_homo(
            rvec=np.asarray(link_rvec).reshape(3, 3).T, tvec=np.asarray(link_tvec)
        )

        self.robot.setConfig(old_config)

        return endeff_pose

    def set_finger_joint_positions(self, joint_positions):
        assert (
            joint_positions.shape[0] == 4
        ), "Finger joint positions should have 4 dimensions"
        robot_config = self.robot.getConfig()
        robot_config[6:] = joint_positions[:]
        self.robot.setConfig(robot_config)

    def set_arm_joint_positions(self, joint_positions):
        assert (
            joint_positions.shape[0] == 6
        ), "Arm joint positions should have 6 dimensions"
        robot_config = self.robot.getConfig()
        robot_config[:6] = joint_positions[:]
        self.robot.setConfig(robot_config)

    def get_arm_joint_positions(self):
        return np.asarray(self.robot.getConfig()[:6])

    def get_finger_joint_positions(self):
        return np.asarray(self.robot.getConfig()[6:])

    def set_joint_positions(self, joint_positions):
        self.robot.setConfig(joint_positions)

    def get_joint_positions(self):
        return np.asarray(self.robot.getConfig())

    def get_subrobot_joint_indices(self, robot, base_link_name, tip_link_name):
        joint_indices = []  # NOTE: Joint indices for the whole robot model
        parent_link = None
        tip_link = robot.link(tip_link_name)
        while True:
            if parent_link is None:
                current_link = tip_link
            else:
                current_link = parent_link

            if (
                current_link.getName() == base_link_name
            ):  # We don't want to go further than the base_link
                break

            # Get the id if the joint is not fixed
            joint_type = robot.getJointType(current_link.getName())
            # print('joint_type: {}'.format(joint_type))
            if joint_type != "weld":
                joint_id = current_link.getIndex()
                joint_indices.append(joint_id)

            # Get the parent id
            parent_id = current_link.getParent()
            if parent_id == -1:
                break
            parent_link = robot.link(parent_id)

        # Revert the joint indices
        joint_indices.reverse()

        return joint_indices

    # Have the single jacobian step
    def single_jacobian_inverse_step(self, desired, compute_type="position"):
        # compute_type: (position, orientation, all)

        # Get the current error
        x_e = self.get_current_error(desired=desired, error_type=compute_type)

        # Get the current jacobian and the necessary change
        current_jacobian = self.get_jacobian(compute_type=compute_type)
        joint_pos_change = np.linalg.pinv(current_jacobian) @ x_e

        return joint_pos_change

    def print(self):
        # print(f'Chain: {self.tip_link_name}')
        for link_id in range(len(self.robot.getConfig())):
            space = " " * (link_id + 1)
            print(f"{space}Link: {self.robot.link(link_id).getName()}")
        print("****")


class FingertipIKFullRobotSolver:
    def __init__(
        self,
        urdf_path,
        desired_finger_types,
        compute_type="position",
    ):

        # Have the compute type eithr all or position only
        # for this version of the implememtation we only accept position IK
        # but we have a separate link to predict if we use orientation
        assert (
            compute_type == "all" or compute_type == "position"
        ), "compute_type parameter passed incorrectly, it can either be position or all"

        # Initialize the world and the robot
        self.world = WorldModel()
        self.robot = self.world.loadRobot(fn=urdf_path)
        self.compute_type = compute_type
        urdf_fingertip_mappings = dict(
            index="allegro_link_d_tip",
            middle="allegro_link_h_tip",
            ring="allegro_link_m_tip",
            thumb="allegro_link_r_tip",
            index_orientation="allegro_link_d",
            middle_orientation="allegro_link_h",
            ring_orientation="allegro_link_m",
            thumb_orientation="allegro_link_r",
        )

        self.fingertip_link_mappings = {}
        for finger_type in desired_finger_types:
            self.fingertip_link_mappings[finger_type] = urdf_fingertip_mappings[
                finger_type
            ]

        # Create the chains
        self.chains = {}
        for finger_type in ["index", "middle", "ring", "thumb"]:
            if finger_type in self.fingertip_link_mappings.keys():
                self.chains[finger_type] = FingertipChain(
                    robot_model=self.robot,
                    base_link_name="kinova_link_base",
                    tip_link_name=self.fingertip_link_mappings[finger_type],
                )

                # If orientation is enabled we would like to have orientation links as the link
                # right before the tip as well
                if compute_type == "all":
                    finger_type_orientation = f"{finger_type}_orientation"
                    self.chains[finger_type_orientation] = FingertipChain(
                        robot_model=self.robot,
                        base_link_name="kinova_link_base",
                        tip_link_name=urdf_fingertip_mappings[
                            finger_type_orientation
                        ],  # We don't need the finger orientation links to be on the fingertip mappings
                    )

    # NOTE: Since there is an arm to be impacted now, fingers cannot go through inverse kinematics
    # separately, that's why inverse kinematics method is implemented above the chains
    def inverse_kinematics(
        self,
        desired_poses,
        current_joint_positions,
        threshold=1e-3,
        learning_rate=1e-2,
        max_iterations=50,
        finger_arm_weight=50,
        desired_orientation_poses=None,
    ):
        # desired_poses: (4,4,4) 4 different pose for each finger wrt the base (kinova_link_base)
        # desired_orientation_poses: if not None, it will be giving the desired poses of the orientation links
        # current_joint_positions: (22,): first 6 joints are for kinova, next 16 joints are for allegro

        # Set the current joint positions
        self.set_joint_positions(
            hand_joint_positions=current_joint_positions[6:],
            arm_joint_positions=current_joint_positions[:6],
        )

        delta_joint_positions = np.zeros(22)

        # Start the jacobian gradient descent
        for _ in range(max_iterations):
            for finger_id, finger_type in enumerate(
                ["index", "middle", "ring", "thumb"]
            ):
                if finger_type in self.fingertip_link_mappings.keys():

                    # Get the current joint positions
                    current_finger_joint_positions = self.chains[
                        finger_type
                    ].get_joint_positions()

                    # Calculate the type of the desired pose
                    from third_person_man.utils import turn_homo_to_frames

                    _, desired_tvec = turn_homo_to_frames(
                        matrix=desired_poses[finger_id]
                    )
                    desired = desired_tvec

                    # Get the joint change for a single jacobian gradient descent step
                    current_finger_joint_change = self.chains[
                        finger_type
                    ].single_jacobian_inverse_step(  # The first 6 joints are kinova's since the urdf is built that way
                        desired=desired_tvec,
                        compute_type="position",  # It should always be calculated as position
                    )

                    # Add the joint change to the current config
                    # Add more weight to the fingers than the arm so that it would prioritize the fingers
                    current_finger_joint_positions[:6] += (
                        learning_rate * current_finger_joint_change[:6]
                    )
                    current_finger_joint_positions[6:] += (
                        finger_arm_weight
                        * learning_rate
                        * current_finger_joint_change[6:]
                    )
                    # Add the joint change to the global delta_joint_positions
                    delta_joint_positions[:6] += (
                        learning_rate * current_finger_joint_change[:6]
                    )
                    delta_joint_positions[
                        6 + (4 * finger_id) : 6 + (4 * (finger_id + 1))
                    ] += (
                        finger_arm_weight
                        * learning_rate
                        * current_finger_joint_change[6:]
                    )

                    # Apply the joint changes to each of the finger's configs - separately for the fingers globally for the arm
                    # Now the finger joint positions also should be set globally for a single finger
                    self.set_global_finger_joint_positions(
                        joint_positions=current_finger_joint_positions[6:],
                        finger_type=finger_type,
                    )
                    self.set_global_arm_joint_positions(
                        joint_positions=current_finger_joint_positions[:6]
                    )

                    # We should redo the whole process for the  the orientation is included
                    if self.compute_type == "all":
                        finger_type_orientation = self._get_orientation_finger_type(
                            finger_type
                        )
                        current_finger_orientation_joints = self.chains[
                            finger_type_orientation
                        ].get_joint_positions()
                        _, or_desired_tvec = turn_homo_to_frames(
                            matrix=desired_orientation_poses[finger_id]
                        )
                        current_finger_joint_change = self.chains[
                            finger_type_orientation
                        ].single_jacobian_inverse_step(
                            desired=or_desired_tvec, compute_type="position"
                        )

                        assert np.isclose(
                            current_finger_orientation_joints,
                            current_finger_joint_positions,
                        ).all(), f"Current finger orietation: {current_finger_orientation_joints} and curret finger joiits: {current_finger_joint_positions} are not the same"
                        orientation_learning_rate = 1e-3
                        current_finger_orientation_joints[:6] += (
                            orientation_learning_rate * current_finger_joint_change[:6]
                        )
                        current_finger_orientation_joints[6:] += (
                            finger_arm_weight
                            * orientation_learning_rate
                            * current_finger_joint_change[6:]
                        )
                        # Add the joint change to the global delta_joint_positions
                        delta_joint_positions[:6] += (
                            orientation_learning_rate * current_finger_joint_change[:6]
                        )
                        delta_joint_positions[
                            6 + (4 * finger_id) : 6 + (4 * (finger_id + 1))
                        ] += (
                            finger_arm_weight
                            * orientation_learning_rate
                            * current_finger_joint_change[6:]
                        )
                        # Apply the joint changes to each of the finger's configs - separately for the fingers globally for the arm
                        # Now the finger joint positions also should be set globally for a single finger
                        self.set_global_finger_joint_positions(
                            joint_positions=current_finger_orientation_joints[6:],
                            finger_type=finger_type,
                        )
                        self.set_global_arm_joint_positions(
                            joint_positions=current_finger_orientation_joints[:6]
                        )

            # Calculate the current error for all fingertips and if all of them are below the threshold, break
            is_done = True
            for finger_id, finger_type in enumerate(
                ["index", "middle", "ring", "thumb"]
            ):
                if finger_type in self.fingertip_link_mappings.keys():
                    current_finger_error = self.chains[finger_type].get_current_error(
                        desired=desired,
                        error_type="position",  # The compute_type for the chain IK module will always be position because the orientation is set differently
                    )
                    if (np.abs(current_finger_error) > threshold).any():
                        is_done = False
                        break

            if is_done:
                return (
                    current_joint_positions + delta_joint_positions,
                    delta_joint_positions,
                )

        return current_joint_positions + delta_joint_positions, delta_joint_positions

    def forward_kinematics(self, current_joint_positions):
        # Set the joint positions and get the fingertip poses wrt the base frame
        # And get the current pose
        # current_joint_positions: (22,) joint positions - first 6 joint positions are for kinova joints, last 16 are for allegro
        self.set_joint_positions(
            hand_joint_positions=current_joint_positions[6:],
            arm_joint_positions=current_joint_positions[:6],
        )

        fingertip_poses = []
        if self.compute_type == "all":
            fingertip_orientation_poses = []
        for finger_type in ["index", "middle", "ring", "thumb"]:
            if finger_type in self.fingertip_link_mappings.keys():
                curr_fingertip_pose = self.chains[finger_type].get_current_pose()
                fingertip_poses.append(curr_fingertip_pose)

                if self.compute_type == "all":
                    finger_type_orientation = self._get_orientation_finger_type(
                        finger_type
                    )
                    curr_ft_orientation_pose = self.chains[
                        finger_type_orientation
                    ].get_current_pose()
                    fingertip_orientation_poses.append(curr_ft_orientation_pose)

        fingertip_poses = np.stack(fingertip_poses, axis=0)
        if self.compute_type == "all":
            fingertip_orientation_poses = np.stack(fingertip_orientation_poses, axis=0)
            return fingertip_poses, fingertip_orientation_poses

        return fingertip_poses

    def set_global_arm_joint_positions(self, joint_positions):
        # NOTE: This sets the arm joint positions for ALL finger chains
        # this is needed since all chains need to have the same arm values
        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            # Set the finger joints
            if finger_type in self.fingertip_link_mappings.keys():
                # Set the arm joint for all fingers
                self.chains[finger_type].set_arm_joint_positions(
                    joint_positions=joint_positions
                )

    def set_global_finger_joint_positions(self, joint_positions, finger_type):
        self.chains[finger_type].set_finger_joint_positions(
            joint_positions=joint_positions
        )
        if self.compute_type == "all":
            finger_type_orientation = self._get_orientation_finger_type(finger_type)
            self.chains[finger_type_orientation].set_finger_joint_positions(
                joint_positions=joint_positions
            )

    def _get_orientation_finger_type(self, finger_type):
        return f"{finger_type}_orientation"

    def set_joint_positions(self, hand_joint_positions, arm_joint_positions):

        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            # Set the finger joints
            if finger_type in self.fingertip_link_mappings.keys():
                self.set_global_finger_joint_positions(
                    joint_positions=hand_joint_positions[4 * i : 4 * (i + 1)],
                    finger_type=finger_type,
                )

        # Set the arm joint for all fingers
        self.set_global_arm_joint_positions(joint_positions=arm_joint_positions)

    def set_hand_joint_positions(self, joint_positions):
        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            if finger_type in self.fingertip_link_mappings.keys():
                finger_joint_pos = joint_positions[4 * i : 4 * (i + 1)]
                self.chains[finger_type].set_finger_joint_positions(
                    joint_positions=finger_joint_pos
                )

    def get_arm_joint_positions(self):
        j = 0
        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            if finger_type in self.fingertip_link_mappings.keys():
                if (
                    j == 0
                ):  # NOTE: This check is only for debugging is the arm positions are the same for different finger chains
                    first_arm_joints = self.chains[
                        finger_type
                    ].get_arm_joint_positions()
                    first_finger_type = finger_type
                else:
                    next_arm_joints = self.chains[finger_type].get_arm_joint_positions()
                    assert np.isclose(
                        first_arm_joints, next_arm_joints
                    ).all(), f"Arm joint positions are different for fingers {first_finger_type}({first_arm_joints}) and {finger_type}({next_arm_joints})"
                j += 1

        return first_arm_joints

    def get_hand_joint_positions(self):
        hand_joint_positions = np.zeros(16)  # If the finger is not included return 0s
        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            if finger_type in self.fingertip_link_mappings.keys():
                curr_finger_joint_pos = self.chains[
                    finger_type
                ].get_finger_joint_positions()
                hand_joint_positions[4 * i : 4 * (i + 1)] = curr_finger_joint_pos[:]

        return hand_joint_positions

    def get_endeff_pose(self, full_robot_joint_positions):
        j = 0
        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            if finger_type in self.fingertip_link_mappings.keys():

                finger_joints = [full_robot_joint_positions[:6], np.zeros(4)]
                finger_joints = np.concatenate(finger_joints, axis=0)

                if (
                    j == 0
                ):  # NOTE: This check is only for debugging is the arm positions are the same for different finger chains

                    first_endeff_pose = self.chains[finger_type].get_endeff_pose(
                        joint_positions=finger_joints
                    )
                    first_finger_type = finger_type
                else:
                    next_endeff_pose = self.chains[finger_type].get_endeff_pose(
                        joint_positions=finger_joints
                    )
                    assert np.isclose(
                        first_endeff_pose, next_endeff_pose
                    ).all(), f"Arm endeff poses are different for fingers {first_finger_type}({first_endeff_pose}) and {finger_type}({next_endeff_pose})"

                # self.chains[finger_type].print()

                j += 1

        return first_endeff_pose

    def get_arm_link_pose(self, joint_positions, link_name):
        j = 0
        for i, finger_type in enumerate(["index", "middle", "ring", "thumb"]):
            if finger_type in self.fingertip_link_mappings.keys():
                finger_joints = [
                    joint_positions[:6],
                    joint_positions[6 + 4 * i : 6 + 4 * (i + 1)],
                ]
                finger_joints = np.concatenate(finger_joints, axis=0)
                if (
                    j == 0
                ):  # NOTE: This check is only for debugging is the arm positions are the same for different finger chains
                    first_link_pose = self.chains[finger_type].get_arm_link_pose(
                        joint_positions=finger_joints, link_name=link_name
                    )
                    first_finger_type = finger_type
                else:
                    next_link_pose = self.chains[finger_type].get_arm_link_pose(
                        joint_positions=finger_joints, link_name=link_name
                    )
                    assert np.isclose(
                        first_link_pose, next_link_pose
                    ).all(), f"Arm {link_name} poses are different for fingers {first_finger_type}({first_link_pose}) and {finger_type}({next_link_pose})"

        return first_link_pose

    def get_ft_pose(self, finger_joint_positions, finger_type):

        finger_pose = self.chains[finger_type].get_fingertip_pose(
            joint_positions=finger_joint_positions
        )

        print("finger_pose: {}".format(finger_pose[:3, 3]))

        return finger_pose

    def print_chains(self):
        for key, values in self.chains.items():
            self.chains[key].print()
