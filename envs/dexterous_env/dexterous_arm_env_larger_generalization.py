import time
import numpy as np

from object_rewards.utils.arm_initialization import get_wrist_object_pose_diff_live
from .dexterous_arm_env_tracking import DexterityEnvTracking


class DexterityEnvLargerGeneralization(DexterityEnvTracking):
    def __init__(self, calibration_dir, policy_residuals, demo_obj_residual, **kwargs):
        super().__init__(**kwargs)
        self.__dict__.update(**kwargs)

        # Demo obj - arm pos difference - calculated separately
        self.demo_obj_robot_pos_diff = np.array(demo_obj_residual)
        self.calibration_dir = calibration_dir
        self.policy_residuals = policy_residuals

    def _reset_state(self):

        # Get the object position and reset to there here
        if self.manual_reset:
            input("Press Enter to continue.. after resetting the objects positions")

        obj_in_base = get_wrist_object_pose_diff_live(
            host=self.host_address,
            camera_port=10005,
            camera_id=0,
            calibration_pics_dir=self.calibration_dir,
            object_prompt=self.text_prompt,
        )
        desired_arm_position = obj_in_base[:3, 3] - self.demo_obj_robot_pos_diff
        resetting_action = dict(
            kinova=np.concatenate(
                [desired_arm_position, self.home_state["kinova"][3:]]
            ),
            allegro=self.home_state["allegro"],
        )
        self.episode_arm_offset = (
            desired_arm_position - self.home_state["kinova"][:3] + self.policy_residuals
        )  # This will ignore the policy residuals if not

        self.deploy_api.send_robot_action(resetting_action)
        self.step_count = 0
        time.sleep(3)

    def step(self, action):
        # Assumption: action - 4,4,4 fingertip homogenous matrices

        # Now add the arm offset to all the fingers
        for finger_homo in action:
            finger_homo[:3, 3] += self.episode_arm_offset

        current_joint_positions = self._get_curr_joint_positions()

        solver_action, _ = self.ik_solver.inverse_kinematics(
            desired_poses=action[:4],
            current_joint_positions=current_joint_positions,
            desired_orientation_poses=None,
        )
        robot_action_dict = dict(
            allegro=solver_action[6:],
            kinova=self._turn_arm_joint_action_to_cartesian(
                arm_hand_action=solver_action
            ),
        )

        self.deploy_api.send_robot_action(robot_action_dict)

        # Get the observations
        obs = self._get_obs()
        reward = self.get_reward()  # This will be rewritten by children classes

        infos = {"is_success": False}
        self.step_count += 1
        if self.max_episode_steps is not None:
            done = truncated = self.step_count >= self.max_episode_steps
        else:
            done = truncated = False

        return obs, reward, done, truncated, infos
