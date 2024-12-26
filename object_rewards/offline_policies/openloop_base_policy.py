import os

from object_rewards.calibration import CalibrateBase, CalibrateFingertips
from object_rewards.utils import get_demo_action_ids, get_initial_kinova_position


class OpenloopHuman2Robot:

    def __init__(
        self,
        data_path,
        demo_num,
        host,
        camera_port,
        camera_id,
        view_num,
        calibration_pics_dir,
    ):
        self.data_path = data_path
        self.demo_path = os.path.join(data_path, f"demonstration_{demo_num}")

        self.demo_action_ids = get_demo_action_ids(
            data_path=data_path, view_num=view_num, demo_num=demo_num
        )

        # Load the calibration classes
        self.calibrate_fingertips = CalibrateFingertips(
            demo_path=self.demo_path,
            realsense_view_num=camera_id,
            host=host,
            camera_port=camera_port,
        )
        self.calibrate_base = CalibrateBase(
            host=host,
            calibration_pics_dir=calibration_pics_dir,
            cam_idx=camera_id,
        )
        self.H_B_C = self.calibrate_base.get_base_to_camera()

    def get_keypoint_to_base(self, frame_id, keypoint_type="fingertips"):
        # Returns fingertips to base positions for orientation and fingertip poses
        if keypoint_type == "fingertips":
            H_F_B = self.calibrate_fingertips.get_fingertips_to_base(
                frame_id=frame_id, base_to_camera=self.H_B_C
            )
        if keypoint_type == "wrist":
            H_F_B = self.calibrate_fingertips.get_wrist_to_base(
                frame_id=frame_id, base_to_camera=self.H_B_C
            )
        if keypoint_type == "finger_roots":
            H_F_B = self.calibrate_fingertips.get_finger_roots_to_base(
                frame_id=frame_id, base_to_camera=self.H_B_C
            )

        return H_F_B

    def initialize_robot_position(self):
        frame_id = self.demo_action_ids[0]
        wrist_to_base = self.get_keypoint_to_base(
            frame_id=frame_id, keypoint_type="wrist"
        )
        fingertips_to_base = self.get_keypoint_to_base(
            frame_id=frame_id, keypoint_type="finger_roots"
        )
        position = get_initial_kinova_position(
            wrist_to_base,
            fingertips_to_base,
            wrist_extend_length=0.05,
            wrist_raise=0.05,
        )

        return position

    def act(self, obs, episode_step, **kwargs):
        is_done = False

        demo_action_id_range = range(self.demo_action_ids[0], self.demo_action_ids[1])
        if episode_step >= len(demo_action_id_range) - 1:
            is_done = True

        frame_id = demo_action_id_range[episode_step]
        action = self.get_keypoint_to_base(
            frame_id=frame_id, keypoint_type="fingertips"
        )

        return action, is_done


class OpenloopH2RRes(OpenloopHuman2Robot):
    def __init__(self, demo_residuals, **kwargs):

        super().__init__(**kwargs)

        self.__dict__.update(**kwargs)

        self.demo_residuals = demo_residuals

    def get_keypoint_to_base(self, frame_id, keypoint_type="fingertips"):
        # Returns fingertips to base positions for orientation and fingertip poses
        H_F_B = super().get_keypoint_to_base(frame_id, keypoint_type)
        # Add the residuals for fingertips
        for i in range(len(self.demo_residuals)):
            H_F_B[:, i, 3] += self.demo_residuals[i]

        return H_F_B
