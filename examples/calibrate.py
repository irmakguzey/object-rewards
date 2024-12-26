from object_rewards.calibration import CalibrateBase

HOST = "172.24.71.240"
CALIBRATION_PICS_DIR = "<calibration-dic>"
CAM_IDX = 0
MARKER_SIZE = 0.05

base_calibr = CalibrateBase(
    host=HOST,
    calibration_pics_dir=CALIBRATION_PICS_DIR,
    cam_idx=CAM_IDX,
)

base_calibr.save_poses()
base_to_camera = base_calibr.calibrate(True, True)
base_calibr.get_calibration_error_in_2d(base_to_camera=base_to_camera)
