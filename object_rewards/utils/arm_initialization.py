# Script to initialize the position of the arm to the top of an object that is detected
# Steps:
# 1- Get the object position using langsam
# 2- Find the depth of that point
# 3- Read the depth and see if it makes sense :P
# 4- Turn that into tvec and homo matrix and move it to the base frame
import os
import pickle
import time

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from openteach.constants import DEPTH_PORT_OFFSET
from openteach.robot.kinova import KinovaArm
from openteach.utils.network import ZMQCameraSubscriber
from PIL import Image as im

from object_rewards.point_tracking.lang_sam import LangSAM
from object_rewards.utils.constants import REALSENSE_INTRINSICS
from object_rewards.utils.visualization import vis_dino_boxes


def get_image(host_address, camera_port, camera_id, is_depth):
    if is_depth:
        image_subscriber = ZMQCameraSubscriber(
            host=host_address,
            port=camera_port + camera_id + DEPTH_PORT_OFFSET,
            topic_type="Depth",
        )
        image, _ = image_subscriber.recv_depth_image()
    else:
        image_subscriber = ZMQCameraSubscriber(
            host=host_address, port=camera_port + camera_id, topic_type="RGB"
        )
        image, _ = image_subscriber.recv_rgb_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_image(demo_path, frame_id, camera_id, is_depth):

    if is_depth:

        file_name = f"{demo_path}/cam_{camera_id}_depth.h5"
        with h5py.File(file_name, "r") as f:
            depth_imgs = f["depth_images"][()]

        img = depth_imgs[frame_id]

    else:
        dir_name = f"{demo_path}/cam_{camera_id}_rgb_images"
        image_path = os.path.join(
            demo_path,
            "{}/frame_{}.png".format(dir_name, str(frame_id).zfill(5)),
        )
        img = cv2.imread(image_path)  # This loads images as PIL image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)


def get_object_position(image, text_prompt):
    langsam = LangSAM()
    masks, boxes, phrases, logits, embeddings = langsam.predict(
        image_pil=im.fromarray(image), text_prompt=text_prompt
    )

    print(f"boxes: {boxes}")
    boxes = boxes.detach().cpu().numpy()

    # Find the box with the smallest size - NOTE: This is pretty hacky and might not work everywhere
    min_box_size, best_box_id = 1280 * 720, 0
    for box_id, box in enumerate(boxes):
        box_size = abs(box[0] - box[2]) * abs(box[1] - box[3])
        if box_size < min_box_size:
            best_box_id = box_id
            min_box_size = box_size

    mean_x, mean_y = (boxes[best_box_id, 0] + boxes[best_box_id, 2]) / 2, (
        boxes[best_box_id, 1] + boxes[best_box_id, 3]
    ) / 2

    img_pts = [mean_x.astype(int), mean_y.astype(int)]
    print(f"img_pts: {img_pts}, mean_x: {mean_x}, mean_y: {mean_y}")
    image = cv2.circle(image, img_pts, radius=5, color=(255, 0, 0), thickness=-1)

    _, ax = plt.subplots(1, 1)
    ax = vis_dino_boxes(ax, image, [boxes[best_box_id]], [logits[best_box_id]])
    plt.savefig("obj_pos_boxes.png", bbox_inches="tight")

    # cv2.imwrite("obj_pos.png", image)

    # np.save("obj_pts.npy", img_pts)

    return img_pts


def get_obj_tvec(
    img_pts, depth_img=None, host_address=None, camera_port=None, camera_id=None
):

    if depth_img is None:
        depth_img = get_image(
            host_address=host_address,
            camera_port=camera_port,
            camera_id=camera_id,
            is_depth=True,
        )

    x = img_pts[0]
    y = img_pts[1]
    z = depth_img[y, x]

    print(depth_img.shape, img_pts)

    intrinsic_matrix = REALSENSE_INTRINSICS
    cam_pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720, intrinsic_matrix=intrinsic_matrix
    )

    fx, fy = cam_pinhole_intrinsics.get_focal_length()
    cx, cy = cam_pinhole_intrinsics.get_principal_point()

    depth_scale = 1000.0

    z = z / depth_scale
    x = (x - cx) / fx * z
    y = (y - cy) / fy * z

    tvec = np.array([x, y, z])

    print(f"calculated tvec: {tvec} - shape: {tvec.shape}")

    return tvec


def get_obj_in_base(obj_tvec, camera_id, calibration_pics_dir):

    H_B_C = np.load(f"{calibration_pics_dir}/homo_base_to_cam_{camera_id}.npy")

    H_O_C = np.eye(4)
    H_O_C[:3, 3] = obj_tvec

    H_O_B = np.linalg.pinv(H_B_C) @ H_O_C

    print(f"H_O_B[:3, 3]: {H_O_B[:3, 3]}")

    return H_O_B


def get_wrist_object_pose_diff_in_demo(
    data_path,
    demo_num,
    host,
    camera_port,
    camera_id,
    calibration_pics_dir,
    object_prompt,
):
    from object_rewards.offline_policies.openloop_base_policy import OpenloopHuman2Robot

    openloop_base_policy = OpenloopHuman2Robot(
        data_path=data_path,
        demo_num=demo_num,
        host=host,
        camera_port=camera_port,
        camera_id=camera_id,
        marker_size=0.05,
        calibration_pics_dir=calibration_pics_dir,
        view_num=camera_id,
    )

    wrist_position = openloop_base_policy.initialize_robot_position(
        wrist_extend_length=0.05
    )

    demo_path = f"{data_path}/demonstration_{demo_num}"
    image_indices_path = os.path.join(demo_path, f"image_indices_cam_{camera_id}.pkl")
    with open(image_indices_path, "rb") as file:
        image_indices = pickle.load(file)

    _, image_id = image_indices[openloop_base_policy.demo_action_ids[0]]

    # Find the object position in the demo
    rgb_img = load_image(
        demo_path=demo_path,
        frame_id=image_id,
        camera_id=camera_id,
        is_depth=False,
    )

    depth_img = load_image(
        demo_path=demo_path,
        frame_id=image_id,
        camera_id=camera_id,
        is_depth=True,
    )

    obj_pts = get_object_position(image=rgb_img, text_prompt=object_prompt)
    obj_tvec_in_camera = get_obj_tvec(img_pts=obj_pts, depth_img=depth_img)
    obj_in_base = get_obj_in_base(
        obj_tvec=obj_tvec_in_camera,
        camera_id=camera_id,
        calibration_pics_dir=calibration_pics_dir,
    )

    print("** IN DEMO {demo_num} OBJ and WRIST")
    print(f"obj_tvec: {obj_in_base[:3, 3]}")
    print(f"wrist_tvec: {wrist_position[:3]}")

    return wrist_position, obj_in_base[:3, 3]


def get_wrist_object_pose_diff_live(
    host, camera_port, camera_id, calibration_pics_dir, object_prompt
):

    # Find the object position in the demo
    rgb_img = get_image(
        host_address=host,
        camera_port=camera_port,
        camera_id=camera_id,
        is_depth=False,
    )

    cv2.imwrite("curr_img.png", rgb_img)

    depth_img = get_image(
        host_address=host,
        camera_port=camera_port,
        camera_id=camera_id,
        is_depth=True,
    )

    # obj_pts = np.load("obj_pts.npy")
    obj_pts = get_object_position(image=rgb_img, text_prompt=object_prompt)
    obj_tvec_in_camera = get_obj_tvec(img_pts=obj_pts, depth_img=depth_img)
    obj_in_base = get_obj_in_base(
        obj_tvec=obj_tvec_in_camera,
        camera_id=camera_id,
        calibration_pics_dir=calibration_pics_dir,
    )

    arm = KinovaArm()
    time.sleep(1)
    arm_cart = arm.get_cartesian_position()

    print("** LIVE OBJ and WRIST **")
    print(f"obj_tvec: {obj_in_base[:3, 3]}")
    print(f"wrist_tvec: {arm_cart[:3]}")

    return obj_in_base


if __name__ == "__main__":
    # Test for moving the object around
    host_address = "172.24.71.240"
    camera_port = 10005
    camera_id = 0
    calibration_pics_dir = "/data/irmak/third_person_manipulation/base_calibration_1"
    object_prompt = "green music box"
    data_path = "/data/irmak/third_person_manipulation/music_box_opening_ground"
    demo_num = 15

    # # Get the arm in base and see how off they are
    arm = KinovaArm()
    time.sleep(1)

    demo_wrist_pose, obj_in_base_tvec = get_wrist_object_pose_diff_in_demo(
        data_path=data_path,
        demo_num=demo_num,
        host=host_address,
        camera_port=camera_port,
        camera_id=camera_id,
        calibration_pics_dir=calibration_pics_dir,
        object_prompt=object_prompt,
    )

    # Get the difference: and by following where the object is always, move the arm there
    demo_obj_robot_pos_diff = obj_in_base_tvec - demo_wrist_pose[:3]

    print(f"demo_obj_robot_pos_diff: {demo_obj_robot_pos_diff}")

    while True:

        input("Move the object and press enter")
        obj_in_base = get_wrist_object_pose_diff_live(
            host=host_address,
            camera_port=camera_port,
            camera_id=camera_id,
            calibration_pics_dir=calibration_pics_dir,
            object_prompt=object_prompt,
        )

        desired_arm_position = obj_in_base[:3, 3] - demo_obj_robot_pos_diff
        desired_arm_pose = np.concatenate([desired_arm_position, demo_wrist_pose[3:]])

        input(
            f"Detected obj_in_base: {obj_in_base[:3, 3]}, Moving the robot to: {desired_arm_pose} press enter if it makes sense"
        )

        arm.move_coords(desired_arm_pose)
