import numpy as np
import open3d as o3d
from einops import rearrange

from object_rewards.utils.constants import *


def get_point_cloud(
    color_img=None, depth_img=None, filter_outliners=False, x_bounds=None, y_bounds=None
):

    intrinsic_matrix = REALSENSE_INTRINSICS
    cam_pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width=1280, height=720, intrinsic_matrix=intrinsic_matrix
    )

    fx, fy = cam_pinhole_intrinsics.get_focal_length()
    cx, cy = cam_pinhole_intrinsics.get_principal_point()

    depth_scale = 1000.0
    xmap, ymap = np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depth_img / depth_scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # Crop the points with x,y is smaller and larger than some set values

    points_depth = np.stack((points_x, -points_y, -points_z), axis=2)
    points_depth = rearrange(points_depth, "h w d -> (h w) d")

    if not color_img is None:
        color_img = color_img / 255.0
        points_color = color_img
        points_color = rearrange(points_color, "h w d -> (h w) d")

    # Crop object
    if not x_bounds is None:
        cropped_idx = []
        x_min_idx = points_depth[:, 0] < x_bounds[1]
        x_max_idx = points_depth[:, 0] > x_bounds[0]
        y_min_idx = points_depth[:, 1] < y_bounds[1]
        y_max_idx = points_depth[:, 1] > y_bounds[0]
        for id in range(points_depth.shape[0]):
            if x_min_idx[id] and x_max_idx[id] and y_min_idx[id] and y_max_idx[id]:
                cropped_idx.append(id)
    else:
        cropped_idx = range(points_depth.shape[0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_depth[cropped_idx, :])
    if not color_img is None:
        pcd.colors = o3d.utility.Vector3dVector(points_color[cropped_idx, :])

    if filter_outliners:
        pcd = remove_outliers(pcd)

    return pcd


def remove_outliers(pcd):
    cl, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.5)
    return cl
