import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors


def project_axes(rvec, tvec, intrinsic_matrix, scale=0.01, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    # img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(
        -1, 3
    )
    axis_points, _ = cv2.projectPoints(points, rvec, tvec, intrinsic_matrix, dist)
    return axis_points


def project_poses(poses, intrinsic_matrix, distortion=None, scale=0.01):
    # Project the axes for each pose
    projected_poses = []
    # print('poses.shape: {}'.format(poses.shape))
    for pose_id in range(len(poses)):
        pose = poses[pose_id]
        rvec, tvec = pose[:3, :3], pose[:3, 3]
        projected_pose = project_axes(
            rvec, tvec, intrinsic_matrix, dist=distortion, scale=scale
        )
        projected_poses.append(projected_pose)
    projected_poses = np.stack(projected_poses, axis=0)

    return projected_poses


def plot_axes(axes, img, color_set=1):
    for axis in axes:
        axis = axis.astype(int)
        if color_set == 1:
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[0].ravel()), (255, 0, 0), 3
            )
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[1].ravel()), (0, 255, 0), 3
            )
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[2].ravel()), (0, 0, 255), 3
            )

        elif color_set == 2:

            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[0].ravel()), (255, 165, 0), 3
            )  # Orange
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[1].ravel()), (128, 128, 0), 3
            )  # Green
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[2].ravel()), (138, 43, 226), 3
            )  # Purple

        elif color_set == 3:

            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[0].ravel()), (255, 153, 153), 3
            )  # Light Red
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[1].ravel()), (204, 255, 204), 3
            )  # Green
            img = cv2.line(
                img, tuple(axis[3].ravel()), tuple(axis[2].ravel()), (153, 255, 255), 3
            )  # Light Blue

    return img


# Will draw points to the given set of points
def plot_points(points, img, color=(255, 0, 0), radius=5):
    for point in points:
        point = point.astype(int)
        img = cv2.circle(
            img, point[3], radius=radius, color=color, thickness=-1
        )  # Here the input is expected to be the output of project_poses - which gives you the points of an axis where the center is the 3rd index

    return img


def plot_rotation_and_translation(
    image, translation, rotation, translation_pos=(1050, 400), ellipse_pos=(900, 400)
):
    # translation: (2,) translation at each axes
    # rotation: (1,) rotation in radians
    # Actual action
    trans_x = translation[0]  # multiplication is only for scaling
    trans_y = translation[1]
    image = cv2.arrowedLine(
        image,
        translation_pos,
        (
            int(translation_pos[0] + trans_x),
            int(translation_pos[1] + trans_y),
        ),  # Y should be removed from the action
        color=(159, 43, 104),
        thickness=3,
    )

    # Draw an ellipse to show the rotate_speed more thoroughly
    axesLength = (50, 50)
    angle = 0
    startAngle = 0
    endAngle = rotation * (180.0 / np.pi)
    image = cv2.ellipse(
        image,
        ellipse_pos,
        axesLength,
        angle,
        startAngle,
        endAngle,
        color=(159, 43, 104),
        thickness=3,
    )

    return image


def concat_imgs(
    img1, img2, orientation="horizontal"
):  # Or it could be vertical as well
    metric_id = 0 if orientation == "horizontal" else 1
    max_metric = max(img1.shape[metric_id], img2.shape[metric_id])
    min_metric = min(img1.shape[metric_id], img2.shape[metric_id])
    scale = min_metric / max_metric
    large_img_idx = np.argmax([img1.shape[metric_id], img2.shape[metric_id]])

    if large_img_idx == 0:
        img1 = cv2.resize(
            img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale))
        )
    else:
        img2 = cv2.resize(
            img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale))
        )

    concat_img = (
        cv2.hconcat([img1, img2])
        if orientation == "horizontal"
        else cv2.vconcat([img1, img2])
    )
    return concat_img


def turn_images_to_video(
    viz_dir, video_fps, video_name="visualization.mp4", is_video_path_relative=True
):
    if is_video_path_relative:
        video_path = os.path.join(viz_dir, video_name)
    else:
        video_path = video_name
    if os.path.exists(video_path):
        os.remove(video_path)
    os.system(
        "ffmpeg -r {} -i {}/%*.png -vf setsar=1:1 {}".format(
            video_fps, viz_dir, video_path  # fps
        )
    )


def turn_video_to_images(dir_path, video_name, images_dir_name, images_fps):
    images_path = os.path.join(dir_path, images_dir_name)
    video_path = os.path.join(dir_path, video_name)
    os.makedirs(images_path, exist_ok=True)
    os.system(f"ffmpeg -i {video_path} -vf fps={images_fps} {images_path}/out%d.png")


# Langsam related visualizations
def display_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for i, mask_np in enumerate(masks):
        axes[i + 1].imshow(mask_np, cmap="gray")
        axes[i + 1].set_title(f"Mask {i+1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig("image_with_masks.png")


def display_image_with_boxes(image, boxes, logits):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis("off")

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(
            logit.item(), 2
        )  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(
            x_min,
            y_min,
            f"Confidence: {confidence_score}",
            fontsize=8,
            color="red",
            verticalalignment="top",
        )

    # plt.show()
    plt.savefig("image_with_boxes.png")


# Method to add dino boxes to the given image with the given confidense scores
def vis_dino_boxes(ax, image, boxes, logits):
    ax.imshow(image)
    ax.set_title("Image with Bounding Boxes")
    ax.axis("off")

    for box, logit in zip(boxes, logits):
        x_min, y_min, x_max, y_max = box
        confidence_score = round(
            logit.item(), 2
        )  # Convert logit to a scalar before rounding
        box_width = x_max - x_min
        box_height = y_max - y_min

        # Draw bounding box
        rect = plt.Rectangle(
            (x_min, y_min),
            box_width,
            box_height,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
        ax.add_patch(rect)

        # Add confidence score as text
        ax.text(
            x_min,
            y_min,
            f"Confidence: {confidence_score}",
            fontsize=8,
            color="red",
            verticalalignment="top",
        )

    plt.tight_layout()
    return ax


# Method to add mask from the sam to the given axis
def vis_sam_mask(ax, mask):
    ax.imshow(mask, cmap="gray")
    ax.axis("off")
    plt.tight_layout()

    return ax


def visualize_offsets_on_img(
    img, offsets, residual_limit, q_values, img_path, plot_q_values_as_mtx=True
):
    # img: an image to plot the given offsets on
    # offsets: offset values that are applied - residual_limit is used to scale the plottings
    # q_values: range of q_values that we receive from various actions

    img = np.flip(img, (1, 2))
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Add the offset
    arrow_steps = img.shape[1] / (offsets.shape[0] + 1)
    arrow_centers_x = np.arange(0, img.shape[1], arrow_steps)
    arrow_centers_y = 430
    arrow_max_size = arrow_steps / 2 - 5

    for i, arrow_center_x in enumerate(arrow_centers_x[1:]):
        curr_offset = offsets[i]
        offset_pixel_size = int(arrow_max_size / residual_limit * curr_offset)

        arrow_center = (int(arrow_center_x), int(arrow_centers_y))

        img = cv2.arrowedLine(
            img.copy(),
            arrow_center,
            (int(arrow_center_x + offset_pixel_size), arrow_centers_y),
            color=(150, 255, 0),
            thickness=3,
        )

    cv2.imwrite("offsetted_img.png", img)

    # Plot the residuals in different plots
    if plot_q_values_as_mtx:
        fig, axs = plt.subplots(
            nrows=1, ncols=offsets.shape[0], figsize=(5 * offsets.shape[0], 5)
        )
        images = []
        for i, q_value in enumerate(q_values):
            q_value = np.expand_dims(q_value, 0)
            q_value = np.repeat(q_value, repeats=20, axis=0)
            images.append(axs[i].imshow(q_value))
            axs[i].label_outer()

        # Find the min and max of all colors for use in setting the color scale.
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        fig.colorbar(images[0], ax=axs, orientation="horizontal", fraction=0.1)

        # Make images respond to changes in the norm of other images (e.g. via the
        # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
        # recurse infinitely!
        def update(changed_image):
            for im in images:
                if (
                    changed_image.get_cmap() != im.get_cmap()
                    or changed_image.get_clim() != im.get_clim()
                ):
                    im.set_cmap(changed_image.get_cmap())
                    im.set_clim(changed_image.get_clim())

        for im in images:
            im.callbacks.connect("changed", update)

    else:
        # Plot q values as just plots
        nrows = 2
        ncols = int(offsets.shape[0] / nrows)
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )
        q_value_range = np.linspace(-residual_limit, residual_limit, q_values.shape[1])
        for i, q_value in enumerate(q_values):
            row_id = int(i / ncols)
            col_id = int(i % ncols)
            axs[row_id, col_id].plot(q_value_range, q_value)
            axs[row_id, col_id].set_ylabel("Q Values")
            axs[row_id, col_id].set_xlabel("Offset Values")
            axs[row_id, col_id].set_title(f"Q Values for Offset: {i}")

    # plt.close()
    plt.savefig("qvalues_plot.png", bbox_inches="tight")

    # Now load the images and concat them vertically
    offset_img = cv2.imread("offsetted_img.png")
    qvalues_img = cv2.imread("qvalues_plot.png")
    if plot_q_values_as_mtx:
        final_img = concat_imgs(
            img1=offset_img, img2=qvalues_img, orientation="vertical"
        )
    else:
        final_img = concat_imgs(
            img1=offset_img, img2=qvalues_img, orientation="horizontal"
        )

    cv2.imwrite(img_path, final_img)
