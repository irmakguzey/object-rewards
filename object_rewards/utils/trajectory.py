from copy import deepcopy as copy

import numpy as np


def get_rot_and_trans(tracks, delta=False):
    # KL divergence with these might not work well since there are negatives and etc
    # tracks: predicted points (T, NUM_POINTS)
    translations, rotations = [], []
    total_translation, total_rotation = (
        np.zeros(2, dtype=np.float32),
        0.0,
    )

    # prev_points = tracks[0, :]
    for i in range(len(tracks)):
        if i == 0:
            prev_points = tracks[0, :]
        else:
            prev_points = tracks[i - 1, :]
        curr_points = tracks[i, :]

        # Translation
        # Calculate the translation difference
        curr_mean = np.mean(curr_points, axis=0)
        prev_mean = np.mean(prev_points, axis=0)
        diff_mean = curr_mean - prev_mean

        if (diff_mean == 0.0).all():
            diff_mean = np.ones(2) * 1.0e-12

        # Add it to the total translation
        total_translation[0] += diff_mean[0]
        total_translation[1] += diff_mean[1]

        # Rotation
        # Bring all the points to the same space
        curr_feat_norm = curr_points - curr_mean
        prev_feat_norm = prev_points - prev_mean

        # Calculate the rotation
        n = np.cross(prev_feat_norm, curr_feat_norm)
        if (n == 0.0).all():
            average_rot = 1e-12
            # print("average_rot = 0")
        else:
            rot = n / np.linalg.norm(n)
            average_rot = np.mean(rot)

        total_rotation += average_rot  # NOTE: If the KL divergence doesn't work well

        if delta:
            translations.append(diff_mean)
            rotations.append(average_rot)
        else:
            translations.append(total_translation.copy())
            rotations.append(copy(total_rotation))

    translations = np.stack(translations, axis=0)
    rotations = np.expand_dims(np.array(rotations), axis=1)

    trajectory = np.concatenate([translations, rotations], axis=-1)

    return trajectory
