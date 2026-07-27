# EE pose <-> 6D rotation conversion, and per-dim normalization.
#
# O_T_EE / O_T_EE_cmd are libfranka 4x4 homogeneous transforms, flattened
# column-major: cols are [rot_x(0:3), rot_y(4:7), rot_z(8:11), trans(12:15)].
# The 6D rep (Zhou et al., 2019) is just the first two rotation columns —
# no need to ever materialize the full 3x3 for our purposes.

import numpy as np


def flat16_to_pos_rot6d(flat):
    """flat[...,16] -> (pos[...,3], rot6d[...,6])."""
    pos = flat[..., 12:15]
    rot6d = np.concatenate([flat[..., 0:3], flat[..., 4:7]], axis=-1)
    return pos, rot6d


def rot6d_to_matrix(rot6d):
    """rot6d[...,6] -> rotation matrix [...,3,3] via Gram-Schmidt (inference-time only)."""
    a1, a2 = rot6d[..., 0:3], rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def pos_rot6d_to_flat16(pos, rot6d):
    """Inverse of flat16_to_pos_rot6d, for streaming a prediction back to the controller."""
    R = rot6d_to_matrix(rot6d)
    flat = np.zeros(pos.shape[:-1] + (16,), dtype=pos.dtype)
    flat[..., 0:3], flat[..., 4:7], flat[..., 8:11] = R[..., :, 0], R[..., :, 1], R[..., :, 2]
    flat[..., 12:15] = pos
    flat[..., 15] = 1.0
    return flat


def compute_stats(x):
    """x[N,D] -> (mean[D], std[D]), std floored to avoid div-by-zero on constant dims."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize(x, mean, std):
    return (x - mean) / std


def denormalize(x, mean, std):
    return x * std + mean
