import numpy as np


def flat16_to_pos_rot6d(flat):
    """Split a column-major flat 4x4 pose into position and 6D rotation."""
    pos = flat[..., 12:15]
    rot6d = np.concatenate([flat[..., 0:3], flat[..., 4:7]], axis=-1)
    return pos, rot6d


def rot6d_to_matrix(rot6d):
    """Convert 6D rotation to a 3x3 rotation matrix via Gram-Schmidt."""
    a1, a2 = rot6d[..., 0:3], rot6d[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = a2 / np.linalg.norm(a2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def pos_rot6d_to_flat16(pos, rot6d):
    """Rebuild a column-major flat 4x4 pose from position and 6D rotation."""
    R = rot6d_to_matrix(rot6d)
    flat = np.zeros(pos.shape[:-1] + (16,), dtype=pos.dtype)
    flat[..., 0:3], flat[..., 4:7], flat[..., 8:11] = R[..., :, 0], R[..., :, 1], R[..., :, 2]
    flat[..., 12:15] = pos
    flat[..., 15] = 1.0
    return flat


def compute_stats(x):
    """Per-dim mean and std of x[N, D], with std floored on constant dims."""
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def normalize(x, mean, std):
    """Zero-mean unit-variance normalization."""
    return (x - mean) / std


def denormalize(x, mean, std):
    """Inverse of normalize."""
    return x * std + mean
