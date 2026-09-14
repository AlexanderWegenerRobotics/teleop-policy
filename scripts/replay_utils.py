# Shared offline open-loop replay logic for verify_stage2.py and eval_open_loop.py:
# load a checkpoint, run it frame-by-frame over an episode's logged
# observations, temporally-ensemble the overlapping predicted chunks.

import os

import h5py
import numpy as np
import torch

from dataset.transforms import denormalize, flat16_to_pos_rot6d, normalize
from models.act.act import ACT

M_ENSEMBLE = 0.01  # ALOHA default: w_i ~ exp(-m*i), i = steps since prediction was made

# same layout as dataset/teleop_dataset.py: per arm, pos(3) + rot6d(6) + grip(1), world frame
POSE_KEYS = {"observations": ("O_T_EE_world", "gripper_width"), "actions": ("O_T_EE_cmd_world", "gripper_cmd")}


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    dcfg, mcfg = ckpt["dataset_config"], ckpt["model_config"]
    n_cameras = len(dcfg["cameras"]["use"])
    action_dim = dcfg["action"]["dims_per_arm"] * len(dcfg["action"]["arms"])
    model = ACT(n_cameras=n_cameras, proprio_dim=action_dim, action_dim=action_dim,
                chunk_size=dcfg["action"]["chunk_size"], **mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dcfg


def load_stats(dcfg):
    npz = np.load(dcfg["normalize"]["stats_file"])
    return {k: npz[k] for k in npz.files}


def load_image(f, cam, t, hw):
    img = f[f"observations/images/{cam}"][t]
    if img.shape[:2] != tuple(hw):
        import cv2
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def pose_at(f, group, arms, t):
    pose_key, grip_key = POSE_KEYS[group]
    parts = []
    for arm in arms:
        pos, rot6d = flat16_to_pos_rot6d(f[f"{group}/{arm}/{pose_key}"][t])
        grip = f[f"{group}/{arm}/{grip_key}"][t]
        parts.append(np.concatenate([pos, rot6d, [grip]]))
    return np.concatenate(parts).astype(np.float32)


def episode_path(dcfg, episode_id):
    return os.path.join(dcfg["data"]["store_root"], str(episode_id).zfill(3), dcfg["data"]["episode_file"])


# runs the model over every `stride`-th ENGAGED frame of an open episode file,
# temporally-ensembles overlapping chunk predictions -> (t0, t1, logged_cmd, ensembled)
def open_loop_replay(f, model, dcfg, stats, device, stride=1):
    arms = dcfg["action"]["arms"]
    cams, hw = dcfg["cameras"]["use"], dcfg["cameras"]["resize_to"]
    K = dcfg["action"]["chunk_size"]
    action_dim = dcfg["action"]["dims_per_arm"] * len(arms)

    T = f["observations/timestamp_ns"].shape[0]
    engaged = np.ones(T, dtype=bool)
    for arm in arms:
        engaged &= f[f"observations/{arm}/state"][:] == 4
    t0, t1 = int(np.where(engaged)[0].min()), int(np.where(engaged)[0].max())

    logged_cmd = np.stack([pose_at(f, "actions", arms, t) for t in range(t0, t1)])

    pending = {t: [] for t in range(t0, t1 + K)}  # absolute frame -> chunk predictions, oldest first
    with torch.no_grad():
        for t in range(t0, t1, stride):
            images = np.stack([load_image(f, cam, t, hw) for cam in cams])
            images = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            proprio = pose_at(f, "observations", arms, t)
            proprio_n = normalize(proprio, stats["proprio_mean"], stats["proprio_std"])
            proprio_t = torch.from_numpy(proprio_n).unsqueeze(0).float().to(device)

            a_hat, _, _ = model(images, proprio_t)  # z=0 at inference (actions=None)
            a_hat = denormalize(a_hat[0].cpu().numpy(), stats["action_mean"], stats["action_std"])
            for i in range(K):
                if t + i in pending:
                    pending[t + i].append(a_hat[i])

    ensembled = np.zeros((t1 - t0, action_dim), dtype=np.float32)
    for t in range(t0, t1):
        preds = pending[t]
        if not preds:
            ensembled[t - t0] = ensembled[t - t0 - 1] if t > t0 else 0.0
            continue
        preds = np.stack(preds)
        ages = np.arange(len(preds))[::-1]  # 0 = most recent prediction
        w = np.exp(-M_ENSEMBLE * ages)
        w /= w.sum()
        ensembled[t - t0] = (w[:, None] * preds).sum(axis=0)

    return t0, t1, logged_cmd, ensembled
