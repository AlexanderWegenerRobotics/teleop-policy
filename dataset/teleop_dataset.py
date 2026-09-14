# Torch Dataset over converted episode.hdf5 files: one random (episode, t)
# sample per __getitem__, ACT-style (chunk of K future commanded actions).

import os

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

try:
    from transforms import flat16_to_pos_rot6d, normalize          # run from dataset/ or with it on sys.path
except ImportError:
    from dataset.transforms import flat16_to_pos_rot6d, normalize  # run as `dataset.teleop_dataset` from repo root


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def read_split(path):
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


CLOSED_WIDTH_M = 0.04  # gripper_cmd is binary 0.0 / 0.08; midpoint splits it


class EpisodicDataset(Dataset):
    def __init__(self, episode_ids, cfg, stats=None):
        self.episode_ids = episode_ids
        self.store_root = cfg["data"]["store_root"]
        self.episode_file = cfg["data"]["episode_file"]
        self.cams = cfg["cameras"]["use"]
        self.img_hw = tuple(cfg["cameras"]["resize_to"])
        self.arms = cfg["action"]["arms"]
        self.K = cfg["action"]["chunk_size"]
        self.train_states = set(cfg["state"]["train_states"])
        self.stats = stats  # dict with 'proprio_mean/std', 'action_mean/std', or None
        # Grasp-window oversampling (see _pick_t / _grasp_window). 1.0 = the
        # original uniform sampler, so this is opt-in and the default retrains
        # bit-for-bit as before.
        sampling = cfg.get("sampling") or {}
        self.grasp_oversample = float(sampling.get("grasp_oversample", 1.0))
        self.grasp_window_s = float(sampling.get("grasp_window_s", 1.5))
        self.rate_hz = float((cfg.get("alignment") or {}).get("rate_hz", 30.0))

    def __len__(self):
        return len(self.episode_ids)

    def _episode_path(self, episode_id):
        folder = str(episode_id).zfill(3)
        return os.path.join(self.store_root, folder, self.episode_file)

    # picks a random ENGAGED timestep, both arms; falls back to any timestep
    def _pick_t(self, f):
        states = [f[f"observations/{arm}/state"][:] for arm in self.arms]
        engaged = np.ones_like(states[0], dtype=bool)
        for s in states:
            engaged &= np.isin(s, list(self.train_states))
        valid = np.where(engaged)[0]
        if len(valid) == 0:
            valid = np.arange(len(states[0]))
        if self.grasp_oversample <= 1.0:
            return int(np.random.choice(valid))
        w = self._grasp_weights(f, valid)
        return int(np.random.choice(valid, p=w))

    def _grasp_window(self, f):
        """Boolean mask over the episode: frames in the approach-to-grasp window.

        Derived entirely from actions/<arm>/gripper_cmd, which is RAW RECORDED
        DATA -- gripper_cmd is logged as exactly 0.0 (closed) or 0.08 (open)
        by data_logger.hpp, so a 0.08 -> 0.0 transition IS the grasp instant.
        No annotation, no labels/ dataset, nothing to hand-label. (episode.hdf5
        does carry labels/arm_*_phase, but this deliberately does not use it:
        depending on it would make the training set only as good as its
        labelling, and the signal is already unambiguous in the actions.)

        The window is the grasp_window_s BEFORE each close, not the close
        itself -- that is the final approach the policy currently gets wrong,
        and it is where the conjunction of "centred over the object" and "at
        grasp depth" actually lives. Measured over 12 episodes, the arm is in
        that joint configuration for only 1.65% of engaged frames while a
        uniform sampler spends 98% of its budget elsewhere.
        """
        n = f[f"actions/{self.arms[0]}/gripper_cmd"].shape[0]
        win = np.zeros(n, dtype=bool)
        span = int(round(self.grasp_window_s * self.rate_hz))
        for arm in self.arms:
            g = f[f"actions/{arm}/gripper_cmd"][:]
            closes = np.where(np.diff((g < CLOSED_WIDTH_M).astype(np.int8)) > 0)[0]
            for c in closes:
                win[max(0, c - span):min(n, c + 1)] = True
        return win

    def _grasp_weights(self, f, valid):
        """Sampling distribution over `valid` that favours the grasp window by
        grasp_oversample:1, normalised so it stays a probability vector however
        many grasp frames the episode happens to contain."""
        win = self._grasp_window(f)[valid]
        w = np.where(win, float(self.grasp_oversample), 1.0)
        return w / w.sum()

    def _load_image(self, f, cam, t):
        img = f[f"observations/images/{cam}"][t]
        if img.shape[:2] != self.img_hw:
            import cv2
            img = cv2.resize(img, (self.img_hw[1], self.img_hw[0]), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def _proprio(self, f, t):
        # world frame (T_base * O_T_EE), not base frame -- see configs/dataset.yaml's
        # proprio.source comment for why.
        parts = []
        for arm in self.arms:
            pos, rot6d = flat16_to_pos_rot6d(f[f"observations/{arm}/O_T_EE_world"][t])
            grip = f[f"observations/{arm}/gripper_width"][t]
            parts.append(np.concatenate([pos, rot6d, [grip]]))
        return np.concatenate(parts).astype(np.float32)

    # commanded action chunk [K, 20] starting at t, padded with the last
    # valid action past episode end; is_pad marks the padded steps
    def _action_chunk(self, f, t):
        T = f["actions/arm_left/O_T_EE_cmd_world"].shape[0]
        end = min(t + self.K, T)
        chunk = []
        for arm in self.arms:
            pos, rot6d = flat16_to_pos_rot6d(f[f"actions/{arm}/O_T_EE_cmd_world"][t:end])
            grip = f[f"actions/{arm}/gripper_cmd"][t:end][:, None]
            chunk.append(np.concatenate([pos, rot6d, grip], axis=-1))
        chunk = np.concatenate(chunk, axis=-1).astype(np.float32)  # [end-t, 20]

        is_pad = np.zeros(self.K, dtype=bool)
        if end - t < self.K:
            pad_n = self.K - (end - t)
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad_n, axis=0)], axis=0)
            is_pad[end - t:] = True
        return chunk, is_pad

    def __getitem__(self, idx):
        episode_id = self.episode_ids[idx]
        with h5py.File(self._episode_path(episode_id), "r") as f:
            t = self._pick_t(f)
            images = np.stack([self._load_image(f, cam, t) for cam in self.cams])  # [C,H,W,3]
            proprio = self._proprio(f, t)
            action, is_pad = self._action_chunk(f, t)

        if self.stats is not None:
            proprio = normalize(proprio, self.stats["proprio_mean"], self.stats["proprio_std"])
            action = normalize(action, self.stats["action_mean"], self.stats["action_std"])

        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()  # [C,3,H,W]
        return {
            "images": images,
            "proprio": torch.from_numpy(proprio).float(),
            "action": torch.from_numpy(action).float(),
            "is_pad": torch.from_numpy(is_pad),
            "episode_id": episode_id,
            "t": t,
        }


def build_datasets(cfg_path):
    """Convenience: returns (train_ds, val_ds, test_ds) using saved normalization stats."""
    cfg = load_cfg(cfg_path)
    stats_path = cfg["normalize"]["stats_file"]
    stats = None
    if os.path.exists(stats_path):
        npz = np.load(stats_path)
        stats = {k: npz[k] for k in npz.files}

    splits_dir = cfg["data"]["splits"]
    train_ids = read_split(os.path.join(splits_dir, "train.txt"))
    val_ids = read_split(os.path.join(splits_dir, "val.txt"))
    test_ids = read_split(os.path.join(splits_dir, "test.txt"))

    return (EpisodicDataset(train_ids, cfg, stats),
            EpisodicDataset(val_ids, cfg, stats),
            EpisodicDataset(test_ids, cfg, stats))
