import os

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from .transforms import flat16_to_pos_rot6d, normalize


def load_cfg(path):
    """Load a YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)


def read_split(path):
    """Read episode ids from a split file."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def engaged_mask(f, arms, train_states=(4,)):
    """Frames where both arms are engaged with a valid command from the control loop."""
    mask = None
    for arm in arms:
        g = f[f"observations/{arm}"]
        m = np.isin(g["state"][:], list(train_states))
        if "cmd_valid" in g:
            m &= g["cmd_valid"][:] == 1
        if "log_src" in g:
            m &= g["log_src"][:] == 0
        mask = m if mask is None else mask & m
    return mask


class EpisodicDataset(Dataset):
    """One random engaged timestep per episode with its K-step action chunk."""

    def __init__(self, episode_ids, cfg, stats=None):
        """Store episode ids, config fields and normalization stats."""
        self.episode_ids = episode_ids
        self.store_root = cfg["data"]["store_root"]
        self.episode_file = cfg["data"]["episode_file"]
        self.cams = cfg["cameras"]["use"]
        self.img_hw = tuple(cfg["cameras"]["resize_to"])
        self.arms = cfg["action"]["arms"]
        self.K = cfg["action"]["chunk_size"]
        self.train_states = set(cfg["state"]["train_states"])
        self.stats = stats

    def __len__(self):
        """Number of episodes."""
        return len(self.episode_ids)

    def _episode_path(self, episode_id):
        """Path to an episode's hdf5 file."""
        folder = str(episode_id).zfill(3)
        return os.path.join(self.store_root, folder, self.episode_file)

    def _pick_t(self, f):
        """Random engaged timestep, else any timestep."""
        engaged = engaged_mask(f, self.arms, self.train_states)
        valid = np.where(engaged)[0]
        if len(valid) == 0:
            valid = np.arange(len(engaged))
        return int(np.random.choice(valid))

    def _load_image(self, f, cam, t):
        """Camera frame at t, resized and scaled to [0, 1]."""
        img = f[f"observations/images/{cam}"][t]
        if img.shape[:2] != self.img_hw:
            import cv2
            img = cv2.resize(img, (self.img_hw[1], self.img_hw[0]), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    def _proprio(self, f, t):
        """World-frame EE pose and gripper width for both arms at t."""
        parts = []
        for arm in self.arms:
            pos, rot6d = flat16_to_pos_rot6d(f[f"observations/{arm}/O_T_EE_world"][t])
            grip = f[f"observations/{arm}/gripper_width"][t]
            parts.append(np.concatenate([pos, rot6d, [grip]]))
        return np.concatenate(parts).astype(np.float32)

    def _action_chunk(self, f, t):
        """Commanded action chunk [K, 20] from t, edge-padded, with pad mask."""
        T = f["actions/arm_left/O_T_EE_cmd_world"].shape[0]
        end = min(t + self.K, T)
        chunk = []
        for arm in self.arms:
            pos, rot6d = flat16_to_pos_rot6d(f[f"actions/{arm}/O_T_EE_cmd_world"][t:end])
            grip = f[f"actions/{arm}/gripper_cmd"][t:end][:, None]
            chunk.append(np.concatenate([pos, rot6d, grip], axis=-1))
        chunk = np.concatenate(chunk, axis=-1).astype(np.float32)

        is_pad = np.zeros(self.K, dtype=bool)
        if end - t < self.K:
            pad_n = self.K - (end - t)
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad_n, axis=0)], axis=0)
            is_pad[end - t:] = True
        return chunk, is_pad

    def __getitem__(self, idx):
        """Sample images, proprio and action chunk from one episode."""
        episode_id = self.episode_ids[idx]
        with h5py.File(self._episode_path(episode_id), "r") as f:
            t = self._pick_t(f)
            images = np.stack([self._load_image(f, cam, t) for cam in self.cams])
            proprio = self._proprio(f, t)
            action, is_pad = self._action_chunk(f, t)

        if self.stats is not None:
            proprio = normalize(proprio, self.stats["proprio_mean"], self.stats["proprio_std"])
            action = normalize(action, self.stats["action_mean"], self.stats["action_std"])

        images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        return {
            "images": images,
            "proprio": torch.from_numpy(proprio).float(),
            "action": torch.from_numpy(action).float(),
            "is_pad": torch.from_numpy(is_pad),
            "episode_id": episode_id,
            "t": t,
        }


def build_datasets(cfg_path):
    """Build train, val and test datasets with saved normalization stats."""
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
