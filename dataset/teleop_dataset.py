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


HEAD_DIMS = 2
AUTHORITY_HUMAN = 1
HUMAN_FRAC_MIN = 0.9


def chunk_human_fraction(human, K):
    """Fraction of each K-step action chunk the operator held, edge-padded."""
    padded = np.concatenate([human, np.repeat(human[-1:], K - 1)]).astype(np.float64)
    csum = np.concatenate([[0.0], np.cumsum(padded)])
    return (csum[K:] - csum[:-K]) / K


def human_mask(f, arms):
    """Frames the operator held on every arm, or None if not an intervention episode."""
    mask = None
    for arm in arms:
        g = f[f"observations/{arm}"]
        if "authority" not in g:
            return None
        m = g["authority"][:] == AUTHORITY_HUMAN
        mask = m if mask is None else mask & m
    return mask if mask is not None and mask.any() else None


def head_cfg(cfg):
    """Head config if head joints are part of proprio and action, else None."""
    head = cfg.get("head") or {}
    return head if head.get("use") else None


def vector_dim(cfg):
    """Proprio and action size: pose and gripper per arm, plus head joints if enabled."""
    n = cfg["action"]["dims_per_arm"] * len(cfg["action"]["arms"])
    return n + (HEAD_DIMS if head_cfg(cfg) else 0)


def _vector(f, arms, sl, pose_key, grip_key, group, head_key):
    """Per-arm pose and gripper, plus head joints, over a slice of frames -> [n, D]."""
    parts = []
    for arm in arms:
        pos, rot6d = flat16_to_pos_rot6d(f[f"{group}/{arm}/{pose_key}"][sl])
        grip = f[f"{group}/{arm}/{grip_key}"][sl][:, None]
        parts += [pos, rot6d, grip]
    if head_key:
        parts.append(f[head_key][sl])
    return np.concatenate(parts, axis=-1).astype(np.float32)


def proprio_vector(f, cfg, sl):
    """Measured state over a slice of frames -> [n, D]."""
    head = head_cfg(cfg)
    return _vector(f, cfg["action"]["arms"], sl, "O_T_EE_world", "gripper_width", "observations",
                   head["proprio"] if head else None)


def action_vector(f, cfg, sl):
    """Commanded action over a slice of frames -> [n, D]."""
    head = head_cfg(cfg)
    return _vector(f, cfg["action"]["arms"], sl, "O_T_EE_cmd_world", "gripper_cmd", "actions",
                   head["action"] if head else None)


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
        self.cfg = cfg

    def __len__(self):
        """Number of episodes."""
        return len(self.episode_ids)

    def _episode_path(self, episode_id):
        """Path to an episode's hdf5 file."""
        folder = str(episode_id).zfill(3)
        return os.path.join(self.store_root, folder, self.episode_file)

    def _pick_t(self, f):
        """Random engaged timestep, restricted to human-authored chunks when present."""
        engaged = engaged_mask(f, self.arms, self.train_states)
        human = human_mask(f, self.arms)
        if human is not None:
            filtered = engaged & human & (chunk_human_fraction(human, self.K) >= HUMAN_FRAC_MIN)
            if filtered.any():
                engaged = filtered
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
        """Measured state vector at t."""
        return proprio_vector(f, self.cfg, slice(t, t + 1))[0]

    def _action_chunk(self, f, t):
        """Commanded action chunk [K, D] from t, edge-padded, with pad mask."""
        T = f["actions/arm_left/O_T_EE_cmd_world"].shape[0]
        end = min(t + self.K, T)
        chunk = action_vector(f, self.cfg, slice(t, end))

        is_pad = np.zeros(self.K, dtype=bool)
        if end - t < self.K:
            pad_n = self.K - (end - t)
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad_n, axis=0)], axis=0)
            is_pad[end - t:] = True
        return chunk, is_pad

    def __getitem__(self, idx):
        """Sample images, proprio and action chunk from one episode."""
        episode_id = self.episode_ids[idx]
        return self._sample(episode_id, None)

    def _sample(self, episode_id, t):
        """Images, proprio and action chunk at t, or at a random engaged t if None."""
        with h5py.File(self._episode_path(episode_id), "r") as f:
            if t is None:
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


class EvalDataset(EpisodicDataset):
    """Fixed engaged timesteps every stride frames over all episodes, for a stable val loss."""

    def __init__(self, episode_ids, cfg, stats=None, stride=10):
        """Enumerate (episode, t) pairs once."""
        super().__init__(episode_ids, cfg, stats)
        self.index = []
        for eid in episode_ids:
            with h5py.File(self._episode_path(eid), "r") as f:
                ts = np.where(engaged_mask(f, self.arms, self.train_states))[0][::stride]
            self.index += [(eid, int(t)) for t in ts]

    def __len__(self):
        """Number of (episode, t) pairs."""
        return len(self.index)

    def __getitem__(self, idx):
        """Sample at a fixed (episode, t)."""
        return self._sample(*self.index[idx])


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
