# Computes per-dim mean/std of proprio+action over the train split only,
# saves to dataset/stats.npz. Run once before training; re-run if the train
# split or action representation changes.

import os
import sys

import h5py
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset"))
from transforms import flat16_to_pos_rot6d, compute_stats


def collect(cfg):
    store_root = cfg["data"]["store_root"]
    episode_file = cfg["data"]["episode_file"]
    arms = cfg["action"]["arms"]
    splits_dir = cfg["data"]["splits"]

    with open(os.path.join(splits_dir, "train.txt")) as f:
        train_ids = [ln.strip() for ln in f if ln.strip()]

    proprio_all, action_all = [], []
    for eid in train_ids:
        path = os.path.join(store_root, str(eid).zfill(3), episode_file)
        with h5py.File(path, "r") as f:
            p_parts, a_parts = [], []
            for arm in arms:
                # world frame -- see configs/dataset.yaml's proprio.source comment
                pos, rot6d = flat16_to_pos_rot6d(f[f"observations/{arm}/O_T_EE_world"][:])
                grip = f[f"observations/{arm}/gripper_width"][:][:, None]
                p_parts.append(np.concatenate([pos, rot6d, grip], axis=-1))

                cpos, crot6d = flat16_to_pos_rot6d(f[f"actions/{arm}/O_T_EE_cmd_world"][:])
                cgrip = f[f"actions/{arm}/gripper_cmd"][:][:, None]
                a_parts.append(np.concatenate([cpos, crot6d, cgrip], axis=-1))
            proprio_all.append(np.concatenate(p_parts, axis=-1))
            action_all.append(np.concatenate(a_parts, axis=-1))

    return np.concatenate(proprio_all, axis=0), np.concatenate(action_all, axis=0)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/dataset.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    proprio, action = collect(cfg)
    proprio_mean, proprio_std = compute_stats(proprio)
    action_mean, action_std = compute_stats(action)

    out_path = cfg["normalize"]["stats_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
              proprio_mean=proprio_mean, proprio_std=proprio_std,
              action_mean=action_mean, action_std=action_std)
    print(f"[ok] {out_path}  proprio {proprio.shape}  action {action.shape}")


if __name__ == "__main__":
    main()
