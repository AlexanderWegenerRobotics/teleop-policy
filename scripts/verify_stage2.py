# Stage 2 gate, offline: feed a training episode's logged images/proprio
# into a checkpoint frame by frame, temporally-ensemble the predicted
# chunks, and compare against the logged commanded trajectory. No sim, no
# robot — checks the model reproduces what it trained on before any
# rollout harness exists.
#
# Usage: python scripts/verify_stage2.py checkpoints/act_sorting/best.pt 000 [stride]

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_utils import episode_path, load_model, load_stats, open_loop_replay


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/act_sorting/best.pt"
    episode_id = sys.argv[2] if len(sys.argv) > 2 else "0"
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dcfg = load_model(ckpt_path, device)
    stats = load_stats(dcfg)
    arms = dcfg["action"]["arms"]

    with h5py.File(episode_path(dcfg, episode_id), "r") as f:
        t0, t1, logged_cmd, ensembled = open_loop_replay(f, model, dcfg, stats, device, stride)

    fig, axes = plt.subplots(len(arms), 4, figsize=(16, 6), sharex=True)
    for row, arm in enumerate(arms):
        off = row * 10
        for c, (dim_idx, label) in enumerate([(0, "x"), (1, "y"), (2, "z"), (9, "gripper")]):
            ax = axes[row, c]
            ax.plot(logged_cmd[:, off + dim_idx], label="logged commanded", alpha=0.8)
            ax.plot(ensembled[:, off + dim_idx], label="predicted (ensembled)", alpha=0.8)
            ax.set_title(f"{arm} {label}")
            if row == 0 and c == 0:
                ax.legend()
    fig.tight_layout()
    out_path = f"stage2_replay_ep{episode_id}.png"
    fig.savefig(out_path, dpi=120)
    print(f"[ok] {out_path}")

    # mixed units (m for pos/gripper, unitless for 6D rot) -- rough magnitude check only
    print(f"mean |predicted - commanded|: {np.abs(ensembled - logged_cmd).mean():.4f}")


if __name__ == "__main__":
    main()
