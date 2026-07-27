# Aggregate open-loop generalization check: runs the same replay+ensembling
# as verify_stage2.py over every episode in a split, reports the error
# distribution instead of one episode at a time. Still offline/no sim — a
# cheap regression check to run after every training run, not a substitute
# for the closed-loop rollout harness.
#
# Usage: python scripts/eval_open_loop.py checkpoints/act_sorting_full/best.pt val [stride]

import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.teleop_dataset import read_split
from replay_utils import episode_path, load_model, load_stats, open_loop_replay


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/act_sorting_full/best.pt"
    split = sys.argv[2] if len(sys.argv) > 2 else "val"
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dcfg = load_model(ckpt_path, device)
    stats = load_stats(dcfg)
    episode_ids = read_split(os.path.join(dcfg["data"]["splits"], f"{split}.txt"))

    errors = []
    for eid in episode_ids:
        with h5py.File(episode_path(dcfg, eid), "r") as f:
            _, _, logged_cmd, ensembled = open_loop_replay(f, model, dcfg, stats, device, stride)
        err = float(np.abs(ensembled - logged_cmd).mean())
        errors.append((eid, err))
        print(f"episode {eid}: {err:.4f}")

    errs = np.array([e for _, e in errors])
    order = np.argsort(errs)[::-1]
    print(f"\n{split} split ({len(errors)} episodes)  mean {errs.mean():.4f}  "
          f"std {errs.std():.4f}  min {errs.min():.4f}  max {errs.max():.4f}")
    print("worst 5:", [(errors[i][0], f"{errors[i][1]:.4f}") for i in order[:5]])

    fig, ax = plt.subplots(figsize=(max(6, len(errors) * 0.4), 4))
    sorted_pairs = sorted(errors, key=lambda x: x[1])
    ax.bar([str(e) for e, _ in sorted_pairs], [v for _, v in sorted_pairs])
    ax.axhline(errs.mean(), color="red", linestyle="--", label=f"mean {errs.mean():.4f}")
    ax.set_ylabel("mean |predicted - commanded|")
    ax.set_title(f"open-loop generalization: {split} split")
    ax.legend()
    plt.xticks(rotation=90)
    fig.tight_layout()
    out_path = f"open_loop_eval_{split}.png"
    fig.savefig(out_path, dpi=120)
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
