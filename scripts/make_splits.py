import argparse
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import read_meta


def main():
    """Write seeded train/val/test episode splits from the episodes in store_root."""
    ap = argparse.ArgumentParser(description="Create train/val/test episode splits.")
    ap.add_argument("config", nargs="?", default="configs/dataset.yaml")
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = cfg["data"]["store_root"]
    keep = set(cfg["data"]["include_success"])
    episode_file = cfg["data"]["episode_file"]

    ids, skipped = [], []
    for folder in sorted(glob.glob(os.path.join(root, "[0-9]" * 3))):
        eid = os.path.basename(folder)
        outcome = read_meta(folder).get("success", "")
        if outcome not in keep:
            skipped.append((eid, outcome or "no outcome"))
        elif not os.path.exists(os.path.join(folder, episode_file)):
            skipped.append((eid, f"no {episode_file}"))
        else:
            ids.append(eid)

    rng = np.random.default_rng(args.seed)
    ids = list(rng.permutation(ids))
    n_val, n_test = round(len(ids) * args.val), round(len(ids) * args.test)
    splits = {"val": ids[:n_val], "test": ids[n_val:n_val + n_test], "train": ids[n_val + n_test:]}

    out_dir = cfg["data"]["splits"]
    os.makedirs(out_dir, exist_ok=True)
    for name, split in splits.items():
        with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
            f.write("".join(f"{eid}\n" for eid in sorted(split)))
        print(f"{name}: {len(split)}")
    for eid, reason in skipped:
        print(f"  [skip] {eid}: {reason}")


if __name__ == "__main__":
    main()
