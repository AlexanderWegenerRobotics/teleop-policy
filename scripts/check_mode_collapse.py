# Cross-episode mode-collapse check, offline: runs open_loop_replay's first
# prediction (t0, always a real inference regardless of stride -- t0 is
# always in range(t0,t1,stride)) over several validation episodes and
# compares how much the predicted initial target varies episode-to-episode
# against a reference scale (obj0's own logged position spread across the
# same episodes). If the model is actually conditioning on the scene, the
# predicted-target spread should be comparable to the scene's own spread; if
# it's much smaller, the model is producing close to the same output
# regardless of what's actually in front of it -- the live-run symptom
# (predicted target holding still, not tracking toward the object) reduced
# to a single number instead of an eyeballed video.
#
# obj0_x/y/z is a cheap per-episode fingerprint (first candidate's logged
# world position, see scene.csv), not a rigorous candidate-to-slot mapping --
# good enough as a reference scale, see backfill_candidate_position.py for
# the real mapping if that's ever needed elsewhere.
#
# Usage: python scripts/check_mode_collapse.py checkpoints/act_sorting_world/best.pt
#        python scripts/check_mode_collapse.py checkpoints/act_sorting_world/best.pt --split val --episodes 10

import argparse
import os
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from replay_utils import episode_path, load_model, load_stats, open_loop_replay


def read_split(dcfg, split):
    path = os.path.join(dcfg["data"]["splits"], f"{split}.txt")
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def scene_obj0_position(dcfg, episode_id):
    """First candidate's logged (x,y,z) from scene.csv, or None if missing."""
    path = os.path.join(dcfg["data"]["store_root"], str(episode_id).zfill(3), "scene.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().strip().split(";")
    cols = ("obj0_x", "obj0_y", "obj0_z")
    if not all(c in header for c in cols):
        return None
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    idx = [header.index(c) for c in cols]
    return data[0, idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="val")
    ap.add_argument("--episodes", type=int, default=8, help="how many split episodes to check")
    ap.add_argument("--stride", type=int, default=4, help="open_loop_replay stride -- coarser is fine")
    ap.add_argument("--seconds-offset", type=float, default=0.0,
                     help="check the prediction this many seconds into ENGAGED instead of t0 -- t0 is "
                          "confounded by every episode starting from ~the same home pose (homogeneous "
                          "proprio regardless of scene), so a flat t0 result alone doesn't distinguish "
                          "'ignores vision' from 'hasn't committed yet'. Pick something past your "
                          "observed time-to-steady-state (e.g. 7-8s) to rule that out.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dcfg = load_model(args.checkpoint, device)
    stats = load_stats(dcfg)
    arms = dcfg["action"]["arms"]
    d = dcfg["action"]["dims_per_arm"]
    rate_hz = dcfg["alignment"]["rate_hz"]
    tick_offset = int(round(args.seconds_offset * rate_hz))
    print(f"[check_mode_collapse] sampling tick_offset={tick_offset} ({args.seconds_offset}s @ {rate_hz}Hz) "
          f"into each episode's ENGAGED window (clipped to episode length if shorter)")

    episode_ids = read_split(dcfg, args.split)[:args.episodes]

    rows = []
    for eid in episode_ids:
        path = episode_path(dcfg, eid)
        if not os.path.exists(path):
            print(f"{eid}: skip (no episode.hdf5)")
            continue
        with h5py.File(path, "r") as f:
            _t0, _t1, _logged_cmd, ensembled = open_loop_replay(f, model, dcfg, stats, device, args.stride)
        t = min(tick_offset, len(ensembled) - 1)
        pred_t = ensembled[t]
        obj0 = scene_obj0_position(dcfg, eid)
        row = {"episode": eid, "obj0": obj0}
        for i, arm in enumerate(arms):
            row[f"{arm}_pos"] = pred_t[i * d: i * d + 3]
        rows.append(row)
        pos_str = "  ".join(f"{arm}={row[f'{arm}_pos']}" for arm in arms)
        obj_str = f"obj0={obj0}" if obj0 is not None else "obj0=? (no scene.csv/columns)"
        print(f"{eid}: {pos_str}   ({obj_str})")

    if len(rows) < 2:
        print("need at least 2 episodes with data to compute spread")
        return

    print()
    for arm in arms:
        positions = np.stack([r[f"{arm}_pos"] for r in rows])
        centroid = positions.mean(axis=0)
        spread = np.linalg.norm(positions - centroid, axis=1)
        print(f"{arm}: predicted t0 target spread across {len(rows)} episodes -- "
              f"mean {spread.mean():.4f}m, max {spread.max():.4f}m from centroid")

    obj0_positions = np.stack([r["obj0"] for r in rows if r["obj0"] is not None])
    if len(obj0_positions) >= 2:
        obj0_centroid = obj0_positions.mean(axis=0)
        obj0_spread = np.linalg.norm(obj0_positions - obj0_centroid, axis=1)
        print(f"\nreference -- obj0's own logged position spread across these same episodes: "
              f"mean {obj0_spread.mean():.4f}m, max {obj0_spread.max():.4f}m from centroid")
        print("If the predicted-target spread above is much smaller than this reference, the "
              "model isn't tracking per-episode scene differences -- consistent with mode collapse. "
              "If they're comparable, the model IS scene-conditioned and the live-run issue is "
              "something else (distribution shift from training, a live-only bug, etc).")
    else:
        print("\n(no obj0 reference available -- can't compare against scene spread, "
              "but the predicted-target spread above is still informative on its own: "
              "near-zero spread across genuinely different episodes is still suspicious)")


if __name__ == "__main__":
    main()
