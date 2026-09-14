# Within-scene candidate-discrimination check, offline: the cross-episode
# spread check (check_mode_collapse.py) only shows the model's target moves
# WITH the scene overall -- it can't tell "picked the right parcel" apart
# from "landed on the average/centroid of whatever parcels were present,"
# which looks identical from outside (coarse reach toward the staging area,
# no commitment to one object) and is the classic CVAE failure mode: collapse
# on the hard multimodal decision (which parcel?) while still conditioning
# fine on the easy near-unimodal part (which general area?).
#
# For each val episode with >=2 logged candidates (scene.csv obj0..objN-1,
# N = n_objects), this compares, at the same tick t (default ~7.5s in, past
# the homing-pose confound -- see check_mode_collapse.py):
#   - which candidate the model's predicted target is nearest to
#   - which candidate the actual logged/commanded (human demo) position is
#     nearest to, at that same tick t
#   - whether the predicted target sits closer to the CENTROID of all
#     candidates than to its own nearest candidate -- the direct signature
#     of "averaged the options" rather than "picked one"
#
# Usage: python scripts/check_candidate_discrimination.py checkpoints/act_sorting_world/best.pt
#        python scripts/check_candidate_discrimination.py checkpoints/act_sorting_world/best.pt --seconds-offset 7.5 --episodes 15

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


def scene_candidates(dcfg, episode_id):
    """[n_objects,3] world positions from scene.csv's obj0..objN-1 columns, or None."""
    path = os.path.join(dcfg["data"]["store_root"], str(episode_id).zfill(3), "scene.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        header = f.readline().strip().split(";")
    if "n_objects" not in header:
        return None
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n_objects = int(data[0, header.index("n_objects")])
    if n_objects < 2:
        return None
    cands = []
    for i in range(n_objects):
        cols = (f"obj{i}_x", f"obj{i}_y", f"obj{i}_z")
        if not all(c in header for c in cols):
            return None
        idx = [header.index(c) for c in cols]
        cands.append(data[0, idx])
    return np.stack(cands)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("--split", default="val")
    ap.add_argument("--episodes", type=int, default=15, help="how many split episodes to check")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--seconds-offset", type=float, default=7.5,
                     help="tick, in seconds into ENGAGED, to compare model vs. demo at -- keep past "
                          "the homing-pose confound (see check_mode_collapse.py)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, dcfg = load_model(args.checkpoint, device)
    stats = load_stats(dcfg)
    arms = dcfg["action"]["arms"]
    d = dcfg["action"]["dims_per_arm"]
    rate_hz = dcfg["alignment"]["rate_hz"]
    tick_offset = int(round(args.seconds_offset * rate_hz))
    print(f"[check_candidate_discrimination] tick_offset={tick_offset} ({args.seconds_offset}s @ {rate_hz}Hz)")

    episode_ids = read_split(dcfg, args.split)[:args.episodes]

    per_arm_rows = {arm: [] for arm in arms}
    n_skipped_single_obj = 0
    n_skipped_no_data = 0

    for eid in episode_ids:
        cands = scene_candidates(dcfg, eid)
        if cands is None:
            n_skipped_single_obj += 1
            continue
        path = episode_path(dcfg, eid)
        if not os.path.exists(path):
            n_skipped_no_data += 1
            continue
        with h5py.File(path, "r") as f:
            _t0, _t1, logged_cmd, ensembled = open_loop_replay(f, model, dcfg, stats, device, args.stride)
        t = min(tick_offset, len(ensembled) - 1, len(logged_cmd) - 1)
        centroid = cands.mean(axis=0)

        for i, arm in enumerate(arms):
            pred_pos = ensembled[t][i * d: i * d + 3]
            actual_pos = logged_cmd[t][i * d: i * d + 3]

            dists_pred = np.linalg.norm(cands - pred_pos, axis=1)
            dists_actual = np.linalg.norm(cands - actual_pos, axis=1)
            nearest_pred = int(np.argmin(dists_pred))
            nearest_actual = int(np.argmin(dists_actual))

            dist_pred_nearest = dists_pred[nearest_pred]
            dist_pred_centroid = np.linalg.norm(pred_pos - centroid)

            row = {
                "episode": eid,
                "n_candidates": len(cands),
                "match": nearest_pred == nearest_actual,
                "dist_pred_nearest": dist_pred_nearest,
                "dist_pred_centroid": dist_pred_centroid,
                "closer_to_centroid": dist_pred_centroid < dist_pred_nearest,
            }
            per_arm_rows[arm].append(row)
            print(f"{eid} {arm}: n_cand={len(cands)}  nearest_pred={nearest_pred} nearest_actual={nearest_actual} "
                  f"{'MATCH' if row['match'] else 'miss'}  "
                  f"dist_to_nearest={dist_pred_nearest:.4f}m dist_to_centroid={dist_pred_centroid:.4f}m"
                  f"{'  <-- closer to centroid than to any candidate' if row['closer_to_centroid'] else ''}")

    print(f"\n(skipped {n_skipped_single_obj} episodes with <2 candidates or missing scene.csv columns, "
          f"{n_skipped_no_data} with no episode.hdf5)")

    print()
    for arm, rows in per_arm_rows.items():
        if not rows:
            print(f"{arm}: no multi-candidate episodes found")
            continue
        match_rate = np.mean([r["match"] for r in rows])
        centroid_rate = np.mean([r["closer_to_centroid"] for r in rows])
        mean_dist_nearest = np.mean([r["dist_pred_nearest"] for r in rows])
        mean_dist_centroid = np.mean([r["dist_pred_centroid"] for r in rows])
        print(f"{arm} ({len(rows)} episodes): "
              f"selection match rate {match_rate:.0%}  |  "
              f"closer-to-centroid-than-to-any-candidate {centroid_rate:.0%}  |  "
              f"mean dist to nearest candidate {mean_dist_nearest:.4f}m vs. mean dist to centroid {mean_dist_centroid:.4f}m")

    print("\nLow match rate + frequently closer to centroid than to any single candidate = the model is "
          "averaging over candidates instead of picking one -- collapse on the discrete selection, not on "
          "overall scene conditioning. High match rate = it's genuinely discriminating and the live-run "
          "symptom is something else (e.g. live-only distribution shift, or a downstream execution/grasp issue).")


if __name__ == "__main__":
    main()
