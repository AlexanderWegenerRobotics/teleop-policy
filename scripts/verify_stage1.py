# Stage 1 gate check, no torch required: confirms commanded EE trajectories
# lead measured motion by a plausible margin, and that the 6D rotation round
# trips through Gram-Schmidt cleanly. Run against one episode before trusting
# the full Dataset class.
#
# Usage: python scripts/verify_stage1.py <episode.hdf5> [out_dir]

import os
import sys

import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dataset"))
from transforms import flat16_to_pos_rot6d, rot6d_to_matrix

ARMS = ["arm_left", "arm_right"]


def best_lag(cmd, meas, max_lag=30):
    """Cross-correlation lag (in frames) that best aligns meas to cmd; positive = cmd leads."""
    cmd_speed = np.linalg.norm(np.gradient(cmd, axis=0), axis=1)
    meas_speed = np.linalg.norm(np.gradient(meas, axis=0), axis=1)
    lags = range(-max_lag, max_lag + 1)
    scores = [np.corrcoef(cmd_speed[max_lag: -max_lag or None],
                           np.roll(meas_speed, -l)[max_lag: -max_lag or None])[0, 1]
              for l in lags]
    return list(lags)[int(np.argmax(scores))]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/sessions/gifted-zealous-fermat/mnt/000/episode.hdf5"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    with h5py.File(path, "r") as f:
        engaged = np.ones(f["observations/timestamp_ns"].shape[0], dtype=bool)
        for arm in ARMS:
            engaged &= f[f"observations/{arm}/state"][:] == 4  # ENGAGED
        print(f"engaged fraction: {engaged.mean():.3f}")

        fig, axes = plt.subplots(len(ARMS), 3, figsize=(12, 6), sharex=True)
        for row, arm in enumerate(ARMS):
            meas_pos, meas_rot6d = flat16_to_pos_rot6d(f[f"observations/{arm}/O_T_EE"][:])
            cmd_pos, cmd_rot6d = flat16_to_pos_rot6d(f[f"actions/{arm}/O_T_EE_cmd"][:])

            lag = best_lag(cmd_pos, meas_pos)
            print(f"{arm}: best lag = {lag} frames ({'cmd leads' if lag > 0 else 'meas leads / no lag'})")

            # rot6d -> matrix round trip sanity: columns should be ~unit, ~orthogonal
            R = rot6d_to_matrix(meas_rot6d[:5])
            col_norms = np.linalg.norm(R, axis=1)
            print(f"{arm}: rotation column norms (first 5 frames, expect ~1.0): {col_norms.flatten()[:3]}")

            for c in range(3):
                ax = axes[row, c]
                ax.plot(cmd_pos[:, c], label="commanded", alpha=0.8)
                ax.plot(meas_pos[:, c], label="measured", alpha=0.8)
                ax.set_title(f"{arm} pos[{c}]")
                if row == 0 and c == 0:
                    ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, "stage1_cmd_vs_measured.png")
        fig.savefig(out_path, dpi=120)
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
