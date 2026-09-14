"""Closed-loop offline replay -- the diagnostic every previous test was missing.

Why this exists
---------------
replay_utils.open_loop_replay feeds the model the LOGGED human-demo proprio at
every tick. That is teacher forcing: the input state is dragged forward by the
demonstrator regardless of what the model predicts. A policy that converges to
a fixed point the moment it has to stand on its own output will still score a
near-perfect L1 under open-loop replay, and will still pass a mode-collapse or
candidate-discrimination check -- because at every tick it is handed a fresh,
in-distribution state it never had to reach by itself.

check_mode_collapse.py and check_candidate_discrimination.py both run through
open_loop_replay, so neither can see this failure mode. The live rollout
(teleop-orchestrator) is the only closed-loop thing in the stack, and it stalls:
commanded EE speed decays ~8x, from demo-typical (~38 mm/s) to a ~4 mm/s floor,
and holds there, with the gripper never actuating.

This script closes the proprio loop offline so that failure is reproducible
without hardware, and separable from every deployment-side suspect (head
heuristic, UDP path, control rate).

What is and isn't closed
------------------------
proprio: CLOSED. Each tick the model is fed the pose it itself commanded last
    tick (perfect-tracking assumption -- the controller is treated as ideal, so
    any stall found here is the policy's, not the controller's).
vision:  OPEN. We replay logged frames; we cannot re-render the scene from a
    counterfactual arm pose. So the images keep showing the demonstrator's arms
    where the demonstrator put them, while the model's believed proprio drifts
    away from that.

That asymmetry makes this test CONSERVATIVE in the direction that matters: the
vision stream keeps handing the policy a correct, on-trajectory view of the
scene, which is strictly more information than it gets live. If it stalls even
with oracle vision, the stall is not a vision/distribution-shift problem.

The --blend sweep
-----------------
--blend b feeds  proprio = (1-b)*logged + b*own_last_command.
b=0 reproduces open_loop_replay exactly (sanity check -- should match its L1);
b=1 is fully closed-loop. Sweeping b shows how much of the offline result was
teacher forcing holding the policy up. A policy that is genuinely closed-loop
stable degrades gracefully; one that is coasting on teacher forcing falls off a
cliff somewhere in the middle.

Usage
-----
    python scripts/check_closed_loop.py \
        --checkpoint checkpoints/act_sorting_world/best.pt \
        --episodes 14 16 22 \
        --blend 0 0.5 1.0 \
        --out results/closed_loop.png

Run from the teleop-policy repo root (same convention as eval_open_loop.py --
replay_utils uses bare `from dataset.transforms import ...`).
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

M_ENSEMBLE = 0.01  # ALOHA default, matches replay_utils.py and policy_module.py
ENGAGED_STATE = 4


# ---------------------------------------------------------------------------
# Core replay. Deliberately takes a `predict` callable rather than a model, so
# the loop logic (chunk bookkeeping, ensembling, feedback) is testable without
# torch -- see _selftest() at the bottom.
# ---------------------------------------------------------------------------
def closed_loop_replay(predict, proprio_log, action_log, K, blend=1.0, stride=1):
    """Replays a single episode with the proprio loop closed by `blend`.

    predict(t, proprio[D]) -> a_hat[K, D]   (denormalized action space)
    proprio_log[T, D]  logged observed pose (teacher-forcing source)
    action_log[T, D]   logged commanded pose (the demo, for reference only)

    Returns ensembled[T', D] -- the model's own commanded trajectory.
    """
    T, D = proprio_log.shape
    n = (T - 1) // stride
    pending: dict[int, list[np.ndarray]] = {}
    ensembled = np.zeros((n, D), dtype=np.float32)

    # The state the model actually gets fed. Seeded from the logged pose so the
    # episode starts on-distribution; from then on it is whatever `blend` says.
    own = proprio_log[0].astype(np.float32).copy()

    for j in range(n):
        t = j * stride
        proprio = (1.0 - blend) * proprio_log[t] + blend * own

        a_hat = predict(t, proprio.astype(np.float32))  # [K, D]
        for i in range(K):
            pending.setdefault(j + i, []).append(a_hat[i])

        preds = np.stack(pending.pop(j, [a_hat[0]]))
        ages = np.arange(len(preds))[::-1]  # 0 = most recent
        w = np.exp(-M_ENSEMBLE * ages)
        w /= w.sum()
        cmd = (w[:, None] * preds).sum(axis=0)

        ensembled[j] = cmd
        # Perfect-tracking assumption: what we commanded is where we now are.
        own = cmd.astype(np.float32)

    return ensembled


def _report_reach_depth(results, demo_logs, sls, arms, blends):
    """Does the model descend as far as the demonstrator did?

    The live rollout hovers ~30 mm above where the demos grasp, and the arm
    tracks its commands to within ~6 mm, so the shortfall is in the commanded
    trajectory rather than the controller. The open question is whether the
    model also under-shoots on training-distribution inputs:

      under-shoots here too -> the z floor is a clamped tail of the action
          distribution and L1 regression fits the conditional median, which
          systematically fails to reach clamped extremes. Fix belongs in
          training (weight the terminal approach, or predict a residual
          against the clamp).
      reaches the floor here -> the model can do it and something about the
          live observation stops it. Fix belongs in deployment.

    Per episode we compare the model's lowest commanded z against that same
    episode's logged floor, so scene-to-scene variation in table height or
    object placement cancels out.
    """
    print(f"\n{'blend':>6} | {'arm':>9} | {'model z-min':>11} | {'demo z-floor':>12} | "
          f"{'shortfall':>10} | {'reached':>8}")
    print("-" * 74)
    for b in blends:
        for a, sl in enumerate(sls):
            zi = sl.start + 2  # x, y, z -> z is the third column of this arm's block
            short, reached, mz, dz = [], 0, [], []
            for ens, demo in zip(results[b], demo_logs):
                m, d = float(ens[:, zi].min()), float(demo[:, zi].min())
                short.append((m - d) * 1000.0)
                mz.append(m)
                dz.append(d)
                # within 5 mm of the floor counts as having got there
                reached += int(m - d < 0.005)
            n = len(short)
            print(f"{b:6.2f} | {arms[a]:>9} | {np.mean(mz):11.4f} | {np.mean(dz):12.4f} | "
                  f"{np.mean(short):+7.1f} mm | {reached:>3}/{n:<4}")
    print("-" * 74)
    print("  per-episode detail (blend 1.00, fully closed-loop):")
    b = max(blends)
    for a, sl in enumerate(sls):
        zi = sl.start + 2
        for ens, demo in zip(results[b], demo_logs):
            m, d = float(ens[:, zi].min()), float(demo[:, zi].min())
            print(f"    {arms[a]:>9}  model {m:.4f}  demo floor {d:.4f}  "
                  f"shortfall {(m - d) * 1000:+6.1f} mm")
    print("\n  Reading: a shortfall near 0 means the model reaches the demo's grasp")
    print("  depth on training-distribution inputs, so the live ~30 mm gap is a")
    print("  deployment problem. A consistently positive shortfall means it never")
    print("  learned to reach the floor, and live is just reproducing that.")


def speed_mm_s(pos: np.ndarray, rate_hz: float) -> np.ndarray:
    """Commanded EE speed, mm/s. Per-second so runs at different tick rates
    stay comparable (the live loop ticks ~9 Hz, this replays at 30 Hz)."""
    d = np.linalg.norm(np.diff(pos, axis=0), axis=1) * 1000.0
    return d * rate_hz


# ---------------------------------------------------------------------------
# Episode IO (mirrors replay_utils.pose_at layout: per arm pos3 + rot6d + grip)
# ---------------------------------------------------------------------------
def _flat_to_pos_rot6d(flat):
    return np.concatenate([flat[..., 12:15], flat[..., 0:3], flat[..., 4:7]], axis=-1)


def load_episode(path, arms):
    with h5py.File(path, "r") as f:
        engaged = np.ones(f[f"observations/{arms[0]}/state"].shape[0], dtype=bool)
        for arm in arms:
            engaged &= f[f"observations/{arm}/state"][:] == ENGAGED_STATE
        idx = np.where(engaged)[0]
        t0, t1 = int(idx.min()), int(idx.max())
        sl = slice(t0, t1)

        def stack(group, pose_key, grip_key):
            parts = []
            for arm in arms:
                pose = _flat_to_pos_rot6d(f[f"{group}/{arm}/{pose_key}"][sl])
                grip = f[f"{group}/{arm}/{grip_key}"][sl][:, None]
                parts.append(np.concatenate([pose, grip], axis=1))
            return np.concatenate(parts, axis=1).astype(np.float32)

        proprio = stack("observations", "O_T_EE_world", "gripper_width")
        action = stack("actions", "O_T_EE_cmd_world", "gripper_cmd")
        images = {cam: f[f"observations/images/{cam}"][sl]
                  for cam in f["observations/images"].keys()}
    return proprio, action, images, t0


def arm_slices(dims_per_arm, n_arms):
    """Column slices of the position (xyz) block for each arm."""
    return [slice(a * dims_per_arm, a * dims_per_arm + 3) for a in range(n_arms)]


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/act_sorting_world/best.pt")
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="episode ids; default = the val split")
    ap.add_argument("--blend", nargs="+", type=float, default=[0.0, 0.5, 1.0],
                    help="0 = open-loop (reproduces replay_utils), 1 = fully closed-loop")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/closed_loop.png")
    args = ap.parse_args(argv)

    # torch imported lazily so the loop logic above stays importable (and
    # self-testable) in environments without it.
    import torch
    sys.path.insert(0, os.getcwd())
    from dataset.transforms import denormalize, normalize
    from replay_utils import load_model, load_stats, episode_path

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, dcfg = load_model(args.checkpoint, device)
    stats = load_stats(dcfg)

    arms = dcfg["action"]["arms"]
    cams, hw = dcfg["cameras"]["use"], tuple(dcfg["cameras"]["resize_to"])
    K = dcfg["action"]["chunk_size"]
    dpa = dcfg["action"]["dims_per_arm"]
    rate_hz = dcfg["alignment"]["rate_hz"] / args.stride

    episodes = args.episodes or [ln.strip() for ln in
                                 open(os.path.join(dcfg["data"]["splits"], "val.txt")) if ln.strip()]

    results = {b: [] for b in args.blend}
    demo_speeds = []

    for ep in episodes:
        path = episode_path(dcfg, ep)
        proprio_log, action_log, images, _ = load_episode(path, arms)

        # Pre-resize the whole image stack once; reused across every blend so
        # the only thing varying between conditions is the proprio feedback.
        import cv2
        cache = {}
        for cam in cams:
            arr = images[cam]
            if arr.shape[1:3] != hw:
                arr = np.stack([cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
                                for im in arr])
            cache[cam] = arr.astype(np.float32) / 255.0

        def predict(t, proprio):
            im = np.stack([cache[c][t] for c in cams])
            im_t = torch.from_numpy(im).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            pn = normalize(proprio, stats["proprio_mean"], stats["proprio_std"])
            p_t = torch.from_numpy(pn).unsqueeze(0).float().to(device)
            with torch.no_grad():
                a_hat, _, _ = model(im_t, p_t)
            return denormalize(a_hat[0].cpu().numpy(), stats["action_mean"], stats["action_std"])

        for b in args.blend:
            ens = closed_loop_replay(predict, proprio_log, action_log, K,
                                     blend=b, stride=args.stride)
            results[b].append(ens)
        demo_speeds.append(action_log)
        print(f"[ep {ep}] replayed {len(proprio_log)} frames x {len(args.blend)} blends")

    # ---- report ----------------------------------------------------------
    sls = arm_slices(dpa, len(arms))
    print(f"\n{'blend':>6} | {'arm':>9} | {'speed 1st qtr':>13} | {'speed last qtr':>14} | "
          f"{'decay':>6} | {'grip travel':>11}")
    print("-" * 78)
    demo_line = {}
    for a, sl in enumerate(sls):
        sp = np.concatenate([speed_mm_s(d[::args.stride, sl], rate_hz) for d in demo_speeds])
        gt = np.mean([np.ptp(d[:, sl.start + 9]) for d in demo_speeds]) * 1000
        demo_line[arms[a]] = (np.mean(sp), gt)

    for b in args.blend:
        for a, sl in enumerate(sls):
            firsts, lasts, grips = [], [], []
            for ens in results[b]:
                sp = speed_mm_s(ens[:, sl], rate_hz)
                n = len(sp)
                firsts.append(np.mean(sp[:n // 4]))
                lasts.append(np.mean(sp[-n // 4:]))
                grips.append(np.ptp(ens[:, sl.start + 9]) * 1000)
            f_, l_ = np.mean(firsts), np.mean(lasts)
            print(f"{b:6.2f} | {arms[a]:>9} | {f_:10.1f} mm/s | {l_:11.1f} mm/s | "
                  f"{f_ / max(l_, 1e-9):5.1f}x | {np.mean(grips):8.2f} mm")
    print("-" * 78)
    for arm, (sp, gt) in demo_line.items():
        print(f"{'demo':>6} | {arm:>9} | {sp:10.1f} mm/s | {'(reference)':>14} | "
              f"{'':>6} | {gt:8.2f} mm")

    _report_reach_depth(results, demo_speeds, sls, arms, args.blend)

    # ---- plot ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib unavailable, skipping figure")
        return

    fig, axarr = plt.subplots(2, len(arms), figsize=(7 * len(arms), 9), sharex=True)
    axarr = np.atleast_2d(axarr)
    for a, sl in enumerate(sls):
        ax = axarr[0, a]
        for b in args.blend:
            sp = np.mean([speed_mm_s(e[:, sl], rate_hz)[:min(len(x) for x in results[b]) - 1]
                          for e in results[b]], axis=0)
            ax.plot(np.arange(len(sp)) / rate_hz, sp, lw=1.6, label=f"blend {b:.2f}")
        dsp = np.mean([speed_mm_s(d[::args.stride, sl], rate_hz)[:len(sp)]
                       for d in demo_speeds if len(d[::args.stride]) > len(sp)] or [np.zeros(len(sp))],
                      axis=0)
        ax.plot(np.arange(len(dsp)) / rate_hz, dsp, "k--", lw=1.0, alpha=0.5, label="demo")
        ax.set_yscale("log")
        ax.set_title(f"{arms[a]} — commanded EE speed (mean over {len(episodes)} episodes)")
        ax.set_ylabel("mm / s")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        ax = axarr[1, a]
        for b in args.blend:
            g = np.mean([e[:, sl.start + 9][:min(len(x) for x in results[b])] for e in results[b]], axis=0)
            ax.plot(np.arange(len(g)) / rate_hz, g, lw=1.6, label=f"blend {b:.2f}")
        ax.set_title("gripper command")
        ax.set_ylabel("gripper (m)")
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Closed-loop offline replay — proprio fed back, vision from logs "
                 "(blend 0 = open-loop / teacher-forced, 1 = fully closed)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"\n[out] wrote {args.out}")


# ---------------------------------------------------------------------------
def _selftest():
    """Exercises the replay loop with synthetic predictors -- no torch, no data.

    Checks the three properties the diagnostic's validity rests on:
      1. blend=0 ignores the model's own output entirely (open-loop equivalence)
      2. blend=1 feeds the model's own command back (a hold-still policy stalls)
      3. a policy that genuinely tracks its input still advances when closed
    """
    K, D, T = 8, 4, 60
    rng = np.random.default_rng(0)
    proprio_log = np.cumsum(rng.normal(0, 0.05, (T, D)), axis=0).astype(np.float32)
    action_log = proprio_log.copy()

    seen: list[np.ndarray] = []

    def identity_policy(t, proprio):
        seen.append(proprio.copy())
        return np.tile(proprio, (K, 1))

    # 1: open-loop equivalence. The property that matters is that the model is
    # fed the logged proprio and nothing else -- NOT that the output equals it.
    # (Under temporal ensembling slot j mixes the chunks predicted at ticks
    # j-K+1..j, so even an identity policy returns a trailing weighted average
    # of recent logged poses, not the current one.)
    seen.clear()
    ens_open = closed_loop_replay(identity_policy, proprio_log, action_log, K, blend=0.0)
    assert np.allclose(np.stack(seen), proprio_log[:len(seen)], atol=1e-5), "blend=0 leaked feedback"
    # each output must be a convex combination of the K poses in its window
    for j in range(K, len(ens_open)):
        win = proprio_log[j - K + 1:j + 1]
        assert (ens_open[j] >= win.min(0) - 1e-4).all() and (ens_open[j] <= win.max(0) + 1e-4).all(), \
            f"blend=0 output at {j} outside its ensembling window"
    # a constant logged trajectory must pass through exactly
    const = np.tile(proprio_log[0], (T, 1))
    assert np.allclose(closed_loop_replay(identity_policy, const, const, K, blend=0.0),
                       proprio_log[0], atol=1e-4), "blend=0 != open loop on constant input"

    # 2: closed-loop, identity policy must freeze at the seed pose.

    seen.clear()
    ens_closed = closed_loop_replay(identity_policy, proprio_log, action_log, K, blend=1.0)
    assert np.allclose(ens_closed, proprio_log[0], atol=1e-4), "blend=1 should freeze a hold policy"
    assert np.ptp(np.stack(seen), axis=0).max() < 1e-4, "blend=1 should feed back its own output"

    # 3: a policy that advances by a fixed delta must keep advancing closed-loop.
    step = np.full(D, 0.01, dtype=np.float32)

    def advancing_policy(t, proprio):
        return np.stack([proprio + step * (i + 1) for i in range(K)])

    ens_adv = closed_loop_replay(advancing_policy, proprio_log, action_log, K, blend=1.0)
    d = np.diff(ens_adv, axis=0)
    assert d.min() > 0, "advancing policy stalled closed-loop -- loop is broken"
    assert abs(d.mean() - 0.01) < 5e-3, f"unexpected closed-loop step size {d.mean():.4f}"

    # ensembling weights: most recent prediction must dominate
    ages = np.arange(5)[::-1]
    w = np.exp(-M_ENSEMBLE * ages)
    w /= w.sum()
    assert w[-1] == w.max(), "recency weighting is inverted"

    print("selftest OK: open-loop equivalence, closed-loop feedback, "
          "advancing-policy sanity, recency weighting")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
