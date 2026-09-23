import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.teleop_dataset import action_vector, engaged_mask, proprio_vector, read_split, vector_dim
from dataset.transforms import denormalize, normalize
from models.act.act import ACT

M_ENSEMBLE = 0.01


def load_model(ckpt_path, device):
    """Load an ACT checkpoint and its dataset config."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    dcfg, mcfg = ckpt["dataset_config"], ckpt["model_config"]
    n_cameras = len(dcfg["cameras"]["use"])
    action_dim = vector_dim(dcfg)
    model = ACT(n_cameras=n_cameras, proprio_dim=action_dim, action_dim=action_dim,
                chunk_size=dcfg["action"]["chunk_size"], **mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dcfg


def load_stats(dcfg):
    """Load normalization stats named in the dataset config."""
    npz = np.load(dcfg["normalize"]["stats_file"])
    return {k: npz[k] for k in npz.files}


def load_image(f, cam, t, hw):
    """Camera frame at t, resized and scaled to [0, 1]."""
    img = f[f"observations/images/{cam}"][t]
    if img.shape[:2] != tuple(hw):
        import cv2
        img = cv2.resize(img, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def episode_path(dcfg, episode_id):
    """Path to an episode's hdf5 file."""
    return os.path.join(dcfg["data"]["store_root"], str(episode_id).zfill(3), dcfg["data"]["episode_file"])


def open_loop_replay(f, model, dcfg, stats, device, stride=1):
    """Run the model over engaged frames and temporally ensemble the chunks."""
    arms = dcfg["action"]["arms"]
    cams, hw = dcfg["cameras"]["use"], dcfg["cameras"]["resize_to"]
    K = dcfg["action"]["chunk_size"]
    action_dim = vector_dim(dcfg)

    engaged = engaged_mask(f, arms, dcfg["state"]["train_states"])
    t0, t1 = int(np.where(engaged)[0].min()), int(np.where(engaged)[0].max())

    logged_cmd = action_vector(f, dcfg, slice(t0, t1))

    pending = {t: [] for t in range(t0, t1 + K)}
    with torch.no_grad():
        for t in range(t0, t1, stride):
            images = np.stack([load_image(f, cam, t, hw) for cam in cams])
            images = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).float().to(device)
            proprio = proprio_vector(f, dcfg, slice(t, t + 1))[0]
            proprio_n = normalize(proprio, stats["proprio_mean"], stats["proprio_std"])
            proprio_t = torch.from_numpy(proprio_n).unsqueeze(0).float().to(device)

            a_hat, _, _ = model(images, proprio_t)
            a_hat = denormalize(a_hat[0].cpu().numpy(), stats["action_mean"], stats["action_std"])
            for i in range(K):
                if t + i in pending:
                    pending[t + i].append(a_hat[i])

    ensembled = np.zeros((t1 - t0, action_dim), dtype=np.float32)
    for t in range(t0, t1):
        preds = pending[t]
        if not preds:
            ensembled[t - t0] = ensembled[t - t0 - 1] if t > t0 else 0.0
            continue
        preds = np.stack(preds)
        ages = np.arange(len(preds))[::-1]
        w = np.exp(-M_ENSEMBLE * ages)
        w /= w.sum()
        ensembled[t - t0] = (w[:, None] * preds).sum(axis=0)

    return t0, t1, logged_cmd, ensembled


def main():
    """Open-loop L1 against logged commands over every episode in a split."""
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/act_sorting/best.pt"
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
    ax.set_title(f"open-loop: {split} split")
    ax.legend()
    plt.xticks(rotation=90)
    fig.tight_layout()
    out_path = f"open_loop_eval_{split}.png"
    fig.savefig(out_path, dpi=120)
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
