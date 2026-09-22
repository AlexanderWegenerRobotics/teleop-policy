import csv
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.teleop_dataset import EpisodicDataset, load_cfg, read_split
from models.act.act import ACT, act_loss


def build_loader(dcfg, split, stats, batch_size, num_workers, shuffle):
    """DataLoader over one split."""
    ids = read_split(os.path.join(dcfg["data"]["splits"], f"{split}.txt"))
    ds = EpisodicDataset(ids, dcfg, stats)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle,
                      pin_memory=True, persistent_workers=num_workers > 0)


def main():
    """Train ACT on L1 + beta * KL over the action chunk."""
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/act_sorting.yaml"
    cfg = load_cfg(cfg_path)
    dcfg = load_cfg(cfg["dataset_config"])
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    torch.manual_seed(tcfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats_path = dcfg["normalize"]["stats_file"]
    if not os.path.exists(stats_path):
        sys.exit(f"missing {stats_path}, run scripts/compute_stats.py first")
    npz = np.load(stats_path)
    stats = {k: npz[k] for k in npz.files}

    train_loader = build_loader(dcfg, "train", stats, tcfg["batch_size"], tcfg["num_workers"], shuffle=True)
    val_loader = build_loader(dcfg, "val", stats, tcfg["batch_size"], tcfg["num_workers"], shuffle=False)

    n_cameras = len(dcfg["cameras"]["use"])
    action_dim = dcfg["action"]["dims_per_arm"] * len(dcfg["action"]["arms"])
    model = ACT(n_cameras=n_cameras, proprio_dim=action_dim, action_dim=action_dim,
                chunk_size=dcfg["action"]["chunk_size"], **mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.1f}M  train episodes: {len(train_loader.dataset)}  "
          f"device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    os.makedirs(tcfg["ckpt_dir"], exist_ok=True)

    metrics_path = os.path.join(tcfg["ckpt_dir"], "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    metrics_file = open(metrics_path, "a", newline="")
    metrics = csv.writer(metrics_file)
    if write_header:
        metrics.writerow(["elapsed_sec", "epoch", "step", "split", "loss", "l1", "kl"])

    step = 0
    epoch = 0
    start_time = time.time()

    def save_ckpt(path, include_optim):
        """Save model, configs and optionally optimizer state."""
        payload = {"model": model.state_dict(), "epoch": epoch, "step": step,
                   "elapsed_sec": time.time() - start_time,
                   "dataset_config": dcfg, "model_config": mcfg}
        if include_optim:
            payload["optimizer"] = opt.state_dict()
        try:
            torch.save(payload, path)
            print(f"[ok] {path}")
            return True
        except (RuntimeError, OSError) as e:
            print(f"[warn] checkpoint save failed ({path}): {e}")
            return False

    ckpt_history = []
    keep_last = tcfg.get("ckpt_keep_last", 3)
    best_val = float("inf")

    buf_loss, buf_l1, buf_kl, buf_meta = [], [], [], []

    def flush_buffer():
        """Write buffered train losses to metrics.csv and print the latest."""
        if not buf_meta:
            return
        loss_vals = torch.stack(buf_loss).tolist()
        l1_vals = torch.stack(buf_l1).tolist()
        kl_vals = torch.stack(buf_kl).tolist()
        for (e, ep, st), lv, l1v, klv in zip(buf_meta, loss_vals, l1_vals, kl_vals):
            metrics.writerow([f"{e:.1f}", ep, st, "train", f"{lv:.6f}", f"{l1v:.6f}", f"{klv:.6f}"])
        metrics_file.flush()
        print(f"epoch {buf_meta[-1][1]} step {buf_meta[-1][2]}  loss {loss_vals[-1]:.4f}  "
              f"l1 {l1_vals[-1]:.4f}  kl {kl_vals[-1]:.4f}  elapsed {buf_meta[-1][0] / 60:.1f}min")
        buf_loss.clear(); buf_l1.clear(); buf_kl.clear(); buf_meta.clear()

    try:
        for epoch in range(tcfg["epochs"]):
            model.train()
            for batch in train_loader:
                images = batch["images"].to(device, non_blocking=True)
                proprio = batch["proprio"].to(device, non_blocking=True)
                action = batch["action"].to(device, non_blocking=True)
                is_pad = batch["is_pad"].to(device, non_blocking=True)

                a_hat, mu, logvar = model(images, proprio, action, is_pad)
                loss, l1, kl = act_loss(a_hat, action, is_pad, mu, logvar, tcfg["kl_weight"])

                opt.zero_grad()
                loss.backward()
                opt.step()

                buf_loss.append(loss.detach()); buf_l1.append(l1.detach()); buf_kl.append(kl.detach())
                buf_meta.append((time.time() - start_time, epoch, step))
                if step % tcfg["log_every"] == 0:
                    flush_buffer()
                step += 1

            if epoch % tcfg["ckpt_every"] == 0 or epoch == tcfg["epochs"] - 1:
                flush_buffer()
                elapsed = time.time() - start_time
                val_l1 = evaluate(model, val_loader, device)
                metrics.writerow([f"{elapsed:.1f}", epoch, step, "val", "", f"{val_l1:.6f}", ""])
                metrics_file.flush()
                print(f"epoch {epoch}  val_l1 {val_l1:.4f}  elapsed {elapsed / 60:.1f}min")

                milestone = os.path.join(tcfg["ckpt_dir"], f"epoch_{epoch}.pt")
                if save_ckpt(milestone, include_optim=False):
                    ckpt_history.append(milestone)
                    while len(ckpt_history) > keep_last:
                        old = ckpt_history.pop(0)
                        try:
                            os.remove(old)
                        except OSError:
                            pass

                save_ckpt(os.path.join(tcfg["ckpt_dir"], "latest.pt"), include_optim=True)

                if val_l1 < best_val:
                    best_val = val_l1
                    save_ckpt(os.path.join(tcfg["ckpt_dir"], "best.pt"), include_optim=False)
    except KeyboardInterrupt:
        flush_buffer()
        print(f"\n[interrupt] epoch {epoch} step {step}, saving checkpoint")
        save_ckpt(os.path.join(tcfg["ckpt_dir"], "interrupt.pt"), include_optim=True)
    finally:
        flush_buffer()
        metrics_file.close()


@torch.no_grad()
def evaluate(model, loader, device):
    """Mean L1 over a loader."""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        images = batch["images"].to(device)
        proprio = batch["proprio"].to(device)
        action = batch["action"].to(device)
        is_pad = batch["is_pad"].to(device)
        a_hat, mu, logvar = model(images, proprio, action, is_pad)
        _, l1, _ = act_loss(a_hat, action, is_pad, mu, logvar, kl_weight=0.0)
        total += l1.item()
        n += 1
    return total / max(n, 1)


if __name__ == "__main__":
    main()
