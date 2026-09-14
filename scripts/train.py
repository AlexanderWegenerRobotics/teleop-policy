# Train ACT: L1 + beta*KL on the action chunk. Same script for the Stage 2
# 10-episode overfit smoke test (set train.n_train_episodes in the config)
# and Stage 4 full training (n_train_episodes: null).
#
# Usage: python scripts/train.py configs/act_sorting.yaml

import copy
import csv
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataset.teleop_dataset import EpisodicDataset, load_cfg, read_split
from models.act.act import ACT, act_loss


def build_loader(cfg, dcfg, split, n_episodes, stats, batch_size, num_workers, shuffle):
    ids = read_split(os.path.join(dcfg["data"]["splits"], f"{split}.txt"))
    if n_episodes:
        ids = ids[:n_episodes]
    ds = EpisodicDataset(ids, dcfg, stats)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=shuffle,
                       pin_memory=True, persistent_workers=num_workers > 0)


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/act_sorting.yaml"
    cfg = load_cfg(cfg_path)
    dcfg = load_cfg(cfg["dataset_config"])
    # Optional per-experiment overrides into the dataset config, so varying one
    # knob (e.g. sampling.grasp_oversample) doesn't need a duplicated copy of
    # dataset.yaml that then drifts out of sync with the original. One level of
    # nesting, which covers every block in dataset.yaml.
    for section, values in (cfg.get("dataset_overrides") or {}).items():
        if isinstance(values, dict):
            dcfg.setdefault(section, {}).update(values)
        else:
            dcfg[section] = values
    if cfg.get("dataset_overrides"):
        print(f"[train] dataset_overrides applied: {cfg['dataset_overrides']}")
    tcfg = cfg["train"]
    mcfg = cfg["model"]

    torch.manual_seed(tcfg["seed"])
    torch.backends.cudnn.benchmark = True  # fixed input size -> let cudnn pick fast kernels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stats = None
    stats_path = dcfg["normalize"]["stats_file"]
    if os.path.exists(stats_path):
        import numpy as np
        npz = np.load(stats_path)
        stats = {k: npz[k] for k in npz.files}
    else:
        sys.exit(f"missing {stats_path} — run scripts/compute_stats.py first")

    n_ep = tcfg.get("n_train_episodes")
    train_loader = build_loader(cfg, dcfg, "train", n_ep, stats,
                                 tcfg["batch_size"], tcfg["num_workers"], shuffle=True)
    val_loader = build_loader(cfg, dcfg, "val", None, stats,
                               tcfg["batch_size"], tcfg["num_workers"], shuffle=False)

    # A SECOND val loader that samples (almost) only the approach-to-grasp
    # window. Overall val L1 averages 20 action dims over every timestep, and
    # the grasp pose is 1.65% of engaged frames -- so it is dominated by
    # transit and transport and can move the wrong way relative to task
    # success. Observed directly: a 600-epoch run with grasp oversampling
    # scored WORSE on val L1 than the baseline (0.178 vs 0.169) while behaving
    # visibly better on the robot (decisive approach, fingers actually
    # closing). Selecting checkpoints on val L1 alone would have thrown that
    # model away. This gives a number that tracks the thing being optimised
    # for, cheaply, without a sim rollout per checkpoint.
    #
    # Same episodes, same normalisation -- only the sampling distribution
    # differs, so l1_grasp and l1 are directly comparable.
    dcfg_grasp = copy.deepcopy(dcfg)
    dcfg_grasp.setdefault("sampling", {})
    dcfg_grasp["sampling"] = dict(dcfg_grasp["sampling"])
    dcfg_grasp["sampling"]["grasp_oversample"] = 1e6  # effectively grasp-window only
    val_grasp_loader = build_loader(cfg, dcfg_grasp, "val", None, stats,
                                     tcfg["batch_size"], tcfg["num_workers"], shuffle=False)

    n_cameras = len(dcfg["cameras"]["use"])
    action_dim = dcfg["action"]["dims_per_arm"] * len(dcfg["action"]["arms"])
    model = ACT(n_cameras=n_cameras, proprio_dim=action_dim, action_dim=action_dim,
                chunk_size=dcfg["action"]["chunk_size"], **mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.1f}M  train episodes: {len(train_loader.dataset)}  "
          f"device: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"])
    os.makedirs(tcfg["ckpt_dir"], exist_ok=True)

    # one row per train step + one per val eval, appended live so a killed run keeps its history
    metrics_path = os.path.join(tcfg["ckpt_dir"], "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    metrics_file = open(metrics_path, "a", newline="")
    metrics = csv.writer(metrics_file)
    if write_header:
        metrics.writerow(["elapsed_sec", "epoch", "step", "split", "loss", "l1", "kl", "l1_grasp"])

    step = 0
    epoch = 0
    start_time = time.time()

    # optimizer state is ~2x the model size (AdamW keeps two extra tensors per param) — only
    # worth it on latest.pt (used to resume). Milestone/best snapshots are model-only.
    def save_ckpt(path, include_optim):
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
            print(f"[warn] checkpoint save failed ({path}): {e} — training continues")
            return False

    ckpt_history = []
    keep_last = tcfg.get("ckpt_keep_last", 3)
    best_val = float("inf")
    best_val_grasp = float("inf")

    # buffered so .item() (a GPU sync point) only runs once every log_every steps instead of
    # every step — without this the CPU stalls on the GPU each step and nothing overlaps
    buf_loss, buf_l1, buf_kl, buf_meta = [], [], [], []

    def flush_buffer():
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
                # Grasp-window val L1 -- the selection metric. See the
                # val_grasp_loader comment for why plain val_l1 is not.
                val_l1_grasp = evaluate(model, val_grasp_loader, device)
                metrics.writerow([f"{elapsed:.1f}", epoch, step, "val", "",
                                  f"{val_l1:.6f}", "", f"{val_l1_grasp:.6f}"])
                metrics_file.flush()
                print(f"epoch {epoch}  val_l1 {val_l1:.4f}  val_l1_grasp {val_l1_grasp:.4f}  "
                      f"elapsed {elapsed / 60:.1f}min")

                milestone = os.path.join(tcfg["ckpt_dir"], f"epoch_{epoch}.pt")
                if save_ckpt(milestone, include_optim=False):
                    ckpt_history.append(milestone)
                    while len(ckpt_history) > keep_last:
                        old = ckpt_history.pop(0)
                        try:
                            os.remove(old)
                            print(f"[cleanup] removed {old}")
                        except OSError:
                            pass

                save_ckpt(os.path.join(tcfg["ckpt_dir"], "latest.pt"), include_optim=True)

                # best.pt is selected on whichever metric select_on names.
                # Default stays "l1" so existing configs behave identically,
                # but "l1_grasp" is the one that tracks task success -- see the
                # val_grasp_loader comment. best_grasp.pt is always written so
                # both candidates survive and can be compared in the sim
                # without paying for a second training run.
                if val_l1 < best_val:
                    best_val = val_l1
                    save_ckpt(os.path.join(tcfg["ckpt_dir"], "best.pt"), include_optim=False)
                if val_l1_grasp < best_val_grasp:
                    best_val_grasp = val_l1_grasp
                    save_ckpt(os.path.join(tcfg["ckpt_dir"], "best_grasp.pt"), include_optim=False)
    except KeyboardInterrupt:
        flush_buffer()
        print(f"\n[interrupt] Ctrl-C at epoch {epoch} step {step} — saving checkpoint")
        save_ckpt(os.path.join(tcfg["ckpt_dir"], "interrupt.pt"), include_optim=True)
    finally:
        flush_buffer()
        metrics_file.close()


@torch.no_grad()
def evaluate(model, loader, device):
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
