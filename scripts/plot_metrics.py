import csv
import sys

import matplotlib.pyplot as plt


def load(path):
    """Read train and val rows from a metrics.csv."""
    train, val = {"step": [], "loss": [], "l1": [], "kl": []}, {"step": [], "l1": []}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "train":
                train["step"].append(int(row["step"]))
                train["loss"].append(float(row["loss"]))
                train["l1"].append(float(row["l1"]))
                train["kl"].append(float(row["kl"]))
            else:
                val["step"].append(int(row["step"]))
                val["l1"].append(float(row["l1"]))
    return train, val


def main():
    """Plot train loss, L1, KL and val L1 next to the metrics file."""
    path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/act_sorting/metrics.csv"
    train, val = load(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train["step"], train["loss"]); axes[0].set_title("train loss")
    axes[1].plot(train["step"], train["l1"], label="train"); axes[1].plot(val["step"], val["l1"], label="val")
    axes[1].set_title("l1"); axes[1].legend()
    axes[2].plot(train["step"], train["kl"]); axes[2].set_title("kl")
    for ax in axes:
        ax.set_xlabel("step")
    fig.tight_layout()

    out_path = path.rsplit(".", 1)[0] + ".png"
    fig.savefig(out_path, dpi=120)
    print(f"[ok] {out_path}")


if __name__ == "__main__":
    main()
