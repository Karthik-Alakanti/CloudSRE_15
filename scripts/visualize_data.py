#!/usr/bin/env python3
from pathlib import Path
import sys, json, yaml
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def main():
    cfg = yaml.safe_load((ROOT/"configs/experiment.yaml").read_text())
    proc = ROOT/cfg["outputs"]["processed_dir"]
    part = ROOT/cfg["outputs"]["partitions_dir"]
    out = ROOT/"results/figures"
    out.mkdir(parents=True, exist_ok=True)
    y = np.load(proc/"y_train.npy")
    labels = json.loads((proc/"label_encoder.json").read_text())
    inv = {v:k for k,v in labels.items()}

    vals, counts = np.unique(y, return_counts=True)
    names = [inv[int(v)] for v in vals]
    plt.figure(figsize=(11,5))
    plt.bar(names, counts)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Records")
    plt.title("UNSW-NB15 training class distribution")
    plt.tight_layout()
    plt.savefig(out/"global_class_distribution.png", dpi=160)
    plt.close()

    meta = json.loads((part/"partition_metadata.json").read_text())
    classes = sorted({int(k) for client in meta["clients"].values()
                      for split in client["class_counts"].values() for k in split})
    matrix = np.zeros((len(meta["clients"]), len(labels)), dtype=int)
    for i, (_, cm) in enumerate(sorted(meta["clients"].items())):
        total_counts = {}
        for split in ("train","validation","calibration"):
            total_counts.update(cm["class_counts"][split])
        for cls, count in total_counts.items():
            matrix[i, int(cls)] += int(count)
    plt.figure(figsize=(12,6))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Records")
    plt.yticks(range(len(meta["clients"])), [f"Client {i}" for i in range(len(meta["clients"]))])
    plt.xticks(range(len(labels)), [inv[i] for i in range(len(labels))], rotation=35, ha="right")
    plt.title("Non-IID client class distribution")
    plt.tight_layout()
    plt.savefig(out/"client_class_heatmap.png", dpi=160)
    plt.close()
    print(f"Saved figures to {out}")

if __name__ == "__main__":
    main()
