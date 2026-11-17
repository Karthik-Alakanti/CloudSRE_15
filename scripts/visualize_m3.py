#!/usr/bin/env python3
"""M3 visualizations: class distribution, client heatmap, convergence, per-class F1."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
M3_RESULTS = ROOT / "results/m3"
M3_METRICS = M3_RESULTS / "metrics"
M3_FIGS = M3_RESULTS / "figures"
M3_FIGS.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Analysis", "Backdoor", "Benign", "DoS", "Exploits",
    "Fuzzers", "Generic", "Reconnaissance", "Shellcode", "Worms",
]


def load_results(tag="_s42"):
    path = M3_METRICS / f"all_results{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    return []


def plot_global_class_distribution():
    """Global class distribution across train and test splits."""
    import pandas as pd
    import yaml

    config_path = ROOT / "configs/experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    ds = config["dataset"]

    # Load raw data to get class counts
    raw_path = ROOT / "data/raw/unsw-nb15" / ds["data_file"]
    df = pd.read_csv(raw_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    from sklearn.model_selection import train_test_split
    y_raw = df["Attack"].astype(str)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y_raw)

    train_counts = df.loc[train_idx, "Attack"].value_counts().sort_index()
    test_counts = df.loc[test_idx, "Attack"].value_counts().sort_index()

    classes = CLASS_NAMES
    train_vals = [int(train_counts.get(c, 0)) for c in classes]
    test_vals = [int(test_counts.get(c, 0)) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_vals, width, label="Train", alpha=0.8)
    ax.bar(x + width/2, test_vals, width, label="Test", alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Records")
    ax.set_title("NF-UNSW-NB15-v2: Class Distribution (Train vs Test)")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(M3_FIGS / "global_class_distribution.png", dpi=150)
    plt.close()


def plot_client_class_heatmap():
    """Client-by-class heatmap of calibration class presence."""
    import yaml

    config_path = ROOT / "configs/experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    partition_meta = json.loads(
        (ROOT / "data/partitions" / "partition_metadata.json").read_text()
    )

    clients = sorted(partition_meta["clients"].keys())
    classes = list(range(10))

    # Build presence matrix: client x class (1 if class present in calibration)
    presence = np.zeros((len(clients), len(classes)))
    for i, cid in enumerate(clients):
        cm = partition_meta["clients"][cid]
        cal_counts = cm["class_counts"]["calibration"]
        for cls in classes:
            if str(cls) in cal_counts and cal_counts[str(cls)] > 0:
                presence[i, cls] = 1

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        presence,
        ax=ax,
        cmap="Blues",
        cbar_kws={"label": "Class Present in Calibration"},
        xticklabels=CLASS_NAMES,
        yticklabels=[f"Client {c}" for c in clients],
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_title("CAA-FedCal: Client Class Presence in Calibration Sets\n"
                 "(Dirichlet α=0.3, 10 clients)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(M3_FIGS / "client_class_heatmap.png", dpi=150)
    plt.close()


def plot_convergence():
    """Training convergence: FedAvg rounds and Centralized epochs."""
    results = load_results("_s42")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # FedAvg convergence
    fedavg_runs = [r for r in results if r["experiment"] == "fedavg"]
    if fedavg_runs:
        ax = axes[0]
        for run in fedavg_runs:
            history = run["history"]
            rounds = [h["round"] + 1 for h in history]
            accs = [h["test_accuracy"] for h in history]
            ax.plot(rounds, accs, marker="o", label=f"Seed {run['seed']}")
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("FedAvg: Test Accuracy per Round")
        ax.legend()
        ax.grid(alpha=0.3)

    # Centralized convergence
    central_runs = [r for r in results if r["experiment"] == "centralized"]
    if central_runs:
        ax = axes[1]
        for run in central_runs:
            history = run["history"]
            epochs = [h["epoch"] + 1 for h in history]
            accs = [h["test_accuracy"] for h in history]
            ax.plot(epochs, accs, marker="o", label=f"Seed {run['seed']}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Centralized MLP: Test Accuracy per Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(M3_FIGS / "convergence.png", dpi=150)
    plt.close()


def plot_per_class_f1_comparison():
    """Per-class F1 comparison across experiments (seed 42)."""
    results = load_results("_s42")

    # Collect per-class F1 for each experiment
    experiments = {}
    for run in results:
        exp = run["experiment"]
        if exp == "weighting_ablation":
            continue  # Skip ablation (same as CAA-FedCal)
        if "per_class" in run:
            experiments[exp] = run["per_class"]

    classes = [str(c) for c in range(10)]
    exp_names = list(experiments.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(classes))
    width = 0.2
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(exp_names))

    for i, exp_name in enumerate(exp_names):
        f1s = [experiments[exp_name][c]["f1"] for c in classes]
        ax.bar(x + offsets[i], f1s, width, label=exp_name.replace("_", " ").title(),
               alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison (Seed 42)")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(M3_FIGS / "per_class_f1_comparison.png", dpi=150)
    plt.close()


def plot_metric_comparison():
    """Bar chart comparing key metrics across experiments."""
    results = load_results("_s42")

    # Collect summary metrics
    summary = []
    for run in results:
        if run["experiment"] == "weighting_ablation":
            continue
        summary.append({
            "Experiment": run["experiment"].replace("_", " ").title(),
            "Seed": run["seed"],
            "Accuracy": run.get("test_accuracy", 0),
            "Macro-F1": run.get("test_macro_f1", 0),
            "Weighted-F1": run.get("test_weighted_f1", 0),
            "Benign FPR": run.get("test_benign_fpr", 0),
        })

    import pandas as pd
    df = pd.DataFrame(summary)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ["Accuracy", "Macro-F1", "Weighted-F1", "Benign FPR"]
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        sns.barplot(data=df, x="Experiment", y=metric, ax=ax, hue="Seed",
                    palette="Set2")
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        if metric == "Benign FPR":
            ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(M3_FIGS / "metric_comparison.png", dpi=150)
    plt.close()


def main():
    print("Generating visualizations...")
    plot_global_class_distribution()
    print("  - global_class_distribution.png")
    plot_client_class_heatmap()
    print("  - client_class_heatmap.png")
    plot_convergence()
    print("  - convergence.png")
    plot_per_class_f1_comparison()
    print("  - per_class_f1_comparison.png")
    plot_metric_comparison()
    print("  - metric_comparison.png")
    print(f"\nAll figures saved to {M3_FIGS}")


if __name__ == "__main__":
    main()
