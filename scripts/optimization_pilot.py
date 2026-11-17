#!/usr/bin/env python3
"""M3 Optimization Pilot — cheap improvements to Centralized MLP and FedAvg.

Preserves all M3 artifacts. Saves new experiments under results/optimization/.
Uses validation data (pooled client validation sets) for model selection —
never test data.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.mlp import (
    IDSMLP, NumpyDataset, create_model, evaluate_model, save_model,
    train_epoch,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
M2_PROCESSED = ROOT / "data/processed"
M2_PARTITIONS = ROOT / "data/partitions"
OPT_RESULTS = ROOT / "results/optimization"
OPT_CHECKPOINTS = OPT_RESULTS / "checkpoints"
OPT_MODELS = OPT_RESULTS / "models"
OPT_METRICS = OPT_RESULTS / "metrics"

for d in (OPT_RESULTS, OPT_CHECKPOINTS, OPT_MODELS, OPT_METRICS):
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 10
INPUT_DIM = 41
BENIGN_CLASS_ID = 2
BATCH_SIZE = 256
DEVICE = "cpu"
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_m2_artifacts():
    X_train = sparse.load_npz(M2_PROCESSED / "X_train.npz").tocsr().toarray()
    y_train = np.load(M2_PROCESSED / "y_train.npy")
    X_test = sparse.load_npz(M2_PROCESSED / "X_test.npz").tocsr().toarray()
    y_test = np.load(M2_PROCESSED / "y_test.npy")
    return X_train, y_train, X_test, y_test


def load_validation_data():
    """Pool all client validation sets for model selection."""
    partition_meta = json.loads(
        (M2_PARTITIONS / "partition_metadata.json").read_text()
    )
    val_X_list, val_y_list = [], []
    for cid in range(10):
        key = f"{cid:02d}"
        val_X_list.append(
            sparse.load_npz(
                M2_PARTITIONS / f"client_{key}_validation.npz"
            ).tocsr().toarray()
        )
        val_y_list.append(
            np.load(M2_PARTITIONS / f"client_{key}_validation_y.npy")
        )
    val_X = np.vstack(val_X_list)
    val_y = np.concatenate(val_y_list)
    return val_X, val_y


def compute_class_weights(y_train, num_classes=NUM_CLASSES):
    """Inverse class frequency weights for CrossEntropyLoss."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1  # avoid division by zero
    weights = len(y_train) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_all_metrics(targets, preds, num_classes=NUM_CLASSES,
                        benign_class_id=BENIGN_CLASS_ID):
    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(num_classes)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    benign_mask = targets == benign_class_id
    benign_fpr = ((preds[benign_mask] != benign_class_id).mean()
                  if len(preds[benign_mask]) > 0 else 0.0)
    minority_classes = [c for c in range(num_classes) if c != benign_class_id]
    minority_recall = recall[[c for c in minority_classes]].mean()

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "benign_fpr": float(benign_fpr),
        "minority_recall": float(minority_recall),
        "per_class": {
            str(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            } for c in range(num_classes)
        },
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(targets)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_centralized(X_train, y_train, val_X, val_y, X_test, y_test,
                      hidden1=128, hidden2=64, epochs=30, lr=0.001,
                      batch_size=256, seed=SEED, use_class_weights=False,
                      tag=""):
    """Train centralized MLP with validation-based model selection."""
    device = torch.device(DEVICE)
    model = create_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                         seed=seed).to(device)

    if use_class_weights:
        class_weights = compute_class_weights(y_train).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(
        NumpyDataset(val_X, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        NumpyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_state = None

    start_time = time.time()
    for epoch in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        # Compute validation macro-F1
        val_preds = val_metrics.pop("predictions")
        val_targets = val_metrics.pop("targets")
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro")

        record = {
            "epoch": epoch,
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_macro_f1": float(val_macro_f1),
        }
        history.append(record)

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train_acc={train_metrics['accuracy']:.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_macro_f1={val_macro_f1:.4f}")

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Best val macro-F1: {best_val_macro_f1:.4f} at epoch {best_epoch+1}")

    # Restore best model
    model.load_state_dict(best_state)

    # Final evaluation on test set (untouched until now)
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    test_preds = test_metrics.pop("predictions")
    test_targets = test_metrics.pop("targets")
    test_probs = test_metrics.pop("probabilities")

    metrics = compute_all_metrics(test_targets, test_preds)
    metrics["val_macro_f1_at_selection"] = float(best_val_macro_f1)
    metrics["best_epoch"] = int(best_epoch)
    metrics["training_time_sec"] = float(elapsed)

    # Save model and metrics
    model_tag = f"opt_{tag}_seed{seed}"
    save_model(model, OPT_MODELS / f"{model_tag}.pt")

    result = {
        "experiment": f"centralized_opt_{tag}",
        "seed": seed,
        "config": {
            "hidden1": hidden1,
            "hidden2": hidden2,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "use_class_weights": use_class_weights,
            "architecture": f"{hidden1}->{hidden2}",
        },
        "metrics": metrics,
        "model_path": str(OPT_MODELS / f"{model_tag}.pt"),
    }
    (OPT_METRICS / f"{model_tag}.json").write_text(
        json.dumps(result, indent=2))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run optimization pilots")
    parser.add_argument("--config", type=str, default="all",
                        help="Config to run: baseline, more_epochs, wider_mlp, lower_lr, class_weighted, all")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed")
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_m2_artifacts()
    val_X, val_y = load_validation_data()
    print(f"Loaded: train={len(X_train)}, val={len(val_X)}, test={len(X_test)}")
    print(f"Class distribution (train): {np.bincount(y_train).tolist()}")
    print(f"Class distribution (val): {np.bincount(val_y).tolist()}")
    print(f"Class distribution (test): {np.bincount(y_test).tolist()}")
    print()

    results = []
    configs = []

    if args.config in ("baseline", "all"):
        configs.append(("baseline", dict(hidden1=128, hidden2=64, epochs=30,
                                         lr=0.001, use_class_weights=False)))
    if args.config in ("more_epochs", "all"):
        configs.append(("more_epochs", dict(hidden1=128, hidden2=64, epochs=60,
                                            lr=0.001, use_class_weights=False)))
    if args.config in ("wider_mlp", "all"):
        configs.append(("wider_mlp", dict(hidden1=256, hidden2=128, epochs=30,
                                          lr=0.001, use_class_weights=False)))
    if args.config in ("lower_lr", "all"):
        configs.append(("lower_lr", dict(hidden1=128, hidden2=64, epochs=30,
                                         lr=0.0005, use_class_weights=False)))
    if args.config in ("class_weighted", "all"):
        configs.append(("class_weighted", dict(hidden1=128, hidden2=64, epochs=30,
                                               lr=0.001, use_class_weights=True)))

    for tag, cfg in configs:
        print(f"\n{'='*60}")
        print(f"Config: {tag} — {cfg}")
        print(f"{'='*60}")
        r = train_centralized(
            X_train, y_train, val_X, val_y, X_test, y_test,
            seed=args.seed, tag=tag, **cfg)
        results.append(r)
        print(f"  RESULT: acc={r['metrics']['accuracy']:.4f}, "
              f"macro_f1={r['metrics']['macro_f1']:.4f}, "
              f"benign_fpr={r['metrics']['benign_fpr']:.6f}, "
              f"minority_recall={r['metrics']['minority_recall']:.4f}")

    # Save aggregate
    (OPT_METRICS / f"pilot_results_seed{args.seed}.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nAll pilot results saved to results/optimization/metrics/")


if __name__ == "__main__":
    main()
