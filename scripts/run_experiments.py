#!/usr/bin/env python3
"""M3 experiment runner: Centralized MLP, FedAvg, and CAA-FedCal calibration.

Experiments:
  A. Centralized MLP (single-node, all training data)
  B. Standard FedAvg (argmax decision)
  C. FedAvg + ordinary pooled calibration (all calibration data pooled centrally)
  D. FedAvg + CAA-FedCal (class-presence-aware weighting)
  E. Weighting ablation (uniform / sample_count / presence)

Resumable: checkpoints saved every round/epoch. Re-running skips completed
seeds/experiments where final artifacts exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.calibration import CAAFedCal, compute_client_histograms, predict_with_thresholds_vectorized
from src.federated.fedavg import FedAvgTrainer
from src.models.centralized import CentralizedTrainer
from src.models.mlp import (IDSMLP, NumpyDataset, average_weights, create_model,
                            evaluate_model)

# ---------------------------------------------------------------------------
# Configuration (locked to M2 artifacts — do not change without approval)
# ---------------------------------------------------------------------------
NUM_CLIENTS = 10
DIRICHLET_ALPHA = 0.3
NUM_CLASSES = 10
INPUT_DIM = 41
BENIGN_CLASS_ID = 2  # "Benign" = 2 in label_encoder.json
SEEDS = [42, 123, 7]  # 3 random seeds

# Training hyperparameters
NUM_ROUNDS = 15  # reduced from 30 for CPU runtime
LOCAL_EPOCHS = 3
BATCH_SIZE = 256
LR = 0.001
CENTRALIZED_EPOCHS = 30  # reduced from 90 for CPU runtime

# Calibration parameters
NUM_BINS = 20
MIN_CLIENTS = 3
MIN_CALIB_RECORDS = 10
ALPHA_BPR = 0.05  # benign FPR budget per class

# Paths
M2_PROCESSED = ROOT / "data/processed"
M2_PARTITIONS = ROOT / "data/partitions"
M3_RESULTS = ROOT / "results/m3"
M3_CHECKPOINTS = M3_RESULTS / "checkpoints"
M3_MODELS = M3_RESULTS / "models"
M3_METRICS = M3_RESULTS / "metrics"
M3_CALIBRATION = M3_RESULTS / "calibration"

for d in (M3_RESULTS, M3_CHECKPOINTS, M3_MODELS, M3_METRICS, M3_CALIBRATION):
    d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Data loading (M2 artifacts, read-only)
# ---------------------------------------------------------------------------
def load_m2_artifacts():
    """Load preprocessed train/test and client partitions from M2."""
    X_train = sparse.load_npz(M2_PROCESSED / "X_train.npz").tocsr().toarray()
    y_train = np.load(M2_PROCESSED / "y_train.npy")
    X_test = sparse.load_npz(M2_PROCESSED / "X_test.npz").tocsr().toarray()
    y_test = np.load(M2_PROCESSED / "y_test.npy")

    label_encoder = json.loads((M2_PROCESSED / "label_encoder.json").read_text())
    inv_label_encoder = {v: k for k, v in label_encoder.items()}

    partition_meta = json.loads(
        (M2_PARTITIONS / "partition_metadata.json").read_text()
    )

    client_data = {}
    for cid in range(NUM_CLIENTS):
        key = f"{cid:02d}"
        cm = partition_meta["clients"][key]
        client_data[cid] = {
            "train_X": sparse.load_npz(
                M2_PARTITIONS / f"client_{key}_train.npz"
            ).tocsr().toarray(),
            "train_y": np.load(M2_PARTITIONS / f"client_{key}_train_y.npy"),
            "validation_X": sparse.load_npz(
                M2_PARTITIONS / f"client_{key}_validation.npz"
            ).tocsr().toarray(),
            "validation_y": np.load(M2_PARTITIONS / f"client_{key}_validation_y.npy"),
            "calibration_X": sparse.load_npz(
                M2_PARTITIONS / f"client_{key}_calibration.npz"
            ).tocsr().toarray(),
            "calibration_y": np.load(M2_PARTITIONS / f"client_{key}_calibration_y.npy"),
            "meta": cm,
        }

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "inv_label_encoder": inv_label_encoder,
        "partition_meta": partition_meta,
        "client_data": client_data,
    }


# ---------------------------------------------------------------------------
# Experiment A: Centralized MLP
# ---------------------------------------------------------------------------
def run_centralized(data, seed, tag=""):
    ckpt_dir = M3_CHECKPOINTS / f"centralized_seed{seed}{tag}"
    trainer = CentralizedTrainer(
        data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        num_classes=NUM_CLASSES, input_dim=INPUT_DIM,
        seed=seed, device=DEVICE,
        checkpoint_dir=ckpt_dir,
    )
    result = trainer.train(epochs=CENTRALIZED_EPOCHS, batch_size=BATCH_SIZE,
                           lr=LR)
    # Evaluate final
    test_loader = DataLoader(
        NumpyDataset(data["X_test"], data["y_test"]),
        batch_size=BATCH_SIZE, shuffle=False)
    metrics = evaluate_model(trainer.model, test_loader,
                             trainer.criterion, torch.device(DEVICE))

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_recall_fscore_support,
                                 confusion_matrix)
    preds = metrics.pop("predictions")
    targets = metrics.pop("targets")
    probs_all = metrics.pop("probabilities")

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    benign_mask = targets == BENIGN_CLASS_ID
    benign_fpr = ((preds[benign_mask] != BENIGN_CLASS_ID).mean()
                  if len(preds[benign_mask]) > 0 else 0.0)
    minority_classes = [c for c in range(NUM_CLASSES) if c != BENIGN_CLASS_ID]
    minority_recall = recall[[c for c in minority_classes]].mean()

    return {
        "experiment": "centralized",
        "seed": seed,
        "history": trainer.history,
        "test_loss": metrics["loss"],
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_benign_fpr": float(benign_fpr),
        "test_minority_recall": float(minority_recall),
        "per_class": {
            str(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "test_predictions": preds.tolist(),
        "test_targets": targets.tolist(),
        "test_probabilities": probs_all.tolist(),
        "model_path": str(M3_MODELS / f"centralized_seed{seed}{tag}.pt"),
    }


# ---------------------------------------------------------------------------
# Experiment B: Standard FedAvg (argmax)
# ---------------------------------------------------------------------------
def run_fedavg(data, seed, tag=""):
    model_path = M3_MODELS / f"fedavg_seed{seed}{tag}.pt"
    if model_path.exists():
        print(f"  FedAvg model already exists at {model_path}, skipping training.")
        # Load and evaluate
        model = create_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                             seed=seed).to(torch.device(DEVICE))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))

        test_loader = DataLoader(
            NumpyDataset(data["X_test"], data["y_test"]),
            batch_size=BATCH_SIZE, shuffle=False)
        metrics = evaluate_model(model, test_loader,
                                 torch.nn.CrossEntropyLoss(),
                                 torch.device(DEVICE))

        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                     f1_score, precision_recall_fscore_support,
                                     confusion_matrix)
        preds = metrics.pop("predictions")
        targets = metrics.pop("targets")
        probs_all = metrics.pop("probabilities")

        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro")
        weighted_f1 = f1_score(targets, preds, average="weighted")
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

        benign_mask = targets == BENIGN_CLASS_ID
        benign_fpr = ((preds[benign_mask] != BENIGN_CLASS_ID).mean()
                      if len(preds[benign_mask]) > 0 else 0.0)
        minority_classes = [c for c in range(NUM_CLASSES) if c != BENIGN_CLASS_ID]
        minority_recall = recall[[c for c in minority_classes]].mean()

        # Need history too - load from checkpoint
        ckpt_dir = M3_CHECKPOINTS / f"fedavg_seed{seed}{tag}"
        history_path = ckpt_dir / "history.json"
        history = []
        if history_path.exists():
            history = json.loads(history_path.read_text())

        return {
            "experiment": "fedavg",
            "seed": seed,
            "history": history,
            "test_loss": float(metrics["loss"]),
            "test_accuracy": float(acc),
            "test_balanced_accuracy": float(bal_acc),
            "test_macro_f1": float(macro_f1),
            "test_weighted_f1": float(weighted_f1),
            "test_benign_fpr": float(benign_fpr),
            "test_minority_recall": float(minority_recall),
            "per_class": {
                str(c): {
                    "precision": float(precision[c]),
                    "recall": float(recall[c]),
                    "f1": float(f1[c]),
                    "support": int(support[c]),
                } for c in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
            "test_predictions": preds.tolist(),
            "test_targets": targets.tolist(),
            "test_probabilities": probs_all.tolist(),
            "model_path": str(model_path),
        }

    ckpt_dir = M3_CHECKPOINTS / f"fedavg_seed{seed}{tag}"
    trainer = FedAvgTrainer(
        data["client_data"], data["X_test"], data["y_test"],
        num_clients=NUM_CLIENTS, num_classes=NUM_CLASSES,
        input_dim=INPUT_DIM, seed=seed, device=DEVICE,
        checkpoint_dir=ckpt_dir,
    )
    result = trainer.train(num_rounds=NUM_ROUNDS, local_epochs=LOCAL_EPOCHS,
                           batch_size=BATCH_SIZE, lr=LR)
    # Save final model
    torch.save(trainer.global_model.state_dict(),
               model_path)

    # Evaluate final with argmax
    test_loader = DataLoader(
        NumpyDataset(data["X_test"], data["y_test"]),
        batch_size=BATCH_SIZE, shuffle=False)
    metrics = evaluate_model(trainer.global_model, test_loader,
                             trainer.criterion, torch.device(DEVICE))

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 f1_score, precision_recall_fscore_support,
                                 confusion_matrix)
    preds = metrics.pop("predictions")
    targets = metrics.pop("targets")
    probs_all = metrics.pop("probabilities")

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    benign_mask = targets == BENIGN_CLASS_ID
    benign_fpr = ((preds[benign_mask] != BENIGN_CLASS_ID).mean()
                  if len(preds[benign_mask]) > 0 else 0.0)
    minority_classes = [c for c in range(NUM_CLASSES) if c != BENIGN_CLASS_ID]
    minority_recall = recall[[c for c in minority_classes]].mean()

    return {
        "experiment": "fedavg",
        "seed": seed,
        "history": trainer.history,
        "test_loss": float(metrics["loss"]),
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_benign_fpr": float(benign_fpr),
        "test_minority_recall": float(minority_recall),
        "per_class": {
            str(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "test_predictions": preds.tolist(),
        "test_targets": targets.tolist(),
        "test_probabilities": probs_all.tolist(),
        "model_path": str(model_path),
    }


# ---------------------------------------------------------------------------
# Experiment C: FedAvg + pooled calibration
# ---------------------------------------------------------------------------
def run_fedavg_pooled_calibration(data, seed, tag=""):
    # First train FedAvg
    fedavg_result = run_fedavg(data, seed, tag=tag)
    model_path = M3_MODELS / f"fedavg_seed{seed}{tag}.pt"

    # Load the trained FedAvg model
    model = create_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                         seed=seed).to(torch.device(DEVICE))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Pool ALL calibration data across clients (centralized calibration)
    pooled_X = []
    pooled_y = []
    for cid in range(NUM_CLIENTS):
        cd = data["client_data"][cid]
        pooled_X.append(cd["calibration_X"])
        pooled_y.append(cd["calibration_y"])
    pooled_X = np.vstack(pooled_X)
    pooled_y = np.concatenate(pooled_y)

    # Fit calibration on pooled data using CAA-FedCal machinery with
    # a single pseudo-client containing all calibration data.
    # "Uniform" weighting over one client = ordinary pooled calibration.
    cal = CAAFedCal(
        num_classes=NUM_CLASSES, num_bins=NUM_BINS,
        min_clients=MIN_CLIENTS, min_calib_records=MIN_CALIB_RECORDS,
        alpha_bpr=ALPHA_BPR, benign_class_id=BENIGN_CLASS_ID,
        weighting="uniform",
    )
    client_calibration = {0: {"X": pooled_X, "y": pooled_y}}
    cal_result = cal.fit(model, client_calibration, device=DEVICE)

    # Save calibration
    cal_path = M3_CALIBRATION / f"pooled_seed{seed}{tag}.json"
    cal.save(cal_path)

    # Evaluate with pooled calibration thresholds (vectorized)
    pred_result = predict_with_thresholds_vectorized(
        model, data["X_test"], cal.thresholds,
        benign_class_id=BENIGN_CLASS_ID,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    preds = pred_result["predictions"]
    probs_all = pred_result["probabilities"]
    targets = data["y_test"]

    # Compute metrics
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score,
                                 precision_recall_fscore_support)

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    benign_mask = targets == BENIGN_CLASS_ID
    benign_fpr = ((preds[benign_mask] != BENIGN_CLASS_ID).mean()
                  if len(preds[benign_mask]) > 0 else 0.0)
    minority_classes = [c for c in range(NUM_CLASSES) if c != BENIGN_CLASS_ID]
    minority_recall = recall[[c for c in minority_classes]].mean()

    torch.save(model.state_dict(),
               M3_MODELS / f"fedavg_pooled_cal_seed{seed}{tag}.pt")

    return {
        "experiment": "fedavg_pooled_calibration",
        "seed": seed,
        "fedavg_history": fedavg_result["history"],
        "calibration": cal_result,
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_benign_fpr": float(benign_fpr),
        "test_minority_recall": float(minority_recall),
        "per_class": {
            str(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "thresholds": cal.thresholds,
        "model_path": str(M3_MODELS / f"fedavg_pooled_cal_seed{seed}{tag}.pt"),
    }


# ---------------------------------------------------------------------------
# Experiment D: FedAvg + CAA-FedCal (presence weighting)
# ---------------------------------------------------------------------------
def run_fedavg_caa_fedcal(data, seed, tag=""):
    # Train FedAvg
    fedavg_result = run_fedavg(data, seed, tag=tag)
    model_path = M3_MODELS / f"fedavg_seed{seed}{tag}.pt"

    model = create_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                         seed=seed).to(torch.device(DEVICE))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Fit CAA-FedCal with presence weighting
    cal = CAAFedCal(
        num_classes=NUM_CLASSES, num_bins=NUM_BINS,
        min_clients=MIN_CLIENTS, min_calib_records=MIN_CALIB_RECORDS,
        alpha_bpr=ALPHA_BPR, benign_class_id=BENIGN_CLASS_ID,
        weighting="presence",
    )
    client_calibration = {
        cid: {"X": data["client_data"][cid]["calibration_X"],
              "y": data["client_data"][cid]["calibration_y"]}
        for cid in range(NUM_CLIENTS)
    }
    cal_result = cal.fit(model, client_calibration, device=DEVICE)

    # Save calibration
    cal_path = M3_CALIBRATION / f"caa_fedcal_seed{seed}{tag}.json"
    cal.save(cal_path)

    # Evaluate with CAA-FedCal thresholds
    cal_result_pred = cal.predict_with_thresholds(
        model, data["X_test"], batch_size=BATCH_SIZE, device=DEVICE)
    preds = cal_result_pred["predictions"]
    probs_all = cal_result_pred["probabilities"]
    targets = data["y_test"]

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix,
                                 f1_score, precision_recall_fscore_support)

    acc = accuracy_score(targets, preds)
    bal_acc = balanced_accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES)))

    benign_mask = targets == BENIGN_CLASS_ID
    benign_preds = preds[benign_mask]
    benign_fpr = (benign_preds != BENIGN_CLASS_ID).mean() if len(benign_preds) > 0 else 0.0

    minority_classes = [c for c in range(NUM_CLASSES) if c != BENIGN_CLASS_ID]
    minority_recall = recall[[c for c in minority_classes]].mean()

    torch.save(model.state_dict(),
               M3_MODELS / f"fedavg_caa_fedcal_seed{seed}{tag}.pt")

    return {
        "experiment": "fedavg_caa_fedcal",
        "seed": seed,
        "fedavg_history": fedavg_result["history"],
        "calibration": cal_result,
        "test_accuracy": float(acc),
        "test_balanced_accuracy": float(bal_acc),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_benign_fpr": float(benign_fpr),
        "test_minority_recall": float(minority_recall),
        "per_class": {
            str(c): {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            } for c in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "model_path": str(M3_MODELS / f"fedavg_caa_fedcal_seed{seed}{tag}.pt"),
    }


# ---------------------------------------------------------------------------
# Experiment E: Weighting ablation
# ---------------------------------------------------------------------------
def run_weighting_ablation(data, seed, tag=""):
    """Run CAA-FedCal with all three weighting schemes on a FedAvg model."""
    model_path = M3_MODELS / f"fedavg_seed{seed}{tag}.pt"
    if not Path(model_path).exists():
        # Train FedAvg first if not already done
        run_fedavg(data, seed, tag=tag)

    model = create_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                         seed=seed).to(torch.device(DEVICE))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    client_calibration = {
        cid: {"X": data["client_data"][cid]["calibration_X"],
              "y": data["client_data"][cid]["calibration_y"]}
        for cid in range(NUM_CLIENTS)
    }

    results = {}
    for weighting in ["uniform", "sample_count", "presence"]:
        cal = CAAFedCal(
            num_classes=NUM_CLASSES, num_bins=NUM_BINS,
            min_clients=MIN_CLIENTS, min_calib_records=MIN_CALIB_RECORDS,
            alpha_bpr=ALPHA_BPR, benign_class_id=BENIGN_CLASS_ID,
            weighting=weighting,
        )
        cal_result = cal.fit(model, client_calibration, device=DEVICE)
        cal.save(M3_CALIBRATION / f"ablation_{weighting}_seed{seed}{tag}.json")

        pred_result = cal.predict_with_thresholds(
            model, data["X_test"], batch_size=BATCH_SIZE, device=DEVICE)
        preds = pred_result["predictions"]
        targets = data["y_test"]

        from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                      f1_score, precision_recall_fscore_support)

        acc = accuracy_score(targets, preds)
        bal_acc = balanced_accuracy_score(targets, preds)
        macro_f1 = f1_score(targets, preds, average="macro")
        weighted_f1 = f1_score(targets, preds, average="weighted")
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
        cm = confusion_matrix_fn(targets, preds)

        benign_mask = targets == BENIGN_CLASS_ID
        benign_fpr = ((preds[benign_mask] != BENIGN_CLASS_ID).mean()
                      if len(preds[benign_mask]) > 0 else 0.0)
        minority_classes = [c for c in range(NUM_CLASSES) if c != BENIGN_CLASS_ID]
        minority_recall = recall[[c for c in minority_classes]].mean()

        results[weighting] = {
            "test_accuracy": float(acc),
            "test_balanced_accuracy": float(bal_acc),
            "test_macro_f1": float(macro_f1),
            "test_weighted_f1": float(weighted_f1),
            "test_benign_fpr": float(benign_fpr),
            "test_minority_recall": float(minority_recall),
            "per_class": {
                str(c): {
                    "precision": float(precision[c]),
                    "recall": float(recall[c]),
                    "f1": float(f1[c]),
                    "support": int(support[c]),
                } for c in range(NUM_CLASSES)
            },
            "confusion_matrix": cm.tolist(),
            "thresholds": cal.thresholds,
        }

    return {
        "experiment": "weighting_ablation",
        "seed": seed,
        "results": results,
    }


def confusion_matrix_fn(targets, preds, labels=None):
    from sklearn.metrics import confusion_matrix
    if labels is None:
        labels = list(range(NUM_CLASSES))
    return confusion_matrix(targets, preds, labels=labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run M3 experiments")
    parser.add_argument("--experiment", choices=["all", "centralized", "fedavg",
                        "pooled", "caa_fedcal", "ablation"], default="all")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only one seed (default: all seeds)")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag for output files")
    args = parser.parse_args()

    data = load_m2_artifacts()
    print(f"Loaded M2 artifacts: train={len(data['X_train'])}, "
          f"test={len(data['X_test'])}")
    print(f"Classes: {data['label_encoder']}")

    seeds = [args.seed] if args.seed is not None else SEEDS
    tag = args.tag

    # Load existing results if any (for accumulation across experiment calls)
    results_path = M3_METRICS / f"all_results{tag}.json"
    all_results = []
    if results_path.exists():
        try:
            all_results = json.loads(results_path.read_text())
        except Exception:
            all_results = []

    # Track which (experiment, seed) combos are already done
    done_keys = {(e["experiment"], e["seed"]) for e in all_results}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")

        if args.experiment in ("all", "centralized"):
            if ("centralized", seed) in done_keys:
                print(f"  Centralized seed {seed} already done, skipping.")
            else:
                print("\n--- Experiment A: Centralized MLP ---")
                r = run_centralized(data, seed, tag=tag)
                all_results.append(r)
                done_keys.add(("centralized", seed))
                print(f"  test_acc={r['test_accuracy']:.4f}, "
                      f"macro_f1={r['test_macro_f1']:.4f}")

        if args.experiment in ("all", "fedavg"):
            if ("fedavg", seed) in done_keys:
                print(f"  FedAvg seed {seed} already done, skipping.")
            else:
                print("\n--- Experiment B: Standard FedAvg ---")
                r = run_fedavg(data, seed, tag=tag)
                all_results.append(r)
                done_keys.add(("fedavg", seed))
                print(f"  test_acc={r['test_accuracy']:.4f}")

        if args.experiment in ("all", "pooled"):
            if ("fedavg_pooled_calibration", seed) in done_keys:
                print(f"  Pooled calibration seed {seed} already done, skipping.")
            else:
                print("\n--- Experiment C: FedAvg + Pooled Calibration ---")
                r = run_fedavg_pooled_calibration(data, seed, tag=tag)
                all_results.append(r)
                done_keys.add(("fedavg_pooled_calibration", seed))
                print(f"  test_acc={r['test_accuracy']:.4f}, "
                      f"macro_f1={r['test_macro_f1']:.4f}, "
                      f"benign_fpr={r['test_benign_fpr']:.4f}")

        if args.experiment in ("all", "caa_fedcal"):
            if ("fedavg_caa_fedcal", seed) in done_keys:
                print(f"  CAA-FedCal seed {seed} already done, skipping.")
            else:
                print("\n--- Experiment D: FedAvg + CAA-FedCal ---")
                r = run_fedavg_caa_fedcal(data, seed, tag=tag)
                all_results.append(r)
                done_keys.add(("fedavg_caa_fedcal", seed))
                print(f"  test_acc={r['test_accuracy']:.4f}, "
                      f"macro_f1={r['test_macro_f1']:.4f}, "
                      f"benign_fpr={r['test_benign_fpr']:.4f}")

        if args.experiment in ("all", "ablation"):
            if ("weighting_ablation", seed) in done_keys:
                print(f"  Weighting ablation seed {seed} already done, skipping.")
            else:
                print("\n--- Experiment E: Weighting Ablation ---")
                r = run_weighting_ablation(data, seed, tag=tag)
                all_results.append(r)
                done_keys.add(("weighting_ablation", seed))
                for w, res in r["results"].items():
                    print(f"  {w}: acc={res['test_accuracy']:.4f}, "
                          f"macro_f1={res['test_macro_f1']:.4f}, "
                          f"benign_fpr={res['test_benign_fpr']:.4f}")

        # Save after each seed
        results_path.write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
