#!/usr/bin/env python3
"""Sensitivity analysis: exclude test records whose feature vectors appear in training.

Addresses the finding that 1,699 feature vectors overlap between train and test
(all with conflicting Attack labels). Reports metrics on both:
  A. Full test split (primary)
  B. Test split minus overlapping records (sensitivity)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.calibration import CAAFedCal, predict_with_thresholds_vectorized
from src.models.mlp import (IDSMLP, NumpyDataset, create_model, evaluate_model)
from src.models.centralized import CentralizedTrainer

M2_PROCESSED = ROOT / "data/processed"
M2_PARTITIONS = ROOT / "data/partitions"
M3_RESULTS = ROOT / "results/m3"
M3_MODELS = M3_RESULTS / "models"
M3_METRICS = M3_RESULTS / "metrics"
M3_SENSITIVITY = M3_RESULTS / "sensitivity"
M3_SENSITIVITY.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 10
INPUT_DIM = 41
BENIGN_CLASS_ID = 2
SEEDS = [42, 123, 7]
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_m2_artifacts():
    X_train = sparse.load_npz(M2_PROCESSED / "X_train.npz").tocsr().toarray()
    y_train = np.load(M2_PROCESSED / "y_train.npy")
    X_test = sparse.load_npz(M2_PROCESSED / "X_test.npz").tocsr().toarray()
    y_test = np.load(M2_PROCESSED / "y_test.npy")
    return X_train, y_train, X_test, y_test


def find_overlapping_test_indices(df, train_indices, test_indices):
    """Find test indices whose feature vectors appear in training set."""
    feature_cols = [c for c in df.columns if c not in ("Attack", "Label")]
    
    train_features = df.loc[train_indices, feature_cols]
    test_features = df.loc[test_indices, feature_cols]
    
    # Create hashable keys
    def make_key(row):
        return tuple(row)
    
    train_keys = set(train_features.apply(make_key, axis=1))
    test_keys = test_features.apply(make_key, axis=1)
    
    overlapping_test_positions = []
    for pos, key in enumerate(test_keys):
        if key in train_keys:
            overlapping_test_positions.append(pos)
    
    return overlapping_test_positions


def compute_metrics(targets, preds, num_classes=NUM_CLASSES, benign_class_id=BENIGN_CLASS_ID):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    
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


def evaluate_model_on_split(model, X, y, tag=""):
    """Evaluate a model on a data split, returning predictions and metrics."""
    test_loader = DataLoader(
        NumpyDataset(X, y), batch_size=BATCH_SIZE, shuffle=False)
    metrics = evaluate_model(model, test_loader,
                             torch.nn.CrossEntropyLoss(),
                             torch.device(DEVICE))
    preds = metrics.pop("predictions")
    targets = metrics.pop("targets")
    return preds, targets


def evaluate_calibrated_on_split(model, thresholds, X, y, tag=""):
    """Evaluate a calibrated model on a data split."""
    pred_result = predict_with_thresholds_vectorized(
        model, X, thresholds,
        benign_class_id=BENIGN_CLASS_ID,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    return pred_result["predictions"], y


def main():
    # Load M2 processed data
    X_train, y_train, X_test, y_test = load_m2_artifacts()
    
    # Load raw data to find overlapping feature vectors
    import yaml
    config_path = ROOT / "configs/experiment.yaml"
    config = yaml.safe_load(config_path.read_text())
    ds = config["dataset"]
    raw_path = ROOT / "data/raw/unsw-nb15" / ds["data_file"]
    df = pd.read_csv(raw_path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    
    # Recreate the stratified split to get train/test indices
    from sklearn.model_selection import train_test_split
    y_raw = df["Attack"].astype(str)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y_raw)
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    
    # Find overlapping test positions (positions in the test array)
    print("Finding overlapping feature vectors between train and test...")
    overlapping_test_positions = find_overlapping_test_indices(df, train_idx, test_idx)
    print(f"  Overlapping test records: {len(overlapping_test_positions)} / {len(test_idx)} "
          f"({100*len(overlapping_test_positions)/len(test_idx):.2f}%)")
    
    # Create mask for non-overlapping test records
    overlap_mask = np.zeros(len(test_idx), dtype=bool)
    overlap_mask[overlapping_test_positions] = True
    non_overlap_mask = ~overlap_mask
    
    # Filter test data
    X_test_clean = X_test[non_overlap_mask]
    y_test_clean = y_test[non_overlap_mask]
    print(f"  Clean test records: {len(y_test_clean)} / {len(y_test)}")
    
    # Class distribution of clean test set
    from collections import Counter
    clean_counts = Counter(y_test_clean)
    print(f"  Clean test class distribution: {dict(sorted(clean_counts.items()))}")
    
    results = {
        "overlapping_test_records": len(overlapping_test_positions),
        "total_test_records": len(y_test),
        "clean_test_records": int(len(y_test_clean)),
        "overlap_percentage": float(100 * len(overlapping_test_positions) / len(y_test)),
        "seeds": {},
    }
    
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed} — Sensitivity Analysis")
        print(f"{'='*60}")
        
        seed_results = {}
        
        # --- Centralized ---
        print("\n  Centralized MLP...")
        ctrainer = CentralizedTrainer(
            X_train, y_train, X_test, y_test,
            num_classes=NUM_CLASSES, input_dim=INPUT_DIM,
            seed=seed, device=DEVICE,
            checkpoint_dir=M3_RESULTS / f"checkpoints/centralized_sens_seed{seed}",
        )
        # Load existing model from checkpoints
        model_path = M3_RESULTS / f"checkpoints/centralized_seed{seed}_s{seed}/final_model.pt"
        if model_path.exists():
            ctrainer.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            print(f"    Model not found at {model_path}, skipping.")
            continue
        
        # Full test
        preds_full, targets_full = evaluate_model_on_split(
            ctrainer.model, X_test, y_test)
        m_full = compute_metrics(targets_full, preds_full)
        
        # Clean test
        preds_clean, targets_clean = evaluate_model_on_split(
            ctrainer.model, X_test_clean, y_test_clean)
        m_clean = compute_metrics(targets_clean, preds_clean)
        
        seed_results["centralized"] = {
            "full_test": m_full,
            "clean_test": m_clean,
        }
        print(f"    Full: acc={m_full['accuracy']:.4f}, macro_f1={m_full['macro_f1']:.4f}")
        print(f"    Clean: acc={m_clean['accuracy']:.4f}, macro_f1={m_clean['macro_f1']:.4f}")
        
        # --- FedAvg (argmax) ---
        print("\n  FedAvg (argmax)...")
        model_path = M3_MODELS / f"fedavg_seed{seed}_s{seed}.pt"
        if not model_path.exists():
            print(f"    Model not found, skipping.")
            continue
        
        model = create_model(input_dim=INPUT_DIM, num_classes=NUM_CLASSES,
                             seed=seed).to(torch.device(DEVICE))
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        preds_full, targets_full = evaluate_model_on_split(
            model, X_test, y_test)
        m_full = compute_metrics(targets_full, preds_full)
        
        preds_clean, targets_clean = evaluate_model_on_split(
            model, X_test_clean, y_test_clean)
        m_clean = compute_metrics(targets_clean, preds_clean)
        
        seed_results["fedavg"] = {
            "full_test": m_full,
            "clean_test": m_clean,
        }
        print(f"    Full: acc={m_full['accuracy']:.4f}, macro_f1={m_full['macro_f1']:.4f}")
        print(f"    Clean: acc={m_clean['accuracy']:.4f}, macro_f1={m_clean['macro_f1']:.4f}")
        
        # --- CAA-FedCal ---
        print("\n  CAA-FedCal...")
        cal_path = M3_RESULTS / f"calibration/caa_fedcal_seed{seed}_s{seed}.json"
        if not cal_path.exists():
            print(f"    Calibration not found at {cal_path}, skipping.")
            continue
        
        cal = CAAFedCal.load(cal_path)
        
        preds_full, targets_full = evaluate_calibrated_on_split(
            model, cal.thresholds, X_test, y_test)
        m_full = compute_metrics(targets_full, preds_full)
        
        preds_clean, targets_clean = evaluate_calibrated_on_split(
            model, cal.thresholds, X_test_clean, y_test_clean)
        m_clean = compute_metrics(targets_clean, preds_clean)
        
        seed_results["caa_fedcal"] = {
            "full_test": m_full,
            "clean_test": m_clean,
        }
        print(f"    Full: acc={m_full['accuracy']:.4f}, macro_f1={m_full['macro_f1']:.4f}")
        print(f"    Clean: acc={m_clean['accuracy']:.4f}, macro_f1={m_clean['macro_f1']:.4f}")
        
        results["seeds"][str(seed)] = seed_results
    
    # Save results
    output_path = M3_SENSITIVITY / "sensitivity_analysis.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nSensitivity analysis saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*60)
    print(f"Overlapping test records: {results['overlapping_test_records']} / {results['total_test_records']} "
          f"({results['overlap_percentage']:.2f}%)")
    print(f"Clean test records: {results['clean_test_records']}")
    print()
    for seed, sr in results["seeds"].items():
        print(f"\nSeed {seed}:")
        for exp_name, exp_data in sr.items():
            full = exp_data["full_test"]
            clean = exp_data["clean_test"]
            acc_diff = clean["accuracy"] - full["accuracy"]
            f1_diff = clean["macro_f1"] - full["macro_f1"]
            print(f"  {exp_name}: "
                  f"full_acc={full['accuracy']:.4f} clean_acc={clean['accuracy']:.4f} (diff={acc_diff:+.4f}), "
                  f"full_mF1={full['macro_f1']:.4f} clean_mF1={clean['macro_f1']:.4f} (diff={f1_diff:+.4f})")


if __name__ == "__main__":
    main()
