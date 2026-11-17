"""CAA-FedCal: Class-Absence-Aware Federated Post-Hoc Calibration.

Implements the algorithm documented in docs/CAAFEDCAL.md.

Key principles:
- Calibration partitions only; test data never used for threshold selection.
- Class-presence-aware weighting: only clients that have a class in calibration
  contribute to that class's histogram.
- Benign false-positive constraint enforced per class.
- Falls back to argmax for classes with insufficient coverage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.mlp import IDSMLP, NumpyDataset


class ClientCalibrationHistogram:
    """Class-conditional score histograms for one client."""

    def __init__(self, num_classes: int, num_bins: int = 20):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
        self.tp_histograms = np.zeros((num_classes, num_bins), dtype=np.float64)
        self.tp_counts = np.zeros(num_classes, dtype=np.int64)
        self.benign_histograms = np.zeros((num_classes, num_bins), dtype=np.float64)
        self.benign_counts = np.zeros(num_classes, dtype=np.int64)
        self.present_classes: set[int] = set()


def compute_client_histograms(model: IDSMLP, calibration_X: np.ndarray,
                              calibration_y: np.ndarray,
                              num_classes: int, num_bins: int = 20,
                              batch_size: int = 256,
                              device: str | torch.device = "cpu",
                              benign_class_id: int = 2) -> ClientCalibrationHistogram:
    """Run frozen model on client calibration records and build histograms.

    Vectorized over batches — no per-record Python loop.
    """
    model.eval()
    dev = torch.device(device)
    hist = ClientCalibrationHistogram(num_classes, num_bins)
    bin_edges = hist.bin_edges

    ds = NumpyDataset(calibration_X, calibration_y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            probs = model.predict_proba(xb)
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            labels = yb.cpu().numpy()  # (B,)
            batch_size_actual = len(labels)

            # True-class scores for all records in batch
            true_scores = probs[np.arange(batch_size_actual), labels]

            # Bin indices for true-class scores
            tp_bin_idx = np.digitize(true_scores, bin_edges) - 1
            tp_bin_idx = np.clip(tp_bin_idx, 0, num_bins - 1)

            # Update present classes
            hist.present_classes.update(labels.tolist())

            # Per-class TP histograms (vectorized via np.add.at)
            for cls in range(num_classes):
                mask = labels == cls
                if not mask.any():
                    continue
                hist.tp_counts[cls] += int(mask.sum())
                np.add.at(hist.tp_histograms[cls], tp_bin_idx[mask], 1)

            # Benign records: for each benign record, add score for each
            # non-benign class to the benign histogram
            benign_mask = labels == benign_class_id
            if benign_mask.any():
                hist.benign_counts[benign_class_id] += int(benign_mask.sum())
                benign_probs = probs[benign_mask]  # (n_benign, C)
                for cls in range(num_classes):
                    if cls == benign_class_id:
                        continue
                    cls_scores = benign_probs[:, cls]
                    b_idx = np.digitize(cls_scores, bin_edges) - 1
                    b_idx = np.clip(b_idx, 0, num_bins - 1)
                    np.add.at(hist.benign_histograms[cls], b_idx, 1)

    return hist


def aggregate_histograms(client_hists: dict[int, ClientCalibrationHistogram],
                         num_classes: int, num_bins: int = 20,
                         weighting: str = "presence",
                         min_clients: int = 3,
                         benign_class_id: int = 2) -> dict[str, Any]:
    """Aggregate client histograms into class-specific threshold candidates.

    Returns per-class:
    - aggregated TP histogram (normalized)
    - aggregated benign histogram (normalized)
    - eligible clients (those with class in calibration)
    - eligibility flag (|C_i| >= min_clients)
    """
    result: dict[str, Any] = {}

    for cls in range(num_classes):
        if cls == benign_class_id:
            continue  # No threshold for benign class

        eligible_clients = sorted(
            c for c, h in client_hists.items() if cls in h.present_classes
        )
        n_eligible = len(eligible_clients)
        eligible = n_eligible >= min_clients

        if not eligible:
            result[str(cls)] = {
                "eligible": False,
                "reason": f"only {n_eligible} clients have class {cls} in calibration (need {min_clients})",
                "eligible_clients": eligible_clients,
            }
            continue

        if weighting == "presence":
            weights = {c: 1.0 / n_eligible for c in eligible_clients}
        elif weighting == "sample_count":
            total_records = sum(client_hists[c].tp_counts[cls] for c in eligible_clients)
            if total_records == 0:
                weights = {c: 1.0 / n_eligible for c in eligible_clients}
            else:
                weights = {c: client_hists[c].tp_counts[cls] / total_records
                           for c in eligible_clients}
        elif weighting == "uniform":
            weights = {c: 1.0 / len(client_hists) for c in eligible_clients}
        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        tp_agg = np.zeros(num_bins, dtype=np.float64)
        benign_agg = np.zeros(num_bins, dtype=np.float64)
        total_tp_records = 0
        total_benign_records = 0

        for c in eligible_clients:
            w = weights[c]
            tp_agg += w * client_hists[c].tp_histograms[cls]
            benign_agg += w * client_hists[c].benign_histograms[cls]
            total_tp_records += int(client_hists[c].tp_counts[cls])
            total_benign_records += int(client_hists[c].benign_counts[cls])

        tp_sum = tp_agg.sum()
        benign_sum = benign_agg.sum()
        if tp_sum > 0:
            tp_agg /= tp_sum
        if benign_sum > 0:
            benign_agg /= benign_sum

        result[str(cls)] = {
            "eligible": True,
            "tp_histogram": tp_agg.tolist(),
            "benign_histogram": benign_agg.tolist(),
            "bin_edges": np.linspace(0.0, 1.0, num_bins + 1).tolist(),
            "eligible_clients": eligible_clients,
            "n_eligible_clients": n_eligible,
            "total_tp_records": int(total_tp_records),
            "total_benign_records": int(total_benign_records),
            "weighting": weighting,
        }

    return result


def select_thresholds(aggregated: dict[str, Any], num_classes: int,
                      alpha_bpr: float = 0.05,
                      min_calib_records: int = 10,
                      benign_class_id: int = 2) -> dict[str, Any]:
    """Select class-specific thresholds under benign-FPR constraint.

    For each eligible class, find the threshold that maximizes TPR
    subject to FPR_benign <= alpha_bpr.
    """
    thresholds: dict[str, Any] = {}

    for cls in range(num_classes):
        if cls == benign_class_id:
            continue

        cls_key = str(cls)
        entry = aggregated.get(cls_key, {})

        if not entry.get("eligible", False):
            thresholds[cls_key] = {
                "threshold": None,
                "eligible": False,
                "reason": entry.get("reason", "not eligible"),
            }
            continue

        if entry.get("total_tp_records", 0) < min_calib_records:
            thresholds[cls_key] = {
                "threshold": None,
                "eligible": False,
                "reason": f"only {entry.get('total_tp_records', 0)} calibration records (need {min_calib_records})",
            }
            continue

        tp_hist = np.array(entry["tp_histogram"])
        benign_hist = np.array(entry["benign_histogram"])
        bin_edges = np.array(entry["bin_edges"])
        num_bins = len(bin_edges) - 1

        best_threshold = None
        best_tpr = -1.0
        best_fpr = 1.0

        # Candidate thresholds: right edge of each bin
        for b in range(num_bins):
            t = bin_edges[b + 1]
            tpr = tp_hist[b:].sum()
            fpr = benign_hist[b:].sum()
            if fpr <= alpha_bpr and tpr > best_tpr:
                best_tpr = float(tpr)
                best_fpr = float(fpr)
                best_threshold = float(t)

        # Try threshold = 1.0 (never predict)
        tpr_at_1 = float(tp_hist[-1:].sum())
        fpr_at_1 = float(benign_hist[-1:].sum())
        if fpr_at_1 <= alpha_bpr and tpr_at_1 > best_tpr:
            best_tpr = tpr_at_1
            best_fpr = fpr_at_1
            best_threshold = 1.0

        if best_threshold is None:
            best_threshold = 1.0
            best_tpr = 0.0
            best_fpr = 0.0

        thresholds[cls_key] = {
            "threshold": best_threshold,
            "eligible": True,
            "tpr": best_tpr,
            "fpr": best_fpr,
            "alpha_bpr": alpha_bpr,
            "constraint_satisfied": best_fpr <= alpha_bpr,
        }

    return thresholds


def predict_with_thresholds_vectorized(model: IDSMLP, X: np.ndarray,
                                       thresholds: dict[str, Any],
                                       benign_class_id: int = 2,
                                       num_classes: int = 10,
                                       batch_size: int = 256,
                                       device: str | torch.device = "cpu") -> dict[str, np.ndarray]:
    """Apply calibrated thresholds to produce predictions (vectorized).

    Decision rule:
    - If any non-benign class has score >= its threshold, predict the
      threshold-satisfied class with the highest score.
    - Otherwise, predict argmax.
    """
    dev = torch.device(device)
    model.eval()
    ds = NumpyDataset(X, np.zeros(len(X), dtype=int))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []
    all_argmax = []

    # Build threshold array: threshold[i] or inf if not eligible
    thr_array = np.full(num_classes, np.inf)
    for cls in range(num_classes):
        if cls == benign_class_id:
            continue
        entry = thresholds.get(str(cls), {})
        if entry.get("eligible", False):
            thr = entry.get("threshold")
            if thr is not None:
                thr_array[cls] = thr

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev)
            probs = model.predict_proba(xb)
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            argmax = probs.argmax(axis=1)

            batch_preds = argmax.copy()

            # For each record, check which non-benign classes satisfy threshold
            # satisfied_mask[i, cls] = True if probs[i, cls] >= thr_array[cls]
            # (benign class excluded: thr_array[benign] = inf, never satisfied)
            satisfied = probs >= thr_array[np.newaxis, :]  # (B, C)
            satisfied[:, benign_class_id] = False  # never threshold-benign

            any_satisfied = satisfied.any(axis=1)

            if any_satisfied.any():
                # For records with at least one satisfied class:
                # Among satisfied classes, pick the one with highest score.
                # Mask out non-satisfied classes by setting their scores to -inf.
                masked_scores = np.where(satisfied, probs, -np.inf)
                best_satisfied_cls = masked_scores.argmax(axis=1)
                batch_preds[any_satisfied] = best_satisfied_cls[any_satisfied]

            all_preds.append(batch_preds)
            all_probs.append(probs)
            all_argmax.append(argmax)

    return {
        "predictions": np.concatenate(all_preds),
        "probabilities": np.concatenate(all_probs),
        "argmax_predictions": np.concatenate(all_argmax),
    }


class CAAFedCal:
    """CAA-FedCal calibration pipeline.

    1. Computes client histograms from calibration partitions.
    2. Aggregates histograms with configurable weighting.
    3. Selects class-specific thresholds under benign-FPR constraint.
    4. Applies thresholds at inference time.
    """

    def __init__(self, num_classes: int = 10, num_bins: int = 20,
                 min_clients: int = 3, min_calib_records: int = 10,
                 alpha_bpr: float = 0.05, benign_class_id: int = 2,
                 weighting: str = "presence"):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.min_clients = min_clients
        self.min_calib_records = min_calib_records
        self.alpha_bpr = alpha_bpr
        self.benign_class_id = benign_class_id
        self.weighting = weighting
        self.client_hists: dict[int, ClientCalibrationHistogram] = {}
        self.aggregated: dict[str, Any] = {}
        self.thresholds: dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, model: IDSMLP, client_calibration: dict[int, dict[str, np.ndarray]],
            device: str | torch.device = "cpu") -> dict[str, Any]:
        """Fit CAA-FedCal on client calibration partitions.

        client_calibration: dict mapping client_id -> {"X": cal_X, "y": cal_y}
        """
        self.client_hists = {}
        for cid, data in client_calibration.items():
            hist = compute_client_histograms(
                model, data["X"], data["y"],
                num_classes=self.num_classes,
                num_bins=self.num_bins,
                device=device,
                benign_class_id=self.benign_class_id,
            )
            self.client_hists[cid] = hist

        self.aggregated = aggregate_histograms(
            self.client_hists, self.num_classes, self.num_bins,
            weighting=self.weighting,
            min_clients=self.min_clients,
            benign_class_id=self.benign_class_id,
        )

        self.thresholds = select_thresholds(
            self.aggregated, self.num_classes,
            alpha_bpr=self.alpha_bpr,
            min_calib_records=self.min_calib_records,
            benign_class_id=self.benign_class_id,
        )

        self.is_fitted = True
        return {
            "aggregated": self.aggregated,
            "thresholds": self.thresholds,
            "client_presence": {
                str(c): sorted(list(h.present_classes))
                for c, h in self.client_hists.items()
            },
        }

    def predict_with_thresholds(self, model: IDSMLP, X: np.ndarray,
                                batch_size: int = 256,
                                device: str | torch.device = "cpu") -> dict[str, np.ndarray]:
        """Apply calibrated thresholds to produce predictions."""
        if not self.is_fitted:
            raise RuntimeError("CAA-FedCal must be fitted before prediction.")
        return predict_with_thresholds_vectorized(
            model, X, self.thresholds,
            benign_class_id=self.benign_class_id,
            num_classes=self.num_classes,
            batch_size=batch_size,
            device=device,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_classes": self.num_classes,
            "num_bins": self.num_bins,
            "min_clients": self.min_clients,
            "min_calib_records": self.min_calib_records,
            "alpha_bpr": self.alpha_bpr,
            "benign_class_id": self.benign_class_id,
            "weighting": self.weighting,
            "is_fitted": self.is_fitted,
            "aggregated": self.aggregated,
            "thresholds": self.thresholds,
            "client_presence": {
                str(c): sorted(list(h.present_classes))
                for c, h in self.client_hists.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CAAFedCal":
        path = Path(path)
        payload = json.loads(path.read_text())
        cal = cls(
            num_classes=payload["num_classes"],
            num_bins=payload["num_bins"],
            min_clients=payload["min_clients"],
            min_calib_records=payload["min_calib_records"],
            alpha_bpr=payload["alpha_bpr"],
            benign_class_id=payload["benign_class_id"],
            weighting=payload["weighting"],
        )
        cal.aggregated = payload["aggregated"]
        cal.thresholds = payload["thresholds"]
        cal.is_fitted = payload["is_fitted"]
        cal.client_hists = {}
        for c_str, classes in payload["client_presence"].items():
            hist = ClientCalibrationHistogram(cal.num_classes, cal.num_bins)
            hist.present_classes = set(classes)
            cal.client_hists[int(c_str)] = hist
        return cal
