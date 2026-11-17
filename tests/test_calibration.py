"""Tests for CAA-FedCal calibration algorithm."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration.calibration import (
    CAAFedCal, ClientCalibrationHistogram, aggregate_histograms,
    compute_client_histograms, predict_with_thresholds_vectorized,
    select_thresholds,
)


# ---------------------------------------------------------------------------
# Helper: dummy model that returns fixed probabilities
# ---------------------------------------------------------------------------
class _DummyModel:
    """Model that returns fixed softmax probabilities per input."""

    def __init__(self, probs_per_record: np.ndarray):
        self.probs = probs_per_record

    def predict_proba(self, x):
        n = len(x) if hasattr(x, "__len__") else 1
        return self.probs[:n] if n > 1 else self.probs[0:1]

    def eval(self):
        return self

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# ClientCalibrationHistogram tests (via compute_client_histograms)
# ---------------------------------------------------------------------------
def test_histogram_present_classes_through_compute():
    X = np.zeros((2, 41), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    probs = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
    ])
    model = _DummyModel(probs)

    hist = compute_client_histograms(
        model, X, y, num_classes=3, num_bins=5,
        device="cpu", benign_class_id=2,
    )

    assert hist.present_classes == {0, 1}
    assert hist.tp_counts[0] == 1
    assert hist.tp_counts[1] == 1
    assert hist.tp_counts[2] == 0


# ---------------------------------------------------------------------------
# compute_client_histograms tests
# ---------------------------------------------------------------------------
def test_compute_client_histograms_vectorized():
    # 4 records, 3 classes
    X = np.zeros((4, 41), dtype=np.float32)
    y = np.array([0, 1, 2, 0], dtype=np.int64)
    # Fixed probabilities: record 0 -> class 0 with score 0.9, etc.
    probs = np.array([
        [0.9, 0.05, 0.05],   # record 0: true=0, score for class 0 = 0.9
        [0.1, 0.8, 0.1],     # record 1: true=1, score for class 1 = 0.8
        [0.05, 0.05, 0.9],   # record 2: true=2, score for class 2 = 0.9
        [0.85, 0.1, 0.05],   # record 3: true=0, score for class 0 = 0.85
    ])
    model = _DummyModel(probs)

    hist = compute_client_histograms(
        model, X, y, num_classes=3, num_bins=5,
        device="cpu", benign_class_id=2,
    )

    # Class 0: 2 records (indices 0, 3), scores 0.9 and 0.85
    assert hist.tp_counts[0] == 2
    # Class 1: 1 record (index 1), score 0.8
    assert hist.tp_counts[1] == 1
    # Class 2: 1 record (index 2), score 0.9
    assert hist.tp_counts[2] == 1
    # Present classes: {0, 1, 2}
    assert hist.present_classes == {0, 1, 2}
    # Benign count: class 2 has 1 benign record
    assert hist.benign_counts[2] == 1


# ---------------------------------------------------------------------------
# aggregate_histograms tests
# ---------------------------------------------------------------------------
def test_aggregate_histograms_presence_weighting():
    # Create two clients with histograms for class 0
    h1 = ClientCalibrationHistogram(num_classes=2, num_bins=5)
    h1.tp_histograms[0] = np.array([0.0, 0.0, 0.0, 0.5, 0.5])  # scores in bins 3,4
    h1.tp_counts[0] = 10
    h1.present_classes.add(0)

    h2 = ClientCalibrationHistogram(num_classes=2, num_bins=5)
    h2.tp_histograms[0] = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # scores in bins 0,1
    h2.tp_counts[0] = 10
    h2.present_classes.add(0)

    client_hists = {0: h1, 1: h2}
    result = aggregate_histograms(
        client_hists, num_classes=2, num_bins=5,
        weighting="presence", min_clients=1, benign_class_id=1,
    )

    cls0 = result["0"]
    assert cls0["eligible"] is True
    assert cls0["n_eligible_clients"] == 2
    # Presence weighting: equal weight 0.5 each
    # Aggregated: 0.5 * [0,0,0,0.5,0.5] + 0.5 * [0.5,0.5,0,0,0]
    # = [0.25, 0.25, 0, 0.25, 0.25]
    expected = np.array([0.25, 0.25, 0.0, 0.25, 0.25])
    np.testing.assert_allclose(cls0["tp_histogram"], expected)


def test_aggregate_histograms_sample_count_weighting():
    h1 = ClientCalibrationHistogram(num_classes=2, num_bins=5)
    h1.tp_histograms[0] = np.array([0.0, 0.0, 0.0, 0.5, 0.5])
    h1.tp_counts[0] = 100  # Many records
    h1.present_classes.add(0)

    h2 = ClientCalibrationHistogram(num_classes=2, num_bins=5)
    h2.tp_histograms[0] = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    h2.tp_counts[0] = 1  # Few records
    h2.present_classes.add(0)

    client_hists = {0: h1, 1: h2}
    result = aggregate_histograms(
        client_hists, num_classes=2, num_bins=5,
        weighting="sample_count", min_clients=1, benign_class_id=1,
    )

    cls0 = result["0"]
    assert cls0["eligible"] is True
    # Sample count weighting: h1 gets 100/101, h2 gets 1/101
    expected = (100/101) * np.array([0.0, 0.0, 0.0, 0.5, 0.5]) + \
               (1/101) * np.array([0.5, 0.5, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(cls0["tp_histogram"], expected, atol=1e-6)


def test_aggregate_histograms_min_clients_filter():
    h1 = ClientCalibrationHistogram(num_classes=2, num_bins=5)
    h1.present_classes.add(0)

    client_hists = {0: h1}  # Only 1 client has class 0
    result = aggregate_histograms(
        client_hists, num_classes=2, num_bins=5,
        weighting="presence", min_clients=3, benign_class_id=1,
    )

    assert result["0"]["eligible"] is False
    assert "only 1 clients" in result["0"]["reason"]


# ---------------------------------------------------------------------------
# select_thresholds tests
# ---------------------------------------------------------------------------
def test_select_thresholds_benign_fpr_constraint():
    # Class 0: TP scores concentrated in high bins, benign scores in low bins
    tp_hist = np.array([0.0, 0.0, 0.0, 0.3, 0.7])  # most TP scores high
    benign_hist = np.array([0.7, 0.3, 0.0, 0.0, 0.0])  # benign scores low

    entry = {
        "eligible": True,
        "tp_histogram": tp_hist.tolist(),
        "benign_histogram": benign_hist.tolist(),
        "bin_edges": np.linspace(0.0, 1.0, 6).tolist(),
        "total_tp_records": 100,
    }

    thresholds = select_thresholds(
        {"0": entry}, num_classes=2,
        alpha_bpr=0.05, min_calib_records=10, benign_class_id=1,
    )

    t0 = thresholds["0"]
    assert t0["eligible"] is True
    assert t0["threshold"] is not None
    assert t0["fpr"] <= 0.05  # Constraint satisfied
    assert t0["tpr"] > 0  # Some TPR achieved


def test_select_thresholds_no_valid_threshold():
    # All TP and benign scores are high — no threshold can satisfy FPR constraint
    tp_hist = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    benign_hist = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    entry = {
        "eligible": True,
        "tp_histogram": tp_hist.tolist(),
        "benign_histogram": benign_hist.tolist(),
        "bin_edges": np.linspace(0.0, 1.0, 6).tolist(),
        "total_tp_records": 100,
    }

    thresholds = select_thresholds(
        {"0": entry}, num_classes=2,
        alpha_bpr=0.05, min_calib_records=10, benign_class_id=1,
    )

    t0 = thresholds["0"]
    assert t0["eligible"] is True
    assert t0["threshold"] == 1.0  # Falls back to never-trigger


def test_select_thresholds_insufficient_records():
    entry = {
        "eligible": True,
        "tp_histogram": [0.0, 0.0, 0.0, 0.5, 0.5],
        "benign_histogram": [0.5, 0.5, 0.0, 0.0, 0.0],
        "bin_edges": np.linspace(0.0, 1.0, 6).tolist(),
        "total_tp_records": 5,  # Below min_calib_records=10
    }

    thresholds = select_thresholds(
        {"0": entry}, num_classes=2,
        alpha_bpr=0.05, min_calib_records=10, benign_class_id=1,
    )

    assert thresholds["0"]["eligible"] is False
    assert "5" in thresholds["0"]["reason"]


def test_select_thresholds_not_eligible():
    entry = {
        "eligible": False,
        "reason": "only 2 clients have class 0 in calibration (need 3)",
        "eligible_clients": [0, 1],
    }

    thresholds = select_thresholds(
        {"0": entry}, num_classes=2,
        alpha_bpr=0.05, min_calib_records=10, benign_class_id=1,
    )

    assert thresholds["0"]["eligible"] is False
    assert thresholds["0"]["threshold"] is None


# ---------------------------------------------------------------------------
# predict_with_thresholds_vectorized tests
# ---------------------------------------------------------------------------
def test_predict_thresholds_no_satisfied():
    """When no attack class meets its threshold, fall back to argmax."""
    X = np.zeros((2, 41), dtype=np.float32)
    probs = np.array([
        [0.4, 0.3, 0.3],   # argmax=0, no class >= threshold 0.9
        [0.2, 0.5, 0.3],   # argmax=1
    ])
    model = _DummyModel(probs)
    thresholds = {
        "0": {"eligible": True, "threshold": 0.9},
        "1": {"eligible": True, "threshold": 0.9},
    }

    result = predict_with_thresholds_vectorized(
        model, X, thresholds,
        benign_class_id=2, num_classes=3,
        batch_size=2, device="cpu",
    )

    # No threshold satisfied -> argmax
    np.testing.assert_array_equal(result["predictions"], [0, 1])


def test_predict_with_thresholds_integration():
    """Integration test with dummy model."""
    X = np.zeros((4, 41), dtype=np.float32)
    # Probabilities: class 2 (benign) has threshold 0.9, class 0 has 0.7
    probs = np.array([
        [0.8, 0.1, 0.1],   # class 0 score 0.8 >= 0.7 -> predict 0
        [0.4, 0.5, 0.1],   # class 1 score 0.5 < 0.9, class 0 0.4 < 0.7 -> argmax=1
        [0.1, 0.1, 0.8],   # benign argmax, no attack >= threshold -> predict 2
        [0.6, 0.3, 0.1],   # class 0 score 0.6 < 0.7 -> argmax=0
    ])
    model = _DummyModel(probs)

    thresholds = {
        "0": {"eligible": True, "threshold": 0.7},
        "1": {"eligible": True, "threshold": 0.9},
    }

    result = predict_with_thresholds_vectorized(
        model, X, thresholds,
        benign_class_id=2, num_classes=3,
        batch_size=4, device="cpu",  # single batch for dummy model
    )

    preds = result["predictions"]
    # Record 0: class 0 score 0.8 >= 0.7 -> predict 0
    assert preds[0] == 0
    # Record 1: class 1 score 0.5 < 0.9, class 0 score 0.4 < 0.7 -> argmax=1
    assert preds[1] == 1
    # Record 2: class 2 (benign) argmax, no attack threshold satisfied -> predict 2
    assert preds[2] == 2
    # Record 3: class 0 score 0.6 < 0.7 -> argmax=0
    assert preds[3] == 0


def test_predict_multiple_thresholds_satisfied():
    """When multiple classes satisfy threshold, pick highest score."""
    X = np.zeros((1, 41), dtype=np.float32)
    probs = np.array([[0.85, 0.9, 0.05]])  # class 0=0.85, class 1=0.9
    model = _DummyModel(probs)

    thresholds = {
        "0": {"eligible": True, "threshold": 0.7},
        "1": {"eligible": True, "threshold": 0.7},
    }

    result = predict_with_thresholds_vectorized(
        model, X, thresholds,
        benign_class_id=2, num_classes=3,
        batch_size=4, device="cpu",  # single batch
    )

    # Both classes satisfy threshold, pick class 1 (higher score 0.9)
    assert result["predictions"][0] == 1


# ---------------------------------------------------------------------------
# CAAFedCal end-to-end tests
# ---------------------------------------------------------------------------
def test_caa_fedcal_fit_and_predict(tmp_path):
    """End-to-end test of CAA-FedCal fit and predict."""
    # Create dummy data
    np.random.seed(42)
    n_clients = 3
    n_classes = 3
    benign_class = 2

    client_calibration = {}
    for cid in range(n_clients):
        n_records = 50
        X = np.random.randn(n_records, 41).astype(np.float32)
        # Assign labels: mostly benign, some of each class
        y = np.random.choice([0, 1, 2], size=n_records, p=[0.3, 0.3, 0.4])
        client_calibration[cid] = {"X": X, "y": y}

    # Dummy model with fixed probabilities
    n_total = sum(len(v["y"]) for v in client_calibration.values())
    all_probs = np.random.dirichlet([1, 1, 1], size=n_total)
    model = _DummyModel(all_probs)

    cal = CAAFedCal(
        num_classes=n_classes, num_bins=5,
        min_clients=1, min_calib_records=5,
        alpha_bpr=0.05, benign_class_id=benign_class,
        weighting="presence",
    )
    result = cal.fit(model, client_calibration, device="cpu")

    assert cal.is_fitted
    assert "thresholds" in result
    assert "aggregated" in result
    assert "client_presence" in result

    # Check each class threshold
    for cls in range(n_classes):
        if cls == benign_class:
            continue
        entry = result["thresholds"][str(cls)]
        assert "threshold" in entry
        assert "eligible" in entry

    # Test predict
    X_test = np.random.randn(10, 41).astype(np.float32)
    test_probs = np.random.dirichlet([1, 1, 1], size=10)
    test_model = _DummyModel(test_probs)

    pred_result = cal.predict_with_thresholds(
        test_model, X_test, batch_size=5, device="cpu",
    )
    assert "predictions" in pred_result
    assert "probabilities" in pred_result
    assert len(pred_result["predictions"]) == 10
    assert all(0 <= p < n_classes for p in pred_result["predictions"])


def test_caa_fedcal_save_load(tmp_path):
    """Test save/load roundtrip."""
    np.random.seed(42)
    client_calibration = {
        0: {"X": np.random.randn(30, 41).astype(np.float32),
            "y": np.array([0]*10 + [1]*10 + [2]*10)},
    }
    all_probs = np.random.dirichlet([1, 1, 1], size=30)
    model = _DummyModel(all_probs)

    cal = CAAFedCal(
        num_classes=3, num_bins=5, min_clients=1,
        min_calib_records=5, alpha_bpr=0.05, benign_class_id=2,
        weighting="presence",
    )
    cal.fit(model, client_calibration, device="cpu")

    save_path = tmp_path / "calibration.json"
    cal.save(save_path)

    cal2 = CAAFedCal.load(save_path)
    assert cal2.is_fitted
    assert cal2.weighting == "presence"
    assert cal2.alpha_bpr == 0.05
    assert cal2.thresholds == cal.thresholds

    # Predictions should match
    X_test = np.random.randn(5, 41).astype(np.float32)
    test_probs = np.random.dirichlet([1, 1, 1], size=5)
    model2 = _DummyModel(test_probs)

    p1 = cal.predict_with_thresholds(model2, X_test, device="cpu")
    p2 = cal2.predict_with_thresholds(model2, X_test, device="cpu")
    np.testing.assert_array_equal(p1["predictions"], p2["predictions"])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_class_absent_from_all_clients():
    """Class absent from all client calibration sets -> not eligible."""
    h1 = ClientCalibrationHistogram(num_classes=3, num_bins=5)
    h1.present_classes.add(0)  # Only classes 0 and 1 present
    h1.present_classes.add(1)

    client_hists = {0: h1}
    result = aggregate_histograms(
        client_hists, num_classes=3, num_bins=5,
        weighting="presence", min_clients=1, benign_class_id=2,
    )

    # Class 2 is benign -> not in result (skipped)
    assert "2" not in result
    # Classes 0 and 1 should be eligible
    assert result["0"]["eligible"] is True
    assert result["1"]["eligible"] is True


def test_single_client_single_class():
    """Single client with single class in calibration."""
    h = ClientCalibrationHistogram(num_classes=2, num_bins=5)
    h.tp_histograms[0] = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    h.tp_counts[0] = 10
    h.present_classes.add(0)

    client_hists = {0: h}
    result = aggregate_histograms(
        client_hists, num_classes=2, num_bins=5,
        weighting="presence", min_clients=1, benign_class_id=1,
    )

    assert result["0"]["eligible"] is True
    assert result["0"]["n_eligible_clients"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
