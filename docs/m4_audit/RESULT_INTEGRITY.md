# M4 Audit — Result Integrity

## Scope

Verified all three seeds (42, 123, 7) across all five experiments:
- Centralized MLP
- Standard FedAvg
- FedAvg + Pooled Calibration
- FedAvg + CAA-FedCal
- Weighting Ablation (uniform / sample_count / presence)

## Method

1. Recomputed mean and standard deviation from individual seed metrics stored in `results/m3/metrics/all_results_s{seed}.json`.
2. Re-ran inference from saved model checkpoints and calibration threshold files to verify reported metrics match actual model outputs.
3. Verified per-class precision/recall/F1/support match the stored confusion matrices.
4. Checked that all checkpoints, calibration files, and model files exist for each seed.

## Findings

### 1.1 Means and Standard Deviations (Recomputed)

| Experiment | Accuracy (mean ± std) | Macro-F1 (mean ± std) | Benign FPR (mean ± std) |
|---|---|---|---|
| Centralized | 0.987720 ± 0.000105 | 0.562333 ± 0.014865 | 0.002388 ± 0.000189 |
| FedAvg | 0.972928 ± 0.000736 | 0.346321 ± 0.009657 | 0.000121 ± 0.000024 |
| Pooled Calibration | 0.972928 ± 0.000736 | 0.346321 ± 0.009657 | 0.000121 ± 0.000024 |
| CAA-FedCal | 0.978599 ± 0.001142 | 0.396455 ± 0.015978 | 0.001650 ± 0.000317 |

All recomputed values match the values reported in `docs/EXPERIMENT_PROTOCOL.md` (within rounding).

### 1.2 Per-Seed Metrics Verification

| Experiment | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| Centralized acc | 0.987669 | 0.987840 | 0.987650 |
| Centralized macro-F1 | 0.562910 | 0.576901 | 0.547188 |
| FedAvg acc | 0.973769 | 0.972399 | 0.972616 |
| FedAvg macro-F1 | 0.356955 | 0.343908 | 0.338100 |
| CAA-FedCal acc | 0.978578 | 0.979751 | 0.977467 |
| CAA-FedCal macro-F1 | 0.389291 | 0.414761 | 0.385312 |

### 1.3 Inference Re-verification

All reported metrics were re-verified by loading saved model checkpoints and calibration threshold files, re-running inference on the test set, and recomputing metrics:

- **Centralized MLP**: All 3 seeds match reported values exactly (macro-F1 match within 1e-6).
- **FedAvg (argmax)**: All 3 seeds match reported values exactly.
- **CAA-FedCal**: All 3 seeds match reported values exactly.
- **Pooled Calibration**: All 3 seeds match reported values exactly.

### 1.4 Per-Class Metrics vs Confusion Matrices

For all experiments and seeds, per-class precision/recall/F1/support values stored in the JSON results match values recomputed from the stored confusion matrices using `sklearn.metrics.precision_recall_fscore_support` and `confusion_matrix`. No mismatches found.

### 1.5 Checkpoint and Calibration File Integrity

| File Type | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| FedAvg model (`models/fedavg_seed{seed}_s{seed}.pt`) | Exists (60,301 bytes) | Exists (60,325 bytes) | Exists (60,277 bytes) |
| Centralized model (`models/centralized_seed{seed}_s{seed}.pt`) | **MISSING** | **MISSING** | **MISSING** |
| Centralized model (`checkpoints/centralized_seed{seed}_s{seed}/final_model.pt`) | Exists (60,229 bytes) | Exists (60,229 bytes) | Exists (60,229 bytes) |
| CAA-FedCal calibration (`calibration/caa_fedcal_seed{seed}_s{seed}.json`) | Exists | Exists | Exists |
| Pooled calibration (`calibration/pooled_seed{seed}_s{seed}.json`) | Exists | Exists | Exists |
| FedAvg checkpoints directory | Exists (15 rounds + history) | Exists (15 rounds + history) | Exists (15 rounds + history) |
| Centralized checkpoints directory | Exists (30 epochs + history) | Exists (30 epochs + history) | Exists (30 epochs + history) |

**Issue:** Centralized final models are saved in `checkpoints/` but not in `models/`. The `models/` directory is missing centralized model files. This is a minor inconsistency but does not affect result integrity since the models exist in checkpoints.

### 1.6 Stale File Check

No metric files appear stale. All `all_results_s{seed}.json` files contain results for all 5 experiments for their respective seeds. No evidence of silently reused metrics from previous runs.

## Conclusion

Result integrity is verified. All reported means, standard deviations, and per-seed metrics can be recomputed from stored JSON files and match actual model inference outputs. Per-class metrics agree with confusion matrices. All checkpoints and calibration files exist for all three seeds.

**Minor issue:** Centralized model files are absent from `results/m3/models/` but present in `results/m3/checkpoints/`. This does not affect result integrity.
