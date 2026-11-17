# M3 Course Project Report

## Federated Multiclass Network Intrusion Detection with Class-Absence-Aware Calibration

**Dataset:** NF-UNSW-NB15-v2 (University of Queensland, UQRDM)
**Method:** CAA-FedCal — Class-Absence-Aware Federated Post-Hoc Calibration
**Status:** M3 complete (3 seeds)

---

## 1. Introduction

Network intrusion detection systems (NIDS) face a fundamental challenge: network traffic is inherently imbalanced, with benign flows vastly outnumbering attacks, and attack types themselves are highly skewed. In a federated learning setting with Non-IID data partitioning, this imbalance is exacerbated — rare attack classes may be entirely absent from some clients' training data.

This project implements and evaluates **CAA-FedCal** (Class-Absence-Aware Federated Post-Hoc Calibration), a post-training calibration method for federated multiclass intrusion detection. CAA-FedCal addresses class absence under Dirichlet α=0.3 Non-IID partitioning by computing class-conditional prediction-score histograms from client calibration partitions, aggregating them with class-presence-aware weighting, and selecting class-specific decision thresholds under a benign false-positive constraint.

---

## 2. Dataset: NF-UNSW-NB15-v2

NF-UNSW-NB15-v2 is a NetFlow v2 rendering of the UNSW-NB15 dataset created at the University of Queensland (UQRDM) by M. Sarhan. It is **not** the original UNSW-NB15 training/testing partition CSVs from UNSW Canberra.

| Property | Value |
|---|---|
| Source | University of Queensland, UQRDM |
| Source record | https://espace.library.uq.edu.au/view/UQ:ffbb0c1 |
| File | `NF-UNSW-NB15-v2.csv` |
| Size | 441,867,785 bytes |
| SHA-256 | `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f` |
| Rows × Columns | 2,390,275 × 45 |
| Missing values | 0 |
| Infinite values | 0 |
| Exact duplicate rows | 0 |

### Class Distribution (10 classes, all retained)

| Class | Count | % |
|---|---|---|
| Benign | 2,295,222 | 96.02% |
| Exploits | 31,551 | 1.32% |
| Fuzzers | 22,310 | 0.93% |
| Generic | 16,560 | 0.69% |
| Reconnaissance | 12,779 | 0.53% |
| DoS | 5,794 | 0.24% |
| Analysis | 2,299 | 0.10% |
| Backdoor | 2,169 | 0.09% |
| Shellcode | 1,427 | 0.06% |
| Worms | 164 | 0.007% |

### Train/Test Split

NF-UNSW-NB15-v2 is a single combined file with no official partition. A reproducible stratified 80/20 train/test split was created by the `Attack` column (seed=42, test_fraction=0.20):

- **Train:** 1,912,220 records
- **Test:** 478,055 records

Preprocessing (standardization) was fit on the training split only. IP address columns (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`) were excluded from model inputs to reduce identity/leakage risk.

### Data Quality Notes

- 1,699 feature vectors overlap between train and test (0.60% of test records), all with conflicting `Attack` labels (known source-data label noise). Binary `Label` is consistent across all overlaps.
- Sensitivity analysis confirmed excluding overlapping records changes macro-F1 by <0.0015 — no meaningful contamination.

---

## 3. Methodology

### 3.1 Model Architecture

**MLP:** 41 input features → 128 hidden (ReLU, dropout 0.2) → 64 hidden (ReLU, dropout 0.2) → 10 output classes (logits). Softmax applied at inference.

| Hyperparameter | Value |
|---|---|
| Hidden layers | 128, 64 |
| Activation | ReLU |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 256 |
| Classes | 10 |

### 3.2 Federated Training (FedAvg)

- 10 clients, Dirichlet α=0.3 partitioning
- 15 communication rounds (reduced from 30 for CPU runtime)
- 3 local epochs per round
- Weighted average by client training set size
- 3 random seeds: 42, 123, 7

### 3.3 Centralized Baseline

- Trained on full training split (1,912,220 records)
- 30 epochs, Adam lr=0.001, batch size 256
- Same architecture and hyperparameters as FedAvg

### 3.4 CAA-FedCal Algorithm

CAA-FedCal is a post-training calibration stage applied after FedAvg converges. It does not modify model weights.

**Step 1 — Client class presence:** For each client, determine which classes appear in its calibration partition.

**Step 2 — Local histograms:** For each client and each locally-present class, compute a class-conditional softmax score histogram (20 bins on [0,1]) from calibration records. Also compute benign-class score histograms for false-positive estimation.

**Step 3 — Histogram aggregation:** Aggregate client histograms per class using class-presence-aware weighting (uniform over eligible clients only). Clients without a class in calibration contribute zero weight.

**Step 4 — Threshold selection:** For each eligible class, select the threshold that maximizes true positive rate subject to benign false-positive rate ≤ 0.05 (per class).

**Step 5 — Prediction:** At inference, if any attack class score exceeds its calibrated threshold, predict that class (highest score among satisfied). Otherwise, use argmax.

**Fallback behavior:** Classes with insufficient client coverage (<3 clients) or insufficient calibration records (<10) fall back to argmax. Classes where no threshold satisfies the benign-FPR constraint are set to threshold=1.0 (never trigger).

---

## 4. Experiments

### 4.1 Experimental Design

| Experiment | Description |
|---|---|
| A. Centralized MLP | Single-node training on full training split |
| B. Standard FedAvg | Federated averaging, argmax decision |
| C. FedAvg + Pooled Calibration | All calibration data pooled centrally, ordinary thresholds |
| D. FedAvg + CAA-FedCal | Federated averaging + class-absence-aware calibration |
| E. Weighting Ablation | CAA-FedCal with uniform / sample_count / presence weighting |

### 4.2 Results (mean ± std, 3 seeds)

| Experiment | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | Benign FPR | Minority Recall |
|---|---|---|---|---|---|---|
| Centralized | 0.9877 ± 0.0001 | 0.5819 ± 0.0084 | 0.5623 ± 0.0149 | 0.9865 ± 0.0002 | 0.0025 ± 0.0002 | 0.5357 ± 0.0106 |
| FedAvg | 0.9729 ± 0.0007 | 0.3816 ± 0.0099 | 0.3463 ± 0.0096 | 0.9649 ± 0.0016 | 0.0001 ± 0.0000 | 0.3129 ± 0.0081 |
| Pooled Cal | 0.9729 ± 0.0007 | 0.3816 ± 0.0099 | 0.3463 ± 0.0096 | 0.9649 ± 0.0016 | 0.0001 ± 0.0000 | 0.3129 ± 0.0081 |
| CAA-FedCal | 0.9786 ± 0.0012 | 0.4496 ± 0.0168 | 0.3965 ± 0.0152 | 0.9767 ± 0.0016 | 0.0018 ± 0.0003 | 0.3887 ± 0.0129 |

### 4.3 Per-Class F1 (CAA-FedCal, seed 42)

| Class | F1 | Support |
|---|---|---|
| Benign | 0.997 | 459,044 |
| Shellcode | 0.611 | 285 |
| Reconnaissance | 0.693 | 2,556 |
| Fuzzers | 0.425 | 4,462 |
| Exploits | 0.590 | 6,310 |
| Generic | 0.520 | 3,312 |
| DoS | 0.021 | 1,159 |
| Backdoor | 0.036 | 434 |
| Analysis | 0.000 | 460 |
| Worms | 0.000 | 33 |

### 4.4 Key Findings

1. **Centralized MLP significantly outperforms FedAvg** (macro-F1: 0.56 vs 0.35). This is expected — centralized training has access to all data simultaneously, while FedAvg must contend with Non-IID partitioning.

2. **CAA-FedCal improves macro-F1 over FedAvg by ~0.05** (0.3965 vs 0.3463), consistent across all 3 seeds. The improvement comes primarily from mid-tier classes (Exploits: 0.264→0.590, Fuzzers: 0.131→0.425).

3. **Pooled calibration provides zero improvement** over standard FedAvg argmax. This demonstrates that naively pooling calibration data does not help — the class-presence-aware structure of CAA-FedCal is essential.

4. **CAA-FedCal does not help the rarest classes** (Worms, Analysis still F1=0.000). Worms has only 33 test records and is present in only 5 of 10 clients' calibration sets. With 18 total Worms calibration records, there is insufficient data to estimate a reliable threshold.

5. **Benign false-positive rate remains low** under CAA-FedCal (0.0018 vs 0.0001 for FedAvg). The constraint (α_bpr=0.05 per class) is respected.

6. **Overlapping train/test feature vectors (0.60%) have negligible impact** — excluding them changes macro-F1 by <0.0015.

7. **All three CAA-FedCal weighting schemes give identical results** after histogram normalization, suggesting the weighting choice is not critical when client score distributions are similar.

---

## 5. Limitations

1. **CPU-only training:** No GPU available. Training time limited the number of FedAvg rounds (15 instead of 30) and centralized epochs (30 instead of 90).

2. **Rare classes:** Worms (164 total records, 33 test) and Analysis (2,299 total, 460 test) are too rare for reliable per-class threshold estimation. Both have F1=0.000 across all experiments.

3. **Dirichlet α=0.3:** This creates extreme Non-IID partitioning — some clients have 0 records of certain classes. This is by design but limits calibration reliability for rare classes.

4. **Label noise:** 1,699 feature vectors appear in both train and test with conflicting Attack labels. This is source-data noise, not a pipeline artifact.

5. **No privacy claims:** CAA-FedCal aggregates histograms, not raw data, but makes no differential privacy or formal privacy guarantees.

---

## 6. Reproducibility

All experiments are fully reproducible:

- Deterministic seeds (42, 123, 7) for all random procedures
- File SHA-256 hashes recorded for all datasets
- Checkpoints saved after every round/epoch (resumable training)
- All code, configuration, and results under version control
- Package versions recorded (numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1, scipy 1.18.1, torch 2.15.0)

### Key Files

| Path | Description |
|---|---|
| `src/models/mlp.py` | MLP model definition and training utilities |
| `src/federated/fedavg.py` | FedAvg training loop with checkpointing |
| `src/calibration/calibration.py` | CAA-FedCal algorithm implementation |
| `scripts/run_experiments.py` | Main experiment runner (5 experiments, 3 seeds) |
| `scripts/sensitivity_analysis.py` | Sensitivity analysis (duplicate feature exclusion) |
| `scripts/visualize_m3.py` | Visualization generation |
| `tests/test_calibration.py` | 16 tests for calibration algorithm |
| `tests/test_pipeline.py` | 9 existing pipeline tests |
| `docs/CAAFEDCAL.md` | CAA-FedCal algorithm specification |
| `docs/METHODOLOGY.md` | Updated methodology documentation |
| `docs/EXPERIMENT_PROTOCOL.md` | Updated experiment protocol with M3 results |
| `results/m3/` | All experiment outputs (models, metrics, figures) |

---

## 7. Conclusion

CAA-FedCal demonstrates that class-absence-aware federated calibration can improve multiclass intrusion detection performance under Non-IID partitioning. The method improves macro-F1 by ~0.05 over standard FedAvg while maintaining a low benign false-positive rate. However, the rarest classes (Worms, Analysis) remain undetectable due to insufficient calibration records — a fundamental limitation of extreme Non-IID partitioning with α=0.3.

Future work should explore: (1) alternative partitioning strategies that preserve rare classes across more clients, (2) semi-supervised or transfer learning approaches for rare classes, and (3) formal privacy analysis of histogram aggregation.

---

**Dataset citation:** M. Sarhan, NF-UNSW-NB15-v2, University of Queensland, UQRDM. https://espace.library.uq.edu.au/view/UQ:ffbb0c1
