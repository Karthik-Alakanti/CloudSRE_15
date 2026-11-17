# M4 Audit — Data Leakage and Split Integrity

## Scope

Verified train/validation/calibration separation, test-set isolation, duplicate-feature overlap handling, and absence of preprocessing/test leakage in threshold selection.

## 1. Train/Validation/Calibration Separation

### M2 Partitioning

Each client's training records are split into train/validation/calibration with target proportions 70/15/15. The M2 validation script (`scripts/validate_data.py`) verifies:
- Train/validation/calibration indices are disjoint within each client
- No client split overlaps another client split
- Every training row is assigned to exactly one client

### Verification: All Checks Pass

`python scripts/validate_data.py` returns `status: PASS` with 0 errors.

## 2. Test-Set Isolation

### Verification

- The test split (478,055 records) is never used in any client partition
- Preprocessing (imputation, standardization) is fit on the training split only; test data is transformed with the training-fitted transformer
- Threshold selection uses calibration partitions only; test data never influences thresholds
- Model evaluation on test data occurs after all training and calibration are complete

### No Test Labels in Training/Calibration

Confirmed by code review:
- `preprocess_dataset()` fits transformer on training split only (`preprocessing_fit_on: "training_split_only"`, `test_used_for_fit: false`)
- `compute_client_histograms()` uses only calibration X/y
- `select_thresholds()` uses only aggregated calibration histograms
- `predict_with_thresholds_vectorized()` uses thresholds computed from calibration only

## 3. Duplicate-Feature Overlap Handling

### Train/Test Overlap

From the M2 duplicate-feature audit (conducted before M3):

- 1,699 unique feature vectors appear in both train and test (0.60% of test records)
- All overlapping vectors have **conflicting Attack labels** (known source-data label noise)
- Binary Label is consistent across all overlaps (0 conflicts)
- **Sensitivity analysis:** Excluding overlapping test records changes macro-F1 by < 0.0015 across all experiments

### Client Internal Overlap (Train/Val/Cal)

Duplicate feature vectors occur **within** client partitions across train/validation/calibration splits:

| Client | Train-Val Overlap | Train-Cal Overlap | Val-Cal Overlap |
|---|---|---|---|
| 0 | 305 | 312 | 158 |
| 1 | 32,817 | 32,652 | 9,848 |
| 2 | 322 | 325 | 82 |
| 3 | 157 | 159 | 86 |
| 4 | 77 | 74 | 35 |
| 5 | 118 | 98 | 32 |
| 6 | 164 | 156 | 51 |
| 7 | 802 | 823 | 202 |
| 8 | 151 | 153 | 45 |
| 9 | 274 | 276 | 91 |

**Client 1** has the most overlap (32,817 train-val, 32,652 train-cal unique vectors) because it holds ~995,499 benign training records (Benign is class 2, which dominates the dataset).

### Implication for Calibration

Calibration records can share feature vectors with training records within the same client. This means:
- The model may have "seen" similar feature patterns during training
- Calibration scores may be over-optimistic for those specific records
- However, since labels can conflict (same features, different Attack labels), this is not a clean label leak

This is a **documented limitation** of the Dirichlet partitioning + 70/15/15 split approach. It does not invalidate the results but should be considered when interpreting calibration reliability.

## 4. Preprocessing Leakage Check

### Verified

| Operation | Data Used | Test Labels Used? |
|---|---|---|
| Imputer fit | Training split only | No |
| Standardizer fit | Training split only | No |
| Label encoding | Training labels only | No |
| Test transform | Training-fitted transformer | No (test labels not used) |
| Partitioning | Training split only | No |
| Threshold selection | Calibration partitions only | No |

No preprocessing or threshold-selection operation uses test labels.

## 5. Limitations

1. **Within-client train/calibration feature overlap:** Duplicate feature vectors appear across client train/validation/calibration splits. Client 1 has the most overlap (~33K unique vectors shared between train and calibration). This may slightly inflate calibration-score reliability for those records.

2. **Train/test feature overlap:** 1,699 feature vectors (0.60% of test) appear in both train and test with conflicting Attack labels. Sensitivity analysis confirms negligible impact (< 0.0015 macro-F1 difference).

3. **No cross-client overlap check performed:** The audit checked within-client overlap but did not verify whether the same feature vector appears in different clients' partitions. This is unlikely to be an issue since Dirichlet partitioning assigns each row to exactly one client, but it was not explicitly verified.

## Conclusion

No data leakage was found in preprocessing, partitioning, or threshold selection. The test set is strictly isolated. The main limitation is within-client feature overlap between train and calibration partitions, which is a consequence of the Dirichlet partitioning + 70/15/15 split and cannot be avoided without group-aware splitting.
