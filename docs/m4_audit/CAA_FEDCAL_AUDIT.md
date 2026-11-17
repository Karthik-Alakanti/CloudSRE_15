# M4 Audit — CAA-FedCal Implementation Trace

## Scope

Traced the complete CAA-FedCal implementation from local client score generation to final prediction, verifying each step of the algorithm documented in `docs/CAAFEDCAL.md`.

## 1. Class-Conditional Score Histogram Construction

### Implementation (`src/calibration/calibration.py:compute_client_histograms`)

For each client `c`, the frozen global model is run on the client's calibration records. For each record `(x, y)`:

1. **True-positive class histogram:** The softmax score `s_y(x)` (score for the true class) is binned into one of 20 uniform bins on [0, 1]. The count is added to `H_c[y][bin]`.

2. **Benign-class histogram:** If `y == Benign` (class ID 2), then for every non-benign class `i != 2`, the score `s_i(x)` is binned and added to `H_benign_c[i][bin]`. This captures how often benign records receive high scores for each attack class.

3. **Class presence:** The true label `y` is added to `P_c` (the set of classes present in client `c`'s calibration set).

### Vectorization

The implementation is vectorized over batches using `np.add.at` for histogram accumulation — no per-record Python loop in the hot path.

### Data Source

Calibration partitions only (`client_{id}_calibration.npz` / `client_{id}_calibration_y.npy`). Training and validation partitions are never used for histogram construction.

## 2. Client Class Presence Determination

### Implementation (`src/calibration/calibration.py:aggregate_histograms`)

For each class `i` (excluding Benign):

```
eligible_clients_i = { c : i in P_c }
n_eligible_i = |eligible_clients_i|
eligible_i = (n_eligible_i >= min_clients)
```

where `min_clients = 3` (configurable).

A class is "present" in a client's calibration set if at least 1 calibration record of that class exists for that client.

### Actual Class Presence (from `partition_metadata.json`)

The partition metadata shows that under Dirichlet α=0.3, class presence is highly uneven:

| Class | Clients with class in calibration | Eligible (>=3 clients) |
|---|---|---|
| Analysis (0) | 7 clients | Yes |
| Backdoor (1) | 7 clients | Yes |
| DoS (3) | 8 clients | Yes |
| Exploits (4) | 7 clients | Yes |
| Fuzzers (5) | 7 clients | Yes |
| Generic (6) | 7 clients | Yes |
| Reconnaissance (7) | 7 clients | Yes |
| Shellcode (8) | 5 clients | Yes |
| Worms (9) | 5 clients | Yes |

All non-benign classes have >= 3 clients with the class in calibration, so all are eligible for threshold estimation.

## 3. Histogram Aggregation

### Implementation (`src/calibration/calibration.py:aggregate_histograms`)

For each eligible class `i`, client histograms are aggregated with configurable weighting:

**Presence weighting (default):**
```
w_c = 1 / n_eligible_i  for each eligible client c
H_agg[i] = sum_{c in eligible_clients_i} w_c * H_c[i]
```

**Sample-count weighting:**
```
w_c = |TP_c[i]| / sum_{c' in eligible_clients_i} |TP_c'[i]|
```

**Uniform weighting:**
```
w_c = 1 / num_clients  for each eligible client c
```

After aggregation, histograms are normalized to sum to 1:
```
H_agg[i][b] /= sum_b H_agg[i][b]
H_benign_agg[i][b] /= sum_b H_benign_agg[i][b]
```

## 4. Threshold Selection

### Implementation (`src/calibration/calibration.py:select_thresholds`)

For each eligible class `i`:

1. Candidate thresholds: right edges of the 20 histogram bins (0.05, 0.10, 0.15, ..., 1.00).
2. For each candidate threshold `t`:
   - TPR_i(t) = sum of normalized TP histogram from bin containing t to end
   - FPR_i(t) = sum of normalized benign histogram from bin containing t to end
3. Select threshold that **maximizes TPR** subject to **FPR <= alpha_bpr (0.05)**.
4. If no threshold satisfies the constraint, set threshold = 1.0 (never triggers).

### Actual Thresholds (all 3 seeds)

| Seed | Class 0 | Class 1 | Class 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9 |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 123 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| 7 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

**Finding:** All thresholds are exactly 0.1 across all classes and all seeds. This is the second bin edge (right edge of bin 1). This suggests that for all classes, FPR at threshold 0.05 exceeds 0.05 (failing the constraint), but FPR at threshold 0.1 is <= 0.05 (satisfying the constraint). The threshold selection picks the lowest threshold that satisfies the constraint while maximizing TPR.

This is a significant finding: the CAA-FedCal thresholds are uniformly 0.1, meaning the method is effectively "classify as attack class i if softmax score for i >= 0.1." This is a very permissive threshold that explains the macro-F1 improvement but also raises questions about whether the threshold selection is meaningfully differentiating between classes.

## 5. Benign False-Positive Constraint Enforcement

### Implementation

The constraint is enforced **per class**, not globally:

For each class `i != Benign`:
```
FPR_i(t_i*) <= alpha_bpr = 0.05
```

This means at most 5% of benign records are misclassified as class `i` (for each attack class individually). The aggregate benign→any-attack rate could be up to 9 * 0.05 = 0.45 in the worst case, but the measured benign FPR is ~0.0018, well below any concerning level.

### Actual Benign FPR (CAA-FedCal)

| Seed | Benign FPR |
|---|---|
| 42 | 0.001991 |
| 123 | 0.001595 |
| 7 | 0.001364 |

All well below the 0.05 per-class constraint.

## 6. Absent and Insufficient Class Handling

### Implementation

| Situation | Threshold | Fallback |
|---|---|---|
| `|C_i| < min_clients` (class absent from too few clients) | None | Argmax |
| Total calibration records < `min_calib_records` (10) | None | Argmax |
| No threshold satisfies FPR <= alpha_bpr | 1.0 | Never triggers (argmax for this class) |
| Class is Benign | N/A | No threshold (benign never has a threshold) |

### Actual Handling

All 9 non-benign classes are eligible (>= 3 clients, >= 10 calibration records). No class falls back to argmax due to ineligibility. Worms (class 9) has 5 clients in calibration, which meets min_clients=3 but is the lowest coverage.

## 7. Conflicting Threshold Override Resolution

### Implementation (`src/calibration/calibration.py:predict_with_thresholds_vectorized`)

For a test record `x` with softmax scores `s = (s_0, ..., s_9)`:

1. **No non-benign class threshold-satisfied:** Predict argmax.
2. **Exactly one non-benign class threshold-satisfied:** Predict that class.
3. **Multiple non-benign classes threshold-satisfied:** Predict the satisfied class with the highest score `s_i`. Ties broken by lower class index.
4. **Benign is argmax and a non-benign class is threshold-satisfied:** The threshold-satisfied class takes precedence (attack detection priority).
5. **Benign is argmax and no threshold satisfied:** Predict Benign.

## 8. Critical Verification: No Training/Test Leakage in Calibration

### Verified

1. **Calibration data is never used for gradient training:** The CAA-FedCal `fit()` method calls `model.eval()` and runs inference with `torch.no_grad()`. Model weights are never modified. The frozen FedAvg model is used only for forward passes on calibration data.

2. **Test set never influences threshold selection:** Thresholds are computed exclusively from calibration partitions. Test data is never passed to `compute_client_histograms`, `aggregate_histograms`, or `select_thresholds`. Test predictions are only computed after thresholds are finalized.

3. **Validation partitions are not used for threshold selection:** Only calibration partitions are used.

## 9. Issues Identified

### Issue 9.1: Uniform Threshold of 0.1

All CAA-FedCal thresholds are exactly 0.1 across all classes and all seeds. This suggests:
- The threshold selection consistently picks the first bin edge that satisfies FPR <= 0.05
- The method is effectively "classify as attack if score >= 0.1"
- This is a very low threshold that may inflate minority-class recall at the cost of some precision
- The thresholds do not meaningfully differentiate between classes

### Issue 9.2: Threshold of 0.1 May Be Trivially Easy to Satisfy

With 10 classes and softmax output, it's common for multiple classes to have scores >= 0.1. The "highest score among satisfied" rule means the prediction usually goes to the argmax class anyway, unless a lower-scoring class happens to cross 0.1 while the argmax class is below 0.1 (rare given softmax properties).

### Issue 9.3: min_clients=3 with Pooled Calibration

When calibration data is pooled across all clients (1 pseudo-client), `n_eligible = 1 < min_clients = 3`, so all classes are ineligible and pooled calibration falls back to pure argmax. This explains why pooled calibration produces identical results to standard FedAvg.

## Conclusion

CAA-FedCal is implemented correctly according to its specification. Calibration data is never used for training, and test data never influences thresholds. However, all thresholds being uniformly 0.1 suggests the method may not be providing meaningful class-specific differentiation. The threshold selection is driven by the benign-FPR constraint, which consistently selects 0.1 as the first valid threshold.
