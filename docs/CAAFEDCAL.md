# CAA-FedCal Algorithm Specification

Class-Absence-Aware Federated Post-Hoc Calibration for Multiclass Intrusion Detection under Non-IID Traffic.

## 1. Overview

CAA-FedCal is a post-training calibration stage applied after a federated model (FedAvg) has converged. It does not modify model weights. Each client uses its local calibration subset (disjoint from train/validation) to compute class-conditional prediction-score summaries. The server aggregates these summaries and selects class-specific decision thresholds subject to a benign false-positive constraint.

## 2. Notation

- `K = 10`: number of clients
- `C = 10`: number of classes (Analysis=0, ..., Worms=9)
- `D_c`: local calibration dataset of client `c`
- `f(x)`: frozen global model producing logits for input `x`
- `s_i(x) = softmax(f(x))[i]`: predicted score for class `i`
- `y_true(x)`: ground-truth label of `x`
- `B = 20`: number of histogram bins (uniform edges on [0, 1])
- `alpha_bpr`: benign false-positive budget (fraction of benign records allowed to be flagged as attack)
- `min_clients`: minimum number of clients that must have a class in calibration for a threshold to be estimated (default: 3)

## 3. Step-by-Step Algorithm

### Step 3.1 — Client class presence determination

For each client `c`, the set of locally present classes in calibration is:

```
P_c = { c_i : exists (x, y) in D_c such that y = c_i }
```

A class `i` is "present" in client `c` iff client `c` has at least 1 calibration record of class `i`.

### Step 3.2 — Local class-conditional score histograms

For each client `c` and each class `i` in `P_c`:

1. Select calibration records where `y_true == i` (true-positive-class records for class `i`).
2. For each such record, compute `s_i(x)` — the softmax score for class `i`.
3. Build a histogram `H_c[i]` over `B` uniform bins on [0, 1] of these scores.
4. Also collect the benign scores for class `i`: for all calibration records where `y_true != i` (but `y_true == Benign` if benign class exists), compute `s_i(x)` and build histogram `H_benign_c[i]`.

Formally:
```
TP_c[i] = { s_i(x) : (x, y) in D_c, y = i }
FP_c[i] = { s_i(x) : (x, y) in D_c, y = Benign }
H_c[i] = histogram(TP_c[i], bins=B, range=[0,1])
H_benign_c[i] = histogram(FP_c[i], bins=B, range=[0,1])
```

### Step 3.3 — Histogram aggregation at server

For each class `i`, let `C_i = { c : i in P_c }` be the set of clients that have class `i` in calibration.

Three weighting schemes are supported:

**Uniform weighting:**
```
w_c = 1 / |C_i|  for c in C_i
```

**Sample-count weighting:**
```
w_c = |TP_c[i]| / sum_{c' in C_i} |TP_c'[i]|
```

**Class-presence-aware weighting (CAA-FedCal proposed):**
```
w_c = 1 / |C_i|  for c in C_i
```
(equivalent to uniform over eligible clients, but only eligible clients contribute — clients without class `i` contribute zero)

The aggregated histograms:
```
H_agg[i] = sum_{c in C_i} w_c * H_c[i]
H_benign_agg[i] = sum_{c in C_i} w_c * H_benign_c[i]
```

Normalized histograms (each sums to 1 over bins):
```
H_agg[i][b] /= sum_b H_agg[i][b]
H_benign_agg[i][b] /= sum_b H_benign_agg[i][b]
```

### Step 3.4 — Class-specific threshold selection

For each class `i` with `|C_i| >= min_clients` and sufficient calibration records:

For each bin edge threshold `t` (candidate threshold from histogram bin edges):

1. Compute True Positive Rate for class `i`:
   ```
   TPR_i(t) = sum_{b: bin_edge(b) >= t} H_agg[i][b]
   ```

2. Compute False Positive Rate for benign records predicted as class `i`:
   ```
   FPR_i(t) = sum_{b: bin_edge(b) >= t} H_benign_agg[i][b]
   ```

3. Select the threshold `t_i*` that maximizes TPR_i(t) subject to FPR_i(t) <= alpha_bpr.

If no threshold satisfies the benign-FPR constraint, set `t_i* = 1.0` (never predict class `i` based on threshold — falls back to argmax).

### Step 3.5 — Benign false-positive constraint enforcement

The benign-FPR constraint is enforced globally across all classes. The total benign false-positive rate is:

```
FPR_benign_total = sum_{i != Benign} P(y = i | y = Benign) * FPR_i(t_i*)
```

However, since we select per-class thresholds independently, the practical enforcement is:

For each class `i != Benign`, require `FPR_i(t_i*) <= alpha_bpr`. This ensures that for any single attack class, at most `alpha_bpr` fraction of benign records are misclassified as that class. The aggregate benign→attack rate across all classes is bounded by `sum_i alpha_bpr` but each individual class respects the per-class budget.

The configuration parameter `alpha_bpr` is set to 0.05 (5% benign FPR per class).

### Step 3.6 — Minimum client coverage

A class `i` is eligible for threshold estimation only if `|C_i| >= min_clients` (default: 3). This ensures thresholds are not based on a single client's potentially idiosyncratic calibration data.

### Step 3.7 — Insufficient calibration records

If a class `i` has `|C_i| >= min_clients` but the total calibration records for class `i` across eligible clients is below a minimum (e.g., 10 total records), the threshold for class `i` is not estimated and the class falls back to argmax decision.

### Step 3.8 — Class absent from all eligible clients

If `|C_i| = 0` (class `i` absent from all client calibration sets) or `|C_i| < min_clients`, no threshold is estimated for class `i`. Prediction for class `i` falls back to standard argmax over all class scores.

### Step 3.9 — Prediction resolution

For a test record `x` with softmax scores `s = (s_0, ..., s_9)`:

1. Compute argmax class: `c_argmax = argmax_i s_i`
2. Check threshold satisfaction for each eligible class `i` (i != Benign, i has threshold `t_i*`):
   - Class `i` is "threshold-satisfied" if `s_i >= t_i*`

3. Decision rules:
   - **If no non-benign class is threshold-satisfied**: predict argmax class (which may be Benign or any class).
   - **If exactly one non-benign class is threshold-satisfied**: predict that class.
   - **If multiple non-benign classes are threshold-satisfied**: predict the threshold-satisfied class with the highest score `s_i`. Ties broken by lower class index.
   - **If Benign is argmax and a non-benign class is threshold-satisfied**: the threshold-satisfied class takes precedence (attack detection priority).
   - **If Benign is argmax and no non-benign class is threshold-satisfied**: predict Benign.

In short: thresholds act as an attack-detection override. If any attack class's score exceeds its calibrated threshold, that class wins (highest score among threshold-satisfied). Otherwise, argmax rules.

## 4. Configuration

| Parameter | Value | Description |
|---|---|---|
| `num_bins` | 20 | Histogram bins on [0, 1] |
| `min_clients` | 3 | Min clients with class in calibration |
| `min_calib_records` | 10 | Min total calibration records for class |
| `alpha_bpr` | 0.05 | Benign FPR budget per class |
| `calibration_split` | 15% | Fraction of client data used for calibration |

## 5. Data Usage Constraints

- Thresholds are computed using calibration partitions only.
- Test labels and test predictions are never used to choose thresholds.
- Validation partitions are not used for threshold selection.
- Histogram aggregation is not differential privacy and makes no privacy claims.

## 6. Fallback Behavior

| Situation | Fallback |
|---|---|
| `|C_i| < min_clients` | Class `i` uses argmax, no threshold |
| Total calibration records < `min_calib_records` | Class `i` uses argmax, no threshold |
| No threshold satisfies `FPR_i <= alpha_bpr` | Class `i` threshold set to 1.0 (never triggers) |
| No class threshold satisfied at inference | Argmax prediction |
| Multiple thresholds satisfied | Highest score among satisfied classes |
