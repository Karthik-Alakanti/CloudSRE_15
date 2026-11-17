# M4 Audit — Weighting Ablation

## Scope

Investigated why uniform, sample-count, and presence weighting all produce identical results in the CAA-FedCal weighting ablation experiment.

## 1. Reported Results

All three weighting schemes produce identical metrics across all 3 seeds:

| Seed | Uniform | Sample-Count | Presence |
|---|---|---|---|
| 42 | acc=0.978578, mF1=0.389291 | acc=0.978578, mF1=0.389291 | acc=0.978578, mF1=0.389291 |
| 123 | acc=0.979751, mF1=0.414761 | acc=0.979751, mF1=0.414761 | acc=0.979751, mF1=0.414761 |
| 7 | acc=0.977467, mF1=0.385312 | acc=0.977467, mF1=0.385312 | acc=0.977467, mF1=0.385312 |

## 2. Aggregation Equations

### Implementation (`src/calibration/calibration.py:aggregate_histograms`)

For each class `i`, eligible clients `C_i` contribute weighted histograms:

```
H_agg[i] = sum_{c in C_i} w_c * H_c[i]
H_benign_agg[i] = sum_{c in C_i} w_c * H_benign_c[i]
```

Where each client histogram `H_c[i]` is **normalized to sum to 1** before aggregation:

```
H_c[i][b] = count_c[i][b] / sum_b count_c[i][b]
```

### Weighting Schemes

| Scheme | Weight for client c |
|---|---|
| Presence | `w_c = 1 / |C_i|` (uniform over eligible clients) |
| Sample-count | `w_c = |TP_c[i]| / sum_{c' in C_i} |TP_c'[i]|` (proportional to calibration record count) |
| Uniform | `w_c = 1 / num_clients` (uniform over ALL clients, but only eligible ones contribute) |

## 3. Why Results Are Identical

### Mathematical Reason

The weighting schemes differ in the weights `w_c`, BUT the final thresholds depend on the **shape** of the aggregated histogram, not just the weights. After threshold selection (maximizing TPR subject to FPR <= 0.05), all three weightings produce the same threshold of 0.1 for every class.

### Verification of Threshold Identity

The saved calibration files confirm that all three weightings produce identical thresholds:

- All thresholds are 0.1 for all classes
- All classes are eligible (>= min_clients=3)
- All classes pass min_calib_records=10

### Why Threshold Selection Ignores Weighting Differences

The threshold selection algorithm iterates through bin edges from lowest to highest and selects the first threshold where FPR <= 0.05 while maximizing TPR. For all classes and all weightings:

1. At threshold 0.05: FPR > 0.05 (constraint violated)
2. At threshold 0.10: FPR <= 0.05 (constraint satisfied)

Since 0.10 is the first bin edge that satisfies the benign-FPR constraint, it is selected regardless of the weighting scheme. The weighting affects the histogram shape but not enough to change which bin edge first satisfies FPR <= 0.05.

### Numerical Evidence

The intermediate aggregated histograms differ between weightings (different `w_c` values produce different weighted averages), but the threshold selection is insensitive to these differences because:

- The benign-FPR constraint (0.05) is loose enough that threshold 0.1 always satisfies it
- The TPR maximization picks the lowest valid threshold
- All weightings agree that 0.1 is the first valid threshold

## 4. Conclusion

**The weighting ablation is non-informative.** All three weighting schemes produce identical thresholds (0.1 for all classes) because the threshold selection is driven by the benign-FPR constraint, which consistently selects 0.1 as the first valid threshold across all weightings.

**Do not claim** that the three strategies perform equally well based on meaningful differences — they produce identical results because the ablation does not actually vary the final thresholds. The ablation tests whether weighting affects histogram aggregation, but the downstream threshold selection is insensitive to these histogram differences.

**Recommendation:** Either (a) tighten the benign-FPR constraint (e.g., alpha_bpr=0.01) to create differentiation between weightings, or (b) remove the ablation as it does not provide informative comparison.
