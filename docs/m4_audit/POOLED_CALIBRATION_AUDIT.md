# M4 Audit — Pooled Calibration Investigation

## Scope

Investigated why pooled calibration and standard FedAvg produce identical reported metrics.

## 1. Reported Results

| Seed | FedAvg (argmax) | Pooled Calibration |
|---|---|---|
| 42 | acc=0.973769, mF1=0.356955 | acc=0.973769, mF1=0.356955 |
| 123 | acc=0.972399, mF1=0.343908 | acc=0.972399, mF1=0.343908 |
| 7 | acc=0.972616, mF1=0.338100 | acc=0.972399→0.972616, mF1=0.338100 |

Metrics are **exactly identical** for all 3 seeds.

## 2. Root Cause: min_clients Filter

### Implementation (`src/calibration/calibration.py:aggregate_histograms`)

Pooled calibration treats all calibration data as belonging to a single pseudo-client (client_id=0). The `aggregate_histograms` function determines eligibility:

```
eligible_clients_i = { c : class i in client c's calibration set }
n_eligible_i = |eligible_clients_i|
eligible_i = (n_eligible_i >= min_clients)
```

With `min_clients = 3` and only 1 pseudo-client in pooled calibration:

```
n_eligible_i = 1 for all classes i
1 < 3 → NOT ELIGIBLE
```

### Result

All 9 non-benign classes are **ineligible** for threshold estimation in pooled calibration. The `select_thresholds` function marks all classes as `eligible=False` with `threshold=None`.

## 3. Prediction Behavior

When all thresholds are ineligible/None, `predict_with_thresholds_vectorized` produces:

- `thr_array` = all `inf` (no class has a finite threshold)
- `satisfied = probs >= inf` = all `False`
- `any_satisfied` = all `False`
- Predictions = pure argmax

### Verified: 0 Prediction Differences

Re-running inference confirmed that pooled calibration predictions are **identical** to pure argmax predictions for all 478,055 test records across all 3 seeds.

| Seed | Pooled vs Argmax Differences |
|---|---|
| 42 | 0 / 478,055 |
| 123 | 0 / 478,055 |
| 7 | 0 / 478,055 |

## 4. Threshold Details (Pooled Calibration)

All classes report:
```json
{
  "threshold": null,
  "eligible": false,
  "reason": "only 1 clients have class X in calibration (need 3)"
}
```

## 5. Conclusion

**Pooled calibration produces identical metrics to FedAvg because no thresholds can be estimated.** The `min_clients=3` requirement cannot be satisfied with a single pooled client (n_eligible=1). All classes fall back to argmax, making pooled calibration mathematically equivalent to standard FedAvg.

This is **not a bug** — it is a legitimate consequence of the minimum client coverage requirement. However, it means:

1. **Pooled calibration does not test whether pooling calibration data helps.** It tests whether a single pooled client can satisfy min_clients=3, which it cannot.
2. **The "pooled calibration" experiment is not a valid comparison** for evaluating whether centralized calibration would improve over FedAvg.
3. **A valid pooled calibration test would require either:** (a) lowering min_clients to 1, or (b) not applying the min_clients filter to pooled calibration.

## 6. Recommendation

The claim "pooled calibration provides zero improvement over FedAvg" is technically correct but misleading. The correct interpretation is: "pooled calibration with min_clients=3 degenerates to argmax because a single client cannot meet the minimum coverage requirement." This does not tell us whether pooling calibration data *per se* would help or hurt.
