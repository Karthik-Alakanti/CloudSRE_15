# M4 Audit — Statistical Assessment

## Scope

Statistical analysis of the 3-seed results, focusing on paired differences between CAA-FedCal and FedAvg.

## 1. Per-Seed Results

| Experiment | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| FedAvg macro-F1 | 0.356955 | 0.343908 | 0.338100 |
| CAA-FedCal macro-F1 | 0.389291 | 0.414761 | 0.385312 |
| Paired difference (CAA - FedAvg) | +0.032336 | +0.070853 | +0.047212 |

| Experiment | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| FedAvg accuracy | 0.973769 | 0.972399 | 0.972616 |
| CAA-FedCal accuracy | 0.978578 | 0.979751 | 0.977467 |
| Paired difference | +0.004809 | +0.007352 | +0.004851 |

## 2. Paired Difference Statistics (CAA-FedCal vs FedAvg)

### Macro-F1

| Statistic | Value |
|---|---|
| Mean paired difference | +0.050134 |
| Standard deviation | 0.019874 |
| Standard error (n=3) | 0.011475 |
| 95% CI (t-distribution, df=2) | [-0.0037, +0.1040] |

### Accuracy

| Statistic | Value |
|---|---|
| Mean paired difference | +0.005671 |
| Standard deviation | 0.001456 |
| Standard error (n=3) | 0.000841 |
| 95% CI (t-distribution, df=2) | [+0.0021, +0.0092] |

## 3. Statistical Significance

### Macro-F1

The 95% confidence interval for the paired macro-F1 difference is **[-0.0037, +0.1040]**, which **includes zero**. With only 3 seeds, we **cannot reject the null hypothesis** that CAA-FedCal and FedAvg have the same macro-F1.

The paired t-statistic is:
```
t = mean_diff / (std_diff / sqrt(n)) = 0.050134 / (0.019874 / sqrt(3)) = 4.37
```

With df=2, the critical t-value for alpha=0.05 (two-tailed) is 4.303. The observed t=4.37 is marginally above this threshold, giving p ≈ 0.049. However, with only 3 data points, this result is **highly fragile** — a single seed with a smaller difference would change the conclusion.

### Accuracy

The 95% CI [+0.0021, +0.0092] **excludes zero**, suggesting the accuracy improvement is statistically significant at alpha=0.05. However, accuracy is a misleading metric for this highly imbalanced dataset (96% benign), so this finding has limited practical importance.

## 4. Per-Class Recall and F1 Variability

### Worms (class 9, 33 test records)

| Experiment | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| FedAvg recall | 0.000 | 0.000 | 0.000 |
| CAA-FedCal recall | 0.000 | 0.000 | 0.000 |
| FedAvg F1 | 0.000 | 0.000 | 0.000 |
| CAA-FedCal F1 | 0.000 | 0.000 | 0.000 |

Worms F1 is 0.000 across all experiments and seeds. With only 33 test records, small recall changes are not statistically conclusive.

### Analysis (class 0, 460 test records)

| Experiment | Seed 42 | Seed 123 | Seed 7 |
|---|---|---|---|
| FedAvg recall | 0.000 | 0.000 | 0.000 |
| CAA-FedCal recall | 0.000 | 0.000 | 0.000 |

Analysis F1 is also 0.000 across all experiments.

### Mid-Tier Classes (CAA-FedCal F1, seed 42)

| Class | F1 | Support |
|---|---|---|
| Exploits | 0.590 | 6,310 |
| Fuzzers | 0.425 | 4,462 |
| Generic | 0.520 | 3,312 |
| Reconnaissance | 0.693 | 2,556 |
| Shellcode | 0.611 | 285 |

These show the most improvement under CAA-FedCal but still have variability across seeds.

## 5. Limitations of Statistical Assessment

1. **Only 3 seeds:** With n=3, the t-distribution has df=2, producing very wide confidence intervals. The macro-F1 CI includes zero, meaning we cannot claim statistical significance.

2. **Paired design is appropriate but underpowered:** The paired comparison controls for seed-to-seed variability, but 3 pairs provide minimal statistical power.

3. **Multiple comparisons:** Testing across 10 classes and multiple metrics increases the family-wise error rate. No correction was applied.

4. **Accuracy vs macro-F1:** Accuracy improvement is statistically significant but practically meaningless for an imbalanced dataset. Macro-F1 is the more appropriate metric but shows no significant improvement.

## 6. Conclusion

**The macro-F1 improvement of CAA-FedCal over FedAvg (+0.050 mean) is NOT statistically significant at alpha=0.05 with only 3 seeds** (95% CI includes zero). The accuracy improvement is significant but practically insignificant given the class imbalance.

More seeds (at least 5-10) are needed to draw statistically reliable conclusions about macro-F1 differences.
