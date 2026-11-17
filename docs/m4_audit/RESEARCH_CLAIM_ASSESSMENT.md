# M4 Audit — Research Claim Assessment

## Scope

Assessed whether the M3 results support the proposed contribution: CAA-FedCal (Class-Absence-Aware Federated Post-Hoc Calibration) for multiclass intrusion detection under Non-IID traffic.

## 1. What Is Demonstrated

### Demonstration 1: CAA-FedCal Improves Macro-F1 Over FedAvg

Across 3 seeds, CAA-FedCal achieves higher macro-F1 than standard FedAvg:

| Seed | FedAvg mF1 | CAA-FedCal mF1 | Improvement |
|---|---|---|---|
| 42 | 0.356955 | 0.389291 | +0.032336 |
| 123 | 0.343908 | 0.414761 | +0.070853 |
| 7 | 0.338100 | 0.385312 | +0.047212 |

Mean improvement: +0.050 (not statistically significant with n=3, 95% CI includes zero).

### Demonstration 2: CAA-FedCal Improves Mid-Tier Classes

CAA-FedCal shows the largest gains on mid-tier classes (Exploits, Fuzzers, Generic, Reconnaissance, Shellcode) rather than the rarest classes:

| Class | FedAvg F1 (seed 42) | CAA-FedCal F1 (seed 42) | Improvement |
|---|---|---|---|
| Exploits | 0.264 | 0.590 | +0.326 |
| Fuzzers | 0.131 | 0.425 | +0.294 |
| Generic | 0.663 | 0.520 | -0.143 |
| Reconnaissance | 0.701 | 0.693 | -0.008 |
| Shellcode | 0.800 | 0.611 | -0.189 |

Mixed results — some classes improve substantially, others degrade.

### Demonstration 3: Benign False-Positive Rate Remains Low

CAA-FedCal maintains a low benign FPR (~0.0018), well below the 0.05 per-class constraint.

### Demonstration 4: Centralized MLP Significantly Outperforms FedAvg

| Experiment | Macro-F1 |
|---|---|
| Centralized | 0.5623 ± 0.0149 |
| FedAvg | 0.3463 ± 0.0096 |
| CAA-FedCal | 0.3965 ± 0.0152 |

The federated gap (centralized - FedAvg ≈ 0.216) is much larger than the CAA-FedCal improvement over FedAvg (≈ 0.050).

## 2. What Is Only Hypothesized

### Hypothesis 1: Class-Presence-Aware Weighting Is Beneficial

The weighting ablation (uniform / sample-count / presence) produced identical results because all thresholds are 0.1 regardless of weighting. **The benefit of class-presence-aware weighting is NOT demonstrated** — the ablation is non-informative.

### Hypothesis 2: Pooled Calibration Is Inferior to CAA-FedCal

Pooled calibration degenerates to argmax because `min_clients=3` cannot be satisfied with a single pooled client. **This does not test whether pooled calibration per se is beneficial or harmful.**

### Hypothesis 3: Class-Specific Thresholds Are Meaningfully Different

All CAA-FedCal thresholds are exactly 0.1 across all classes and all seeds. **The thresholds do not meaningfully differentiate between classes**, calling into question whether the "class-specific" aspect of CAA-FedCal provides any benefit over a single global threshold.

### Hypothesis 4: CAA-FedCal Helps Rare Classes

Worms (164 total records, 33 test) and Analysis (2,299 total, 460 test) have F1=0.000 across all experiments and seeds. **CAA-FedCal does not help the rarest classes.**

## 3. What Cannot Be Claimed

| Claim | Status | Reason |
|---|---|---|
| "CAA-FedCal significantly improves macro-F1" | Cannot claim | 95% CI includes zero with n=3 |
| "Class-presence-aware weighting is essential" | Cannot claim | Ablation is non-informative (all weightings identical) |
| "Pooled calibration is inferior" | Cannot claim | Pooled calibration degenerates to argmax due to min_clients filter |
| "Class-specific thresholds are meaningful" | Cannot claim | All thresholds are 0.1 (no differentiation) |
| "CAA-FedCal helps rare classes (Worms, Analysis)" | Cannot claim | F1=0.000 for both classes across all seeds |
| "CAA-FedCal is novel" | Cannot claim | Novelty gate still pending; literature review incomplete |

## 4. Most Important Limitations

### Limitation 1: Only 3 Seeds

With n=3, statistical power is extremely low. The macro-F1 confidence interval includes zero. At least 5-10 seeds are needed for reliable conclusions.

### Limitation 2: All Thresholds Are 0.1

The uniform threshold of 0.1 across all classes suggests the threshold selection is not providing class-specific differentiation. This may mean:
- The benign-FPR constraint (0.05) is too loose
- The histogram binning (20 bins) is too coarse
- The score distributions are similar across classes

### Limitation 3: Rarest Classes Undetectable

Worms (33 test records) and Analysis (460 test records) have F1=0.000. CAA-FedCal cannot create information that doesn't exist — with so few calibration records, no meaningful threshold can be estimated for these classes.

### Limitation 4: CPU-Only Training, Reduced Rounds

FedAvg was run for 15 rounds (reduced from 30) and Centralized for 30 epochs (reduced from 90) due to CPU runtime constraints. Results may differ with full training.

### Limitation 5: Within-Client Feature Overlap

Duplicate feature vectors appear across client train/validation/calibration partitions (most notably Client 1 with ~33K overlapping vectors). This may slightly inflate calibration-score reliability.

### Limitation 6: No Novelty Claim Verified

The M1 novelty gate is still pending. The CAA-FedCal contribution has not been confirmed as novel relative to existing federated IDS and calibration research.

## 5. Differentiation from Existing Research

### Similar Work

- **Laridi et al. (2024, Scientific Reports):** Federated threshold estimation from aggregated score histograms under benign-FPR constraint — the original proposal was rejected due to overlap with this work.
- **Maddock et al. (2025, arXiv:2510.01987):** Related federated calibration work.

### CAA-FedCal's Claimed Differentiation

1. **Class-absence awareness:** Explicitly handles classes absent from some clients' calibration sets
2. **Class-presence-aware weighting:** Only clients with a class in calibration contribute to that class's threshold
3. **NF-UNSW-NB15-v2 dataset:** Uses the UQ NetFlow v2 rendering rather than original UNSW-NB15

### Assessment

The differentiation is **modest and unverified**:
- Class-absence awareness is a reasonable design choice but its benefit is not demonstrated (all classes are eligible in this experiment)
- Class-presence-aware weighting produces identical results to other weightings (ablation is non-informative)
- NF-UNSW-NB15-v2 is a different dataset rendering but the core problem (multiclass IDS calibration) is the same

**Cannot claim novelty** without completing the novelty gate literature review.

## 6. Overall Assessment

| Aspect | Assessment |
|---|---|
| CAA-FedCal improves macro-F1 over FedAvg | Demonstrated (+0.050), but not statistically significant (n=3) |
| Class-presence-aware weighting is beneficial | Not demonstrated (ablation non-informative) |
| Pooled calibration is inferior | Not tested (degenerates to argmax) |
| Class-specific thresholds are meaningful | Not demonstrated (all thresholds = 0.1) |
| Rare classes are helped | Not demonstrated (F1=0.000 for Worms, Analysis) |
| Novelty claim | Cannot claim (novelty gate pending) |

## 7. Recommendations

1. **Increase seeds to 5-10** before claiming any statistical improvement
2. **Tighten benign-FPR constraint** (e.g., alpha_bpr=0.01) to create meaningful threshold differentiation
3. **Investigate why all thresholds are 0.1** — this may indicate a threshold selection issue
4. **Complete the novelty gate** before making any novelty claims
5. **Consider whether CAA-FedCal's complexity is justified** given the modest, non-significant improvement over FedAvg

## Conclusion

The results demonstrate a **modest, non-statistically-significant improvement** in macro-F1 from CAA-FedCal over standard FedAvg. The claimed mechanisms (class-presence-aware weighting, class-specific thresholds) are not validated by the ablation analysis. No novelty claim can be made at this time.
