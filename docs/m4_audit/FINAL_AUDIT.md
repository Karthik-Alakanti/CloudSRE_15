# M4 Audit — Final Audit Report

## Summary

This audit reviewed the complete M3 implementation: models, calibration algorithm, experiments, results, and methodology. All 25 existing tests pass. No new training was performed. No existing results were modified.

## Audit Documents

| Document | Status | Key Finding |
|---|---|---|
| `RESULT_INTEGRITY.md` | Complete | All metrics verified; centralized models missing from `models/` dir |
| `CAA_FEDCAL_AUDIT.md` | Complete | All thresholds are 0.1; calibration never leaks to training/test |
| `POOLED_CALIBRATION_AUDIT.md` | Complete | Pooled calibration degenerates to argmax (min_clients=3 not met) |
| `ABLATION_AUDIT.md` | Complete | Weighting ablation is non-informative (all weightings identical) |
| `LEAKAGE_AUDIT.md` | Complete | No preprocessing/test leakage; within-client feature overlap exists |
| `STATISTICAL_ASSESSMENT.md` | Complete | Macro-F1 improvement not statistically significant (n=3) |
| `RESEARCH_CLAIM_ASSESSMENT.md` | Complete | No novelty claim supported; limitations documented |

## Critical Issues Requiring Attention

### Issue 1: All CAA-FedCal Thresholds Are 0.1

Every class threshold is exactly 0.1 across all 3 seeds. This suggests the threshold selection consistently picks the first bin edge satisfying FPR <= 0.05. The class-specific differentiation is not demonstrated.

**Recommended action:** Investigate threshold selection logic; consider tightening alpha_bpr.

### Issue 2: Pooled Calibration Is Not a Valid Comparison

Pooled calibration with min_clients=3 degenerates to argmax (1 client < 3 required). The experiment does not test whether pooling calibration data helps.

**Recommended action:** Either lower min_clients for pooled calibration or remove the pooled calibration experiment.

### Issue 3: Weighting Ablation Is Non-Informative

Uniform, sample-count, and presence weighting produce identical results because threshold selection is insensitive to weighting differences.

**Recommended action:** Remove or redesign the ablation experiment.

### Issue 4: Statistical Power

Only 3 seeds. The macro-F1 improvement (+0.050) is not statistically significant (95% CI includes zero).

**Recommended action:** Increase to 5-10 seeds before claiming improvement.

### Issue 5: Rare Classes Undetectable

Worms (33 test records) and Analysis (460 test records) have F1=0.000 across all experiments.

**Recommended action:** Acknowledge this limitation; consider whether rare classes can be addressed at all.

### Issue 6: Centralized Models Missing from `models/` Directory

Centralized final models exist in `checkpoints/` but not in `models/`.

**Recommended action:** Copy centralized models to `models/` for consistency, or update documentation.

## Tests

All 25 existing tests pass:
- 16 calibration tests (`tests/test_calibration.py`)
- 9 pipeline tests (`tests/test_pipeline.py`)

No new tests were added. The audit verified existing implementation properties without changing methodology.

## Overall Conclusion

M3 is implemented and results are internally consistent. However, the research contribution is **not yet supported by statistically significant evidence**. The CAA-FedCal improvement over FedAvg is modest (+0.050 macro-F1) and not statistically significant with 3 seeds. The claimed mechanisms (class-presence-aware weighting, class-specific thresholds) are not validated by the ablation analysis.

**Recommendation:** Do not draft a paper or claim novelty until:
1. More seeds are run (5-10)
2. Threshold selection is investigated (all thresholds = 0.1)
3. The novelty gate is completed
4. The ablation and pooled calibration experiments are redesigned

## Files Preserved

All existing artifacts are preserved:
- `results/m3/models/` — model checkpoints
- `results/m3/checkpoints/` — per-round/per-epoch checkpoints
- `results/m3/calibration/` — CAA-FedCal and pooled thresholds
- `results/m3/metrics/` — all experiment results (JSON)
- `results/m3/sensitivity/` — sensitivity analysis
- `results/m3/figures/` — visualizations
- `data/raw/` — raw dataset (untouched)
- `data/processed/` — preprocessed data (untouched)
- `data/partitions/` — client partitions (untouched)
