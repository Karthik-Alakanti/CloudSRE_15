# Experiment protocol and stage gates

## Current stage: M3 (complete)

Model training and calibration experiments have been executed for 3 seeds.

## M1 — literature and novelty review (status: direction selected, gate pending)

Original proposal (federated class-specific threshold calibration from aggregated score histograms under benign-FPR constraint): **NOT APPROVED.**
- Core mechanism overlaps Laridi et al. (2024, Scientific Reports) and Maddock et al. (2025, arXiv:2510.01987).
- Does not address class absence under Dirichlet α=0.3 partitioning.

Selected direction: **Candidate 1 — Class-Absence-Aware Federated Calibration (CAA-FedCal)**.
- Documented in `docs/CANDIDATE_DIRECTIONS.md` and `docs/CAAFEDCAL.md`.
- Novelty gate still pending: the specific combination (class-absence-aware federated calibration + benign-FP constraint + NF-UNSW-NB15-v2 + Dirichlet α=0.3) must be confirmed as not covered by existing work before any novelty claim.

- [x] Read the full 2024 summary-statistics federated threshold paper (Laridi et al., 2024) — DONE (PMC11535521, arXiv:2410.09284)
- [x] Compare it against the proposed supervised multiclass IDS calibration method — DONE
- [x] Search 2023–2026 literature for federated score calibration and per-class thresholds — DONE
- [x] Create a claim-to-evidence table in `docs/NOVELTY_REVIEW.md` — DONE
- [x] Decide whether to proceed, narrow, or replace the contribution — DONE (replaced with Candidate 1)
- [ ] Confirm Candidate 1's specific combination is not covered by existing work — PENDING (requires full-text review of Maddock et al. 2025 and PROTEAN 2025)

## M2 — dataset pipeline (COMPLETE for NF-UNSW-NB15-v2)

- [x] Obtain NF-UNSW-NB15-v2 from University of Queensland (UQRDM) — placed at `data/raw/unsw-nb15/fe6cb615d161452c_MOHANAD_A4706/data/NF-UNSW-NB15-v2.csv`
- [x] Verify file identity: SHA-256 `05019e3ac8d55b3f074f9f042623bbbb1af2527b8b78558c69353ccb2ccefa3f`, 2,390,275 rows × 45 columns
- [x] Inspect schema: target column `Attack`, binary label `Label`, no `id` column, IP columns `IPV4_SRC_ADDR`/`IPV4_DST_ADDR`
- [x] Create reproducible stratified 80/20 train/test split (seed=42, test_fraction=0.20)
- [x] Label frequencies, class frequencies, missingness, infinities, duplicates recorded
- [x] Fit preprocessing on training split only
- [x] Transform test split with training-fitted transformer
- [x] Partition training rows across 10 clients with Dirichlet alpha=0.3
- [x] Split each client into train/validation/calibration (70/15/15)
- [x] Run validation script — PASSED
- [x] Generate class-distribution and client-distribution visualizations
- [x] Save dataset hashes and pipeline configuration
- [x] Update documentation for NF-UNSW-NB15-v2

## M3 — model training and calibration (COMPLETE for 3 seeds)

**Configuration:**
- 10 federated clients, Dirichlet alpha=0.3
- 41 transformed features, 10 classes
- MLP: 128 → 64 hidden units, ReLU, dropout 0.2
- Adam, learning rate 0.001, batch size 256
- Centralized: 30 epochs; FedAvg: 15 rounds × 3 local epochs (reduced from 30 rounds for CPU runtime)
- 3 random seeds: 42, 123, 7

**Experiments implemented:**
- A. Centralized MLP (single-node baseline)
- B. Standard FedAvg (argmax decision)
- C. FedAvg + ordinary pooled calibration
- D. FedAvg + CAA-FedCal (class-presence-aware weighting)
- E. Weighting ablation (uniform / sample_count / presence)

**Key results (mean ± std across 3 seeds):**
| Experiment | Accuracy | Macro-F1 | Weighted-F1 | Benign FPR |
|---|---|---|---|---|
| Centralized | 0.9877 ± 0.0001 | 0.5623 ± 0.0149 | 0.9865 ± 0.0002 | 0.0025 ± 0.0002 |
| FedAvg | 0.9729 ± 0.0007 | 0.3463 ± 0.0096 | 0.9649 ± 0.0016 | 0.0001 ± 0.0000 |
| Pooled Cal | 0.9729 ± 0.0007 | 0.3463 ± 0.0096 | 0.9649 ± 0.0016 | 0.0001 ± 0.0000 |
| CAA-FedCal | 0.9786 ± 0.0012 | 0.3965 ± 0.0152 | 0.9767 ± 0.0016 | 0.0018 ± 0.0003 |

**Key findings:**
1. Centralized MLP significantly outperforms FedAvg (macro-F1: 0.56 vs 0.35)
2. CAA-FedCal improves macro-F1 over FedAvg by ~0.05 (consistent across seeds)
3. Pooled calibration provides zero improvement over FedAvg argmax (identical results)
4. CAA-FedCal improves mid-tier classes (Exploits, Fuzzers) but not rarest (Worms, Analysis still F1=0)
5. Overlapping train/test feature vectors (0.60% of test) have negligible impact on metrics (<0.0015 macro-F1 difference)
6. All three CAA-FedCal weighting schemes give identical results after histogram normalization

**Artifacts:**
- `results/m3/models/` — model checkpoints (FedAvg, Centralized, calibrated variants)
- `results/m3/checkpoints/` — per-round/per-epoch checkpoints (resumable)
- `results/m3/calibration/` — CAA-FedCal and pooled calibration thresholds
- `results/m3/metrics/` — all experiment results (JSON)
- `results/m3/sensitivity/` — sensitivity analysis (duplicate feature exclusion)
- `results/m3/figures/` — visualizations

## M2 corrections applied (documented in `docs/M2_AUDIT.md`)

1. Added cross-split duplicate-record leakage check to `scripts/inspect_dataset.py` (EXPERIMENT_PROTOCOL stop condition).
2. Added package version metadata to `src/data/pipeline.py` preprocessing metadata (METHODOLOGY reproducibility requirement).
3. Added per-client split seeds to `scripts/partition.py` partition metadata (full reproducibility).
4. Documented n=1/n=2 class calibration absence as a known limitation (not a bug).

## M2 acceptance criteria

`python scripts/validate_data.py` must pass all hard checks:

- Train and test feature matrices have identical feature dimension and finite values.
- No target/identifier columns appear in model feature list.
- Train and test source files are distinct and have expected schema.
- Every training row is assigned to exactly one client.
- No client split overlaps another client split.
- For each client, train/validation/calibration indices are disjoint.
- The official test set is not used in any client partition.
- Label mapping is consistent and no test labels are unseen.
- Data shapes and metadata agree.

Warnings (e.g. a rare class absent from a client) must be reviewed, not silently ignored.

## Stop conditions

Stop and ask for review if:

- files are not the official intended dataset;
- labels or columns differ unexpectedly;
- there are duplicate records across train/test that could indicate leakage;
- preprocessing produces non-finite values;
- class labels are missing or unexpectedly merged;
- partitioning loses or duplicates rows;
- rare class counts make the planned calibration statistically unstable.

## No-training gate

The no-training gate has been lifted. M3 experiments are complete. M4 (further analysis, report writing) may proceed with user approval.
