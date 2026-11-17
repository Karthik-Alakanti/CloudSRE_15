# Methodology (working specification)

## Experimental unit

One federated multiclass network IDS experiment on **NF-UNSW-NB15-v2** (University of Queensland, UQRDM). Ten simulated clients are created from the stratified training split using class-wise Dirichlet allocation with alpha=0.3 and a fixed seed.

NF-UNSW-NB15-v2 is a single combined NetFlow v2 CSV (2,390,275 rows, 45 columns) with no official train/test partition. The pipeline creates a reproducible stratified 80/20 train/test split by the `Attack` column (seed=42) before preprocessing and partitioning.

## Labels

Use `Attack` as the multiclass target. The ten classes are: Analysis, Backdoor, Benign, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms. Actual spellings and counts are read from the CSV and preserved in metadata. Do not hard-code category counts or drop rare labels.

The binary `Label` column (0=Benign, 1=Attack) is excluded from model inputs. There is no `id` column in NF-UNSW-NB15-v2.

## Feature policy

- Exclude target columns `Attack`, `Label`.
- No `id` column exists in NF-UNSW-NB15-v2.
- Exclude IP address columns `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` by default (identity/leakage reduction).
- No `Stime`/`Ltime` timestamp columns exist in NF-UNSW-NB15-v2.
- Keep remaining candidate traffic features: ports, protocol, flow counters, packet sizes, TCP flags, durations, throughput, retransmissions, DNS/FTP/ICMP fields.
- Fail if required target column is absent.
- The implementation records exactly which columns were used/dropped.

This feature policy is a starting benchmark policy, not a claim that every remaining feature is safe from dataset-specific artifacts. Inspect feature documentation (`NetFlow_v2_Features.csv`) and check for duplicates/constant columns before finalizing.

## Preprocessing

1. Read the single NF-UNSW-NB15-v2 CSV.
2. Create a reproducible stratified train/test split by `Attack` (test_fraction=0.20, seed=42). Every row is assigned to exactly one split.
3. Normalize label strings (trim whitespace only; do not merge labels silently).
4. Separate target from predictors.
5. Convert +/- infinity to missing.
6. Fit imputers, categorical one-hot encoder and numeric standardizer on training split records only.
7. Transform test split records with the already-fitted transformer.
8. Encode target labels using a mapping learned from training labels; fail if test contains an unseen label.
9. Persist transformer, feature names, label mapping, split metadata and transformed matrices.

The current implementation uses a central fit on the training split for a reproducible simulation. This is not a fully decentralized preprocessing protocol. A production FL implementation would need a separately designed local/secure preprocessing approach.

## Client partitioning

- Partition only the processed training split records.
- For each class, draw a Dirichlet allocation over 10 clients using alpha=0.3.
- Preserve each sample exactly once.
- Then split each client partition into train/validation/calibration subsets with target proportions 70/15/15.
- Rare classes may not occur in every client's subset. Record class counts and warn; do not fabricate samples.
- Calibration records must never be used for local gradient updates.
- The official/test split is not distributed among clients and is never used for calibration.

## Class absence under Dirichlet alpha=0.3

With alpha=0.3, 10 clients, and 10 classes, rare classes are structurally absent from many clients. Worms has only 131 training records; under Dirichlet partitioning it lands in a small subset of clients, and several clients have 0 Worms records in their calibration sets. This is expected and is recorded in `partition_metadata.json`.

**Calibration reliability:** Some clients have fewer than 5 calibration examples for a given class (reported as warnings by `validate_data.py`). When a class has n=1 in a client, all of it goes to train (0 validation, 0 calibration). When n=2, validation gets 0. This means some clients cannot estimate class-specific thresholds for absent-in-calibration classes. This is a known limitation, not a bug, and is addressed by the class-absence-aware calibration design (Candidate 1) which only aggregates from clients that have the class in calibration.

## Model architecture (M3)

**MLP**: 41 input features → 128 hidden (ReLU, dropout 0.2) → 64 hidden (ReLU, dropout 0.2) → 10 output classes (logits). Softmax applied at inference for class scores.

**Centralized MLP**: Trained on the full training split (1,912,220 records) for 30 epochs, Adam lr=0.001, batch size 256.

**FedAvg**: 10 clients, 15 communication rounds (reduced from 30 for CPU runtime), 3 local epochs per round, batch size 256, Adam lr=0.001. Weighted average by client training set size.

**Implemented experiments**:
- A. Centralized MLP (single-node baseline)
- B. Standard FedAvg (argmax decision)
- C. FedAvg + ordinary pooled calibration (all calibration data pooled centrally)
- D. FedAvg + CAA-FedCal (class-presence-aware weighting, documented in docs/CAAFEDCAL.md)
- E. Weighting ablation (uniform / sample_count / presence)

CAA-FedCal is a post-training calibration stage. It does not modify model weights. Each client computes class-conditional score histograms from its calibration partition. The server aggregates histograms with class-presence-aware weighting and selects class-specific thresholds under a benign false-positive constraint (alpha_bpr=0.05). At inference, if any attack class score exceeds its calibrated threshold, that class wins; otherwise argmax applies.

## Planned comparisons (later)

- Centralized MLP, argmax.
- FedAvg, argmax.
- FedProx, argmax.
- FedAvg with centrally pooled calibration (diagnostic upper reference; not privacy-preserving).
- FedAvg with proposed aggregated client-score calibration.

Do not implement these until data validation and novelty gate pass.

## Reproducibility

Record seed, file hashes, row counts, feature columns, class mapping, transformer settings, client indices/counts, split counts, per-client split seeds, and package versions. All random procedures must use explicit seeds.
