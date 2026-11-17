# Research problem and novelty gate

## Working title
**Federated Multiclass Intrusion Detection with Privacy-Conscious Distributed Score Calibration under Non-IID Traffic**

## Core problem
Federated learning permits multiple network participants to train a shared IDS without centralizing their raw traffic. However, heterogeneous local traffic distributions and class imbalance can produce a global classifier whose overall score obscures weak detection of underrepresented attack classes.

## Provisional research question
Can class-specific decision thresholds estimated from aggregated client-local score summaries improve minority-class detection while keeping the benign false-positive rate below a predefined operational budget?

## Provisional hypothesis
A post-training distributed calibration stage may improve the operating point of a federated multiclass classifier without changing its learned model parameters. It may improve minority attack recall at a controlled benign false-positive rate. This is a hypothesis to test, not a promised result.

## Why the earlier class-aware focal-loss idea was dropped
The supplied FL-IIDS paper already includes Class Gradient Balancing Loss and Sample Label Smoothing Loss. FIST-G2 explicitly addresses minority-traffic bias using server-side GAN boundary-sample fine-tuning. Rebranding class weighting/focal loss as the contribution would create substantial overlap.

## Proposed mechanism (not frozen)
1. Train a conventional multiclass global classifier using FedAvg.
2. Each client uses a disjoint local calibration subset not used for gradient updates.
3. Each client runs the frozen global model on its local calibration records.
4. Clients compute class-conditional score summaries (initial implementation candidate: fixed-bin histograms of the predicted score for each class, split into true-positive-class and other-class records).
5. The server aggregates summaries and selects class-specific thresholds subject to a benign false-positive constraint.
6. Evaluate the calibrated decision policy on the untouched official test set.

No raw calibration records or individual predictions are sent to the server. However, histograms can still leak information, especially for small counts. This is not differential privacy and must not be described as a formal privacy guarantee.

## Critical novelty gate — unresolved
A 2024 paper on federated anomaly detection using summary-statistics-based thresholding is directly relevant. Before asserting novelty, obtain and inspect its full text and compare:
- anomaly detection versus supervised multiclass classification;
- global versus per-class thresholds;
- score-statistic representation;
- client heterogeneity and class absence;
- false-positive constraints;
- privacy threat model and any protection of summary statistics;
- evaluation datasets and baselines.

Also search 2023–2026 work on federated calibration, class-wise thresholding, cost-sensitive multiclass decisions, and intrusion detection.

**Do not write “first”, “novel”, or “privacy-preserving” as established facts until this gate is resolved.** If the idea is already covered, change the contribution before implementing the calibration module.

## Scope boundaries
- Primary task: network intrusion detection, multiclass `attack_cat`.
- Dataset: UNSW-NB15 official train/test partitions.
- Federated simulation: 10 clients, Dirichlet concentration alpha=0.3 (initial protocol setting).
- Initial classifier planned later: lightweight MLP.
- Main FL baselines planned later: FedAvg and FedProx.
- Calibration proposal: class-specific thresholds from aggregated client-local score summaries.
- Excluded from initial scope: concept drift, incremental learning, GAN augmentation, SHAP, poisoning defense, client drop-out studies, and cross-dataset generalization.

## Source-derived constraints from the supplied base papers
- FL-IIDS uses dynamic example memory, Class Gradient Balancing Loss, Sample Label Smoothing Loss, and relay-client reconstruction. Its conclusion identifies Non-IID federated incremental IDS as future work.
- FIST-G2 addresses client drift using gradient memory and minority-traffic bias using GAN boundary samples.
- XAI-FedIDS explicitly separates client training, trust-validation, and held-out test subsets to avoid using the same validation data for weighting and reported performance.
- The cloud IDS review supplied by the user is labelled a working draft. Treat it as a direction-setting document, not a systematic review.

## Decision status
M1 is **not complete** until the relevant threshold-estimation paper and recent related work have been compared. M2 data engineering can proceed independently.
