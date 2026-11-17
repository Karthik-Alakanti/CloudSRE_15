# Candidate Research Directions

This document proposes three candidate research directions for the federated multiclass IDS project on UNSW-NB15. The original proposal (federated class-specific threshold calibration from aggregated score histograms under a benign-FPR constraint) is documented in `docs/NOVELTY_REVIEW.md` and remains available for comparison. M1 novelty was **NOT APPROVED** for the original proposal because the core mechanism (federated thresholding from aggregated client-local score statistics under Non-IID) is already covered by Laridi et al. (2024, Scientific Reports).

All three candidates below address the **rare-class problem created by Dirichlet Non-IID partitioning (alpha=0.3)**, under which rare attack classes (e.g., Worms) may be absent from many clients' local data. This makes per-client model training and per-client threshold estimation unreliable for those classes.

None of the candidates has been verified as novel. Each requires a full-text literature comparison before any novelty claim.

---

## Candidate 1: Class-Absence-Aware Federated Calibration (CAA-FedCal)

### 1. Proposed title
**Class-Absence-Aware Federated Post-Hoc Calibration for Multiclass Intrusion Detection under Non-IID Traffic**

### 2. Research question
Can a federated post-hoc calibration method that explicitly accounts for class absence across clients improve minority-class detection while keeping the benign false-positive rate within a predefined operational budget?

### 3. Exact methodological contribution
- Train a global multiclass classifier with FedAvg on the UNSW-NB15 training partition (10 clients, Dirichlet alpha=0.3).
- Each client runs the frozen global model on its **local calibration subset** (15% of its partition, disjoint from train/validation).
- Each client computes class-conditional score summaries **only for classes present in its local calibration set**; absent classes contribute nothing.
- The server aggregates per-class score histograms **weighted by the number of clients that actually have that class** in their calibration set (not by total sample count alone).
- Class-specific thresholds are selected only for classes with sufficient client coverage (configurable minimum, e.g., >= 3 clients).
- The benign false-positive budget is enforced at the server level during threshold selection.
- Evaluate on the untouched official test set.

### 4. Closest existing papers
| Paper | Venue/Year | Key details |
|---|---|---|
| Maddock, Cormode & Maple (2025) | arXiv:2510.01987 | "Private Federated Multiclass Post-hoc Calibration". Federated histogram binning and temperature scaling for multiclass calibration under non-IID. Weighted binning combines client histograms with a weighting scheme based on example counts per class. Datasets: seven benchmarks (not UNSW-NB15). Does not address class absence specifically. |
| Laridi, Palmer & Tam (2024) | Scientific Reports 14:26704 | Federated thresholding from summary statistics of autoencoder reconstruction errors under Non-IID. Binary anomaly detection, not multiclass supervised IDS. Single global threshold, not per-class. |
| MDPI Applied Sciences 16(2):801 (2026) | Applied Sciences 2026 | Per-class thresholds with benign-FP cap via centralized validation grid search on FL-IDS. Not federated calibration; not UNSW-NB15. |

### 5. What those papers already accomplish
- Maddock et al. (2025): Show that federated post-hoc calibration (binning/scaling) works in heterogeneous FL settings; weighting by example count helps stability under label skew.
- Laridi et al. (2024): Show that federated summary-statistic thresholding improves anomaly detection under Non-IID.
- MDPI Applied Sciences (2026): Show that per-class thresholds with a benign-FP cap improve minority-class F1 in centralized FL-IDS evaluation.

### 6. What remains unresolved
- None of the above explicitly handles **class absence**: when a class is missing from a client's calibration set, that client has no score summary for that class. Weighting by example count alone does not distinguish "class present but few examples" from "class entirely absent".
- No work combines federated class-conditional calibration with an explicit benign-FP constraint on **UNSW-NB15** with **Dirichlet alpha=0.3** partitioning where rare-class absence is structurally guaranteed.
- The interaction between class absence and threshold reliability (small-histogram instability) is not quantified in existing FL-IDS work.

### 7. Proposed experiment
1. Preprocess UNSW-NB15 train/test (training-only fit, test transform).
2. Partition train into 10 clients via Dirichlet alpha=0.3.
3. Split each client into train/validation/calibration (70/15/15).
4. Train global MLP with FedAvg (baseline).
5. Each client computes class-conditional score histograms on its calibration set for locally present classes only.
6. Server aggregates with class-presence weighting and selects class-specific thresholds under benign-FP budget.
7. Evaluate FedAvg (argmax) vs. CAA-FedCal on official test set.
8. Ablation: uniform weighting vs. presence-weighted aggregation; with/without minimum client coverage threshold.

### 8. Baselines
- Centralized MLP, argmax (trained on all training data).
- FedAvg, argmax (standard federated baseline).
- CAA-FedCal with uniform weighting (ablation).
- CAA-FedCal with presence weighting (proposed).

### 9. Evaluation metrics
- Primary: Macro-F1 on official test set.
- Secondary: Per-class recall (especially minority classes: Worms, Shellcode, Analysis), per-class FPR, benign FPR, overall FPR.
- Calibration: classwise expected calibration error (cwECE) on calibration sets.

### 10. Expected implementation complexity
Medium. Requires: FedAvg training loop, local calibration scoring, histogram construction, server-side aggregation with presence weighting, threshold search under FPR constraint. All implementable with numpy/scikit-learn. No GANs, transformers, or explainability.

### 11. Local computational requirements
Low. Lightweight MLP (128-64 hidden units), 10 clients, 30 rounds, local epochs=3, batch=256. Runs on CPU in minutes. UNSW-NB15 train ~175K rows, test ~82K rows.

### 12. Main risks and failure modes
- Rare classes (Worms) may be present in too few clients (<3) to support reliable threshold estimation; the method would skip thresholds for those classes, falling back to argmax.
- If benign FPR budget is too tight, all class-specific thresholds may be conservative, yielding no minority-recall improvement.
- Histogram binning with few calibration samples per class per client may produce unstable thresholds.
- Results may show no statistically significant improvement over FedAvg argmax.

---

## Candidate 2: Prototype-Guided Class-Specific Thresholding for Non-IID FL-IDS

### 1. Proposed title
**Prototype-Guided Class-Specific Thresholding for Federated Multiclass IDS under Dirichlet Non-IID Partitioning**

### 2. Research question
Can exchanging class prototypes (rather than score histograms) for locally present classes, combined with class-specific decision thresholds, improve rare-attack detection under Dirichlet Non-IID partitioning where rare classes are absent from many clients?

### 3. Exact methodological contribution
- Train a global multiclass classifier with FedAvg.
- Each client computes class prototypes (mean feature vector of correctly classified samples per class) on its local validation set, **only for classes present locally**.
- Server aggregates prototypes per class using only clients that have that class (class-presence-weighted prototype averaging).
- The global model's classification head is adjusted using aggregated prototypes: for each class, the decision boundary is shifted toward the aggregated prototype, and a class-specific threshold is applied to the prototype-adjusted score.
- Thresholds are selected on each client's local calibration set, then aggregated at the server (median across clients that have the class).
- Evaluate on the untouched official test set.

### 4. Closest existing papers
| Paper | Venue/Year | Key details |
|---|---|---|
| Chennoufi et al. (2025) | arXiv:2507.05524, ESORICS 2025 | "PROTEAN: Federated Intrusion Detection in Non-IID Environments through Prototype-Based Knowledge Sharing". Exchanges class prototypes among clients to improve detection under Non-IID. Datasets: IIoT and 5G-NIDD (not UNSW-NB15). No threshold component; prototypes used for feature alignment, not decision thresholds. |
| Sun et al. (2024) | arXiv:2408.07152, ISC 2024 | "FedMADE: Dynamic aggregation for minority attack detection". Dynamic aggregation weights based on per-class performance. CICIoT2023. No prototype sharing, no thresholding. |
| Jin et al. (2025) | Computer Networks 259:111116 | "Federated dual correction IDS". GAN boundary samples for rare traffic. UNSW-NB15 and CICIDS2018. GAN-based, excluded by project constraints. |

### 5. What those papers already accomplish
- PROTEAN (2025): Show prototype exchange improves FL-IDS under Non-IID attack distributions; allows clients to learn about attacks not present in their data.
- FedMADE (2024): Show dynamic aggregation weights improve minority attack classification by up to 71%.
- Jin et al. (2025): Show GAN-based rare-traffic augmentation improves heterogeneous FL-IDS on UNSW-NB15.

### 6. What remains unresolved
- PROTEAN exchanges prototypes for **all classes**, not specifically addressing the statistical reliability problem when a class is absent from most clients.
- No work combines prototype-based knowledge sharing with **class-specific decision thresholds** under a benign-FP constraint.
- No work evaluates prototype-based FL-IDS on **UNSW-NB15** with **Dirichlet alpha=0.3** and 10 clients.
- The interaction between prototype quality (few samples per class) and threshold stability is not studied.

### 7. Proposed experiment
1. Preprocess and partition as in Candidate 1.
2. Train global MLP with FedAvg.
3. Each client computes class prototypes on validation set for locally present classes.
4. Server aggregates prototypes per class (presence-weighted).
5. Global model applies prototype-adjusted scoring with class-specific thresholds selected on local calibration sets.
6. Evaluate FedAvg argmax vs. prototype-guided thresholding on official test set.
7. Ablation: prototype aggregation with/without presence weighting; threshold aggregation by median vs. mean.

### 8. Baselines
- Centralized MLP, argmax.
- FedAvg, argmax.
- PROTEAN-style prototype aggregation without thresholds (ablation).
- Prototype-guided thresholding (proposed).

### 9. Evaluation metrics
- Primary: Macro-F1 on official test set.
- Secondary: Per-class recall for minority classes, per-class FPR, benign FPR.
- Prototype quality: intra-class variance of aggregated prototypes vs. local prototypes.

### 10. Expected implementation complexity
Medium-High. Requires: prototype computation, prototype-aggregated classification head modification, threshold selection per client, server-side threshold aggregation. More complex than Candidate 1 because it modifies the model's decision mechanism, not just post-hoc thresholds.

### 11. Local computational requirements
Low-Medium. Same MLP training as Candidate 1 plus prototype computation (negligible overhead). CPU feasible.

### 12. Main risks and failure modes
- Prototypes for rare classes with few samples may be noisy/unrepresentative.
- If a class is present in only 1-2 clients, prototype aggregation is essentially copying that client's prototype, providing no benefit.
- Modifying the classification head may destabilize the trained model.
- Results may not outperform simpler FedAvg with per-class thresholds.

---

## Candidate 3: Hierarchical Benign-First with Federated Class-Specific Attack Thresholds (HB-STAT)

### 1. Proposed title
**Hierarchical Benign-First Classification with Federated Class-Specific Attack Thresholds under Non-IID Partitioning**

### 2. Research question
Can a two-stage decision policy—a high-recall benign-vs-attack gate followed by federated class-specific attack thresholds—improve minority-class attack recall while keeping the benign false-positive rate within budget in federated multiclass IDS?

### 3. Exact methodological contribution
- Stage 1: Binary classifier (benign vs. any attack) trained federated with FedAvg. Threshold set for high benign recall (low false negatives on attacks).
- Stage 2: For samples passing the gate as "attack", apply class-specific thresholds from federated class-conditional score distributions (computed on calibration sets, only for locally present classes).
- Server aggregates per-attack-class score summaries from clients that have the class; selects thresholds under a benign-FPR budget applied to Stage 1 false positives.
- The benign-FPR budget is split: Stage 1 controls benign→attack false positives; Stage 2 controls misclassification among attack classes.
- Evaluate on the untouched official test set.

### 4. Closest existing papers
| Paper | Venue/Year | Key details |
|---|---|---|
| MDPI Applied Sciences 16(2):801 (2026) | Applied Sciences 2026 | "Federated Learning for IDS Under Class Imbalance". Benign gate + per-class attack thresholds via **centralized** validation grid search. Benign-FP cap ≤500 per 10K benign. CICIDS-2017, InSDN-OVS, 5G-NIDD (not UNSW-NB15). Not federated calibration. |
| Laridi et al. (2024) | Scientific Reports 2024 | Federated thresholding from summary statistics. Anomaly detection (binary), not multiclass hierarchical. |
| Maddock et al. (2025) | arXiv:2025 | Federated post-hoc calibration (binning/scaling). Multiclass, but single-stage, not hierarchical benign-first. |

### 5. What those papers already accomplish
- MDPI Applied Sciences (2026): Show benign gate + per-class thresholds with benign-FP cap improves rare-class F1 in centralized FL-IDS.
- Laridi et al. (2024): Federated thresholding works for anomaly detection under Non-IID.
- Maddock et al. (2025): Federated calibration (binning/scaling) works for multiclass under non-IID.

### 6. What remains unresolved
- The MDPI Applied Sciences (2026) benign-gate + per-class threshold approach is **centralized** (validation grid search), not federated.
- No work combines a **federated** benign-vs-attack gate with **federated** class-specific attack thresholds under a benign-FP budget.
- No work evaluates this hierarchical approach on **UNSW-NB15**.
- The benign-FP budget allocation between Stage 1 and Stage 2 is not studied.

### 7. Proposed experiment
1. Preprocess and partition as in Candidate 1.
2. Train binary benign-vs-attack MLP with FedAvg (Stage 1).
3. Train multiclass attack classifier with FedAvg (Stage 2, attack samples only).
4. Each client computes class-conditional score summaries on calibration set for locally present attack classes.
5. Server aggregates summaries (presence-weighted) and selects attack-class thresholds under benign-FP budget.
6. Evaluate: FedAvg argmax (single-stage) vs. HB-STAT (two-stage) on official test set.
7. Ablation: Stage 1 threshold variation (benign recall vs. FPR trade-off).

### 8. Baselines
- Centralized MLP, argmax.
- FedAvg, argmax (single-stage multiclass).
- HB-STAT with centralized threshold selection (diagnostic reference).
- HB-STAT with federated threshold selection (proposed).

### 9. Evaluation metrics
- Primary: Macro-F1 on official test set.
- Secondary: Benign recall (Stage 1 gate quality), per-class attack recall, benign FPR, overall FPR.
- Gate efficiency: fraction of test samples passed to Stage 2.

### 10. Expected implementation complexity
Medium. Requires: two-stage model training, binary + multiclass FedAvg, federated threshold aggregation at Stage 2 only. More components than Candidate 1 but each is straightforward.

### 11. Local computational requirements
Low-Medium. Two models instead of one, but both are lightweight MLPs. CPU feasible.

### 12. Main risks and failure modes
- Stage 1 benign gate may be too aggressive (too many benign→attack false positives) or too conservative (misses attacks).
- The two-stage structure complicates the benign-FP budget accounting.
- If Stage 1 misses many minority attacks, Stage 2 cannot recover them.
- Results may not outperform a well-tuned single-stage FedAvg with per-class thresholds.

---

## Comparison Table

| Criterion | Candidate 1: CAA-FedCal | Candidate 2: Prototype-Guided | Candidate 3: HB-STAT |
|---|---|---|---|
| Research problem | Class absence in federated calibration | Prototype sharing + thresholds under Non-IID | Hierarchical benign-first + federated attack thresholds |
| Novelty defensibility | High — class-absence-aware federated calibration not in existing work | Medium — PROTEAN does prototypes; no threshold component | Medium — benign gate exists centrally, not federated |
| Scientific value | High — directly addresses the structural rare-class problem of Dirichlet α=0.3 | Medium — prototypes help but don't solve threshold reliability | Medium — hierarchical design is operationally appealing |
| Reproducibility | High — pure post-hoc calibration, no model modification | Medium — modifies classification head | Medium — two-stage training adds complexity |
| Implementation complexity | Medium | Medium-High | Medium |
| Local compute requirements | Low | Low-Medium | Low-Medium |
| Closest prior work overlap | Maddock et al. 2025 (federated binning, no class absence) | PROTEAN 2025 (prototypes, no thresholds/UNSW-NB15) | MDPI Applied Sciences 2026 (centralized benign gate+thresholds) |
| Main risk | Rare classes absent from too few clients | Prototype noise for rare classes | Stage 1 gate misses minority attacks |
| Uses UNSW-NB15 | Yes | Yes | Yes |
| Uses Dirichlet α=0.3, 10 clients | Yes | Yes | Yes |
| Benign-FP budget | Yes | Yes | Yes (split across stages) |
| Requires model training | Yes (FedAvg baseline) | Yes (FedAvg + head modification) | Yes (two-stage FedAvg) |

---

## Recommendation

**Recommend Candidate 1: Class-Absence-Aware Federated Calibration (CAA-FedCal).**

### Rationale
1. **Directly addresses the structural problem**: Under Dirichlet α=0.3 with 10 clients and 10 classes on UNSW-NB15, rare classes (Worms, Shellcode, Analysis) are guaranteed to be absent from many clients. Candidate 1 explicitly handles this by only aggregating score summaries from clients that have the class, weighted by class presence. This is the most defensible research gap: existing federated calibration work (Maddock et al. 2025) weights by example count but does not distinguish class absence from class scarcity.

2. **Minimal novelty risk**: The core of federated post-hoc calibration exists (Maddock et al. 2025; Laridi et al. 2024), but the **class-absence-aware weighting** combined with **benign-FP constraint on UNSW-NB15** is not covered by existing work. The contribution is specific and verifiable.

3. **Simplest implementation**: Pure post-hoc calibration on top of a standard FedAvg baseline. No model architecture changes, no two-stage training, no prototype computation. Lowest risk of implementation bugs invalidating results.

4. **Cleanest ablation story**: Compare presence-weighted vs. uniform aggregation; with/without minimum client coverage; federated vs. centralized threshold selection. All baselines use the same partitions and model.

5. **Feasible on local hardware**: Lightweight MLP, 10 clients, 30 rounds, CPU-only. Runs in minutes.

### Conditions for proceeding
- M1 novelty review must be updated to document that Candidate 1's specific combination (class-absence-aware federated calibration with benign-FP constraint on UNSW-NB15) is not covered by existing work.
- M2 data validation must pass with official CSVs.
- User must approve the direction before implementation.

---

## Original Proposal (Documented for Comparison)

**Title**: Federated Multiclass Intrusion Detection with Privacy-Conscious Distributed Score Calibration under Non-IID Traffic

**Research question**: Can class-specific decision thresholds estimated from aggregated client-local score summaries improve minority-class detection while keeping the benign false-positive rate below a predefined operational budget?

**Status**: NOT APPROVED (M1). Core mechanism overlaps with Laridi et al. (2024) and Maddock et al. (2025). The class-absence problem is not addressed.

**Why superseded**: Candidate 1 retains the core calibration idea but adds the class-absence-aware mechanism that addresses the actual structural problem created by Dirichlet Non-IID partitioning, making the contribution more defensible.
