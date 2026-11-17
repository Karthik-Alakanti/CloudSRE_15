# M3 Optimization Pilot — Results and Assessment

## Overview

Investigated cheap optimization opportunities for the Centralized MLP and FedAvg baselines from M3. All experiments preserve M3 artifacts and save new results under `results/optimization/`. Validation data (pooled client validation sets) was used for model selection — test data was never used for tuning.

## Pilot Experiments (seed 42)

| Experiment | Config | Accuracy | Macro-F1 | Benign FPR | Minority Recall |
|---|---|---|---|---|---|
| Centralized (M3 baseline) | 128→64, 30ep, lr=0.001 | 0.9877 | 0.5629 | 0.0025 | 0.5357 |
| FedAvg (M3 baseline) | 10 clients, 15 rounds | 0.9738 | 0.3570 | 0.0001 | 0.3129 |
| CAA-FedCal (M3 baseline) | FedAvg + calibration | 0.9786 | 0.3893 | 0.0020 | 0.3887 |
| class_weighted | 128→64, 30ep, lr=0.001, weighted CE | 0.9824 | 0.5685 | 0.0067 | 0.6714 |
| wider_mlp (5ep) | 256→128, 5ep, lr=0.001 | 0.9871 | 0.5148 | 0.0025 | 0.4731 |
| **lower_lr (BEST)** | **128→64, 30ep, lr=0.0005** | **0.9879** | **0.5770** | **0.0020** | **0.5455** |

## Best Configuration: lower_lr (0.0005)

Reducing the Adam learning rate from 0.001 to 0.0005 with the same architecture (128→64) and same epochs (30) produces consistent improvements across all metrics:

| Metric | M3 Baseline | Optimized | Change |
|---|---|---|---|
| Accuracy | 0.9877 | 0.9879 | +0.0002 |
| Macro-F1 | 0.5629 | 0.5770 | +0.0141 (+2.5% rel.) |
| Benign FPR | 0.0025 | 0.0020 | -0.0005 (-20% rel.) |
| Minority Recall | 0.5357 | 0.5455 | +0.0098 |

Per-class F1 improvements (lower_lr vs M3 centralized):
- Worms: 0.514 → **0.606** (+18% rel.)
- Backdoor: 0.031 → **0.039** (+26% rel.)
- Reconnaissance: 0.772 → **0.826** (+7% rel.)
- Exploits: 0.755 → **0.812** (+7.5% rel.)
- Generic: 0.846 → 0.848 (similar)
- Shellcode: 0.853 → 0.846 (similar)
- Fuzzers: 0.762 → 0.698 (-8% rel.)
- DoS: 0.097 → 0.095 (similar)
- Analysis: 0.000 → 0.000 (unchanged — too few samples)

## Other Pilots

**class_weighted CE (30 epochs):** Accuracy drops to 0.9824 (from 0.9877) as benign FPR increases to 0.0067. Macro-F1 slightly improves to 0.5685. Worms F1 gets worse (0.391 vs 0.514). Not recommended — trades accuracy and Worms detection for minority-class recall.

**wider_mlp 256→128 (5 epochs):** Not converged. At 5 epochs, acc=0.9871, mF1=0.5148. Needs more epochs to compare fairly with 30-epoch baseline.

## Limitations

1. **Single seed (42):** All optimization pilots ran on seed 42 only. Results should be confirmed on seeds 123 and 7 before claiming generalizability.
2. **wider_mlp not fully evaluated:** Only 5 epochs (vs 30 for baseline). May perform better with full training but requires ~35 minutes.
3. **No FedAvg optimization tested:** All pilots were on centralized MLP. FedAvg optimization (more rounds, more local epochs, different LR) remains unexplored.
4. **99% accuracy not reached:** Best accuracy is 0.9879, still below 99.00%. The error rate is 1.21%, requiring ~20% relative reduction to reach 99%.

## Recommendation

Adopt **lower_lr = 0.0005** as the default learning rate for centralized MLP training. It improves accuracy, macro-F1, and benign FPR simultaneously with no additional compute cost. Run on seeds 123 and 7 to confirm consistency before finalizing.

## Artifacts

All results saved under `results/optimization/`:
- `models/opt_lower_lr_seed42.pt` — best model checkpoint
- `metrics/opt_lower_lr_seed42.json` — full metrics including per-class breakdown and confusion matrix
- `metrics/opt_class_weighted_seed42.json` — class-weighted CE results
- `metrics/opt_wider_mlp_5ep_seed42.json` — wider MLP (5 epochs) results
