# PU-Net benchmark

CD / P2M, x1e-4, lower is better.

| Method | 10K 1% | 10K 2% | 10K 3% | 50K 1% | 50K 2% | 50K 3% |
|---|---|---|---|---|---|---|
| Bilateral | 3.65 / 1.34 | 5.01 / 2.02 | 7.00 / 3.56 | 0.88 / 0.23 | 2.38 / 1.39 | 6.30 / 4.73 |
| PCNet | 3.52 / 1.15 | 7.47 / 3.97 | 13.10 / 8.74 | 1.05 / 0.35 | 1.45 / 0.61 | 2.29 / 1.29 |
| DMRDenoise | 4.48 / 1.72 | 4.98 / 2.12 | 5.89 / 2.85 | 1.16 / 0.47 | 1.57 / 0.80 | 2.43 / 1.53 |
| GLR | 2.96 / 1.05 | 3.77 / 1.31 | 4.91 / 2.11 | 0.70 / 0.16 | 1.59 / 0.83 | 3.84 / 2.71 |
| ScoreDenoise | 2.52 / 0.46 | 3.69 / 1.07 | 4.71 / 1.94 | 0.72 / 0.15 | 1.29 / 0.57 | 1.93 / 1.04 |
| PD-Flow | 2.13 / 0.38 | 3.25 / 1.01 | 5.19 / 2.52 | 0.65 / 0.16 | 1.42 / 0.78 | 3.90 / 2.86 |
| I-PFN | 2.31 / 0.37 | 3.43 / 0.90 | 5.24 / 2.50 | 0.66 / 0.12 | 1.05 / 0.43 | 2.54 / 1.65 |
| P2P-Bridge | 2.28 / 0.39 | 3.20 / 0.81 | 3.99 / 1.42 | 0.59 / 0.09 | 0.90 / 0.32 | 1.56 / 0.84 |
| **Ours** | 2.89 / 0.24 | 4.07 / 0.53 | 5.29 / 0.96 | 0.76 / 0.07 | 1.40 / 0.33 | 2.77 / 1.05 |

**Read the CD values only.** P2M is not calibrated against the published
definition (0.17x on a known baseline) and is shown for completeness only.

PU-Net only - the harness supports PC-Net too but it was not run for this
report. Checkpoint is epoch 46/60; resuming to 60 reproduced these numbers
to within ~1-2%, confirming it as a stable optimum rather than an early stop.
