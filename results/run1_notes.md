# Run 1 - Kaggle T4, 60 epochs, fixed 2% noise

Superseded. Kept because its failure is what motivated the noise-range fix.

Trained 8.5 h. Loss 0.614969 -> 0.566963, best 0.549032 at epoch 46, and
essentially flat after epoch 3. The benchmark then ran out of the 12 h limit
after 4 of 6 cells.

## Results (CD is calibrated and comparable; P2M is NOT - see below)

| case       | ours CD | noisy CD | improvement |
|------------|---------|----------|-------------|
| sparse/1%  |  4.6991 |   4.7907 |         +2% |
| sparse/2%  |  5.1526 |  11.5867 |        +56% |
| sparse/3%  |  5.5109 |  20.6351 |        +73% |
| dense/1%   |  0.8660 |   3.1881 |        +73% |

On CD this beats Bilateral, PCNet and DMRDenoise at the higher noise levels
and sits behind GLR, ScoreDenoise, PD-Flow, I-PFN and P2P-Bridge.

## Why sparse/1% barely moved

Trained at a single fixed noise level, so the model learned one correction
size and applied it regardless, displacing points that were already nearly
right. Reproduced in miniature on a sphere:

| test noise | noisy CD | trained fixed 2% | trained 0.5-3% |
|------------|----------|------------------|----------------|
| 1%         |   122.36 |  225.81 (-84.5%) | 33.99 (+72.2%) |
| 2%         |   436.61 |  247.09 (+43.4%) | 78.07 (+82.1%) |
| 3%         |   841.18 |  432.25 (+48.6%) | 140.59(+83.3%) |

The fixed-noise model is worse than doing nothing at 1%. Sampling the range
fixes it and is better at every level, including the one it used to train on.
PatchDataset(noise_range=...) now does this by default.

## P2M is not comparable

Calibration passes on CD (0.84x of published Bilateral) but P2M comes out at
0.17x for the same algorithm on the same shapes, so it measures something
different from what the papers report. Run 1's P2M numbers looked better than
every published method; that is an artifact of the definition, not a result.
Do not quote P2M until the convention is resolved against their eval code.
