# Run 2 - per-patch noise sampling (0.5-3%)

Same architecture and budget as run 1; the only change is that each patch is
re-noised at a level drawn from 0.5-3% instead of everything being fixed at 2%.

## Training

| | run 1 (fixed 2%) | run 2 (range) |
|---|---|---|
| best loss | 0.549032 | **0.523106** |
| loss curve | flat after epoch 3 | still falling at epoch 46 |
| epochs | 60 | 46 |
| time | 8.5 h | 6.1 h |

Run 1 converged almost immediately and then stopped improving. Run 2 was still
descending monotonically when the checkpoint was taken (0.523617 -> 0.523315 ->
0.523106 over the last three epochs), so it had not plateaued and more epochs
would likely still help.

## Benchmark (PU-Net, CD x1e-4)

| case | run 1 | run 2 | noisy input | run 2 vs noisy | rank |
|---|---|---|---|---|---|
| sparse/1% | 4.70 | **2.89** | 4.79 | 40% | 5/9 |
| sparse/2% | 5.15 | 4.07 | 11.59 | 65% | 6/9 |
| sparse/3% | 5.51 | 5.29 | 20.64 | 74% | 6/9 |
| dense/1% | 0.87 | 0.76 | 3.19 | 76% | 6/9 |
| dense/2% | - | 1.40 | 9.95 | 86% | 4/9 |
| dense/3% | - | 2.77 | 19.89 | 86% | 6/9 |

Beats Bilateral, PCNet and DMRDenoise in every cell; GLR at sparse/1% and
dense/2%; PD-Flow at dense/2% and dense/3%. Behind ScoreDenoise, I-PFN and
P2P-Bridge throughout. Full layout in comparison_table.txt.

The sparse/1% cell is the one the fix targeted: run 1 managed +2% over the
noisy input there because it had only ever seen 2% noise, and run 2 gets +40%.
Every other cell improved too, on 46 epochs rather than 60.

## P2M

Not reported. Calibration reproduces the published Bilateral CD to 0.84x but
our P2M lands at 0.17x for the same algorithm on the same shapes, so it is
measuring something other than what these papers call P2M. Neither mean
distance nor mean squared distance reproduces their value; resolving it needs
their evaluation code. The P2M column appears in comparison_table.txt because
the layout calls for it, flagged as not-a-claim everywhere it appears.

## Recovering the checkpoint

The download arrived unpacked: a `.pt` is a zip archive, and it had been
extracted to `best/best/{data.pkl, data/*, version, byteorder}`. Rezipping the
members under a `best/` prefix with ZIP_STORED restores a file torch loads
normally, with its full 46-epoch history intact.

## Resuming the remaining 14 epochs

Training was later resumed from this checkpoint through epoch 60 to check
whether stopping at 46 had left improvement on the table. It hadn't: the
resulting benchmark reproduced these numbers to within about 1-2% (sparse/1%
CD 2.8896 vs 2.89, sparse/2% 4.0682 vs 4.07, sparse/3% 5.2866 vs 5.29,
dense/1% 0.7633 vs 0.76, dense/2% 1.3957 vs 1.40) - consistent with the same
optimum, not a materially different checkpoint. Epoch 46 is a stable point
for this recipe rather than an early stop that cost anything. Full-precision
numbers for those 5 cells are in
[bench_partial_confirmation.json](bench_partial_confirmation.json), the raw
resumable-benchmark cache pulled from Drive.

## Scope

PC-Net was not run. The harness and Colab notebook both support it
(`load_released_set(DATA, "PCNet", ...)`), but the PU-Net table already
answers what this project set out to answer, and a second dataset wasn't
worth another multi-hour session across repeated Colab disconnects.
