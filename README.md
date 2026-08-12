# Point cloud denoising with a geometry-aware transformer

A 3D scan comes back as a cloud of points that are all slightly off the real
surface. This denoises it: for every point, look at its local neighbourhood
and predict where the point should actually sit.

The architecture is multi-scale EdgeConv for local descriptors, then a
transformer whose attention is biased by relative 3D offsets, then an MLP that
predicts a displacement per point.

New to any of this? [**EXPLAINED.md**](EXPLAINED.md) walks through the whole
project in plain language, no prior knowledge assumed, including every bug
that got found along the way and why the fix worked. [**NOTES.md**](NOTES.md)
covers every concept and term this project uses plus interview questions
about it, if you're studying this project rather than just reading about it.

## Status

Trained on the PU-Net training set and benchmarked against the published
comparison table.

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

CD / P2M x1e-4, lower is better. **Read the CD columns only** - P2M is not
calibrated against the published definition (0.17x on a known baseline) and is
shown for completeness. Full table in
[results/comparison_table.txt](results/comparison_table.txt).

On CD this ranks 4th to 6th of nine depending on the cell, beating Bilateral,
PCNet and DMRDenoise everywhere, GLR at sparse/1% and dense/2%, and PD-Flow at
dense/2% and dense/3%. Behind ScoreDenoise, I-PFN and P2P-Bridge throughout.

Against the noisy input it removes 40% of the Chamfer error at 1% noise and
86% at 3%. Per-cell detail in
[results/benchmark_run2.txt](results/benchmark_run2.txt).

The checkpoint is epoch 46 of a 60-epoch run: training was resumed for the
remaining 14 to check whether stopping early had cost anything, and it hadn't
- see [The last epoch is not the best epoch](#the-last-epoch-is-not-the-best-epoch).
This table is the practical ceiling for this training recipe at this budget,
not a truncated result.

## Why attention needs geometry

Plain self-attention is permutation invariant, so it has no idea where points
sit relative to each other. For denoising that is exactly the wrong property:
whether a neighbour should influence a point depends on which direction it is
in and how far away. So the offset between every pair of points goes through a
small MLP and the result is added to the attention logits as a per-head bias.

## How a patch is built

Each point becomes the centre of one patch: its k nearest neighbours,
translated so the centre is at the origin, rotated into the patch's own
principal-axis frame, and scaled to roughly unit size. The network sees every
patch in a canonical orientation instead of having to learn rotation
invariance from scratch.

At evaluation time only the **centre** point of each patch is used. Every
point is the centre of exactly one patch, so every point gets exactly one
prediction and overlapping patches cannot overwrite each other.

## Two things that were easy to get wrong

Both of these were live bugs in the earlier version of this project, and both
made it degrade point clouds rather than clean them. Neither showed up as a
crash or a rising loss, which is why they are now covered by tests.

**The input and its target must be rotated by the same matrix.** The patch
frame gives a rotation R. Rotating the noisy patch by R and its ground-truth
target by R transpose puts them in different frames. Because R is derived from
each patch's own PCA, the mismatch is different for every sample, so the
network is asked to learn something that is not a function. On a test patch
the input-to-target gap was 0.040 that way versus 0.016 done correctly.
Covered by `test_noisy_and_clean_land_in_the_same_frame`.

**Neighbourhoods have to adapt to point density.** A fixed radius of 5% of the
bounding box sounds reasonable and holds only about 4 neighbours on a
1500-point sphere, below the minimum needed to define a frame. A third of the
points then got no patch at all and silently stayed noisy. Chamfer splits into
two halves and the damage only shows up in one of them:

| | accuracy (pred to gt) | coverage (gt to pred) | total |
|---|---|---|---|
| noisy input | 283.35 | 134.12 | 417.47 |
| radius patches | 255.79 | 165.97 | 421.75 |
| k-NN patches | 0.25 | 11.34 | 11.59 |

The points that did get denoised landed closer to the surface, but the third
left untouched wrecked coverage and the total came out worse. Switching to
k-NN covers every point and drops the supervision ceiling by 36x. Covered by
`test_every_point_gets_a_patch`.

## Fixed-noise training breaks the low-noise case

Run 1 trained at a single fixed 2% noise level, and the benchmark showed what
that costs: +56% at 2%, +73% at 3%, and +2% at 1%. The model learned one
correction size and applied it regardless, so where the input was already
close it displaced points that did not need moving.

Sampling the level per patch instead fixed it. Same architecture, same budget,
measured on the real benchmark:

| case | run 1 (fixed 2%) | run 2 (range 0.5-3%) |
|---|---|---|
| sparse/1% | 4.70 | **2.89** |
| sparse/2% | 5.15 | 4.07 |
| sparse/3% | 5.51 | 5.29 |
| dense/1% | 0.87 | 0.76 |

The targeted cell went from +2% over the noisy input to +40%, and every other
cell improved too - on a checkpoint that saw 46 epochs rather than 60.

The mechanism, isolated on a sphere:

| test noise | noisy CD | trained at fixed 2% | trained 0.5-3% |
|---|---|---|---|
| 1% | 122.36 | 225.81 (**84.5% worse**) | 33.99 (72.2% better) |
| 2% | 436.61 | 247.09 (43.4% better) | 78.07 (**82.1% better**) |
| 3% | 841.18 | 432.25 (48.6% better) | 140.59 (**83.3% better**) |

At 1% the fixed-noise model is worse than not denoising at all. Sampling the
range wins at every level, including the one the fixed model trained on.
`PatchDataset(noise_range=(0.005, 0.03))` is now the default.

## The last epoch is not the best epoch

Loss is not guaranteed to keep falling to the final epoch. On a real 60-epoch
run it bottomed around epoch 45 (~0.523) and rose back to ~0.535 by epoch 60,
even with the learning rate still decaying - the model started overfitting or
destabilizing past that point, and best.pt correctly stopped updating there.

`train()` used to return whatever model the loop happened to end on rather
than the best-loss checkpoint. That silently mattered: a notebook benchmarking
the returned model right after training was scoring the worse epoch-60 weights
while best.pt on disk still correctly pointed at epoch 46. `train()` now always
hands back best.pt's weights regardless of which epoch the loop stopped at.
Covered by `test_train_returns_the_best_checkpoint_not_the_last_epoch`.

Training was later resumed from epoch 46 through the remaining 14 to check
whether stopping there had left something on the table. It hadn't: the
resulting benchmark reproduced the epoch-46 numbers to within about 1-2%
(e.g. sparse/1% CD 2.8896 versus the original run's 2.89), consistent with
the same optimum rather than a meaningfully different checkpoint. Epoch 46
is a stable point for this recipe, not an early stop that cost anything.

## Undertraining looks like a bug

An undertrained model has learned a partial displacement that overshoots, and
that scores worse than not denoising at all. Same setup, same code, different
budget:

| epochs | patches/shape | CD | vs noisy input |
|---|---|---|---|
| 4 | 128 | 250.88 | 54.8% worse |
| 10 | 256 | 18.07 | 88.8% better |
| 20 | 512 | 3.66 | 97.7% better |

So a bad short run says nothing about the architecture. Check the loss curve
before changing anything.

## Benchmark

`scripts/benchmark.py` runs the PU-Net / PC-Net grid: two resolutions (10K
sparse, 50K dense) by three noise levels (1/2/3%), scored by Chamfer distance
and point-to-mesh, reported x1e-4. Results print straight into the published
comparison table.

Classical baselines (`bilateral`, `laplacian`) and the `identity` no-op run
through the same measurement path as the model, so nothing is compared across
different code.

**Calibrate before quoting any number.** The bilateral filter's scores appear
in the published tables, so scoring it here says whether our normalization,
Chamfer convention and noise model match theirs:

```bash
python scripts/benchmark.py --data data --calibrate
```

Every reported metric is checked separately, because gating on Chamfer alone
let a real problem through: CD agreed at 0.84x while P2M sat at 0.17x for the
same algorithm on the same shapes, and the run reported PASS. Current state:

```
CD   ratio 0.84x  PASS      QUOTABLE: CD
P2M  ratio 0.17x  FAIL      DO NOT QUOTE: P2M
```

It also reports INCONCLUSIVE if run on substitute shapes rather than the
released test set, where any difference mixes the metric convention up with
the shapes being easier or harder. That is not a calibration, so it refuses to
certify. See [docs/benchmark.md](docs/benchmark.md) for the data.

## Usage

```bash
pip install -r requirements.txt
pytest                                    # 38 tests
pytest -m "not slow"                      # skip the training checks
```

Training expects a directory of clean clouds as `.npy` of shape (N, 3). Noise
is added on the fly, so one clean dataset covers every noise level.

```bash
python scripts/train.py --data data/train --out runs/exp1 --epochs 50
python scripts/evaluate.py --checkpoint runs/exp1/best.pt --data data/test
```

Evaluation scores the noisy input alongside the denoised output, because
"did this help?" is the only question worth asking and a table without the
baseline cannot answer it:

```
Improvement
----------------------------------------
CD          51.5388 ->    46.6609     +9.5%
```

Pass `--meshes` to also report point-to-mesh distance. P2M is the more honest
metric: Chamfer can be improved by points clustering near ground-truth
samples, whereas P2M measures distance to the actual surface.

## Layout

```
pointdenoise/
  geometry.py   patch frames and noise models
  data.py       k-NN patch extraction and supervision
  model.py      EdgeConv, relative-position attention, denoiser
  losses.py     robust Chamfer plus repulsion
  metrics.py    CD and P2M in the published benchmark convention
  engine.py     training and inference loops
  baselines.py  bilateral and Laplacian, used to calibrate the harness
  benchmark.py  PU-Net/PC-Net protocol and comparison table
scripts/        train.py, evaluate.py, benchmark.py
docs/           benchmark.md - getting the test data
tests/          38 tests, including regression tests for the bugs above
configs/        default.yaml
```

## Metric conventions

Denoising papers report Chamfer distance in at least two conventions that
differ by a factor of two, so numbers are only comparable if the convention
matches. `metrics.py` normalizes both clouds to the unit sphere, sums the mean
squared nearest-neighbour distance in both directions, and reports the result
multiplied by 1e4, which reproduces the published Bilateral CD to 0.84x.

P2M is unresolved. Neither mean distance nor mean squared distance reproduces
the published value, so the papers use a definition this project has not
matched yet and P2M is excluded from any comparison. Resolving it needs their
evaluation code rather than another guess.

## Scope

This result is PU-Net only. `scripts/benchmark.py` and the Colab notebook both
support PC-Net too (`load_released_set(DATA, "PCNet", ...)`, same calibrated
harness), but running it wasn't worth another multi-hour Colab session for a
second dataset rather than a different result - the PU-Net table already
answers the question this project set out to answer: reproduce the published
protocol correctly and see where an honestly-measured implementation lands.

## What is left

- **Resolve the P2M convention** against the published evaluation code, which
  would make the second metric quotable instead of excluded.
- **Close the CD gap.** ScoreDenoise, I-PFN and P2P-Bridge are ahead in every
  cell. Those use different formulations (score matching, iterative filtering,
  a Schrodinger bridge) rather than better tuning of this one.
