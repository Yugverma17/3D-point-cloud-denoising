# Point cloud denoising with a geometry-aware transformer

A 3D scan comes back as a cloud of points that are all slightly off the real
surface. This denoises it: for every point, look at its local neighbourhood
and predict where the point should actually sit.

The architecture is multi-scale EdgeConv for local descriptors, then a
transformer whose attention is biased by relative 3D offsets, then an MLP that
predicts a displacement per point.

## Status

Working, trained on toy data so far. On a noisy sphere it removes 97.7% of the
Chamfer error (162.06 to 3.66, x1e-4) given 20 epochs. The next step is the
PU-Net / PC-Net benchmark so the numbers can be compared against published
methods.

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

## Usage

```bash
pip install -r requirements.txt
pytest                                    # 23 tests
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
scripts/        train.py, evaluate.py
tests/          23 tests, including regression tests for both bugs above
configs/        default.yaml
```

## Metric conventions

Denoising papers report Chamfer distance in at least two conventions that
differ by a factor of two, so numbers are only comparable if the convention
matches. `metrics.py` normalizes both clouds to the unit sphere, sums the mean
squared nearest-neighbour distance in both directions, and reports the result
multiplied by 1e4. Before comparing against any published table, calibrate
against one number from that table.

## Next

- PU-Net and PC-Net benchmark harness at 10K/50K points and 1/2/3% noise
- Train on the full shape set rather than the toy data used so far
- Compare against published baselines once the harness is calibrated
