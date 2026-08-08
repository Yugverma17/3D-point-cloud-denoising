# Getting the benchmark data

The comparison tables in ScoreDenoise, PD-Flow, I-PFN and P2P-Bridge all use
the same two test sets. To put our numbers in those tables we have to evaluate
on the same files, not on our own resampling of the same meshes.

## Why the exact files matter

Two things vary between anyone's regenerated version and the released one:

- **Poisson-disk sampling.** Resampling a mesh to 10,000 points does not give
  the same 10,000 points twice. Different point placement changes Chamfer
  distance directly.
- **The noise draw.** "1% Gaussian noise" fixes the standard deviation, not
  the actual displacements. A different random seed gives a different cloud
  and a different score.

Neither difference is large, but both are the same size as the gaps between
methods in the table. ScoreDenoise sits at 2.52 and PD-Flow at 2.13 on
PU-Net sparse/1%: a resampling difference can easily be worth more than that
0.39, so a regenerated set cannot tell you which method is better.

Use the released files.

## Where to get them

The test sets originate with the ScoreDenoise release
(<https://github.com/luost26/score-denoise>). Its README links a data archive
containing `PUNet` and `PCNet` at both resolutions with noise already applied.
The P2P-Bridge repo (<https://github.com/matvogel/P2P-Bridge>) uses the same
data and documents its layout too.

Download and unpack so the tree looks like this:

```
data/benchmark/
  PUNet_10000_poisson/          clean, 10K points, one .xyz per shape
  PUNet_10000_poisson_0.01/     same shapes with 1% noise
  PUNet_10000_poisson_0.02/
  PUNet_10000_poisson_0.03/
  PUNet_50000_poisson/          the dense versions
  PUNet_50000_poisson_0.01/
  ...
  meshes/                       ground-truth meshes, needed for P2M
```

`load_released_set` expects exactly those directory names. If the archive you
get uses different ones, rename rather than editing the loader, so the layout
stays the one the docs describe.

## Calibrate before quoting anything

```bash
python scripts/benchmark.py --data data/benchmark --calibrate
```

This scores the bilateral filter, whose numbers are in the published table
(PU-Net sparse/1%: CD 3.65, P2M 1.34), and compares. Three outcomes:

- **PASS** - our normalization, Chamfer convention and noise model agree with
  theirs. Our own numbers can be quoted alongside.
- **FAIL** - they do not. A ratio near 2 or 0.5 almost always means the
  Chamfer convention differs (some papers halve it, some do not); a large
  ratio usually means the noise scale or the unit-sphere normalization
  differs. Fix the harness before publishing any comparison.
- **INCONCLUSIVE** - you ran it on substitute shapes rather than the released
  set. The difference then mixes the metric convention with the shapes being
  easier or harder, which is not a calibration.

The bilateral implementation here is not identical to whichever one produced
the published row, so exact agreement is not expected; the tolerance is a
loose 35% band meant to catch convention errors, not small differences.

## Without the released data

`--meshes` regenerates clouds from a directory of meshes:

```bash
python scripts/benchmark.py --meshes data/meshes --method bilateral
```

This is useful for development and for comparing your own variants against
each other. Numbers from it must not be placed in a table next to published
results, and `calibrate` will refuse to certify them.

## Running the grid

```bash
# a classical baseline
python scripts/benchmark.py --data data/benchmark --method bilateral

# a trained checkpoint, across every resolution and noise level
python scripts/benchmark.py --data data/benchmark --checkpoint runs/exp1/best.pt

# the noisy input itself, i.e. what "doing nothing" scores
python scripts/benchmark.py --data data/benchmark --method identity
```

`--no-p2m` skips the point-to-mesh metric, which is much slower than Chamfer
because it queries the mesh surface rather than a point set. Keep it on for
anything you intend to report: Chamfer can be improved by points clustering
near ground-truth samples, whereas P2M measures distance to the actual
surface, so a method that scores well on CD and badly on P2M has usually just
bunched its points up.
