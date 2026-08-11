# Run 2 - per-patch noise sampling (0.5-3%)

Same architecture and budget as run 1; the only change is that each patch is
re-noised at a level drawn from 0.5-3% instead of everything being fixed at 2%.

| | run 1 (fixed 2%) | run 2 (range) |
|---|---|---|
| best loss | 0.549032 | **0.523106** |
| loss curve | flat after epoch 3 | still falling at epoch 46 |
| epochs | 60 | 46 |
| time | 8.5 h | 6.1 h |

Run 1 converged almost immediately and then stopped improving. Run 2 is still
descending monotonically at the point the checkpoint was taken (0.523617 ->
0.523315 -> 0.523106 over the last three epochs), so it had not yet plateaued
and more epochs would likely still help.

Benchmark numbers pending - needs a GPU pass, see kaggle/benchmark_pointdenoise.ipynb.

## Recovering the checkpoint

The download arrived unpacked: a `.pt` is a zip archive, and it had been
extracted to `best/best/{data.pkl, data/*, version, byteorder}`. Rezipping the
members under a `best/` prefix with ZIP_STORED restores a file torch loads
normally.
