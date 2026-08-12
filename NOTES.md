# Study notes and interview prep

Two parts. Part 1 is every concept and term this project actually uses,
explained simply enough that no background is assumed. Part 2 is interview
questions this project is a genuinely good answer to, split by role, with
real answers grounded in what actually happened while building it (numbers,
bugs, file names), not generic textbook answers.

If a term below still isn't clicking, [EXPLAINED.md](EXPLAINED.md) covers the
same ground with more room to breathe, full analogies and a walk through the
whole project stage by stage.

---

## Part 1: Concepts and terms

### The shapes themselves

**Point cloud.** A shape made of individual dots scattered in 3D space,
instead of a smooth surface. Picture a statue covered edge to edge in tiny
glowing stickers: erase the statue and you can still tell what it was, just
from the pattern the stickers make.

**Mesh.** The "connect the dots" version of a shape: the same dots, but with
flat triangles stretched between them to make a solid surface. This project
uses meshes as the true, exact reference when measuring point-to-mesh
distance, since a mesh has no ambiguity about where the real surface is,
unlike a point cloud which only samples it.

**Noise.** Random, unwanted error. Here specifically: every point in a scan
being nudged a little off from its true position, the same way a photo
comes out blurry if the camera shook while the shutter was open. This
project adds noise on purpose during training and testing, at a known
amount (1%, 2%, or 3% of the object's size), so it's possible to measure
exactly how well that noise gets undone.

**Denoising.** Pushing each noisy point back toward where it should truly
be, without erasing the actual shape underneath.

### Geometry and math tools

**PCA (principal component analysis).** A way to find the "natural
directions" a cluster of points is spread out along. Imagine a cloud of
points shaped like a long, flat pancake: PCA finds that the pancake is
longest in one direction, a bit shorter in a second direction, and almost
flat in the third. This project uses PCA to rotate every patch into a
consistent, predictable orientation before the network sees it, the same
way you'd always turn a book right-side up before reading it, regardless of
how it was originally handed to you.

**Rotation matrix.** A small grid of numbers that, when applied to a set of
points, spins them around without changing their size or shape, only their
orientation. This project computes a custom rotation matrix for every patch
individually, from that patch's own PCA directions.

**k-nearest neighbors (k-NN).** For a given point, its `k` closest other
points, measured by straight-line distance. This project builds every patch
from a point's 256 nearest neighbors, specifically instead of "everyone
within some fixed distance," because a fixed distance can leave a point
with almost no neighbors in a sparse area of the cloud (see the patch
coverage bug in EXPLAINED.md Part 4).

**Normalization (unit sphere).** Shrinking or growing a shape so it always
fits inside a ball of the exact same size (radius 1), regardless of how big
the original object was. This makes measurements comparable across a huge
statue and a tiny figurine; without it, "how far off is this point" would
mean something completely different depending on the object's actual scale.

### Neural networks

**Neural network.** A program built from a chain of simple math operations
(mostly multiplying numbers together and adding them up) whose exact
numbers, called weights, get tuned automatically by practicing on examples,
rather than being hand-written by a programmer.

**Weights / parameters.** The actual adjustable numbers inside a neural
network. This project's network has about 4.8 million of them. Training
means slowly nudging all 4.8 million toward values that make the network's
guesses better.

**Forward pass.** Running an input through the network to get an output.
Here: feeding in a noisy patch and getting a suggested nudge back out.

**EdgeConv.** A way of building a description of a point's local
neighborhood: for each point, look at its `k` nearest neighbors, compute
"how is each neighbor different from me," and combine those differences
into a compact fingerprint. Borrowed from a technique originally built for
processing point clouds directly (from a paper called DGCNN).

**Attention / transformer.** A technique, most famous from modern chatbots,
where every item in a group gets to "check in" with every other item and
decide how much each one should matter to it. Normally this has no built-in
sense of physical space, it only compares abstract number patterns, so this
project adds the actual 3D direction and distance between every pair of
points directly into the attention calculation. That's what "relative
position bias" means in the code: teaching attention to actually care about
3D geometry instead of only abstract similarity.

**Multi-head attention.** Instead of computing one single "who matters to
whom" pattern, running several smaller versions side by side (this project
uses 8) and combining them. Different heads can end up specializing in
different kinds of relationships, the same way a panel of judges might each
focus on a different aspect of a performance.

**Residual / displacement prediction.** Rather than asking the network "what
is the correct position," this project asks it "how much should this move,
and in what direction" and adds that small nudge to the existing position.
Predicting a small correction is an easier problem than predicting an
absolute position from scratch, and it comes with a safety property: if the
network's very last layer starts out outputting exactly zero (which this
project deliberately sets up), an untrained network does nothing at all to
its input. It can never make a cloud worse before it has learned anything.

### Training

**Loss function.** The formula that scores how wrong the network's current
guess is. Training is the process of adjusting weights to make this number
smaller.

**Chamfer distance.** This project's main loss and its main measurement of
success. For every predicted point, find the nearest true point and measure
that distance; do the same the other way around, from every true point to
its nearest predicted point; average both sets together. Checking both
directions matters because checking only one direction can be gamed by
bunching all your points onto a few correct spots and ignoring the rest of
the shape.

**Repulsion loss.** An extra rule added on top of Chamfer distance,
specifically penalizing points that end up too close to their neighbors.
Without it, a network can satisfy Chamfer distance by clumping points
together in a few "safe" spots rather than spreading them evenly across the
real surface.

**Gradient descent / backpropagation.** The actual mechanism that adjusts a
network's weights: figure out, for every single weight, which direction of
small change would reduce the loss, and nudge every weight a little in that
direction. Backpropagation is the specific, efficient method used to
compute all those "which direction" answers at once, working backward
through the network from the loss.

**Learning rate.** How big each of those nudges is. Too large and training
can overshoot and bounce around without settling; too small and training
crawls. This project starts at 0.001 and shrinks it smoothly over training
using a schedule called cosine annealing, similar to a dart player making
big adjustments early and much finer ones as they close in on the bullseye.

**AdamW optimizer.** The specific algorithm used to turn "which direction
reduces the loss" into an actual weight update. It keeps a running memory of
recent gradients to smooth out the updates, and separately applies "weight
decay," a gentle pull toward smaller weight values overall, which helps
avoid the network relying too heavily on any single weight.

**Gradient clipping.** A safety cap on how large a single weight update is
allowed to be, even if the raw calculation suggests a huge one. Prevents one
unusually bad batch of training data from throwing the whole network wildly
off course.

**Epoch.** One complete pass through every training example.

**Batch.** A small group of examples (32, here) processed together at once,
rather than the whole dataset in one go, mostly because of memory limits.

**Checkpoint.** A saved snapshot of everything a network currently knows,
written to disk so training can be safely paused and resumed. This project
saves two: `last.pt` (always the most recent epoch) and `best.pt` (only
overwritten when an epoch beats every previous one). See the fourth bug in
EXPLAINED.md for why that distinction turned out to matter.

**Overfitting.** When a network starts memorizing quirks specific to its
training examples rather than learning the general pattern, which can
actually make its performance get *worse* the longer it trains, even as its
score on the training data itself keeps looking better. This project's own
loss curve shows this directly: it improves until about epoch 45, then
climbs back up through epoch 60.

### Evaluation

**Benchmark.** A fixed, shared test everyone agrees to use, so different
methods can be fairly compared against each other on identical conditions.

**Calibration.** Checking your own measurement against something with a
known, publicly agreed answer before trusting anything else you measure. If
your ruler doesn't agree with a value everyone knows, don't trust it on
values nobody has already verified for you either.

**Point-to-mesh distance (P2M).** A second way of measuring closeness,
against the true smooth surface rather than a set of sampled points. Not
used in this project's reported results: the calibration check found this
project's P2M implementation landing at roughly a sixth of the published
value for an identical method on identical shapes, meaning it measures
something different from what the source papers mean by P2M.

**Sparse / dense.** In this project's benchmark, "sparse" means a cloud
sampled down to 10,000 points and "dense" means 50,000 points.

### Engineering practices

**Resumable / fault-tolerant.** Designed so that an interruption (a
crashed process, a disconnected cloud session) costs only the work in
progress, not everything done before it. This project's training resumes
from its last saved epoch, and its benchmark saves each individual test
result to disk the moment it finishes rather than only at the very end.

**Unit test / regression test.** A small, automated check that a specific
piece of code behaves correctly, run repeatedly so a bug that was already
found and fixed can't silently come back unnoticed. This project has 38 of
them, four of which exist specifically because of real bugs found during
development.

**GPU (graphics processing unit).** A processor originally built for video
game graphics that happens to be extremely fast at the repetitive math
neural networks require. Training this project's network took over 4 hours
per epoch on a regular processor (CPU) and a few minutes per epoch on a
free cloud GPU (a Kaggle or Colab T4).

---

## Part 2: Interview questions and answers

These are written to be answered *using this project as the example*, with
real numbers rather than generic definitions. Each one includes what a
strong answer actually sounds like.

### For a Software Engineering (SDE) interview

**Q: Walk me through a bug you found and fixed in a real project.**

The frame-rotation bug is the strongest one to tell: during training, the
noisy input patch and its correct-answer target both need rotating into the
same coordinate frame, or they can never be compared meaningfully.
`pca_alignment` returned a rotation `R` and the code applied `R` to the
input but the inverse, `R^-1`, to the target, a one-character-scale mistake
with a large effect because `R` is different for every single patch. I
caught it by writing a test that constructs a synthetic patch, applies both
versions of the rotation, and measures the distance between input and
target: 0.040 with the bug present versus 0.016 with the correct rotation,
against ground truth that the true gap should be close to 0.016. Point
being: I didn't just eyeball the fix, I wrote a test that fails on the old
code and passes on the new code, which is what `test_noisy_and_clean_land_in_the_same_frame`
does now, permanently.

**Q: How do you design for a system that might get interrupted partway
through?**

Save progress at the smallest unit of work that's expensive to redo, not at
the end of the whole job. This project's benchmark runs 12 separate test
conditions, each taking real GPU minutes. The first version only wrote
results to disk after all 12 finished, so a Colab disconnect after 4 of
them lost all 4. I changed it to write a small JSON file to disk immediately
after each individual condition finishes, and to check that file first
before recomputing anything. A later session, even a different Google
account with an empty Drive, picks up exactly where the last one stopped.
Training resumability works the same way: every epoch writes a checkpoint
with the model, optimizer, and scheduler state, so `resume=` reconstructs
training exactly where it left off, not just the weights.

**Q: How do you handle a dependency or environment that behaves differently
across platforms?**

The training notebook needed to run on both Kaggle and Google Colab, which
mount uploaded data at different filesystem paths (`/kaggle/input/...` vs
`/root/.cache/kagglehub/...`). Rather than hardcode either path, the setup
cell searches a list of candidate roots for a directory containing a known
marker file, and on Google Drive specifically, wraps the mount attempt in a
try/except so a failure (in practice, a browser blocking Google's sign-in
popup) falls back to an unpersisted local directory and prints the likely
fix, instead of crashing the whole notebook on cell one.

**Q: Tell me about a time you had to debug something without being able to
reproduce it locally.**

A downloaded 58 MB checkpoint file failed to load with no informative error
message. Since a `.pt` file is a zip archive under the hood, I checked
whether it had been silently extracted rather than downloaded as a file
(some browsers do this automatically), found a `best/best/{data.pkl,
data/*}` folder structure exactly matching torch's internal zip layout, and
wrote a small script to re-zip those files uncompressed under the correct
top-level name. Loaded correctly afterward with its full 46-epoch training
history intact. The lesson: when a binary file "won't load," check whether
it's actually still a file.

**Q: How do you keep duplicated logic from drifting out of sync?**

Early on, the tracking and measurement logic was about to get copy-pasted
into a second notebook. Instead it went into one shared module,
`pointdenoise/engine.py` and friends, imported by both the CLI script and
both notebooks. When I later fixed the "returns last epoch instead of best
epoch" bug, fixing it once in `engine.py` fixed it everywhere that imports
`train()`, rather than needing the same fix applied and re-verified in three
separate copies.

### For a Machine Learning interview

**Q: Why process the point cloud in small patches instead of feeding the
whole cloud into the network at once?**

Two reasons. Computationally, a cloud has tens of thousands of points, and
comparing every point to every other point (which attention does, at cost
that grows with the square of how many points you feed in) becomes far too
expensive at full scale. More importantly, whether a point needs to move is
mostly a local decision: it depends on the shape of the surface immediately
around it, not on a point on the other side of the object. Patches of 256
neighbors give the network exactly the information that's actually relevant
to each decision, and keep the attention computation a fixed, manageable
size regardless of how big the input cloud is.

**Q: Why give attention explicit access to 3D positions rather than letting
it learn geometric relationships on its own?**

Standard attention only sees the query and key vectors it's given; it has
no built-in notion of physical space unless that information is somehow
encoded into those vectors, which is an awkward, indirect way to teach a
network "closer things usually matter more, and direction matters too."
This project instead computes the actual 3D offset between every pair of
points, passes it through a small MLP, and adds the result directly onto
the attention scores as a bias. That gives the network a direct, unambiguous
signal about geometry rather than asking it to reconstruct that from
scratch, and it's a small amount of extra compute for what turned out to be
a meaningful architectural choice.

**Q: You found that training with a single fixed noise level hurt
performance. Explain what happened and why the fix worked.**

Training only ever showed the network patches with exactly 2% noise added.
The network learned one specific correction size and applied it
indiscriminately. Tested at 1% noise, an amount that's already fairly close
to correct, that same fixed correction consistently overshot, and measured
Chamfer distance came out 84.5% *worse* than doing nothing at all on a
synthetic test. The benchmark told the same story on real data: only a 2%
improvement over the noisy input at 1% noise, versus 56% and 73% at 2% and
3%. The fix was to sample a random noise level between 0.5% and 3% for
every single training patch instead of a fixed value, which is much closer
to how the network will actually be used (it doesn't know in advance how
noisy new data will be). That one change took the 1% case from +2% to +40%
improvement, and every other noise level improved too, including the 2%
level the old model had been exclusively trained on.

**Q: How do you know your model isn't just overfitting to the benchmark?**

Two separate lines of evidence. First, the loss curve itself: training loss
dropped until about epoch 45 and rose again through epoch 60, which is
overfitting showing up directly in the training signal, not just inferred
from a gap between train and test performance. `best.pt` is the checkpoint
from the epoch that never got beaten, and the code now explicitly returns
that checkpoint rather than whatever epoch the loop happens to stop on.
Second, I resumed training from that epoch-46 checkpoint through the
remaining 14 epochs specifically to check whether stopping early had cost
anything, and the resulting benchmark scores landed within 1-2% of the
original, confirming epoch 46 is a genuine, repeatable optimum for this
setup rather than an artifact of one particular training run.

**Q: How would you improve this model's results further?**

The gap to the top three published methods (ScoreDenoise, I-PFN,
P2P-Bridge) isn't a tuning gap, it's a different-formulation gap: those use
score matching, iterative refinement over multiple passes, or a
Schrödinger-bridge-based diffusion approach, rather than a single
displacement prediction per patch like this project does. The most direct
next step actually available in this codebase is `iters` in
`denoise_cloud`, which re-runs the whole model on its own output; that's
already wired up but untested at `iters > 1`. Beyond that, adding an
iterative refinement loop into training itself, rather than only at
inference time, is the kind of change that would move this from "a good
single-pass method" toward the family the top methods belong to.

**Q: What's the point of the repulsion loss, and what would happen without
it?**

Chamfer distance alone can be satisfied by predictions that clump together:
if every predicted point lands extremely close to just a handful of true
points, the "predicted to nearest true" half of Chamfer distance can look
great while the "true to nearest predicted" half suffers, and visually the
result would be uneven clumps rather than a smooth, evenly-covered surface.
Repulsion adds an explicit penalty (a Gaussian falloff on the distance to a
point's own nearest neighbors) that pushes points apart once they get too
close, encouraging the kind of even spread Chamfer distance alone doesn't
guarantee on its own.

### For a Data Science interview

**Q: How do you decide whether a result is actually comparable to
published numbers, rather than just similar-looking?**

Calibration: run a method whose score is already published (this project
uses a bilateral filter) through your own measurement pipeline, on the
exact same test shapes, and compare. This project's Chamfer distance came
out at 0.84x the published bilateral score, comfortably inside a tolerance
band meant to catch real convention mismatches (a factor of 2 usually means
a squared-vs-unsquared or one-sided-vs-two-sided definition difference), so
CD numbers can be fairly placed in the same table as everyone else's. The
same check on P2M came back at 0.17x, nowhere near that band, so P2M got
excluded from the reported comparison entirely rather than quoted anyway.
Calibration is what turns "this number looks good" into "this number can
be trusted."

**Q: Your evaluation reports an average across shapes. What does that
average hide, and how would you check?**

Averages hide variance between individual cases. On a quick check with the
bilateral baseline on the PU-Net sparse/1% test set, the mean Chamfer
distance was 3.08, but individual shapes ranged from 1.61 up to 8.35, and
the three worst shapes alone accounted for a third of the total error. A
mean by itself can't tell you whether a method is consistently decent or
excellent on most shapes and badly wrong on a few, and those are very
different stories for someone deciding whether to trust the method on a new
shape. `scripts/benchmark.py` supports writing per-shape results to a CSV
specifically so that spread is visible, not just the summary number.

**Q: How would you explain this project's headline result to someone
non-technical?**

Out of nine methods tested on the same fair test, six of them make the
scanned points meaningfully more accurate than doing nothing at all, and
this one is in the middle of that group: clearly better than the older,
simpler approaches, and behind the three newest ones, which use
fundamentally different techniques rather than just being more carefully
tuned versions of the same idea. Compared to leaving the data untouched,
this method removes 40% of the positioning error at low noise levels and
86% at high noise levels, which is the number that actually demonstrates
the method works, independent of how it stacks up against everyone else's
submission.

**Q: What would make you distrust a benchmark result, even a good one?**

If a metric's calibration check disagreed with the published value by a
large factor and the number still got reported anyway. That's exactly why
P2M isn't in this project's final table: not because the underlying method
did badly on P2M, but because there's currently no way to know whether the
computed value means the same thing as everyone else's P2M, and quoting it
regardless would be presenting an unverified number as if it had been
checked. The honest move when a metric can't be calibrated is to say so
and exclude it, not to quietly report it because it happens to look good.
