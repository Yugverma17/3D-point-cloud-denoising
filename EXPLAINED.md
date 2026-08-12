# This project, explained simply

This document explains the whole project in plain language, from "what even
is a point cloud" all the way down to why one specific line of code was
wrong. No prior knowledge assumed. If a word might be unfamiliar, it gets
explained the first time it shows up, and again in the [glossary](#glossary)
at the end.

If you only read one paragraph, read this one: a 3D scan gives you a cloud
of dots that trace out the shape of an object, but the dots are never quite
in the right place, like a photo taken with shaky hands. This project builds
a computer program that looks at those shaky dots and nudges each one back
toward where it should really be. It's tested the same way scientists test
any new method: on a fixed set of practice shapes, against a fixed set of
older methods, using a fair, agreed-upon ruler for "how good is good."

---

## Part 1: What's a point cloud, and what's wrong with it?

Imagine you had a statue of a horse, and you covered it in ten thousand tiny
stickers, each one glowing a different color. If you took a photo of just
the stickers and erased the horse underneath, you'd still be able to tell
it's a horse, just from the pattern the dots make. That's a point cloud: a
big pile of individual points in 3D space that, together, trace out the
shape of something.

Real point clouds come from 3D scanners: devices that bounce light or lasers
off an object and measure how long it takes to bounce back, over and over,
from thousands of slightly different spots. Robots use them to "see" the
world in 3D. Self-driving cars use them to notice pedestrians. Video game
studios use them to turn a real statue into a 3D model.

Here's the problem: no scanner is perfect. Every single dot ends up a tiny
bit off from where it should truly be, like tracing a picture with a hand
that won't stop shaking. Zoom in and the horse's smooth leg turns into a
fuzzy little cloud of dots scattered around where the leg actually is. That
fuzziness is called **noise**, and it makes the point cloud harder to use
for almost anything downstream: 3D printing, robot navigation, measuring the
object accurately.

**Denoising** means taking that shaky, fuzzy set of dots and nudging each
one back toward its true position, undoing the shake without erasing the
actual shape. That's the entire goal of this project: build something that
takes noisy dots in, and gives cleaner dots out.

## Part 2: How do you prove you're actually good at this?

Here's a trap that's easy to fall into: build something that seems to clean
up dots, try it on a couple of examples, and declare victory. The problem is
you have no idea if you're actually good, or just got lucky, or are worse
than something a first-year student could build in an afternoon.

Scientists solve this with a **benchmark**: a fixed, shared test that
everyone agrees to use. For point cloud denoising, that benchmark works like
this:

1. Take 20 real 3D shapes (a camel, a cow, a chair, some machine parts,
   even a plain sphere). These come from something called **PU-Net**. There's
   a second, smaller set of 10 shapes too, called **PC-Net**, for extra
   variety.
2. For each shape, sample it down to a fixed number of points: either
   10,000 (called "sparse") or 50,000 (called "dense").
3. On purpose, shake every point by a known, controlled amount: 1%, 2%, or
   3% of the object's size. This "known amount" matters a lot, because now
   everyone testing their method knows exactly how far off the dots
   actually are, which means you can measure exactly how well you undid it.
4. Everyone runs their denoiser on these exact same noisy files, and reports
   how close their cleaned-up dots land to the real, un-shaken positions.

The result is a big scoreboard: one row per method, one column per test
condition (10K points at 1% noise, 50K points at 3% noise, and so on), and
in each cell, a number saying "how far off, on average, were this method's
cleaned-up dots."

That number is called **Chamfer Distance**, usually shortened to **CD**.
Here's how to picture it: for every dot you cleaned up, find the single
closest correct dot and measure that distance. Do the same thing in reverse,
for every correct dot, find the closest dot you produced. Average both sets
of distances together. That combination matters because it stops a cheap
trick: if you only measured "how close is my cleaned dot to a correct one,"
you could cheat by squishing all your dots onto just a handful of correct
spots and ignoring the rest of the shape entirely. Measuring in both
directions means you also get penalized for leaving parts of the shape
uncovered. Lower CD is better; it means your dots ended up closer to where
they should be.

There's a second measurement called **P2M** ("point to mesh") that's
supposed to measure distance to the *actual smooth surface* of the object
rather than to sampled dots. It's the more honest of the two numbers in
principle, because Chamfer distance can be gamed a little by bunching points
up near sampled dots. This project tried to compute P2M and got a number 6
times smaller than everyone else's published P2M scores for the exact same
method on the exact same shapes. That's a strong sign this project's P2M is
measuring something different from what the published papers mean by P2M,
not that this project is secretly amazing. So P2M got dropped from the
final results entirely rather than quoted as if it meant something. More on
that in [Part 9](#part-9-checking-if-were-actually-any-good).

## Part 3: The plan for actually fixing the dots

Here's the core idea, in one sentence: for every single dot, look at its
nearby neighbor dots, and use that little neighborhood to guess which
direction and how far that one dot should move.

Why look at just the neighbors, and not the whole cloud at once? Two
reasons. First, a cloud has tens of thousands of points, and comparing every
point to every other point at once is far too much information to make
sense of. Second, and more importantly, whether a dot needs to move mostly
depends on what's immediately around it, the same way you can tell if one
brick in a wall is crooked by looking at the bricks right next to it,
without needing to study the whole building.

So the pipeline for one single dot looks like this:

1. Grab its 256 closest neighbor dots. This little neighborhood is called a
   **patch**.
2. Rotate and resize that patch into a standard, predictable orientation
   (details below, this turns out to matter a lot).
3. Feed the patch into a neural network (a trained computer program, more
   in [Part 5](#part-5-the-robot-brain)), which outputs a small nudge: "move
   this much, in this direction."
4. Apply the nudge. That's the cleaned-up position for that one dot.

Repeat for every dot in the cloud, and you've denoised the whole thing.

## Part 4: Building one patch

This part of the project lives in two files: `pointdenoise/geometry.py` and
`pointdenoise/data.py`.

### Picking the neighbors

The first version of this picked "every point within some fixed distance"
of the center dot. That sounds reasonable, but here's the problem: imagine
you're standing in a huge, mostly empty gymnasium with only three other
people scattered far away, versus standing in a full classroom of thirty
people close together. "Everyone within 10 feet of me" finds a full patch
in the classroom and almost nobody in the gym. Point clouds have the exact
same problem: some areas are packed with dots, others are sparse. A fixed
distance rule quietly leaves some dots with almost no neighbors, and a patch
that thin can't be built into a stable, reliable orientation.

That's exactly what happened here: with a fixed distance, a third of all
the points in a test cloud ended up with too few neighbors to work with, and
those points just never got denoised at all, staying just as noisy as when
they started. The fix was simple once found: instead of "everyone within X
feet," use "my 256 closest neighbors, whoever they are and however far
away." That always gives a full patch, no matter how crowded or sparse that
part of the cloud is.

### Rotating the patch (and the biggest bug in the whole project)

If you handed the network the same little neighborhood of dots, but rotated
to a totally random angle every single time, it would have to somehow learn
that a horse's leg looks the same whether it's tilted 10 degrees or 190
degrees. That's a much harder problem than it needs to be. The fix is to
always rotate every patch into the *same predictable orientation* before
the network ever sees it, the same way you'd always pack a suitcase the
same way so you can find things without thinking about it.

The rotation is computed from the patch's own shape (its three main
directions of spread, found with something called PCA), giving each patch
its own custom rotation that straightens it out consistently.

Here's where the biggest bug in the entire project happened. During
training, the network needs two things for each patch: the noisy input, and
the correct answer (where those points should really be). Both need to be
rotated into the same frame, or the network can never learn anything
sensible from them. Picture a language student being given a question in
French, but the answer key in German. No matter how hard that student
studies, they can never correctly match questions to answers, because
they're not even in the same language. That's what was happening here: the
noisy input was rotated one way, and the correct answer was accidentally
rotated the *opposite* way. And because each patch gets its own custom
rotation, this mismatch was different every single time, so there was no
consistent pattern for the network to ever discover, no matter how long it
trained. On a test patch, the input ended up 2.5 times further from its
supposedly-correct answer than it should have been. The fix was a one-line
correction: rotate both the input and the answer using the exact same
rotation.

### Making patches a consistent size

A patch taken from a huge statue and a patch taken from a tiny figurine
should look similar to the network, the same way a photo of a mountain and
a photo of a pebble can be resized to fill the same size frame. Each patch
gets divided by its own radius so it always roughly fills a ball of size 1,
regardless of how big the original object was.

### Adding practice noise, and the second-biggest bug

To train the network, it needs lots of examples of "here's a noisy patch,
here's what it should really look like." The noisy examples are made by
starting from clean, correct shapes and deliberately shaking every point by
a controlled random amount, the same way the benchmark does.

The first version of training always shook points by exactly 2%, every
single time. That sounds tidy, but it taught the network exactly one lesson:
"always correct for 2% worth of noise." Applied to a patch that only had 1%
noise, an amount that's already fairly close to correct, the network still
applied its full "2% worth" of correction and *overshot*, actually pushing
points further from correct than if it had done nothing at all. It's like a
doctor who has only ever treated patients with a high fever: give them a
patient with no fever, and their trained instinct is still to prescribe
fever medicine, which does more harm than good.

The measured damage was dramatic: at low noise (1%), the trained network
made the Chamfer distance 84% *worse* than doing nothing. The fix was to
train on a random noise amount every time, somewhere between 0.5% and 3%,
instead of always exactly 2%. After that fix, the same network improved
every single noise level, including the one it used to be trained on
exclusively.

## Part 5: The robot brain

This lives in `pointdenoise/model.py`. A **neural network** is a computer
program built out of a lot of simple math operations (mostly multiplying
numbers together and adding them up) chained together, where the exact
numbers used (called weights) get slowly adjusted during training until the
whole chain reliably turns "noisy patch in" into "good correction out."
Nobody hand-writes those weights; they get discovered automatically by
practicing on lots of examples.

This project's network has two main ingredients:

**EdgeConv**, which builds a description of each point's immediate
neighborhood. For every dot, it looks at its 20 (or 10, for a finer pass)
nearest neighbors and asks "how are you different from me," combining all
those little differences into a compact fingerprint. Think of it as each
dot quickly interviewing the dots standing right next to it.

**Attention**, borrowed from the same kind of technology behind modern
chatbots, which lets every dot in the patch "check in" with every *other*
dot in the patch, not just its immediate neighbors. Normally, attention has
no built-in sense of physical space; it just looks at abstract number
patterns. That's the wrong tool for a job about 3D positions, so this
project adds something extra: the actual 3D direction and distance between
every pair of dots gets fed into the attention mechanism directly, so a dot
that's far away and in an unhelpful direction gets naturally down-weighted,
and a close, well-placed neighbor gets more say. EdgeConv handles "what's
immediately around me," and attention handles "let me also get the fuller
picture from everyone else in this patch."

The network's final output for each point isn't a brand new position, it's
a small **displacement**: "move this much, in this direction." That choice
matters for a subtle but important reason: the very last layer of the
network starts out deliberately set to output exactly zero. That means an
untrained, brand new network does nothing at all to its input, it can never
make a cloud worse before it's learned anything, because "do nothing" is
its starting point. As training progresses, it slowly learns to push those
zero-nudges into useful, non-zero ones.

## Part 6: Learning from mistakes

This lives in `pointdenoise/losses.py`. A **loss function** is how you tell
the network "here's how wrong you were," so it knows which direction to
adjust itself.

The main loss is Chamfer distance, the same measurement described in
[Part 2](#part-2-how-do-you-prove-youre-actually-good-at-this), used here
during training instead of just for the final scoreboard. One tweak: rather
than using the raw squared distance, this project takes the square root of
each distance before averaging. That softens the effect of the occasional
badly-placed point, so one unlucky mistake doesn't dominate the whole
training signal and cause overreactions, the same way a coach correcting a
gymnast's routine should focus on the overall performance rather than
completely reworking the routine because of one wobble.

There's a second loss called **repulsion**, whose entire job is to stop
points from clumping together too closely. Chamfer distance alone can be
satisfied by bunching a bunch of points onto the same small area, similar
to how kids in a game might all crowd into one corner if nobody tells them
to spread out. The repulsion loss specifically penalizes points that end up
too close to their neighbors, encouraging an even spread across the shape's
actual surface.

## Part 7: Practicing (the training loop)

This lives in `pointdenoise/engine.py`.

Training happens in **epochs**: one epoch means the network has practiced
on every training shape once. This project trains for up to 60 epochs, and
within each epoch, the training data is split into small **batches**
(groups of 32 patches at a time) rather than shown all at once, mostly
because a computer's memory can only hold so much at a time, the same way
you'd read a book one page at a time rather than trying to see every page
at once.

After every batch, the network's weights get nudged slightly, based on how
wrong its guesses were (Part 6). How big that nudge is is called the
**learning rate**, and this project starts with relatively large nudges and
gradually shrinks them as training goes on, following a smooth curve called
"cosine annealing." Picture a dart player: early throws involve big
adjustments to get roughly on target, and later throws involve much
smaller, finer adjustments to nail the bullseye.

After every single epoch, the current state of the network gets saved to
disk as a **checkpoint**, essentially a snapshot or save file of everything
the network currently knows. Two checkpoints get kept: `last.pt`, which is
always just overwritten with the most recent epoch, and `best.pt`, which
only gets overwritten when a new epoch beats every previous epoch's score.
Saving checkpoints like this means training can be safely interrupted (a
computer crash, or in this project's case, several Google Colab sessions
disconnecting mid-run) and resumed later from exactly where it left off,
rather than starting completely over.

### The third bug: trusting the wrong save file

For a long time, it seemed reasonable to assume that more training is
always better, and that whatever epoch training happens to stop on is the
best one. That assumption turned out to be wrong. A real 60-epoch training
run showed the loss (the "how wrong am I" number from Part 6) steadily
dropping until around epoch 45, and then it started climbing back up again,
even though the learning rate was still shrinking the whole time. That's a
classic sign of **overfitting**: the network started memorizing quirky
patterns specific to its practice examples rather than learning the general
rule, the same way a student who over-studies practice questions can start
picking up weird tricks that don't actually work on the real exam.

`best.pt` correctly kept pointing at epoch 46 the whole time, since nothing
after that ever beat it. But the code that trains the network was handing
back whatever the network looked like at the very last epoch it ran, not
the officially-best one, the equivalent of a photographer handing over the
very last, slightly blurry photo from a shoot instead of the best one they
actually took. This mattered in practice: a benchmark run right after
training accidentally used the worse "last epoch" version instead of the
genuinely best one. The fix makes the training code always reload and hand
back the actual best checkpoint, regardless of which epoch the loop happens
to stop on.

## Part 8: All four bugs, in one place

| # | What was wrong | What it felt like | The fix |
|---|---|---|---|
| 1 | Input and its correct answer were rotated in opposite directions | A question in French, answer key in German | Rotate both by the same amount |
| 2 | Patches used "everyone within X distance," which left sparse areas with too few neighbors | Trying to fill a classroom-sized group using only people within reach in an empty gym | Use "my 256 closest neighbors" instead, which always works |
| 3 | Training always used exactly 2% noise | A doctor who only ever treats a fever, and overmedicates a patient who doesn't have one | Train with noise randomly varied between 0.5% and 3% |
| 4 | Training returned whichever epoch it happened to stop on, not the best one | Handing over the last, blurriest photo instead of the best one from the shoot | Always reload and return the actual best checkpoint |

Each of these is also backed by an automated test in `tests/`, specifically
checking that the bug can't silently come back.

## Part 9: Checking if we're actually any good

This lives in `pointdenoise/baselines.py` and `pointdenoise/benchmark.py`.

Before trusting a fancy new method, it helps to also test some simple,
well-understood ones for comparison, the same way you'd compare a new bike
against a regular bike before comparing it to a race car. This project
includes a classical method called a **bilateral filter**, which slides
each point a little along its estimated surface direction based on a
weighted average of nearby points, no learning involved, just a formula.

That bilateral filter turns out to be the single most important tool in
this whole project, for a reason that has nothing to do with how good it
is at denoising. Its scores are published in the same official tables this
project compares against. That means it can be used as a **calibration**
check: if this project's own measuring code reports a bilateral filter
score close to the officially published one, that's strong evidence the
measuring code is measuring the same thing everyone else is, in the same
units, using the same rules. If it doesn't match, nothing else this project
measures can be trusted either, no matter how good the results look,
because there'd be no way to know if the "ruler" itself is broken.

That's exactly what happened here: Chamfer distance matched the published
bilateral score closely (about 84% of the way there, well within a normal
range of implementation differences), so CD numbers from this project can
be fairly placed next to the published numbers. P2M did not match at all;
this project's P2M came out about six times smaller than the published
P2M for that same filter on those same shapes. That's a clear sign of a
units or definition mismatch somewhere, not evidence that this project's
denoiser secretly beats everyone at P2M. So P2M gets left out of the real
comparison entirely, even though quietly including it would have made the
results look better than they honestly are.

## Part 10: The actual story of training this thing

The first full training attempt used CPU only (no graphics card), which
turned out to be wildly impractical: one epoch alone took over 4 hours,
which would have made a full 60-epoch run take roughly 10 days. Training
moved to free cloud GPUs instead, first on Kaggle, later on Google Colab,
each of which gives a shared graphics card (a T4) that trains the same
model in minutes per epoch instead of hours.

The first real GPU run trained at a single fixed 2% noise level for 60
epochs, and its results were fine at higher noise but barely helped at all
at low noise (bug #3 above). Fixing that (randomizing the noise per patch)
and retraining gave a real improvement at every single noise level.

Along the way, the process ran into a long string of practical obstacles,
each one worth mentioning because none of them are really about machine
learning: a downloaded checkpoint file that arrived accidentally unzipped
and had to be manually repackaged; a Kaggle notebook that failed because a
required input dataset was never attached; a Colab notebook that failed to
even start because a privacy-focused browser blocked Google's sign-in
popup; and several Colab sessions that simply disconnected mid-run, losing
whatever hadn't been saved yet. Each of these got fixed by making the
notebooks more careful about saving progress constantly (to Google Drive,
cell by cell) rather than only at the very end, so a disconnect only costs
the one step that was in progress, not everything done before it.

That resilience mattered directly: partway through the benchmark, one Colab
session disconnected after finishing only 4 of 12 test conditions. Because
progress was now being saved after each individual test condition, a later
session could pick up exactly where the last one left off instead of
starting the whole benchmark over.

There was also a moment of real confusion during this process: a resumed
training run said "resuming at epoch 46," which briefly looked like a
mistake, as if the wrong, older checkpoint had been uploaded by accident.
The actual explanation turned out to be simpler and more interesting: that
genuinely was the right, best checkpoint, exactly as bug #4 predicts. To
double check, training was resumed all the way through the remaining 14
epochs, and the resulting benchmark scores came back within about 1-2% of
the original epoch 46 numbers, confirming epoch 46 really is a stable,
genuine best point for this particular training setup, not an accident of
stopping early.

The very last decision made was to skip testing on the second benchmark set
(PC-Net, 10 extra shapes) entirely. The tooling fully supports it and could
run it at any time, but chasing a second full test across yet another
multi-hour cloud session, after several had already disconnected partway
through, wasn't worth it: the first test set (PU-Net) already answers the
question this project set out to answer.

## Part 11: The final scorecard, explained

Here's the real result, Chamfer distance only (see Part 9 for why P2M
doesn't appear):

| Method | 10K pts, 1% noise | 10K pts, 2% noise | 10K pts, 3% noise | 50K pts, 1% noise | 50K pts, 2% noise | 50K pts, 3% noise |
|---|---|---|---|---|---|---|
| Bilateral (simple formula, no learning) | 3.65 | 5.01 | 7.00 | 0.88 | 2.38 | 6.30 |
| PCNet | 3.52 | 7.47 | 13.10 | 1.05 | 1.45 | 2.29 |
| DMRDenoise | 4.48 | 4.98 | 5.89 | 1.16 | 1.57 | 2.43 |
| GLR | 2.96 | 3.77 | 4.91 | 0.70 | 1.59 | 3.84 |
| ScoreDenoise | 2.52 | 3.69 | 4.71 | 0.72 | 1.29 | 1.93 |
| PD-Flow | 2.13 | 3.25 | 5.19 | 0.65 | 1.42 | 3.90 |
| I-PFN | 2.31 | 3.43 | 5.24 | 0.66 | 1.05 | 2.54 |
| P2P-Bridge | 2.28 | 3.20 | 3.99 | 0.59 | 0.90 | 1.56 |
| **This project** | **2.89** | **4.07** | **5.29** | **0.76** | **1.40** | **2.77** |

Every number is "how far off, on average, were the cleaned dots," in tiny
units where lower means better. Reading straight down: this project sits
somewhere in the middle of the pack. It clearly beats Bilateral, PCNet, and
DMRDenoise in every single test condition, and it beats GLR and PD-Flow in
a couple of the harder ones. It doesn't beat ScoreDenoise, I-PFN, or
P2P-Bridge in any test condition.

That's a completely honest, respectable place to land, and here's why:
those top three methods aren't just better-tuned versions of the same idea
used here, they use fundamentally different approaches (matching statistical
"scores" of where points should be, iteratively refining predictions over
many passes, or a technique borrowed from a completely different branch of
math called a Schrödinger bridge). Beating them would take a genuinely
different idea, not just more training time on this one. Compared to doing
nothing at all (just leaving the noisy points as they were), this project's
method removes 40% of the error at low noise and 86% at high noise, which
is the number that actually shows the method works, separate from how it
stacks up against everyone else's homework.

## Part 12: Try it yourself

The [main README](README.md) has the full setup instructions: how to
install everything, run the automated tests, retrain the network, and
reproduce the benchmark table above from scratch. This document exists to
explain *why* everything in that codebase is built the way it is; the
README explains *how* to actually run it.

## Glossary

**Point cloud** - a shape represented as a big pile of individual dots in
3D space, instead of a smooth surface.

**Mesh** - the "connect the dots" version of a shape, made of flat
triangles instead of loose points. Used here as ground truth to measure the
more honest P2M distance against.

**Noise** - random, unwanted error. Here, each point in a scan being
slightly off from where it should truly be.

**Denoising** - the process of removing noise; nudging each noisy point
back toward its true position.

**Patch** - a small neighborhood of points around one center point, used as
the input the network actually looks at.

**Neural network** - a computer program built from many simple math
operations chained together, whose exact numbers (weights) are learned
automatically from examples rather than hand-written.

**Attention** - a technique that lets every item in a group "check in"
with every other item, weighing how much each one should influence the
others.

**Epoch** - one full pass through every training example.

**Batch** - a small group of training examples processed together at once.

**Checkpoint** - a saved snapshot of everything a neural network currently
knows, so training can be paused and resumed.

**Learning rate** - how big a correction the network makes after each
mistake during training.

**Loss function** - the formula that measures how wrong the network's
current guess is, which is what training tries to minimize.

**Chamfer Distance (CD)** - the main measurement in this project: the
average distance from each predicted point to its nearest true point, and
back the other way too.

**P2M (point to mesh)** - a similar measurement, but against the smooth
true surface instead of a set of points. Not used in this project's final
results because the numbers didn't match the published definition.

**Overfitting** - when a network starts memorizing quirks of its specific
training examples instead of learning the general pattern, which can make
it perform worse on new, unseen data even as it keeps "improving" on the
data it's already seen.

**Benchmark** - a fixed, shared test that everyone uses, so results from
different methods can be fairly compared.

**Calibration** - checking that your own measurement matches a known,
trusted reference value, before trusting anything else you measure.

**GPU** - a graphics card. Originally built for video games, it happens to
be extremely fast at the kind of repetitive math neural networks need,
which is why training moves from hours or days on a regular processor down
to minutes on one.

**Sparse / dense** - in this project, "sparse" means a cloud sampled down
to 10,000 points, and "dense" means 50,000 points.
