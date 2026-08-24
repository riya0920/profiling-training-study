# Profiling-Driven Training Optimisation Study

An instrumented baseline, then an optimisation ladder where every rung changes
exactly one thing and the order is chosen by the profile rather than by a blog
post. Negative results stay in the table.

> **Status: ~100% of what this machine can host.** Instrumentation, the ladder,
> repeats-and-report, committed profiler traces, **weak and strong DDP scaling
> studies executed across real processes**, an **ablation that refuted its own
> hypothesis**, a **cost-to-target-accuracy study**, and a **CPU-pinning
> experiment that had to validate its own instrument first** are done — all on
> **CPU**, labelled as such everywhere. No GPU exists here, so there is no GPU
> number anywhere, and `torch.compile` genuinely fails on this box. See
> [Roadmap](#roadmap).

## DDP scaling, actually executed — weak and strong

No GPU here, so the study runs DDP over **gloo** across real OS processes on CPU.
Genuinely distributed; labelled CPU/gloo rather than dressed up as a GPU result.

**Weak scaling** (per-worker batch fixed — "more data in the same time"):

| world | global batch | lr | samples/s (mean ± std) | speedup | efficiency |
|---|---|---|---|---|---|
| 1 | 32 | 0.0100 | 43.6 ± 1.6 | 1.00x | **100%** |
| 2 | 64 | 0.0200 | 69.8 ± 1.7 | 1.60x | **80%** |
| 4 | 128 | 0.0400 | 103.7 ± 13.9 | 2.38x | **59%** |

**Strong scaling** (global batch fixed — "the same job, faster"):

| world | per-worker batch | lr | samples/s (mean ± std) | speedup | efficiency |
|---|---|---|---|---|---|
| 1 | 128 | 0.0100 | 51.2 ± 4.0 | 1.00x | **100%** |
| 2 | 64 | 0.0100 | 71.8 ± 8.3 | 1.40x | **70%** |
| 4 | 32 | 0.0100 | 101.0 ± 6.4 | 1.97x | **49%** |

Strong is worse at every size, as it should be: the per-worker batch shrinks, so
each worker does less compute per synchronisation and fixed per-step overhead is
amortised over less work.

### A bug the strong-scaling run exposed

The LR was scaling against the **per-worker** batch — right by accident under
weak scaling, where they coincide. Under strong scaling it printed
`lr 0.01 → 0.02 → 0.04` while the global batch never moved from 128. That is a
hyperparameter change wearing a scaling study's clothes. Fixed to reference a
fixed base batch; strong now holds `lr = 0.01` throughout.

## The measurement that refused to cooperate

An earlier version of this README stated **"16.7% is measured communication"**
and attributed the scaling gap to it. **That number was noise and the claim was
wrong.** Repeating the paired `no_sync` comparison seven times per worker shows
why:

| world | comm fraction (median) | spread across 7 trials |
|---|---|---|
| 2 | 6.3% | **±20.7 points** |
| 4 | 2.3% | **±22.3 points** |

The spread is three to ten times the median; consecutive runs of the same config
gave 15.2% and 1.2%. On a contended CPU this estimator is a coin flip.
**Communication cost is unmeasured on this hardware**, and the median is now
reported *with* its spread so a reader can see it is unusable.

### The ablation refuted the fallback hypothesis too

With communication unmeasurable, the remaining explanation was CPU contention
from the in-process decode. The ablation is one command — rerun world 4 with
`--decode-cost 0`:

| configuration | efficiency at world 4 |
|---|---|
| decode_cost = 60 (normal) | 59% ± 13.9 |
| decode_cost = 0 (ablation) | **65% ± 0.9** |

Removing *all* CPU-side data work recovers ~6 points, and the two overlap within
the baseline's spread. **Decode contention is not the dominant term either.**

Both candidate explanations have been tested and neither accounts for the ~35–41%
loss. What remains untested is core oversubscription (4 workers plus gloo threads
on 4 physical cores) and DDP's gradient-bucketing overhead.

**Naming an unexplained gap is more useful than assigning it to whichever cause
was measured last.** Full write-up in [docs/SCALING.md](docs/SCALING.md).

## Committed profiler traces

`python -m trainlab.profile_trace` writes Chrome traces to `results/traces/`,
loadable in `chrome://tracing` or Perfetto. A screenshot is a claim about a
trace; the trace is the evidence, so the trace is what is committed.

| config | data_wait | forward | backward | optimizer |
|---|---|---|---|---|
| baseline | **25.2%** | 33.2% | 38.7% | 2.9% |
| +prefetch | **0.4%** | 47.7% | 48.3% | 3.6% |

Data-wait collapses from a quarter of step time to essentially nothing, which is
the entire argument for attacking the dataloader before touching precision or
layout.

## The one rule

> You do not get to optimise anything you have not measured.

Enforced in code, not by discipline:

* `_assert_single_change` **fails the run** if a rung changes more than one knob
  (batch size and its pre-declared LR rule count as one change, because the LR
  follows from the batch).
* Warmup steps are excluded from every measurement, and how many were excluded is
  recorded in the row.
* Every row records its device, timer kind, and torch version, so rows measured
  on different hardware can never be silently merged into one table.
* Rungs that cannot run on the current device are recorded as **skipped with a
  reason**; rungs that crash are recorded as **failed with the error**. A missing
  row and a deliberately omitted row look identical in a table, so neither is
  allowed to be missing.

## Run it

```bash
pip install -r requirements.txt
make test                                  # 12 tests, all about the instrument
make ladder                                # run every rung, 3 repeats each
make report                                # regenerate REPORT.md + figures from the ledger
```

On a GPU box the same command runs the CUDA-only rungs (`pin_memory`, `tf32`)
that are skipped on CPU, and the timer switches from `perf_counter` to
`torch.cuda.Event` automatically — wall-clock timing around an async kernel
launch measures how fast Python queued the work, not how long the GPU took.

## What the profile actually said

The instrumented FP32 baseline spends **~37% of step time in data wait**. That
single number set the entire rung order: while the device is idle a third of the
time waiting for input, precision and layout optimisations cannot help, because
kernels are not the constraint.

The obvious first move — mixed precision — would have been the wrong one here,
and the profile said so before any time was spent on it. That is the whole
argument for the word order in "profiling-driven distributed training".

Measured results, the generated figures, and the separability analysis are in
**[REPORT.md](REPORT.md)**, regenerated from `results/ladder_cpu.json` by
`python -m trainlab.report`. Nothing in that file is typed by hand.

## The synthetic workload, and why it is not cheating

`SyntheticImages` carries a tunable per-sample CPU cost that does **real numpy
work**, not `sleep`. That matters: a sleep is trivially hidden by any number of
workers and would make the dataloader rung look better than it is, whereas real
work competes for cores and the GIL the way JPEG decode and augmentation actually
do. The cost is set so the baseline reproduces the most common real training
bottleneck (CPU input pipeline starving the accelerator) on any machine, without
downloading ImageNet.

The trade this makes is stated plainly: these are **not** ImageNet numbers, the
model is deliberately small, and the absolute throughput means nothing outside
this workload. What transfers is the method and the rung order, not the values.

## bf16 vs fp16

`amp_dtype` defaults to **bf16**. bf16 keeps fp32's exponent range and loses
mantissa bits; fp16 keeps mantissa and loses range, which is why fp16 needs a
`GradScaler` to stop small gradients flushing to zero. The code enables the
scaler *only* for fp16 (`scaler = GradScaler(enabled=amp and dtype is fp16)`) —
running a scaler with bf16 is a common cargo-cult that costs a little time and
buys nothing.

What breaks if this choice is wrong: on hardware without native bf16 (pre-Ampere
NVIDIA, most CPUs without AMX), bf16 autocast is emulated and is **slower than
fp32** — which is exactly what the measured ladder on this CPU shows, and why
that rung is a negative result in the table rather than a deleted row.

## Cost

The direct compute cost of this study was **$0**: it ran on local hardware.
`python -m trainlab.cost --ledger results/ladder_cpu.json --rate 0.35` turns any
ledger into a cost-to-target table and a break-even for scaling, but the rate is
a required input rather than a hard-coded constant, because published on-demand
prices are not what anyone's bill actually says.

## Cost to a target *accuracy* — the question the ladder cannot answer

Every number in the optimisation ladder is samples per second. That is right for
*"is the training loop efficient?"* and wrong for the question anyone pays for,
which is *"what does it cost to get a model this good?"*

    cost to target = samples_to_target / samples_per_second x rate

**Both factors.** The ladder measures only the denominator, so it is structurally
blind to any change that moves throughput and convergence in opposite directions —
and so is every "we got a 2x speedup" measured the same way.

`make accuracy` trains each configuration until held-out accuracy crosses 85% on
two consecutive evaluations, at $0.35/hour:

| config | samples/s | steps to target | samples to target | $ to target | rank by throughput | rank by cost |
|---|---|---|---|---|---|---|
| batch64 | 106.0 | 187.5 | 12,000 | **$0.0110** | 1 | 1 |
| batch256_linear | 83.0 | 137.5 | 35,200 | $0.0412 | **2** | **4** |
| batch128_linear | 71.2 | 162.5 | 20,800 | $0.0284 | 3 | 3 |
| batch128_unscaled | 68.2 | 137.5 | 17,600 | $0.0251 | 4 | 2 |
| batch128_bf16 | 31.7 | 162.5 | 20,800 | $0.0639 | 5 | 5 |

**The winners agree and the orderings do not.** `batch256_linear` is the
second-fastest configuration and the second-most-expensive route to the target —
a throughput ladder promotes it while it costs **3.75x** the cheapest path to the
same accuracy. Across the table the spread is **5.8x**, and none of it is visible
in samples/s.

The mechanism is clean and it is exactly the textbook one: bigger batches **did**
cut the number of updates needed (187.5 → 137.5 steps) and raised the number of
*samples* 2.9x doing it. On CPU the throughput gain never arrived to compensate.

### A reporting bug this had first

The first version compared only `rank_by_throughput[0]` against
`rank_by_cost[0]`, found them equal, and printed *"the throughput winner is also
the cheapest here"*. True, and it buried the finding: a ladder does not just pick
a winner, it **ranks every rung**, and the result lives entirely in the rungs that
move. The comparison now covers full orderings and names the configuration
throughput over-rates — the one a ladder would promote and a budget would not.

## CPU pinning: the experiment that had to validate its own instrument

The ladder's biggest win was `num_workers=4`. On an 8-core box that leaves the
main process sharing all 8 cores with 4 decode workers — everybody oversubscribed,
threads migrating, caches thrown away. Does partitioning the cores beat sharing
them, even though partitioning gives each side strictly fewer?

Four arms, because `pinned_split` bundles *two* interventions and an experiment
that cannot separate its own interventions has measured neither:

| arm | main | workers | threads | samples/s (median of 7) | range |
|---|---|---|---|---|---|
| `default` | all | all | default | 87.8 | 79.2–105.2 |
| `threads_capped` | all | all | 4 | 99.5 | 65.3–129.0 |
| `pinned_split` | 0–3 | 4–7 | 4 | **140.6** | 129.4–146.3 |
| `pinned_half_shared` | 0–3 | 0–3 | 4 | 132.5 | 113.9–138.5 |

**Pinning is worth +41.3%**, an effect 3.5x the measured noise floor.

**The mechanism is not the one predicted.** `pinned_half_shared` puts both sides
on the *same* four cores — same core count, shared instead of disjoint — and
captures nearly all of the gain. Disjointness is worth a further +6.1%, which is
*inside* the noise floor and not separable. So what helps is confining the
training process to four cores **at all**; whether the dataloader gets its own
four is not established. Without that fourth arm the obvious reading of "+41.3%"
is "partitioning removed contention", and this data does not support it.

### The calibration arm did more work than the experiment

PyTorch already defaults to **4 threads** here — 8 logical cores, 4 physical — so
`threads_capped` is a no-op, and `default` and `threads_capped` are **the same
configuration measured twice**. That accident is the most useful thing in the
table, twice over.

**It measures the noise floor.** The gap between two identical configurations is
not an effect, it is error: **11.8%** at 7 repeats. Every other number has to
clear it, and the disjointness result does not.

**It validates the separability test itself.** At **3 repeats** those two
identical arms came out *non-overlapping* — 48.9–67.0 against 81.9–92.5. So
non-overlap did **not** imply a real effect: the test scored a false positive on
the one chance it was given, and every `separable` flag in that run was unsound.
Two 3-repeat runs duly disagreed on the **sign** of every effect — disjointness
was +8.4% and "separable" in one, −15.2% and not in the next.

So the module computes `separability_test_valid` and **refuses to issue a verdict
when the calibration arms fail their own overlap check**. At 3 repeats it reports
NO CONCLUSION; at 7 the arms overlap as they must and the verdicts become
readable. The precondition is checkable *before* the result is read, which is the
whole point — otherwise the 3-repeat run would have shipped whichever sign it
happened to produce, with a confident range test behind it.

### Two bugs found by running it

* **`worker_init_fn` was a closure**, and the docstring justified that choice on
  the grounds that it runs in the worker *process* under spawn. The reasoning was
  right and the conclusion backwards: spawn **pickles** the init function to reach
  the child, and a local function is not picklable. It failed instantly with
  `PicklingError: Can't pickle local object`. It is a module-level class now.
* **The dataloader kwargs were unguarded for `num_workers=0`.** Torch raises on
  `prefetch_factor` there rather than ignoring it. The study never runs with zero
  workers — contention is the whole point — so it only fired under test, which is
  precisely where an unguarded version would have sat unnoticed until someone
  reused the helper.

## Roadmap

| Milestone | Status |
|---|---|
| Instrumented baseline, phase breakdown, unattributed time | done |
| Optimisation ladder, one change per rung, enforced | done |
| Repeats with mean±std + separability analysis | done |
| Generated REPORT.md with generated figures | done |
| Negative + skipped + failed rungs kept in the ledger | done |
| Cost ledger and scaling break-even calculator | done |
| DDP weak-scaling executed at 1/2/4 workers (CPU/gloo) | done |
| Measured comm fraction explaining the efficiency gap | done |
| Committed profiler traces, before/after data-wait | done |
| `docs/SCALING.md` incl. "when is DDP the wrong tool" | done |
| **Same study on real GPUs with NCCL** | impossible here: no GPU |
| Contention-vs-communication ablation, executed | done |
| Strong-scaling counterpart (global batch fixed) | done |
| Comm fraction repeated and reported with its spread | done |
| **`torch.compile` rung** | fails on this box: no MSVC toolchain (in the ledger) |
| Cost-to-target-accuracy: throughput and cost rank the rungs differently | done |
| CPU-pinning experiment, with a calibration arm that validates its own test | done |
| **A quiet dedicated machine to shrink the 11.8% noise floor** | not available here |

## Honesty notes

* **Every measured number in this repo is CPU.** `torch.cuda.is_available()` is
  `False` and no GPU number is quoted anywhere.
* **Two torch versions are represented.** The ladder, the traces and the scaling
  studies were measured on `2.11.0+cpu`; the cost-to-accuracy and CPU-pinning
  studies on `2.13.0+cpu`, because the toolchain moved underneath this repo
  mid-build. Each result file records its own version. Nothing is compared
  *across* that boundary, and the two are not silently pooled into one table.
* **The scaling numbers are CPU/gloo across real OS processes**, not GPUs across
  NVLink. They are executed rather than projected, and the efficiency figures
  (100% / 80% / 59%) belong to this interconnect. The *shape* — efficiency
  falling as the comm fraction grows — is what transfers; the percentages are not
  a prediction about NCCL.
  (An earlier version of this note said there was no scaling number at all. That
  was true when it was written and stopped being true when the study ran, and a
  stale honesty note is worse than none: it is a false claim in the section whose
  whole job is not making them.)
* The `+compile` rung genuinely fails on this machine (`InductorError: Compiler:
  cl is not found`) because Inductor's CPU backend needs MSVC. That is in the
  ledger as a failed row with the error, not omitted.
* **Absolute throughput here moved by more than 2x between sessions.**
  `pinned_split` measured 63.8 samples/s while this laptop was carrying unrelated
  work at 100% CPU and 140.6 on a quiet one. Only within-session comparisons carry
  weight, which is why every arm is re-run inside each invocation rather than
  compared against a stored baseline.
* **`separable_from_control` is thinner than the effect size suggests**: 129.4
  against 129.0 is a margin of 0.4 samples/s. The claim that pinning helps rests
  on the effect being 3.5x the noise floor, not on that hair-width range test.
* **The cost-to-accuracy task is synthetic**, so its convergence shape belongs to
  the generator. What transfers is that the two rankings can disagree and that
  only measuring both catches it — not the crossover point.
* **Stopping resolution differs across batch sizes.** Accuracy is evaluated every
  25 steps, so `samples_to_target` is quantised to 1,600 samples at batch 64 and
  6,400 at batch 256 — up to 18% of that row. It does not explain a 2.9x gap, and
  it is a real bias in the large-batch direction.
* Absolute throughput is meaningless outside this synthetic workload. The rung
  *order* is the transferable result; the rung *values* are not.
