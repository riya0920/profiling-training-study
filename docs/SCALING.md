# DDP scaling study (CPU / gloo) — and a measurement that refused to cooperate

Produced by `python -m trainlab.scaling`. Raw data in `results/`.

## What this is, and what it is not

This machine has **no GPU**. The study runs DistributedDataParallel over the
**gloo** backend across real OS processes on CPU: separate processes, real
gradient all-reduce, real barriers. Every number is labelled CPU/gloo.

**Transfers to a GPU cluster:** the method — weak *and* strong scaling run
separately, an LR rule fixed before the runs, warmup excluded, three
non-overlapping measurement windows, an ablation to test the explanation, and a
comm-fraction measurement reported with its spread.

**Does not transfer:** the numbers. gloo over loopback is nothing like NCCL over
NVLink, and CPU workers contend for the very cores the all-reduce runs on.

## Protocol, fixed before the first run

| decision | value | why |
|---|---|---|
| scaling modes | **weak and strong, separately** | Different experiments with different answers. Weak asks "more data in the same time"; strong asks "the same job, faster". |
| LR rule | linear against a **fixed reference batch** | See the bug below. |
| warmup | 5% of steps, excluded | The first steps pay allocator growth. |
| windows | 3 non-overlapping, mean ± std | One window cannot separate result from noise. |
| threads per worker | **1** | Otherwise the 1-worker baseline already uses every core and "efficiency" measures thread contention. |

### A bug the strong-scaling run exposed

The LR was being scaled against the **per-worker** batch. Under weak scaling
those coincide, so it was right by accident. Under strong scaling — global batch
held at 128 — it printed `lr 0.01 → 0.02 → 0.04` while the global batch never
changed. That is a hyperparameter change wearing a scaling study's clothes, and
it would have made the strong-scaling curve meaningless.

Fixed: the rule now scales against a fixed reference batch, so strong scaling
holds `lr = 0.01` throughout and weak scaling still scales with world size.
`test_scaling_protocol_is_weak_and_lr_follows_global_batch` pins it.

## Weak scaling (per-worker batch fixed at 32)

| world | global batch | lr | samples/s (mean ± std) | speedup | efficiency | comm fraction |
|---|---|---|---|---|---|---|
| 1 | 32 | 0.0100 | 43.6 ± 1.6 | 1.00x | **100%** | 0.0% |
| 2 | 64 | 0.0200 | 69.8 ± 1.7 | 1.60x | **80%** | 6.3% ± 20.7 |
| 4 | 128 | 0.0400 | 103.7 ± 13.9 | 2.38x | **59%** | 2.3% ± 22.3 |

## Strong scaling (global batch fixed at 128)

| world | per-worker batch | lr | samples/s (mean ± std) | speedup | efficiency |
|---|---|---|---|---|---|
| 1 | 128 | 0.0100 | 51.2 ± 4.0 | 1.00x | **100%** |
| 2 | 64 | 0.0100 | 71.8 ± 8.3 | 1.40x | **70%** |
| 4 | 32 | 0.0100 | 101.0 ± 6.4 | 1.97x | **49%** |

Strong scaling is **worse at every world size**, which is the expected shape: the
per-worker batch shrinks as workers are added, so each worker does less compute
per synchronisation and the fixed per-step overhead is amortised over less work.

## The comm-fraction measurement does not work here, and that is the finding

An earlier version of this document stated *"16.7% is measured communication"*
and attributed the scaling gap to it. **That number was noise, and reporting it
as a measurement was wrong.**

The comm fraction is estimated by timing a backward pass with gradient sync
against the same backward inside `model.no_sync()`. Repeating that paired
comparison seven times per worker and reporting the spread shows why the single
measurement was untrustworthy:

| world | comm fraction (median) | spread across 7 trials |
|---|---|---|
| 2 | 6.3% | **±20.7 points** |
| 4 | 2.3% | **±22.3 points** |

**The spread is three to ten times the median.** Consecutive runs of the same
configuration produced 15.2% and 1.2%. On a contended CPU where the collective
threads share cores with the compute, this estimator is a coin flip, and no
attribution built on it is defensible.

The honest position: **communication cost is unmeasured on this hardware.** The
median is reported with its spread so a reader can see it is unusable, rather
than quoted as a clean number.

## The ablation, which refuted the fallback hypothesis too

With communication unmeasurable, the remaining hypothesis was CPU contention from
the in-process data decode. The ablation is one command: rerun world 4 with
`--decode-cost 0`, removing that work entirely.

| configuration | efficiency at world 4 |
|---|---|
| decode_cost = 60 (normal) | 59% ± 13.9 |
| **decode_cost = 0 (ablation)** | **65% ± 0.9** |

Removing *all* CPU-side data work recovers about 6 percentage points — and the
two figures overlap within the ±13.9 spread of the baseline. **Decode contention
is not the dominant term either.**

So both candidate explanations have now been tested and neither accounts for the
~35–41% loss at four workers. What remains, untested:

* **Core oversubscription.** Four training processes plus gloo's collective
  threads on 8 logical / 4 physical cores. Testing this needs either fewer
  workers than physical cores or explicit CPU pinning.
* **DDP gradient bucketing overhead**, which is per-step and independent of the
  data pipeline.

**Naming an unexplained gap is more useful than assigning it to whichever cause
was measured last.** The next experiment is stated rather than the conclusion.

## When is DDP the wrong tool?

* **The model does not fit on one device** → DDP replicates the full model per
  worker. FSDP / tensor / pipeline parallelism instead.
* **The model is tiny** → exactly this study. Per-step synchronisation is a fixed
  cost against a small compute budget, so efficiency falls fast. At `SmallCNN`
  size, scaling past two workers on this hardware is already a poor trade.
* **The bottleneck is the input pipeline** → adding workers multiplies the data
  problem. The optimisation ladder runs *before* the scaling study for this
  reason.
* **The global batch is already at the limit the LR schedule tolerates** → weak
  scaling grows it further, and past some point the linear rule destabilises
  regardless of hardware.

## Cost

`python -m trainlab.cost --ledger results/ladder_cpu.json --rate 0.35` converts a
ledger into a cost-to-target table and a scaling break-even. This study's direct
compute cost was **$0** — local hardware — so the rate is a required input rather
than a constant. At the measured efficiencies, four rented workers would cost 4x
for 2.4x the throughput: a bad trade unless wall-clock is worth more than 1.7x
the money.
