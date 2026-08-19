# DDP weak-scaling study (CPU / gloo)

Produced by `python -m trainlab.scaling --world-sizes 1 2 4 --steps 30`.
Raw data: `results/scaling_cpu_gloo.json`.

## What this is, and what it is not

This machine has **no GPU**. The study therefore runs DistributedDataParallel
over the **gloo** backend across real OS processes on CPU. That is genuinely
distributed data-parallel training — separate processes, real gradient
all-reduce, real synchronisation barriers — and every number below is labelled
CPU/gloo.

**What transfers to a GPU cluster:** the method. Weak scaling with a fixed
per-worker batch, an LR rule declared before the run, warmup excluded, three
non-overlapping measurement windows, and a *measured* communication fraction to
explain the efficiency gap rather than a hand-wave.

**What does not transfer:** the numbers. gloo over loopback has nothing like
NCCL-over-NVLink bandwidth, and CPU workers contend for the very cores the
all-reduce runs on. **A 4-worker CPU efficiency figure says nothing about 4-GPU
efficiency**, and it is not offered as a proxy for one.

## Protocol, fixed before the first run

| decision | value | why |
|---|---|---|
| scaling mode | **weak** — per-worker batch fixed at 32 | Strong scaling (fixed global batch, shrinking per-worker batch) is a *different* experiment with a different answer. Conflating them is the most common error in scaling tables. |
| LR rule | linear in global batch (Goyal et al.) | Declared in `workload.py` before any run, so this is not a hyperparameter search wearing a scaling study's clothes. |
| warmup | 5% of steps, excluded | The first steps pay allocator growth and worker startup. |
| windows | 3 non-overlapping, mean ± std | One window cannot distinguish a result from noise. |
| threads per worker | **1** (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`) | Critical. Without it, the 1-worker baseline already uses every core, and "scaling efficiency" would measure thread contention rather than communication. |

## Results

Hardware: Windows 11, Intel64 Family 6 Model 126 (Ice Lake mobile), 8 logical
CPUs. Backend gloo, device CPU, 30 steps per run.

| world | global batch | lr | samples/s (mean ± std) | speedup | ideal | efficiency | comm fraction |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 0.0100 | 48.1 ± 1.9 | 1.00x | 1.00x | **100%** | 0.0% |
| 2 | 64 | 0.0200 | 69.4 ± 4.5 | 1.44x | 2.00x | **72%** | 1.7% |
| 4 | 128 | 0.0400 | 97.3 ± 12.7 | 2.02x | 4.00x | **51%** | 16.7% |

## Where did the missing 49% go?

At 4 workers the efficiency is 51%, so roughly half the added compute produced
no throughput. The honest accounting:

**1. Communication: 16.7% (measured).** The comm fraction is measured directly,
not inferred — a backward pass with gradient sync is timed against the identical
backward inside `model.no_sync()`, which skips the all-reduce entirely. The
difference is non-overlapped communication. It rises 0% → 1.7% → 16.7%, which is
the expected shape: gloo's all-reduce cost grows with participant count while the
per-worker gradient volume stays fixed.

**2. Core contention: the rest, and it is the larger term.** This is the part a
GPU study would not have. Each worker is pinned to one intra-op thread, but each
worker also runs its own dataloader decode *in-process* (`num_workers=0` inside
the DDP run), and gloo's collective threads run on the same 8 logical cores. At
world size 4 the machine is oversubscribed: 4 training processes + 4 decode
workloads + collective traffic on 8 logical (4 physical) cores.

**3. Variance grows with world size** — ±1.9 at one worker, ±12.7 at four. With
three windows that spread is wide enough that the 51% figure should be read as
"roughly half", not as 51.0%.

**The measurement I would run next** to separate (1) from (2): re-run at world
size 4 with `decode_cost=0`, removing the CPU-side data work entirely. If
efficiency jumps, contention dominates; if it stays near 51%, communication does.
That is one command and it is the honest next step rather than a conclusion
asserted here.

## When is DDP the wrong tool?

* **The model does not fit on one device** → DDP replicates the full model per
  worker, so it cannot help. FSDP / tensor parallelism / pipeline parallelism.
* **The model is tiny** → exactly this study. Communication is a fixed cost per
  step against a small compute budget, so the comm fraction climbs fast. At
  `SmallCNN` size, scaling past 2 workers on this hardware is already a poor
  trade.
* **The bottleneck is the input pipeline** → adding workers multiplies the data
  problem instead of solving it. The optimisation ladder exists to check this
  first, and it is why the ladder runs before the scaling study rather than after.
* **Batch size is already at the limit of what the LR schedule tolerates** →
  weak scaling grows the global batch, and past some point the linear LR rule
  destabilises regardless of hardware.

## Cost

`python -m trainlab.cost --ledger results/ladder_cpu.json --rate 0.35` converts
any ledger into a cost-to-target table and a break-even for scaling. The direct
compute cost of **this** study was **$0** — it ran on a local laptop — so the
rate is a required input rather than a hard-coded constant. On rented hardware
the 4-worker row above would cost 4x the 1-worker row for 2.02x the throughput,
which at these efficiencies is a bad trade unless wall-clock is worth more than
2x the money.
