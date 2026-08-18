# Profiling-Driven Training Optimisation Study

An instrumented baseline, then an optimisation ladder where every rung changes
exactly one thing and the order is chosen by the profile rather than by a blog
post. Negative results stay in the table.

> **Status: ~40% built.** The instrumentation, the ladder, the repeat-and-report
> machinery and the generated REPORT.md are done and **measured on CPU**. The DDP
> harness is written but **has not been run on multiple GPUs**, so this repo
> contains **no scaling-efficiency number**. See [Roadmap](#roadmap).

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

## Roadmap (the remaining ~60%)

| Milestone | Status |
|---|---|
| Instrumented baseline, phase breakdown, unattributed time | done |
| Optimisation ladder, one change per rung, enforced | done |
| Repeats with mean±std + separability analysis | done |
| Generated REPORT.md with generated figures | done |
| Negative + skipped + failed rungs kept in the ledger | done |
| Cost ledger and scaling break-even calculator | done |
| DDP harness (weak scaling, comm-fraction measurement) | **written, never run** |
| **Scaling efficiency at 1/2/4 GPUs on the same GPU model** | not measured |
| **Measured explanation of the scaling gap (comm fraction)** | not measured |
| **`torch.compile` rung** | fails on this box: no MSVC toolchain (recorded in the ledger) |
| **Committed profiler trace + before/after trace screenshots** | not done |
| **Cost-to-target-accuracy table (1-GPU-optimised vs 4-GPU)** | needs the GPU runs |

## Honesty notes

* **Every measured number in this repo is CPU.** `torch` here is `2.11.0+cpu` and
  `torch.cuda.is_available()` is `False`. No GPU number is quoted anywhere.
* **There is no scaling efficiency number.** `ddp.py` is written, documents its
  measurement protocol, and declares weak-vs-strong scaling up front, but it has
  not been executed on multiple devices. An unrun harness produces no result, and
  inventing "87%" here would be precisely the failure this project is about.
* The `+compile` rung genuinely fails on this machine (`InductorError: Compiler:
  cl is not found`) because Inductor's CPU backend needs MSVC. That is in the
  ledger as a failed row with the error, not omitted.
* Absolute throughput is meaningless outside this synthetic workload. The rung
  *order* is the transferable result; the rung *values* are not.
