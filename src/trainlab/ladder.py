"""The optimisation ladder: one change per run, measured, in profiled order.

    python -m trainlab.ladder --steps 60            # run the ladder
    python -m trainlab.ladder --rung baseline       # run a single rung

Rules enforced by the code, not by discipline:
  * every rung inherits the previous rung's config and changes exactly ONE thing
    (`_assert_single_change` fails the run otherwise)
  * negative results stay in the ledger -- a rung that loses is a finding
  * every row records the device, timer kind, and torch version, so two rows
    measured on different hardware can never be silently compared
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .profiling import StepTimer, bottleneck
from .workload import SmallCNN, SyntheticImages, scaled_lr, warmup_steps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")


@dataclass(frozen=True)
class RunConfig:
    name: str
    batch_size: int = 64
    lr: float = 0.01
    num_workers: int = 0
    persistent_workers: bool = False
    prefetch_factor: int = 2
    pin_memory: bool = False
    amp: bool = False
    amp_dtype: str = "bf16"
    channels_last: bool = False
    tf32: bool = False
    compile: bool = False
    note: str = ""


# The ladder. Order is set by the profile, not by a blog post: the baseline
# breakdown shows data-wait dominating, so dataloader rungs come first. Kernel-
# level rungs (tf32/amp/channels_last/compile) are pointless while the device is
# idle waiting for input.
LADDER = [
    RunConfig("baseline", note="instrumented FP32 reference, single-process loading"),
    RunConfig("workers", num_workers=4, note="attack the profiled data-wait stall"),
    RunConfig("workers+persistent", num_workers=4, persistent_workers=True,
              note="stop paying worker startup every epoch"),
    RunConfig("+prefetch", num_workers=4, persistent_workers=True, prefetch_factor=4,
              note="deeper queue to absorb jitter in per-sample cost"),
    RunConfig("+pin_memory", num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True,
              note="CUDA only: pinned staging buffers enable async H2D copies"),
    RunConfig("+tf32", num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True, tf32=True,
              note="CUDA only: TF32 matmul/conv on Ampere+"),
    RunConfig("+amp", num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True, tf32=True,
              amp=True, note="mixed precision; bf16 chosen over fp16, see REPORT"),
    RunConfig("+channels_last", num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True,
              tf32=True, amp=True, channels_last=True, note="NHWC layout for conv kernels"),
    RunConfig("+batch128", num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True,
              tf32=True, amp=True, channels_last=True, batch_size=128, lr=scaled_lr(0.01, 64, 128),
              note="batch doubled WITH the pre-declared linear LR rule + warmup"),
    RunConfig("+compile", num_workers=4, persistent_workers=True, prefetch_factor=4, pin_memory=True,
              tf32=True, amp=True, channels_last=True, batch_size=128, lr=scaled_lr(0.01, 64, 128),
              compile=True, note="torch.compile; warmup cost reported separately from steady state"),
]

TUNABLE = [f.name for f in dataclasses.fields(RunConfig) if f.name not in ("name", "note")]


def _assert_single_change(prev: RunConfig, cur: RunConfig):
    """A ladder rung that changes two things measures nothing."""
    changed = [f for f in TUNABLE if getattr(prev, f) != getattr(cur, f)]
    # batch_size and lr move together by design -- the LR rule is a consequence
    # of the batch change, not an independent knob.
    if set(changed) == {"batch_size", "lr"}:
        return
    if len(changed) > 1:
        raise ValueError("rung %r changes %d things (%s); one change per rung" % (cur.name, len(changed), changed))


def run_one(cfg: RunConfig, steps: int, device: torch.device, dataset_n: int = 20_000, decode_cost: int = 400) -> dict:
    torch.manual_seed(0)
    if cfg.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ds = SyntheticImages(n=dataset_n, decode_cost=decode_cost)
    loader_kwargs = dict(batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    if cfg.num_workers > 0:
        loader_kwargs.update(persistent_workers=cfg.persistent_workers, prefetch_factor=cfg.prefetch_factor)
    if device.type == "cuda":
        loader_kwargs.update(pin_memory=cfg.pin_memory)
    loader = DataLoader(ds, **loader_kwargs)

    model = SmallCNN().to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.compile:
        model = torch.compile(model)

    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    warm = warmup_steps(steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    loss_fn = nn.CrossEntropyLoss()
    amp_dtype = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and amp_dtype is torch.float16)

    timer = StepTimer(device)
    # Warmup steps are excluded from the measurement: the first steps pay cudnn
    # autotuning, allocator growth and (with compile) graph capture. Including
    # them is the most common way a speedup table gets quietly wrong.
    n_warmup = 5 if not cfg.compile else 8
    seen, done, t_start, warm_wall = 0, 0, None, None

    it = iter(loader)
    t_iter = time.perf_counter()
    while done < steps + n_warmup:
        with timer.phase("data"):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x = x.to(device, non_blocking=cfg.pin_memory)
            y = y.to(device, non_blocking=cfg.pin_memory)
            if cfg.channels_last:
                x = x.to(memory_format=torch.channels_last)

        with timer.phase("forward"):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp):
                out = model(x)
                loss = loss_fn(out, y)

        with timer.phase("backward"):
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        with timer.phase("optimizer"):
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            sched.step()

        done += 1
        if done == n_warmup:
            # Start the clock only after warmup, and reset the phase totals too.
            timer.flush()
            timer.reset()
            warm_wall = time.perf_counter() - t_iter
            t_start = time.perf_counter()
        elif done > n_warmup:
            timer.step_done()
            seen += x.shape[0]

    wall = time.perf_counter() - t_start
    summary = timer.summary(seen, wall)
    summary.update(
        {
            "run": cfg.name,
            "config": dataclasses.asdict(cfg),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or platform.machine(),
            "torch": torch.__version__,
            "warmup_steps_excluded": n_warmup,
            "warmup_wall_s": warm_wall,
            "final_loss": float(loss.detach()),
        }
    )
    summary["bottleneck"] = bottleneck(summary)
    return summary


def _aggregate(reps: list) -> dict:
    """Collapse repeated runs of one rung into mean/std. Keeps every raw value."""
    import statistics

    base = dict(reps[0])
    for key in ("samples_per_s", "ms_per_step", "data_pct", "forward_pct", "backward_pct",
                "optimizer_pct", "unattributed_pct", "wall_s"):
        vals = [r[key] for r in reps]
        base[key] = statistics.fmean(vals)
        base[key + "_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
    base["repeats"] = len(reps)
    base["samples_per_s_raw"] = [r["samples_per_s"] for r in reps]
    return base


def run_ladder(steps: int, device: torch.device, rungs=None, dataset_n: int = 20_000, decode_cost: int = 400,
               repeats: int = 3) -> dict:
    selected = [r for r in LADDER if rungs is None or r.name in rungs]
    rows, prev = [], None
    for cfg in selected:
        if prev is not None:
            _assert_single_change(prev, cfg)
        # Rungs that are no-ops on this device are skipped and SAID to be skipped,
        # rather than run and reported as a 1.00x "result" that means nothing.
        if device.type != "cuda" and cfg.name in ("+pin_memory", "+tf32"):
            rows.append({"run": cfg.name, "skipped": True,
                         "reason": "CUDA-only rung; no effect on %s" % device.type,
                         "config": dataclasses.asdict(cfg)})
            prev = cfg
            continue
        print("running rung: %s" % cfg.name)
        try:
            # Repeat each rung and report mean +/- std. Single-shot ladder tables
            # are how a 10% run-to-run variance gets published as a "10% speedup":
            # measured variance on this workload is large enough that several
            # adjacent rungs are not separable, and the table has to say so.
            reps = [run_one(cfg, steps, device, dataset_n, decode_cost) for _ in range(repeats)]
            row = _aggregate(reps)
        except Exception as exc:
            # A rung that cannot run on this machine is recorded as a failure with
            # the reason, not dropped. A ladder table with a silently missing row
            # is indistinguishable from one where the row was omitted because it
            # looked bad.
            print("  FAILED: %s" % type(exc).__name__)
            rows.append({"run": cfg.name, "failed": True, "error": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                         "config": dataclasses.asdict(cfg)})
            prev = cfg
            continue
        base = next((r for r in rows if r.get("run") == "baseline" and "samples_per_s" in r), None)
        prev_measured = next((r for r in reversed(rows) if "samples_per_s" in r), None)
        if base:
            row["cumulative_speedup"] = row["samples_per_s"] / base["samples_per_s"]
        if prev_measured:
            row["step_speedup"] = row["samples_per_s"] / prev_measured["samples_per_s"]
        rows.append(row)
        print("  %.1f +/- %.1f samples/s  data=%.0f%% fwd=%.0f%% bwd=%.0f%%  cum=%.2fx"
              % (row["samples_per_s"], row.get("samples_per_s_std", 0.0), row["data_pct"],
                 row["forward_pct"], row["backward_pct"], row.get("cumulative_speedup", 1.0)))
        prev = cfg
    return {"rows": rows, "steps_per_run": steps, "repeats": repeats, "device": str(device),
            "decode_cost": decode_cost, "dataset_n": dataset_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--rung", action="append", default=None)
    ap.add_argument("--dataset-n", type=int, default=20_000)
    ap.add_argument("--decode-cost", type=int, default=400)
    ap.add_argument("--repeats", type=int, default=3, help="runs per rung; the table reports mean +/- std")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    result = run_ladder(args.steps, device, args.rung, args.dataset_n, args.decode_cost, args.repeats)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "ladder_%s.json" % device.type)
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
