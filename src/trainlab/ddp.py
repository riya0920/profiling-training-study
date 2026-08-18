"""DDP scaling harness. Scaffolding is here; the scaling numbers are NOT.

    torchrun --nproc_per_node=4 -m trainlab.ddp --steps 200

This module is written and runnable but has **not been run on multiple GPUs**,
because this study was executed on a CPU-only machine. No scaling efficiency
number appears anywhere in this repository. Writing the harness and reporting an
imagined 87% would be exactly the failure mode the project is supposed to
demonstrate resistance to.

What is fixed here BEFORE any run, so the eventual numbers are not a
hyperparameter search wearing a scaling study's clothes:

  * LR scaling rule (linear) and warmup fraction (5%), declared in workload.py
  * measurement windows: 3 non-overlapping windows after warmup, mean +/- std
  * the same global batch per step is NOT held constant -- per-GPU batch is held
    constant, so this measures WEAK scaling. Strong scaling (fixed global batch,
    shrinking per-GPU batch) is a different experiment with a different answer,
    and conflating them is the most common error in scaling tables.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .profiling import StepTimer
from .workload import SmallCNN, SyntheticImages, scaled_lr, warmup_steps


def setup() -> tuple:
    """Initialise from torchrun's environment. nccl on CUDA, gloo elsewhere."""
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return rank, world, local_rank, device


def measure_comm_fraction(model: DDP, device: torch.device, steps: int = 20) -> float:
    """Estimate the fraction of step time spent in gradient all-reduce.

    This is the number that explains a scaling gap. "We got 87% efficiency" is a
    claim; "we got 87% and 12% of step time is all-reduce that DDP could not
    overlap with backward" is a measurement. Without this, the gap is a hand-wave.

    Method: time a step with DDP gradient sync enabled against the same step
    inside `no_sync()`, which skips the all-reduce entirely. The difference is
    the non-overlapped communication cost.
    """
    x = torch.randn(32, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    loss_fn = nn.CrossEntropyLoss()

    def _time(sync: bool) -> float:
        ctx = model.no_sync() if not sync else _null_context()
        t0 = time.perf_counter()
        with ctx:
            for _ in range(steps):
                loss_fn(model(x), y).backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    _time(True)  # warmup
    with_sync = _time(True)
    without_sync = _time(False)
    return max(with_sync - without_sync, 0.0) / with_sync if with_sync else 0.0


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def run(steps: int, batch_size: int, base_lr: float, dataset_n: int, decode_cost: int) -> dict:
    rank, world, local_rank, device = setup()

    ds = SyntheticImages(n=dataset_n, decode_cost=decode_cost)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=4, persistent_workers=True, prefetch_factor=4,
                        pin_memory=device.type == "cuda", drop_last=True)

    model = SmallCNN().to(device)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # Weak scaling: per-GPU batch fixed, so the global batch grows with world size
    # and the LR follows the pre-declared linear rule against the GLOBAL batch.
    global_batch = batch_size * world
    lr = scaled_lr(base_lr, batch_size, global_batch)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    warm = warmup_steps(steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    loss_fn = nn.CrossEntropyLoss()
    timer = StepTimer(device)

    # Three non-overlapping windows, so the reported spread is across independent
    # stretches of the run rather than three reads of the same warm cache.
    windows, per_window = [], max(steps // 3, 1)
    it, seen, t0 = iter(loader), 0, time.perf_counter()
    for step in range(steps + 5):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step == 4:
            t0, seen = time.perf_counter(), 0
        elif step > 4:
            seen += x.shape[0]
            if (step - 4) % per_window == 0:
                elapsed = time.perf_counter() - t0
                windows.append(seen * world / elapsed)
                t0, seen = time.perf_counter(), 0

    comm = measure_comm_fraction(model, device) if world > 1 else 0.0
    result = {
        "world_size": world,
        "per_gpu_batch": batch_size,
        "global_batch": global_batch,
        "lr": lr,
        "scaling": "weak",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(local_rank) if device.type == "cuda" else "cpu",
        "window_samples_per_s": windows,
        "comm_fraction": comm,
        "timer": timer.timer_kind,
    }
    if rank == 0:
        print(json.dumps(result, indent=2))
    if world > 1:
        dist.destroy_process_group()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64, help="PER GPU batch (weak scaling)")
    ap.add_argument("--base-lr", type=float, default=0.01)
    ap.add_argument("--dataset-n", type=int, default=20_000)
    ap.add_argument("--decode-cost", type=int, default=400)
    args = ap.parse_args()
    run(args.steps, args.batch_size, args.base_lr, args.dataset_n, args.decode_cost)


if __name__ == "__main__":
    main()
