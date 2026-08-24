"""DDP weak-scaling study, actually executed.

    python -m trainlab.scaling --world-sizes 1 2 4 --steps 40

This machine has no GPU, so the study runs **DDP over the gloo backend across
real OS processes on CPU**. That is a genuine distributed data-parallel run --
separate processes, real gradient all-reduce, real synchronisation -- and it is
labelled as CPU/gloo everywhere rather than dressed up as a GPU result.

**What transfers from a CPU/gloo scaling curve and what does not:**

* Transfers: the *method* (weak scaling, fixed per-worker batch, pre-declared LR
  rule, N independent measurement windows), and the shape of the story -- that
  efficiency falls as communication's share of step time rises.
* Does NOT transfer: the numbers. gloo over loopback has completely different
  bandwidth and latency from NCCL over NVLink, and CPU workers contend for the
  same cores that the all-reduce runs on. A 4-worker CPU efficiency figure says
  nothing about 4-GPU efficiency.

The protocol is fixed BEFORE any run, so the result cannot be a hyperparameter
search wearing a scaling study's clothes:

* **Weak scaling**: per-worker batch is held constant, so the global batch grows
  with world size. Strong scaling (fixed global batch) is a different experiment
  with a different answer, and conflating them is the most common error here.
* **LR rule**: linear in the global batch (Goyal et al.), declared in workload.py.
* **Warmup**: 5% of steps, excluded from measurement.
* **Windows**: 3 non-overlapping windows after warmup, reported mean +/- std.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")


def _worker_main():
    """Runs inside each spawned process."""
    import statistics

    import torch
    import torch.distributed as dist
    import torch.nn as nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from .workload import SmallCNN, SyntheticImages, scaled_lr, warmup_steps

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    steps = int(os.environ["STEPS"])
    batch = int(os.environ["PER_WORKER_BATCH"])
    decode_cost = int(os.environ["DECODE_COST"])
    dataset_n = int(os.environ["DATASET_N"])
    base_lr = float(os.environ["BASE_LR"])

    # gloo is the CPU collective backend; nccl is the GPU one. The study ran on
    # gloo for a long time because this machine had no GPU -- the backend is a
    # parameter, not an assumption, so the same code produces both curves.
    backend = os.environ.get("BACKEND", "gloo")
    use_cuda = backend == "nccl"
    if use_cuda:
        torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world)

    # Each worker gets ONE intra-op thread. Without this, torch spreads a single
    # worker across all cores and a 4-worker run would be compared against a
    # 1-worker run that was already using the whole machine -- which would make
    # "scaling efficiency" measure thread contention rather than communication.
    torch.set_num_threads(1)
    torch.manual_seed(0)

    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    ds = SyntheticImages(n=dataset_n, decode_cost=decode_cost)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=0, drop_last=True)

    model = SmallCNN().to(device)
    if world > 1:
        model = DDP(model, device_ids=[rank] if use_cuda else None)

    def _sync():
        # CUDA is asynchronous: without this, perf_counter() below would time
        # kernel *launches* and report them as compute, and the all-reduce would
        # look free because it had not happened yet.
        if use_cuda:
            torch.cuda.synchronize()

    global_batch = batch * world
    # The LR rule scales against a FIXED reference batch, not the per-worker
    # batch. Using the per-worker batch as the reference is correct by accident
    # under weak scaling (where they coincide) and WRONG under strong scaling: it
    # would scale the LR with world size while the global batch is held constant,
    # which is a hyperparameter change masquerading as a scaling result. Caught
    # by the strong-scaling run printing lr 0.01 -> 0.02 -> 0.04 at a fixed
    # global batch of 128.
    ref_batch = int(os.environ.get("REF_BATCH", batch))
    lr = scaled_lr(base_lr, ref_batch, global_batch)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    warm = warmup_steps(steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    loss_fn = nn.CrossEntropyLoss()

    n_warmup = 5
    per_window = max((steps) // 3, 1)
    windows, seen, t0 = [], 0, None

    it = iter(loader)
    for step in range(steps + n_warmup):
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

        if step == n_warmup - 1:
            _sync()
            if world > 1:
                dist.barrier()      # start every worker's clock together
            t0, seen = time.perf_counter(), 0
        elif step >= n_warmup:
            seen += x.shape[0]
            if (step - n_warmup + 1) % per_window == 0:
                _sync()
                elapsed = time.perf_counter() - t0
                windows.append(seen / elapsed)     # per-worker samples/sec
                t0, seen = time.perf_counter(), 0

    # Communication fraction: time a backward WITH gradient sync against the same
    # backward inside no_sync(), which skips the all-reduce. The difference is
    # the non-overlapped communication cost, and it is what explains a scaling
    # gap instead of hand-waving at one.
    comm_fraction = 0.0
    if world > 1:
        import torch as _t

        x = _t.randn(batch, 3, 32, 32, device=device)
        y = _t.randint(0, 10, (batch,), device=device)

        def _time(sync: bool, reps: int = 8) -> float:
            _sync()
            t = time.perf_counter()
            for _ in range(reps):
                if sync:
                    loss_fn(model(x), y).backward()
                else:
                    with model.no_sync():
                        loss_fn(model(x), y).backward()
            _sync()
            return time.perf_counter() - t

        # REPEATED and reported as a median. A single synced-vs-no_sync
        # comparison is extremely noisy on a contended CPU: consecutive runs of
        # the same configuration produced 15.2% and 1.2%, which is not a
        # measurement, it is a coin flip. The median of several paired trials is
        # stable enough to reason about; the spread is reported alongside it so a
        # reader can see how much to trust it.
        _time(True, 3)          # warm the allocator and the collective
        fractions = []
        for _ in range(7):
            with_sync, without_sync = _time(True), _time(False)
            if with_sync > 0:
                fractions.append(max(with_sync - without_sync, 0.0) / with_sync)
        fractions.sort()
        comm_fraction = statistics.median(fractions) if fractions else 0.0
        comm_fraction_spread = (fractions[-1] - fractions[0]) if len(fractions) > 1 else 0.0

    result = {
        "rank": rank,
        "world_size": world,
        "per_worker_samples_per_s": windows,
        "comm_fraction": comm_fraction,
        "comm_fraction_spread": locals().get("comm_fraction_spread", 0.0),
        "lr": lr,
        "global_batch": global_batch,
    }
    out_dir = os.environ["OUT_DIR"]
    with open(os.path.join(out_dir, "rank-%d.json" % rank), "w") as fh:
        json.dump(result, fh)

    if world > 1:
        dist.destroy_process_group()


def run_world_size(world: int, steps: int, batch: int, decode_cost: int, dataset_n: int,
                   base_lr: float, port: int, ref_batch: int = None,
                   backend: str = "gloo") -> dict:
    out_dir = os.path.join(RESULTS, "scaling_ws%d_%s" % (world, backend))
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, f))

    env = dict(os.environ)
    env.update({
        "WORLD_SIZE": str(world), "STEPS": str(steps), "PER_WORKER_BATCH": str(batch),
        "DECODE_COST": str(decode_cost), "DATASET_N": str(dataset_n), "BASE_LR": str(base_lr),
        "REF_BATCH": str(ref_batch or batch),
        "OUT_DIR": out_dir, "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
        "TRAINLAB_DDP_WORKER": "1",
        "OMP_NUM_THREADS": "1",
        "BACKEND": backend,
    })

    procs = []
    t0 = time.perf_counter()
    for rank in range(world):
        e = dict(env)
        e["RANK"] = str(rank)
        procs.append(subprocess.Popen([sys.executable, "-m", "trainlab.scaling"], env=e,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    outs = [p.communicate() for p in procs]
    wall = time.perf_counter() - t0

    for (o, e), p in zip(outs, procs):
        if p.returncode != 0:
            raise RuntimeError("rank failed (%d): %s" % (p.returncode, e.decode()[-600:]))

    ranks = []
    for rank in range(world):
        with open(os.path.join(out_dir, "rank-%d.json" % rank)) as fh:
            ranks.append(json.load(fh))

    # Aggregate throughput = sum over workers of their per-worker rate.
    per_window_totals = []
    n_windows = min(len(r["per_worker_samples_per_s"]) for r in ranks)
    for w in range(n_windows):
        per_window_totals.append(sum(r["per_worker_samples_per_s"][w] for r in ranks))

    return {
        "world_size": world,
        "per_worker_batch": batch,
        "global_batch": ranks[0]["global_batch"],
        "lr": ranks[0]["lr"],
        "windows": per_window_totals,
        "aggregate_samples_per_s": statistics.fmean(per_window_totals),
        "std": statistics.stdev(per_window_totals) if len(per_window_totals) > 1 else 0.0,
        "comm_fraction": statistics.fmean([r["comm_fraction"] for r in ranks]),
        "comm_fraction_spread": statistics.fmean([r.get("comm_fraction_spread", 0.0) for r in ranks]),
        "wall_s": wall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world-sizes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32, help="PER WORKER batch (weak scaling)")
    ap.add_argument("--decode-cost", type=int, default=60)
    ap.add_argument("--dataset-n", type=int, default=4000)
    ap.add_argument("--base-lr", type=float, default=0.01)
    ap.add_argument("--port", type=int, default=29517)
    ap.add_argument("--strong", action="store_true",
                    help="strong scaling: hold the GLOBAL batch fixed and shrink the per-worker batch")
    ap.add_argument("--tag", default=None, help="suffix for the results filename")
    ap.add_argument("--backend", default="gloo", choices=["gloo", "nccl"],
                    help="gloo runs the workers on CPU; nccl puts rank i on cuda:i")
    args = ap.parse_args()

    rows = []
    for i, world in enumerate(args.world_sizes):
        # Strong scaling holds the GLOBAL batch fixed, so each worker gets a
        # SMALLER slice as the world grows. That is a different experiment from
        # weak scaling and answers a different question: weak scaling asks "can I
        # train on more data in the same time", strong scaling asks "can I train
        # the same job faster". They have different answers and conflating them
        # is the most common error in a scaling table.
        batch = max(args.batch // world, 1) if args.strong else args.batch
        print("running world_size=%d (per-worker batch %d) ..." % (world, batch))
        # Reference batch for the LR rule: the global batch at world size 1.
        ref_batch = args.batch
        row = run_world_size(world, args.steps, batch, args.decode_cost, args.dataset_n,
                             args.base_lr, args.port + i, ref_batch=ref_batch, backend=args.backend)
        rows.append(row)
        print("  aggregate %.1f +/- %.1f samples/s   comm_fraction=%.3f (spread %.3f)"
              % (row["aggregate_samples_per_s"], row["std"], row["comm_fraction"],
                 row.get("comm_fraction_spread", 0.0)))

    base = rows[0]["aggregate_samples_per_s"]
    base_world = rows[0]["world_size"]
    for row in rows:
        ideal = base * (row["world_size"] / base_world)
        row["speedup"] = row["aggregate_samples_per_s"] / base
        row["ideal_speedup"] = row["world_size"] / base_world
        row["efficiency"] = row["speedup"] / row["ideal_speedup"]

    def _gpu_info():
        import torch as _t
        if args.backend != "nccl":
            return (None, 0, _t.__version__)
        return (_t.cuda.get_device_name(0), _t.cuda.device_count(), _t.__version__)

    _gpu = _gpu_info()
    report = {
        "scaling": ("strong (global batch fixed, per-worker batch shrinks)" if args.strong
                    else "weak (per-worker batch fixed, global batch grows)"),
        "decode_cost": args.decode_cost,
        "backend": args.backend,
        "device": "cuda" if args.backend == "nccl" else "cpu",
        "hardware": {"platform": platform.platform(),
                     "processor": platform.processor() or platform.machine(),
                     "cpu_count": os.cpu_count(),
                     "accelerator": _gpu[0], "gpu_count": _gpu[1], "torch": _gpu[2]},
        "steps_per_run": args.steps,
        "threads_per_worker": 1,
        "rows": rows,
    }
    os.makedirs(RESULTS, exist_ok=True)
    suffix = args.tag or ("strong" if args.strong else "weak")
    stem = "cpu_gloo" if args.backend == "gloo" else "gpu_nccl"
    path = os.path.join(RESULTS, "scaling_%s_%s.json" % (stem, suffix))
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    print()
    print("| world | global batch | lr | samples/s (mean ± std) | speedup | ideal | efficiency | comm fraction |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| %d | %d | %.4f | %.1f ± %.1f | %.2fx | %.2fx | %.0f%% | %.1f%% ± %.1f |"
              % (r["world_size"], r["global_batch"], r["lr"], r["aggregate_samples_per_s"],
                 r["std"], r["speedup"], r["ideal_speedup"], 100 * r["efficiency"],
                 100 * r["comm_fraction"], 100 * r.get("comm_fraction_spread", 0.0)))
    print("\nwrote", path)


if __name__ == "__main__":
    if os.environ.get("TRAINLAB_DDP_WORKER") == "1":
        _worker_main()
    else:
        main()
