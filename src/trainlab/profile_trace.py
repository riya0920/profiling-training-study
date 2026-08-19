"""Capture torch.profiler traces for the baseline and the optimised config.

    python -m trainlab.profile_trace

Produces, for each config:
  * `results/traces/<name>.json`  -- a Chrome trace, loadable in chrome://tracing
    or https://ui.perfetto.dev. This is the artifact a reviewer can open.
  * a committed top-operator table, so the finding survives without the viewer.

The point of committing a trace rather than a screenshot: a screenshot is a claim
about a trace, and a trace is the evidence. Anyone can re-open these and check
that the data-wait region really did collapse.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader

from .ladder import LADDER, RunConfig
from .workload import SmallCNN, SyntheticImages

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRACES = os.path.join(ROOT, "results", "traces")


def capture(cfg: RunConfig, steps: int, dataset_n: int, decode_cost: int) -> dict:
    torch.manual_seed(0)
    ds = SyntheticImages(n=dataset_n, decode_cost=decode_cost)
    kwargs = dict(batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=cfg.num_workers)
    if cfg.num_workers > 0:
        kwargs.update(persistent_workers=cfg.persistent_workers, prefetch_factor=cfg.prefetch_factor)
    loader = DataLoader(ds, **kwargs)

    model = SmallCNN()
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(TRACES, exist_ok=True)
    path = os.path.join(TRACES, "%s.json" % cfg.name.replace("+", "plus_"))

    # wait/warmup/active: the first steps are skipped so allocator growth and
    # worker startup do not dominate the trace.
    sched = schedule(wait=2, warmup=2, active=steps, repeat=1)
    it = iter(loader)

    with profile(activities=[ProfilerActivity.CPU], schedule=sched, record_shapes=False,
                 with_stack=False) as prof:
        for _ in range(steps + 4):
            with torch.profiler.record_function("data_wait"):
                try:
                    x, y = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, y = next(it)
                if cfg.channels_last:
                    x = x.to(memory_format=torch.channels_last)
            with torch.profiler.record_function("forward"):
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=cfg.amp):
                    loss = loss_fn(model(x), y)
            with torch.profiler.record_function("backward"):
                opt.zero_grad(set_to_none=True)
                loss.backward()
            with torch.profiler.record_function("optimizer"):
                opt.step()
            prof.step()

    prof.export_chrome_trace(path)

    # Pull the annotated regions back out so the finding is readable without a
    # trace viewer.
    regions = {}
    for evt in prof.key_averages():
        if evt.key in ("data_wait", "forward", "backward", "optimizer"):
            regions[evt.key] = evt.cpu_time_total / 1000.0   # ms
    total = sum(regions.values()) or 1.0

    top = []
    for evt in sorted(prof.key_averages(), key=lambda e: -e.cpu_time_total)[:12]:
        if evt.key in regions:
            continue
        top.append({"op": evt.key, "cpu_ms": round(evt.cpu_time_total / 1000.0, 2),
                    "calls": evt.count})

    return {
        "config": cfg.name,
        "trace": os.path.relpath(path, ROOT).replace("\\", "/"),
        "regions_ms": {k: round(v, 2) for k, v in regions.items()},
        "regions_pct": {k: round(100.0 * v / total, 1) for k, v in regions.items()},
        "top_operators": top[:8],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--dataset-n", type=int, default=3000)
    ap.add_argument("--decode-cost", type=int, default=400)
    args = ap.parse_args()

    by_name = {c.name: c for c in LADDER}
    out = []
    for name in ("baseline", "+prefetch"):
        cfg = by_name[name]
        print("profiling %s ..." % name)
        out.append(capture(cfg, args.steps, args.dataset_n, args.decode_cost))

    path = os.path.join(ROOT, "results", "profile_regions.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)

    print()
    print("| config | data_wait | forward | backward | optimizer | trace |")
    print("|---|---|---|---|---|---|")
    for r in out:
        p = r["regions_pct"]
        print("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | `%s` |"
              % (r["config"], p.get("data_wait", 0), p.get("forward", 0),
                 p.get("backward", 0), p.get("optimizer", 0), r["trace"]))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
