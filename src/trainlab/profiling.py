"""Step-time instrumentation.

The rule this whole project is built on: you do not get to optimise anything you
have not measured. So the baseline is instrumented *first*, the breakdown tells
you which rung to climb next, and every rung is one change with a before/after.

On CUDA, timing uses `torch.cuda.Event` and not `time.perf_counter`, because
kernel launches are asynchronous -- wall-clock around a forward pass measures how
fast Python queued the work, not how long the GPU took. On CPU the two coincide
and `perf_counter` is used directly. Which timer produced a number is recorded in
the result, so a CPU-measured table is never mistaken for a GPU one.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import torch


class StepTimer:
    """Accumulates per-phase time across steps: data wait / forward / backward / optimiser."""

    PHASES = ("data", "forward", "backward", "optimizer")

    def __init__(self, device: torch.device):
        self.device = device
        self.cuda = device.type == "cuda"
        self.reset()

    @property
    def timer_kind(self) -> str:
        return "cuda_event" if self.cuda else "perf_counter"

    def reset(self):
        self.totals = {p: 0.0 for p in self.PHASES}
        self.n_steps = 0
        self._pending = []

    @contextmanager
    def phase(self, name: str):
        assert name in self.PHASES, name
        if self.cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            # Deliberately NOT synchronising here: a sync inside the hot loop
            # serialises the pipeline and changes the thing being measured.
            # Events are resolved once per epoch in `flush()`.
            self._pending.append((name, start, end))
        else:
            t0 = time.perf_counter()
            yield
            self.totals[name] += time.perf_counter() - t0

    def flush(self):
        """Resolve pending CUDA events. Call once at epoch end, never per step."""
        if not self.cuda or not self._pending:
            return
        torch.cuda.synchronize()
        for name, start, end in self._pending:
            self.totals[name] += start.elapsed_time(end) / 1000.0
        self._pending = []

    def step_done(self):
        self.n_steps += 1

    def summary(self, n_samples: int, wall_s: float) -> dict:
        self.flush()
        measured = sum(self.totals.values())
        out = {
            "timer": self.timer_kind,
            "steps": self.n_steps,
            "wall_s": wall_s,
            "samples_per_s": n_samples / wall_s if wall_s > 0 else 0.0,
            "ms_per_step": 1000.0 * wall_s / max(self.n_steps, 1),
        }
        for p in self.PHASES:
            out["%s_s" % p] = self.totals[p]
            out["%s_pct" % p] = 100.0 * self.totals[p] / measured if measured else 0.0
        # Wall time not attributed to any phase: python overhead, logging, the
        # gaps between instrumented regions. Reporting it keeps the percentages
        # honest instead of silently renormalising them to 100%.
        out["unattributed_s"] = max(wall_s - measured, 0.0)
        out["unattributed_pct"] = 100.0 * out["unattributed_s"] / wall_s if wall_s else 0.0
        return out


def bottleneck(summary: dict) -> str:
    """Name the phase to attack next. This is what picks the rung order."""
    phases = {p: summary.get("%s_pct" % p, 0.0) for p in StepTimer.PHASES}
    worst = max(phases, key=phases.get)
    if phases["data"] > 20.0:
        # Data wait above ~20% means the accelerator is idle waiting on the input
        # pipeline. No amount of kernel optimisation helps until that is fixed,
        # which is why the ladder always starts here when it shows up.
        return "data"
    return worst
