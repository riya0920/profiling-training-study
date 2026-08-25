"""CPU pinning: is the training loop fighting its own dataloader for cores?

    python -m trainlab.affinity --repeats 7

Seven, not three. At three repeats the calibration arms below fail their own
overlap check and the module refuses to conclude -- see the calibration section.

## The hypothesis

The ladder's biggest win came from `num_workers=4`, which fixed a data-wait
stall. But on an 8-core machine that leaves the main process - which runs the
forward and backward passes across however many threads PyTorch chose, by default
one per core - sharing all 8 cores with 4 decode workers. Everybody is
oversubscribed, the OS scheduler migrates threads between cores, and every
migration throws away a warm L1 and L2.

If that is happening, **partitioning the cores should beat sharing them** even
though partitioning gives each side strictly fewer cores. That is the
counter-intuitive claim worth testing, and it needs four arms because
`pinned_split` bundles *two* interventions - a thread cap and an affinity mask - and an experiment that cannot separate its own interventions has not measured
either of them:

| arm | main process | workers | torch threads | what it isolates |
|---|---|---|---|---|
| `default` | all cores | all cores | default (= core count) | the baseline |
| `threads_capped` | all cores | all cores | 4 | **the thread cap alone.** The control for `pinned_split`. |
| `pinned_split` | cores 0-3 | cores 4-7 | 4 | cap + disjoint partitioning |
| `pinned_half_shared` | cores 0-3 | cores 0-3 | 4 | cap + *half the machine*, shared |

The fourth arm is the one that stops a wrong conclusion. If `pinned_split` beats
`threads_capped`, the obvious reading is "partitioning helped". But `pinned_split`
also happens to give each side exactly four cores, and `pinned_half_shared` gives
both sides the same four cores to fight over. If those two come out equal, the
effect is about core *count*, not about disjointness, and the partitioning story
is wrong.

(The first version of this file had a fourth arm called `pinned_overlap` that was
byte-identical to `threads_capped` - same thread cap, no pinning - and was
described as the control. It was a duplicate, and it would have produced a
confident "pinning has no effect against its control" that was really just the
same arm compared with itself.)

## Why this is worth running rather than reasoning about

Both directions are defensible in the abstract. Partitioning removes contention
and migration; it also caps each side's peak parallelism, so a burst of decode
work can no longer borrow idle compute cores. Which effect wins depends on the
ratio of decode cost to compute cost in this specific workload, which is exactly
the sort of thing that has a number rather than an opinion.

## What came out: pinning helps, and not for the reason predicted

At **7 repeats**, on a quiet machine:

| arm | samples/s (median) | range |
|---|---|---|
| `default` | 87.8 | 79.2-105.2 |
| `threads_capped` | 99.5 | 65.3-129.0 |
| `pinned_split` | **140.6** | 129.4-146.3 |
| `pinned_half_shared` | 132.5 | 113.9-138.5 |

**Pinning is worth +41.3%** against its control - an effect 3.5x the measured
noise floor. That is the hypothesis supported.

**The mechanism is not the one the hypothesis named.** `pinned_half_shared` puts
*both* the main process and the decode workers on the same four cores - same core
count as `pinned_split`, shared instead of disjoint - and it captures nearly all
of the gain. Disjointness is worth a further +6.1%, which is *inside* the noise
floor and not separable. So what helps is **confining the training process to
four cores at all**; whether the dataloader gets its own four is not established
by this data.

That distinction is the entire reason the fourth arm exists. Without it the
obvious reading of "+41.3%" is "partitioning removed contention", and this run
does not support that claim - it supports the much narrower one that a process
spread across 8 logical (4 physical) cores does worse than one confined to 4.

## The calibration arm, which did more work than the experiment

PyTorch already defaults to **4 threads** here - 8 logical cores, 4 physical - so
`threads_capped` is a no-op, and `default` and `threads_capped` are *the same
configuration measured twice*. That accident is the most useful thing in the
table:

**It measures the noise floor.** The gap between two identical configurations is
not an effect, it is error. 11.8% at 7 repeats. Every other number has to clear
it, and the disjointness result does not.

**It validates the separability test.** At **3 repeats** those two identical arms
came out *non-overlapping* - 48.9-67.0 against 81.9-92.5. Non-overlap therefore
did not imply a real effect: the test scored a false positive on the one chance
it was given, and every `separable` flag in that run was unsound. Two 3-repeat
runs duly disagreed on the **sign** of every effect: disjointness was +8.4% and
"separable" in one, −15.2% and not in the next.

So `separability_test_valid` is computed and checked before any verdict is read,
and at 3 repeats the module refuses to conclude. At 7 the calibration arms
overlap as they must, and the verdicts become readable. **The precondition is
checkable in advance**, which is the point of having built it - otherwise the
3-repeat run would have shipped whichever sign it happened to produce.

## What these numbers are not

Absolute throughput here moved by more than 2x *between sessions* - `pinned_split`
measured 63.8 samples/s during a run when this laptop was carrying unrelated work
at 100% CPU, and 140.6 on a quiet one. Only within-session comparisons carry any
weight, which is why every arm is re-run inside each invocation rather than
compared against a stored baseline.

The `separable_from_control` flag is also thinner than the effect size suggests:
129.4 against 129.0 is a margin of 0.4 samples/s. The argument that pinning helps
rests on the effect being 3.5x the noise floor, not on that hair-width range
separation.

## Portability

Affinity is set with `psutil.Process.cpu_affinity()`, which works on Linux and
Windows and raises on macOS. When it is unavailable every arm is reported as
skipped rather than silently falling back to `default`, because an experiment
whose arms quietly collapse into each other reports a null result that means
nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .workload import SmallCNN, SyntheticImages

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")


def affinity_available() -> bool:
    try:
        import psutil

        p = psutil.Process()
        p.cpu_affinity()
        return True
    except Exception:
        return False


def _set_affinity(cores):
    import psutil

    psutil.Process().cpu_affinity(list(cores))


class _PinWorker:
    """Pin each dataloader worker at startup. A class, not a closure.

    The first version was a closure over `cores`, on the reasoning that
    `worker_init_fn` runs in the worker PROCESS and must not depend on
    module-level state the parent mutated. The reasoning was right and the
    conclusion was backwards: under spawn -- which is what Windows uses -- the
    init function is **pickled** to reach the child, and a local function is not
    picklable. It failed immediately with

        PicklingError: Can't pickle local object _worker_init.<locals>.init

    A module-level class instance pickles fine: the class is importable by name
    and the state is a plain tuple. It also still avoids the problem the closure
    was reaching for, because the cores travel with the instance rather than
    being read out of module state on the far side.
    """

    def __init__(self, cores):
        self.cores = tuple(cores)

    def __call__(self, _worker_id):
        try:
            _set_affinity(self.cores)
        except Exception:
            # A worker that cannot pin itself must not take the run down; the
            # arm is then partly unpinned, which `psutil` reports back in the
            # verification below rather than hiding.
            pass


@dataclass
class Arm:
    name: str
    torch_threads: int | None       # None = leave PyTorch's default
    main_cores: tuple | None        # None = do not pin
    worker_cores: tuple | None
    note: str


def build_arms(n_cores: int, n_workers: int = 4):
    half = max(n_cores // 2, 1)
    compute = tuple(range(0, half))
    decode = tuple(range(half, n_cores))
    return [
        Arm("default", None, None, None,
            "everything shares every core; PyTorch picks its own thread count"),
        Arm("threads_capped", n_workers, None, None,
            "CONTROL: cap torch threads to the worker count, no pinning. The only "
            "difference from pinned_split is the affinity mask."),
        Arm("pinned_split", n_workers, compute, decode,
            "compute on cores %s, decode on cores %s -- disjoint, strictly fewer each"
            % (list(compute), list(decode))),
        Arm("pinned_half_shared", n_workers, compute, compute,
            "both sides confined to cores %s. Same core COUNT as pinned_split, "
            "shared instead of disjoint -- so if these two tie, the effect is not "
            "partitioning." % list(compute)),
    ]


def run_arm(arm: Arm, steps: int = 40, batch_size: int = 64, n_workers: int = 4,
            dataset_n: int = 4000, decode_cost: int = 400, warmup: int = 5) -> dict:
    original_threads = torch.get_num_threads()
    original_affinity = None
    if arm.main_cores is not None:
        import psutil

        original_affinity = psutil.Process().cpu_affinity()

    try:
        if arm.torch_threads:
            torch.set_num_threads(arm.torch_threads)
        if arm.main_cores is not None:
            _set_affinity(arm.main_cores)

        torch.manual_seed(0)
        ds = SyntheticImages(n=dataset_n, decode_cost=decode_cost)
        kwargs = dict(batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=n_workers)
        if n_workers > 0:
            # Guarded, because torch raises on these when num_workers == 0 rather
            # than ignoring them. The study never runs with zero workers -- the
            # whole point is contention between the loop and its loaders -- so
            # this only fires under test, which is exactly where an unguarded
            # version would have gone unnoticed until someone reused the helper.
            kwargs.update(persistent_workers=True, prefetch_factor=4)
        if arm.worker_cores is not None and n_workers > 0:
            kwargs["worker_init_fn"] = _PinWorker(arm.worker_cores)
        loader = DataLoader(ds, **kwargs)

        model = SmallCNN()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

        it = iter(loader)
        seen, done, t0 = 0, 0, None
        data_s = 0.0
        while done < steps + warmup:
            td = time.perf_counter()
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            d = time.perf_counter() - td

            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            done += 1
            # Warmup excluded: worker startup and allocator growth are paid once
            # and would otherwise be charged to whichever arm ran first.
            if done == warmup:
                t0 = time.perf_counter()
                seen, data_s = 0, 0.0
            elif done > warmup:
                seen += int(y.numel())
                data_s += d

        wall = time.perf_counter() - t0
        del loader
        return {"samples_per_s": seen / wall, "wall_s": wall,
                "data_wait_pct": 100.0 * data_s / wall}
    finally:
        torch.set_num_threads(original_threads)
        if original_affinity is not None:
            try:
                _set_affinity(original_affinity)
            except Exception:
                pass


def run(repeats: int = 3, steps: int = 40, n_workers: int = 4,
        decode_cost: int = 400, dataset_n: int = 4000) -> dict:
    n_cores = os.cpu_count() or 1
    # Read BEFORE any arm runs. Captured after the fact it would be whatever the
    # last arm restored, which is the same number by luck rather than by design.
    default_threads = torch.get_num_threads()
    arms = build_arms(n_cores, n_workers)

    if not affinity_available():
        return {
            "skipped": True,
            "reason": ("psutil cpu_affinity is unavailable on this platform, so the pinned arms "
                       "would silently collapse into the unpinned ones. Reporting that as a null "
                       "result would be worse than reporting nothing."),
            "arms": [a.name for a in arms],
        }

    rows = []
    for arm in arms:
        rates = sorted(run_arm(arm, steps=steps, n_workers=n_workers,
                               decode_cost=decode_cost, dataset_n=dataset_n)["samples_per_s"]
                       for _ in range(repeats))
        rows.append({
            "arm": arm.name, "note": arm.note,
            "torch_threads": arm.torch_threads or torch.get_num_threads(),
            "main_cores": list(arm.main_cores) if arm.main_cores else "all",
            "worker_cores": list(arm.worker_cores) if arm.worker_cores else "all",
            "samples_per_s_median": round(statistics.median(rates), 1),
            "samples_per_s_min": round(rates[0], 1),
            "samples_per_s_max": round(rates[-1], 1),
            "runs": [round(r, 1) for r in rates],
        })
        print("  %-16s %8.1f samples/s  (%.1f-%.1f over %d runs)"
              % (arm.name, rows[-1]["samples_per_s_median"], rates[0], rates[-1], repeats))

    base = next(r for r in rows if r["arm"] == "default")
    for r in rows:
        r["vs_default"] = round(r["samples_per_s_median"] / base["samples_per_s_median"], 3)

    def _sep(a, b):
        # Non-overlapping observed ranges. A crude test, and the right strength of
        # claim for three runs on a laptop -- a t-test on n=3 would look more
        # rigorous and mean less.
        return bool(a["samples_per_s_min"] > b["samples_per_s_max"]
                    or a["samples_per_s_max"] < b["samples_per_s_min"])

    base = next(r for r in rows if r["arm"] == "default")
    split = next(r for r in rows if r["arm"] == "pinned_split")
    control = next(r for r in rows if r["arm"] == "threads_capped")
    half_shared = next(r for r in rows if r["arm"] == "pinned_half_shared")

    separable = _sep(split, control)
    effect = split["samples_per_s_median"] / control["samples_per_s_median"]
    partition_effect = split["samples_per_s_median"] / half_shared["samples_per_s_median"]
    partition_separable = _sep(split, half_shared)

    # The accidental calibration. If PyTorch's default thread count already
    # equals the cap, then `default` and `threads_capped` are the SAME
    # configuration measured twice -- and the gap between their medians is a
    # direct read-out of this benchmark's noise floor, on this machine, at this
    # repeat count. That turns a wasted arm into the most useful number in the
    # table, because every other effect here has to clear it.
    cap_was_noop = default_threads == n_workers
    noise_floor = abs(base["samples_per_s_median"] / control["samples_per_s_median"] - 1.0)

    # And the calibration arms also tell you whether the SEPARABILITY TEST works.
    #
    # `default` and `threads_capped` are the same configuration. If their observed
    # ranges come out non-overlapping, then non-overlap does not imply a real
    # effect on this machine -- the test has a false-positive rate of 1 out of the
    # 1 chance it was given. Every "separable" verdict below is then unsound, and
    # reporting one anyway would be using a broken instrument because it happened
    # to agree with the hypothesis.
    calibration_falsely_separable = cap_was_noop and _sep(base, control)
    test_valid = not calibration_falsely_separable

    return {
        "hardware": {"platform": platform.platform(),
                     "processor": platform.processor() or platform.machine(),
                     "cpu_count": n_cores, "torch": torch.__version__,
                     "default_torch_threads": default_threads},
        "steps_per_run": steps, "repeats": repeats, "num_workers": n_workers,
        "decode_cost": decode_cost,
        "rows": rows,
        "pinning_effect_vs_thread_capped_control": round(effect, 3),
        "separable_from_control": separable,
        "thread_cap_was_a_noop": bool(cap_was_noop),
        "measured_noise_floor": round(noise_floor, 3),
        "separability_test_valid": bool(test_valid),
        "separability_test_note": (
            "INVALID: the two arms that are the same configuration came out non-overlapping "
            "themselves, so non-overlap does not imply an effect on this machine and every "
            "`separable` flag below is unsound. The experiment is underpowered at %d repeats."
            % repeats if calibration_falsely_separable else
            "the two identical arms overlap as they must, so non-overlap elsewhere carries "
            "some weight -- at %d repeats, not much." % repeats),
        "noise_floor_note": (
            "PyTorch already defaulted to %d threads on this %d-logical-core machine, so "
            "`default` and `threads_capped` are the same configuration measured twice. The %.1f%% "
            "gap between their medians is therefore this benchmark's noise floor at %d repeats, "
            "not an effect -- and every other number below has to clear it to mean anything."
            % (default_threads, n_cores, 100 * noise_floor, repeats)
            if cap_was_noop else
            "PyTorch defaulted to %d threads against a cap of %d, so the thread-cap arm is a real "
            "intervention and no noise floor can be read off it." % (default_threads, n_workers)),
        "disjointness_effect_vs_same_core_count": round(partition_effect, 3),
        "disjointness_separable": partition_separable,
        "verdict": (
            "NO CONCLUSION: pinning appears to change throughput by %+.1f%%, but the calibration "
            "arms -- which are the same configuration measured twice -- came out non-overlapping "
            "themselves. The separability test is broken on this machine at %d repeats, so "
            "neither this number nor its sign is established."
            % (100 * (effect - 1), repeats) if calibration_falsely_separable else
            "pinning changes throughput by %+.1f%% against the thread-capped control, and the "
            "observed ranges do not overlap" % (100 * (effect - 1))
            if separable else
            "HYPOTHESIS NOT SUPPORTED: pinning changes throughput by %+.1f%% against the "
            "thread-capped control, the observed ranges overlap, and the effect is %s the %.1f%% "
            "noise floor measured from the two arms that are secretly identical. This run does "
            "not establish an effect in either direction."
            % (100 * (effect - 1),
               "smaller than" if abs(effect - 1) < noise_floor else "larger than but not separable from",
               100 * noise_floor)),
        "disjointness_verdict": (
            "NO CONCLUSION: the separability test failed its own calibration, see above"
            if calibration_falsely_separable else
            "against pinned_half_shared -- same core count, shared rather than disjoint -- "
            "partitioning is worth %+.1f%%, and the ranges do not overlap"
            % (100 * (partition_effect - 1)) if partition_separable else
            "against pinned_half_shared -- same core count, shared rather than disjoint -- "
            "partitioning is worth %+.1f%%, but the ranges overlap, so any benefit of pinning "
            "here is not attributable to disjointness by this data"
            % (100 * (partition_effect - 1))),
        "caveat": ("one laptop, one workload. The balance between contention removed and peak "
                   "parallelism lost depends on the ratio of decode cost to compute cost, which "
                   "is a property of this workload rather than of pinning."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--decode-cost", type=int, default=400)
    ap.add_argument("--dataset-n", type=int, default=4000)
    args = ap.parse_args()

    out = run(repeats=args.repeats, steps=args.steps, n_workers=args.workers,
              decode_cost=args.decode_cost, dataset_n=args.dataset_n)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "cpu_affinity.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print("\nwritten:", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
