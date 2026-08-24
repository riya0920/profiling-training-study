"""Cost to a target ACCURACY — the question the throughput ladder cannot answer.

    python -m trainlab.accuracy --target 0.85 --rate 0.35

## Why this exists

Every number in the optimisation ladder is samples per second. That is the right
metric for the question *"is the training loop efficient?"* and the wrong metric
for the question anyone actually pays for, which is *"what does it cost to get a
model this good?"*

The two come apart for one reason: **a change can move throughput and convergence
in opposite directions.** Batch size is the obvious case — doubling it usually
raises samples/s and often raises the number of samples needed to reach a target,
because a doubled batch does not halve the number of updates required. The ladder
sees the first effect and is structurally blind to the second, and so is every
"we got a 2x speedup" claim measured the same way.

    cost to target = samples_to_target / samples_per_second x rate

Both factors. A ladder that optimises only the denominator can make the product
worse, and the only way to know is to measure the numerator too.

## What is measured

Each configuration trains from the same seed until held-out accuracy first
crosses the target, and reports:

  * **samples_to_target** — the convergence term the ladder never sees
  * **samples_per_s** — the throughput term, measured on the same run
  * **hours** and **usd** to the target, at a rate the caller supplies

Then it ranks the configurations by throughput and by cost-to-target separately.
**If those two rankings differ, the ladder's winner is not the cheapest model.**

## The honest limits

* The task is synthetic, so the *shape* of its convergence is a property of this
  generator. What transfers is the method and the fact that the two rankings can
  disagree — not the specific crossover.
* Held-out accuracy is a **noisy stopping rule**. One lucky evaluation can trip
  the threshold early, so the crossing requires the target to be met on two
  consecutive evaluations. Without that, the whole table measures evaluation
  noise.
* Every configuration gets the same seed and the same data order, which removes
  one source of variance and does not remove the rest. Repeats are reported.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .workload import SmallCNN, SyntheticImages, scaled_lr, warmup_steps

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results")


@dataclass
class AccConfig:
    name: str
    batch_size: int = 64
    lr: float = 0.01
    amp: bool = False
    num_workers: int = 4
    note: str = ""


# Deliberately a *small* set, all sharing the dataloader settings the ladder
# already established as optimal. The variable under study is the one that
# plausibly trades throughput against convergence; holding everything else fixed
# is what makes the comparison mean anything.
CONFIGS = [
    AccConfig("batch64", batch_size=64, lr=0.01,
              note="the ladder's converged dataloader settings, fp32"),
    AccConfig("batch128_linear", batch_size=128, lr=scaled_lr(0.01, 64, 128),
              note="batch doubled WITH the pre-declared linear LR rule"),
    AccConfig("batch128_unscaled", batch_size=128, lr=0.01,
              note="the control: batch doubled and LR left alone, which is the mistake"),
    AccConfig("batch256_linear", batch_size=256, lr=scaled_lr(0.01, 64, 256),
              note="pushing the batch far enough that the trade should be visible"),
    AccConfig("batch128_bf16", batch_size=128, lr=scaled_lr(0.01, 64, 128), amp=True,
              note="does reduced precision cost convergence as well as throughput?"),
]


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += int((model(x).argmax(1) == y).sum())
        total += int(y.numel())
    model.train()
    return correct / max(total, 1)


def train_to_target(cfg: AccConfig, target: float, device, train_n: int = 4000,
                    val_n: int = 1000, decode_cost: int = 120, max_steps: int = 900,
                    eval_every: int = 25, seed: int = 0) -> dict:
    """Train until held-out accuracy crosses `target` twice in a row.

    Twice in a row, not once. Held-out accuracy on 1,000 samples has a standard
    error near a percentage point, so a single crossing of an 85% threshold is
    substantially a coin flip — and since the whole table is *when* the crossing
    happened, a noisy rule does not add error to the answer, it becomes the
    answer.

    Evaluation time is excluded from the throughput measurement and reported
    separately. Charging evaluation to training makes a configuration that
    evaluates less often look faster, which is a knob, not a speedup.
    """
    torch.manual_seed(seed)
    train_ds = SyntheticImages(n=train_n, decode_cost=decode_cost, seed=1)
    val_ds = SyntheticImages(n=val_n, decode_cost=decode_cost, seed=99)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        num_workers=cfg.num_workers,
                        **({"persistent_workers": True, "prefetch_factor": 4}
                           if cfg.num_workers > 0 else {}))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = SmallCNN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9)
    warm = warmup_steps(max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm))
    loss_fn = nn.CrossEntropyLoss()

    step, samples, train_s, eval_s = 0, 0, 0.0, 0.0
    hits, crossed_at = 0, None
    curve = []
    it = iter(loader)

    while step < max_steps:
        t0 = time.perf_counter()
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.amp):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        train_s += time.perf_counter() - t0

        step += 1
        samples += int(y.numel())

        if step % eval_every == 0:
            te = time.perf_counter()
            acc = evaluate(model, val_loader, device)
            eval_s += time.perf_counter() - te
            curve.append({"step": step, "samples": samples, "val_acc": acc})
            hits = hits + 1 if acc >= target else 0
            if hits >= 2:
                crossed_at = {"step": step, "samples": samples, "val_acc": acc}
                break

    return {
        "config": asdict(cfg),
        "reached_target": crossed_at is not None,
        "target": target,
        "steps_to_target": crossed_at["step"] if crossed_at else None,
        "samples_to_target": crossed_at["samples"] if crossed_at else None,
        "final_val_acc": curve[-1]["val_acc"] if curve else 0.0,
        "train_s": train_s,
        "eval_s": eval_s,
        "samples_per_s": samples / train_s if train_s else 0.0,
        "steps_run": step,
        "curve": curve,
    }


def _reading(winners_agree, orders_identical, by_throughput, by_cost, biggest,
             cheapest, dearest, by_cost_rows=()) -> str:
    """State the result at the resolution the data supports.

    Three genuinely different outcomes, and only the first is the headline the
    module was written hoping for:

      1. the winners differ -- a throughput ladder ships the wrong config
      2. the winners agree but the ORDER does not -- a throughput ladder ranks a
         costly config right behind the winner, which is how it gets picked the
         moment the winner has some other drawback
      3. the orderings are identical -- throughput was a sufficient proxy here

    Reporting (2) as "the throughput winner is also the cheapest" is technically
    true and hides the finding.
    """
    if not by_cost:
        return "nothing reached the target; there is no cost-to-target to report"
    if not winners_agree:
        return ("the throughput winner (%s) is NOT the cheapest way to reach the target (%s). "
                "A ladder measured in samples/s would have shipped the wrong configuration."
                % (by_throughput[0], by_cost[0]))
    if orders_identical:
        return ("throughput and cost-to-target rank every configuration identically here, so "
                "throughput was a sufficient proxy for this workload -- which is a result about "
                "this workload, not a general licence to skip the convergence term")
    spread = dearest["usd_to_target"] / cheapest["usd_to_target"]
    if biggest is None:
        return ("the winners agree and the orderings differ only in configurations throughput "
                "UNDER-rates, which costs nobody anything. Across the table the dearest "
                "configuration costs %.2fx the cheapest." % spread)
    over = next(r for r in by_cost_rows if r["name"] == biggest["name"])
    return (
        "the winners agree and the ORDERINGS DO NOT. %s is rank %d of %d by throughput and only "
        "rank %d of %d by cost -- a throughput ladder promotes it while it costs %.2fx the "
        "cheapest route to the same accuracy. Across the whole table the dearest configuration "
        "costs %.2fx the cheapest, and none of that spread is visible in samples/s."
        % (biggest["name"], biggest["rank_by_throughput"], len(by_throughput),
           biggest["rank_by_cost"], len(by_cost),
           over["usd_to_target"] / cheapest["usd_to_target"], spread))


def run(target: float = 0.85, rate_usd_per_hour: float = 0.35, repeats: int = 2,
        max_steps: int = 900, train_n: int = 4000, decode_cost: int = 120) -> dict:
    device = torch.device("cpu")
    rows = []
    for cfg in CONFIGS:
        reps = [train_to_target(cfg, target, device, train_n=train_n, decode_cost=decode_cost,
                                max_steps=max_steps, seed=s) for s in range(repeats)]
        reached = [r for r in reps if r["reached_target"]]

        tp = sum(r["samples_per_s"] for r in reps) / len(reps)
        if reached:
            s2t = sum(r["samples_to_target"] for r in reached) / len(reached)
            hours = s2t / tp / 3600.0
            row = {
                "name": cfg.name, "note": cfg.note,
                "reached": len(reached), "of": repeats,
                "samples_per_s": round(tp, 1),
                "samples_to_target": round(s2t, 1),
                "steps_to_target": round(sum(r["steps_to_target"] for r in reached) / len(reached), 1),
                "hours_to_target": hours,
                "usd_to_target": hours * rate_usd_per_hour,
                "final_val_acc": round(max(r["final_val_acc"] for r in reps), 4),
            }
        else:
            # A configuration that never reaches the target has no cost-to-target
            # at all. Reporting its throughput next to the others' costs -- as a
            # throughput-only ladder implicitly does -- is the exact error this
            # module exists to make visible.
            row = {
                "name": cfg.name, "note": cfg.note, "reached": 0, "of": repeats,
                "samples_per_s": round(tp, 1),
                "samples_to_target": None, "steps_to_target": None,
                "hours_to_target": None, "usd_to_target": None,
                "final_val_acc": round(max(r["final_val_acc"] for r in reps), 4),
            }
        rows.append(row)
        print("  %-20s %8.1f samples/s  %s  final acc %.3f"
              % (cfg.name, row["samples_per_s"],
                 ("%7.0f samples -> $%.4f" % (row["samples_to_target"], row["usd_to_target"]))
                 if row["usd_to_target"] is not None else "  never reached target",
                 row["final_val_acc"]))

    by_throughput = [r["name"] for r in sorted(rows, key=lambda r: -r["samples_per_s"])]
    costed = [r for r in rows if r["usd_to_target"] is not None]
    by_cost = [r["name"] for r in sorted(costed, key=lambda r: r["usd_to_target"])]

    # Compare the FULL orderings, not just the winners.
    #
    # The first version only checked `by_throughput[0] == by_cost[0]` and printed
    # "the throughput winner is also the cheapest" -- which was true, and buried
    # the actual result. The winner agreeing is the least interesting cell in the
    # table: a ladder does not only pick a winner, it ranks every rung, and this
    # study's real finding lives in the rungs that move.
    tput_rank = {n: i for i, n in enumerate(by_throughput)}
    cost_rank = {n: i for i, n in enumerate(by_cost)}
    moves = sorted(
        ({"name": n,
          "rank_by_throughput": tput_rank[n] + 1,
          "rank_by_cost": cost_rank[n] + 1,
          "moved": cost_rank[n] - tput_rank[n]} for n in cost_rank),
        key=lambda m: -abs(m["moved"]))
    # The dangerous mover is the one throughput ranks TOO HIGH -- promoted by
    # samples/s, demoted by cost. A config that moves the other way is one the
    # ladder was merely pessimistic about, which costs nobody anything.
    overrated = [m for m in moves if m["moved"] > 0]
    biggest = max(overrated, key=lambda m: m["moved"]) if overrated else None
    orders_identical = by_throughput[: len(by_cost)] == by_cost
    winners_agree = bool(by_cost) and by_throughput[0] == by_cost[0]

    cheapest = min(costed, key=lambda r: r["usd_to_target"]) if costed else None
    dearest = max(costed, key=lambda r: r["usd_to_target"]) if costed else None
    cost_spread = (dearest["usd_to_target"] / cheapest["usd_to_target"]) if cheapest else None
    return {
        "hardware": {"platform": platform.platform(),
                     "processor": platform.processor() or platform.machine(),
                     "cpu_count": os.cpu_count(), "torch": torch.__version__},
        "target_val_accuracy": target,
        "rate_usd_per_hour": rate_usd_per_hour,
        "repeats": repeats,
        "max_steps": max_steps,
        "rows": rows,
        "rank_by_throughput": by_throughput,
        "rank_by_cost_to_target": by_cost,
        "winners_agree": winners_agree,
        "orderings_identical": orders_identical,
        "rank_changes": moves,
        "cost_spread_dearest_over_cheapest": round(cost_spread, 2) if cost_spread else None,
        "reading": _reading(winners_agree, orders_identical, by_throughput, by_cost, biggest,
                            cheapest, dearest, by_cost_rows=costed),
        "caveat": ("synthetic task on CPU. The convergence shape belongs to this generator, so "
                   "the crossover point does not transfer -- what transfers is that the two "
                   "rankings can disagree, and that only measuring both catches it."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.85)
    ap.add_argument("--rate", type=float, default=0.35, help="USD per hour for the machine")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--train-n", type=int, default=4000)
    ap.add_argument("--decode-cost", type=int, default=120)
    args = ap.parse_args()

    out = run(target=args.target, rate_usd_per_hour=args.rate, repeats=args.repeats,
              max_steps=args.max_steps, train_n=args.train_n, decode_cost=args.decode_cost)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "cost_to_accuracy.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
    print("\nwritten:", os.path.relpath(path, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
