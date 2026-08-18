"""Cost ledger: turn throughput into dollars, and dollars into a decision.

The point of a cost ledger is not bookkeeping, it is answering "is scaling worth
the money at this size?" -- a question that has a numeric answer and is almost
never asked in portfolio projects.

    python -m trainlab.cost --ledger results/ladder_cpu.json --rate 0.35
"""
from __future__ import annotations

import argparse
import json

# Published on-demand rates are inputs, not facts about your bill: spot/preemptible
# capacity, committed-use discounts and idle time between runs all move the real
# number. They are passed in rather than hard-coded so the ledger cannot silently
# go stale.
EXAMPLE_RATES_USD_PER_HOUR = {
    "local-cpu": 0.0,
    "example-1xGPU": 0.35,
    "example-4xGPU": 1.40,
}


def samples_per_dollar(samples_per_s: float, rate_usd_per_hour: float) -> float:
    if rate_usd_per_hour <= 0:
        return float("inf")
    return samples_per_s * 3600.0 / rate_usd_per_hour


def cost_to_target(samples_needed: float, samples_per_s: float, rate_usd_per_hour: float) -> dict:
    hours = samples_needed / samples_per_s / 3600.0 if samples_per_s > 0 else float("inf")
    return {"hours": hours, "usd": hours * rate_usd_per_hour}


def scaling_worth_it(single_tp: float, single_rate: float, multi_tp: float, multi_rate: float,
                     samples_needed: float) -> dict:
    """The actual decision, with the arithmetic shown.

    Scaling buys wall-clock and costs money whenever efficiency < 1. Whether that
    trade is worth taking depends on how much the wall-clock is worth, which is a
    business input the engineer does not get to invent -- so this returns both
    numbers and the break-even, not a verdict.
    """
    one = cost_to_target(samples_needed, single_tp, single_rate)
    many = cost_to_target(samples_needed, multi_tp, multi_rate)
    hours_saved = one["hours"] - many["hours"]
    extra_usd = many["usd"] - one["usd"]
    return {
        "single": one,
        "scaled": many,
        "hours_saved": hours_saved,
        "extra_usd": extra_usd,
        "usd_per_hour_saved": (extra_usd / hours_saved) if hours_saved > 0 else float("inf"),
        "verdict_input_needed": "worth it iff an engineer-hour of waiting is worth more than usd_per_hour_saved",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--rate", type=float, default=0.0, help="USD per hour for the device that produced the ledger")
    ap.add_argument("--samples-needed", type=float, default=5_000_000)
    args = ap.parse_args()

    with open(args.ledger) as fh:
        rows = [r for r in json.load(fh)["rows"] if "samples_per_s" in r]

    print("| rung | samples/s | hours to %.0e samples | USD @ $%.2f/h |" % (args.samples_needed, args.rate))
    print("|---|---|---|---|")
    for r in rows:
        c = cost_to_target(args.samples_needed, r["samples_per_s"], args.rate)
        print("| %s | %.1f | %.2f | %.2f |" % (r["run"], r["samples_per_s"], c["hours"], c["usd"]))

    best = max(rows, key=lambda r: r["samples_per_s"])
    base = rows[0]
    saved = cost_to_target(args.samples_needed, base["samples_per_s"], args.rate)["usd"] - \
        cost_to_target(args.samples_needed, best["samples_per_s"], args.rate)["usd"]
    print("\nOptimisation saves $%.2f per %.0e samples at $%.2f/h (%s vs baseline)."
          % (saved, args.samples_needed, args.rate, best["run"]))
    if args.rate == 0:
        print("Rate is $0.00 (local hardware): this study's direct compute cost was $0. "
              "The table above is the template for a rented-hardware rerun, not a bill.")


if __name__ == "__main__":
    main()
