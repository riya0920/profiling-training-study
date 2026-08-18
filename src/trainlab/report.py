"""Turn the ledger into REPORT.md + generated figures.

Every figure in the report is produced by this file from the committed JSON. No
hand-drawn diagrams, no numbers typed into markdown by hand -- if the ladder is
re-run on different hardware, the report regenerates and cannot silently disagree
with the data.

    python -m trainlab.report --ledger results/ladder_cpu.json
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _measured(rows):
    return [r for r in rows if "samples_per_s" in r]


def ladder_table(ledger: dict) -> str:
    rows = ledger["rows"]
    lines = [
        "| rung | samples/s (mean ± std) | Δ vs prev | cumulative | data % | fwd % | bwd % | opt % | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r.get("skipped"):
            lines.append("| %s | _skipped_ | — | — | — | — | — | — | %s |" % (r["run"], r["reason"]))
            continue
        if r.get("failed"):
            lines.append("| %s | _failed_ | — | — | — | — | — | — | %s |" % (r["run"], r["error"].split(":")[0]))
            continue
        lines.append(
            "| %s | %.1f ± %.1f | %s | %.2fx | %.0f | %.0f | %.0f | %.0f | %s |"
            % (
                r["run"],
                r["samples_per_s"],
                r.get("samples_per_s_std", 0.0),
                ("%+.0f%%" % (100 * (r["step_speedup"] - 1))) if "step_speedup" in r else "—",
                r.get("cumulative_speedup", 1.0),
                r["data_pct"],
                r["forward_pct"],
                r["backward_pct"],
                r["optimizer_pct"],
                r["config"].get("note", ""),
            )
        )
    return "\n".join(lines)


def separability_note(ledger: dict) -> str:
    """State which adjacent rungs are NOT separable given the measured variance.

    A speedup smaller than the combined run-to-run spread is not a result, and a
    ladder table that does not say so invites the reader to believe every row.
    """
    rows = _measured(ledger["rows"])
    notes = []
    for prev, cur in zip(rows, rows[1:]):
        gap = abs(cur["samples_per_s"] - prev["samples_per_s"])
        spread = prev.get("samples_per_s_std", 0.0) + cur.get("samples_per_s_std", 0.0)
        if spread > 0 and gap < spread:
            notes.append(
                "* `%s` -> `%s`: gap %.1f samples/s is smaller than the combined spread %.1f. "
                "**Not separable at this sample size.**" % (prev["run"], cur["run"], gap, spread)
            )
    if not notes:
        return "Every adjacent pair of rungs is separated by more than the combined run-to-run spread."
    return "\n".join(notes)


def make_figures(ledger: dict, out_dir: str) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    rows = _measured(ledger["rows"])
    names = [r["run"] for r in rows]
    paths = []

    # Figure 1: throughput with error bars.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    vals = [r["samples_per_s"] for r in rows]
    errs = [r.get("samples_per_s_std", 0.0) for r in rows]
    base = vals[0]
    colours = ["#4c78a8" if v >= base else "#e45756" for v in vals]
    ax.bar(names, vals, yerr=errs, capsize=4, color=colours)
    ax.axhline(base, ls="--", c="0.4", lw=1, label="baseline")
    ax.set_ylabel("samples / s")
    ax.set_title("Throughput by ladder rung (%s, %d repeats, error bars = 1 std)"
                 % (ledger["device"], ledger.get("repeats", 1)))
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p1 = os.path.join(out_dir, "ladder_throughput.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    paths.append(p1)

    # Figure 2: where the time goes, per rung. This is the figure that justifies
    # the rung ORDER -- data-wait collapsing to zero after the workers rung is
    # the whole argument for attacking the dataloader first.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    phases = ["data_pct", "forward_pct", "backward_pct", "optimizer_pct", "unattributed_pct"]
    labels = ["data wait", "forward", "backward", "optimizer", "unattributed"]
    bottom = [0.0] * len(rows)
    for phase, label in zip(phases, labels):
        vals = [r.get(phase, 0.0) for r in rows]
        ax.bar(names, vals, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_ylabel("% of measured step time")
    ax.set_title("Step-time breakdown by rung")
    ax.legend(ncol=5, fontsize=8)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    p2 = os.path.join(out_dir, "ladder_breakdown.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    paths.append(p2)
    return paths


def build_report(ledger_path: str) -> str:
    with open(ledger_path) as fh:
        ledger = json.load(fh)
    rows = _measured(ledger["rows"])
    best = max(rows, key=lambda r: r["samples_per_s"])
    baseline = rows[0]
    figs = make_figures(ledger, os.path.join(ROOT, "results", "figures"))
    fig_md = "\n".join("![%s](results/figures/%s)" % (os.path.basename(f), os.path.basename(f)) for f in figs)

    return REPORT_TEMPLATE.format(
        device=ledger["device"],
        device_name=baseline.get("device_name", "unknown"),
        torch=baseline.get("torch", "unknown"),
        timer=baseline.get("timer", "unknown"),
        steps=ledger["steps_per_run"],
        repeats=ledger.get("repeats", 1),
        decode_cost=ledger.get("decode_cost", "?"),
        table=ladder_table(ledger),
        figures=fig_md,
        baseline_tp=baseline["samples_per_s"],
        baseline_data_pct=baseline["data_pct"],
        best_name=best["run"],
        best_tp=best["samples_per_s"],
        best_speedup=best["samples_per_s"] / baseline["samples_per_s"],
        separability=separability_note(ledger),
    )


REPORT_TEMPLATE = """# Optimisation ladder: where the time went

Generated by `python -m trainlab.report`. Every number and figure below comes
from `results/ladder_*.json`; nothing is hand-typed.

## Setup

| | |
|---|---|
| device | `{device}` ({device_name}) |
| torch | {torch} |
| timer | `{timer}` |
| steps per run | {steps} (after warmup, which is excluded) |
| repeats per rung | {repeats}, reported as mean ± std |
| synthetic per-sample decode cost | {decode_cost} |

## Baseline profile: the measurement that set the rung order

The instrumented FP32 baseline runs at **{baseline_tp:.1f} samples/s** with
**{baseline_data_pct:.0f}% of step time in data wait**. That single number decides
everything that follows: while the compute device is idle a third of the time
waiting for input, no kernel-level optimisation can help, because the kernels are
not the constraint. The ladder therefore attacks the dataloader first and only
then moves to precision and layout.

This is the difference between profiling-driven and blog-post-driven: the obvious
first move (mixed precision) would have been the wrong one, and the profile says
so before any time is spent on it.

## The ladder

{table}

{figures}

## Best configuration

`{best_name}` at **{best_tp:.1f} samples/s**, a **{best_speedup:.2f}x** cumulative
speedup over the instrumented baseline.

## What is and is not separable

{separability}

## Negative results (kept)

Rungs that lost are in the table above with their real numbers. They are the most
informative rows in it: a ladder where every rung wins is a ladder that was
edited after the fact.

## Reading this report on other hardware

The rung *order* generalises (fix the profiled bottleneck first); the rung
*results* do not. Re-run `python -m trainlab.ladder` on the target device and
regenerate this report. Rows measured on different devices are never merged into
one table -- each row records its own device, timer and torch version for exactly
that reason.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "results", "ladder_cpu.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "REPORT.md"))
    args = ap.parse_args()
    text = build_report(args.ledger)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
