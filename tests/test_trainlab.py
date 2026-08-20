"""Tests for the measurement machinery.

If the timer is wrong, every number in the report is wrong and no amount of
careful methodology elsewhere saves it. These tests are about the instrument.
"""
import time

import pytest
import torch

from trainlab.ladder import LADDER, RunConfig, _aggregate, _assert_single_change
from trainlab.profiling import StepTimer, bottleneck
from trainlab.workload import SmallCNN, SyntheticImages, scaled_lr, warmup_steps


def test_timer_attributes_time_to_the_right_phase():
    t = StepTimer(torch.device("cpu"))
    with t.phase("data"):
        time.sleep(0.05)
    with t.phase("forward"):
        time.sleep(0.01)
    s = t.summary(n_samples=10, wall_s=0.06)
    assert s["data_s"] > s["forward_s"]
    assert s["data_pct"] > 60


def test_summary_reports_unattributed_time_instead_of_renormalising():
    """Wall time not inside any phase must be visible, not silently absorbed."""
    t = StepTimer(torch.device("cpu"))
    with t.phase("forward"):
        time.sleep(0.01)
    s = t.summary(n_samples=1, wall_s=1.0)
    assert s["unattributed_s"] > 0.9
    assert s["unattributed_pct"] > 85


def test_timer_records_which_clock_produced_the_numbers():
    assert StepTimer(torch.device("cpu")).timer_kind == "perf_counter"


def test_bottleneck_prefers_data_wait_above_the_threshold():
    # data at 25% is not the largest phase, but it is the one to fix first
    s = {"data_pct": 25.0, "forward_pct": 40.0, "backward_pct": 30.0, "optimizer_pct": 5.0}
    assert bottleneck(s) == "data"
    s2 = {"data_pct": 5.0, "forward_pct": 30.0, "backward_pct": 60.0, "optimizer_pct": 5.0}
    assert bottleneck(s2) == "backward"


def test_ladder_rungs_change_exactly_one_thing():
    """The rule the whole study rests on, enforced across the real ladder."""
    for prev, cur in zip(LADDER, LADDER[1:]):
        _assert_single_change(prev, cur)


def test_multi_change_rung_is_rejected():
    a = RunConfig("a")
    b = RunConfig("b", num_workers=4, amp=True)
    with pytest.raises(ValueError):
        _assert_single_change(a, b)


def test_batch_and_lr_may_move_together():
    a = RunConfig("a", batch_size=64, lr=0.01)
    b = RunConfig("b", batch_size=128, lr=0.02)
    _assert_single_change(a, b)  # must not raise: the LR rule follows from the batch change


def test_lr_scaling_rules():
    assert scaled_lr(0.01, 64, 128, "linear") == pytest.approx(0.02)
    assert scaled_lr(0.01, 64, 256, "sqrt") == pytest.approx(0.02)
    with pytest.raises(ValueError):
        scaled_lr(0.01, 64, 128, "magic")


def test_warmup_is_never_zero():
    assert warmup_steps(1) >= 1
    assert warmup_steps(1000, 0.05) == 50


def test_aggregate_reports_spread_and_keeps_raw_values():
    reps = [
        {"samples_per_s": 100.0, "ms_per_step": 10.0, "data_pct": 10.0, "forward_pct": 40.0,
         "backward_pct": 45.0, "optimizer_pct": 5.0, "unattributed_pct": 0.0, "wall_s": 1.0},
        {"samples_per_s": 120.0, "ms_per_step": 8.0, "data_pct": 12.0, "forward_pct": 40.0,
         "backward_pct": 43.0, "optimizer_pct": 5.0, "unattributed_pct": 0.0, "wall_s": 1.0},
    ]
    agg = _aggregate(reps)
    assert agg["samples_per_s"] == pytest.approx(110.0)
    assert agg["samples_per_s_std"] > 0
    assert agg["samples_per_s_raw"] == [100.0, 120.0]
    assert agg["repeats"] == 2


def test_dataset_is_deterministic_per_index():
    ds = SyntheticImages(n=16, decode_cost=1)
    a, la = ds[3]
    b, lb = ds[3]
    assert torch.allclose(a, b)
    assert la == lb


def test_model_forward_shape():
    m = SmallCNN(n_classes=10, width=8)
    out = m(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_scaling_protocol_is_weak_and_lr_follows_global_batch():
    """The two decisions that make a scaling table meaningful, asserted."""
    from trainlab.workload import scaled_lr

    per_worker = 32
    for world in (1, 2, 4, 8):
        global_batch = per_worker * world
        lr = scaled_lr(0.01, per_worker, global_batch)
        # Linear rule against the GLOBAL batch: doubling workers doubles the LR.
        assert abs(lr - 0.01 * world) < 1e-9


def test_scaling_report_efficiency_is_internally_consistent():
    """Guards against a table whose speedup and efficiency disagree."""
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "results", "scaling_cpu_gloo_weak.json")
    if not os.path.exists(path):
        import pytest

        pytest.skip("scaling study not run in this checkout")
    report = json.load(open(path))
    base = report["rows"][0]
    for row in report["rows"]:
        expected_speedup = row["aggregate_samples_per_s"] / base["aggregate_samples_per_s"]
        assert abs(row["speedup"] - expected_speedup) < 1e-6
        assert abs(row["efficiency"] - row["speedup"] / row["ideal_speedup"]) < 1e-6
        # Efficiency above 1 at a larger world size would mean superlinear
        # scaling, which for this workload means the baseline was mismeasured.
        assert row["efficiency"] <= 1.05


def test_comm_fraction_rises_with_world_size():
    """The measured explanation for the efficiency gap must behave sensibly."""
    import json
    import os

    path = os.path.join(os.path.dirname(__file__), "..", "results", "scaling_cpu_gloo_weak.json")
    if not os.path.exists(path):
        import pytest

        pytest.skip("scaling study not run in this checkout")
    rows = json.load(open(path))["rows"]
    fractions = [r["comm_fraction"] for r in rows]
    assert fractions[0] == 0.0, "a single worker performs no all-reduce"
    # Deliberately NOT asserting monotonicity any more. Repeating the paired
    # no_sync comparison showed a spread three to ten times the median, so this
    # estimator is too noisy on a contended CPU to carry an ordering assertion.
    # What IS assertable is that the spread is reported at all, so a reader can
    # see the number is unusable rather than trusting it.
    for r in rows[1:]:
        assert "comm_fraction_spread" in r, "an unusable estimate must ship with its spread"
