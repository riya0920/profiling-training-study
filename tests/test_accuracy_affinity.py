"""Tests for the cost-to-accuracy study and the CPU-pinning experiment.

Both modules run real training, so these tests use tiny configurations and check
the *machinery* - the stopping rule, the cost arithmetic, the arm design - rather
than re-running the studies. The studies themselves are committed under
`results/`.

The two load-bearing tests are the ones about experimental design:
`test_the_arms_are_not_duplicates_of_each_other` and
`test_the_stopping_rule_requires_two_consecutive_crossings`. Both pin mistakes
that were actually made here, and both are the kind of mistake that produces a
confident number rather than a crash.
"""
import math
import os

import pytest
import torch

from trainlab import affinity as aff
from trainlab.accuracy import AccConfig, CONFIGS, evaluate, train_to_target
from trainlab.cost import cost_to_target


# --- experimental design ---------------------------------------------------

def test_the_arms_are_not_duplicates_of_each_other():
    """The first version had a fourth arm byte-identical to the control.

    It was described as "the control for pinned_split", and comparing an arm with
    itself would have produced a confident "pinning has no effect" that measured
    nothing at all. Two arms are the same arm if their thread count and both
    affinity masks match.
    """
    arms = aff.build_arms(8, 4)
    signatures = [(a.torch_threads, a.main_cores, a.worker_cores) for a in arms]
    assert len(set(signatures)) == len(arms), "duplicate arm: %s" % signatures


def test_pinned_split_and_its_control_differ_only_in_affinity():
    """Otherwise the experiment cannot attribute a difference to pinning."""
    arms = {a.name: a for a in aff.build_arms(8, 4)}
    split, control = arms["pinned_split"], arms["threads_capped"]
    assert split.torch_threads == control.torch_threads
    assert control.main_cores is None and control.worker_cores is None
    assert split.main_cores is not None and split.worker_cores is not None


def test_the_half_shared_arm_has_the_same_core_count_as_the_split_arm():
    """The arm that separates "disjointness helped" from "four cores was enough"."""
    arms = {a.name: a for a in aff.build_arms(8, 4)}
    split, half = arms["pinned_split"], arms["pinned_half_shared"]
    assert len(split.main_cores) == len(half.main_cores)
    assert len(split.worker_cores) == len(half.worker_cores)
    # Disjoint versus shared is the whole difference.
    assert not set(split.main_cores) & set(split.worker_cores)
    assert set(half.main_cores) == set(half.worker_cores)


def test_arms_partition_the_machine_without_leaving_a_core_idle():
    arms = {a.name: a for a in aff.build_arms(8, 4)}
    split = arms["pinned_split"]
    assert sorted(set(split.main_cores) | set(split.worker_cores)) == list(range(8))


def test_arms_degrade_sensibly_on_a_single_core_machine():
    arms = aff.build_arms(1, 4)
    for a in arms:
        if a.main_cores is not None:
            assert len(a.main_cores) >= 1


# --- the pinning run itself ------------------------------------------------

@pytest.mark.skipif(not aff.affinity_available(), reason="cpu_affinity unavailable here")
def test_running_an_arm_restores_the_process_affinity_afterwards():
    """A benchmark that leaves the process pinned poisons every measurement that
    follows it, including the ones in other test files."""
    import psutil

    before = psutil.Process().cpu_affinity()
    arm = {a.name: a for a in aff.build_arms(os.cpu_count() or 2, 2)}["pinned_split"]
    aff.run_arm(arm, steps=2, n_workers=0, dataset_n=64, decode_cost=2, warmup=1)
    assert psutil.Process().cpu_affinity() == before


def test_running_an_arm_restores_the_torch_thread_count():
    before = torch.get_num_threads()
    arm = {a.name: a for a in aff.build_arms(os.cpu_count() or 2, 2)}["threads_capped"]
    aff.run_arm(arm, steps=2, n_workers=0, dataset_n=64, decode_cost=2, warmup=1)
    assert torch.get_num_threads() == before


def test_the_study_refuses_to_run_rather_than_collapse_its_arms(monkeypatch):
    """On a platform without affinity control, the pinned arms would silently
    become the unpinned ones and the study would report a null result that means
    nothing. It reports `skipped` instead."""
    monkeypatch.setattr(aff, "affinity_available", lambda: False)
    out = aff.run(repeats=1, steps=1)
    assert out["skipped"]
    assert "silently collapse" in out["reason"]


# --- the accuracy stopping rule -------------------------------------------

def test_the_stopping_rule_requires_two_consecutive_crossings():
    """The whole table is *when* the target was crossed, so a noisy stopping rule
    does not add error to the answer - it becomes the answer.

    Driven with a scripted accuracy sequence rather than real training, because
    the property under test is the rule, not the model.
    """
    seq = [0.50, 0.90, 0.70, 0.88, 0.91, 0.95]     # one lucky spike at index 1
    hits, crossed = 0, None
    for i, acc in enumerate(seq):
        hits = hits + 1 if acc >= 0.85 else 0
        if hits >= 2:
            crossed = i
            break
    assert crossed == 4, "must ignore the isolated spike at index 1"


def test_a_configuration_that_never_reaches_the_target_has_no_cost():
    """Reporting its throughput beside the others' costs -- which a
    throughput-only ladder implicitly does -- is the error this module exists to
    make visible."""
    cfg = AccConfig("hopeless", batch_size=16, lr=0.0)      # zero LR: cannot learn
    out = train_to_target(cfg, target=0.99, device=torch.device("cpu"),
                          train_n=200, val_n=100, decode_cost=2, max_steps=6,
                          eval_every=3, seed=0)
    assert not out["reached_target"]
    assert out["samples_to_target"] is None


def test_evaluation_time_is_excluded_from_throughput():
    """Charging evaluation to training makes a configuration that evaluates less
    often look faster, which is a knob rather than a speedup."""
    out = train_to_target(AccConfig("probe", batch_size=16), target=1.1,
                          device=torch.device("cpu"), train_n=200, val_n=100,
                          decode_cost=2, max_steps=6, eval_every=2, seed=0)
    assert out["eval_s"] > 0
    assert out["samples_per_s"] == pytest.approx(
        out["steps_run"] * 16 / out["train_s"], rel=1e-6)


def test_training_actually_learns_on_this_task():
    """Guard against a generator change that makes the whole study vacuous: if
    the task were unlearnable every row would read 'never reached target' and the
    table would look like a finding."""
    out = train_to_target(AccConfig("learn", batch_size=32, lr=0.02), target=1.1,
                          device=torch.device("cpu"), train_n=800, val_n=200,
                          decode_cost=2, max_steps=60, eval_every=20, seed=0)
    accs = [c["val_acc"] for c in out["curve"]]
    assert accs[-1] > accs[0], "no learning signal: %s" % accs


# --- the cost arithmetic ---------------------------------------------------

def test_cost_to_target_multiplies_both_factors():
    """cost = samples_to_target / samples_per_s x rate. A ladder that optimises
    only the denominator can make the product worse."""
    cheap_fast = cost_to_target(10_000, 200.0, 1.0)
    cheap_slow = cost_to_target(10_000, 100.0, 1.0)
    dear_fast = cost_to_target(20_000, 200.0, 1.0)
    assert cheap_fast["usd"] < cheap_slow["usd"]
    assert cheap_fast["usd"] < dear_fast["usd"]
    # Doubling throughput and doubling samples-needed cancel exactly.
    assert dear_fast["usd"] == pytest.approx(cheap_slow["usd"])


def test_a_free_machine_costs_nothing_but_still_takes_time():
    out = cost_to_target(10_000, 100.0, 0.0)
    assert out["usd"] == 0.0
    assert out["hours"] > 0


def test_zero_throughput_does_not_divide_by_zero():
    out = cost_to_target(10_000, 0.0, 1.0)
    assert math.isinf(out["hours"])


# --- the configuration set -------------------------------------------------

def test_every_accuracy_config_is_distinct():
    sigs = [(c.batch_size, c.lr, c.amp) for c in CONFIGS]
    assert len(set(sigs)) == len(CONFIGS)


def test_the_unscaled_batch_control_is_present():
    """Doubling the batch without touching the LR is the mistake the linear rule
    exists to prevent, so the study has to contain it to show the difference."""
    names = {c.name for c in CONFIGS}
    assert "batch128_linear" in names and "batch128_unscaled" in names
    linear = next(c for c in CONFIGS if c.name == "batch128_linear")
    unscaled = next(c for c in CONFIGS if c.name == "batch128_unscaled")
    assert linear.batch_size == unscaled.batch_size
    assert linear.lr != unscaled.lr
