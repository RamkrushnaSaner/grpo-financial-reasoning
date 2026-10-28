import pytest
from conftest import SAMPLE, make_trained

from grpo.data import FinancialTask
from grpo.evaluation import (
    ask_both,
    compare,
    evaluate,
    format_comparison,
    render_ab,
    run_probes,
    summarize_probes,
)
from grpo.stats import (
    mcnemar_exact,
    paired_binary,
    paired_bootstrap_ci,
    standard_error_pp,
    window_mean,
)

TASK = FinancialTask(version=1)


# ---------------------------------------------------------------- held-out evaluation
def test_evaluate_returns_bounded_metrics_and_one_record_per_question(policy, cfg):
    r = evaluate(policy, TASK, cfg)
    for k in ("acc_overall", "format", "pass@1", f"pass@{cfg.evaluation.k}"):
        assert 0.0 <= r[k] <= 1.0
    assert r["n"] == cfg.evaluation.n and len(r["records"]) == cfg.evaluation.n
    rec = r["records"][0]
    assert rec["index"] == TASK.train_pool and {"correct", "format", "tokens", "text",
                                                 "pass_at_1", "pass_at_k"} <= set(rec)
    assert sum(r[f"n_{g}"] for g in ("1step", "2step") if f"n_{g}" in r) == r["n"]


def test_sampled_metrics_are_reproducible(policy, cfg):
    make_trained(policy, scale=2.0)
    a, b = evaluate(policy, TASK, cfg), evaluate(policy, TASK, cfg)
    assert [x["n_correct"] for x in a["records"]] == [x["n_correct"] for x in b["records"]]
    assert [x["text"] for x in a["records"]] == [x["text"] for x in b["records"]]


# ---------------------------------------------------------------- probes
def test_ask_both_with_sampling_restores_adapter(policy, cfg):
    """Regression for the reused-context-manager crash, which only fired with sampling."""
    make_trained(policy, scale=2.0)
    mag = policy.adapter_magnitude()
    out = ask_both(policy, TASK, SAMPLE, cfg, n_samples=3)
    assert set(out) == {"BASE", "TRAINED"} and "pass_k" in out["TRAINED"]
    assert policy.adapter_magnitude() == pytest.approx(mag)
    assert "TRAINED" in render_ab(SAMPLE, out)


def test_ask_both_restores_adapter_when_generation_raises(policy, cfg):
    make_trained(policy, scale=2.0)
    mag = policy.adapter_magnitude()

    def boom(*a, **k):
        raise RuntimeError("generation failed")
    policy.model.generate = boom
    with pytest.raises(RuntimeError):
        ask_both(policy, TASK, SAMPLE, cfg)
    assert policy.adapter_magnitude() == pytest.approx(mag)      # NOT left at base


def test_run_probes_and_summary(policy, cfg):
    make_trained(policy)
    qs = [dict(SAMPLE, id="A1", capability="x", expect="IMPROVE"),
          dict(SAMPLE, id="C1", capability="y", expect="NO CHANGE", answer=None, answer_type=None)]
    rows = run_probes(policy, TASK, qs, cfg, verbose=False)
    assert len(rows) == 2 and rows[1]["base_exact"] is None
    assert "PROBE SUMMARY" in summarize_probes(rows)


# ---------------------------------------------------------------- statistics
def test_mcnemar_exact_values():
    # The two extreme pairings consistent with Run 6's 10/60 -> 26/60 accuracy.
    assert mcnemar_exact(10, 26) == pytest.approx(0.0113, abs=1e-4)
    assert mcnemar_exact(0, 16) == pytest.approx(3.05e-5, rel=1e-2)
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(3, 3) == 1.0


def test_paired_binary_counts_changed_questions():
    t = paired_binary([True, True, False, False], [True, False, True, True])
    assert (t["worse"], t["better"], t["before"], t["after"]) == (1, 2, 0.5, 0.75)


def test_paired_bootstrap_is_deterministic_and_brackets_the_mean():
    a = paired_bootstrap_ci([0, 0, 0, 0], [1, 1, 0, 1])
    assert a == paired_bootstrap_ci([0, 0, 0, 0], [1, 1, 0, 1])
    assert a["ci_low"] <= a["mean_diff"] == 0.75 <= a["ci_high"]


def test_window_mean_and_standard_error():
    s = [0.0] * 5 + [1.0] * 5
    assert window_mean(s, True, 5) == 0.0 and window_mean(s, False, 5) == 1.0
    assert window_mean([None, 1.0, 3.0], True, 3) == 2.0
    assert standard_error_pp(60) == pytest.approx(6.45, abs=0.01)


# ---------------------------------------------------------------- comparison
def _ev(correct, fmt, tokens=None):
    recs = [dict(index=400 + i, subgroup="1step" if i % 2 else "2step", correct=c, format=f,
                 tokens=(tokens or [50] * len(correct))[i], pass_at_1=float(c), pass_at_k=float(c))
            for i, (c, f) in enumerate(zip(correct, fmt, strict=True))]
    from grpo.evaluation import summarize_records
    return summarize_records(recs, k=6)


def test_compare_uses_paired_tests_and_flags_format_collapse():
    n = 40
    before = _ev([i < 6 for i in range(n)], [i < 30 for i in range(n)])
    after = _ev([i < 20 for i in range(n)], [i < 5 for i in range(n)])
    comp = compare(before, after)
    rows = {r["key"]: r for r in comp["rows"]}
    assert comp["paired"] and rows["acc_overall"]["test"] == "McNemar"
    assert (rows["acc_overall"]["better"], rows["acc_overall"]["worse"]) == (14, 0)
    assert rows["acc_overall"]["verdict"] == "significant"
    assert rows["format"]["verdict"] == "significant" and rows["format"]["delta_pp"] < 0
    assert {"acc_1step", "acc_2step"} <= set(rows)
    text = format_comparison(comp)
    assert "format compliance fell" in text and "correct_gated" in text
    assert "improved accuracy" in comp["verdict"]


def test_compare_without_records_reports_deltas_untested():
    """Run 6's saved results have no per-question records: no significance claimed."""
    before = {"acc_overall": 0.1667, "format": 0.6167, "n": 60, "mean_len": 57.0}
    after = {"acc_overall": 0.4333, "format": 0.2000, "n": 60, "mean_len": 47.0}
    comp = compare(before, after)
    assert not comp["paired"]
    assert all(r["test"] == "none" for r in comp["rows"])
    assert "not computed" in comp["verdict"]
    assert rows_delta(comp, "acc_overall") == pytest.approx(26.66, abs=0.05)


def rows_delta(comp, key):
    return next(r["delta_pp"] for r in comp["rows"] if r["key"] == key)


class _CoinTask(FinancialTask):
    """Correctness depends on the sampled text, so sampled metrics vary with the RNG."""

    def is_correct(self, text, sample):
        return len(text) % 2 == 0


def test_evaluation_seeds_its_own_sampling(policy, cfg):
    """Regression: pass@k used to depend on how much randomness earlier stages used."""
    import torch
    task = _CoinTask(version=1)
    make_trained(policy, scale=1.0)
    torch.manual_seed(1)
    a = evaluate(policy, task, cfg.with_overrides({"evaluation.k": 6}))
    torch.manual_seed(2)
    b = evaluate(policy, task, cfg.with_overrides({"evaluation.k": 6}))
    assert [x["n_correct"] for x in a["records"]] == [x["n_correct"] for x in b["records"]]
    c = evaluate(policy, task, cfg.with_overrides({"evaluation.k": 6, "experiment.seed": 7}))
    assert [x["n_correct"] for x in c["records"]] != [x["n_correct"] for x in a["records"]]
