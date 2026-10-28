"""The real grpo_step and training loop, driven by the tiny CPU policy."""
import os

import numpy as np
import pytest
import torch
from conftest import SAMPLE, make_trained, tiny_config

from grpo.algorithm import build_prompt_ids, grpo_step
from grpo.data import FinancialTask
from grpo.rewards import CompositeReward, RewardComponent, build_reward
from grpo.testing import tiny_policy
from grpo.tracking import read_metrics
from grpo.training import train

TASK = FinancialTask(version=1)


def _length_reward():
    # reward varies with completion length, guaranteeing a non-degenerate group
    return CompositeReward([RewardComponent("len", lambda t, s: len(t) / 10.0, 1.0)])


def _step(policy, cfg, reward, lr=0.5, opt=None):
    opt = opt or torch.optim.SGD(policy.trainable_parameters(), lr=lr)
    return grpo_step(policy, build_prompt_ids(policy, TASK.messages(SAMPLE)), SAMPLE, reward,
                     opt, cfg)


def test_step_updates_only_lora_and_ratio_is_exactly_one(policy, cfg):
    torch.manual_seed(3)
    base_before = policy.model.base.weight.clone()
    res = _step(policy, cfg, _length_reward())
    assert not res.skipped and np.isfinite(res.loss) and res.grad_norm > 0
    assert res.clip_frac == 0.0 and res.ratio_dev < 1e-6       # mu=1, dropout=0
    assert res.kl >= 0.0 and len(res.texts) == cfg.algorithm.group_size
    assert policy.is_trained()                                  # LoRA moved
    assert torch.equal(policy.model.base.weight, base_before)    # base frozen


def test_step_skips_degenerate_group(policy, cfg):
    res = _step(policy, cfg, CompositeReward([RewardComponent("c", lambda t, s: 1.0, 1.0)]))
    assert res.skipped and res.grad_norm is None and not policy.is_trained()


def test_mu_greater_than_one_engages_clipping(policy, tmp_path):
    torch.manual_seed(3)
    res = _step(policy, tiny_config(tmp_path, algorithm=dict(mu=4)), _length_reward(), lr=5.0)
    assert res.clip_frac > 0.0


def test_lora_dropout_breaks_the_ratio_identity(cfg):
    """Reproduces the Run 6 finding on CPU: logp_old is scored in eval mode and
    logp_new in train mode, so dropout makes rho != 1 even at mu=1."""
    p = make_trained(tiny_policy(lora_dropout=0.5))
    torch.manual_seed(3)
    res = _step(p, cfg, _length_reward(), lr=0.1)
    assert not res.skipped and res.ratio_dev > 1e-3


def test_train_writes_metrics_and_caches_weights_not_just_metrics(policy, cfg, tmp_path):
    run_dir = tmp_path / "run"
    result = train(policy, TASK, cfg, run_dir, log=lambda *a: None)
    mag = policy.adapter_magnitude()
    assert mag > 0 and len(result["history"]["kl"]) == cfg.training.steps
    lines = read_metrics(run_dir / "metrics.jsonl")
    assert [r["step"] for r in lines] == list(range(1, cfg.training.steps + 1))
    for key in ("loss", "grad_norm", "kl", "exact_rate", "format_rate", "partial_mass",
                "reward_mean", "reward_std", "n_distinct", "pass_at_g", "var_correct_graded"):
        assert key in lines[0], key
    policy.reset()
    cached = train(policy, TASK, cfg, run_dir, log=lambda *a: None)    # cache hit
    assert policy.adapter_magnitude() == pytest.approx(mag)            # weights restored
    assert cached["history"] == result["history"]


def test_changed_settings_invalidate_the_training_cache(policy, cfg, tmp_path):
    run_dir = tmp_path / "run"
    first = train(policy, TASK, cfg, run_dir, log=lambda *a: None)
    msgs = []
    second = train(policy, TASK, cfg.with_overrides({"training.steps": 3}), run_dir,
                   log=msgs.append)
    assert any("stale" in m for m in msgs)
    assert second["fingerprint"] != first["fingerprint"] and len(second["history"]["kl"]) == 3


def test_missing_adapter_file_forces_retraining(policy, cfg, tmp_path):
    run_dir = tmp_path / "run"
    train(policy, TASK, cfg, run_dir, log=lambda *a: None)
    os.remove(run_dir / "adapter.pt")
    policy.reset()
    train(policy, TASK, cfg, run_dir, log=lambda *a: None)
    assert policy.is_trained() and (run_dir / "adapter.pt").exists()


def test_gated_reward_trains(policy, tmp_path):
    cfg = tiny_config(tmp_path, reward=dict(components=[
        dict(name="correct_gated", params=dict(lam=0.3, kappa=0.2)),
        dict(name="format", weight=0.1)]))
    result = train(policy, TASK, cfg, tmp_path / "g", log=lambda *a: None)
    assert "var_correct_gated" in result["history"]


def test_reward_is_built_from_config(cfg):
    r = build_reward(cfg.reward, TASK)
    assert r.describe() == {"correct_graded": 1.0, "format": 0.1}


def test_step_ends_answers_on_every_stop_token(policy, cfg, monkeypatch):
    """Regression: an answer ending on a second stop id (here 0, also the pad id)
    must not be scored into its padding."""
    from types import SimpleNamespace

    import grpo.algorithm.step as step_mod
    plen = 4
    seqs = torch.tensor([[5, 6, 7, 8, 9, 10, 1, 0],          # stops on 1
                         [5, 6, 7, 8, 9, 0, 0, 0],           # stops on 0
                         [5, 6, 7, 8, 9, 10, 11, 12]] * 2)   # never stops
    monkeypatch.setattr(step_mod, "generate_group", lambda *a, **k: (seqs, plen))
    policy.model.generation_config = SimpleNamespace(eos_token_id=[1, 0])
    res = _step(policy, cfg, _length_reward())
    assert res.completion_lengths == [3, 2, 4] * 2


def test_partial_credit_telemetry_ignores_the_training_lambda(tmp_path):
    """Regression: measuring partial credit at each run's own lambda made the metric
    scale with the hyperparameter (it inflated a reported signal ~13x)."""
    from grpo.algorithm import StepResult
    from grpo.training import rollout_telemetry
    sample = {"answer": 39.0, "answer_type": "percentage"}
    texts = ["Answer: 37.33", "Answer: 39"]
    tel = {}
    for lam in (0.3, 0.45):
        cfg = tiny_config(tmp_path, reward=dict(components=[
            dict(name="correct_graded", params=dict(lam=lam, kappa=0.2))]))
        reward = build_reward(cfg.reward, TASK)
        totals, rows = reward.score_group(texts, sample)
        res = StepResult(loss=0.0, texts=texts, rewards=totals, breakdown=rows,
                         completion_lengths=[5, 5])
        tel[lam] = rollout_telemetry(TASK, res, sample, reward, cfg)["partial_mass"]
    assert tel[0.3] == tel[0.45] == pytest.approx(0.2422, abs=1e-3)
