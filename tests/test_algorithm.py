import math

import pytest
import torch
from conftest import SAMPLE, make_trained, tiny_config

from grpo.algorithm import (
    ADVANTAGES,
    LOSSES,
    build_prompt_ids,
    completion_mask,
    group_advantages,
    grpo_loss,
    grpo_step,
    true_lengths,
)
from grpo.config import RewardComponentConfig, RewardConfig
from grpo.data import FinancialTask
from grpo.rewards import CompositeReward, RewardComponent, build_reward
from grpo.stats import pass_at_k
from grpo.testing import tiny_policy
from grpo.validation import clip_engagement_test, gradient_direction_test, validate_math


def test_advantages_are_centred_and_normalised():
    a = group_advantages(torch.tensor([1.1, 0.1, 1.1, 0.1, 0.1, 0.0]))
    assert abs(a.mean()) < 1e-5 and abs(a.std() - 1) < 0.02


def test_degenerate_group_is_skipped():
    assert group_advantages([0.5] * 6) is None


def test_k3_kl_is_non_negative():
    d = torch.linspace(-3, 3, 2001)
    assert (torch.exp(d) - d - 1).min() >= -1e-6


@pytest.mark.parametrize("rho,adv,clipped", [(1.5, 1.0, True), (0.5, 1.0, False),
                                             (1.5, -1.0, False), (1.0, 1.0, False)])
def test_clipping_cases(rho, adv, clipped):
    r, a = torch.tensor(rho), torch.tensor(adv)
    obj = torch.min(r * a, torch.clamp(r, 0.8, 1.2) * a)
    assert (not torch.isclose(obj, r * a)) == clipped


def test_mask_uses_plen_minus_one():
    assert completion_mask(1, 6, 3, [6])[0].tolist() == [0., 0., 1., 1., 1.]
    m = completion_mask(2, 8, 3, [6, 8])
    assert m[0].tolist() == [0., 0., 1., 1., 1., 0., 0.]
    assert m[1].tolist() == [0., 0., 1., 1., 1., 1., 1.]


def test_mask_regression_first_generated_token_is_scored():
    """Masking from plen (the historical bug) would leave position plen-1 at 0."""
    assert completion_mask(1, 6, 3, [6])[0, 2] == 1.0


def test_true_lengths_include_the_stop_token():
    seqs = torch.tensor([[5, 6, 7, 11, 12, 1, 0, 0],
                         [5, 6, 7, 11, 12, 13, 14, 1],
                         [5, 6, 7, 11, 12, 13, 14, 15]])
    assert true_lengths(seqs, 3, 1) == [6, 8, 8]


def test_true_lengths_stop_on_any_stop_token():
    """An answer ending on a second stop id (which may equal the pad id) must not be
    treated as running into its padding."""
    seqs = torch.tensor([[5, 6, 7, 11, 0, 0, 0, 0],      # ends on 0, padded with 0
                         [5, 6, 7, 11, 12, 1, 0, 0]])
    assert true_lengths(seqs, 3, [1]) == [8, 6]
    assert true_lengths(seqs, 3, [0, 1]) == [5, 6]


def test_clip_fraction_is_zero_when_ratio_is_one():
    lp = torch.randn(3, 5)
    _, kl, cf = grpo_loss(lp, lp.detach(), lp.detach(), torch.tensor([1.0, -0.5, -0.5]),
                          torch.ones(3, 5))
    assert cf == 0.0 and abs(kl) < 1e-7


def test_gradient_direction():
    g = gradient_direction_test()
    assert g["delta"][0] > 0 and all(x < 0 for x in g["delta"][1:])
    assert g["corr"] > 0.9


def test_clipping_inert_at_mu1_engaged_after():
    fr = clip_engagement_test(mu=4)
    assert fr[0] == 0.0 and max(fr) > 0.0


@pytest.mark.parametrize("n,c,k,expected", [(6, 0, 1, 0.0), (6, 6, 1, 1.0), (6, 1, 6, 1.0),
                                            (6, 1, 1, 1 / 6), (6, 3, 2, 0.8)])
def test_pass_at_k(n, c, k, expected):
    assert pass_at_k(n, c, k) == pytest.approx(expected)


def test_validate_math_end_to_end():
    out = validate_math(verbose=False)
    assert out["grad_corr"] > 0.9 and out["clip_fracs"][0] == 0.0


def test_grpo_loss_uses_k3_not_naive_kl():
    """With logp_ref = logp_new + 1, k3 = e - 2 while the naive estimator gives -1."""
    lp = torch.zeros(2, 4)
    _, kl, _ = grpo_loss(lp, lp, lp + 1.0, torch.tensor([1.0, -1.0]), torch.ones(2, 4))
    assert kl == pytest.approx(math.e - 2, abs=1e-5)


# ---------------------------------------------------------------- golden regression
# Values produced by the ORIGINAL (pre-refactor) implementation on the same
# inputs. On the machine that recorded them the refactored code reproduces them
# exactly; other CPUs take different float32 kernel paths (AVX-512 vs AVX2),
# which moves results by ~1e-7, so the tolerance is float32 noise, not a margin
# for real changes (any change to the update moves these values far more).
GOLDEN = {  # arm, mu -> (loss, kl, clip_frac, sum |lora|)
    ("graded", 1): (0.03772466257214546, 0.9583109021186829, 0.0, 69.49972481373698),
    ("graded", 3): (-0.001306295394897461, 0.8914210200309753, 0.1607142835855484,
                    69.48584760725498),
    ("binary", 3): (-0.0076995291747152805, 0.9067458510398865, 0.1785714328289032,
                    69.44497679267079),
}
ARMS = {"graded": (("correct_graded", 1.0, {"lam": 0.3, "kappa": 0.2}), ("format", 0.1, {})),
        "binary": (("correct_binary", 1.0, {}), ("format", 0.1, {}))}


@pytest.mark.parametrize("arm,mu", list(GOLDEN))
def test_update_matches_the_original_implementation(arm, mu):
    cfg = tiny_config(algorithm=dict(mu=mu))
    task = FinancialTask(version=1)
    torch.manual_seed(0)
    pol = make_trained(tiny_policy(), scale=0.3)
    reward = build_reward(RewardConfig(tuple(RewardComponentConfig(*c) for c in ARMS[arm])), task)
    opt = torch.optim.AdamW(pol.trainable_parameters(), lr=1e-2)
    torch.manual_seed(5)
    res = grpo_step(pol, build_prompt_ids(pol, task.messages(SAMPLE)), SAMPLE, reward, opt, cfg)
    lora = float(sum(v.double().abs().sum() for k, v in pol.model.state_dict().items()
                     if "lora_" in k))
    assert (res.loss, res.kl, res.clip_frac) == pytest.approx(GOLDEN[(arm, mu)][:3], abs=1e-5)
    assert lora == pytest.approx(GOLDEN[(arm, mu)][3], abs=1e-4)


# ---------------------------------------------------------------- extension points
def _length_reward():
    return CompositeReward([RewardComponent("len", lambda t, s: len(t) / 10.0, 1.0)])


def test_registered_advantage_and_loss_are_used_by_the_step(policy):
    calls = []

    @ADVANTAGES.register("test_mean_only")
    def mean_only(rewards):
        calls.append("adv")
        return rewards - rewards.mean() if rewards.std() > 0 else None

    @LOSSES.register("test_wrapped")
    def wrapped(*args, epsilon, beta):
        calls.append("loss")
        return grpo_loss(*args, epsilon=epsilon, beta=beta)

    cfg = tiny_config(algorithm=dict(advantage="test_mean_only", loss="test_wrapped"))
    torch.manual_seed(3)
    opt = torch.optim.SGD(policy.trainable_parameters(), lr=0.5)
    task = FinancialTask(size=10, train_pool=8)
    res = grpo_step(policy, build_prompt_ids(policy, task.messages(SAMPLE)), SAMPLE,
                    _length_reward(), opt, cfg)
    assert not res.skipped and calls == ["adv", "loss"]


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):
        LOSSES.register("grpo")(lambda *a, **k: None)
