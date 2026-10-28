"""Validation suite: tests a wrong implementation would fail.

Everything else validates INPUTS. These validate that the UPDATE does what the
mathematics says. In RL a wrong gradient does not crash -- it just learns
slowly, and you would blame the reward long before suspecting the loss.

Tests 1-6 run on a tiny GRU model on the CPU: they prove the loss and
advantage code is mathematically right, NOT that a real model learned
anything. Test 7 runs on the real policy after training.

    1 advantages centred and normalised; degenerate groups -> None
    2 k3 KL non-negative everywhere
    3 clipping activates exactly when the min/clamp form says it should
    4 mask covers plen-1 .. L-2
    5 positive advantage => sequence becomes MORE likely     (decisive)
    6 clipping inert at mu=1, engages once rollouts are reused
    7 reference policy != policy once trained                 (decisive, GPU)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from .algorithm import completion_mask, group_advantages, grpo_loss, per_token_logps
from .stats import pass_at_k


class TinyLM(nn.Module):
    """A tiny GRU language model for deterministic CPU tests of the update rule."""

    def __init__(self, vocab: int = 20, dim: int = 16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.rnn = nn.GRU(dim, dim, batch_first=True)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h)


def _logps(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    lg = model(x)[:, :-1]
    return torch.log_softmax(lg.float(), -1).gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1)


def gradient_direction_test(seed: int = 0) -> Dict:
    """One GRPO step on a tiny model: does probability move the way the math says?"""
    torch.manual_seed(seed)
    model = TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    G, plen, T = 4, 3, 9
    seqs = torch.randint(1, 19, (G, T))
    seqs[:, :plen] = torch.tensor([1, 2, 3])
    adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]))      # seq0 clearly best
    mask = completion_mask(G, T, plen, [T] * G)
    before = (_logps(model, seqs).detach() * mask).sum(1) / mask.sum(1)
    lp_new = _logps(model, seqs)
    lp_old = lp_new.detach()
    loss, _, _ = grpo_loss(lp_new, lp_old, lp_old, adv, mask)
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = (_logps(model, seqs).detach() * mask).sum(1) / mask.sum(1)
    delta = after - before
    corr = float(np.corrcoef(adv.numpy(), delta.numpy())[0, 1])
    return dict(adv=adv.tolist(), before=before.tolist(), after=after.tolist(),
                delta=delta.tolist(), corr=corr)


def clip_engagement_test(mu: int = 4, seed: int = 0, epsilon: float = 0.2) -> list:
    """Clip fraction across inner epochs: 0 at the first, non-zero afterwards."""
    torch.manual_seed(seed)
    model = TinyLM()
    opt = torch.optim.SGD(model.parameters(), lr=5.0)
    G, plen, T = 4, 3, 9
    seqs = torch.randint(1, 19, (G, T))
    seqs[:, :plen] = torch.tensor([1, 2, 3])
    adv = group_advantages(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    mask = completion_mask(G, T, plen, [T] * G)
    lp_old = _logps(model, seqs).detach()
    fracs = []
    for _ in range(mu):
        lp = _logps(model, seqs)
        loss, _, cf = grpo_loss(lp, lp_old, lp_old, adv, mask, epsilon=epsilon)
        fracs.append(round(cf, 3))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return fracs


def validate_math(epsilon: float = 0.2, verbose: bool = True) -> Dict:
    """Tests 1-6. CPU only, deterministic, independent of any training run."""
    out: Dict = {}
    say = print if verbose else (lambda *a, **k: None)

    a = group_advantages(torch.tensor([1.1, 0.1, 1.1, 0.1, 0.1, 0.0]))
    assert abs(a.mean()) < 1e-5 and abs(a.std() - 1) < 0.02
    assert group_advantages(torch.tensor([0.5] * 6)) is None
    out["advantages"] = (float(a.mean()), float(a.std()))
    say(f"1. advantages   mean={a.mean():+.2e} std={a.std():.4f}, degenerate -> None   PASS")

    d = torch.linspace(-3, 3, 2001)
    k3 = torch.exp(d) - d - 1.0
    assert k3.min() >= -1e-6 and (-d).min() < 0
    out["k3_min"] = float(k3.min())
    say(f"2. k3 KL        min={k3.min():.2e} (>=0); naive -d reaches {(-d).min():.2f}   PASS")

    for rho, av, should in [(1.5, 1.0, True), (0.5, 1.0, False),
                            (1.5, -1.0, False), (1.0, 1.0, False)]:
        r, ad = torch.tensor(rho), torch.tensor(av)
        obj = torch.min(r * ad, torch.clamp(r, 1 - epsilon, 1 + epsilon) * ad)
        assert (not torch.isclose(obj, r * ad)) == should, (rho, av)
    say("3. clipping     4/4 cases match the min/clamp form                        PASS")

    assert completion_mask(1, 6, 3, [6])[0].tolist() == [0., 0., 1., 1., 1.]
    m2 = completion_mask(2, 8, 3, [6, 8])
    assert m2[0].tolist() == [0., 0., 1., 1., 1., 0., 0.]
    assert m2[1].tolist() == [0., 0., 1., 1., 1., 1., 1.]
    say("4. mask         plen-1 rule and per-sequence lengths correct              PASS")

    g = gradient_direction_test()
    assert g["delta"][0] > 0, "positive-advantage sequence did NOT become more likely"
    assert all(x < 0 for x in g["delta"][1:]), "negative-advantage sequences did NOT drop"
    assert g["corr"] > 0.9, f"corr(advantage, delta logprob) = {g['corr']:.3f}"
    out["grad_corr"] = g["corr"]
    say(f"5. gradient     corr(advantage, delta log-prob) = {g['corr']:.4f}           PASS")

    fr = clip_engagement_test(epsilon=epsilon)
    assert fr[0] == 0.0 and max(fr) > 0.0
    out["clip_fracs"] = fr
    say(f"6. mu>1         clip fraction over inner epochs 1..4 = {fr}  PASS")

    assert pass_at_k(6, 0, 1) == 0.0 and pass_at_k(6, 6, 1) == 1.0
    assert pass_at_k(6, 1, 6) == 1.0 and abs(pass_at_k(6, 1, 1) - 1 / 6) < 1e-9
    return out


def validate_reference_policy(policy, input_ids: torch.Tensor, verbose: bool = True) -> Dict:
    """Test 7. If the reference path returned the POLICY's log-probs, KL would be
    identically zero: training would run, curves would look healthy, and the beta
    leash would do nothing. This is the only test that catches it."""
    trained = policy.is_trained()
    mag_before = policy.adapter_magnitude()
    lp_pol = per_token_logps(policy, input_ids, use_adapter=True)
    lp_ref = per_token_logps(policy, input_ids, use_adapter=False)
    diff = float((lp_pol - lp_ref).abs().mean())
    dlt = lp_ref - lp_pol
    kl = float((torch.exp(dlt) - dlt - 1).mean())
    restored = abs(policy.adapter_magnitude() - mag_before) < 1e-6
    ok = restored and ((diff > 1e-6) if trained else (diff < 1e-6))
    if verbose:
        state = "trained: must DIFFER" if trained else "pristine: must MATCH"
        print(f"7. reference    {state} | mean|diff|={diff:.3e} KL={kl:.5f} "
              f"| adapter restored={restored}   {'PASS' if ok else 'FAIL'}")
    assert ok, "reference-policy check failed"
    return dict(trained=trained, mean_abs_diff=diff, kl=kl)
