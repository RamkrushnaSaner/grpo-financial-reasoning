"""Policy losses. A loss maps per-token log-probabilities and per-answer
advantages to a scalar to minimise.

Signature every registered loss must accept:

    loss_fn(logp_new, logp_old, logp_ref, adv, mask, *, epsilon, beta)
        -> (loss tensor, mean KL (float), clip fraction (float))

    logp_*  [G, T-1]  log p(token t+1 | tokens <= t) under the current policy
                      (with grad), the rollout policy, and the frozen reference
    adv     [G]       one advantage per answer
    mask    [G, T-1]  1 on the answer's tokens, 0 on prompt and padding

Register a variant with ``@LOSSES.register("name")``; select it with
``algorithm.loss``.
"""
from __future__ import annotations

from typing import Tuple

import torch

from ..registry import Registry

LOSSES: Registry = Registry("loss")


def k3_kl(logp_ref: torch.Tensor, logp_new: torch.Tensor) -> torch.Tensor:
    """Per-token KL estimate e^d - d - 1 with d = logp_ref - logp_new (Schulman's
    k3). Unlike the naive -d it is always >= 0: a negative KL estimate would
    reward drifting away from the reference."""
    delta = logp_ref - logp_new
    return torch.exp(delta) - delta - 1.0


@LOSSES.register("grpo")
def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor,
              adv: torch.Tensor, mask: torch.Tensor, epsilon: float = 0.2,
              beta: float = 0.04) -> Tuple[torch.Tensor, float, float]:
    """The GRPO objective (DeepSeekMath), negated for minimisation:

        J = mean_i [ 1/|o_i| sum_t ( min(rho*A_i, clip(rho, 1-eps, 1+eps)*A_i) - beta*KL ) ]

    with rho = exp(logp_new - logp_old). At mu = 1 and no dropout, rho == 1
    exactly, the clip never binds, and the gradient is A_i * grad log pi plus
    the KL pull. The per-answer 1/|o_i| normalisation is the original GRPO
    form; it has a known length bias (Liu et al. 2025).
    """
    ratio = torch.exp(logp_new - logp_old)
    s1 = ratio * adv.unsqueeze(1)
    s2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv.unsqueeze(1)
    kl = k3_kl(logp_ref, logp_new)
    per_token = torch.min(s1, s2) - beta * kl
    per_resp = (per_token * mask).sum(1) / mask.sum(1).clamp(min=1)   # length-normalise
    loss = -per_resp.mean()                                            # ascend J
    with torch.no_grad():
        denom = mask.sum().clamp(min=1)
        clip_frac = (((s2 < s1) & (mask > 0)).float().sum() / denom).item()
        kl_val = ((kl * mask).sum() / denom).item()
    return loss, kl_val, clip_frac
