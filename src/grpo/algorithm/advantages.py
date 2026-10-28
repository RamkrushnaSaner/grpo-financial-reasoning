"""Advantage estimators: turn a group's rewards into per-answer advantages.

An estimator takes the G rewards for one question and returns G advantages,
or ``None`` when the group carries no learning signal (the step is skipped).
Register a variant with ``@ADVANTAGES.register("name")`` and select it with
``algorithm.advantage``.
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

from ..registry import Registry

ADVANTAGES: Registry = Registry("advantage estimator")


@ADVANTAGES.register("group_std")
def group_advantages(rewards: Union[torch.Tensor, Sequence[float]],
                     eps: float = 1e-4) -> Optional[torch.Tensor]:
    """GRPO's estimator: A_i = (r_i - mean) / std, within one question's group.

    The group mean is a Monte-Carlo estimate of the policy's expected reward on
    this question: an unbiased baseline that costs nothing, which is why GRPO
    needs no critic. Returns None for a degenerate group (zero spread), which
    carries no information. ``std`` is the unbiased (N-1) estimate.

    Known property (Liu et al. 2025, "Dr. GRPO"): dividing by the group std
    gives a group whose rewards differ only slightly the same update size as
    a group with right and wrong answers.
    """
    r = rewards if torch.is_tensor(rewards) else torch.tensor(rewards, dtype=torch.float32)
    s = r.std()
    if s < 1e-6:
        return None
    return (r - r.mean()) / (s + eps)
