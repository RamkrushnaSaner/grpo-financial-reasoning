"""One GRPO update for one question.

    1 rollouts      sample G answers from the current policy
    2 rewards       score each answer (the reward function is given)
    3 advantages    group-relative; skip the step if the group carries no signal
    4 mask          keep the answer tokens only
    5 log-probs     reference (adapter zeroed) and rollout ("old") policies
    6 update        mu inner epochs of loss -> backward -> clip -> optimizer step

The step knows nothing about the task: it receives the prompt ids, the sample
(passed through to the reward function) and the reward function. Task-level
telemetry is computed by the caller from the returned texts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .advantages import ADVANTAGES
from .losses import LOSSES
from .rollout import completion_mask, generate_group, per_token_logps, true_lengths


@dataclass
class StepResult:
    loss: Optional[float]               # None when the group was degenerate (skipped)
    kl: float = 0.0
    clip_frac: float = 0.0
    ratio_dev: float = 0.0              # max |rho - 1| at the first inner epoch: exactly 0 at
                                        # mu=1 without dropout; > 0 means the identity broke
    grad_norm: Optional[float] = None   # pre-clipping gradient norm of the last inner epoch
    texts: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    breakdown: List[Dict[str, float]] = field(default_factory=list)
    completion_lengths: List[int] = field(default_factory=list)

    @property
    def skipped(self) -> bool:
        return self.loss is None


def grpo_step(policy, prompt_ids: torch.Tensor, sample: dict, reward, optimizer,
              cfg) -> StepResult:
    """``cfg`` is the full ``Config`` (algorithm, sampling and training sections)."""
    algo = cfg.algorithm
    advantage_fn = ADVANTAGES.get(algo.advantage)
    loss_fn = LOSSES.get(algo.loss)

    seqs, plen = generate_group(policy, prompt_ids, algo.group_size, cfg.sampling)    # 1
    lengths = true_lengths(seqs, plen, policy.stop_token_ids())
    texts = [policy.tokenizer.decode(seqs[i, plen:lengths[i]], skip_special_tokens=True)
             for i in range(algo.group_size)]

    totals, rows = reward.score_group(texts, sample)                                   # 2
    result = StepResult(loss=None, texts=texts, rewards=totals, breakdown=rows,
                        completion_lengths=[L - plen for L in lengths])

    adv = advantage_fn(torch.tensor(totals, device=policy.device, dtype=torch.float32))  # 3
    if adv is None:
        return result                                                                   # skip

    mask = completion_mask(algo.group_size, seqs.shape[1], plen, lengths, policy.device)  # 4
    logp_ref = per_token_logps(policy, seqs, use_adapter=False)                         # 5
    logp_old = per_token_logps(policy, seqs, use_adapter=True).detach()

    policy.model.train()
    for epoch in range(algo.mu):                                                        # 6
        logp_new = per_token_logps(policy, seqs, use_adapter=True, requires_grad=True)
        if epoch == 0:
            # Direct check of the rho == 1 identity; the clip fraction is only an
            # indirect symptom (small deviations never cross the +/-epsilon bound).
            with torch.no_grad():
                dev = (torch.exp(logp_new - logp_old) - 1.0).abs()
                result.ratio_dev = float(dev[mask > 0].max()) if mask.any() else 0.0
        loss, result.kl, result.clip_frac = loss_fn(logp_new, logp_old, logp_ref, adv, mask,
                                                    epsilon=algo.epsilon, beta=algo.beta)
        optimizer.zero_grad()
        loss.backward()
        result.grad_norm = float(torch.nn.utils.clip_grad_norm_(
            policy.trainable_parameters(), cfg.training.grad_clip))
        optimizer.step()
        result.loss = loss.item()
    return result
