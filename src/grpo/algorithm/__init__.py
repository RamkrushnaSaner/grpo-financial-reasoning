"""The GRPO algorithm: advantage estimators, losses, rollout primitives and the
update step. Nothing here knows about the task or the reward design."""
from .advantages import ADVANTAGES, group_advantages
from .losses import LOSSES, grpo_loss, k3_kl
from .rollout import (
    build_prompt_ids,
    completion_mask,
    generate_group,
    per_token_logps,
    true_lengths,
)
from .step import StepResult, grpo_step

__all__ = ["ADVANTAGES", "LOSSES", "StepResult", "build_prompt_ids", "completion_mask",
           "generate_group", "group_advantages", "grpo_loss", "grpo_step", "k3_kl",
           "per_token_logps", "true_lengths"]
