"""Rewards: a weighted sum of registered components, built from config.

    reward:
      components:
        - {name: correct_graded, weight: 1.0, params: {lam: 0.3, kappa: 0.2}}
        - {name: format, weight: 0.1}

Only differences between answers to the SAME question matter (advantages are
normalised within the group), so weights set the trade-off between
components, not the absolute scale.
"""
from .components import (
    REWARD_COMPONENTS,
    correct_binary,
    correct_gated,
    correct_graded,
    format_component,
    graded_score,
    partial_credit,
    partial_mass,
    validate_reward,
)
from .composite import CompositeReward, RewardComponent


def build_reward(reward_cfg, task) -> CompositeReward:
    """A ``CompositeReward`` from a ``RewardConfig`` for the given task."""
    comps = []
    for c in reward_cfg.components:
        factory = REWARD_COMPONENTS.get(c.name)
        try:
            fn = factory(task, **c.params)
        except TypeError as e:
            raise ValueError(f"reward component {c.name!r}: bad params {c.params}: {e}") from None
        comps.append(RewardComponent(c.name, fn, c.weight))
    return CompositeReward(comps, name="+".join(c.name for c in reward_cfg.components))


__all__ = ["REWARD_COMPONENTS", "CompositeReward", "RewardComponent", "build_reward",
           "correct_binary", "correct_gated", "correct_graded", "format_component",
           "graded_score", "partial_credit", "partial_mass", "validate_reward"]
