"""A reward is a weighted sum of named components, with per-component telemetry
and fault isolation.

* ablation = remove or re-weight a component in the config; never edit code
* the per-component breakdown makes effects attributable and shows saturation
* a component that raises or returns NaN scores 0 and is counted, instead of
  halting training (an exception inside a rollout loop looks like a hang)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

ComponentFn = Callable[[str, dict], float]


@dataclass
class RewardComponent:
    """One named signal. ``fn`` must be pure: (completion text, sample) -> float."""
    name: str
    fn: ComponentFn
    weight: float
    enabled: bool = True


class CompositeReward:
    def __init__(self, components: List[RewardComponent], name: str = "reward"):
        self.name = name
        self.components = components
        self.failures: Dict[str, int] = {c.name: 0 for c in components}

    @property
    def active(self) -> List[RewardComponent]:
        return [c for c in self.components if c.enabled]

    def score(self, text: str, sample: dict) -> Tuple[float, Dict[str, float]]:
        breakdown, total = {}, 0.0
        for c in self.active:
            try:
                v = float(c.fn(text, sample))
                if not math.isfinite(v):
                    raise ValueError("non-finite reward")
            except Exception:
                self.failures[c.name] += 1      # never propagate: silent-halt risk
                v = 0.0
            breakdown[c.name] = v
            total += c.weight * v
        return total, breakdown

    def score_group(self, texts: List[str], sample: dict) -> Tuple[List[float], List[Dict]]:
        totals, rows = [], []
        for t in texts:
            tot, bd = self.score(t, sample)
            totals.append(tot)
            rows.append(bd)
        return totals, rows

    def component_variance(self, rows: List[Dict[str, float]]) -> Dict[str, float]:
        """Variance WITHIN a group. ~0 => the component contributes no gradient
        for this question (advantages are invariant to a shared shift)."""
        return {c.name: float(np.var([r[c.name] for r in rows])) for c in self.active}

    def describe(self) -> Dict[str, float]:
        return {c.name: c.weight for c in self.active}
