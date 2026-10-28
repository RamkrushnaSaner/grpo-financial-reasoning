"""Built-in reward components for tasks with a numeric answer.

A component factory receives the task and its config ``params`` and returns a
pure ``(text, sample) -> float``. Components use the task's parser and
verifier, so they work for any task whose samples carry a numeric
``"answer"``. Register a new one with ``@REWARD_COMPONENTS.register("name")``.

    correct_binary  1 if the answer verifies, else 0 (parser fallback allowed)
    correct_graded  1 if it verifies, else partial credit lam*exp(-rel_err/kappa)
    correct_gated   as correct_graded, but credit only for the contracted format
    format          1 if the completion follows the output format
"""
from __future__ import annotations

import math
import warnings
from typing import Callable, Optional, Sequence

from ..data.base import Sample
from ..data.financial import verify_answer
from ..registry import Registry

REWARD_COMPONENTS: Registry = Registry("reward component")


# ------------------------------------------------------------------ the graded formula
def partial_credit(p: float, g: float, lam: float, kappa: float) -> float:
    """lam * exp(-|p - g| / (kappa * max(1, |g|))). Depends only on the relative
    error, so a 25% miss scores the same whether the truth is 39% or $8B."""
    return lam * math.exp(-abs(p - g) / (kappa * max(1.0, abs(g))))


def graded_score(pred: Optional[float], sample: Sample, verify: Callable, lam: float,
                 kappa: float) -> float:
    """1.0 if ``verify(pred, sample)``, partial credit otherwise, 0 if unusable.

    The finiteness guard matters: float('nan') parses fine, flows through
    exp(), and poisons every advantage in the group without an error.
    """
    gt = sample.get("answer")
    if pred is None or gt is None:
        return 0.0
    try:
        p, g = float(pred), float(gt)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(p) and math.isfinite(g)):
        return 0.0
    if verify(p, sample):
        return 1.0
    return partial_credit(p, g, lam, kappa)


def _check_lam(lam: float) -> None:
    if lam >= 0.5:
        warnings.warn(f"lam={lam} violates 2*lam < 1: two partial credits now outrank one "
                      "exact answer, so guessing can become reward-optimal. Only appropriate "
                      "for a deliberate lambda-violation experiment.", stacklevel=3)


# ------------------------------------------------------------------ components
@REWARD_COMPONENTS.register("correct_binary")
def correct_binary(task) -> Callable[[str, Sample], float]:
    return lambda t, s: float(task.verify(task.parse(t), s))


@REWARD_COMPONENTS.register("correct_graded")
def correct_graded(task, lam: float = 0.3, kappa: float = 0.2) -> Callable[[str, Sample], float]:
    _check_lam(lam)
    return lambda t, s: graded_score(task.parse(t), s, task.verify, lam, kappa)


@REWARD_COMPONENTS.register("correct_gated")
def correct_gated(task, lam: float = 0.3, kappa: float = 0.2) -> Callable[[str, Sample], float]:
    """Credit ONLY when the answer honours the output contract: the parser runs
    in strict mode, so a missing or malformed tag earns nothing. Format becomes
    a gate rather than an addend, and correctness can no longer starve the
    format signal. Proposed fix for the Run 6 format collapse; not yet
    validated by a training run."""
    _check_lam(lam)
    return lambda t, s: graded_score(task.parse(t, strict=True), s, task.verify, lam, kappa)


@REWARD_COMPONENTS.register("format")
def format_component(task) -> Callable[[str, Sample], float]:
    return lambda t, s: float(task.follows_format(t))


# ------------------------------------------------------------------ telemetry + checks
def partial_mass(task, texts: Sequence[str], sample: Sample, lam: float, kappa: float) -> float:
    """Mean partial credit among answers the task marks incorrect, always measured
    at a FIXED (lam, kappa). Measuring at each run's own training lam made the
    metric scale with the hyperparameter (it inflated a reported signal ~13x)."""
    vals = [graded_score(task.parse(t), sample, task.verify, lam, kappa)
            for t in texts if not task.is_correct(t, sample)]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _percentage_verify(p, sample) -> bool:
    return verify_answer(p, sample["answer"], sample.get("answer_type", "percentage")) == 1.0


def validate_reward(lam: float = 0.3, kappa: float = 0.2) -> None:
    """Six properties of the graded formula, each guarding a failure mode. Raises."""
    assert lam < 0.5, "SAFETY: two partial credits would outrank one exact answer"

    def g_(p, g, at="percentage"):
        return graded_score(p, {"answer": g, "answer_type": at}, _percentage_verify, lam, kappa)

    g, errs = 39.0, [1, 2, 5, 10, 20, 40]
    up = [g_(g + e, g) for e in errs]
    down = [g_(g - e, g) for e in errs]
    assert all(up[i] > up[i + 1] for i in range(len(up) - 1)), "not monotonic"
    assert all(abs(a - b) < 1e-12 for a, b in zip(up, down, strict=True)), "not symmetric"
    distinct = {round(g_(p, g), 6) for p in [39.0, 38.7, 37.33, 30.0, 23.0, 5.0, None]}
    assert len(distinct) >= 3, "degenerate: too few distinct values"
    assert abs(g_(39.0 * 1.25, 39.0) - g_(8.0 * 1.25, 8.0, "dollar")) < 1e-9, "not scale-free"
    for bad in [None, "abc", float("nan"), float("inf")]:
        v = g_(bad, 5.0, "dollar")
        assert math.isfinite(v) and 0.0 <= v <= 1.0, f"bad input {bad!r} -> {v}"
    assert g_(0.0, 0.0, "dollar") == 1.0
    assert g_(1e9, 5.0, "dollar") < 1e-6
