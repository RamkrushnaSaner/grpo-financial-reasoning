"""Synthetic financial reasoning with numerically verifiable answers.

Four templates (growth, margin, sum of quarters, one-year projection), one or
two arithmetic steps each. Because every answer is a number, the reward is a
program rather than a learned model: the RLVR setting.

Dataset versions
----------------
``version: 1`` is the generator that produced the reported run (Run 6),
kept byte-for-byte so that run stays reproducible. It has a labelling bug:
the growth label is computed from the exact growth rate, but the question
shows the new revenue rounded to one decimal, so the label is often not what
the shown numbers give (63 of 104 growth questions in the training pool, 11
of the first 60 held-out questions). Example: "$1.2B compared to $1.0B" is
20%, labelled 17%.

``version: 2`` draws exactly the same questions (identical random draws and
text) and computes every label from the numbers the question shows, rounded
to two decimals. Use it for all new experiments.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

import numpy as np

from .base import TASKS, Sample, Task
from .financial_probes import PROBES

TOLERANCE: Dict[str, float] = {"percentage": 0.5, "dollar": 0.1, "ratio": 0.01, "integer": 0.0}

# ------------------------------------------------------------------ output contract
ANSWER_TAG = "Answer:"
ANSWER_RE = re.compile(r"Answer:\s*\$?(-?\d+(?:\.\d+)?)")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text: str, allow_fallback: bool = True) -> Optional[float]:
    """The last ``Answer: <number>`` wins. With ``allow_fallback`` the last number
    anywhere in the text is used when there is no tag. Never raises.

    The number pattern is ``-?\\d+(?:\\.\\d+)?``, not ``[\\d.]+``: the latter
    matches strings such as ``8.25.0.0`` and ``float()`` then raises inside a
    rollout loop.

    Measured: the fallback decouples correctness from format. In Run 6 the
    model learned to write "Final answer: 8.7", which the fallback parses, so
    format compliance fell 61.7% -> 20.0%. ``correct_gated`` uses
    ``allow_fallback=False``.
    """
    m = ANSWER_RE.findall(text)
    if m:
        return float(m[-1])
    if not allow_fallback:
        return None
    nums = NUMBER_RE.findall(text)
    return float(nums[-1]) if nums else None


def has_answer_tag(text: str) -> bool:
    return ANSWER_TAG in text


def verify_answer(predicted, ground_truth, answer_type, tolerance: Optional[Dict] = None) -> float:
    """1.0 if ``predicted`` matches within the type's absolute tolerance, else 0.0.
    None-safe and NaN-safe; never raises (an exception inside a rollout loop
    looks like a hang, not an error)."""
    if predicted is None or ground_truth is None:
        return 0.0
    try:
        p, g = float(predicted), float(ground_truth)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(p) and math.isfinite(g)):
        return 0.0
    if answer_type == "integer":
        return 1.0 if round(p) == round(g) else 0.0
    tol = (tolerance or TOLERANCE).get(answer_type, TOLERANCE.get(answer_type, 0.01))
    return 1.0 if abs(p - g) <= tol else 0.0


# ------------------------------------------------------------------ generator
def _make(rng: np.random.RandomState, version: int) -> Sample:
    """One question. Both versions make the same random draws in the same
    order, so they produce identical questions; only the labels differ."""
    qt = str(rng.choice(["growth", "margin", "comparison", "projection"]))
    if qt == "growth":                                                  # 2-step
        ro = round(rng.uniform(0.5, 10), 1)
        g = round(rng.uniform(-0.2, 0.5), 2)
        rn = round(ro * (1 + g), 1)
        answer = round(g * 100, 1) if version == 1 else round((rn - ro) / ro * 100, 2)
        return dict(context=f"In Q3 2024, the company reported revenue of ${rn}B, "
                            f"compared to ${ro}B in Q3 2023.",
                    question="What was the year-over-year revenue growth rate in percent?",
                    answer=answer, answer_type="percentage", steps=2, qtype=qt)
    if qt == "margin":                                                  # 1-step
        rev = round(rng.uniform(1, 15), 1)
        m = round(rng.uniform(0.05, 0.4), 2)
        profit = round(rev * m, 2)
        answer = round(m * 100, 1) if version == 1 else round(profit / rev * 100, 2)
        return dict(context=f"The company generated revenue of ${rev}B and operating "
                            f"profit of ${profit}B in the fiscal year.",
                    question="What was the operating margin in percent?",
                    answer=answer, answer_type="percentage", steps=1, qtype=qt)
    if qt == "comparison":                                              # 1-step
        q1, q2, q3 = (round(rng.uniform(1, 5), 1) for _ in range(3))
        answer = round(q1 + q2 + q3, 1 if version == 1 else 2)
        return dict(context=f"Quarterly revenues: Q1=${q1}B, Q2=${q2}B, Q3=${q3}B.",
                    question="What is the total revenue in $B for the first three quarters?",
                    answer=answer, answer_type="dollar", steps=1, qtype=qt)
    cur = round(rng.uniform(2, 10), 1)                                  # 2-step projection
    g = round(rng.uniform(0.05, 0.25), 2)
    pct = round(g * 100, 0)
    answer = round(cur * (1 + g), 1) if version == 1 else round(cur * (1 + pct / 100), 2)
    return dict(context=f"Current annual revenue is ${cur}B with a {pct:.0f}% annual growth rate.",
                question="What is the projected revenue in $B for next year?",
                answer=answer, answer_type="dollar", steps=2, qtype=qt)


def generate(n: int, seed: int, version: int) -> List[Sample]:
    if version not in (1, 2):
        raise ValueError(f"financial task version must be 1 or 2, got {version}")
    rng = np.random.RandomState(seed)
    return [_make(rng, version) for _ in range(n)]


# ------------------------------------------------------------------ prompt
# Measured: a 0.5B model ignores format INSTRUCTIONS but copies format
# DEMONSTRATIONS. Without the worked example the first run stalled (format 0%,
# 75-100% of steps degenerate). The prompt defines the contract the reward checks.
SYSTEM_PROMPT = ("You are a precise financial analyst. Reason step by step, then end "
                 "with your final answer on its own line as: Answer: <number>")
ONE_SHOT_USER = ("Revenue was $2.0B this year and $1.6B last year.\n\n"
                 "What was the revenue growth rate in percent?")
ONE_SHOT_ASSISTANT = "Growth = (2.0 - 1.6) / 1.6 = 0.4 / 1.6 = 0.25 = 25%\nAnswer: 25"


def build_messages(sample: Sample) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ONE_SHOT_USER},
        {"role": "assistant", "content": ONE_SHOT_ASSISTANT},
        {"role": "user", "content": f"{sample.get('context', '')}\n\n{sample['question']}"},
    ]


# ------------------------------------------------------------------ task
@TASKS.register("financial")
class FinancialTask(Task):
    name = "financial"

    def __init__(self, size: int = 500, seed: int = 42, train_pool: int = 400, version: int = 2,
                 tolerance: Optional[Dict[str, float]] = None):
        self.version = version
        self.tolerance = {**TOLERANCE, **(tolerance or {})}
        self.samples = generate(size, seed, version)
        if not 0 < train_pool < size:
            raise ValueError("train_pool must be inside (0, size)")
        self.train_pool = train_pool

    @classmethod
    def from_config(cls, cfg) -> "FinancialTask":
        return cls(size=cfg.size, seed=cfg.seed, train_pool=cfg.train_pool,
                   version=cfg.version, tolerance=cfg.tolerance)

    def messages(self, sample: Sample) -> List[dict]:
        return build_messages(sample)

    def parse(self, text: str, strict: bool = False) -> Optional[float]:
        return extract_answer(text, allow_fallback=not strict)

    def verify(self, pred, sample: Sample) -> bool:
        return verify_answer(pred, sample.get("answer"), sample.get("answer_type"),
                             self.tolerance) == 1.0

    def follows_format(self, text: str) -> bool:
        return has_answer_tag(text)

    def subgroup(self, sample: Sample) -> str:
        return f"{sample['steps']}step"

    def probes(self) -> List[Sample]:
        return [dict(p) for p in PROBES]
