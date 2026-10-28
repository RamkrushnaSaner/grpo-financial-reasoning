"""The task interface: everything domain-specific that training and evaluation
need, behind one small class.

A task owns its questions, its prompt, its answer parser and its verifier.
The GRPO step never looks inside a sample; reward components and the
evaluator call ``parse`` / ``verify`` / ``follows_format`` on the task. A new
domain is one ``Task`` subclass registered in ``TASKS``; nothing in
``grpo.algorithm`` or ``grpo.training`` changes.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..registry import Registry

Sample = Dict  # a plain dict; tasks define their own fields ("question", "answer", ...)

TASKS: Registry = Registry("task")


def build_task(cfg) -> "Task":
    """The task named by ``cfg.name`` (a ``TaskConfig``)."""
    return TASKS.get(cfg.name).from_config(cfg)


class Task:
    """Base class. Subclasses fill ``samples`` and ``train_pool`` and implement
    ``messages``, ``parse``, ``verify`` and ``follows_format``."""

    name: str = "task"
    samples: List[Sample]
    train_pool: int

    @classmethod
    def from_config(cls, cfg) -> "Task":
        """Build from a ``TaskConfig``. Override if the task takes other settings."""
        raise NotImplementedError

    # ------------------------------------------------------------ prompt
    def messages(self, sample: Sample) -> List[dict]:
        """Chat messages for one question (system, examples, user)."""
        raise NotImplementedError

    # ------------------------------------------------------------ scoring
    def parse(self, text: str, strict: bool = False) -> Optional[float]:
        """The answer extracted from a completion, or None. ``strict`` accepts
        only the contracted output format (no fallbacks)."""
        raise NotImplementedError

    def verify(self, pred: Optional[float], sample: Sample) -> bool:
        """Whether a parsed answer matches the sample's ground truth."""
        raise NotImplementedError

    def follows_format(self, text: str) -> bool:
        raise NotImplementedError

    def is_correct(self, text: str, sample: Sample) -> bool:
        """The evaluation metric: held-out accuracy and the arm-independent
        ``exact_rate`` telemetry use this, never the training reward."""
        return self.verify(self.parse(text), sample)

    def subgroup(self, sample: Sample) -> str:
        """Stratification key for evaluation breakdowns (e.g. ``"1step"``)."""
        return "all"

    def probes(self) -> List[Sample]:
        """Hand-written questions for the base-vs-trained probe (optional)."""
        return []

    # ------------------------------------------------------------ splits
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Sample:
        return self.samples[i]

    def eval_indices(self, n: int) -> List[int]:
        """The first ``n`` held-out questions. Fixed, so before/after are paired."""
        held = list(range(self.train_pool, len(self.samples)))
        if n > len(held):
            raise ValueError(f"asked for {n} held-out questions, only {len(held)} exist")
        return held[:n]

    def schedule(self, steps: int, seed: int) -> List[int]:
        """A fixed sequence of training questions (drawn with replacement from the
        training pool), shared by every run with the same seed, so no run can win
        by drawing easier questions."""
        return [int(i) for i in np.random.RandomState(seed).randint(0, self.train_pool, size=steps)]
