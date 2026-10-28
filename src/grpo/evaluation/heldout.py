"""Held-out evaluation: greedy accuracy, format compliance and pass@k, with one
record per question so before/after comparisons can be paired.

Greedy for evaluation, sampled for training. Training samples to explore;
evaluation takes the argmax to measure what the policy believes. (One early
run showed sampled accuracy 5.8% -> 20% while greedy stayed flat: the
"improvement" was luck in which questions were drawn.)
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch

from ..algorithm import build_prompt_ids
from ..stats import pass_at_k
from ..utils import set_seed


def evaluate(policy, task, cfg, *, n: Optional[int] = None, k: Optional[int] = None,
             show: Optional[int] = None, log: Callable = print) -> dict:
    """Evaluate the policy on the first ``n`` held-out questions.

    Returns summary metrics (``acc_overall``, ``acc_<subgroup>``,
    ``n_<subgroup>``, ``format``, ``mean_len``, ``pass@1``, ``pass@k``) plus
    ``records``: one dict per question (index, subgroup, correct, format,
    tokens, text and, with k > 0, n_correct / pass@1 / pass@k).

    Sampling for pass@k is seeded with ``experiment.seed`` at the start, so the
    sampled numbers do not depend on how much randomness was used before.
    """
    n = cfg.evaluation.n if n is None else n
    k = cfg.evaluation.k if k is None else k
    show = cfg.evaluation.show if show is None else show
    if k:
        set_seed(cfg.experiment.seed)
    policy.model.eval()
    tok, sp = policy.tokenizer, cfg.sampling
    records = []
    t0 = time.perf_counter()
    for j, idx in enumerate(task.eval_indices(n)):
        s = task[idx]
        pids = build_prompt_ids(policy, task.messages(s))
        plen = pids.shape[1]
        with torch.no_grad():
            out = policy.model.generate(pids, max_new_tokens=sp.max_new_tokens, do_sample=False,
                                        pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, plen:], skip_special_tokens=True)
        rec = dict(index=int(idx), subgroup=task.subgroup(s),
                   correct=bool(task.is_correct(text, s)), format=bool(task.follows_format(text)),
                   tokens=int(out.shape[1] - plen), text=text)
        if k:
            with torch.no_grad():
                sm = policy.model.generate(pids, max_new_tokens=sp.max_new_tokens, do_sample=True,
                                           temperature=sp.temperature, top_p=sp.top_p,
                                           num_return_sequences=k, pad_token_id=tok.pad_token_id)
            c = int(sum(task.is_correct(tok.decode(sm[i, plen:], skip_special_tokens=True), s)
                        for i in range(k)))
            rec.update(n_correct=c, pass_at_1=pass_at_k(k, c, 1), pass_at_k=pass_at_k(k, c, k))
        records.append(rec)
        if j < show:
            log(f"  [{s['question'][:44]}... gt={s.get('answer')}]\n  {text[:170]}")
    return summarize_records(records, k, eval_sec=time.perf_counter() - t0)


def summarize_records(records: list, k: int, eval_sec: float = 0.0) -> dict:
    by_group = defaultdict(list)
    for r in records:
        by_group[r["subgroup"]].append(r["correct"])
    out = {f"acc_{g}": float(np.mean(v)) for g, v in sorted(by_group.items())}
    out.update({f"n_{g}": len(v) for g, v in sorted(by_group.items())})
    out["acc_overall"] = float(np.mean([r["correct"] for r in records]))
    out["format"] = float(np.mean([r["format"] for r in records]))
    out["mean_len"] = float(np.mean([r["tokens"] for r in records]))
    out["n"] = len(records)
    out["k"] = k
    out["eval_sec"] = eval_sec
    if k:
        out["pass@1"] = float(np.mean([r["pass_at_1"] for r in records]))
        out[f"pass@{k}"] = float(np.mean([r["pass_at_k"] for r in records]))
    out["records"] = records
    return out
