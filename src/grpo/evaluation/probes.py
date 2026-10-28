"""Question-level probe: the same question through the BASE and the TRAINED
policy, side by side.

Fair by construction: one model, one prompt path, a seed reset before every
generation, identical decoding. The only difference is which LoRA tensors are
loaded, and the pristine snapshot (B = 0) IS the base model. The trained
adapter is always restored afterwards, even if generation raises.
"""
from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np
import torch

from ..algorithm import build_prompt_ids
from ..stats import pass_at_k
from ..utils import set_seed


def _generate(policy, pids, max_new_tokens, greedy, sampling, n=1):
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=policy.tokenizer.pad_token_id)
    if greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p,
                  num_return_sequences=n)
    with torch.no_grad():
        return policy.model.generate(pids, **kw)


def ask_both(policy, task, sample: dict, cfg, reward=None, greedy: bool = True,
             n_samples: int = 0, seed: int = 1234, max_new_tokens: Optional[int] = None) -> dict:
    """Run one question through BASE and TRAINED. ``reward`` (optional) is the
    training reward, shown per answer when the question has a ground truth."""
    trained = policy.snapshot()
    mnt = max_new_tokens or sample.get("max_new_tokens") or cfg.sampling.max_new_tokens
    pids = build_prompt_ids(policy, task.messages(sample))
    plen = pids.shape[1]
    has_gt = sample.get("answer") is not None
    tok = policy.tokenizer
    out = {}
    try:
        for label, weights in (("BASE", policy.initial_lora), ("TRAINED", trained)):
            policy.restore(weights)
            policy.model.eval()
            set_seed(seed)
            t0 = time.perf_counter()
            gen = _generate(policy, pids, mnt, greedy, cfg.sampling)
            sec = time.perf_counter() - t0
            text = tok.decode(gen[0, plen:], skip_special_tokens=True)
            r = dict(text=text, tokens=int(gen.shape[1] - plen), sec=sec, pred=task.parse(text),
                     fmt=bool(task.follows_format(text)), truncated=(gen.shape[1] - plen) >= mnt)
            if has_gt:
                r["exact"] = bool(task.is_correct(text, sample))
                if reward is not None:
                    r["reward"], r["reward_parts"] = reward.score(text, sample)
            if n_samples:
                set_seed(seed)
                sm = _generate(policy, pids, mnt, False, cfg.sampling, n=n_samples)
                texts = [tok.decode(sm[i, plen:], skip_special_tokens=True)
                         for i in range(n_samples)]
                r["fmt_rate"] = float(np.mean([task.follows_format(t) for t in texts]))
                if has_gt:
                    c = int(sum(task.is_correct(t, sample) for t in texts))
                    r.update(n_correct=c, n_samples=n_samples, pass_1=pass_at_k(n_samples, c, 1),
                             pass_k=pass_at_k(n_samples, c, n_samples))
            out[label] = r
    finally:
        policy.restore(trained)
    return out


def render_ab(sample: dict, out: dict, W: int = 84) -> str:
    L = ["=" * W, " QUESTION", "=" * W]
    if sample.get("context"):
        L.append(f" {sample['context']}")
    L.append(f" {sample['question']}")
    if sample.get("answer") is not None:
        L.append(f" ground truth: {sample['answer']} ({sample.get('answer_type')})")
    for label, title in (("BASE", "BASE     (pristine adapter = original model)"),
                         ("TRAINED", "TRAINED  (GRPO adapter)")):
        r = out[label]
        L += ["", "=" * W, f" {title}", "=" * W, (r["text"].strip() or "(empty output)")[:900],
              "-" * W]
        line = (f" tokens {r['tokens']:<4} time {r['sec']:5.2f}s  "
                f"format {'YES' if r['fmt'] else 'NO '}  parsed {str(r['pred']):>10}")
        if r["truncated"]:
            line += "  [TRUNCATED]"
        L.append(line)
        if "exact" in r:
            verdict = "CORRECT" if r["exact"] else "WRONG"
            if "reward" in r:
                parts = " + ".join(f"{k} {v:.3f}" for k, v in r["reward_parts"].items())
                L.append(f" reward: {parts} -> TOTAL {r['reward']:.3f} ({verdict})")
            else:
                L.append(f" {verdict}")
        if "pass_k" in r:
            L.append(f" over {r['n_samples']} samples: {r['n_correct']} correct | "
                     f"pass@1 {r['pass_1']:.2f} pass@{r['n_samples']} {r['pass_k']:.2f} | "
                     f"format rate {r['fmt_rate']:.0%}")
    return "\n".join(L)


def run_probes(policy, task, questions: List[dict], cfg, reward=None, greedy: bool = True,
               n_samples: int = 0, seed: int = 1234, log: Callable = print,
               verbose: bool = True) -> List[dict]:
    rows = []
    for q in questions:
        o = ask_both(policy, task, q, cfg, reward=reward, greedy=greedy, n_samples=n_samples,
                     seed=seed)
        if verbose:
            log(render_ab(q, o))
        rows.append(dict(id=q["id"], capability=q.get("capability", ""), group=q["id"][0],
                         expect=q.get("expect", ""),
                         base_fmt=o["BASE"]["fmt"], trained_fmt=o["TRAINED"]["fmt"],
                         base_tokens=o["BASE"]["tokens"], trained_tokens=o["TRAINED"]["tokens"],
                         base_exact=o["BASE"].get("exact"), trained_exact=o["TRAINED"].get("exact"),
                         base_text=o["BASE"]["text"], trained_text=o["TRAINED"]["text"]))
    return rows


GROUP_NAMES = (("A", "in-distribution"), ("B", "near-distribution"),
               ("C", "out-of-distribution"), ("D", "adversarial"))


def summarize_probes(rows: List[dict], W: int = 100) -> str:
    L = ["=" * W, " PROBE SUMMARY", "=" * W,
         f" {'id':<5}{'capability':<26}{'expected':<14}{'format':<10}{'length':<14}correct",
         " " + "-" * (W - 4)]
    for r in rows:
        ex = ("n/a" if r["base_exact"] is None else
              f"{'Y' if r['base_exact'] else 'n'} -> {'Y' if r['trained_exact'] else 'n'}")
        fmt = "{} -> {}".format(int(r["base_fmt"]), int(r["trained_fmt"]))
        ln = "{} -> {}".format(r["base_tokens"], r["trained_tokens"])
        L.append(f" {r['id']:<5}{r['capability'][:25]:<26}{r['expect']:<14}{fmt:<10}{ln:<14}{ex}")
    L += ["", " BY EXPECTATION GROUP"]
    for grp, name in GROUP_NAMES:
        g = [r for r in rows if r["group"] == grp]
        if not g:
            continue
        sc = [r for r in g if r["base_exact"] is not None]
        acc = (f"correct {sum(r['base_exact'] for r in sc)}/{len(sc)} -> "
               f"{sum(r['trained_exact'] for r in sc)}/{len(sc)}") if sc else "not scored"
        L.append(f" {grp} {name:<22}{acc:<26}format "
                 f"{np.mean([r['base_fmt'] for r in g]):.0%} -> "
                 f"{np.mean([r['trained_fmt'] for r in g]):.0%}")
    sc = [r for r in rows if r["base_exact"] is not None]
    if sc:
        L.append(f"\n overall scored: {sum(r['base_exact'] for r in sc)}/{len(sc)} -> "
                 f"{sum(r['trained_exact'] for r in sc)}/{len(sc)}")
    L.append(" Read each row against its EXPECTED column. A handful of questions at one seed "
             "shows WHAT changed, not HOW MUCH.")
    return "\n".join(L)
