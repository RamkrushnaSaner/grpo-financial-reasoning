"""The training loop: one GRPO step per scheduled question, with telemetry,
checkpointing and a fingerprinted cache.

Files written to the run directory:

    adapter.pt      trained LoRA tensors
    train.json      config fingerprint, history (one list per metric), summary
    metrics.jsonl   one record per step, streamed while training

A finished run is reused only if its fingerprint matches the current training
settings AND its adapter file exists. Restoring metrics without weights once
left the model at base state while every label said "GRPO".
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .algorithm import StepResult, build_prompt_ids, grpo_step
from .config import fingerprint_of
from .rewards import CompositeReward, build_reward, partial_mass
from .stats import pass_at_k, window_mean
from .tracking import MetricsWriter, get_logger
from .utils import load_json, save_json, set_seed

# Sections whose settings change what training produces.
TRAINING_SECTIONS = ("experiment.seed", "model", "task", "reward", "algorithm", "sampling",
                     "training", "evaluation.telemetry_lam", "evaluation.telemetry_kappa")


def training_fingerprint(cfg) -> str:
    return fingerprint_of(cfg, TRAINING_SECTIONS)


def rollout_telemetry(task, res: StepResult, sample: dict, reward: CompositeReward, cfg) -> Dict:
    """Per-step measurements of the sampled group. Arm-independent: exact
    correctness is always the task's evaluation metric, and partial credit is
    always measured at the FIXED telemetry (lam, kappa), never at a run's own
    training settings, so runs with different rewards stay comparable."""
    exact = [task.is_correct(t, sample) for t in res.texts]
    n_correct = int(sum(exact))
    g = len(res.texts)
    return dict(
        exact_rate=float(np.mean(exact)),
        format_rate=float(np.mean([task.follows_format(t) for t in res.texts])),
        partial_mass=partial_mass(task, res.texts, sample, cfg.evaluation.telemetry_lam,
                                  cfg.evaluation.telemetry_kappa),
        reward_mean=float(np.mean(res.rewards)),
        reward_std=float(np.std(res.rewards)),
        n_distinct=len({round(r, 6) for r in res.rewards}),
        pass_at_1=pass_at_k(g, n_correct, 1),
        pass_at_g=pass_at_k(g, n_correct, g),
        mean_len=float(np.mean(res.completion_lengths)),
        comp_var=reward.component_variance(res.breakdown),
    )


def summarize_history(hist: Dict[str, list], window: int) -> Dict[str, float]:
    return dict(
        steps=len(hist["skipped"]),
        degenerate_rate=float(np.mean(hist["skipped"])),
        exact_first=window_mean(hist["exact_rate"], True, window),
        exact_last=window_mean(hist["exact_rate"], False, window),
        format_first=window_mean(hist["format_rate"], True, window),
        format_last=window_mean(hist["format_rate"], False, window),
        kl_last=window_mean(hist["kl"], False, window),
        clip_frac_mean=float(np.mean(hist["clip_frac"])),
        ratio_dev_max=float(np.max(hist["ratio_dev"])),
    )


def train(policy, task, cfg, run_dir, log=None) -> dict:
    """Train (or restore) the adapter for ``cfg`` in ``run_dir``. Returns train.json."""
    log = log or get_logger().info
    run_dir = Path(run_dir)
    adapter_path, result_path = run_dir / "adapter.pt", run_dir / "train.json"
    fp = training_fingerprint(cfg)

    if result_path.exists():
        cached = load_json(result_path)
        if cached.get("fingerprint") == fp and policy.load_adapter(str(adapter_path)):
            log(f"training cached: metrics loaded AND weights restored "
                f"(adapter magnitude {policy.adapter_magnitude():.4f})")
            return cached
        log("training cache stale or adapter missing: retraining "
            "(otherwise the model would silently stay at base state)")

    reward = build_reward(cfg.reward, task)
    a = cfg.algorithm
    log(f"train: reward={reward.describe()} G={a.group_size} adv={a.advantage} loss={a.loss} "
        f"eps={a.epsilon} beta={a.beta} mu={a.mu} lr={cfg.training.lr} "
        f"dropout={cfg.model.lora_dropout} seed={cfg.experiment.seed}")
    policy.reset()
    set_seed(cfg.experiment.seed)
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=cfg.training.lr)
    metrics = MetricsWriter(run_dir / "metrics.jsonl")
    w = cfg.training.window
    steps = cfg.training.steps

    hist, samples = defaultdict(list), {}
    t0 = time.time()
    for step, qi in enumerate(task.schedule(steps, cfg.training.schedule_seed)):
        sample = task[qi]
        prompt_ids = build_prompt_ids(policy, task.messages(sample))
        res = grpo_step(policy, prompt_ids, sample, reward, optimizer, cfg)
        tel = rollout_telemetry(task, res, sample, reward, cfg)
        record = dict(step=step + 1, question=qi, skipped=float(res.skipped), loss=res.loss,
                      grad_norm=res.grad_norm, kl=res.kl, clip_frac=res.clip_frac,
                      ratio_dev=res.ratio_dev, **{k: v for k, v in tel.items() if k != "comp_var"},
                      **{f"var_{k}": v for k, v in tel["comp_var"].items()})
        metrics.write(record)
        for k, v in record.items():
            if k not in ("step", "question"):
                hist[k].append(v)
        if step in (0, steps // 2, steps - 1):
            samples[str(step)] = res.texts[0]
        if (step + 1) % w == 0:
            s = slice(-w, None)
            log(f"  step {step + 1:3d} | exact {np.mean(hist['exact_rate'][s]):.3f} | "
                f"pass@G {np.mean(hist['pass_at_g'][s]):.3f} | "
                f"fmt {np.mean(hist['format_rate'][s]):.3f} | "
                f"skip {np.mean(hist['skipped'][s]):.0%} | "
                f"KL {np.mean(hist['kl'][s]):.4f} | clip {np.mean(hist['clip_frac'][s]):.4f} | "
                f"|rho-1| {np.max(hist['ratio_dev'][s]):.1e} | {(time.time() - t0) / 60:.1f}m")

    minutes = (time.time() - t0) / 60
    policy.save_adapter(str(adapter_path))
    log(f"training done in {minutes:.1f} min | adapter saved "
        f"(magnitude {policy.adapter_magnitude():.4f})")
    if any(reward.failures.values()):
        log(f"WARNING reward-component failures: {reward.failures}")
    result = dict(fingerprint=fp, config=cfg.to_dict(), minutes=minutes, samples=samples,
                  failures=reward.failures, summary=summarize_history(hist, w),
                  history={k: list(v) for k, v in hist.items()})
    save_json(result, result_path)
    return result


def load_trained(policy, run_dir) -> Optional[dict]:
    """Restore a finished run's adapter into ``policy`` (None if there is none)."""
    run_dir = Path(run_dir)
    if not (run_dir / "train.json").exists():
        return None
    if not policy.load_adapter(str(run_dir / "adapter.pt")):
        return None
    return load_json(run_dir / "train.json")
