"""Experiment management: one directory per run, cached stages, and the end-to-end
pipeline.

    <output_dir>/<name>/
      config.yaml       the fully resolved config that produced this run
      manifest.json     fingerprints, versions, git commit, command, timestamps
      run.log           human-readable log
      metrics.jsonl     one record per training step
      train.json        training history and summary
      adapter.pt        trained LoRA weights
      eval_after.json   held-out evaluation of the trained policy (with records)
      comparison.json   paired before/after statistics
      probes.json       base-vs-trained answers to the probe questions
      report.md         everything above in one page
      curves.png, before_after.png
    <output_dir>/_baselines/<fingerprint>.json
                        base-model evaluations, shared by every run whose
                        model, task, sampling and evaluation settings match

Every stage is reused only when the settings that can change its output are
unchanged (fingerprints), so re-running a config resumes instead of repeating.
"""
from __future__ import annotations

import datetime
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import yaml

from .algorithm import build_prompt_ids
from .config import Config, fingerprint_of, load_config
from .data import build_task
from .evaluation import (
    compare,
    evaluate,
    format_comparison,
    plot_before_after,
    plot_training,
    render_report,
    run_probes,
    summarize_probes,
)
from .model import load_policy
from .rewards import build_reward
from .stats import window_mean
from .tracking import get_logger
from .training import train, training_fingerprint
from .utils import file_sha256, load_json, save_json, set_seed
from .validation import validate_math, validate_reference_policy

# Settings that change a held-out evaluation of the BASE model.
EVAL_KEYS = ("model.loader", "model.name", "model.load_in_4bit", "task", "sampling",
             "evaluation.n", "evaluation.k", "experiment.seed")
# Settings that change how the policy is constructed (a loaded policy can be reused across
# configs that agree on these).
MODEL_KEYS = ("model",)


def run_dir_for(cfg: Config) -> Path:
    return Path(cfg.experiment.output_dir) / cfg.experiment.name


def cached_json(path: Path, fingerprint: str, compute: Callable[[], dict], log=print) -> dict:
    """Return the stored result if its fingerprint matches, else compute and store."""
    if path.exists():
        stored = load_json(path)
        if stored.get("_fingerprint") == fingerprint:
            log(f"cache hit: {path}")
            return stored
        log(f"cache stale (settings changed): recomputing {path}")
    result = compute()
    result["_fingerprint"] = fingerprint
    save_json(result, path)
    return result


def _git_state() -> Optional[Dict[str, object]]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return dict(commit=commit, dirty=dirty)
    except Exception:
        return None


def _versions() -> Dict[str, Optional[str]]:
    from importlib.metadata import PackageNotFoundError, version
    out: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def write_manifest(run_dir: Path, cfg: Config, config_path: Optional[str]) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    manifest = dict(
        name=cfg.experiment.name,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        config_path=config_path, command=" ".join(sys.argv),
        fingerprints=dict(training=training_fingerprint(cfg),
                          evaluation=fingerprint_of(cfg, EVAL_KEYS)),
        versions=_versions(), git=_git_state(), platform=platform.platform())
    save_json(manifest, run_dir / "manifest.json")
    return manifest


def run_pipeline(cfg: Config, *, config_path: Optional[str] = None, probes: Optional[bool] = None,
                 policy=None) -> dict:
    """validate -> BEFORE -> train -> reference check -> AFTER -> compare -> probe -> report."""
    run_dir = run_dir_for(cfg)
    say = get_logger(log_file=run_dir / "run.log").info
    write_manifest(run_dir, cfg, config_path)
    say(f"run '{cfg.experiment.name}' -> {run_dir}/")

    say("== 1/6 mathematical validation (CPU) ==")
    val = validate_math(cfg.algorithm.epsilon, verbose=False)
    say(f"   gradient-direction corr {val['grad_corr']:.4f}; "
        f"clip fractions {val['clip_fracs']}  PASS")

    set_seed(cfg.experiment.seed)
    policy = policy if policy is not None else load_policy(cfg.model)
    task = build_task(cfg.task)
    eval_fp = fingerprint_of(cfg, EVAL_KEYS)

    say("== 2/6 BEFORE: base model ==")

    def measure_before():
        policy.reset()
        assert not policy.is_trained(), "the baseline must be measured on the pristine adapter"
        return evaluate(policy, task, cfg, log=say)
    before = cached_json(Path(cfg.experiment.output_dir) / "_baselines" / f"{eval_fp}.json",
                         eval_fp, measure_before, say)

    say("== 3/6 GRPO training ==")
    train_result = train(policy, task, cfg, run_dir, say)
    ref = validate_reference_policy(policy, build_prompt_ids(policy, task.messages(task[0])),
                                    verbose=False)
    say(f"   reference vs trained policy: mean |diff log p| {ref['mean_abs_diff']:.3e} PASS")

    say("== 4/6 AFTER: trained model ==")
    if not policy.is_trained():
        raise RuntimeError("adapter is untrained: AFTER would equal BEFORE")
    after_fp = fingerprint_of(cfg, EVAL_KEYS, extra=dict(
        training=train_result["fingerprint"], adapter=file_sha256(run_dir / "adapter.pt")))
    after = cached_json(run_dir / "eval_after.json", after_fp,
                        lambda: evaluate(policy, task, cfg, log=say), say)

    say("== 5/6 comparison ==")
    comp = compare(before, after)
    comparison_text = format_comparison(comp)
    save_json(comp, run_dir / "comparison.json")
    say("\n" + comparison_text)
    plot_training(train_result["history"], str(run_dir / "curves.png"), cfg.algorithm.group_size)
    plot_before_after(before, after, str(run_dir / "before_after.png"),
                      f"Before vs after {cfg.training.steps} GRPO steps (n={after['n']})")

    probes_text = None
    use_probes = cfg.evaluation.probes if probes is None else probes
    if use_probes and task.probes():
        say("== 6/6 question-level probe ==")
        rows = run_probes(policy, task, task.probes(), cfg, reward=build_reward(cfg.reward, task),
                          n_samples=cfg.evaluation.k, log=say)
        save_json(rows, run_dir / "probes.json")
        probes_text = summarize_probes(rows)
        say("\n" + probes_text)

    (run_dir / "report.md").write_text(render_report(
        name=cfg.experiment.name, config_path=config_path, fingerprint=train_result["fingerprint"],
        validation=val, reference=ref, train=train_result, comparison=comparison_text,
        probes=probes_text))
    say(f"artifacts -> {run_dir}/ (report.md, comparison.json, metrics.jsonl, curves.png, ...)")
    return dict(run_dir=str(run_dir), before=before, after=after, comparison=comp,
                train=train_result, policy=policy)


# ------------------------------------------------------------------ several runs
def summarize_runs(run_dirs: Sequence[Path]) -> List[dict]:
    rows = []
    for d in map(Path, run_dirs):
        if not (d / "train.json").exists() or not (d / "comparison.json").exists():
            rows.append(dict(run=d.name, status="incomplete"))
            continue
        tr, comp = load_json(d / "train.json"), load_json(d / "comparison.json")
        acc = next((r for r in comp["rows"] if r["key"] == "acc_overall"), {})
        fmt = next((r for r in comp["rows"] if r["key"] == "format"), {})
        h = tr["history"]
        w = tr["config"]["training"]["window"]
        rows.append(dict(
            run=d.name, status="done", acc_before=acc.get("before"), acc_after=acc.get("after"),
            p_value=acc.get("p_value"), format_after=fmt.get("after"),
            degenerate=tr["summary"]["degenerate_rate"],
            distinct=float(sum(h["n_distinct"]) / len(h["n_distinct"])),
            exact_gain=(window_mean(h["exact_rate"], False, w)
                        - window_mean(h["exact_rate"], True, w)),
            kl_last=tr["summary"]["kl_last"]))
    return rows


def format_summary(rows: List[dict]) -> str:
    head = (f"{'run':<22}{'base':>7}{'after':>8}{'p':>8}{'format':>8}{'degen':>7}{'distinct':>9}"
            f"{'gain':>8}{'KL':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        if r["status"] != "done":
            lines.append(f"{r['run']:<22}  ({r['status']})")
            continue
        p = "n/a" if r["p_value"] is None else f"{r['p_value']:.2g}"
        lines.append(f"{r['run']:<22}{r['acc_before']:>7.1%}{r['acc_after']:>8.1%}{p:>8}"
                     f"{r['format_after']:>8.1%}{r['degenerate']:>7.1%}{r['distinct']:>9.2f}"
                     f"{r['exact_gain']:>+8.3f}{r['kl_last']:>8.4f}")
    lines += ["", "base/after: held-out greedy accuracy; p: paired McNemar test vs the base model; "
                  "degen: share of skipped steps; distinct: distinct rewards per group; gain: "
                  "sampled exact correctness, last minus first training window. One seed per run: "
                  "compare runs with care."]
    return "\n".join(lines)


def sweep(config_paths: Sequence[str], overrides=None, probes: Optional[bool] = None) -> List[dict]:
    """Run several configs in sequence, reusing the loaded model while the model
    settings stay the same, then summarise them side by side."""
    policy, policy_fp, run_dirs = None, None, []
    for path in config_paths:
        cfg = load_config(path, overrides)
        fp = fingerprint_of(cfg, MODEL_KEYS)
        if fp != policy_fp:
            policy = None
            _free_gpu()
            policy_fp = fp
        out = run_pipeline(cfg, config_path=str(path), probes=probes, policy=policy)
        policy = out["policy"]
        run_dirs.append(Path(out["run_dir"]))
    return summarize_runs(run_dirs)


def _free_gpu() -> None:
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
