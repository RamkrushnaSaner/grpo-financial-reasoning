"""Command line interface: ``grpo <command>`` (or ``python -m grpo <command>``).

    grpo validate                         CPU checks of the reward and the GRPO math (~5 s)
    grpo config  CONFIG [--set k=v ...]   print the fully resolved config
    grpo run     CONFIG [--set k=v ...]   the full pipeline for one config
    grpo sweep   CONFIG ... [--set ...]   several configs in sequence, then a summary table
    grpo summarize RUN_DIR ...            the summary table for finished runs
    grpo ask     CONFIG --question ...    one question through the base and trained policy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import yaml


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="YAML config file")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override a setting, e.g. --set algorithm.mu=4 (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="grpo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("validate", help="CPU checks of the reward and the GRPO mathematics")

    p = sub.add_parser("config", help="print the fully resolved config")
    _add_config(p)

    p = sub.add_parser("run", help="run the full pipeline for one config")
    _add_config(p)
    p.add_argument("--no-probes", action="store_true", help="skip the question-level probe")

    p = sub.add_parser("sweep", help="run several configs, then summarise them")
    p.add_argument("configs", nargs="+", help="YAML config files")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", dest="overrides",
                   help="override applied to every config (repeatable)")
    p.add_argument("--no-probes", action="store_true")

    p = sub.add_parser("summarize", help="summary table of finished run directories")
    p.add_argument("run_dirs", nargs="+")

    p = sub.add_parser("ask", help="one question through the base and the trained policy")
    _add_config(p)
    p.add_argument("--question", required=True)
    p.add_argument("--context", default="")
    p.add_argument("--answer", type=float, help="ground truth, if known")
    p.add_argument("--answer-type", default="dollar",
                   choices=["dollar", "percentage", "ratio", "integer"])
    p.add_argument("--samples", type=int, default=0, help="also sample N answers for pass@k")
    p.add_argument("--sampled", action="store_true", help="sampled instead of greedy decoding")
    p.add_argument("--run-dir", help="trained run to load (default: the config's run directory)")
    return ap


def cmd_validate() -> None:
    from .rewards import validate_reward
    from .validation import validate_math
    print("reward properties")
    validate_reward()
    print("   bounded / monotonic / symmetric / distinct / scale-free / edge-safe   PASS\n")
    print("GRPO mathematics (tiny CPU model)")
    out = validate_math(verbose=True)
    print(f"\nALL PASS (gradient-direction correlation {out['grad_corr']:.4f})")
    print("Test 7 (reference != trained policy) needs a trained adapter; `grpo run` performs it.")


def cmd_ask(args) -> None:
    from .config import load_config
    from .data import build_task
    from .evaluation import ask_both, render_ab
    from .experiment import run_dir_for
    from .model import load_policy
    from .rewards import build_reward
    from .training import load_trained

    cfg = load_config(args.config, args.overrides)
    run_dir = Path(args.run_dir) if args.run_dir else run_dir_for(cfg)
    missing = SystemExit(f"no trained run in {run_dir}: run `grpo run {args.config}` first")
    if not (run_dir / "adapter.pt").exists():
        raise missing                                  # fail before loading the model
    policy = load_policy(cfg.model)
    if load_trained(policy, run_dir) is None:
        raise missing
    task = build_task(cfg.task)
    sample = dict(context=args.context, question=args.question, answer=args.answer,
                  answer_type=args.answer_type if args.answer is not None else None)
    out = ask_both(policy, task, sample, cfg, reward=build_reward(cfg.reward, task),
                   greedy=not args.sampled, n_samples=args.samples)
    print(render_ab(sample, out))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        cmd_validate()
    elif args.command == "config":
        from .config import load_config
        print(yaml.safe_dump(load_config(args.config, args.overrides).to_dict(), sort_keys=False),
              end="")
    elif args.command == "run":
        from .config import load_config
        from .experiment import run_pipeline
        cfg = load_config(args.config, args.overrides)
        run_pipeline(cfg, config_path=args.config, probes=False if args.no_probes else None)
    elif args.command == "sweep":
        from .experiment import format_summary, sweep
        rows = sweep(args.configs, args.overrides, probes=False if args.no_probes else None)
        print("\n" + format_summary(rows))
    elif args.command == "summarize":
        from .experiment import format_summary, summarize_runs
        print(format_summary(summarize_runs([Path(d) for d in args.run_dirs])))
    elif args.command == "ask":
        cmd_ask(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
