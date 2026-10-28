#!/usr/bin/env python
"""Reintroduce every historical bug into a temporary copy of the repository and
check that the test suite fails for each one (README, "Testing").

    python tools/mutation_test.py          # ~10 minutes on a laptop CPU

Exit code 0 only if every mutation is caught.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (description, file, original text, mutated text). Each original must occur exactly once.
MUTATIONS = [
    ("mask starts at plen", "src/grpo/algorithm/rollout.py",
     "m[i, plen - 1:L - 1] = 1.0", "m[i, plen:L - 1] = 1.0"),
    ("flipped gradient sign", "src/grpo/algorithm/losses.py",
     "loss = -per_resp.mean()", "loss = per_resp.mean()"),
    ("advantage not divided by std", "src/grpo/algorithm/advantages.py",
     "return (r - r.mean()) / (s + eps)", "return r - r.mean()"),
    ("naive KL instead of k3", "src/grpo/algorithm/losses.py",
     "return torch.exp(delta) - delta - 1.0", "return -delta"),
    ("reference silently equals policy", "src/grpo/algorithm/rollout.py",
     "state_ctx = nullcontext() if use_adapter else policy.use_base()", "state_ctx = nullcontext()"),
    ("[\\d.]+ number regex", "src/grpo/data/financial.py",
     'ANSWER_RE = re.compile(r"Answer:\\s*\\$?(-?\\d+(?:\\.\\d+)?)")',
     'ANSWER_RE = re.compile(r"Answer:\\s*\\$?([\\d.]+)")'),
    ("NaN reward guard removed", "src/grpo/rewards/components.py",
     "    if not (math.isfinite(p) and math.isfinite(g)):\n        return 0.0\n    if verify(p, sample):",
     "    if verify(p, sample):"),
    ("cache restores metrics but not weights", "src/grpo/training.py",
     'if cached.get("fingerprint") == fp and policy.load_adapter(str(adapter_path)):',
     'if cached.get("fingerprint") == fp:'),
    ("single-use context manager", "src/grpo/model.py",
     "    def use_base(self) -> _UseBase:\n        return _UseBase(self)",
     "    def use_base(self):\n        import contextlib\n\n        @contextlib.contextmanager\n"
     "        def gen():\n            saved = self.snapshot()\n            self.restore(self.initial_lora)\n"
     "            try:\n                yield self\n            finally:\n                self.restore(saved)\n"
     "        if not hasattr(self, '_cm'):\n            self._cm = gen()\n        return self._cm"),
    ("gated component re-enables the fallback", "src/grpo/rewards/components.py",
     "graded_score(task.parse(t, strict=True), s, task.verify, lam, kappa)",
     "graded_score(task.parse(t), s, task.verify, lam, kappa)"),
    ("probe leaves the model at base after an error", "src/grpo/evaluation/probes.py",
     "    finally:\n        policy.restore(trained)", "    except KeyError:\n        policy.restore(trained)"),
    ("ratio != 1 at mu=1", "src/grpo/algorithm/step.py",
     "logp_old = per_token_logps(policy, seqs, use_adapter=True).detach()",
     "logp_old = (per_token_logps(policy, seqs, use_adapter=True) + 1e-3).detach()"),
    ("telemetry measured at the training lambda", "src/grpo/training.py",
     "partial_mass=partial_mass(task, res.texts, sample, cfg.evaluation.telemetry_lam,",
     "partial_mass=partial_mass(task, res.texts, sample, "
     "cfg.reward.components[0].params.get('lam', 0.3),"),
    ("v2 growth labels from the exact rate again", "src/grpo/data/financial.py",
     "round((rn - ro) / ro * 100, 2)", "round(g * 100, 2)"),
    ("before/after compared unpaired", "src/grpo/evaluation/report.py",
     "    paired = bool(common)", "    paired = False"),
    ("training cache keyed by name only", "src/grpo/training.py",
     'if cached.get("fingerprint") == fp and policy.load_adapter(str(adapter_path)):',
     'if policy.load_adapter(str(adapter_path)):'),
    ("evaluation sampling not seeded", "src/grpo/evaluation/heldout.py",
     "    if k:\n        set_seed(cfg.experiment.seed)", "    if False:\n        set_seed(cfg.experiment.seed)"),
    ("stop detection uses only the tokenizer EOS", "src/grpo/algorithm/step.py",
     "true_lengths(seqs, plen, policy.stop_token_ids())",
     "true_lengths(seqs, plen, policy.tokenizer.eos_token_id)"),
]


def main() -> int:
    missed = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, rel, old, new in MUTATIONS:
            work = Path(tmp) / "repo"
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(
                "results", "__pycache__", ".pytest_cache", ".git", "*.egg-info"))
            f = work / rel
            src = f.read_text()
            if src.count(old) != 1:
                print(f"STALE   {name}: the original text no longer occurs exactly once in {rel}")
                missed.append(name)
                continue
            f.write_text(src.replace(old, new))
            env = {**os.environ, "PYTHONPATH": str(work / "src")}
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
                               cwd=work, capture_output=True, text=True, env=env)
            first = next((ln for ln in r.stdout.splitlines() if ln.startswith("FAILED")), "")
            caught = r.returncode != 0
            print(f"{'CAUGHT ' if caught else 'MISSED!'} {name:<46} {first[:90]}", flush=True)
            if not caught:
                missed.append(name)
    print(f"\n{len(MUTATIONS) - len(missed)} of {len(MUTATIONS)} mutations caught")
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main())
