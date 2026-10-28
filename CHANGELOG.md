# Changelog

## 2.0.0 — modular architecture

The GRPO update is unchanged: on identical inputs the refactored step reproduces
the loss, KL, clip fraction and updated weights of 1.1.0 (exactly on the same
machine, within float32 noise across CPUs;
`test_update_matches_the_original_implementation`).

**Structure**
- Split into `data` (tasks), `rewards` (components + composite), `algorithm`
  (advantages, losses, rollout, step), `model`, `training`, `evaluation`,
  `experiment`, `tracking`, `stats`, with registries for tasks, reward
  components, advantage estimators, losses and model loaders.
- The GRPO step no longer knows the task: it takes prompt ids, a sample and a
  reward object. Task-level telemetry moved to the training loop.
- One CLI, `grpo` (`validate`, `config`, `run`, `sweep`, `summarize`, `ask`),
  replaces the four scripts.
- Nested YAML config with `extends:` and `--set section.key=value` overrides.
- One directory per run with resolved config, manifest (fingerprints, library
  versions, git commit), `metrics.jsonl`, and fingerprinted stage caches; base
  model evaluations shared across runs.
- A tiny CPU stand-in model (`model.loader: tiny`, `configs/cpu_smoke.yaml`)
  runs the whole pipeline in seconds; an integration test runs the real
  Hugging Face + peft path on a tiny locally built Qwen2.

**Fixes**
- Dataset version 2: growth (and margin) labels computed from the numbers the
  question shows. Version 1, which produced Run 6, is kept and reproduces the
  original dataset byte for byte.
- Before/after comparisons are paired: per-question records are saved and
  tested with exact McNemar (binary metrics) and a paired bootstrap
  (continuous metrics). The single-proportion "SE" verdicts are gone.
- The training cache is keyed by a fingerprint of every setting that affects
  training, not by run name.
- Evaluation seeds its own pass@k sampling (`experiment.seed`).
- Generation stops on every stop token the model uses (tokenizer EOS plus
  generation config), not only the tokenizer's EOS.
- Loss, gradient norm and mean reward are logged every step.

**Corrections to reported results** (docs/RESULTS.md, docs/LESSONS.md)
- Run 6 significance restated as paired tests; 1-step and 2-step gains are not
  established on their own.
- 11 of Run 6's 60 held-out labels are inconsistent with their question.
- Probe A4 was credited for wrong arithmetic within the ±$0.10 tolerance.

**Migration from 1.1.0**

| 1.1.0 | 2.0.0 |
|---|---|
| `python scripts/run_pipeline.py --config configs/default.yaml` | `grpo run configs/run6.yaml` |
| `configs/recommended.yaml` | `configs/gated.yaml` |
| `python scripts/validate.py` | `grpo validate` |
| `python scripts/compare.py ...` | `grpo ask CONFIG --question ...` |
| `python scripts/run_experiments.py` | `grpo sweep configs/base.yaml configs/ablations/*.yaml` |
| `reward_arm: graded` | `reward.components: [correct_graded, format]` |
| `lam`, `kappa`, `format_weight` | component `params` and `weight` |
| `n_steps`, `eval_n`, `eval_k`, `window` | `training.steps`, `evaluation.n`, `evaluation.k`, `training.window` |
| `results/<config>/main.json`, `adapters/main.pt` | `results/<name>/train.json`, `adapter.pt` |
| `RewardRegistry`, `make_registry(arm)` | `CompositeReward`, `build_reward(config, task)` |
| `grpo.engine` | `grpo.algorithm` |
| `FinancialReasoningDataset` | `FinancialTask` (`task.version`) |

## 1.1.0 — repository release
- Refactored the master notebook into the `grpo` package with a CLI, configs, and docs.
- **Added** `graded_gated` reward arm, closing the parser-fallback loophole behind the format collapse (unvalidated).
- **Added** `lora_dropout` config and `max |ρ−1|` telemetry, which pinned down why clip fraction was 0.0038 instead of 0 at μ=1.
- **Added** fingerprinted caching: a cached evaluation is reused only if every setting affecting it is unchanged.
- **Added** 89 CPU tests, including end-to-end runs of the real pipeline and scripts on a tiny model; mutation-tested against all twelve historical bugs.
- **Fixed** `get_logger` ignoring a new log file once any handler existed.
- **Fixed** subgroup standard errors using the overall n (1-step significance was overstated: 5.4 → 3.3 SE).
- **Corrected** two claims: truncation did not cause the baseline drop; λ = 0.6's accuracy cost is 0.95 SE, not 1.2 SE.

## 1.0.0 — master notebook (Run 6)
- Explicit adapter-state machine replacing peft's single-use `disable_adapter()`.
- Validation test 7: the reference policy must differ from the policy once trained.
- Result: accuracy 16.7% → 43.3% (+26.7 pp, 4.1 SE); format 61.7% → 20.0%.

## 0.5.0 — adversarial runs
- λ = 0.6, β = 0 and format-weight 1.0 against a control. Three detector architectures failed on real data; one because partial-credit telemetry was measured at each run's own λ.

## 0.4.0 — reward ablation
- Binary vs graded vs graded-without-format. Degenerate groups 3.3% → 0.0%.

## 0.3.0 — reward registry
- Graded correctness, the λ < 0.5 bound, affine-invariance demonstration, offline validation.

## 0.2.0 — GRPO on Qwen2.5-0.5B
- 4-bit + LoRA, reference model by adapter toggling. The first run stalled at 0% format; a one-shot prompt fixed it: 27.5% → 52.5%.

## 0.1.0 — toy model
- From-scratch GRPO on a 600K character-level model: format learned, correctness 0% across 300 steps.
