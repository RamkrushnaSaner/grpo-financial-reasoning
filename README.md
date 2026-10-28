# GRPO from First Principles

**A small, verified, modular implementation of Group Relative Policy Optimization (the RL algorithm behind DeepSeekMath and DeepSeek-R1), applied to verifiable financial reasoning on Qwen2.5-0.5B-Instruct.**

![python](https://img.shields.io/badge/python-3.10%2B-blue) ![pytorch](https://img.shields.io/badge/pytorch-2.1%2B-ee4c2c) ![license](https://img.shields.io/badge/license-MIT-green) ![tests](https://img.shields.io/badge/tests-146-brightgreen)

No TRL and no wrappers: rollouts, rewards, group-relative advantages, the clipped surrogate, the KL penalty to a reference policy and the update are written out and tested. Tasks, reward components, advantage estimators, losses and model loaders are registered by name and chosen in a YAML file, so a new experiment changes one block of config and at most one function. It trains on a free Colab T4 in about 16 minutes.

---

## Results so far

One reported run (Run 6, `configs/run6.yaml`): 60 GRPO steps, one seed, 60 held-out questions, greedy decoding. Both models answered the same questions, so the tests are paired.

| metric | base Qwen | after GRPO | change | paired test |
|---|---|---|---|---|
| Overall accuracy | 16.7% | **43.3%** | **+26.7 pp** | significant under any pairing (McNemar p ≤ 0.011) |
| 1-step accuracy (n=23) | 21.7% | 56.5% | +34.8 pp | not established alone (p between 0.008 and 0.096) |
| 2-step accuracy (n=37) | 13.5% | 35.1% | +21.6 pp | not established alone (p between 0.008 and 0.096) |
| **Format compliance** | **61.7%** | **20.0%** | **−41.7 pp** | significant under any pairing (p ≤ 0.0005) |
| pass@1 / pass@6 (sampled) | 15.8% / 48.3% | 24.4% / 58.3% | +8.6 / +10.0 pp | weak |

Run 6 did not save per-question results, so the p-values are given as the range over every pairing consistent with the totals; the pipeline now saves them.

**What this shows.** On the four training templates the model's default answer is right much more often, mostly by choosing better formulas (e.g. one year of growth instead of ten). It also learned to drop the required `Answer:` tag: the parser's fallback ("last number in the text") let it earn the correctness reward without the format, so correctness starved format. The fix (`correct_gated`, `configs/gated.yaml`) has **not yet been validated** by a training run.

**What this does not show.** General reasoning: four synthetic templates, one seed, held out by index rather than by question type. Also, 11 of the 60 held-out labels in that run's dataset cannot be derived from their own question (a generator bug, fixed in dataset version 2); the comparison is still fair because both models faced the same labels, but the absolute accuracies are understated. Details: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Quickstart

**CPU: tests, validation and a full pipeline run on a tiny stand-in model (no download, about a minute)**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
make test        # the test suite
make validate    # reward properties and GRPO math checks
make check       # the whole pipeline with the tiny CPU model; numbers meaningless
```

**GPU (a Colab T4 is enough)**

```bash
pip install -e ".[train,dev]"
make smoke       # ~5 min: every stage on the real model
make reproduce   # Run 6 (configs/run6.yaml)
make base        # the same method on the corrected dataset (configs/base.yaml)
make gated       # the unvalidated fix for the format loophole (configs/gated.yaml)
make ablations   # base + reward and lambda ablations, then a summary table
```

On Colab, open [`notebooks/quickstart_colab.ipynb`](notebooks/quickstart_colab.ipynb).

**Ask one question to the base and the trained model**

```bash
grpo ask configs/run6.yaml --samples 6 \
  --context "Current annual revenue is \$4.8B with a 6% annual growth rate." \
  --question "What is the projected revenue in \$B for next year?" \
  --answer 5.09 --answer-type dollar
```

---

## How it works

```
question ──> one-shot prompt (the output contract the reward checks)
         ──> sample G=6 answers at T=1.0          the only source of exploration
         ──> reward = 1.0·correctness + 0.1·format
         ──> A_i = (r_i − mean) / std              the group mean replaces PPO's critic
         ──> skip if all six tie                   zero spread = zero information
         ──> mask to the answer tokens             positions plen−1 .. L−2
         ──> log-probs: reference (adapter zeroed), rollout, current
         ──> loss = −mean[min(ρA, clip(ρ)A) − β·KL], length-normalised
         ──> update the LoRA adapters only
```

$$\mathcal{J}(\theta)=\mathbb{E}\Big[\tfrac{1}{G}\textstyle\sum_{i}\tfrac{1}{|o_i|}\sum_{t}\big(\min(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1\pm\epsilon)\hat A_i)-\beta\,\mathbb{D}_{KL}\big)\Big],\qquad \hat A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}$$

The group mean is a Monte-Carlo estimate of the policy's expected reward on that question: an unbiased baseline that costs nothing, which is what lets GRPO drop PPO's value network. With one update per group (μ = 1), ρ = 1 exactly and the update is REINFORCE with that baseline plus the KL pull. Every term maps to one function: [`docs/METHOD.md`](docs/METHOD.md).

---

## Repository layout

```
configs/            base.yaml (every setting, documented), run6.yaml, gated.yaml,
                    smoke.yaml, cpu_smoke.yaml, ablations/
src/grpo/
  config.py         typed config sections; YAML with extends and --set overrides
  registry.py       name -> implementation lookup: the extension mechanism
  data/             tasks: questions, prompt, answer parser, verifier, probe questions
  rewards/          reward components and their weighted, fault-isolated combination
  algorithm/        advantage estimators, losses, rollouts, the GRPO update step
  model.py          the LoRA policy (adapter state machine) and model loaders
  testing.py        tiny CPU stand-in model for tests and cpu_smoke
  training.py       the training loop: telemetry, metrics stream, fingerprinted checkpoint
  evaluation/       held-out evaluation, base-vs-trained probe, paired comparison, report
  experiment.py     run directories, manifests, cached stages, pipeline, sweeps
  tracking.py       run log and metrics.jsonl
  stats.py          pass@k, McNemar, paired bootstrap, window means
  validation.py     mathematical checks of the update
  cli.py            the `grpo` command
tests/              CPU test suite (+ an optional Hugging Face integration test)
docs/               ARCHITECTURE, METHOD, RESULTS, LESSONS, TECH_DEBT, LEARNING_NOTES
notebooks/          quickstart_colab.ipynb; archive/grpo_master.ipynb (produced Run 6)
```

### What each module does

- **`config`**: eight frozen sections (`experiment`, `model`, `task`, `reward`, `algorithm`, `sampling`, `training`, `evaluation`). Unknown keys, invalid values and unregistered component names are rejected when a config is loaded, before any GPU time is spent.
- **`data`**: a `Task` owns its questions, chat prompt, answer parser (`parse`, with a `strict` mode), verifier (`verify`), format check and probe questions. `FinancialTask` generates four templates (growth, margin, sum of quarters, projection); `task.version: 1` is the generator behind Run 6, `version: 2` asks the same questions with labels computed from the numbers shown.
- **`rewards`**: components are factories `(task, **params) -> fn(text, sample)`: `correct_binary`, `correct_graded` (partial credit λ·exp(−rel_err/κ)), `correct_gated` (credit only with the `Answer:` contract) and `format`. `CompositeReward` sums them with weights, isolates failures (a crashing or NaN component scores 0 and is counted) and reports per-component variance.
- **`algorithm`**: `group_std` advantages, the `grpo` loss with the k3 KL estimator, batched group sampling, stop-token-aware lengths, the completion mask, per-token log-probs for the current and reference policies, and `grpo_step`. It never looks inside a sample.
- **`model`**: `Policy` owns every adapter transition (`snapshot`, `restore`, `reset`, re-enterable `use_base`); the reference policy is the pristine adapter (LoRA B = 0 *is* the base model). Loaders: `hf_lora` (Hugging Face + peft, 4-bit by default) and `tiny`.
- **`training`**: one GRPO step per scheduled question; arm-independent telemetry (exact correctness from the task, partial credit at a fixed λ); `metrics.jsonl` streamed per step; the adapter is reused only if the training fingerprint matches.
- **`evaluation`**: greedy held-out accuracy, format and pass@k with one record per question; the base-vs-trained probe; paired McNemar / bootstrap comparison; plots and `report.md`.
- **`experiment`**: the end-to-end pipeline (validate → baseline → train → reference check → evaluate → compare → probe → report), run directories with manifests, shared baseline cache, sweeps and summaries.

Design and extension guide: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Configuration

Every setting is in [`configs/base.yaml`](configs/base.yaml) with a comment. Other configs extend it:

| config | purpose |
|---|---|
| `base.yaml` | the Run 6 method on the corrected dataset (version 2); the starting point for new experiments |
| `run6.yaml` | reproduces the reported run, including its known issues (dataset version 1) |
| `gated.yaml` | `correct_gated` + `lora_dropout: 0.0`; not yet validated |
| `smoke.yaml` | ~5-minute GPU check of every stage |
| `cpu_smoke.yaml` | seconds on any CPU with the tiny model; used by `make check` and CI |
| `ablations/*.yaml` | binary reward, no format reward, λ = 0.6 violation |

```yaml
extends: base.yaml          # a new experiment lists only what differs
algorithm:
  mu: 2
reward:
  components:
    - {name: correct_gated, weight: 1.0, params: {lam: 0.3, kappa: 0.2}}
    - {name: format, weight: 0.1}
```

Any setting can be overridden on the command line: `grpo run configs/base.yaml --set algorithm.mu=2 --set experiment.name=mu2`. `grpo config CONFIG` prints the fully resolved config.

---

## Running experiments reproducibly

```bash
grpo run CONFIG [--set k=v ...] [--no-probes]   # one run
grpo sweep CONFIG CONFIG ... [--set k=v ...]    # several runs, then a summary table
grpo summarize RUN_DIR RUN_DIR ...              # summary table of finished runs
```

Each run writes `<output_dir>/<name>/` (the name defaults to the config file's stem): the resolved `config.yaml`, a `manifest.json` (settings fingerprints, library versions, git commit, command), `run.log`, `metrics.jsonl` (one record per step: loss, gradient norm, KL, clip fraction, reward statistics, telemetry), `train.json`, `adapter.pt`, `eval_after.json` (with per-question records), `comparison.json`, `probes.json`, `report.md` and plots. Base-model evaluations are cached in `<output_dir>/_baselines/` and shared across runs with the same model, task, sampling and evaluation settings.

Every stage is cached under a fingerprint of the settings that can change its output: re-running a config resumes, and changing a setting reruns exactly the stages it affects. Sampled generation on a GPU is not bitwise deterministic, so repeated runs are comparable rather than identical; greedy evaluation is deterministic. For claims about one method versus another, run several seeds (`--set experiment.seed=N`) and a larger held-out set (see [`docs/TECH_DEBT.md`](docs/TECH_DEBT.md), A2).

---

## Extending

Register a function and name it in the config:

```python
from grpo.algorithm import ADVANTAGES

@ADVANTAGES.register("group_mean")          # centre only, no std division
def group_mean(rewards):
    return None if rewards.std() < 1e-6 else rewards - rewards.mean()
```

```yaml
algorithm:
  advantage: group_mean
```

The same pattern covers tasks (`grpo.data.TASKS`), reward components (`grpo.rewards.REWARD_COMPONENTS`), losses (`grpo.algorithm.LOSSES`) and model loaders (`grpo.model.MODEL_LOADERS`). Contracts and examples: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#extending).

---

## Testing

`make test` runs the suite on CPU without downloading a model. The tiny stand-in model drives the **real** step, training loop, evaluation, probe and CLI end to end. If `transformers` and `peft` are installed, an integration test also runs the real Hugging Face + peft path on a tiny, locally built Qwen2 (no 4-bit: bitsandbytes needs a GPU).

**The refactor did not change the algorithm.** A golden test checks that the update reproduces the loss, KL, clip fraction and updated weights recorded from the pre-refactor implementation (exactly on the machine that recorded them; within float32 noise, about 1e-7, on CPUs that take other kernel paths), and that dataset version 1 reproduces the original 500 questions byte for byte.

**Every historical bug is a regression test, and the suite is mutation-tested.** Each silent failure found during development was reintroduced into a copy of the code; the suite fails for every one:

| reintroduced bug | caught by (most direct test) |
|---|---|
| mask starts at `plen`, dropping the first generated token | `test_mask_regression_first_generated_token_is_scored` |
| flipped gradient sign | `test_gradient_direction` |
| advantage not divided by std | `test_advantages_are_centred_and_normalised` |
| naive KL estimator instead of k3 | `test_grpo_loss_uses_k3_not_naive_kl` |
| reference policy silently equals the policy (KL ≡ 0) | `test_reference_differs_when_trained_and_adapter_survives` |
| `[\d.]+` number regex | `test_malformed_number_does_not_crash` |
| NaN reward guard removed | `test_validation_suite_passes_and_rejects_unsafe_lambda` |
| cache restores metrics but not weights | `test_train_writes_metrics_and_caches_weights_not_just_metrics` |
| single-use context manager reused | `test_use_base_is_reentrant` |
| gated component re-enables the parser fallback | `test_gated_component_closes_the_format_loophole` |
| probe leaves the model at base after an error | `test_ask_both_restores_adapter_when_generation_raises` |
| ρ ≠ 1 at μ = 1 | `test_step_updates_only_lora_and_ratio_is_exactly_one` |
| partial-credit telemetry measured at the training λ | `test_partial_credit_telemetry_ignores_the_training_lambda` |
| growth labels from the exact rate (dataset v2) | `test_version2_asks_the_same_questions_with_consistent_labels` |
| before/after compared unpaired | `test_compare_uses_paired_tests_and_flags_format_collapse` |
| training cache keyed by run name only | `test_changed_settings_invalidate_the_training_cache` |
| evaluation sampling not seeded | `test_evaluation_seeds_its_own_sampling` |
| stop detection uses only the tokenizer's EOS | `test_step_ends_answers_on_every_stop_token` |

18 of 18 mutations make the suite fail (`make mutation` reruns the check). The golden test (`test_update_matches_the_original_implementation`) additionally catches every mutation of the update itself, on both AVX-512 and generic CPU kernels.

---

## Known gaps

Before the next experiment: run the GPU path once after the refactor (`make smoke`, then `make reproduce`); enlarge the held-out set and use several seeds (60 questions detect a 10-point difference between runs less than 60% of the time); train `gated.yaml`; decide on the answer tolerance (±$0.10 credited a wrong multiplication). Method limitations that are deliberately unchanged (one question per update, std-normalised advantages, per-answer length normalisation) have extension points ready. Full list: [`docs/TECH_DEBT.md`](docs/TECH_DEBT.md). Lessons and falsified claims: [`docs/LESSONS.md`](docs/LESSONS.md).

---

## References

- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300 — introduces GRPO.
- DeepSeek-AI, 2025. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:2501.12948
- Liu et al., 2025. *Understanding R1-Zero-Like Training: A Critical Perspective.* arXiv:2503.20783 — length and std normalisation biases.
- Schulman et al., 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
- Schulman, 2020. *Approximating KL Divergence* (blog post) — the k3 estimator.
- Chen et al., 2021. *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374 — unbiased pass@k.
- Hu et al., 2021. *LoRA.* arXiv:2106.09685 · Dettmers et al., 2023. *QLoRA.* arXiv:2305.14314

## License

MIT — see [`LICENSE`](LICENSE).
