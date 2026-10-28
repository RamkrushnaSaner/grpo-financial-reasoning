# Architecture

The package separates **what is learned** (task, reward), **how it is learned**
(advantage estimator, loss, update step), **what is learned with** (model
loader) and **how it is measured** (evaluation, statistics). Each of the
swappable pieces is a registered function or class selected by name in a
config file, so an experiment that changes one of them changes one block of
YAML and, at most, adds one function.

## Module map

```
src/grpo/
  config.py            typed, frozen config sections; YAML loading (extends, --set overrides)
  registry.py          Registry: name -> implementation (the extension mechanism)
  data/
    base.py            Task interface, TASKS registry, build_task
    financial.py       the financial task: generator (v1, v2), prompt, parser, verifier
    financial_probes.py  15 hand-written probe questions
  rewards/
    composite.py       CompositeReward: weighted sum, fault isolation, per-component telemetry
    components.py      REWARD_COMPONENTS: correct_binary, correct_graded, correct_gated, format
    __init__.py        build_reward(config, task)
  algorithm/
    advantages.py      ADVANTAGES: group_std
    losses.py          LOSSES: grpo (+ k3_kl)
    rollout.py         prompt ids, group sampling, stop handling, mask, per-token log-probs
    step.py            grpo_step: one update for one question
  model.py             Policy (adapter state machine), MODEL_LOADERS: hf_lora, tiny
  testing.py           the tiny CPU model and tokenizer behind the "tiny" loader
  training.py          the training loop, telemetry, fingerprinted checkpoint
  evaluation/
    heldout.py         greedy accuracy, format, pass@k, one record per question
    probes.py          base vs trained answers to the same question
    report.py          paired before/after comparison, plots, run report
  stats.py             pass@k, McNemar, paired bootstrap, window means
  tracking.py          per-run log file, metrics.jsonl writer
  utils.py             seeding, JSON, file hashing
  validation.py        mathematical checks of the update (tests 1-7)
  experiment.py        run directories, manifests, cached stages, pipeline, sweep
  cli.py               the `grpo` command
```

## Dependencies point one way

```
            cli
             |
         experiment ---------------------------+
          /    |     \                         |
   training  evaluation  validation            |
      |    \     |      /                      |
      |     algorithm  (task-agnostic)         |
      |        |                               |
   rewards ---data         model (+ testing)   |
        \      |          /                    |
         registry, config, stats, tracking, utils
```

- `algorithm` never imports `data` or `rewards`: `grpo_step` receives the
  prompt ids, the sample (passed through untouched) and a reward object.
- `rewards` uses the task only through its methods (`parse`, `verify`,
  `follows_format`); `data` never imports `rewards`.
- Importing the package does not import transformers, peft or bitsandbytes;
  only the `hf_lora` loader does, when called.

## One training step

```
task[qi] ──> task.messages ──> build_prompt_ids ──> generate_group (G samples)
                                                         │
                                  texts ◄── decode ◄── true_lengths (stop ids)
                                    │
   reward.score_group(texts, sample) ──> ADVANTAGES[algorithm.advantage] ──> None? skip
                                                         │
          completion_mask ── per_token_logps(ref) ── per_token_logps(old)
                                                         │
   for mu epochs: per_token_logps(new) ──> LOSSES[algorithm.loss] ──> backward, clip, step
                                                         │
   StepResult(loss, kl, clip_frac, ratio_dev, grad_norm, texts, rewards, breakdown)
                                                         │
   training.rollout_telemetry(task, result) ──> metrics.jsonl
```

## Interfaces

| piece | registry | contract |
|---|---|---|
| Task | `grpo.data.TASKS` | subclass `Task`: `from_config`, `messages`, `parse(text, strict)`, `verify(pred, sample)`, `follows_format`; optional `subgroup`, `probes` |
| Reward component | `grpo.rewards.REWARD_COMPONENTS` | `factory(task, **params) -> fn(text, sample) -> float` |
| Advantage estimator | `grpo.algorithm.ADVANTAGES` | `fn(rewards: Tensor[G]) -> Tensor[G] or None` (None skips the step) |
| Loss | `grpo.algorithm.LOSSES` | `fn(logp_new, logp_old, logp_ref, adv, mask, *, epsilon, beta) -> (loss, kl, clip_frac)` |
| Model loader | `grpo.model.MODEL_LOADERS` | `fn(model_config) -> Policy` (any model whose state dict has `lora_` tensors) |

Registered today: tasks `financial`; reward components `correct_binary`,
`correct_graded`, `correct_gated`, `format`; advantages `group_std`; losses
`grpo`; model loaders `hf_lora`, `tiny`. A config naming anything else is
rejected when it is loaded, before any model is touched.

## Extending

A component must be registered before a config that names it is loaded. For a
permanent addition, put it next to the built-ins (the package imports them on
start-up, so the CLI sees it). For a quick trial in a notebook, register it in
the notebook, then call the Python API:

```python
from grpo.config import load_config
from grpo.experiment import run_pipeline
from grpo.rewards import REWARD_COMPONENTS

@REWARD_COMPONENTS.register("brevity")
def brevity(task, max_chars=400):
    return lambda text, sample: float(len(text) <= max_chars)

cfg = load_config("configs/base.yaml", ["experiment.name=brevity_trial"])
cfg = cfg.with_overrides({"reward.components": [
    {"name": "correct_graded", "weight": 1.0, "params": {"lam": 0.3, "kappa": 0.2}},
    {"name": "brevity", "weight": 0.05, "params": {"max_chars": 300}}]})
run_pipeline(cfg)
```

**A reward component** (`grpo/rewards/components.py`): a factory taking the
task and its `params`, returning `fn(text, sample) -> float` (as `brevity`
above), then in YAML:

```yaml
reward:
  components:
    - {name: correct_graded, weight: 1.0, params: {lam: 0.3, kappa: 0.2}}
    - {name: brevity, weight: 0.05, params: {max_chars: 300}}
```

**An advantage estimator** (`grpo/algorithm/advantages.py`):

```python
@ADVANTAGES.register("group_mean")          # centre only, no std division
def group_mean(rewards):
    return None if rewards.std() < 1e-6 else rewards - rewards.mean()
```
then `algorithm.advantage: group_mean`.

**A loss** (`grpo/algorithm/losses.py`): same signature as `grpo_loss`; then
`algorithm.loss: <name>`. Return the mean KL and clip fraction for telemetry.

**A task** (`grpo/data/<name>.py`, imported from `grpo/data/__init__.py`):
subclass `Task`, generate or load `samples`, set `train_pool`, implement the
methods above, decorate with `@TASKS.register("<name>")`. The numeric reward
components work unchanged for any task whose samples carry a numeric
`"answer"`.

**A model** (`grpo/model.py`): register a loader returning a `Policy`. The
reference policy is the pristine adapter, so the model must use LoRA with
B initialised to zero (peft does).

Every extension point is covered by a test that registers a throwaway
component and runs it through the real code (`test_custom_component_plugs_in_by_name`,
`test_registered_advantage_and_loss_are_used_by_the_step`).

## Experiments and caching

`grpo run CONFIG` writes one directory per run, `<output_dir>/<name>/`:

| file | contents |
|---|---|
| `config.yaml` | the fully resolved config |
| `manifest.json` | fingerprints, library versions, git commit, command, time |
| `run.log` | the log |
| `metrics.jsonl` | one record per training step (loss, grad norm, KL, clip, rewards, telemetry) |
| `train.json` | history, summary, training fingerprint |
| `adapter.pt` | trained LoRA tensors |
| `eval_after.json` | held-out evaluation of the trained policy, with per-question records |
| `comparison.json` | paired before/after statistics |
| `probes.json` | base vs trained probe answers |
| `report.md`, `curves.png`, `before_after.png` | summaries |

Base-model evaluations go to `<output_dir>/_baselines/<fingerprint>.json` and
are shared by every run whose model, task, sampling and evaluation settings
match. Each stage is reused only when its fingerprint matches (the settings
that can change its output); otherwise it is recomputed. Re-running a config
therefore resumes rather than repeats, and changing one setting reruns exactly
the stages it affects.

## Testing strategy

- **Unit tests** for every pure function (advantages, loss, mask, parser,
  verifier, reward formula, statistics, config loading).
- **Regression tests** for every historical silent bug (see
  [LESSONS.md](LESSONS.md)), and a **golden test**: the refactored update
  reproduces loss, KL, clip fraction and updated weights recorded from the
  pre-refactor code.
- **End-to-end tests** of the real pipeline and CLI with the tiny CPU model
  (`configs/cpu_smoke.yaml`).
- **Integration test** of the Hugging Face + peft path on a tiny locally
  built Qwen2 (skipped unless the `train` extras are installed).
- **Documentation tests** check that commands, configs and paths named in the
  docs exist.
- **Mutation testing** (`tools/mutation_test.py`, `make mutation`) reintroduces
  each historical bug into a copy of the code and requires the suite to fail.
