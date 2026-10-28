# Lessons: bugs, corrections, and falsified claims

Almost every failure in this project was **silent**: training ran, curves moved, and nothing raised. Each one below is now a regression test (see the README's mutation-testing table).

## Silent bugs

| # | bug | why it was silent | fix |
|---|---|---|---|
| 1 | mask started at `plen` | dropped the first generated token from every gradient | `plen-1`; asserted |
| 2 | `[\d.]+` number regex | `float("8.25.0.0")` raised mid-rollout and looked like a hang | `-?\d+(?:\.\d+)?` |
| 3 | NaN reward | `float('nan')` parses, flows through `exp()`, poisons every advantage in the group | `isfinite` guard |
| 4 | prompt/reward contract mismatch | the format reward never fired; the run stalled at 0% format | one-shot worked example |
| 5 | gradient checkpointing + adapter toggling | `CheckpointError`: 404 tensors saved vs 57 recomputed | checkpointing off, score one sequence at a time |
| 6 | partial-credit telemetry at each run's own λ | graded credit is λ·exp(−ρ/κ), so the metric scaled with the hyperparameter; ~93% of a reported "hacking signal" was arithmetic | fixed `telemetry_lam` |
| 7 | endpoint trend estimator | single steps average only G=6 samples: ~4× noisier than a window | `window_mean` |
| 8 | cache restored metrics but not weights | the "GRPO" model was silently the base model; the A/B showed no difference | `train_run` restores weights or retrains |
| 9 | reused a single-use context manager | peft's `disable_adapter()` is a generator; re-entry raised `AttributeError` | re-enterable `Policy.use_base()` |
| 10 | no test that reference ≠ policy | a no-op reference gives KL ≡ 0; the leash silently does nothing | validation test 7 |
| 11 | parser fallback in the reward path | correctness fired without the output contract; format 61.7% → 20.0% | `graded_gated` arm (unvalidated) |
| 12 | LoRA dropout at μ=1 | `logp_old` in eval mode, `logp_new` in train mode, so ρ ≠ 1 | `lora_dropout: 0.0`; `max |ρ−1|` logged |
| 13 | `get_logger` returned early if any handler existed | a second run in the same session never wrote its log file | one run = one log file |
| 14 | subgroup rows used the overall n for their standard error | 1-step significance reported as 5.4 SE instead of 3.3 SE | per-row n |

| 15 | growth labels computed from the exact rate, question shows a rounded revenue | training and evaluation ran normally; 11 of the 60 held-out labels cannot be derived from their question | dataset `version: 2` (v1 kept for reproduction) |
| 16 | before/after "z" divided a difference by the SE of one proportion, and per-question outcomes were not saved | the overall verdict happened to survive; subgroup claims did not | per-question records + exact McNemar test |
| 17 | training cache keyed by run name only | a changed config with the same name silently reused the old adapter | training fingerprint stored in `train.json` |
| 18 | pass@k sampling not seeded at the start of an evaluation | BEFORE and AFTER sampled numbers depended on how much randomness earlier (or cached) stages had used | `evaluate` seeds with `experiment.seed` |
| 19 | stop detection used only the tokenizer's EOS id | an answer ending on a second stop token (which may equal the pad id) would be scored into its padding; not observed in Run 6, found by review | `Policy.stop_token_ids()` merges tokenizer and generation config |

Bugs 13 and 14 were found while building the first repository; 15–19 during the v2.0.0 refactor.

## Claims falsified by the project's own experiments

1. **"The format reward is inert once high."** The affine-invariance theorem is true, but its precondition (saturation) was not met: format variance exceeded correctness variance at 60 steps.
2. **"Removing format is a clean ablation arm."** It was the only arm whose exact correctness did not improve.
3. **Three detector architectures for reward hacking.** v1 watched trends and missed a level shift; v2 compared learning to zero instead of to the control; v3 scored 3/3 only because of contaminated telemetry (bug 6).
4. **"Truncation at 112 tokens caused the baseline to drop from 27.5% to 16.7%."** Runs 4 and 6 measured identical baselines at 112 and 160 tokens. The cause was eval-set composition.
5. **"λ = 0.6 cost 7.5 pp at 1.2 SE."** With the correct n=40, it is 0.95 SE: suggestive, not significant.
6. **"GRPO narrows the pass@1↔pass@G gap."** In run 6 it widened by 1.4 pp. Inside the noise, but the prediction was not confirmed.
7. **"A4 shows the trained model answering correctly."** It chose the right formula (one year, not ten) but computed 4.8 × 1.06 = 5.184 instead of 5.088; the ±$0.10 tolerance around the label 5.09 scored it correct.
8. **"+26.7 pp at 4.1 SE; 1-step at 3.3 SE."** The SE was that of a single proportion. Paired, the overall gain is significant under any pairing of questions (McNemar p ≤ 0.011); the 1-step and 2-step gains are not established on their own (p up to 0.096).

## Principles that came out of this

- **Measure against a control, not against zero.** Reward hacking at λ = 0.6 looked like a healthy run in isolation.
- **The measurement stick must never move with the optimisation target.** Score every arm with the same metric at the same fixed parameters.
- **Level, trend, and relative-to-control are different questions.** Every failed detector answered the wrong one.
- **Read the completions.** Metrics said "all clear" several times while four paragraphs of model output showed the problem.
- **Mutation-test the tests.** Three of twelve reintroduced bugs initially passed the suite.
- **Quote the error bar with the number, for the right n, and pair what is paired.**
- **Check that every label can be derived from its own question.** A generator bug is invisible to every metric that trusts the labels.
