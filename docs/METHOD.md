# Method

## The objective

$$\mathcal{J}(\theta)=\mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\Big(\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)-\beta\,\mathbb{D}_{KL}\Big)\Big]$$

## Equation → code

| term | meaning | implementation |
|---|---|---|
| $o_1..o_G$ | G sampled completions of one question | `grpo.algorithm.rollout.generate_group` (one batched call) |
| $r_i$ | reward for completion i | `grpo.rewards.CompositeReward.score_group` (built from `reward.components`) |
| $\hat A_i=(r_i-\bar r)/\sigma_r$ | group-relative advantage; `None` if σ≈0 | `grpo.algorithm.advantages.group_advantages` (registered as `group_std`) |
| $\sum_t$ over completion tokens | score the model's own words only, positions `plen-1 .. L-2` | `grpo.algorithm.rollout.completion_mask`, `true_lengths` |
| $\pi_\theta$, $\pi_{ref}$ | policy and reference log-probs | `grpo.algorithm.rollout.per_token_logps(use_adapter=True/False)` |
| $\rho=\pi_\theta/\pi_{old}$ | importance ratio; exactly 1 at μ=1 without dropout | `grpo.algorithm.losses.grpo_loss` (registered as `grpo`) |
| $\mathbb{D}_{KL}\approx e^{\delta}-\delta-1$ | k3 estimator, δ = log π_ref − log π_θ, always ≥ 0 | `grpo.algorithm.losses.k3_kl` |
| $1/\lvert o_i\rvert$ | length normalisation | `grpo.algorithm.losses.grpo_loss` |
| update | LoRA parameters only, gradient-norm clipped, AdamW | `grpo.algorithm.step.grpo_step` |

The refactor into these modules did not change the update: `tests/test_algorithm.py::test_update_matches_the_original_implementation` checks loss, KL, clip fraction and the updated weights against values recorded from the pre-refactor code (exact on the recording machine; within float32 noise on other CPUs).

## Why the group mean is enough

The policy-gradient theorem gives ∇J = E[(R − b)∇log π_θ] for any baseline b that does not depend on the sampled answer, because E[b∇log π_θ] = b∇Σπ_θ = b∇1 = 0. Subtracting b leaves the gradient unbiased and lowers its variance. PPO learns b with a value network. GRPO uses the mean reward of the G attempts at the same question — a Monte-Carlo estimate of exactly that quantity, for free.

**Cost:** a noisier baseline than a trained critic, and every token in a completion receives the same advantage.

**At μ = 1** (one update per group, as in DeepSeekMath and TRL's default), the rollout policy and the policy being updated are the same weights, so ρ = 1 and the clip never binds. The gradient is then $\hat A_i\nabla\log\pi_\theta$ plus the KL pull: REINFORCE with the group mean as baseline. The loss *value* is about β × KL (advantages average to zero), so it is not a progress metric.

## Reward

$$r=\begin{cases}1 & |p-a^*|\le\tau\\ \lambda\exp\!\big(-\tfrac{|p-a^*|}{\kappa\max(1,|a^*|)}\big)&\text{otherwise}\\ 0&\text{unparseable}\end{cases}\quad+\;0.1\cdot\mathbb{1}[\text{``Answer:''}]$$

- Depends only on relative error, so it is scale-free.
- **λ < 0.5**: two partial credits must never outrank one exact answer, or guessing becomes reward-optimal.
- **Affine invariance**: Â is unchanged by r → αr + β (α > 0). Only differences between answers to the same question matter, and a component that scores every group member identically contributes exactly zero gradient — once it has actually saturated.
- **Consequence of dividing by the group std:** in a group where every answer is wrong, tiny differences in partial credit (or the format bonus alone) are scaled up to full-size advantages. The 10:1 weighting holds only in groups that mix right and wrong answers. (Liu et al. 2025 call this a question-level difficulty bias.)

Reward components (`grpo.rewards.components`, selected in `reward.components`):

| component | correctness | note |
|---|---|---|
| `correct_binary` | pass/fail | parser fallback allowed |
| `correct_graded` | partial credit | parser fallback allowed (the Run 6 reward) |
| `correct_gated` | partial credit **only** for `Answer: <number>` | proposed fix; not yet validated |
| `format` | — | 1 if the completion contains `Answer:` |

The four reward "arms" of the original project are combinations of these: `binary` = correct_binary + 0.1·format, `graded` = correct_graded + 0.1·format, `graded_nofmt` = correct_graded, `graded_gated` = correct_gated + 0.1·format.

## Provenance

| component | label | note |
|---|---|---|
| group sampling, group-relative advantage | **GRPO** | the defining idea |
| clipped surrogate | **GRPO** | inherited from PPO |
| KL to a frozen reference, k3 estimator | **GRPO** | as in DeepSeekMath |
| length-normalised token loss | **GRPO** | the original form; has a known length bias |
| sequence advantage on every token | **GRPO** | crude, unbiased |
| 4-bit base + LoRA | simplification | restricted policy space, quantisation noise |
| reference by restoring pristine LoRA tensors | simplification | exact: LoRA B = 0 *is* the base model |
| one question per step | simplification | noisier gradient than a batch of questions |
| gradient checkpointing off | simplification | required by adapter toggling |
| graded reward, λ bound, gated component | ours | |
| format reward at 10:1 | ours | anti-hacking constraint |
| one-shot prompt | ours | the reward could not fire without it |
| synthetic dataset | ours | four templates, 1–2 steps |

## Design decisions

- **Explicit adapter state.** `Policy` owns every transition — `snapshot`, `restore`, `reset`, `use_base` — instead of relying on a library context manager's lifetime.
- **Arm-independent telemetry.** Exact correctness is always the task's evaluation metric; partial credit is always measured at the fixed `evaluation.telemetry_lam`.
- **Paired schedule and paired evaluation.** Every run sees the same question sequence; before/after are compared question by question (McNemar).
- **Fingerprinted caches.** A cached stage is reused only if every setting that can change its output is unchanged.
- **Direct ratio telemetry.** `max |ρ−1|` is logged at the first inner epoch; clip fraction is only an indirect symptom.
