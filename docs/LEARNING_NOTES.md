> **Provenance note.** These notes were written before Run 6 and describe GRPO conceptually; they do not reference the package's code. For current numbers see [RESULTS.md](RESULTS.md). Superseded below, per [LESSONS.md](LESSONS.md): the truncation explanation for the baseline drop and for the 100% → 57.5% format drop (runs 4 and 6 contradict it); the "λ = 0.6 cost 7.5 points" result (0.95 SE at n=40: suggestive, not significant); and every standard-error verdict, which used the SE of one proportion rather than a paired test. All runs quoted here used dataset version 1, whose growth labels are often inconsistent with the question.

> **Note on the companion notebook.** "`grpo_industry_grade.ipynb`" below refers to an earlier notebook; the run reported in RESULTS.md came from `notebooks/archive/grpo_master.ipynb`, and the maintained implementation is the `src/grpo` package.

# GRPO — Master Learning & Reference Notes

*Companion to `grpo_industry_grade.ipynb`. Every number cited here comes from a run in this project.*

**Legend:** 🔴 **Must Know** · 🟡 **Important** · 🔵 **Advanced / Optional**

**The running analogy used throughout:** you are a coach training a student who learns only from the grades you give. The student attempts the same problem six times, you grade all six, and they infer from the grades what to do more of.

---

# Part 1 — Fundamentals and Intuition

## 1.1 What GRPO is 🔴

**Simple intuition.** Ask the model the same question six times. Grade all six answers. Tell the model: *"whatever you did in the above-average answers, do more of that; whatever you did in the below-average answers, do less."* Repeat thousands of times.

That's the entire algorithm. Everything else is machinery making each phrase precise.

**Why it matters.** GRPO (Group Relative Policy Optimization) is the algorithm behind DeepSeekMath and DeepSeek-R1. Its innovation is deleting the most expensive component of PPO — the critic network — and replacing it with a free arithmetic trick: compare each attempt to the average of its siblings. That single substitution roughly halves the memory bill, which is why a 0.5B model can be RL-trained on a free Colab T4.

**The name decoded:**
- **Group** — six attempts at the same question form a group
- **Relative** — each attempt is scored *relative to its group's average*, not against an absolute standard
- **Policy Optimization** — we adjust the model (the "policy") to make good attempts more likely

**Common mistake.** Thinking GRPO teaches the model new skills. It doesn't. It **redistributes probability** toward behaviors the model already occasionally produces. This is the project's single most important finding — see 1.2.

---

## 1.2 The central law: RL elicits, pretraining creates 🔴

**Simple intuition.** You can coach a club tennis player into a strong one — they occasionally hit great shots you can reinforce. You cannot coach someone who has never held a racket by shouting scores at them.

**The evidence from this project.** A 600K-parameter character-level model was trained with a *correct* GRPO implementation for 300 steps:

| metric | result |
|---|---|
| Format compliance | 94% → 97.5% ✅ |
| Correctness | **0% → 0%** ❌ |

The algorithm worked perfectly. The model learned formatting because formatting *sometimes happened*. It learned zero arithmetic because a character-level model **never once produced a correct calculation** in ~2,400 sampled attempts. There was nothing to reinforce.

The identical algorithm on pretrained Qwen2.5-0.5B: **27.5% → 52.5%**.

**Why it matters.** This is why real reasoning models start from large pretrained LLMs. It also gives you a diagnostic: if your correctness curve is flat, ask *"does the base model ever get this right?"* before touching hyperparameters.

**Common mistake.** Blaming the algorithm, the learning rate, or the reward when the real problem is that the base model cannot do the task at all. Check `pass@G` first (Section 11.2).

---

# Part 2 — RL and Policy Basics You Actually Need

## 2.1 The policy 🔴

**Simple intuition.** A language model is a probability machine. At every word it holds a probability for each possible next word. "Training" means nudging those probabilities.

**Why it matters.** This framing converts a fuzzy goal ("make the model better") into something calculus can act on ("shift probability mass"), because probabilities are differentiable in the model's weights.

**Mathematics.**

$$\pi_\theta(o \mid q) = \prod_{t=1}^{T} \pi_\theta(o_t \mid q, o_{<t})$$

Read: *the probability of the whole answer is the product of each token's probability given everything before it.* Products are awkward, so we work in logs:

$$\log \pi_\theta(o \mid q) = \sum_{t=1}^{T} \log \pi_\theta(o_t \mid q, o_{<t})$$

**Implementation.** `per_token_logps()` returns exactly the terms of that sum — one log-probability per position.

**Example.** Writing "Answer: 25" is a path through ~5 forks in the road. Each fork has a probability. The path's probability is their product.

**Common mistake.** Thinking of the model as producing *an answer*. It produces *a distribution*; the answer is one sample from it. Almost every GRPO concept depends on that distinction.

---

## 2.2 The policy gradient theorem 🟡

**Simple intuition.** *Sample answers from the model; for each one, push its probability up in proportion to its reward.* Good answer → strong push toward it.

**Mathematics (derived in four steps).**

We want to maximize expected reward: $J(\theta) = \mathbb{E}_{o \sim \pi_\theta}[R(o)]$.

1. Write it as a sum: $J(\theta) = \sum_o \pi_\theta(o) R(o)$
2. Differentiate — only $\pi_\theta$ depends on $\theta$: $\nabla J = \sum_o \nabla\pi_\theta(o) R(o)$
3. **The log-derivative trick.** Since $\nabla \log f = \nabla f / f$, we have $\nabla\pi_\theta = \pi_\theta \nabla\log\pi_\theta$. Substitute:

$$\nabla J = \sum_o \pi_\theta(o)\,\nabla\log\pi_\theta(o)\,R(o) = \mathbb{E}_{o\sim\pi_\theta}\big[R(o)\,\nabla_\theta \log \pi_\theta(o)\big]$$

**Why this is beautiful.** The grader $R$ is a Python function — not differentiable. It never needed to be. It only *scales* the push. This is what makes RL with verifiable rewards possible.

**Common mistake.** Believing the reward function must be differentiable. It must not be — ours is regex + comparison.

---

## 2.3 Baselines: why we subtract the mean 🔴

**Simple intuition.** If all rewards are positive, *every* answer gets pushed up — bad ones just get pushed less. Learning works on average but is horribly noisy. Subtracting the average means above-average answers go **up** and below-average answers go **down**.

**Mathematics.** For any baseline $b$ that doesn't depend on the answer:

$$\nabla J = \mathbb{E}\big[(R(o) - b)\,\nabla\log\pi_\theta(o)\big]$$

This is **legal** because the correction term vanishes:

$$\mathbb{E}[b\,\nabla\log\pi_\theta] = b\sum_o \pi_\theta \frac{\nabla\pi_\theta}{\pi_\theta} = b\,\nabla\!\sum_o \pi_\theta(o) = b\,\nabla 1 = 0$$

**Unbiased, but lower variance.** That's the whole point of a baseline.

**Why it matters.** *What baseline to use* is the question that separates PPO from GRPO. PPO trains a neural network to predict it. **GRPO uses the group's own mean.** Free, no extra model, half the memory.

**Common mistake.** Thinking the baseline biases the gradient. The proof above shows it cannot, as long as it doesn't depend on the specific answer being scored.

---

# Part 3 — Groups, Rollouts, Rewards

## 3.1 Rollout 🔴

**Simple intuition.** One attempt at a question — the student writing one solution on scratch paper.

**Why it matters.** RL learns from the model's *own output*, not from provided answers. No rollouts, no data.

**Implementation.** `generate_group()` — one batched call with `num_return_sequences=G`. Generation is **~85% of wall-clock time**, so batching here is the biggest speed win available.

**Common mistake.** Generating rollouts in a Python loop instead of one batched call. Roughly 3–5× slower for identical results.

---

## 3.2 Group 🔴

**Simple intuition.** The six attempts at the *same* question, generated together. The student solves one problem six times, slightly differently each time.

**Why it matters.** The variation between attempts **is** the learning signal. There is no other source. If all six were identical, there'd be nothing to compare.

**Implementation.** `CFG.group_size = 6`. Sampling at `temperature=1.0` creates the diversity.

**Example.** Six attempts at "what's the operating margin?" might produce: 39 ✓, 38.98 ✓, 37.33 (near miss), 256.5 (inverted the ratio), 30 (guess), and one refusal.

**Common mistake — worth internalizing.** Reducing group size to save time. With fewer draws, **ties become more likely**, which increases wasted steps and can cost you more than the time saved.

---

## 3.3 Degenerate groups (the "skip") 🔴

**Simple intuition.** All six attempts scored identically → nothing to compare → zero information → skip the update.

**Mathematics.** All rewards equal ⇒ $\sigma \approx 0$ ⇒ the advantage $\frac{r_i - \bar r}{\sigma}$ is undefined. Genuinely zero information, not a numerical inconvenience.

**Implementation.** `group_advantages()` returns `None` when `std < 1e-6`; the runner counts it.

**Why this is your master diagnostic.** The same number meant three different things at different points in this project:

| skip rate | meaning | when we saw it |
|---|---|---|
| 75–100% | Task too hard — all attempts wrong | the stalled first Qwen run |
| 0–10% | **Healthy learning band** | after the prompt fix |
| rising again | Task too easy — all attempts *right* | as format saturated |

**Common mistake.** Assuming a rising skip rate means training is failing. It can equally mean training *succeeded* and the questions are now too easy — which is what happened in this project when format compliance saturated and stopped providing contrast.

---

## 3.4 Reward 🔴

**Simple intuition.** The grade on one attempt.

**Why it matters.** In RL the reward *is* the specification of what you want. The algorithm is generic; **all domain knowledge lives in the reward.** This is where the real engineering craft sits.

**Implementation.** Our registry:

```
reward = 1.0 × correctness  +  0.1 × format
```

**The 10:1 ratio is an anti-hacking constraint**, not a stylistic choice. An auxiliary reward must never be able to outbid the terminal objective — otherwise "always format, never compute" becomes optimal.

**Verifiable rewards.** Ours are *programs with ground truth* (regex-extract the number, compare within tolerance). They cannot be sweet-talked, only satisfied. This regime — RL from Verifiable Rewards (RLVR) — is exactly what DeepSeek-R1 used for math and code.

**Common mistake — the silent killer.** A broken reward function **does not crash**. Training runs, curves move, and the model quietly learns something else. Hence: every component is a pure function, exception-isolated, and validated offline before it touches a GPU.

---

## 3.5 Graded (dense) rewards and λ 🔴

**Simple intuition.** Instead of ✓/✗, give partial credit that shrinks with distance. Six wrong attempts stop being six identical zeros and become six distinguishable near-misses.

**Why it matters.** Under binary reward, on a group where the truth was 39%:

| attempt | binary reward | graded reward |
|---|---|---|
| 37.33 (near miss, right method) | **0.1** | **0.342** |
| 256.5 (inverted the ratio) | **0.1** | **0.100** |

Binary told the model these two mistakes were *equally bad*.

**Mathematics.**

$$r(p, a^*) = \begin{cases}
1.0 & |p - a^*| \le \tau \quad \text{(exact)}\\[4pt]
\lambda\exp\!\left(-\dfrac{|p-a^*|}{\kappa \max(1,|a^*|)}\right) & \text{(partial)}\\[4pt]
0 & p \text{ unparseable}
\end{cases}$$

Substituting relative error $\rho = |p - a^*|/|a^*|$ gives $r = \lambda e^{-\rho/\kappa}$ — **the reward depends only on relative error**, so a 25% miss scores identically whether the truth is 39% or \$8B. That's "scale-free."

**Measured effect.**

| | binary | graded |
|---|---|---|
| Distinct reward values per group | 2.50 | **5.07** |
| Degenerate (skipped) steps | 3.3% | **0.0%** |

**Common mistake.** Setting λ too high — see 8.2, where λ=0.6 cost 7.5 percentage points.

---

# Part 4 — Advantages and Group-Relative Normalization

## 4.1 The advantage 🔴

**Simple intuition.** How much better an attempt was than its group's average, measured in units of the group's spread. Grading on a curve — within the student's own six attempts.

**Mathematics.**

$$\hat A_i = \frac{r_i - \operatorname{mean}(r_1 \dots r_G)}{\operatorname{std}(r_1 \dots r_G)}$$

Dividing by the standard deviation normalizes across questions: a 0.2 reward edge means more in a tight group than in a wildly varied one.

**Implementation.** `group_advantages()`. A properly computed advantage vector has **mean ≈ 0 and std ≈ 1** — verified in the notebook's Part 4 (measured: mean = +2.98e-08, std = 0.9998).

**Example.** Rewards `[1.1, 0.1, 1.1, 0.1, 0.1, 0.0]` → advantages `[+1.41, −0.65, +1.41, −0.65, −0.65, −0.86]`. The two correct answers get pushed up hard; everything else gets pushed down.

**Common mistake.** Forgetting the `+ eps` in the denominator, or setting it too large. It prevents division blow-ups but slightly breaks the invariance in 4.2.

---

## 4.2 Affine invariance — the theorem that governs reward design 🔴

**Simple intuition.** GRPO only ever sees *differences within a group*. Multiply every reward by 100 or add 7 to all of them — the rankings don't move, so the learning doesn't change.

**Mathematics.** For any $\alpha > 0$:

$$\hat A_i = \frac{(\alpha r_i + \beta) - \overline{(\alpha r + \beta)}}{\operatorname{std}(\alpha r + \beta)} = \frac{\alpha(r_i - \bar r)}{\alpha \operatorname{std}(r)} = \frac{r_i - \bar r}{\operatorname{std}(r)}$$

**Two consequences you will use constantly:**

1. **Absolute reward scale is irrelevant.** Whether correctness is "worth 1" or "worth 100" changes nothing. **Only ratios between components matter.** Stop agonizing over magnitudes.

2. **A component scoring every group member identically contributes exactly zero gradient.** It is the $\beta$ in the transform above — it cancels perfectly. Like a teacher adding 5 bonus points to every paper: nobody's relative grade moved.

**⚠️ The precondition matters — and it refuted a claim in this project.** At 60 steps from base, format compliance was *not* saturated:

| component | within-group variance |
|---|---|
| format | **0.2167** |
| correctness | 0.1293 |

Format was the **larger** signal, and the ablation arm that removed it was the only one whose exact correctness did not improve. The theorem is true; its *saturation* precondition was not met.

**Common mistake.** Retiring a component because the theorem says saturated components are inert — without first *measuring* whether it has actually saturated. Check `component_variance()`.

---

# Part 5 — Probability Ratios and Clipping

## 5.1 The importance ratio ρ 🟡

**Simple intuition.** How much more likely the *current* model makes a token, compared to the model that actually **generated** it.

**Mathematics.**

$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid \cdot)}{\pi_{\theta_{old}}(o_{i,t} \mid \cdot)}$$

**Why it exists.** Generation is expensive (~85% of wall-clock). If you reuse one batch of rollouts for several gradient steps, the model has moved and the data is slightly stale. The ratio re-weights it honestly. That's importance sampling.

**Implementation reality — know this cold.** With **μ = 1** (one update per rollout batch), $\pi_{old} = \pi_\theta$ at loss time, so **ρ ≡ 1 exactly** and the machinery is dormant. Our notebook measures this: clip fraction across μ = 1..4 is **`[0.0, 0.917, 0.917, 0.917]`** — provably inert at μ=1, engaging the instant rollouts are reused.

**Common mistake.** Claiming clipping "protects your training" when μ=1. It cannot. It is inert by construction, and saying otherwise in an interview is a red flag to anyone who knows the algorithm.

---

## 5.2 Clipping 🟡

**Simple intuition.** A cap on how far one batch may push the model. A coach who never lets one good practice session rewrite the entire playbook.

**Mathematics.**

$$\min\big(\rho\hat A,\ \operatorname{clip}(\rho, 1-\epsilon, 1+\epsilon)\hat A\big)$$

Once the policy has already moved a token's probability by more than ±20% ($\epsilon = 0.2$) relative to the data-generating policy, stop pushing. It's a trust region enforced with `min` and `clamp`.

**The asymmetry (worth understanding).** The `min` means clipping only bites when it would *reduce* the objective:

| ρ | advantage | clipped? |
|---|---|---|
| 1.5 | +1 | ✅ yes — already moved far enough in a good direction |
| 0.5 | +1 | ❌ no — still room to increase |
| 1.5 | −1 | ❌ no — pushing down is fine |

**Common mistake.** Assuming clipping always applies. It's a one-sided guard, and at μ=1 it never fires at all.

---

# Part 6 — KL Divergence and the Reference Policy

## 6.1 The reference policy 🔴

**Simple intuition.** A frozen snapshot of the model *before* RL started — the anchor we measure drift against.

**Why it matters.** Without a fixed reference point, "how far has the model changed?" has no answer.

**Implementation — the elegant part.** We don't store a second model. LoRA adapters are *additive corrections* to frozen base weights, so:

```python
with model.disable_adapter():
    logp_ref = per_token_logps(seqs)   # this IS the base model
```

One model in memory plays both roles. Not an approximation — turning off the adapters gives you exactly the pre-RL model.

**Common mistake (that bit this project).** Combining this with gradient checkpointing. Checkpointing discards activations and recomputes them during backward, but `disable_adapter()` changes the model state between the two passes:

```
CheckpointError: 404 tensors saved during forward vs 57 during recomputation
```

Fix: disable checkpointing, compensate by scoring one sequence at a time.

---

## 6.2 KL divergence and the k3 estimator 🟡

**Simple intuition.** A number measuring how differently the model behaves now versus the reference. A fitness tracker showing how far the student's habits have shifted from day one.

**Why it matters.** Chasing reward hard enough can drift a model into degenerate text that games the grader while forgetting how to write. KL is how you detect that.

**Mathematics — the k3 estimator.** With $\delta = \log\pi_{ref} - \log\pi_\theta$:

$$\mathbb{D}_{KL} \approx e^{\delta} - \delta - 1$$

**Why not the naive $-\delta$?** Because it can go **negative** (measured: naive min = −3.00 over δ ∈ [−3, 3]), and a KL estimate that goes negative destabilizes training. The k3 form is provably ≥ 0 everywhere (measured min = 0.00), zero at δ=0, low-variance, and differentiable.

**Implementation.** `kl = torch.exp(delta) - delta - 1.0`

**Measured values in this project:** ~0.029 in healthy runs; 0.062 (2.14×) when β was set to 0.

**Common mistake.** Using `logp_ref - logp_new` directly as "the KL." It's an unbiased estimate of KL but with much higher variance and no non-negativity guarantee.

---

# Part 7 — The Loss

## 7.1 The complete GRPO objective 🔴

$$\mathcal{J}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\Big(\min\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t}, 1{-}\epsilon, 1{+}\epsilon)\hat A_i\big) - \beta\,\mathbb{D}_{KL}\Big)\right]$$

**Reading it left to right:** average over the group → average over each response's tokens → (the clipped policy-gradient term) minus (the KL leash).

**Implementation.**

```python
per_token = torch.min(s1, s2) - cfg.beta * kl
per_resp  = (per_token * mask).sum(1) / mask.sum(1)   # length-normalized
loss      = -per_resp.mean()                          # minus: maximize → minimize
```

**Common mistake.** Forgetting the minus sign. Optimizers *descend*; we want to *ascend* the objective.

---

## 7.2 Length normalization 🟡

**Simple intuition.** Divide each response's score by its length, so long answers don't get more influence just for being long.

**Why it matters.** Without `1/|o_i|`, a 150-token answer contributes 3× the gradient of a 50-token one purely by token count — and the model learns that **verbosity buys influence**.

**Implementation.** `/ mask.sum(1)`

---

## 7.3 Masking — where silent bugs live 🔴

**Simple intuition.** Only grade the model's *own words*, not the question you handed it.

**The tricky bit.** Models predict the *next* token, so output position $t$ holds the probability of token $t+1$. Completion tokens $x_{plen} \dots x_{L-1}$ are therefore scored at positions $plen-1 \dots L-2$.

**Implementation.**

```python
mask[i, plen-1 : L_i-1] = 1.0
```

**Common mistake — the most dangerous bug in this project.** Masking from `plen` instead of `plen-1` **silently drops the first generated token from every gradient update**. For an `Answer:`-anchored reward, that is the most valuable token in the sequence. No exception is raised. Training runs, curves move, learning is crippled.

Also: score **through** the EOS token. Stopping is a behavior; if it earned no reward, rambling would never be penalized.

---

# Part 8 — The Parameters

## 8.1 β (beta) — the KL leash 🔴

| | |
|---|---|
| **What** | How strongly to pull the model back toward the reference |
| **Our value** | 0.04 (DeepSeekMath default) |
| **Analogy** | The length of a dog's lead |
| **Where** | `per_token = min(s1,s2) - beta * kl` |

| setting | effect |
|---|---|
| β = 0 | No leash. **Measured: KL more than doubled** (0.062 vs 0.029) |
| β = 0.04 ✅ | Healthy — KL stayed at 0.029 |
| β too large | Model pinned to its original self, cannot improve |

---

## 8.2 λ (lambda) — max partial credit 🔴

| | |
|---|---|
| **What** | How much credit a *wrong but close* answer earns |
| **Our value** | 0.3 |
| **Analogy** | A teacher giving points for correct working despite a wrong final number |
| **Where** | `lam * exp(-rho/kappa)` in the reward |

**The safety bound.** Computing must stay reward-optimal:

$$\mathbb{E}[r \mid \text{compute}] = q + (1-q)\lambda\delta_c \quad>\quad \mathbb{E}[r \mid \text{guess}] = \lambda\delta_g$$

Conservatively this gives $\lambda < q$ (the success rate); the crude sufficient version is $2\lambda < 1$ — **two partial credits must never outrank one exact answer.**

| setting | measured effect |
|---|---|
| λ = 0 | Back to pass/fail; degenerate groups return |
| λ = 0.3 ✅ | Safe — computing beats guessing |
| λ ≥ 0.5 ❌ | "Guess plausibly" becomes optimal |
| **λ = 0.6** | **−7.5 pp accuracy, learning rate roughly halved** |

**This is a safety parameter, not a tuning knob.**

---

## 8.3 λ vs β — the confusion to avoid 🔴

These sound similar and do completely unrelated jobs.

| | **λ (lambda)** | **β (beta)** |
|---|---|---|
| Controls | How generous the *grading* is | How tight the *leash* is |
| Lives in | The reward function | The loss function |
| Guards against | Guessing instead of computing | Drifting into gibberish |
| Our value | 0.3 | 0.04 |
| When broken | −7.5 pp accuracy | KL doubled |
| One-liner | **Grading generosity** | **Safety leash** |

λ is about *what counts as a good answer*. β is about *how far the model may change*.

---

## 8.4 The other parameters 🟡

| parameter | value | intuition | if changed |
|---|---|---|---|
| **Group size G** | 6 | Attempts per question | Larger: better comparisons, slower. Smaller: **more ties** — costs more than it saves |
| **ε (clip range)** | 0.2 | Max ±20% probability move per batch | Inert at μ=1 regardless |
| **Learning rate** | 1e-5 | Step size, LoRA-appropriate | Too high: instability; too low: no visible learning in 60 steps |
| **Temperature** | 1.0 | Rollout diversity — the *only* exploration mechanism | Lower: attempts converge, ties rise. Higher: more noise, more contrast |
| **κ (kappa)** | 0.2 | How far partial credit reaches (~2.5× relative error) | Too small: credit dies instantly. Too large: signal flattens |
| **μ (inner epochs)** | 1 | Updates per rollout batch | μ>1: reuses expensive rollouts, **activates clipping** |

---

# Part 9 — How the Model Actually Learns

## 9.1 Gradient and update 🔴

**Simple intuition.** The gradient is the coach's feedback (which direction to change). The update is the student actually changing the habit.

**Implementation.**

```python
loss.backward()                    # compute gradients
clip_grad_norm_(params, 1.0)       # cap the total step size
optimizer.step()                   # apply — LoRA adapters ONLY
```

The 4-bit base model is frozen. Only ~1% of parameters (the LoRA adapters) change.

---

## 9.2 Verifying the gradient is correct 🔴

**This is the check almost nobody runs, and it's the one that matters.**

**The test.** Build a tiny model, give one sequence a positive advantage and three negative, take one GRPO step, then measure whether log-probabilities moved as the math predicts.

**Measured result from the notebook (Part 4):**

| sequence | advantage | log-prob before → after | Δ |
|---|---|---|---|
| seq0 | **+1.500** | −2.9891 → −2.5455 | **+0.444** ✅ |
| seq1 | −0.500 | −3.0823 → −3.1512 | −0.069 ✅ |
| seq2 | −0.500 | −3.0430 → −3.1797 | −0.137 ✅ |
| seq3 | −0.500 | −3.0287 → −3.1415 | −0.113 ✅ |

**corr(advantage, Δ log-prob) = 0.9948**

**Why this matters.** A flipped sign, a misaligned mask, or a mis-normalized advantage would all fail this test — and **nothing else in your pipeline would notice.** Training would just learn slowly, and you'd blame the reward or the hyperparameters.

**Common mistake.** Validating inputs (rewards, parsing, masks) but never the update itself.

---

# Part 10 — The Complete Training Loop

## 10.1 One GRPO step, as a sentence 🔴

> Sample a question → sample 6 attempts → grade them → compute group-relative advantages (skip if all tied) → score every token's log-probability under the policy and, adapters-off, under the reference → assemble the clipped, KL-penalized, masked, length-normalized loss → backprop into the LoRA adapters only → repeat.

## 10.2 As a diagram

```
QUESTION
   ↓
one-shot PROMPT  ← defines the contract the reward checks
   ↓
sample G=6 attempts at temperature 1.0  ← the only exploration
   ↓
REWARD REGISTRY: 1.0×correctness + 0.1×format
   ↓
group-relative advantage (r − mean)/std  ← replaces PPO's critic
   ↓  (skip if all six tie — no contrast, no gradient)
clipped surrogate − β·KL, masked to completion tokens, length-normalized
   ↓
update LoRA adapters only
   ↓
TELEMETRY: skip rate · component variance · KL · clip fraction · pass@G
```

## 10.3 The prompt is part of the loop 🔴

**Finding #2 of this project.** The first Qwen run **stalled completely**: format 0%, 75–100% of steps wasted, accuracy unchanged.

**Cause.** A 0.5B model reads *"give your answer as `Answer: <number>`"* and outputs a bare number anyway. The format reward therefore **never fired once**, and no step-by-step reasoning was elicited.

**Fix.** One worked example added as a prior conversation turn.

| metric | before | after |
|---|---|---|
| Baseline accuracy | 7.5% | **27.5%** |
| Format compliance | 0% | **100%** |
| Wasted steps | 80% | **10%** |
| Final accuracy | — | **52.5%** |

**The generalization:** the prompt defines the *contract* the reward function checks. If they disagree, the model is punished for obeying instructions and the reward can never fire. **Small models ignore format descriptions but reliably copy format demonstrations.**

---

# Part 11 — Evaluation and Interpreting Results

## 11.1 Greedy vs sampled evaluation 🔴

**Training samples; evaluation takes the argmax.** Training samples to *explore* — the variation is the learning signal. Evaluation takes the most likely token to measure what the policy actually *believes*, noise-free.

**Common mistake.** Reporting sampled training accuracy as your headline. In this project, one run showed sampled accuracy 5.8% → 20% while greedy eval stayed **flat** — the "improvement" was luck in which questions got drawn.

---

## 11.2 pass@k 🟡

**Simple intuition.** pass@1 = "does the model usually get it right?" pass@G = "does *any* of G attempts get it right?"

**Mathematics (unbiased estimator).**

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$

where $n$ = samples drawn, $c$ = number correct.

**Why this pair is the perfect test of GRPO's central thesis.** GRPO cannot create capability, only concentrate it. So theory predicts **the pass@1 ↔ pass@G gap should narrow during training**: the model becomes reliable at things it could already occasionally do.

**How to read the gap:**
- **Wide gap remaining** → capability exists, more RL would help
- **Closed gap** → you've hit the model's ceiling; you need a bigger model or better data, not more RL

---

## 11.3 The four curves and what they mean 🔴

| curve | healthy pattern | warning sign |
|---|---|---|
| **Exact correctness** | Rising | Flat → check pass@G: is the capability there at all? |
| **Skip rate** | 0–15% | >50% (too hard) or rising late (too easy) |
| **KL** | Growing smoothly to ~0.03 | Spiking (instability) or flat at 0 (nothing learning) |
| **Format** | Rising then saturating | Flat at 0 → **the reward can't fire** (check the prompt) |

---

## 11.4 Statistical honesty 🔴

**The binding constraint in this project.** At n=60 eval samples, the standard error on accuracy is **≈ ±6 pp**. The measured graded-vs-binary gap was 8.3 pp — about **1.4 standard errors**, single seed.

**Therefore:**
- The *mechanism* result (skip rate 3.3% → 0.0%, averaged over 60 steps) is **solid**
- The *accuracy* result (43.3% vs 35.0%) is **suggestive, not established**

**Common mistake.** Quoting an 8-point improvement from a single seed at n=60 as a finding. Prefer metrics averaged over many *steps* (like skip rate) over metrics from few *eval samples*.

---

# Part 12 — Common Mistakes and Misconceptions

## 12.1 Bugs that fail silently 🔴

Every one of these was found in this project. **None raised an exception.**

| bug | what happens | fix |
|---|---|---|
| **Mask off-by-one** | First generated token dropped from every gradient | `plen-1`, not `plen` |
| **Fragile regex** | `[\d.]+` matches `8.25.0.0` → `float()` raises mid-rollout → looks like a hang | `-?\d+(?:\.\d+)?` |
| **NaN in reward** | `float('nan')` parses, propagates through `exp()`, poisons every advantage | Explicit `math.isfinite()` guard |
| **Truncated completions** | `max_new_tokens` too low cuts answers before `Answer:` — format fell 100% → 57.5% | 160, not 112 |
| **Dead reward component** | Component never fires; you think it's contributing | Track `component_variance()` |
| **Saturated component** | Contributes *exactly zero* gradient, invisibly | Same check |

---

## 12.2 Measurement mistakes 🔴

**The measurement stick must never move with the target.** Two versions of this bit this project:

1. **Scoring each arm with its own reward.** If arm B optimizes graded credit, it must still be *measured* on binary correctness — otherwise you're comparing rulers of different lengths.

2. **Telemetry that scales with a hyperparameter.** Partial-credit mass was computed with each run's *own* λ. Since graded credit is literally $\lambda e^{-\rho/\kappa}$, doubling λ doubled the metric with **zero behavioral change** — contaminating ~93% of an apparent "signal." Fix: always measure at a fixed reference λ.

**Noisy estimators.** Using single-step endpoints (`series[-1] - series[0]`) for a trend, where each point averages only G=6 samples. Simulation: **3.8× noisier** than a 15-step window.

---

## 12.3 Conceptual misconceptions 🔴

| misconception | reality |
|---|---|
| "GRPO teaches new skills" | It redistributes probability toward existing behaviors |
| "Clipping protects my training" | At μ=1, ρ ≡ 1 and clipping is **provably inert** |
| "Reward scale matters" | Only *ratios between components* matter (affine invariance) |
| "A saturated component is safe to keep" | It contributes exactly zero — but **verify saturation first** |
| "Reward hacking causes collapse" | It causes **slower learning that looks healthy** |
| "More gradient steps = more capability" | Not automatic — check whether accuracy actually followed |

---

## 12.4 Reward hacking presents as slower learning 🔴

**The most important practical lesson in this project.**

| run | held-out accuracy | learning gain |
|---|---|---|
| control (λ=0.3) | **42.5%** | +0.145 |
| hacked (λ=0.6) | **35.0%** | +0.077 |

The hacked run **still improved**. Its own curves look like a healthy training run. It just learned at roughly half the rate and ended 7.5 points lower.

**Run in isolation, you would have shipped it.**

**The practical rule: always pair a reward change with a control run.** The damage is *relative underperformance*, which is invisible in absolute terms.

---

# Part 13 — GRPO vs PPO vs SFT

## 13.1 The comparison table 🔴

| | **SFT** | **PPO** | **GRPO** |
|---|---|---|---|
| Learns from | One reference answer | Own sampled attempts | Own attempts, **in groups** |
| Signal | Match tokens exactly | Reward + learned critic | Reward vs **group mean** |
| Baseline | None needed | **Value network** (2nd model) | **Group mean** (free) |
| Models in memory | 1 | Policy + critic + reference | Policy + reference (free via adapters) |
| Can discover new strategies? | ❌ Only imitates | ✅ | ✅ |
| Multiple valid answers? | ❌ Punished | ✅ | ✅ |
| Failure modes | Can't exceed its labels | Critic instability, cost | Blunt token credit, degenerate groups |

## 13.2 The one-line versions 🔴

- **SFT teaches the *path*; RL rewards the *destination*.**
- **GRPO is PPO with the critic replaced by "compare siblings"** — trading statistical finesse for half the memory, which is the right trade when rewards are verifiable and hardware is limited.

## 13.3 The honest trade-off 🟡

GRPO applies the *whole answer's* advantage to *every token* — the correct calculation and the filler words alike. Crude, but unbiased: across thousands of groups, tokens that genuinely cause success appear disproportionately in high-advantage answers.

PPO's critic can (in principle) tell tokens apart. **That's the real cost of GRPO's simplicity**, and it's what a strong interviewer will probe.

---

# Part 14 — Practical and Industry Considerations

## 14.1 Making it fit on cheap hardware 🟡

| technique | saving | cost |
|---|---|---|
| **4-bit base weights** | ~2 GB → ~0.5 GB | Quantization noise |
| **LoRA adapters** | ~1% of params train | Restricted policy space vs full fine-tuning |
| **Reference via `disable_adapter()`** | Eliminates a second model entirely | None — it's exact |
| **Chunked log-prob scoring** | Bounds the `[G, T, 152k-vocab]` logit spike | Slightly slower |

**The non-obvious memory fact:** on a real LLM the hog isn't the weights, it's the **logits**. Scoring `[G, T]` sequences produces a `[G, T, vocab]` tensor — ~1.5 GB in fp32 for G=6, T=400.

## 14.2 Rewards are code, therefore rewards need CI 🟡

- Each component is a **versioned artifact** with regression tests
- A reward change **invalidates prior checkpoint comparisons** and must be recorded in every model's metadata
- The single most common production failure in RLVR pipelines: comparing models trained under different reward versions

## 14.3 Reliability 🟡

- **Never let a reward component raise.** Wrap, default to 0, count the failure, alert above ~1%. An exception inside a rollout loop halts training silently — it looks like a hang, not an error.
- **Checkpoint to persistent storage.** Colab disconnects are a when, not an if.
- **Deployment:** the inference-time validator should **share the parser** with the training-time reward. Divergence between "what training rewarded" and "what production validates" is a subtle, expensive bug class.

## 14.4 Security 🔵

Reward functions consume **untrusted model output**. Regex parsing is safe; `eval()` on model output would be catastrophic. And a sufficiently optimized policy can learn to emit text that *breaks your parser in its favor* — **the reward function is an attack surface, and the attacker is your own model.**

## 14.5 Known limitations of this implementation 🟡

- **Single seed** everywhere; no variance estimates
- **Eval n=40–60** → ±6–8 pp, larger than several measured effects (**the binding constraint**)
- **Runs stop while still improving**
- **Clipping inert** at the default μ=1
- **All-correct degeneracy unaddressed** — partial credit fixes all-*wrong* groups; as accuracy rises, all-*correct* groups dominate. Analytically the benefit falls from ~97% at 27% accuracy to ~10% at 65%. That needs a **difficulty curriculum**, not a better reward.
- **Synthetic data**, four templates

---

# ONE-PAGE CHEAT SHEET

## The equation

$$\mathcal{J}(\theta) = \mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t}\Big(\min\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t}, 1{\pm}\epsilon)\hat A_i\big) - \beta\,\mathbb{D}_{KL}\Big)\right]$$

$$\hat A_i = \frac{r_i - \operatorname{mean}(r)}{\operatorname{std}(r)} \qquad \rho_{i,t} = \frac{\pi_\theta}{\pi_{old}} \qquad \mathbb{D}_{KL} \approx e^{\delta} - \delta - 1$$

## The parameters

| symbol | name | value | one-liner |
|---|---|---|---|
| **G** | group size | 6 | attempts per question |
| **λ** | max partial credit | 0.3 | grading generosity — must be < 0.5 |
| **β** | KL coefficient | 0.04 | safety leash length |
| **ε** | clip range | 0.2 | max ±20% probability move |
| **κ** | decay scale | 0.2 | how far partial credit reaches |
| **μ** | inner epochs | 1 | updates per rollout (ρ ≡ 1 at μ=1) |
| **lr** | learning rate | 1e-5 | LoRA-appropriate step size |
| **T** | temperature | 1.0 | rollout diversity — the only exploration |

## The invariants

1. **Only within-group differences matter** — scale and shift are invisible
2. **Uniform components contribute exactly zero** — verify saturation before retiring
3. **Tied groups teach nothing** — skip and count
4. **The measurement stick must never move with the target**

## The numbers from this project

| result | value |
|---|---|
| Accuracy, base → GRPO | 27.5% → **52.5%** |
| Reproducibility (3 runs) | 42.5%, 43.3%, 52.5% |
| Degenerate groups, binary → graded | 3.3% → **0.0%** |
| Distinct reward values per group | 2.50 → **5.07** |
| Gradient-direction correlation | **0.9948** |
| Clip fraction, μ = 1..4 | `[0.0, 0.917, 0.917, 0.917]` |
| Cost of λ = 0.6 | **−7.5 pp** |
| Format vs correctness variance @60 steps | 0.2167 vs 0.1293 |

---

# THE COMPLETE WORKFLOW IN SIMPLE STEPS

1. **Load a pretrained model** — 4-bit + LoRA. It must already *sometimes* do the task.
2. **Write a prompt with a worked example** — it defines the contract your reward checks.
3. **Write a reward function** — verifiable, exception-proof, unit-tested offline.
4. **Validate the reward** — bounded, monotonic, scale-free, edge-safe, ≥3 distinct values.
5. **Validate the gradient** — positive advantage must raise log-probability (corr > 0.9).
6. **Measure a baseline** — greedy, held out, with pass@k.
7. **For each step:** sample a question → generate 6 attempts → grade → compute advantages → skip if tied → score under policy and reference → build the loss → update adapters.
8. **Log telemetry every step** — skip rate, KL, component variance, clip fraction, pass@G.
9. **Evaluate greedily** on held-out data.
10. **Read 20 completions.** Metrics summarize; text reveals.
11. **Compare against a control** — for any reward change, always.
12. **State your error bars.** n=60 means ±6 pp.

---

# KEY EQUATIONS WITH INTUITION

| equation | says |
|---|---|
| $\log\pi_\theta(o\|q)=\sum_t \log\pi_\theta(o_t\|q,o_{<t})$ | An answer's probability is the product of its tokens' — logs turn it into a sum |
| $\nabla J = \mathbb{E}[R\,\nabla\log\pi_\theta]$ | Sample answers, push each one's probability up in proportion to its reward |
| $\mathbb{E}[b\nabla\log\pi_\theta]=b\nabla 1=0$ | Subtracting any answer-independent baseline is free — unbiased, lower variance |
| $\hat A_i=\frac{r_i-\bar r}{\sigma}$ | "Was this better than my other attempts?" — the group mean *is* the baseline |
| $\hat A(\alpha r+\beta)=\hat A(r)$ | Scale and shift are invisible; only component *ratios* matter |
| $\rho=\pi_\theta/\pi_{old}$ | How stale is this rollout? (Exactly 1 when μ=1) |
| $\min(\rho\hat A,\operatorname{clip}(\rho)\hat A)$ | Don't let one batch move things too far |
| $e^\delta-\delta-1$ | KL, estimated in a way that can't go negative |
| $\frac{1}{\|o_i\|}\sum_t$ | Long answers shouldn't get more influence for being long |
| $\text{pass@}k=1-\binom{n-c}{k}/\binom{n}{k}$ | Does the capability exist anywhere in the distribution? |
| $2\lambda<1$ | Two partial credits must never beat one exact answer |

---

# INTERVIEW READINESS CHECKLIST

You should be able to explain each of these confidently, with an example from your own runs.

## Core algorithm 🔴
- [ ] What GRPO is, in one sentence
- [ ] Why it's called *Group Relative*
- [ ] Derive the policy gradient via the log-derivative trick
- [ ] Why subtracting a baseline is unbiased but reduces variance
- [ ] Why GRPO needs no critic, and what that costs
- [ ] What a degenerate group is and why it produces zero gradient
- [ ] Why the k3 KL estimator beats the naive one

## Implementation 🔴
- [ ] The `plen−1` masking rule and why off-by-one fails silently
- [ ] How `disable_adapter()` gives a free reference model
- [ ] Why clipping is inert at μ=1 — **and be able to prove it**
- [ ] Why the real memory hog is logits, not weights
- [ ] How you verified the gradient is correct

## Reward design 🔴
- [ ] Why absolute reward scale doesn't matter but ratios do
- [ ] Why a saturated component contributes exactly zero
- [ ] The λ < 0.5 bound and what happens when you break it
- [ ] Why an auxiliary reward must never outbid the terminal one
- [ ] How you'd detect reward hacking

## Diagnostics 🟡
- [ ] What skip rate tells you — and its three different meanings
- [ ] Why greedy and sampled evaluation differ
- [ ] What the pass@1 / pass@G gap tells you
- [ ] What error bars your evaluation actually supports

## Comparisons 🟡
- [ ] GRPO vs PPO — what's removed and what it costs
- [ ] GRPO vs SFT — path vs destination
- [ ] When you'd choose each

## Judgment 🔴
- [ ] A time your own experiment falsified your hypothesis
- [ ] Why you always run a control alongside a reward change
- [ ] Which of your results are proven vs suggestive, and why

---

## The three things to lead with in an interview

1. **"I implemented GRPO from scratch and verified the gradient is correct."** Most candidates implement; almost none validate. Correlation 0.9948 between advantage and change in log-probability is a concrete, checkable claim.

2. **"I broke my own reward on purpose to see what would happen."** Setting λ=0.6 cost 7.5 points and — crucially — *looked like a healthy run in isolation*. That's a story about engineering judgment, not just implementation.

3. **"Three of my hypotheses were falsified by my own experiments."** The format-inertness precondition, an ablation arm design, and a monitoring architecture. An implementation that has never been wrong in public has usually just never been checked.
