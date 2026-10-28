# Results

Every number here comes from an executed run. Runs used **different evaluation settings**, so numbers are compared only within a run unless stated otherwise. Runs 4–6 used the original generator (dataset version 1, with the growth-label bug described under Run 6); runs 2–3 used an earlier version of it. Runs 1–5 quote a rough standard error, SE = 100·√(0.25/n) pp for one proportion at its own n; that is a precision guide, not a test of a difference. From v2.0.0 the pipeline saves per-question results and compares before/after with the paired McNemar test.

**Reproducing Run 6.** `grpo run configs/run6.yaml` uses the same settings. Greedy numbers should match on the same GPU type and library versions; sampled numbers (training, pass@k) are comparable, not bit-identical, because GPU sampling is not deterministic and evaluation now seeds pass@k sampling at its start.

## Run index

| run | what | steps | eval n | max tokens | notes |
|---|---|---|---|---|---|
| 1 | toy 600K char-level model | 300 | — | — | the negative control |
| 2 | Qwen, plain prompt | 100 | 40 | 160 | stalled |
| 3 | Qwen + one-shot prompt | 100 | 40 | 160 | first successful run |
| 4 | reward ablation, 3 arms | 60/arm | 60 | 112 | arm-vs-arm valid |
| 5 | adversarial runs | 60/run | 40 | 112 | partial-credit telemetry contaminated (see LESSONS) |
| **6** | **master pipeline** | **60** | **60** | **160** | **`configs/run6.yaml`; the reported run** |

---

## Run 6 — the reported run

Produced by `notebooks/archive/grpo_master.ipynb`; reproduced by `configs/run6.yaml` (dataset version 1). The raw per-question outputs of this run were not saved, which limits the statistics below.

**Before vs after** (greedy, held-out questions 400–459, same questions for both models):

| metric | n | BEFORE | AFTER | change | paired test (range over possible pairings) |
|---|---|---|---|---|---|
| Overall accuracy | 60 | 16.7% (10) | 43.3% (26) | +26.7 pp | McNemar p from 0.011 to 0.00003: **significant under any pairing** |
| 1-step accuracy | 23 | 21.7% (5) | 56.5% (13) | +34.8 pp | p from 0.096 to 0.008: probably real, not established alone |
| 2-step accuracy | 37 | 13.5% (5) | 35.1% (13) | +21.6 pp | p from 0.096 to 0.008: probably real, not established alone |
| Format compliance | 60 | 61.7% (37) | 20.0% (12) | **−41.7 pp** | p from 0.0005 to 6e-8: **significant under any pairing** |
| pass@1 (6 samples) | 60 | 15.8% | 24.4% | +8.6 pp | weak (per-question values not saved) |
| pass@6 | 60 | 48.3% | 58.3% | +10.0 pp | weak |
| Mean length | 60 | 57 tok | 47 tok | −10 tok | |

**Correction (v2.0.0).** The original report gave "4.1 SE", "3.3 SE", "2.6 SE" and "6.5 SE", dividing each change by the standard error of a *single* proportion. A before/after change on the same questions needs a paired test, which depends on how many questions went right→wrong. That count is unknown because per-question results were not saved, so the table gives the range over every pairing consistent with the totals. The pipeline now saves per-question records and reports the exact McNemar test.

**Data-quality caveat (found in v2.0.0).** In dataset version 1, the growth template's label is computed from the exact growth rate while the question shows the new revenue rounded to one decimal. 11 of these 60 held-out questions (all growth, all 2-step) have a label the shown numbers do not give (e.g. "$1.2B compared to $1.0B": 20%, labelled 17%). A model that calculates perfectly from the question could score at most 49/60 (81.7%) overall and 26/37 on 2-step. Both models faced the same labels, so the *change* is a fair comparison; the absolute accuracies are understated and the 1-step/2-step split is distorted. Version 2 fixes the labels.

**Training** (15-step windows): exact correctness (sampled) 0.256 → 0.111 → 0.278 → 0.333; pass@6 0.733 → 0.867; format 0.333 → 0.444; KL 0.0019 → 0.0276; degenerate groups **0.0%** of 60 steps; clip fraction 0.0038; 15.8 min; adapter magnitude 355.58. Each training step uses a different question, so window-to-window changes mix learning with question difficulty.

**Validation**: gradient-direction correlation 0.9948; advantages mean +2.98e-08 / std 0.9998; k3 KL minimum 0.0; clip fraction over inner epochs `[0.0, 0.917, 0.917, 0.917]` (tests 1–6 run on a tiny CPU model: they verify the loss math, not the training run); reference vs trained policy mean |Δlog p| = 0.148, sample KL = 0.073 (test 7, on the trained model: PASS).

### Interpretation

**1. Accuracy improved significantly** on the four training templates, at one seed.

**2. Format compliance collapsed, more significantly still.** The trained model writes `Final answer: 8.7`. `extract_answer` falls back to "the last number in the text", so that completion earns the full 1.0 correctness reward and forfeits only the 0.1 format reward. Within a group, `Final answer: 8.7` (reward 1.0) beats `Answer: 24.3` (reward 0.1), so the policy moves toward whatever style co-occurs with correct answers. The 10:1 weighting stops format from *outbidding* correctness; it does nothing to stop correctness from *starving* format once the parser decouples them. The step-0 rollout already said `Final answer:`, so this style was present in the sampling distribution from the start.

**3. Training format rose while evaluation format fell** (0.333 → 0.444 sampled, 61.7% → 20.0% greedy). The tails of the distribution still produce `Answer:`; the mode moved away from it. Training telemetry could not have caught this. Only the greedy evaluation did.

**4. Mostly sharpening, probably.** Greedy accuracy rose 26.7 pp while sampled pass@1 rose 8.6 pp, and the trained model's greedy accuracy (43.3%) is below the base model's pass@6 (48.3%). That pattern fits RL concentrating probability on answers the base model could already sample. It is consistent with that reading, not proof of it; the pass@1↔pass@6 gap moved 32.5% → 33.9%, inside the noise.

**5. The mode is far better than the samples.** Greedy 43.3% vs sampled pass@1 24.4%. Reported accuracy depends heavily on decoding; always quote both.

**6. The ρ ≡ 1 identity was violated.** Clip fraction 0.0038 where the mathematics predicts exactly 0 at μ=1. Cause: `logp_old` is scored in eval mode and `logp_new` in train mode, so LoRA dropout (0.05) makes them differ. Reproduced on CPU (`test_lora_dropout_breaks_the_ratio_identity`); `configs/gated.yaml` sets dropout to 0.

### Question-level A/B (15 questions)

| group | expectation | correct | format |
|---|---|---|---|
| A in-distribution | improve | 0/4 → 2/4 | 50% → 25% |
| B near-distribution | unknown | 2/3 → 3/3 | 67% → 67% |
| C out-of-distribution | no change | not scored | 0% → 0% |
| D adversarial | varies | 2/5 → 3/5 | 83% → 17% |
| **all** | | **4/12 → 8/12** | **60% → 27%** |

**Changes**
- **A1, sum of three values.** Base: `(2.7 + 2.4 + 3.6) * 3 = 24.3`; trained: `2.7 + 2.4 + 3.6 = 8.7`. Stopped multiplying the sum by 3.
- **A4, projection.** Base: `4.8 × 1.06^10 = 8.3134` (ten years of growth for a one-year question). Trained: `4.8 × 1.06 = 5.184`. **Correction:** the formula is right but the arithmetic is wrong (4.8 × 1.06 = 5.088). It was scored correct because the label is 5.09 and the dollar tolerance is ±0.10. The improvement is formula selection, not a correct answer.
- **B2, reverse direction.** Base looped in LaTeX and hit the 160-token cap; trained solved `3.0 / 0.25 = 12` in 108 tokens.
- **D3, trivial lookup.** Base invented `$3.3 + $1.8 = $5.1`; trained answered 3.3.

**Failures worth naming**
- **A3.** Both models computed `1.4 / 4.5 = 33%`; the answer is 31.1%. A shared arithmetic error that never discriminated within a group, so GRPO had no signal to fix it.
- **D5.** Trained model computed `−0.8 / 4.0 = −0.2` correctly and forgot to multiply by 100. A new failure mode: right division, wrong unit.
- **D1, unanswerable.** Trained model wrote `$2.7 + $2.4 + $2.1 = $7.2`, inventing Q3. Neither model abstains; the trained model's fabrication is more plausible-looking. The reward gives "cannot determine" the lowest score in any group, so nothing in training encourages abstaining.
- **C1, syllogism.** The correct answer is "no" (Priya is not stated to be an analyst). Both models answered "yes".

---

## Earlier runs

### Run 1 — toy character-level model (negative control)
Format compliance 94% → 97.5%; **correctness 0% across 300 steps**. No correct completion ever appeared in a group, so there was nothing to reinforce. *RL elicits capability; pretraining creates it.*

### Run 2 — Qwen, plain prompt
Format 0%; 75–100% of steps degenerate; baseline 7.5%. A 0.5B model ignored the format instruction and wrote bare numbers, so the format reward never fired.

### Run 3 — Qwen + one-shot prompt
| | before | after |
|---|---|---|
| Overall | 27.5% | **52.5%** |
| 1-step | 31.3% | 56.3% |
| 2-step | 25.0% | 50.0% |
| Format | 50.0% | 100.0% |

n=40 (SE ±7.9 pp): the +25 pp gain is 3.2 SE. Adding one worked example lifted the baseline from 7.5% and cut degenerate groups from ~80% to ~10%.

### Run 4 — reward ablation (60 steps per arm)
| arm | held-out | degenerate | distinct rewards / group | format variance |
|---|---|---|---|---|
| binary | 35.0% | 3.3% | 2.50 | 0.1375 |
| **graded** | **43.3%** | **0.0%** | **5.07** | 0.2167 |
| graded, no format | 33.3% | 0.0% | 4.50 | — |

- **Solid:** graded rewards eliminate degenerate groups (averaged over 60 steps).
- **Not established:** graded vs binary on accuracy (+8.3 pp ≈ 1.3 SE, one seed).
- **Refuted:** that the format component was inert. Its within-group variance (0.2167) exceeded correctness (0.1293), and removing it was the only arm whose exact correctness did not improve.

### Run 5 — adversarial runs (n=40, SE ±7.9 pp)
| run | break | held-out | final KL |
|---|---|---|---|
| control | — | 42.5% | 0.0290 |
| λ = 0.6 | violates 2λ < 1 | 35.0% | 0.0268 |
| β = 0 | KL leash removed | 40.0% | 0.0622 (2.14×) |

- **λ = 0.6: −7.5 pp is 0.95 SE — not significant.** Earlier write-ups called it "1.2 SE" using the wrong n. The learning-gain comparison (+0.145 vs +0.077) depends on the estimator: 0.53 of control with 15-step windows, 0.86 with 30-step windows. Treat as suggestive.
- **β = 0** drove KL to 2.14× control within 60 steps, with no measurable accuracy cost yet.
- The raised format-weight run pushed mean training format compliance to about 0.83 vs about 0.49 for control (read from the logged level chart); its held-out accuracy was not captured in the logged output and is not reported.

---

## Consistency across runs

The graded configuration reached 52.5% (run 3), 43.3% (run 4), 42.5% (run 5) and 43.3% (run 6). These runs differ in steps, eval size and token limit, so this shows consistency, not a controlled replication.

Runs 4 and 6 measured an **identical** baseline — 16.7% accuracy, 61.7% format, 57 mean tokens — at 112 and 160 tokens respectively. The base model's greedy completions average 57 tokens, so the 112 cap barely binds. The difference from run 3's 27.5% baseline comes from the eval set (400–439 vs 400–459), not truncation.
