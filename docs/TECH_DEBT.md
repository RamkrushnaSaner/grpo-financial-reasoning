# Known gaps and technical debt

What to address before trusting new experiments, in priority order. Each item
says what is wrong, the evidence, and where in the code a fix belongs. Items
marked **fixed** were resolved in v2.0.0 and are listed so the history is clear.

## A. Before the next GPU experiment

**A1. The GPU path has not been run since the refactor.** The update is
identical to the original on CPU, within float32 noise across CPU types
(`test_update_matches_the_original_implementation`),
and the real Hugging Face + peft path runs end to end on a tiny locally built
Qwen2 (`tests/test_hf_integration.py`). Not exercised: 4-bit loading
(bitsandbytes needs a GPU) and the real Qwen tokenizer and weights.
*Action:* `make smoke` on a Colab T4, then `make reproduce` and check that the
greedy accuracy and format numbers match Run 6 (16.7% → 43.3%, 61.7% → 20.0%).

**A2. The evaluation is too small to compare runs.** With 60 held-out questions
a paired test detects a 10-point difference between two runs only 31–58% of the
time (simulated, depending on how many questions flip the other way); 80% power
needs about 15–20 points. With 200 questions, 10 points is detected 88–100% of
the time. *Action:* set `task.size: 600` and `evaluation.n: 200` (questions
0–499 are unchanged by a larger size). Run at least three seeds
(`--set experiment.seed=...`) before claiming a difference between methods.

**A3. The reward has a known loophole.** `correct_graded` parses the last
number when the `Answer:` tag is missing, so correctness can be earned without
the output contract; in Run 6 format compliance fell 61.7% → 20.0%.
`configs/gated.yaml` (`correct_gated`) closes it but has never been trained.
*Action:* run `gated.yaml` next to `base.yaml` with the same seeds.

**A4. The tolerance credits wrong arithmetic.** Dollar answers are accepted
within ±0.10 and percentages within ±0.5 points, absolute. Probe A4 was scored
correct for 4.8 × 1.06 = 5.184 (true 5.088). *Action:* decide on a tighter or
relative tolerance (`task.tolerance`) before the next comparison; it changes
both the reward and the evaluation metric, so compare runs only at equal
tolerance (the fingerprint already enforces this for cached stages).

## B. Method limitations (deliberately unchanged; extension points exist)

**B1. One question per update.** Each optimizer step uses 6 answers to one
question, so the gradient is noisy (DeepSeekMath trains on large batches of
questions). A batched trainer would accumulate `grpo_step` losses over several
questions before `optimizer.step()`; the change belongs in
`grpo/training.py` (the step already separates rollout, loss and update).

**B2. Dividing by the group std amplifies tiny differences.** In a group where
every answer is wrong, the format bonus or tiny partial credits become
full-size advantages. Alternative estimators (e.g. mean-centring only, as in
Dr. GRPO) can be registered in `grpo/algorithm/advantages.py` and selected with
`algorithm.advantage`.

**B3. Per-answer length normalisation has a known length bias** (long wrong
answers are penalised less per token). Token-level or constant normalisation
can be registered in `grpo/algorithm/losses.py` and selected with
`algorithm.loss`. No length blow-up was observed (mean length fell 57 → 47).

**B4. Every token gets the same advantage.** Shared arithmetic errors (probe A3)
never discriminate within a group, so GRPO has no signal to fix them.

**B5. LoRA dropout 0.05 in the base config** breaks ρ = 1 at μ = 1 (clip
fraction 0.0038). Kept because it is the Run 6 method; `gated.yaml` uses 0.
Decide the default once A3 is resolved.

**B6. No abstention.** Every training question is answerable, so "cannot
determine" always gets the lowest reward in its group (probe D1 fabricated a
number). Matters only if the task should include unanswerable questions.

## C. Measurement and engineering

**C1. No held-out monitoring during training**, so no early stopping or
checkpoint selection; the adapter evaluated is always the last one.

**C2. Training telemetry is confounded by question difficulty**: one question
per step means window-to-window changes in training accuracy mix learning with
which questions were drawn. Only the held-out comparison is paired.

**C3. Memory and speed.** Gradient checkpointing is off (it conflicts with
adapter swapping) and sequences are scored one at a time; generation is most of
the wall-clock. Fine for 0.5B on a T4; a larger model needs a different
reference-policy mechanism.

**C4. The adapter is a raw tensor file** (`adapter.pt`), loadable by
`Policy.load_adapter` only. Using it elsewhere needs a conversion to a peft
adapter (`save_pretrained`).

**C5. The probe set has 15 questions**: it shows *what* changed, not how much.

**C6. `docs/LEARNING_NOTES.md` predates Run 6** and quotes some superseded
numbers; its provenance note lists them.

## Fixed in v2.0.0

| issue | fix |
|---|---|
| Growth labels inconsistent with their question (63/104 train, 11/60 held-out) | dataset `version: 2`; `version: 1` kept for reproduction |
| Significance computed with the SE of one proportion; no per-question results | per-question records, exact McNemar, paired bootstrap |
| Training cache keyed by run name, not settings | training fingerprint in `train.json` |
| pass@k sampling depended on earlier RNG use | evaluation seeds with `experiment.seed` |
| Stop detection used only the tokenizer EOS id | `Policy.stop_token_ids()` |
| Loss, gradient norm and mean reward not logged | per-step `metrics.jsonl` |
| Domain logic inside the GRPO step | `Task` interface; the step is task-agnostic |
| Reward arms hard-coded | reward components composed in config |
