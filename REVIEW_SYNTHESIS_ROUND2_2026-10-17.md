# Review synthesis, Round 2 — Gemini and ChatGPT, 2026-10-17

Commit this file. Record items R2-1 to R2-9 in DECISIONS.md BEFORE any result they affect
exists. PREREGISTRATION.md is not edited; new experiments are pre-registered in DECISIONS.md
before their first result. This file does not change tonight's queue (D67 corrected evals).

## Where both reviewers agree

1. **Corrected evaluation comes first.** Unmerged adapters on the exact 4-bit training base,
   with every control on the same base (already D67, running tonight).
2. **Run the two extra uniform-credit seeds.** With one control seed, S1b/S2/S3 cannot speak
   to run-to-run variability of the control. One GPU night.
3. **Frame the paper as the world-versus-reasoning uncertainty decomposition.** The credit-
   assignment mechanism and the merge bug are supporting results, not the headline.
4. **Merged results are superseded,** kept in a methodology / reproducibility section with
   the mechanism and lost-update percentages. ChatGPT: do not call them a control
   (rounding is a structured perturbation, not controlled noise). Adopted.
5. **Distillation is informative but secondary.** It tests capacity and transfer, not
   spontaneous self-awareness.

## Decisions and amendments

**R2-1. Evaluation hygiene (ChatGPT).** Add a regression test: for a fixed set of prompts,
the reloaded model reproduces the training-time logits and the training-time σ distribution
within tolerance. Any eval loader change must pass it. Record base revision, bitsandbytes
version, compute dtype, quantization config and adapter hash in every eval tape.

**R2-2. SFT pipeline revalidation.** The SFT adapter lost 41% of its entries when merged into
bf16. The merged SFT model remains the start point of every existing RL run, so existing
comparisons stay internally consistent, but FINDINGS #4/#5 SFT numbers describe an attenuated
SFT. For all new SFT (Phase 12): never merge into bf16. Either merge in fp32 and verify outputs
before and after, or train RL on the unmerged SFT adapter.

**R2-3. What σ should and should not predict (ChatGPT's "most likely wrong").** Under the A1
decomposition, σ on noise-free problems should NOT be judged by correlation with total error;
that is the epistemic channel's job (p). Family N's result is reframed as: "a single σ asked to
carry epistemic uncertainty fails to rank errors". This motivates two channels; it is not a
claim that small models cannot express epistemic uncertainty (ChatGPT: that claim is
unsupported). The write-up uses this framing.

**R2-4. Seed-level statistics (ChatGPT).** With 3 vs 3 seeds the smallest exact two-sided
permutation p is 0.10. Report seed-wise effect sizes and matched-step trajectories alongside
the pre-registered problem-level tests, and do not describe pooled problem-level significance
as seed-level significance.

**R2-5. Publication criterion (ChatGPT), stated now:** the positive claim requires a
reproducible, baseline-beating advantage on at least one conditional, held-out uncertainty task
(not pooled coverage alone). If that fails, the controlled negative result is the paper.

**R2-6. Uniform-credit seeds 1 and 2:** scheduled for the next free night, identical protocol
to split credit (same SFT start, seeds, hyperparameters, step 287), evaluated with the corrected
loader. S1b/S2/S3 are then reported 3 vs 3 as a robustness analysis next to the pre-registered
1-control verdict.

**R2-7. Citations to verify before use:** Zeng, Xuan, Cui & Voigt, "Thinking Out Loud: Do
Reasoning Models Know When They're Right?" (EMNLP 2025, arXiv 2504.06564); Leng, Huang, Zhu &
Huang, "Taming Overconfidence in LLMs: Reward Calibration in RLHF" (ICLR 2025); Dettmers et al.,
QLoRA (arXiv 2305.14314); Gneiting, Balabdaoui & Raftery (2007), "Probabilistic forecasts,
calibration and sharpness", JRSS-B 69(2).

## New experiments (pre-register each before results)

**X1. Shallow text model on the propagation pairs (Gemini; CPU only, next).** Train a
gradient-boosted tree to predict ln σ\* from problem text alone (TF-IDF plus numeric features
parsed from the givens) on noisy generator problems (2.0.0, training seeds, noise 1–5%). Run it
on the frozen A6 and A6b pairs with the D59/D61/D64 analysis. Question: can surface features
propagate uncertainty? This sets the bar for the LLM before Phase 12.

**X2. Within-problem correctness pairs (ChatGPT; the decisive epistemic test).**
- Build about 500 held-out problems (training-split seeds not used elsewhere) where k = 8
  samples from the corrected SFT model produce at least one correct and one incorrect answer.
- For each problem, form (correct, incorrect) answer pairs. Problem difficulty is held fixed.
- Score each method on pairwise accuracy: how often the correct answer gets more confidence
  (lower σ, or higher p where available). Methods:
  - stated σ (SFT, and split-credit seed 0, corrected loader);
  - the A5 text baseline (problem + answer + reasoning);
  - a problem-text-only baseline (should score 0.5 by construction; sanity check);
  - an answer-conditioned hidden-state probe: logistic regression on the hidden state right
    after the completed numeric answer, fitted on separate training seeds;
  - random-init probe control.
- Reading: probe succeeds but stated σ fails → the information exists; readout or training
  is deficient. Only the text model succeeds → the signal is in the text, not accessible to the
  chosen hidden-state readout. Everything fails → earlier baseline advantages were mostly
  difficulty prediction, not answer-specific awareness.

**X3. Distillation capacity test (both; secondary).** SFT the same 4-bit base and LoRA (r = 16)
to output (a) out-of-fold A5 teacher σ, and (b) as a control, a σ fitted directly to true
error labels. Evaluate on corrected loaders: in-distribution, depth 5, X2 pairs. Reading: if the
student reaches a rank correlation with error of 0.5 or more, RL's 0.08 is an optimization
problem, not a capacity limit. It does not show RL-produced self-awareness.

## Phase 12 design (pre-register after X1 and X2 report)

Two channels, each evaluated separately before any combined forecast:

**Epistemic channel, p.**
- Event: "the calculation is correct given the displayed measurements",
  |ln μ − ln f(x_meas)| ≤ τ (A1). Never scored against the noise-free truth, so measurement
  noise cannot mislabel a sound calculation.
- p restricted to discrete values {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99}%.
- Trained by Brier score with the answer held fixed (E1 fixed-prefix machinery).

**Aleatoric channel, σ.**
- Trained only on computation-correct answers to noisy problems.
- Reward: expected crps_log against the generator's known predictive distribution for that
  problem (Gemini). Proper, verifiable, and far less noisy than scoring one realized outcome.
  Evaluation still uses realized-truth proper scores.
- Arms (one variable each):
  - **implicit:** σ stated directly after "±";
  - **explicit:** one propagation line in `<steps>` before the answer
    (e.g. `sigma_rel = sqrt((2·0.02)² + 0.01²) = 0.045`). Gemini's main warning: a 1.5B model
    cannot do multi-step propagation silently in one token; it will likely copy the largest
    input percentage and fail A6;
  - **supervised σ\* baseline** (ChatGPT): SFT directly on σ\* with a log-scale loss, no RL.
- Primary tests: A6/A6b joint verdict (D59/D61/D64), Mincer–Zarnowitz on computation-correct
  answers, conditional coverage across depth × noise cells, against the X1 text model.

**Combined forecast (secondary).** The distribution for wrong calculations needs its own
justification (a broad Gaussian centred on a wrong answer may be misspecified). Report the
combined CRPS only after each channel is evaluated on its own.

## Proposed schedule (laptop free only at night)

| night | GPU work | daytime CPU work |
|---|---|---|
| Sat 17 | D67 corrected evals (running) | — |
| Sun 18 | uniform-credit seeds 1–2 (R2-6) | R2 amendments; R2-1 test; X1; X2 problem selection and pre-registration |
| Mon 19 | X2 sampling + hidden states; X3 distillation pilot | X2 analysis; Phase 12 SFT data (implicit / explicit formats) |
| Tue 20+ | Phase 12 SFT (fp32 merge or unmerged), then channel pilots | Phase 12 pre-registration after X1/X2 |

Expected effect on the timeline: write-up moves to about the end of November.
