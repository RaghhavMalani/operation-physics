# Follow-up feasibility: three exploratory items (night of 2026-10-30/31; item 3 on Kaggle, 2026-11-01/02)

**Status: EXPLORATORY, for a follow-up project. Not part of the paper (D148's freeze applies to it).** Nothing here is
pre-registered as a hypothesis or confirmatory. The plan and every analysis rule were written in D151 (now in
`docs/exploratory/DECISIONS_FOLLOWUP.md`, D156) and pushed (`a8938c7`) before any of this track's data existed. Later changes each have their own entry, written
before the data they affect: D152 (item 3's decoding), D153 (item 3's local stop), D157 (item 3 on Kaggle) and D158
(the readouts' memory settings). D159 records the readouts' run and one missing comparison.

| item | code | config | results |
|---|---|---|---|
| 1. uncertainty-guided test-time compute | `physcal/eval/followup_ttc.py` | `configs/followup.yaml` `ttc` | `results/followup/ttc/{x2,mc_a}.json` |
| 2. exact arithmetic on the model's own lines | `physcal/eval/followup_arith.py` | `arith` | `results/followup/arith/results.json` |
| 3. scale check of the epistemic wall | `physcal/eval/followup_scale.py` | `scale` | `results/followup/scale/` |

Tests: `tests/test_followup.py`. CIs are 95% bootstrap intervals over problems (item 1, item 3) or over pairs (item 2,
and MC-A in item 1).

## Short version

1. **Adaptive sampling saves tokens mostly by not sampling. On this model, sampling more buys little.**
   - **The fixed-k curve is flat.** For the SFT 1.5B on X2's 1,200 candidates, majority@8 reaches 0.382 against 0.339
     for one sample (+4.3 points) at 8× the tokens.
   - **The headline policy** was chosen on half the problems and reported on the other half: D75, "stop after one
     sample if its stated σ is small, else draw until two agree".
     - It uses 26% [24%, 27%] of majority@8's tokens, at 3.6 [2.2, 4.9] points less accuracy.
     - At matched accuracy it saves 21% [12%, 30%] of the tokens of the fixed-k curve.
   - **The non-inferior policies** (accuracy within ±0.3 points of majority@8, in-sample) save 10–21% of the tokens:
     "stop at 3 agreeing" (B3) and "stop when the majority cannot be overturned" (C).
   - **MC-A's raw spread as a two-stage stopping rule fails.** For flagging run 0's errors it is worse than a fixed
     number of re-asks on every set: the matched-AUROC saving is −0.8 to −1.6 on A6c, CIs below 0.
2. **A calculator on the model's own lines does not help; it hurts slightly.**
   - **The anchors reproduce:** (a) −0.0397 and (c) −0.0417, as in FINDINGS #35.
   - **E,** every S formula re-evaluated at the model's own values with exact aggregation and copy, scores −0.0413:
     E − (a) = −0.0016 [−0.0034, −0.0003]. It is below RSS and X1.
   - **E_chain** also re-derives the intermediates exactly from the model's own formulas. It gets back to (a): −0.0005
     [−0.0018, +0.0006]. Descriptively it beats RSS and X1 at depths 1–2 and loses at depths 4–5.
   - **What this shows:** the model's arithmetic slips are not where the forecast loss is. Its wrong formulas and wrong
     chains are, and its sloppy aggregation partly offsets them (#35's pattern).
3. **The 7B tells its own right answers from its wrong ones far better than the SFT 1.5B, by every readout. This is
   not yet a scale effect.** It was sampled on Kaggle (D157) and read locally (D159).
   - **Within-problem pairwise accuracy:** the 7B on 844 kept problems vs the SFT 1.5B (X2) on the same 1,176-candidate
     prefix (504 kept):
     - self-consistency 0.827 vs 0.645 (+0.182 [+0.148, +0.215]);
     - direct P(yes) 0.683 vs 0.526 (+0.157 [+0.125, +0.189]);
     - hidden-state probe 0.753 vs 0.540 published (≈ +0.21 [+0.18, +0.24], approximate);
     - length 0.634 vs 0.534 (+0.100 [+0.067, +0.131]).
   - **By D151's rule the wall moves:** a single-sample readout beats the SFT 1.5B's with a CI excluding 0.
     Self-consistency still leads.
   - **But the comparison changes model size, format, training, decoding and the kept problems at once.** The native
     1.5B control, which would separate scale from the rest, did not run. Length alone also gains 0.10, so part of
     the gain is a surface signal, and the 7B probe has no random-init control.

## 1. Uncertainty-guided test-time compute (offline)

**Data.** X2's k = 8 samples: the corrected SFT 1.5B, T = 1, 1,200 candidates (`results/x2/sft_k8`). There is no new
generation.
- **Answers and correctness** are X2's.
- **Agreement:** within 1% relative.
- **The majority answer:** the drawn answer with the most agreeing drawn answers, ties to the earliest drawn.
- **Cost:** completion tokens. The prompt is shared and not counted.
- **Averaging:** every number is the mean over 200 random draw orders per problem. The fixed order 0..7 agrees: 0.338 /
  0.374 for majority@1 / @8.

**Policies (D151's grid, fixed in advance):**
- **A:** 2 samples. If they agree, stop; otherwise draw all 8.
- **B_c:** draw until some answer has c votes.
- **C:** stop when the leader cannot be overtaken by the samples left.
- **D_q:** after one sample, stop if its stated relative σ is ≤ the q-quantile on the selection half; otherwise
  continue as B_2.

**The headline, chosen on the selection half and reported on the evaluation half** (604 problems):
- D75 had the largest matched saving on the selection half (+0.235; next B3 +0.203).
- On the evaluation half:

| | accuracy | tokens / problem | samples | Δ accuracy vs maj@8 | tokens vs maj@8 | matched-accuracy saving |
|---|---|---|---|---|---|---|
| D75 | 0.353 [0.321, 0.385] | 402 [377, 428] | 1.92 | −0.036 [−0.049, −0.022] | 0.26 [0.24, 0.27] | **+0.21 [+0.12, +0.30]** |
| maj@8 | 0.388 [0.350, 0.426] | 1,563 | 8 | – | 1 | – |

**Every policy, all 1,200 problems (in-sample, descriptive):**

| policy | accuracy | tokens | samples | Δ acc vs maj@8 | tokens / maj@8 | matched saving |
|---|---|---|---|---|---|---|
| maj@1 | 0.339 [0.317, 0.361] | 195 | 1.00 | −0.043 [−0.054, −0.032] | 0.12 | – |
| maj@2 | 0.340 [0.318, 0.362] | 391 | 2.00 | −0.042 [−0.053, −0.031] | 0.25 | – |
| maj@3 | 0.356 [0.333, 0.380] | 587 | 3.00 | −0.026 [−0.035, −0.016] | 0.37 | – |
| maj@4 | 0.365 [0.342, 0.390] | 782 | 4.00 | −0.017 [−0.025, −0.009] | 0.50 | – |
| maj@5 | 0.373 [0.349, 0.397] | 978 | 5.00 | −0.009 [−0.016, −0.003] | 0.63 | – |
| maj@6 | 0.377 [0.352, 0.402] | 1,173 | 6.00 | −0.005 [−0.010, −0.000] | 0.75 | – |
| maj@7 | 0.380 [0.355, 0.406] | 1,369 | 7.00 | −0.002 [−0.005, +0.001] | 0.87 | – |
| maj@8 | 0.382 [0.356, 0.409] | 1,564 | 8.00 | – | 1.00 | – |
| A | 0.379 [0.353, 0.405] | 1,269 | 6.08 | −0.003 [−0.005, −0.001] | 0.81 | +0.02 [−0.23, +0.12] |
| B2 | 0.374 [0.350, 0.399] | 954 | 4.53 | −0.008 [−0.014, −0.001] | 0.61 | +0.10 [−0.06, +0.25] |
| B3 | 0.381 [0.355, 0.407] | 1,240 | 5.98 | −0.001 [−0.005, +0.003] | 0.79 | +0.14 [−0.03, +0.21] |
| B4 | 0.382 [0.356, 0.408] | 1,366 | 6.69 | −0.000 [−0.002, +0.002] | 0.87 | +0.11 [−0.12, +0.13] |
| C | 0.382 [0.356, 0.408] | 1,414 | 7.02 | −0.000 [−0.002, +0.001] | 0.90 | +0.08 [−0.16, +0.10] |
| D25 | 0.366 [0.343, 0.391] | 794 | 3.71 | −0.015 [−0.022, −0.008] | 0.51 | +0.02 [−0.12, +0.14] |
| D50 | 0.359 [0.336, 0.383] | 600 | 2.84 | −0.023 [−0.031, −0.015] | 0.38 | +0.07 [−0.06, +0.18] |
| D75 | 0.350 [0.328, 0.373] | 398 | 1.91 | −0.032 [−0.041, −0.022] | 0.25 | +0.22 [+0.16, +0.28] |

- **By depth (in-sample):** majority@8's gain over one sample is +3.5 / +3.0 / +4.8 / +5.7 points at depths 1–4. B3
  stays within 0.6 points of majority@8 at every depth, with 62% / 79% / 81% / 85% of its tokens. The saving is
  largest where the model is most often right, since agreement comes early there.
- **The stated-σ gate is the one "uncertainty-guided" policy that buys a large saving.** The SFT model's σ carries no
  within-problem signal (X2, FINDINGS #19). Between problems, a small first-sample σ still says "more samples will not
  change the answer" often enough to cut 74% of the tokens. The cost is 3–4 points of accuracy that voting would
  have bought.

**MC-A's raw spread as a stopping rule** (secondary, descriptive; B-v2 seed 0, run 0's answer kept; MC-A's own draw
order, which reproduces #37's AUROC curve 0.638 / 0.710 / 0.745 at k = 2 / 4 / 8):
- **The two-stage rule:** draw 2 re-asks; stop if their spread ≤ θ; otherwise use all 8.
- **It loses AUROC for "run 0 is wrong" on every set and every θ.** On A6c's evaluation half, θ = q0.75 (the best on
  the selection half) gives AUROC 0.663 [0.608, 0.716] at 939 re-ask tokens. Fixed k = 3 gives 0.700 at 751 tokens.
  - The matched-AUROC saving is −0.84 [−0.98, −0.24].
  - On A6 and A6b the same θ gives −0.92 and −0.76.
- **The reason:** a k = 2 spread and a k = 8 spread are on different scales. Mixing them across problems breaks the
  ranking that the AUROC measures. A spread-based stopping rule needs a calibrated, k-aware score (e.g. a sequential
  test), not a raw threshold.

**Caveats (item 1):**
- **The idea is prior art.** Early stopping of self-consistency on agreement, as in "adaptive-consistency" and
  "early-stopping self-consistency" (references to verify before any write-up), is known. Here it is only measured on
  this model and task.
- **The fixed-k curve is flat** (+4.3 points over 8×). Small accuracy differences then map to large token differences,
  which makes the matched saving noisy (its in-sample CIs are wide).
- **One model** (the SFT 1.5B), one temperature (T = 1), one task family, 1,200 problems.
- **Prompt tokens are not counted.** With a shared prefix they are paid once per problem by every policy.
- **The D gate's thresholds** are tiny relative σ values (0.0001 / 0.0007 / 0.003). The SFT model often states a near-0
  σ, so the gate is close to "stop if σ ≈ 0".

## 2. Exact arithmetic on the model's own lines (offline)

**Data.** D-pos-v2, seeds 0–2, on the frozen R4-5 set (3,060 problems per seed; 3,039 with predictive draws), read
only. **This is not FINDINGS #35:**
- #35 swapped in the builder's exact values (oracle swaps).
- Here nothing comes from the builder. Each S line's own formula is re-evaluated at the values the model itself wrote,
  with a restricted arithmetic AST. The σ line is recomputed from those, and σ = X′·|μ|.
- The structure (terms, formulas, levels) is unchanged.

**Coverage:**
- **Re-evaluated:** 3,016–3,023 problems per seed. 95% of the S formulas evaluate; the rest keep the σ line's number.
- **Excluded:** 4–31 problems where the S lines and σ terms differ in number, and 13–37 that do not decompose.
- **How much σ moves:** E moves σ by more than 25% on 10–14% of problems.

| variant | expected crps_log | − (a) | − RSS | − X1 | share of RSS→σ\* gap | A6c slope |
|---|---|---|---|---|---|---|
| (a) as stated (anchor −0.0397 ✓) | −0.0397 | – | +0.0002 [−0.0003, +0.0007] | −0.0005 [−0.0010, −0.0001] | +7% | +0.837 [+0.69, +0.98] |
| (c) #35 (anchor −0.0417 ✓) | −0.0417 | −0.0020 [−0.0047, −0.0003] | −0.0018 | −0.0025 | −60% | +0.826 |
| (c′) the model's σ-line terms, exact aggregation + copy | −0.0423 | −0.0025 [−0.0054, −0.0005] | −0.0023 | −0.0031 | −79% | +0.814 |
| **E: S re-evaluated, exact aggregation + copy** | **−0.0413** | **−0.0016 [−0.0034, −0.0003]** | −0.0014 [−0.0033, −0.0000] | −0.0021 [−0.0041, −0.0007] | −48% | +0.826 [+0.70, +0.95] |
| E_S: S re-evaluated, own aggregation + copy kept | −0.0403 | −0.0006 [−0.0014, −0.0000] | −0.0004 [−0.0013, +0.0003] | −0.0011 [−0.0020, −0.0004] | −14% | +0.849 [+0.71, +0.99] |
| E_chain: intermediates re-derived too | −0.0402 | −0.0005 [−0.0018, +0.0006] | −0.0003 [−0.0016, +0.0008] | −0.0010 [−0.0024, +0.0000] | −9% | +0.829 [+0.71, +0.95] |

- **E − (c′)** = +0.0009 [−0.0015, +0.0040]: fixing the S-line arithmetic itself does nothing measurable.
- **E_chain − (c′)** = +0.0021 [−0.0002, +0.0051].
- **By depth** (expected crps_log, depths 1–5):

| source | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|
| (a) | −0.0355 | −0.0372 | −0.0397 | −0.0422 | −0.0440 |
| E | −0.0350 | −0.0371 | −0.0410 | −0.0446 | −0.0489 |
| E_chain | **−0.0343** | **−0.0368** | −0.0397 | −0.0426 | −0.0476 |
| RSS | −0.0371 | −0.0381 | −0.0395 | −0.0426 | −0.0423 |
| X1 | −0.0362 | −0.0374 | −0.0388 | −0.0417 | −0.0419 |
| σ\* | −0.0340 | −0.0352 | −0.0367 | −0.0398 | −0.0391 |

**Reading (D151's rules):**
- **"Exact arithmetic helps":** no. E − (a)'s CI is below 0.
- **"Beats RSS / X1":** no, for both.
- **Why** (with #35's decomposition): the loss is in wrong value-dependent formulas, substituted wrong intermediates,
  chain selection and the copy's decimal shifts.
  - Exact evaluation fixes only the last of these, and makes the absurd root-sum-squares that the model's own sloppy X
    used to pull back count in full ((c′) is the worst row).
  - Re-deriving the intermediates (E_chain) removes part of the substitution error and recovers the loss, but no more.
  - **Where it would help:** at depths 1–2, where formulas are mostly right, E_chain is the best non-oracle source,
    descriptively. A calculator would pay off only together with correct derivatives, i.e. at shallow depth or with a
    symbolic differentiator (#35's "exact sensitivities" row: +51% of the gap).

**Caveats (item 2):**
- one model (D-pos-v2, three seeds), the frozen R4-5 set, one evaluator;
- "the model's values" are parsed from its text by fixed rules (tested), and 5% of the formulas do not evaluate;
- descriptive, with no multiple-comparison correction;
- E_chain's depth-1–2 advantage is a subgroup seen after the fact, labelled descriptive.

## 3. Scale check of the epistemic wall: the 7B, sampled on Kaggle and read locally

**The question (D151):** within a problem, X2's SFT 1.5B barely tells its right answers from its wrong ones by any
single-sample readout (pairwise accuracy 0.53–0.54); self-consistency does better (0.643). Does the 7B of the same
family do better?

**What ran:**
- **Sampling on Kaggle (D157):**
  - the official Qwen2.5-Math-7B-Instruct (revision `ef9926d7…`) in fp16;
  - vLLM 0.29.0 on 2 × T4, with D152's decoding (T 0.7, top_p 0.8, top_k 20), up to 3,072 new tokens, k = 8;
  - **all 1,176 candidates** below the development indices, in one 5.0 h session (15.4 s / problem), with no retry.
- **The ingest (D158)** passed every check: the manifest, the pool, the seeds, and every prompt re-rendered with the
  local tokenizer. The local re-grade changed 0 of 9,408 rows.
- **The readouts on the local 4060 (D159):**
  - direct P(yes) and the hidden states of all 29 layers, on the 6,133 answered answers of the kept problems, using
    the 4-bit copy of the same weights;
  - then D151's analysis, on the CPU.

**What was sampled:**
- answered 91.1%, truncated 8.7% (at 3,072 tokens), correct per sample 51.4%;
- by depth 1 → 4: correct 78 / 61 / 39 / 29%, truncated 4 / 7 / 10 / 14%;
- **kept (mixed) problems:** 844 of 1,176 (71.8%); by depth, 158 / 223 / 232 / 231 of 294. The SFT 1.5B keeps 504
  of the same 1,176 (42.9%);
- 8,476 within-problem (correct, incorrect) pairs.

**Results** (X2's within-problem pairwise accuracy, ties 0.5, mean over kept problems; 95% bootstrap over problems,
B = 10,000):

| readout | 7B, native format (844 kept) | SFT 1.5B X2, same prefix (504 kept) | SFT 1.5B X2, published (516 kept) | 7B − SFT 1.5B, same prefix |
|---|---|---|---|---|
| self-consistency (share of the other 7 within 1%) | **0.827** [0.809, 0.845] | 0.645 [0.617, 0.673] | 0.643 | +0.182 [+0.148, +0.215] |
| direct P(yes) (X4 direct) | **0.683** [0.663, 0.701] | 0.526 [0.500, 0.551] | 0.526 | +0.157 [+0.125, +0.189] |
| hidden-state probe, cross-fitted | **0.753** [0.734, 0.771] | – (not refit; D159) | 0.540 [0.514, 0.565] | ≈ +0.21 [+0.18, +0.24] vs published (approximate) |
| length only (shorter = more confident) | 0.634 [0.614, 0.654] | 0.534 [0.510, 0.558] | 0.535 | +0.100 [+0.067, +0.131] |

- **The differences are unpaired,** because the kept sets differ (D151).
- **The probe's difference is approximate:**
  - it is normal, from the two bootstrap CIs, against the full pool's published number;
  - D151's re-run of X2's cross-fit on the prefix was not done (D159).
- **By depth** (pairwise accuracy at depths 1–4):

| readout | d1 | d2 | d3 | d4 |
|---|---|---|---|---|
| self-consistency | 0.881 | 0.856 | 0.804 | 0.785 |
| direct P(yes) | 0.746 | 0.672 | 0.668 | 0.664 |
| hidden-state probe | 0.787 | 0.724 | 0.749 | 0.759 |
| length | 0.652 | 0.660 | 0.624 | 0.607 |
| *SFT 1.5B probe (X2, published)* | *0.565* | *0.520* | *0.541* | *0.538* |

- **The probe's choice:**
  - the 7B probe chose layer 19 of 29 (index 0 is the embeddings) with C = 0.001 in all five outer folds; inner-CV
    AUROC 0.773–0.781;
  - C = 0.001 is the edge of D151's grid, which was not extended;
  - X2's 1.5B probe chose layer 1 in four of five folds (layer 11 in one).

**Reading (D151's rules, descriptive):**
- **"The wall moves with scale": yes, by D151's rule.** A single-sample readout of the 7B (direct P(yes), and the
  probe) exceeds the SFT 1.5B's with a CI excluding 0. The rule's first comparison, against the native 1.5B, is
  missing (D157, difference 13), so this is the rule's fallback.
- **Self-consistency still leads:** 0.827, against 0.753 (probe) and 0.683 (P(yes)). The CIs do not overlap.
- **What the comparison cannot say is that size caused it.** Between the two columns, all of these change at once:
  - model size (7B vs 1.5B);
  - the format: the native `\boxed{}` chain-of-thought vs our `<steps>` / `<answer>` format with a stated σ;
  - training: no task training vs our SFT;
  - decoding and budget: T 0.7, top_p 0.8, top_k 20 and 3,072 tokens, vs T 1 with no top-p / top-k and 512 tokens (D152);
  - who reads: the 7B's answers were sampled in fp16 and read by a 4-bit copy, while X2's model sampled and read
    itself (D157, difference 3);
  - the problems: 72% vs 43% of the prefix are kept. The 7B's pairs include many depth-3–4 problems that the 1.5B gets
    all wrong, and its errors are of another kind.
- **Part of the gain is a surface signal.**
  - Length alone gains +0.100: the 7B's wrong answers are more often the long ones.
  - P(yes) and the probe both stay above length (0.683 and 0.753, against 0.634; the CIs do not overlap). No paired
    readout-vs-length test was specified.
- **The probe has no random-init control** (D151's stated limit).
  - In X2, the random-init probe was within noise of the real one.
  - Without the control, the 7B probe's 0.753 may partly read the text's surface (length, position, the digits)
    rather than what the model "knows".

**Every difference from the 1.5B setup:**

| | SFT 1.5B (X2) | 7B (this run) | entry |
|---|---|---|---|
| model | Qwen2.5-Math-1.5B-Instruct + our SFT (merged, 4-bit at load) | Qwen2.5-Math-7B-Instruct, the official weights, no task training | D151; D157 (2) |
| format | `<steps>` / `<answer>` with a stated σ | Qwen2.5-Math's chain-of-thought prompt and `\boxed{}`; the problem without its `<steps>` sentence | D151 |
| decoding | T 1, top_p 1, top_k 0; 512 new tokens | T 0.7, top_p 0.8, top_k 20; 3,072 new tokens | D152; D157 (4, 5) |
| sampler | HF + bitsandbytes, 4-bit, on the 4060; one torch seed per chunk | vLLM 0.29.0 in fp16, tensor-parallel 2 on 2 × T4; one seed per sample | D157 (1, 2, 4) |
| who reads | the same 4-bit model | the local 4-bit copy of the same weights (`unsloth/…-bnb-4bit`) | D157 (3) |
| answer and grade | X2's parser for our format | the last box (the D151 + D154 parser); a truncation is no answer; re-graded at ingest | D151; D154; D157 (11) |
| P(yes) turn | X4's `x4_v1` | the same, with "your \boxed{}" in place of "your <answer>"; context up to 4,096 tokens | D151; D157 (12) |
| probe position | the "±" token | the closing brace of the last box | D151 |
| probe control | random-init, within noise of the real probe | none | D151 |
| problems | X2's 1,200 candidates (516 kept; 504 on the 7B's prefix) | the first 1,176 (844 kept) | D157 |
| the 1.5B in the native format | – | not run (D151's control, "if time allows") | D157 (13) |

**Caveats (item 3):**
- one prompt, one model family, one synthetic domain;
- the readouts read a 4-bit copy of the sampler's weights;
- truncated samples (8.7%; 14% at depth 4) are no answer and in no pair, so the kept set misses the errors that run out
  of tokens;
- descriptive, with no multiple-comparison correction.

**Before Kaggle: the local feasibility measurement** (D152–D154; development problems only, never analyzed):
1. **X2's decoding (T = 1, no truncation) does not work for the 7B.** On the first development problem, 7 of 8
   samples ran to the 1,024-token cap, and one switched to Chinese mid-solution. D152 switched to Qwen2.5's documented
   sampling (T 0.7, top_p 0.8, top_k 20) before any non-development completion existed.
2. **At that decoding, D151's 120 s / problem rule fired:** 139 s and 155 s per problem at 1,024 tokens. The samples
   were legitimate derivations still unfinished at 1,024 tokens, so the local main stage did not run (D153).
3. **A development-only measurement at 2,048 tokens** (3 kinematics problems, depths 1–3) sized a local run:

| | 7B (4-bit) | native 1.5B (4-bit, untrained) |
|---|---|---|
| weights in memory / peak at batch 8 | 5.23 / 6.70 GiB (the 6.7 GiB cap) | 1.12 / 2.33 GiB |
| batch 8 × 2,048 tokens | **out of memory**: halves to 2 × 4 (depth 2), then 2 × 2 (depth 3) | fits |
| decoding rate | ~6.6 steps / s at batch 8 (bitsandbytes dequantizes on every step) | ~27 steps / s |
| s / problem, depths 1 / 2 / 3 | **277 / 474 / 852** | 23 / 67 / 46 |
| complete answers (tokens) | 347–808 / 1,117–1,559 / 1,050–1,862; 4 of 24 at the 2,048 cap | 310–630 / 1,015–1,859 / 480–1,236 |
| answered correct (re-graded, D154) | 13 of 20; all 3 problems have right and wrong answers | 3 of 24 |
| direct P(yes) / hidden states (29 layers), s per answer | 0.89 / 0.81 (both logit checks pass) | 0.14 / 0.12 |

- **What that meant:** about 50 problems per night on the 4060, roughly 20 kept, too few to measure the wall. Hence
  Kaggle (D157).
- **The planning estimate was not what ran.** It assumed 1,500–3,000 tokens / s on an A100-class GPU. Kaggle's
  2 × T4 sampled at 568 tokens / s on the 20 timed candidates (D158), and the whole job took 5.0 h.

## What this means for a follow-up

- **Test-time compute:** for this model, "spend more samples where the model is unsure" mostly means "do not sample
  when the first answer's stated σ is small".
  - That buys a 74% token cut for 3–4 points of accuracy. At matched accuracy it saves about a fifth of the tokens
    versus the fixed-k curve.
  - The no-loss policies save only 10–21%, because voting itself buys little (+4.3 points for 8×).
  - **A follow-up worth running:** a model whose voting gain is large, where an agreement stop has more to save.
  - **Not worth pursuing:** a perturbation-spread stop rule, which needs a k-aware score.
- **Tools for arithmetic:** a calculator on the model's own derivation does not fix the forecast. The bottleneck is
  the derivative (formula) and the chain, as #35 found.
  - **The promising variant:** exact symbolic differentiation of the model's own chain (#35's "exact sensitivities":
    +51% of the RSS-to-σ\* gap), together with re-derived intermediates (E_chain helps at depths 1–2).
- **Scale:** in its native format, the 7B tells its own right answers from its wrong ones well above chance with
  one sample (P(yes) 0.68, probe 0.75), where the SFT 1.5B sits at 0.53–0.54. Self-consistency still leads (0.83).
  - **It is not yet a scale result.** Size, format, training, decoding and the kept problems all change together.
  - **What would make it one:**
    - the native 1.5B on the same stack and settings (D151 (3)), e.g. a second Kaggle run, which needs the sampler
      generalized beyond the 7B;
    - a random-init control for the 7B probe;
    - a paired readout-vs-length test.

## Process notes and limits

- **Hand-launched runs and the guards (D154):**
  - **The fault:** the hand-launched feasibility runs put Python in its own process group (`timeout … setsid …`), so
    the temperature and stall guards watched the wrong group.
  - **Found** during the third 7B run and re-pointed. The GPU peaked at 84.3 °C, so no pause was ever due.
  - **The queue scripts are not affected.**
- **The paper's D-entry count (D155, then D156):**
  - **D155** made the paper's macro skip this track's entries.
  - **D156** replaced that: the track's entries moved to `docs/exploratory/DECISIONS_FOLLOWUP.md`, and the macro was
    reverted. The paper stays at 150, and `paper/` is unchanged.
- **Item 3's readouts (D159):**
  - The chain was restarted once to move its GPU deadline, at a cost of ~6 min.
  - 16 temperature pauses took 480 s of 2.5 h of GPU stages.
- **Everything here is exploratory:**
  - item 1 re-reads one model's samples;
  - item 2 re-reads one model's logs;
  - item 3 is one Kaggle sampling run of the 7B, with local readouts.
  - None of it is in the paper, and no hypothesis was tested.
