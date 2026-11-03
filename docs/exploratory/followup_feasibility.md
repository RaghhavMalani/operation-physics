# Follow-up feasibility: three exploratory items (night of 2026-10-30/31; item 3 on Kaggle, 2026-11-01/02)

**Status: EXPLORATORY, for a follow-up project. Not part of the paper (D148's freeze applies to it).** Nothing here is
pre-registered as a hypothesis or confirmatory. The plan and every analysis rule were written in D151 (now in
`docs/exploratory/DECISIONS_FOLLOWUP.md`, D156) and pushed (`a8938c7`) before any of this track's data existed. Later changes each have their own entry, written
before the data they affect: D152 (item 3's decoding), D153 (item 3's local stop), D157 (item 3 on Kaggle) and D158
(the readouts' memory settings). D159 records the 7B readouts' run. D160 (written before their data) adds the controls,
D161 one reproduction retry, and D162 records the controls' run.

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
3. **Scale moves the model's own verdict, not what a linear probe can read.**
   - **The setup:** the 7B and the off-the-shelf 1.5B were sampled the same way on Kaggle (D157, D160) and read
     locally (D159–D162). Only the model's size differs.
   - **7B − native 1.5B, within-problem pairwise accuracy** (unpaired, D151's primary):
     - direct P(yes) 0.683 vs 0.572: +0.110 [+0.082, +0.138]. **By D151's rule, the wall moves with scale.**
     - hidden-state probe 0.753 vs 0.758: −0.005 [−0.031, +0.020]. **No scale effect.**
     - self-consistency 0.827 vs 0.774: +0.053 [+0.025, +0.081]. It still leads at both sizes.
   - **D159's probe gap to the SFT 1.5B was not scale.** The native 1.5B's probe beats the SFT 1.5B's by +0.219
     [+0.188, +0.249], as much as the 7B's does (+0.214). Format, training and decoding account for it.
   - **Most of every probe's signal is surface.**
     - The same probe on random weights reads 0.664 (7B) and 0.673 (1.5B).
     - The real weights add +0.088 and +0.085, with CIs excluding 0, the same at both sizes.

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

## 3. Scale check of the epistemic wall: the 7B and the native 1.5B, sampled on Kaggle and read locally

**The question (D151):** within a problem, X2's SFT 1.5B barely tells its right answers from its wrong ones by any
single-sample readout (pairwise accuracy 0.53–0.54); self-consistency does better (0.643). Does a larger model of the
same family do better, and is that size, or the format?

**What ran:**
- **Sampling on Kaggle,** the same run for both models:
  - the official Qwen2.5-Math-7B-Instruct (D157) and Qwen2.5-Math-1.5B-Instruct (D160), fp16;
  - vLLM 0.29.0 on 2 × T4, with D152's decoding (T 0.7, top_p 0.8, top_k 20), up to 3,072 new tokens, k = 8;
  - the same seeds and the same 1,176 candidates;
  - the prompts are byte-identical (checked);
  - each model ran in one clean session: the 7B in 5.0 h, the 1.5B in 1.55 h.
- **The ingests (D158, D162)** passed every check; the local re-grade changed 0 rows.
- **The readouts on the local 4060** (D159, D162): direct P(yes); the hidden states of all 29 layers, on each model's
  4-bit copy; the same hidden states on a random-weights copy of each architecture (D160); then D151's analysis with
  D160's comparisons.
- **The SFT 1.5B's probe on the same 1,176 candidates** comes from X2's cross-fit, refit under X2's settings. It
  reproduces X2 exactly (D161, after a first attempt under other thread settings was refused).

**What was sampled:**

| | 7B | native 1.5B |
|---|---|---|
| answered | 91.1% | 97.6% |
| truncated (3,072 tokens) | 8.7% | 1.0% |
| correct per sample (depths 1–4) | 51.4% (78 / 61 / 39 / 29%) | 41.4% (73 / 49 / 28 / 16%) |
| kept (mixed) problems | 844 (71.8%) | 766 (65.1%) |
| kept by depth 1 / 2 / 3 / 4 | 158 / 223 / 232 / 231 | 167 / 224 / 215 / 160 |
| only correct / only wrong | 216 / 116 | 155 / 255 |
| answers read out; within-problem pairs | 6,133; 8,476 | 6,000; 8,849 |

- **Kept by both:** 599 problems. The SFT 1.5B keeps 504 of the same 1,176 (42.9%).

**Results** (X2's within-problem pairwise accuracy, ties 0.5, mean over kept problems; 95% bootstrap over problems,
B = 10,000):

| readout | 7B (844) | native 1.5B (766) | 7B − native 1.5B, unpaired | the same, paired (599 common) | SFT 1.5B X2, same prefix (504) |
|---|---|---|---|---|---|
| self-consistency | **0.827** [0.809, 0.845] | 0.774 [0.753, 0.796] | +0.053 [+0.025, +0.081] | +0.076 [+0.045, +0.106] | 0.645 [0.617, 0.673] |
| direct P(yes) (X4 direct) | 0.683 [0.663, 0.701] | 0.572 [0.553, 0.592] | **+0.110 [+0.082, +0.138]** | +0.120 [+0.090, +0.150] | 0.526 [0.500, 0.551] |
| hidden-state probe, cross-fitted | 0.753 [0.734, 0.771] | 0.758 [0.740, 0.775] | **−0.005 [−0.031, +0.020]** | −0.003 [−0.029, +0.024] | 0.539 [0.513, 0.564] |
| the same probe on random weights | 0.664 [0.646, 0.683] | 0.673 [0.653, 0.692] | −0.008 [−0.035, +0.019] | +0.001 [−0.029, +0.029] | – |
| length only (shorter = more confident) | 0.634 [0.614, 0.654] | 0.598 [0.578, 0.619] | +0.036 [+0.008, +0.065] | +0.040 [+0.011, +0.070] | 0.534 [0.510, 0.558] |

**The controls** (paired within a model, over the same kept problems):

| | 7B | native 1.5B |
|---|---|---|
| probe − random-weights probe | +0.088 [+0.068, +0.109] | +0.085 [+0.064, +0.105] |
| P(yes) − length | +0.048 [+0.027, +0.071] | −0.026 [−0.050, −0.002] |
| probe − length | +0.118 [+0.098, +0.139] | +0.159 [+0.138, +0.182] |

- **The probe's scale effect net of the surface:** (probe − random) for the 7B minus the 1.5B is +0.003
  [−0.026, +0.032] (unpaired).
- **Against the SFT 1.5B on the same prefix** (unpaired), 7B − SFT / native 1.5B − SFT:

| readout | 7B − SFT | native 1.5B − SFT |
|---|---|---|
| self-consistency | +0.182 [+0.148, +0.215] | +0.129 [+0.094, +0.164] |
| P(yes) | +0.157 [+0.125, +0.189] | +0.047 [+0.014, +0.079] |
| probe | +0.214 [+0.182, +0.245] | +0.219 [+0.188, +0.249] |
| length | +0.100 [+0.067, +0.131] | +0.064 [+0.033, +0.096] |

- **By depth** (pairwise accuracy, depths 1 / 2 / 3 / 4):

| readout | 7B | native 1.5B |
|---|---|---|
| self-consistency | 0.881 / 0.856 / 0.804 / 0.785 | 0.848 / 0.818 / 0.731 / 0.695 |
| direct P(yes) | 0.746 / 0.672 / 0.668 / 0.664 | 0.619 / 0.574 / 0.561 / 0.537 |
| hidden-state probe | 0.787 / 0.724 / 0.749 / 0.759 | 0.770 / 0.762 / 0.753 / 0.746 |
| random-weights probe | 0.697 / 0.677 / 0.638 / 0.656 | 0.709 / 0.676 / 0.649 / 0.661 |
| length | 0.652 / 0.660 / 0.624 / 0.607 | 0.644 / 0.609 / 0.594 / 0.542 |

- **The probes' choices** (layer of 29, index 0 = the embeddings; C):
  - the 7B: layer 19, C 0.001 in all five folds (the grid's edge, not extended);
  - the native 1.5B: layers 18–21, C 0.01;
  - the random-weights copies: early layers (the 7B's layer 1; the 1.5B's layers 2–3), C 0.001;
  - X2's SFT probe: layer 1 in four of five folds.

**Reading (D160's rules, descriptive):**
- **"The wall moves with scale": yes, through the model's own verdict.**
  - The 7B's direct P(yes) on its own answer beats the native 1.5B's by +0.110 [+0.082, +0.138] (paired, on the
    599 common problems: +0.120).
  - This comparison changes only the model's size: the same prompts, decoding, seeds, sampling stack and readout
    arrangement.
  - The 7B's verdict also beats length (+0.048), and the 1.5B's does not (−0.026).
- **Not through the probe.**
  - A linear probe reads as much from the 1.5B's hidden states as from the 7B's: −0.005 [−0.031, +0.020].
  - Net of the random-weights control, the difference is +0.003 [−0.026, +0.032].
  - D160's rule counts the probe only with both conditions; the second fails.
- **The random-weights control.**
  - Both real probes beat their random-weights copies (+0.088 and +0.085, CIs excluding 0), so each reads more than
    the text's surface.
  - But the surface carries most of the signal: random weights alone reach 0.66–0.67, above length.
  - The native answers' surface (their length, their wording, where the box sits) separates right from wrong far
    better than the SFT format's did.
- **D159's 7B − SFT probe gap (+0.214) was not scale.** The native 1.5B shows the same gap (+0.219). For P(yes) the
  picture differs: the native format adds +0.047 at 1.5B, and size adds +0.110 on top.
- **Self-consistency still leads at both sizes** (0.827 and 0.774) and gains with size (+0.053 unpaired, +0.076
  paired).
- **What still differs between the 7B and the native 1.5B:**
  - the kept problems: 844 vs 766, 599 shared; the 7B keeps more depth-4 problems;
  - truncation: 8.7% vs 1.0%. A truncated sample is no answer, so the 7B's runaway errors leave its pairs.
  - The paired comparisons on the 599 shared problems agree with the unpaired ones.

**Caveats (item 3):**
- one prompt, one model family, one synthetic domain;
- the readouts read 4-bit copies of the sampled fp16 weights;
- one random-weights draw per architecture;
- truncated samples are in no pair;
- descriptive, with no multiple-comparison correction.

**Every difference between the SFT 1.5B (X2) and the Kaggle runs.** The 7B and the native 1.5B differ from each other
only in the weights.

| | SFT 1.5B (X2) | 7B and native 1.5B (Kaggle) | entry |
|---|---|---|---|
| model | Qwen2.5-Math-1.5B-Instruct + our SFT (merged, 4-bit at load) | Qwen2.5-Math-7B-Instruct / -1.5B-Instruct, the official weights, no task training | D151; D157 (2); D160 |
| format | `<steps>` / `<answer>` with a stated σ | Qwen2.5-Math's chain-of-thought prompt and `\boxed{}`; the problem without its `<steps>` sentence | D151 |
| decoding | T 1, top_p 1, top_k 0; 512 new tokens | T 0.7, top_p 0.8, top_k 20; 3,072 new tokens | D152; D157 (4, 5) |
| sampler | HF + bitsandbytes, 4-bit, on the 4060; one torch seed per chunk | vLLM 0.29.0 in fp16, tensor-parallel 2 on 2 × T4; one seed per sample | D157 (1, 2, 4) |
| who reads | the same 4-bit model | the local 4-bit copy of the same weights (`unsloth/…-bnb-4bit`) | D157 (3) |
| answer and grade | X2's parser for our format | the last box (the D151 + D154 parser); a truncation is no answer; re-graded at ingest | D151; D154; D157 (11) |
| P(yes) turn | X4's `x4_v1` | the same, with "your \boxed{}" in place of "your <answer>"; context up to 4,096 tokens | D151; D157 (12) |
| probe position | the "±" token | the closing brace of the last box | D151 |
| probe control | random-init, within noise of the real probe (X2) | random weights, the same protocol: 0.664 / 0.673 | D160 |
| problems | X2's 1,200 candidates (516 kept; 504 on the prefix) | the first 1,176 (844 / 766 kept) | D157; D160 |

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
- **Scale:** size helps the model judge its own answer, not what a linear probe reads.
  - **The 7B's own P(yes)** separates its right answers from its wrong ones better than the 1.5B's (+0.110), with
    everything else equal.
  - **A probe on the hidden states** reads the same at both sizes (0.75–0.76), and most of it is surface: random
    weights reach 0.66–0.67.
  - **The SFT format's wall** (0.53–0.54 for every single-sample readout) comes mostly from the format, training and
    decoding, not from the model's size.
  - **A follow-up worth running:** what in the native answers makes the surface so informative (length, hedging,
    re-checks, where the box sits). Also whether an SFT format that keeps those signs keeps the probe's 0.76.

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
- **The controls (D160–D162):**
  - **X2's probe refit** was refused once by its exact gate: D160 had misread X2's thread settings. The one retry
    under X2's documented settings reproduced it exactly (D161).
  - **The two readout chains ran one after the other,** never at once, with 38 temperature pauses in all.
  - **The 7B's random-weights copy** (15 GB, gitignored) was deleted on 2026-11-03 (D163); it is regenerable from its seed.
- **Everything here is exploratory:**
  - item 1 re-reads one model's samples;
  - item 2 re-reads one model's logs;
  - item 3 is two Kaggle sampling runs (the 7B and the native 1.5B), with local readouts and controls.
  - None of it is in the paper, and no hypothesis was tested.
