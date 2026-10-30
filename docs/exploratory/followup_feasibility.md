# Follow-up feasibility: three exploratory items (night of 2026-10-30/31)

**Status: EXPLORATORY, for a follow-up project. Not part of the paper (D148's freeze applies to it).** Nothing here is
pre-registered as a hypothesis or confirmatory. The plan and every analysis rule were written in DECISIONS.md D151 and
pushed (`a8938c7`) before any of this track's data existed. Two later changes each have their own entry, written
before the data they affect: D152 (item 3's decoding) and D153 (item 3's stop).

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
3. **The 7B scale check cannot be run on this GPU at a useful size (D153). Its main stage did not run.**
   - **The rate:** Qwen2.5-Math-7B-Instruct (4-bit) fits in memory, but decodes at ~6.6 steps/s.
   - **The length:** its native chain-of-thought needs more than 1,024 tokens from depth 2 up.
   - **The rule fired:** D151's 120 s / problem stop rule, at 139 s and 155 s.
   - **What ran instead:** a development-only measurement at 2,048 tokens (§3) sizes what a real run would cost.
     - At a budget that completes, a problem takes 277–852 s (depths 1–3), and batch 8 runs out of memory.
     - That is about 50 problems per night on the 4060, roughly 20 kept: too few to measure the wall.
     - A real run needs a cloud GPU or a faster inference stack. The code, the tested parser, the readouts and a queue
       script are ready.

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

## 3. Scale check of the epistemic wall (GPU): not run at a useful size; what it would cost

**The plan (D151):**
- **The model:** Qwen2.5-Math-7B-Instruct (Unsloth's 4-bit mirror, revision `2ab7fb37…`, content hash
  `9d8d80f688a04194`), in its native `\boxed{}` format.
- **Sampling:** k = 8 per problem on a prefix of X2's frozen candidates.
- **The comparison:** self-consistency, X4-style direct P(yes) and a cross-fitted hidden-state probe, scored by X2's
  within-problem pairwise accuracy, against the 1.5B.
- **The stop rule:** above 120 s / problem, stop and report.

**What happened (D152, D153; development problems only, never analyzed):**
1. **X2's decoding (T = 1, no truncation) does not work for the 7B.** On the first development problem, 7 of 8
   samples ran to the 1,024-token cap, and one switched to Chinese mid-solution. D152 switched both models to
   Qwen2.5's documented sampling (T 0.7, top_p 0.8, top_k 20) before any non-development completion existed.
2. **At that decoding the rule fired:** 139 s and 155 s per problem at 1,024 tokens.
   - **The samples are not loops.** A 2-sample check shows legitimate step-by-step derivations still in phase 3 of 4
     at 1,024 tokens: the native chain-of-thought needs more than 1,024 tokens from depth 2 up.
   - **The main stage was not run** (D153), as D151 says.
3. **A development-only measurement at 2,048 tokens** (3 kinematics problems, depths 1–3) sized a real run:

| | 7B (4-bit) | native 1.5B (4-bit, untrained) |
|---|---|---|
| weights in memory / peak at batch 8 | 5.23 / 6.70 GiB (the 6.7 GiB cap) | 1.12 / 2.33 GiB |
| batch 8 × 2,048 tokens | **out of memory**: halves to 2 × 4 (depth 2), then 2 × 2 (depth 3) | fits |
| decoding rate | ~6.6 steps / s at batch 8 (bitsandbytes dequantizes on every step) | ~27 steps / s |
| s / problem, depths 1 / 2 / 3 | **277 / 474 / 852** | 23 / 67 / 46 |
| complete answers (tokens) | 347–808 / 1,117–1,559 / 1,050–1,862; 4 of 24 at the 2,048 cap | 310–630 / 1,015–1,859 / 480–1,236 |
| answered correct (re-graded, D154) | 13 of 20; all 3 problems have right and wrong answers | 3 of 24 |
| direct P(yes) / hidden states (29 layers), s per answer | 0.89 / 0.81 (both logit checks pass) | 0.14 / 0.12 |

- **The time per problem** is set by the longest of the 8 samples: one runaway sample holds the whole batch. With the
  out-of-memory halving, the halves run one after another.
- **What a night buys on this GPU:**
  - at ~550 s / problem (more at depth 4) an 8-hour night gives about 50 problems;
  - with X2's 43% mixed share that is about 20 kept problems. The development problems hint at a higher share for
    the 7B, but they are 3;
  - a within-problem pairwise accuracy would then carry a CI of about ±0.1.
  - **That cannot separate a 7B readout from the 1.5B's** (X2's probe 0.540; X4 direct 0.526).
- **No within-problem result exists for the 7B.** The question "does the epistemic wall move with scale?" is open.

**What a real run needs (estimates for planning, not measurements):**
- **A serving stack with continuous batching and a paged KV cache,** e.g. vLLM with AWQ / GPTQ kernels rather than
  bitsandbytes, on a GPU with room for the KV cache. A runaway sample then no longer holds the others, and the 8
  samples of a problem share one prefill.
- **The scale of the job:** 1,200 problems × 8 samples × ~1,300 tokens ≈ 12.5 M generated tokens. At an assumed
  1,500–3,000 tokens / s for a 7B on one A100-class GPU, that is about 1.2–2.3 h, and the readouts are a few minutes.
  This throughput is an assumption, not measured here.
- **The control:** the native 1.5B on the same stack and settings separates scale from format and SFT. Locally it is
  cheap (≈ 45 s / problem), but on these development problems it is rarely right in its native format, so it may
  yield few kept problems.
- **The parser** (D151 + D154) is tested and handles both models' development answers: units inside the box, after
  it, or absent.
- **Alternatives kept for the follow-up (D153):**
  - a depth-1-only run, which fits the 1,024 budget locally (~140 s / problem);
  - a short-answer format, which would need SFT;
  - the 7B as an external judge of the SFT 1.5B's X2 answers, which is a different question.

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
- **Scale:** the 7B question needs a cloud GPU or a faster inference stack. On the 4060 it is not measurable at a
  useful size. Everything else is ready: the code, the parser, the readouts, and the queue (`scripts/queue_followup.sh`).

## Process notes and limits

- **Hand-launched runs and the guards (D154):**
  - **The fault:** the hand-launched feasibility runs put Python in its own process group (`timeout … setsid …`), so
    the temperature and stall guards watched the wrong group.
  - **Found** during the third 7B run and re-pointed. The GPU peaked at 84.3 °C, so no pause was ever due.
  - **The queue scripts are not affected.**
- **Everything here is one night's exploratory work:**
  - item 1 re-reads one model's samples;
  - item 2 re-reads one model's logs;
  - item 3 produced development-only measurements.
  - None of it is in the paper, and no hypothesis was tested.
