# Review synthesis, Round 3 — Gemini and ChatGPT, 2026-10-20

Commit this file. Record R3-1 to R3-10 in DECISIONS.md before any result they affect.
Tonight's X4 changes (R3-1, R3-2) are already in tonight's launch prompt.

## Where both reviewers agree

1. **X2/X3 reading holds, with limits.** Tested estimators capture between-problem
   difficulty, not within-problem answer correctness. Do not claim a model-size limit
   (ChatGPT), and do not claim RL's failure is purely optimization without testing it.
2. **X4 must separate passive inspection from active recomputation.** (Both: direct mode.)
3. **Supervised σ\* first, as the propagation ceiling,** before spending RL nights.
4. **Explicit arm: masking every propagation number is high risk.** Both say SFT should
   contain real propagation numbers (a positive control), with format-only as an ablation.
5. **p (confidence) is not part of the primary Phase 12 result.** Defer RL on p.
6. **Expected CRPS against the known distribution is sound** (lower variance), with the
   Gaussian CRPS-optimal scale as the target (already A2), and Monte Carlo σ\* not called exact.

## Decisions

**R3-1. X4 additions (tonight).** Direct mode and the perturbed control (Gemini).
Primary uncertainty for X4 is a cluster bootstrap over problems (516), not over pairs
(ChatGPT). Report extra inference tokens per mode. Later, not tonight: yes/no order
variant (affirmative bias), and a deterministic arithmetic checker as a ceiling.

**R3-2. X2 robustness (CPU, post-hoc, labelled exploratory).**
- Re-report X2's CIs with a problem-level cluster bootstrap.
- Clearly separated pairs (right ≤ 0.25%, wrong ≥ 10%).
- Completion length, right vs wrong, and a length-only pairwise baseline.
- **Self-consistency baseline (ChatGPT's "independent re-solve"):** confidence in an answer =
  fraction of the problem's other 7 samples that agree with it within 1%. Uses existing X2
  samples, no GPU. Costs 8 samples at inference; report that.

**R3-3. Optimization pilots before Phase 12 RL (Gemini + our addition).** E1 had clean credit
yet learned a constant σ (ρ ≈ 0.01) while supervised training on the same labels reached
0.58. Two pilots on the E1 machinery, ~2.7 h each:
- **E1c:** σ advantages normalized across the whole batch instead of within each prompt.
- **E1d:** higher learning rate (RL's is ~40× below SFT's).
Pre-register both; whichever moves ρ sets the RL configuration for Phase 12.

**R3-4. Phase 12 primary scope: aleatoric σ propagation only.** Stages:
- **A (CPU):** X1 plus noise heuristics (max, mean, root-sum-square of stated noise),
  scoring audit, oracle-centred diagnostic.
- **B:** supervised σ\* model, 3 seeds (the ceiling).
- **C:** implicit σ-RL, 1 pilot with the R3-3 configuration; 3 seeds only if it shows signal.
- **D:** explicit propagation, 1 diagnostic seed each: SFT with real propagation numbers
  (positive control) and format-only SFT (ablation). Full seeds only if it resolves B vs C.
- **E:** p (confidence) deferred to a follow-up. If X4 works, verification-derived p becomes
  its teacher or baseline.

**R3-5. New counterfactual set A6c: noise-allocation pairs (ChatGPT's "most likely wrong").**
Keep the same multiset of noise percentages but swap which inputs receive them (e.g.
y = a·b², 1% on a and 5% on b, then reversed). Visible percentages are identical; correct σ\*
differs. This defeats "copy the largest percentage". Primary propagation test alongside
A6/A6b. Compare against X1 and the R3-4 heuristics.

**R3-6. Tolerances and selection bias.**
- Confidence event: |ŷ − y_display| / |y_display| ≤ 1%, y_display from the displayed givens.
  Sensitivity analyses at 0.1% and 0.5%.
- σ propagation, primary: **oracle-centred** diagnostic on all problems (the stated σ scored
  with μ replaced by f(x_meas)), which avoids selecting on correctness (ChatGPT).
- σ propagation, secondary: correctly computed answers with
  τ_σ = max(0.25%, 0.25·σ\*) (Gemini).
- Noise-0 problems handled separately (log σ\* undefined).

**R3-7. Statistics.** Bootstrap over problems or templates, never over overlapping pairs.
Describe the credit-assignment result as 3 vs 3 seeds, not 9 independent replications.

**R3-8. Claim strength.** Part II is stated "for the tested estimators and this model", with
no claim about model size. Defences: X4, candidate verification, and optionally a later
inference-only scale check on a 7B model.

**R3-9. Explicit arm mechanics.** SFT teaches the copy step (answer σ = line result × |μ|);
split credit routes the σ reward to the line's number tokens as well as the answer's σ.
Specify any p loss coefficient separately if p is ever trained (a zero weight in the total
reward does not mean zero effect through separately normalized token advantages).

**R3-10. References to verify before citing:** Huang et al., "Large Language Models Cannot
Self-Correct Reasoning Yet" (ICLR 2024, arXiv 2310.01798); Zhang et al., "Generative
Verifiers" (ICLR 2025, arXiv 2408.15240); Tian et al., "Just Ask for Calibration" (EMNLP 2023,
arXiv 2305.14975); Weng et al., "Large Language Models are Better Reasoners with
Self-Verification" (Findings of EMNLP 2023, arXiv 2212.09561); Xia et al., "Calibration Is Not
Enough" (arXiv 2601.08064); Ma et al., "From token probabilities to calibrated confidence"
(arXiv 2608.07827); Zeng et al. (arXiv 2504.06564) and Leng et al. (arXiv 2410.09724), which a
reviewer called "now verified" — check them yourself.

## Proposed schedule

| night | GPU | daytime CPU |
| --- | --- | --- |
| Tue 20 | X4 main + direct + perturbed (to 11:30) | — |
| Wed 21 | E1c + E1d pilots; X4 remaining controls if time | R3 amendments; A6c build; heuristics; Phase 12 stage A |
| Thu 22 | Phase 12 SFT data check; supervised σ\* 3 seeds | Phase 12 pre-registration (after E1c/E1d) |
| Fri 23+ | implicit σ-RL pilot; explicit diagnostic seeds | analysis |
