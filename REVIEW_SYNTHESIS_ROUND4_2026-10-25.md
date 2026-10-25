# Review synthesis, Round 4 — Gemini and ChatGPT, 2026-10-25

Commit this file. Record R4-1 to R4-9 in DECISIONS.md before any result they affect.

## Where both reviewers agree

1. **Top priority is D-pos-v2 against a matched implicit control on the same corrected
   data** (mixed per-input noise levels, value-dependent sensitivities derived as visible
   steps). Act 3 is currently negative partly because training never contained the
   behaviour being tested; that must be fixed before any capability claim.
2. **The thesis is too strong.** Do not claim "uncertainty is not felt inside one forward
   pass". E1d showed a capability appearing after a learning-rate change, which argues
   against intrinsic-capability conclusions. Use the narrower framing (R4-8).
3. **Report sensitivities split by type** (power law vs value-dependent) and test transfer
   to held-out compositions and numerical regimes, so "memorized a longer template" can be
   ruled out.
4. **Single-model and synthetic-only are serious but fixable:** scope claims to this model
   and benchmark; add an inference-only cross-family replication later.
5. **Write-up items:** provenance ledger of every pre-registered hypothesis and its outcome;
   the bf16 merge pitfall as its own box; k-cost curves; problem-clustered CIs.

## Where they differ, and our call

- **MC-A:** Gemini calls it a primary contribution (the aleatoric dual of self-consistency);
  ChatGPT calls it a baseline, since model arithmetic errors contaminate the spread.
  **Call:** run it as a diagnostic with oracle and robust variants and a stopping rule
  (R4-4); promote it only if results justify that.
- **MC-B (model writes a simulation program):** Gemini says drop (scope creep); ChatGPT says
  prototype cheaply with few-shot prompting, since it is the clean "computing vs predicting"
  arm. **Call:** defer; prototype after tonight with its own pre-registration. Note that
  PAL and Program-of-Thoughts limit its novelty.

## Decisions

**R4-1. D-pos-v2 pilot (1 seed) and matched control B-v2 (1 seed), tonight.**
Same new data for both: noise levels 0/1/2/5% assigned independently per given; every
value-dependent sensitivity derived as its own visible step (e.g. s_a = a/(a−b) = … = 1.50);
power-law sensitivities stated as exponents. Same SFT recipe, budget and pinned base as
D-pos. A6c is in-distribution for these arms; say so.

**R4-2. Grade the line, not only the answer (Gemini's "most likely wrong").** For D-pos-v2,
score three σ values: (a) the stated answer σ; (b) the σ the line states; (c) a
verifier-computed σ from the model's own written sensitivities and levels, combined exactly.
If (c) propagates and (a) does not, the failure is arithmetic, not propagation.

**R4-3. Success criteria and stop/go, fixed now.**
- Propagation: A6c slope with a 90% CI lower bound above 0; a point estimate ≥ 0.5 is the
  practical threshold for "meaningful". A6b slope equivalent to 0 (D59 margin).
- Report everything split by sensitivity type and by depth.
- Go to 3 seeds of both arms only if D-pos-v2 meets the propagation criterion on (a), (b)
  or (c), and record which.

**R4-4. MC-A diagnostic (inference only).** On A6c first (all 1,020 problems), then A6/A6b
if time allows:
- perturb each displayed given by its stated noise, k = 8 fixed seeds; greedy decoding;
- variants: raw (SD of ln answers); robust (drop runs whose sequence of intermediate step
  names differs from the unperturbed run, then MAD × 1.4826); oracle (exact solver on the
  same perturbations, the attainable upper bound);
- report the drop rate, and the k = 2/4/8 curve from the same logs;
- stopping rule: if robust MC-A does not beat the RSS heuristic on A6c, do not increase k.

**R4-5. Frozen confirmatory set.** Generate and freeze a new set of sensitivity,
invariance and allocation pairs from fresh seeds, untouched until any 3-seed confirmatory
run. The current A6/A6b/A6c become development sets.

**R4-6. Three estimands, reported separately:** aleatoric (σ vs σ\* on valid computations
and oracle-centred), epistemic (correctness discrimination), end-to-end (proper score of
the delivered answer, unconditional).

**R4-7. Later, each with its own pre-registration:** cross-family inference replication
(X2, X4, MC-A); more E1d seeds plus one unfrozen joint run at lr 5e-5; MC-B prototype;
compositional transfer (train depths 1–3, test 4–5, held-out optics); X4 with native
chain-of-thought.

**R4-8. Framing for the paper (ChatGPT's wording, adapted):** "Predicting uncertainty is
not the same as computing it. In a controlled small-model physics benchmark, confidence
about whether a particular answer is correct is revealed by cross-sample agreement rather
than by single-response estimators, while propagation of measurement uncertainty is hard to
learn implicitly, motivating a controlled comparison with explicit and executable
computation." Headline figure: "Where does uncertainty information come from?", with
panels for correctness discrimination vs sampling cost, sensitivity-pair response and
invariance.

**R4-9. References to verify before citing:** Wang et al., Self-Consistency (arXiv
2203.11171); Gao et al., PAL (arXiv 2211.10435); Chen et al., Program of Thoughts (arXiv
2211.12588); Kamath et al., Selective Question Answering under Domain Shift (ACL 2020);
JCGM 101:2008 (Monte Carlo propagation of distributions); Wang and Stengel-Eskin (ICLR 2026)
and Jang et al. (ICML 2026), which are unverified.
