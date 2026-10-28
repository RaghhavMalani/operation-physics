# Review synthesis, Round 5 — Gemini and ChatGPT, 2026-10-28

Commit this file. Record R5-1 to R5-10 in DECISIONS.md before any result they affect.

## Central message (both reviewers)

Learning how uncertainty should propagate (structural correctness) is not the same as
forecasting it well (forecast quality). D-pos-v2's confirmed propagation (A6c +0.816) and its
unsupported score secondaries (K6 vs RSS, K7 vs X1) are reported with equal prominence; this
tension is the paper's central contribution. Never describe K6 as a "tie": failure to reject
is not equivalence. Report score differences with intervals, normalized against the
RSS-to-oracle gap where possible.

## Decisions

**R5-1. Paper framing.** Working title: "When Language Models Quote Uncertainty: Separating
Reasoning Errors from Measurement Noise" ("Quoting Volatility" stays the project name).
Claims are scoped to this model, benchmark and training procedures: no "small models cannot",
no "explicit propagation gives better calibration", no "all bf16 merging is unsafe". Headline
figure: three panels (optimization; within-problem correctness discrimination; propagation vs
forecast quality), using real frozen-evaluation data only.

**R5-2. Tail-error autopsy (CPU, descriptive, no new pre-registration).** On the frozen
confirmatory D-pos-v2 logs, classify the worst 5% of problems by cause: wrong sensitivity
(power-law vs value-dependent), arithmetic in the line, wrong level read, aggregation, copy,
chain selection, format; by depth. Report a 95%-trimmed expected score alongside the full one.

**R5-3. Hybrid-oracle ablation (CPU, diagnostic, labelled as such).** From the logged lines,
recompute σ with each component replaced by its exact value independently: exact
sensitivities + model's levels and aggregation; model's sensitivities + exact aggregation
(this is σ source (c)); exact levels; fully exact (ceiling). Shows where the loss comes from.
Never presented as a deployable method.

**R5-4. RSS-fallback guardrail (post-hoc, guarded against rescue bias).** Rule: fall back to
RSS when the written line fails to parse or σ/RSS lies outside fixed bounds. Choose the
bounds on the development sets only, freeze them in a D-entry, then apply once to the
confirmatory logs. Labelled post-hoc exploratory.

**R5-5. Cross-model replication (the main remaining experiment; new pre-registration).**
D-pos-v2 vs B-v2 SFT on a different model family, same data, protocol and frozen
confirmatory set. Candidates: Gemma-2-2B-it, Llama-3.2-3B-Instruct, Llama-3.2-1B-Instruct.
First a feasibility check per candidate: fits the 8 GB GPU in 4-bit with LoRA at the needed
sequence length, step time, and baseline format-following and arithmetic accuracy. Pick
one, pre-register, run 1 seed per arm as a pilot, then 3 seeds if the pilot passes the same
criterion as R4-3. Report format-following and base accuracy separately so a weaker model
does not hide the mechanism.

**R5-6. Fresh counterfactual generalization (optional, if nights remain).** Unseen
coefficient changes inside value-dependent relations (new Jacobians, same surface form),
plus unseen formula compositions. New frozen set; own pre-registration.

**R5-7. Cut from this paper:** nested Monte Carlo; MC-B SFT; joint answer-generation RL at
lr 5e-5 (state E1d as exploratory and scope Act 1's claim to the frozen-answer setting);
retraining D-pos-v2 on depth 4–5 (it would destroy the held-out depth claim); Kelly and
finance metrics; new hyperparameter searches; the optics family unless R5-6 is done.
Nested MC goes to future work, with ChatGPT's sequence (semantic validity first, then
execution tests, then input sampling).

**R5-8. Reporting rules.**
- Statistical unit: report uncertainty over both seeds and problems (clustered or
  hierarchical); never treat 5,390 pairs or 3 seeds on shared problems as independent.
- Sample agreement: report discrimination against inference tokens and GPU-seconds.
- MC-B: split failures into interface (no program, signature, rejected) vs semantic
  (runs but computes the wrong chain).
- Depth 5 is extrapolation in chain length, not to new physical laws; say so.
- A chronological decision timeline separating confirmatory from exploratory analyses.
- A reproducibility appendix: reload, precision and adapter checks; superseded merged
  results kept and labelled.

**R5-9. Venue.** Public arXiv preprint around 20 November 2026, then a 2027 workshop on LM
evaluation, uncertainty or AI for science. A main-track submission only after cross-model
replication and genuine family or formula transfer.

**R5-10. References.** Fix Gneiting & Raftery (2007): JASA, not JRSS-B. Verify before
citing: Damani et al. (arXiv 2507.16806); Park et al. (arXiv 2605.20740); Guo et al. (arXiv
2604.05306); Grünefeld et al. (arXiv 2605.07776); Wang et al. (arXiv 2609.32470); Kapoor et
al. (arXiv 2406.08391); PAL (arXiv 2211.10435); Program of Thoughts (arXiv 2211.12588).
