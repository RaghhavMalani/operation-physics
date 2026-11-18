# arXiv submission metadata (paper #1)

Source bundle: `paper/arxiv_submission.tar.gz`, built by `scripts/paper/arxiv_bundle.sh` from the committed sources
(main.tex, sections/, generated/macros.tex, generated/tables/, generated/figures/headline.pdf, main.bbl). It compiles
with pdflatex alone (no BibTeX run, as on arXiv): 36 pages, no errors, no undefined references or citations, no
overfull boxes (TeX Live 2026, checked 2026-11-18). The numbers below are the generated values in
`generated/macros.tex` at the release pin; if the paper is rebuilt, re-check the abstract against them.

## Title

When Language Models Quote Uncertainty: Separating Reasoning Errors from Measurement Noise

## Authors

Raghhav Malani (PES University, Bengaluru)

## Abstract

Plain text, one paragraph, the paper's abstract with the generated values filled in (1,653 characters; arXiv's
limit is 1,920). It uses the Unicode characters σ and ⋆, which arXiv accepts in metadata; for an ASCII-only form,
write "sigma" and "sigma*".

```text
A numerical answer with a stated uncertainty is useful only if the uncertainty is honest problem by problem. We study this with a generator of multi-step physics problems that computes, for each problem, the benchmark's reference uncertainty σ⋆ under the specified measurement model (a Monte Carlo estimate), sets measurement noise independently of chain depth, and builds counterfactual pairs that separate propagation from surface reading. Training Qwen2.5-Math-1.5B-Instruct with a reward built on a proper scoring rule, we find three things. First, with sequence-level credit the stated σ collapses; giving its tokens their own advantage raises 1σ coverage from 10.8% to 41.8%, but σ still does not rank errors. Second, on the fine-tuned model and its terse format, the tested single-completion signals (stated σ, a hidden-state probe, self-verification) provide little within-problem discrimination between correct and incorrect answers; agreement among samples discriminates better, at 8.0 times the answer-generation tokens. Third, trained on visible sensitivity derivations, the model states a σ that propagates measurement noise on fresh problems in a pre-registered test (slope +0.82, ideal one), yet its expected score is not shown to beat a root-sum-square heuristic: it closes 7% of the gap to σ⋆, and the loss lies in derivatives of value-dependent relations. Learning how uncertainty propagates is not the same as forecasting it well. Results are for one small model on a synthetic benchmark; an exploratory replication on a second model family returned NO-GO: its sensitivity criterion passed, but the pilot did not establish invariance.
```

## Categories

- **Primary:** cs.LG (Machine Learning)
- **Suggested cross-lists:** cs.CL (the subject is language models and their verbalized uncertainty) and stat.ML
  (proper scoring rules, calibration and coverage tests). cs.AI is optional; arXiv moderators may trim cross-lists,
  so two is the safer number.

## Comments

```text
36 pages, 2 figures, 13 tables. Code, frozen evaluation sets and logs: https://github.com/RaghhavMalani/operation-physics
```

The repository must be public before the comments line is submitted.

## License

**Recommended: CC BY 4.0** (Creative Commons Attribution 4.0). Anyone may reuse and adapt the paper with
attribution, which suits an open benchmark paper, and it meets most funders' open-access requirements. The choice
cannot be made more restrictive after announcement, so decide before submitting. The alternative, arXiv's default
non-exclusive distribution license, lets arXiv distribute the paper and grants no reuse rights to anyone else.

The license covers the paper only. The repository has no LICENSE file, so the code is unlicensed (all rights
reserved) until one is added.

## Before submitting

- `main.tex` sets `\date{Draft}`, so the title page reads "Draft". Change it (e.g. to `\date{}` or a date) in the
  sources and follow the release-pin procedure (`paper/README.md`) before rebuilding the bundle, or submit with it as
  it is.
- The .tex comments go out with the source (arXiv publishes it). They cite FINDINGS and D-entries, which are already
  public in the repository; one in `sections/appendix.tex` and one in `sections/acknowledgments.tex` say "your
  instruction". Nothing in them is private, but strip them first if you prefer.
