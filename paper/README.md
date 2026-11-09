# The paper

*When Language Models Quote Uncertainty: Separating Reasoning Errors from Measurement Noise* (working title; R5-1).

## Build

```bash
uv run python scripts/paper/build.py
```

```bash
cd paper && tectonic main.tex
```

`latexmk -pdf main.tex` works as well (pdflatex + BibTeX); the generated files are ASCII so arXiv's pdflatex accepts
them. For an arXiv upload, keep the `.bbl` (`tectonic --keep-intermediates main.tex`) and include `generated/`.

## Where the numbers come from

- **`scripts/paper/build.py`** writes everything under `generated/` from committed files only. It refuses a source
  that git does not track or that has uncommitted changes.
  - `generated/macros.tex`: every value the prose uses, as `\pv{key}`. An undefined key stops the LaTeX build.
  - `generated/tables/*.tex`: each starts with the source files and their git blobs.
  - `generated/figures/`: the headline figure (`scripts/figures/headline.py`) and its sidecar JSON.
  - `generated/sources.json`, `generated/SOURCES.md`: every artifact's sources.
- **No number is typed into the drafted sections** (`tests/test_paper.py` checks it). Every claim cites a FINDINGS
  number or a D-entry in a LaTeX comment.

## Status (2026-11-09)

- **Drafted:** Setup (§3), Methods (§4), Evaluation protocol (§5), the results (§6 optimization, §7 within-problem
  correctness signals, §8 propagation of measurement noise, §9 generalization; D178), Acknowledgments, and one
  exploratory paragraph in Limitations (D166).
- **Stubs:** the abstract, Introduction, Related work (references to be verified first, R5-10), Discussion, the rest
  of Limitations, appendices.
