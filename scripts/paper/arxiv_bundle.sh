#!/usr/bin/env bash
# Build paper/arxiv_submission.tar.gz (ARXIV_METADATA.md): the sources, the generated files, the figure and the
# .bbl, no build artifacts. Builds in a temporary directory with pdflatex + BibTeX (arXiv's compiler), then checks
# that a fresh extraction compiles with pdflatex alone, as arXiv does. Needs pdflatex and bibtex on PATH.
set -euo pipefail
P="$(cd "$(dirname "$0")/../../paper" && pwd)"
W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
cd "$P"
FILES=(main.tex sections/*.tex generated/macros.tex generated/tables/*.tex generated/figures/headline.pdf)
mkdir -p "$W/build" "$W/check"
cp --parents "${FILES[@]}" refs.bib "$W/build/"
cd "$W/build"
latex_run() { pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null; }
latex_run; bibtex main >/dev/null; latex_run; latex_run
tar --owner=0 --group=0 --sort=name --mtime='2026-11-18 00:00Z' -czf "$P/arxiv_submission.tar.gz" "${FILES[@]}" main.bbl
cd "$W/check" && tar xzf "$P/arxiv_submission.tar.gz"
latex_run; latex_run; latex_run
if grep -qE '^! |undefined|Overfull' main.log; then grep -E '^! |undefined|Overfull' main.log; exit 1; fi
grep -o 'Output written.*' main.log
tar tzf "$P/arxiv_submission.tar.gz"
