#!/usr/bin/env bash
# R4-5 (D113): build the frozen confirmatory pair sets, one worker, nice 19, one after another.
cd "$(dirname "$0")/../.." || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for s in a6 a6b a6c; do
  echo "$(date -Is) start $s"
  nice -n 19 uv run python -m physcal.env.pairs export "$s" --confirmatory > "results/confirmatory_build/$s.json" 2> "results/confirmatory_build/$s.log"
  echo "$(date -Is) $s exit $?"
done
