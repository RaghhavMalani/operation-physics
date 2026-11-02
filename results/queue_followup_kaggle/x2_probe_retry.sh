#!/usr/bin/env bash
# D161: the X2 probe refit's one retry, under X2's documented settings (4 jobs x 4 BLAS threads), after the GPU stage.
cd "$(dirname "$0")/../.."
L=results/queue_followup_kaggle/chain.log
until grep -q "end fs_hidden_random_7b" "$L"; do sleep 30; done
echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] D161 retry starts (the GPU stage ended)" >> "$L"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= nice -n 19 \
  uv run python -m physcal.eval.followup_scale x2_probe > results/queue_followup_kaggle/x2_probe_retry.log 2>&1
echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] D161 retry ended: exit $?" >> "$L"
