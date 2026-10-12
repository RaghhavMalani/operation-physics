#!/usr/bin/env bash
# Between GPU stages: if the GPU is hotter than the limit, pause in 10-minute steps until it cools (guardrail).
LIMIT=${1:-87}
while true; do
  t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -1)
  if [ "${t:-0}" -le "$LIMIT" ]; then echo "GPU ${t}C <= ${LIMIT}C, continuing"; break; fi
  echo "$(date -Is) GPU ${t}C > ${LIMIT}C: pausing 600s"; sleep 600
done
