#!/usr/bin/env bash
# GPU temperature guard for one running queue stage (the overnight rule, D91: GPU > 87 C -> pause 10 min, also inside
# a stage). While the stage's process group lives, read the GPU temperature every POLL_S seconds; above MAX_C, stop the
# whole group (SIGSTOP: timeout, uv and python), wait PAUSE_S, then continue it (SIGCONT). PID-based, no pgrep -f.
#
# usage:  scripts/gpu_temp_guard.sh PGID LOGFILE &   # PGID: the stage's `timeout` PID (timeout leads its own group)
set -u
pg=$1 log=$2
MAX_C=${MAX_C:-87}
PAUSE_S=${PAUSE_S:-600}
POLL_S=${POLL_S:-30}
TEMP_CMD=${TEMP_CMD:-nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits}
glog() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] guard($pg): $*" >> "$log"; }
temp() { t=$($TEMP_CMD 2>/dev/null | head -1 | tr -dc '0-9.'); echo "${t:-0}"; }
hot() { awk -v t="$1" -v m="$MAX_C" 'BEGIN {exit !(t + 0 > m + 0)}'; }  # the driver reports 55.70, not 55 (D129)

sleep "$POLL_S"  # the stage's timeout makes its process group first
while kill -0 "$pg" 2>/dev/null; do
  t=$(temp)
  if hot "$t"; then
    if kill -STOP -- "-$pg" 2>/dev/null; then
      glog "GPU at ${t}C > ${MAX_C}C: stage stopped for ${PAUSE_S}s"
      sleep "$PAUSE_S"
      kill -CONT -- "-$pg" 2>/dev/null
      glog "stage continued (GPU at $(temp)C)"
    else
      glog "GPU at ${t}C > ${MAX_C}C, but process group $pg could not be stopped"
    fi
  fi
  sleep "$POLL_S"
done
