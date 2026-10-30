#!/usr/bin/env bash
# GPU temperature guard for one running queue stage (the overnight rule, D91: GPU > 87 C -> pause, also inside a stage).
# While the stage's process group lives, read the GPU temperature every POLL_S seconds; above MAX_C, stop the whole
# group (SIGSTOP: timeout, uv and python). D143 (hysteresis): during the pause, read it every CHECK_S seconds and continue
# the group (SIGCONT) at the first reading at or below RESUME_C, or after PAUSE_S seconds at most (until D143 every pause
# lasted PAUSE_S). An unreadable temperature never ends a pause early. Every pause is logged with its actual length.
# PID-based, no pgrep -f.
#
# usage:  scripts/gpu_temp_guard.sh PGID LOGFILE &   # PGID: the stage's `timeout` PID (timeout leads its own group)
set -u
pg=$1 log=$2
MAX_C=${MAX_C:-87}
RESUME_C=${RESUME_C:-75}   # D143
PAUSE_S=${PAUSE_S:-600}    # the longest pause (integer seconds)
CHECK_S=${CHECK_S:-30}     # D143: the reading interval during a pause (integer seconds)
POLL_S=${POLL_S:-30}
TEMP_CMD=${TEMP_CMD:-nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits}
glog() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] guard($pg): $*" >> "$log"; }
reading() { $TEMP_CMD 2>/dev/null | head -1 | tr -dc '0-9.'; }   # empty if unreadable
temp() { t=$(reading); echo "${t:-0}"; }
hot() { awk -v t="$1" -v m="$MAX_C" 'BEGIN {exit !(t + 0 > m + 0)}'; }  # the driver reports 55.70, not 55 (D129)
cooled() { [ -n "$1" ] && awk -v t="$1" -v r="$RESUME_C" 'BEGIN {exit !(t + 0 <= r + 0)}'; }

sleep "$POLL_S"  # the stage's timeout makes its process group first
while kill -0 "$pg" 2>/dev/null; do
  t=$(temp)
  if hot "$t"; then
    if kill -STOP -- "-$pg" 2>/dev/null; then
      t0=$(date +%s)
      glog "GPU at ${t}C > ${MAX_C}C: stage stopped (continues at <= ${RESUME_C}C, after ${PAUSE_S}s at most)"
      why="the ${PAUSE_S}s maximum"
      while left=$(( PAUSE_S - ($(date +%s) - t0) )); [ "$left" -gt 0 ]; do
        sleep $(( CHECK_S < left ? CHECK_S : left ))
        kill -0 "$pg" 2>/dev/null || { why="the stage ended"; break; }
        r=$(reading)
        if cooled "$r"; then why="GPU at ${r}C <= ${RESUME_C}C"; break; fi
      done
      kill -CONT -- "-$pg" 2>/dev/null
      case $why in GPU*) ;; *) why="$why; GPU at $(temp)C" ;; esac  # a cooled pause already names its reading (D145)
      glog "stage continued after $(( $(date +%s) - t0 ))s paused ($why)"
    else
      glog "GPU at ${t}C > ${MAX_C}C, but process group $pg could not be stopped"
    fi
  fi
  sleep "$POLL_S"
done
