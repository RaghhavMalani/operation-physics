#!/usr/bin/env bash
# Stall watchdog for one running GPU stage (D132). A Windows GPU driver reset (nvlddmkm 153; D131) can leave the stage's
# CUDA process alive and silent until its timeout, hours later. While the stage's process group lives, check every
# POLL_S seconds when its log was last written. Silence counts from the later of that write and the last poll at which
# the group was stopped by the temperature guard (a process in state T): a pause is not a stall. After STALL_S seconds
# of silence, log BLOCKED and kill the group (SIGTERM, then SIGKILL after GRACE_S); the chain sees a non-zero exit and
# goes on to the next stage. PID-based, no pgrep -f.
#
# usage:  scripts/stall_guard.sh PGID STAGE_LOG CHAIN_LOG &   # PGID: the stage's `timeout` PID (timeout leads its group)
set -u
pg=$1 slog=$2 log=$3
STALL_S=${STALL_S:-900}
POLL_S=${STALL_POLL_S:-30}
GRACE_S=${STALL_GRACE_S:-30}
glog() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] stall($pg): $*" >> "$log"; }
stopped() { ps -eo pgid=,stat= | awk -v g="$pg" '$1 == g && $2 ~ /^T/ {f = 1} END {exit !f}'; }
written() { stat -c %Y "$slog" 2>/dev/null || echo 0; }

last=$(date +%s)  # the stage's start counts as activity: its log may not have been written yet
while kill -0 "$pg" 2>/dev/null; do
  now=$(date +%s)
  if stopped; then
    last=$now
  else
    w=$(written); [ "$w" -gt "$last" ] && last=$w
    if [ $((now - last)) -gt "$STALL_S" ]; then
      glog "BLOCKED: $(basename "$slog") silent for $((now - last))s while not temperature-paused: killing process group $pg"
      kill -TERM -- "-$pg" 2>/dev/null
      for ((i = 0; i < GRACE_S; i++)); do kill -0 "$pg" 2>/dev/null || exit 0; sleep 1; done
      kill -KILL -- "-$pg" 2>/dev/null && glog "process group $pg outlived SIGTERM by ${GRACE_S}s: SIGKILL"
      exit 0
    fi
  fi
  sleep "$POLL_S"
done
