#!/usr/bin/env bash
# D160: the 1.5B readout chain starts only after the 7B random-weights chain has exited and released the queue lock
# (never two GPU chains at once).
cd "$(dirname "$0")/../.."
L=results/queue_followup_kaggle/chain.log
PID7=${PID7:?}
while kill -0 "$PID7" 2>/dev/null; do sleep 30; done
until flock -n results/queue_followup_kaggle/queue.lock true; do sleep 30; done
echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] the 7B chain (PID $PID7) has exited; starting the 1.5B chain" >> "$L"
MODEL=1.5b KAGGLE_ZIP="$HOME/kaggle_downloads/followup_1.5b_scale.zip" GPU_DEADLINE=09:00 \
  exec scripts/queue_followup_kaggle.sh >> results/queue_followup_kaggle/nohup.log 2>&1 < /dev/null
