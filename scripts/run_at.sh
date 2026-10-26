#!/usr/bin/env bash
# Start a command at a wall-clock time in IST, PID-tracked (your instruction, 2026-10-26: tonight's queue arms itself
# for 23:00 IST and launches without a message). Writes its own PID to PID_FILE, sleeps in POLL_S steps until START
# (HH:MM IST, today; already past -> at once), then execs the command, so the same PID becomes the chain's PID.
# Cancel before START with `kill $(cat PID_FILE)`; after START the same kill stops the chain from starting further
# stages (a running stage's child is not touched).
#
# usage:  START=23:00 PID_FILE=results/queue_r4_seeds/waiter.pid setsid nohup bash scripts/run_at.sh CMD... > LOG 2>&1 &
set -u
START=${START:?START=HH:MM (IST) is required}
PID_FILE=${PID_FILE:?PID_FILE is required}
POLL_S=${POLL_S:-30}
[ $# -gt 0 ] || { echo "usage: START=HH:MM PID_FILE=... run_at.sh CMD..." >&2; exit 2; }
mkdir -p "$(dirname "$PID_FILE")"
echo $$ > "$PID_FILE"
target=$(TZ=Asia/Kolkata date -d "$START" +%s) || exit 2
say() { echo "[$(TZ=Asia/Kolkata date +%H:%M:%S)] run_at($$): $*"; }
say "armed for $(TZ=Asia/Kolkata date -d "@$target" '+%Y-%m-%d %H:%M') IST: $*"
while now=$(date +%s); [ "$now" -lt "$target" ]; do
  s=$((target - now)); [ "$s" -gt "$POLL_S" ] && s=$POLL_S
  sleep "$s"
done
say "starting: $*"
exec "$@"
