# Free-space guard for the queue scripts (your instruction, 2026-10-23; DECISIONS.md D111). Sourced by every
# scripts/queue_*.sh after its `cd`; each stage launcher (run_stage, run_gpu, run_cpu) calls `disk_guard NAME || exit 75`
# before it starts a stage. The WSL disk image lives on the Windows volume MLStorage, which filled on the night of
# 2026-10-22/23 (STATUS_DISK_FULL). `df` inside WSL shows the image's virtual size, not the host's free space, so the
# guard asks Windows (powershell.exe). Below MIN_FREE_GB GiB free, or if the free space cannot be read (fail closed),
# it logs BLOCKED through the caller's log() and the queue stops cleanly: no new stage starts; running children are
# not touched. DRY_RUN=1 only logs the reading. DISK_GUARD=off skips the check (an explicit bypass, logged).
DISK_GUARD_LABEL=${DISK_GUARD_LABEL:-MLStorage}
MIN_FREE_GB=${MIN_FREE_GB:-15}               # GiB (2^30 bytes): the stricter reading of "15 GB"
DISK_GUARD=${DISK_GUARD:-on}
DISK_GUARD_TRIES=${DISK_GUARD_TRIES:-3}
DISK_GUARD_RETRY_S=${DISK_GUARD_RETRY_S:-5}
DISK_GUARD_TIMEOUT_S=${DISK_GUARD_TIMEOUT_S:-60}
PS_EXE=${PS_EXE:-/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe}

disk_free_bytes() {  # free bytes on the volume labelled $DISK_GUARD_LABEL; fails unless exactly one volume matches
  local i out
  for ((i = 1; i <= DISK_GUARD_TRIES; i++)); do
    out=$(timeout "$DISK_GUARD_TIMEOUT_S" "$PS_EXE" -NoProfile -NonInteractive -Command \
      "\$v = @(Get-Volume -FileSystemLabel '$DISK_GUARD_LABEL' -ErrorAction Stop); if (\$v.Count -ne 1) { exit 2 }; \$v[0].SizeRemaining" \
      2>/dev/null | tr -dc '0-9')
    [ -n "$out" ] && { echo "$out"; return 0; }
    [ "$i" -lt "$DISK_GUARD_TRIES" ] && sleep "$DISK_GUARD_RETRY_S"
  done
  return 1
}

disk_guard() {  # disk_guard STAGE_NAME: 0 = enough space (or dry run / bypass); 1 = BLOCKED, logged
  local name=$1 free gib
  if [ "$DISK_GUARD" = off ]; then log "disk guard OFF (DISK_GUARD=off) before $name"; return 0; fi
  if ! free=$(disk_free_bytes); then
    [ "${DRY_RUN:-0}" = 1 ] && { log "DRY disk guard: free space on $DISK_GUARD_LABEL unreadable"; return 0; }
    log "BLOCKED $name: free space on $DISK_GUARD_LABEL could not be read via powershell.exe; queue stopped"
    return 1
  fi
  gib=$((free / 1073741824))
  if [ "${DRY_RUN:-0}" = 1 ]; then log "DRY disk guard: $gib GiB free on $DISK_GUARD_LABEL (minimum $MIN_FREE_GB)"; return 0; fi
  if [ "$gib" -lt "$MIN_FREE_GB" ]; then
    log "BLOCKED $name: $gib GiB free on $DISK_GUARD_LABEL < $MIN_FREE_GB GiB; queue stopped"
    return 1
  fi
  log "disk: $gib GiB free on $DISK_GUARD_LABEL"
  return 0
}
