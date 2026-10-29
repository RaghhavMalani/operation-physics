# Lab notebook

Running record of what ran, why, the decisions taken, the exact commands and the commits. Newest night last. Times
are IST. The DECISIONS.md entries are the formal record; this file is the working log that leads to them.

---

## 2026-10-29/30 night: recovery, resumable evals, temperature hysteresis, then the XM queue

Brief (your message, ~01:10 IST): (1) recovery checks, (2) resumable evals, (3) temperature-guard hysteresis,
(4) tonight's queue (`GPU_DEADLINE=13:00`), (5) watch overnight, then MORNING.md.

### 01:11 Step 1: recovery checks

- **Clock and system:** 01:11 IST on Fri 30 Oct; WSL up 3 min (Windows booted at 00:57 IST).
- **git:**
  - `git fsck --full`: dangling objects only, exit 0.
  - `git fetch`: local `main` = `origin/main` = `ac78c5f`.
  - Uncommitted: the six queue logs from 05:51 onward, `xm_train_b2.log`, the D-pos-v2 adapter's untracked weights
    and tokenizer files, and the PID / lock files.
- **Processes:** none. Recorded PIDs 10129, 53297, 102507, 10252 and 102646 are all dead.
  - `flock -n` on `results/queue_xm/queue.lock` and `followup/followup.lock`: both free.
- **GPU:** 57.4 °C, 916 MiB used, no compute process.
- **Disk:** MLStorage has 47.17 GiB free (`Get-Volume -FileSystemLabel MLStorage`).
- **Integrity:**
  - `find . ( -path ./.venv -o -path ./.git -o -path ./.cache ) -prune -o -type f -newermt 2026-10-28T12:30:00Z`
    listed 64 files.
  - `.venv/bin/python -I scripts/integrity_check.py "$PWD" files.tsv` found 0 problems (exit 0).
  - Its one note: `xm_train_b2.log` has no final newline (a freeze mid-progress-bar).
- **Windows System log, 28 Oct 18:00 to now:**
  - no nvlddmkm event (no TDR) and no BugCheck;
  - normal service events until `shutdown.exe` (User32 1074) at 09:54:29 on the user's account;
  - the next boot says "last shutdown's success status was true";
  - a sleep and resume at 10:13–10:14, a reboot at 17:23 and one at 00:57 (30 Oct).
- **Night 28/29 pause accounting**, from chain.log (stage start → its pauses):

  | stage | pauses | paused | stage wall | share |
  |---|---|---|---|---|
  | SFT dpos2 | 23 | 230 min | 365 min | 63% |
  | eval dpos2 | 5 | 50 min | 300 min | 17% |
  | SFT b2 | 12 | ≥ 110 min | ~189 min | ≥ 58% (63% counting the last pause) |
  | night (18:46:37 → 09:00:50) | 40 | 390 min | 854 min | 46% |

  The GPU read 53–64 °C at every resume, so the fixed 10-min pause cooled it 25–35 °C below the trip point.
- **Set aside:** `results/stage_xm/b2/seed0` (an empty `checkpoints/` dir and nothing else) went to
  `results/discarded/xm-b2-seed0-hang-2026-10-29/seed0`. `results/stage_xm/b2/` was removed (empty).
- **Written:**
  - D140 (this recovery);
  - D141 (family H declared not run, with reasons);
  - MORNING.md for night 28/29;
  - `.gitignore` lines for stage_xm adapters, partial files and the XM queues' PID / lock files;
  - decision_timeline.md and paper_outline.md now cite D141.
