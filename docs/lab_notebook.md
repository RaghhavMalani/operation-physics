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
- **Commit:** `d911d45` (recovery, D140, D141, MORNING.md for 28/29), pushed.

### 01:30 Step 2: resumable evals (D142)

- **Design** (`physcal/eval/run_eval.py`):
  - **Batches:** `HFBackend.__call__` keeps its partition: length-sorted, cut every `batch_size`. It now runs it through
    `generate_batched`.
  - **`PartialCompletions`** writes one JSONL line per finished batch to `completions.partial.jsonl`, with a single
    O_APPEND write and fsync per line.
  - **On restart** it reads the file back and skips the batches already done.
  - **Header:** an identity (model, adapter hash, generation config, BOS switch, prompts hash, package versions).
    - **A file is moved aside, never mixed in,** on a mismatch, on damage before the last line, or on a batch line
      that does not match the partition.
    - **A torn last line** is cut off.
  - **`run(..., resume=True)`:** greedy evals only. It records a `resume` block in metrics.json / meta / summary.md
    and deletes the partial file after metrics.json.
- **Why batch granularity:** greedy outputs can depend on batch composition (left padding; D137). Re-using the
  single-run partition is what makes byte identity possible at all.
- **CPU tests:** `tests/test_resume_eval.py`, 8 passed (44 s).
- **The GPU acceptance check:** `physcal/eval/resume_check.py`, 30 dev problems (D137's equivalence set).
  - **The passes:** (a) uninterrupted, (b) SIGKILL once ≥ half its batches are on disk, (c) resume.
  - **Accepted iff** (c) ≡ (a) byte for byte and (c) both read and generated batches.
  - **How it runs:** through the queue library's `run_stage` under the guards, `STALL_S=3600` (the orchestrator's own
    log is quiet while its child runs write elsewhere):

    ```
    cd scripts && HF_HUB_OFFLINE=1 QUEUE_LOG=<scratch>/rcq QUEUE_LIB_ONLY=1 STALL_S=3600 \
      bash -c '. ./queue_r4_seeds.sh && GUARD=1 run_stage resume_check_llama 3600 \
               uv run python -m physcal.eval.resume_check check llama --out <scratch>/rc_llama'
    ```

  - **Two launcher false starts** (exit 75 before any GPU work): the library `cd`s to `$(dirname $0)/..`, so it must
    be sourced as `. ./queue_r4_seeds.sh` from `scripts/`, as the tests do.
- **Llama check started 01:35:37.**

### 01:40 Step 3: temperature hysteresis (D143)

- **`scripts/gpu_temp_guard.sh`:**
  - **Unchanged:** the trip at > 87 °C.
  - **Changed:** during a pause it reads every 30 s (`CHECK_S`) and SIGCONTs at the first reading ≤ 75 °C
    (`RESUME_C`), or after 600 s (`PAUSE_S`) at most.
  - **Fail closed:** an empty reading never ends a pause.
  - **The log line:** "stage continued after Ns paused (reason; GPU at T C)".
- **`cool()` in `scripts/queue_r4_seeds.sh`:** the same rule between stages, with its log line "pause ended after Ns".
- **Tests:**
  - `tests/test_temp_guard.py`: 6 passed;
  - the D129 / D132 guard tests in `tests/test_r4_confirm.py`: 13 passed together.

### 01:45 Step 4 prep: tonight's queue (D144)

- **`scripts/queue_xm2.sh`:**
  - **The order:** eval dpos2 (resumable, ≤ 3 attempts) → go rule (CPU) → MC-B A6b → SFT B-v2 (gate + projection).
  - **Not run:** seed 1, B-v2 eval, `xm_pilot`.
- **Tests:** `tests/test_queue_xm2.py`, 6 passed.
  - **They cover:** dry-run order, skips, partial-file count, and the projection stopping a run that cannot finish
    while leaving one that can.
  - **A test-only zombie:** the first version of the "can finish" test hung, because the fake stage was the test's
    unreaped child and a zombie still answers `kill -0`. In the queue, `run_stage`'s `wait` reaps the stage.

### 01:56 GPU check, Llama: ACCEPTED

- **The result** (`results/stage_xm/checks/resume_llama.json`): 30 / 30 byte-identical (text, n_tokens, truncation,
  reward), batch 6, 5 batches.
- **The three passes:**
  - (a) single: 478 s (fingerprint check pass; 5 batches in 165 s of generation).
  - (b) SIGKILLed after 434 s, with 3 of 5 batches on disk.
  - (c) resumed: 336 s. It read 3 batches, generated 2, and logged "resuming, 3 of 5 batches on disk (session 2)".
- **D143 in live use:** the guard paused run (b) at 01:49:09 (87.2 °C) and continued it after 30 s, at 63.9 °C.
  - The fixed rule would have idled 600 s.
  - The completions stayed byte-identical across that mid-generation SIGSTOP.
- **The Qwen check started at 01:56.**

### 02:05 GPU check, Qwen: ACCEPTED

- **The result** (`results/stage_xm/checks/resume_qwen.json`): 30 / 30 byte-identical, batch 16, 2 batches.
- **The three passes:** (a) 183 s; (b) SIGKILLed after 154 s, with 1 of 2 on disk; (c) 138 s, reading 1 and
  generating 1.
- **D142's acceptance criterion is met for both models.** The full test suite started at 02:05 (GPU idle, default
  threads).

### 02:16 Full test suite

- **The command:** `uv run pytest -p no:cacheprovider -rs` (02:05–02:16, GPU idle).
- **The result:** 696 passed, 3 failed.
- **The 3 failures:** `test_disk_guard.py::test_every_queue_script_guards_every_stage_launcher` on `queue_xm.sh`,
  `queue_xm_followup.sh` (both failing since 28 Oct) and `queue_xm2.sh`.
  - **The cause:** they source the guard through `queue_r4_seeds.sh` in library mode, not on the line after their `cd`.
  - **The fix:** the test now accepts library mode, with three checks: only settings before the library, no launcher
    of their own, stages via `run_stage`.
  - **After the fix:** 27 / 27 in that file, so 699 / 699 overall.
- **Dry run of `queue_xm2.sh`:** the order is eval → go → MC-B gen / exec / analyze → B-v2 SFT, with a B-v2 estimate of
  146 min at a share of 0.
- **Host VRAM** with nothing in WSL: 828–834 MiB, so the launch cap is 5.9 GiB.

### 02:18 Launch (D144)

```
mkdir -p results/queue_xm2
GPU_DEADLINE=13:00 setsid nohup scripts/queue_xm2.sh >> results/queue_xm2/nohup.log 2>&1 < /dev/null &
```

- **The chain:** PID 35695, started 02:18:47, 641 min left.
- **Host VRAM:** 741–751 MiB, so the launch cap is 5.9 GiB.
- **`xm_eval_dpos2_a1`** started 02:19:07 (timeout 38,455 s = the deadline). Free space: 47 GiB.
- **The R2-1 reload check passed** at ~02:24: adapter logit delta 3.97; tokenizer matches; 196 bnb-4bit modules.
- **The partial file:**
  - **Header identity:** `lora_hash 3c99b36c58b68995`, 340 batches.
  - **Batches:** 2 on disk by 02:26 (~43 s per batch on the longest prompts).
  - **The GPU then:** 81 °C, 5.86 GiB device-wide.
- **The watcher:** this session runs `watch.sh 35695 results/queue_xm2 <next line> 3600` in the background. It exits on a
  stage end / BLOCKED / SKIP / go rule / B-v2 line, on a suspected stall (stage log silent > 15 min while not
  SIGSTOPped), on the chain's exit, or on a 1-h heartbeat.

### 03:26 Heartbeat 1: eval running, two short pauses

- **`xm_eval_dpos2_a1`:** 630 / 2,040 problems generated after 3,685 s, and 105 / 340 batches on disk.
  - The prompts run longest first, so the rate should improve: ~5.8 s / problem so far.
- **Temperature pauses:** 2, each 30 s.
  - 03:02:11 at 87.8 °C, continued at 64.4 °C.
  - 03:15:13 at 87.7 °C, continued at 63.8 °C.
  - Under the 28/29 rule each would have been 600 s. That is 19 min of GPU time saved so far.
- **GPU:** 81.4 °C, 6,905 MiB device-wide, 93 % utilization. Free space: 47.2 GiB.
- **Cosmetic issue:** the guard's resume line repeats the reading ("GPU at 64.40C <= 75C; GPU at 64.40C").
  - It is left as is, because the guard script is running and bash reads scripts as it goes. Tidy up after the night.
