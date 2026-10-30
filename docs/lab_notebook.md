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

### 04:27 Heartbeat 2: eval at 59 %, no new pauses

- **`xm_eval_dpos2_a1`:** 1,206 / 2,040 problems after 7,339 s, and 201 / 340 batches on disk.
  - In the last hour: 576 problems at ~6.3 s each. That is not faster than the first hour, even though the prompts
    get shorter.
  - At that rate the remaining 834 take ~1.5 h, so the eval ends at ~05:55.
- **Temperature:** no pause since 03:15. Still 2 pauses, 60 s in total. The GPU holds at ~81 °C under load.
- **GPU:** 6,178 MiB device-wide, 93 % utilization. Free space: 47.2 GiB.

### 05:27 Heartbeat 3: eval at 87 %, one more short pause

- **`xm_eval_dpos2_a1`:** 1,782 / 2,040 problems after 10,875 s, and 297 / 340 batches on disk.
  - In the last hour: 576 problems at ~6.1 s each.
  - The remaining 258 take ~26 min, so generation ends at ~05:55. After that the eval scores the records and writes
    metrics.json.
- **Temperature:** pause 3 at 05:08:23 (87.6 °C), continued after 31 s (64.1 °C). The night's total is 3 pauses, 91 s.
- **GPU:** 80.6 °C, 6,895 MiB device-wide. Free space: 47.2 GiB.

### 05:44 Eval done: D-pos-v2 Llama s0 on A6c + A6b (exit 0)

- **`xm_eval_dpos2_a1`:** ended 05:44:35 with exit 0, in one attempt.
  - Stage wall time 12,330 s (3 h 25 min), of which 12,283 s were generation.
  - 3 temperature pauses took 91 s, so 0.7 % of the stage was paused. On 28/29 the same eval was paused 17 % of its
    wall time.
- **Resume block** (D142, first real use):
  - session 1: 0 of 340 batches read, 340 generated;
  - nothing moved aside;
  - the partial file was deleted after metrics.json, as designed.
- **Memory:** torch max reserved 5.86 GiB; nvidia-smi peak 6.76 GiB device-wide.
- **The queue's commit:** `2f0435f`, pushed. It holds metrics.json, summary.md, the plots, `rollouts.jsonl.gz`
  (1.7 MB), meta, and the stage log.
- **The headline numbers** (summary.md):

  | split | n | acc ≤ 1 % | median rel. err | 1σ cov | 2σ cov | median rel σ | parse fail |
  |---|---|---|---|---|---|---|---|
  | a6b_pairs | 1,020 | 8.3 % | 0.215 | 24.0 % | 34.7 % | 0.026 | 0.0 % |
  | a6c_pairs | 1,020 | 10.2 % | 0.124 | 28.5 % | 44.8 % | 0.037 | 0.4 % |

  - **Accuracy falls with depth on A6b:** 20.1 % at depth 1, down to 2.9 % at depth 5.
  - **Energy is the weak family:** 3.0 % on A6b, 4.4 % on A6c.
  - **Coverage on computation-correct answers is near nominal** (A6b at 1 % noise: 66.7 %). The pooled under-coverage
    comes from wrong answers.
  - **The standing post-hoc scale (× 18.1)** takes 1σ coverage to 65.5 % on A6b and 85.2 % on A6c.
  - **The go rule judges this record**, not these headline numbers.
- **`xm_go` started 05:44:40:** CPU only, 1 thread, nice 19, `CUDA_VISIBLE_DEVICES=`.

### 05:45 Go rule (D137 / D139) on D-pos-v2 Llama s0: NO-GO, failing on C3 with only 4 A6b pairs

- **`xm_go`:** 05:44:40 → 05:44:58, exit 0. The queue's commit is `37d8723`, pushed. The output is
  `results/stage_xm/go_dpos2/{go.json, summary.md}`.
- **The verdict:** **NO-GO**, with K1 True, C2 True, C3 False (C1 True).

  | criterion | estimate | 90 % CI | pairs | rule | met |
  |---|---|---|---|---|---|
  | K1 / C1 | A6c slope +0.785 | [+0.595, +0.974] | 505 | one-sided p < 0.05 (p = 0.0) | yes |
  | C2 | A6c slope +0.785 | | 505 | ≥ 0.5 | yes |
  | C3 | A6b slope +0.567 | [−3.55, +843] | 4 | CI inside (−0.3, +0.3); ≥ 3 pairs | no |

- **Why C3 failed:** too few computation-correct A6b pairs, not an A6b slope shown to be away from 0.
  - D137 flagged exactly this risk in advance: Qwen s0 had 17 such pairs, and a model with weaker arithmetic "may have
    too few for C3 to hold, whatever its propagation. Then the pilot is NO-GO, reported with that reason."
  - Llama's A6b accuracy is 8.3 % overall, falling to 2.9 % at depth 5. With 4 pairs, the bootstrap CI is unbounded in
    practice.
- **The pre-registered consequence:** no seed expansion. It is reported as a failed replication at one seed, with the
  same prominence as a success.
  - What did replicate: on Llama, the answer σ propagates on A6c (slope 0.785, CI clear of 0 and of 0.5).
  - What could not be shown: the A6b control.
  - Tonight's queue had no seed-1 stage either way (D144).
- **`mc_b_gen_a6b_pairs`** started 05:45:02 (timeout 4,800 s). The loader check is fine: tokenizer matches, 196 bnb-4bit
  modules.

### 06:18 MC-B A6b (D132, descriptive) done, and B-v2 SFT started

- **The stages** (MC-B uses Qwen2.5-Math-1.5B-Instruct 4-bit, as on A6c):
  - `mc_b_gen_a6b_pairs`: 05:45:02 → 06:18:02, exit 0, 1,982 s. 2 temperature pauses of 30 s (05:54 at 87.6 °C;
    06:10 at 87.2 °C). Commit `023bcfa`.
  - `mc_b_exec_a6b_pairs`: 10 s, exit 0, CPU, 1 thread, nice 19.
  - `mc_b_analyze_a6b_pairs`: 21 s, exit 0. Commit `876f4ef` (`results/mc_b/a6b_pairs/{executions.jsonl, results.json,
    summary.md}`).
- **The programs** (A6b, 1,020 problems, all used):
  - failure rate 87.7 %: no_program 16.5 %, signature 69.1 %, exception 1.7 %, rejected 0.4 %, bad_output 0.1 %;
  - truncated 11.1 %;
  - run 0 computation-correct 1.3 %;
  - by depth, failure runs from 83.8 % at depth 1 to 98.5 % at depth 5.
  - A6c had 84.8 % failure and 0.5 % correct, so it is the same picture.
- **MC-B's σ is collapsed:** median ln(σ/σ\*) = −10.4, 1σ / 2σ coverage 6.5 % / 8.9 % (k = 2). The σ reference points:

  | σ source | expected crps_log | 1σ / 2σ cov | A6b slope [90 % CI] |
  |---|---|---|---|
  | mc_b_k8 | −0.0402 | 7.9 % / 10.1 % | −0.018 [−0.13, +0.09] |
  | mc_a_robust | −0.0512 | 67.0 % / 86.0 % | +0.103 [+0.01, +0.19] |
  | dpos2_a (Qwen) | −0.0327 | 68.7 % / 95.1 % | +0.010 [−0.02, +0.05] |
  | rss | −0.0300 | 65.9 % / 90.6 % | 0 |
  | sigma_star | −0.0292 | 68.0 % / 95.1 % | 0 |

- **The differences** (descriptive on A6b, no Holm; A6c holds the tests):
  - M1, MC-B − RSS: −0.0102 [−0.0109, −0.0094].
  - M2, MC-B − X1: −0.0105.
  - M3, MC-B − D-pos-v2 (a): −0.0075 [−0.0091, −0.0056].
  - **M4, MC-B − MC-A robust: +0.0110 [+0.0058, +0.0168].** This is the opposite sign to A6c (−0.0037).
    - The likely reason: MC-A robust has a non-zero A6b slope (+0.103, CI clear of 0) on a set where the right slope is
      0.
  - **M5:** the A6b slope is −0.018 [−0.15, +0.11]. The summary labels this row "MC-B's A6c slope". That is a label
    bug in the summary text; the number is A6b's. To fix after the night; no analysis code is touched tonight.
- **The per-depth slope table on A6b is degenerate** (values in the millions, CIs of ±10¹⁰, and sigma_star at +1.000
  per depth against 0.000 pooled). Within a depth, A6b pairs barely differ in ln truth. Do not read it.
- **The conclusion:** A6b agrees with FINDINGS #33. MC-B fails as a σ source because its programs fail, not because of
  its σ rule.
- **`xm_train_b2`** started 06:18:42 with `--stop-at 12:45`. Timeout 23,779 s.
  - The start gate's estimate: 148 min = 8,800 s / (1 − 0.011). The paused share tonight is 1.1 %. The projection check
    starts 30 min in.

### 06:49 B-v2 projection 1: ends ~08:36

- **The projection:** 171 / 750 steps in 28:51, pauses included, which projects an end at 08:36 IST. That is before
  12:45, so the run continues. The pace is ~10.1 s / step wall; 28/29's estimate was 11.0 s.
- **Temperature:** 2 pauses in B-v2 so far, 30 s each (06:24:43 at 87.4 °C; 06:48:15 at 87.2 °C). Tonight's total is
  7 pauses, 211 s.
- **GPU:** 82.6 °C, 5,145 MiB, 100 % utilization.

### 08:04 Heartbeat: B-v2 at 82 %, still projecting ~08:37

- **The projections** (15-min interval), each ending well before 12:45:

  | time | steps | wall | projected end |
  |---|---|---|---|
  | 07:03 | 257 | 44:24 | 08:38 |
  | 07:18 | 346 | 59:22 | 08:38 |
  | 07:33 | 434 | 1:14:17 | 08:37 |
  | 07:48 | 520 | 1:29:19 | 08:38 |
  | 08:03 | 612 | 1:44:23 | 08:37 |

- **Temperature:**
  - **B-v2 so far:** 9 pauses, 30–31 s each, all at 87.1–87.9 °C, continued at 60.3–61.6 °C. The GPU trips roughly
    every 10–15 min under SFT load.
  - **Tonight:** 14 pauses, 422 s.
- **GPU:** 83.7 °C, 5,515 MiB. Free space: 47.2 GiB.
- **The watcher change** (scratchpad `watch.sh`): it no longer wakes on "B-v2 projection … before 12:45" lines. It
  still wakes on "cannot finish", a stage end, BLOCKED, a stall, the chain's exit, and the hourly heartbeat.

### 08:31 Chain done; post-run checks and two fixes (D145)

- **`xm_train_b2`:** ended 08:31:25, exit 0.
  - 750 / 750 steps, 132 min wall; 11 pauses, 331 s. Train loss 0.155. Peak reserved 4.48 GiB.
  - The fingerprint is recorded. The queue's commit is `f31dcc7`, pushed.
  - The last projections (08:18: 698 / 750) said 08:37; it ended 08:31.
- **Chain 35695 exited at 08:31:28:** "tonight 16 temperature pauses, 482s paused of 22276s GPU-stage time".
- **Post-run checks:**
  - **Clean:** no queue process, no GPU compute process (785 MiB device-wide, i.e. the host); 47.17 GiB free;
    `HEAD` = `origin/main` = `f31dcc7`.
  - **The integrity check** on the 64 files written since 20:48 UTC: 0 problems. It parsed both JSONL tapes and the gzip
    (2,040 lines = its sibling), and the B-v2 safetensors is structurally ok (392 tensors, 0 all-zero).
  - **The queue lock was still held at 08:32** by PID 93166, a `sleep 900` in the chain's process group, child of init,
    started 08:18:45. It was the B-v2 projection's sleep.
    - `kill "$proj"` had ended the subshell but not the sleep, and the sleep inherited fd 8 (the lock).
    - It exited by itself at 08:33:45, and the lock was then free.
- **D145, the fixes:**
  - `scripts/queue_xm2.sh`: `b2_projection … 8>&- &`, and `stop_projection` (SIGSTOP, kill children, SIGTERM,
    SIGCONT, wait).
  - `scripts/gpu_temp_guard.sh`: one reading on the cooled-pause line.
  - **The tests:** 1 new in `tests/test_queue_xm2.py`, 2 tightened in `tests/test_temp_guard.py`. Both files: 13 passed.
  - **The old stop pattern, re-run in a scratch test:** the orphan `sleep 900` survived and held the harness's pipes
    until its 120 s timeout. That confirms the bug and that the test catches it.
- **The full suite after the night** (08:40–08:51, GPU idle, with the D145 fixes): 700 passed, 0 failed, 0 skipped.
- **MORNING.md** for 29/30 is written, and PROGRESS.md is updated.
