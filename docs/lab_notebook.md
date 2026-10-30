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

---

## 2026-10-30 day: your decisions on the 29/30 report (D146)

Brief (your message, ~10:00 IST):
1. NO-GO accepted; report D137's descriptive A6b slope on all answered pairs.
2. Launch the B-v2 Llama eval (resumable) and `xm_pilot` now, with `GPU_DEADLINE=13:00`.
3. MC-B M5 label fix + A6b re-run.
4. FINDINGS #38.
5. An R5-6 estimate.

### 10:10 D146, the queue, the launch

- **D146 recorded** before anything ran.
- **`scripts/queue_xm2.sh`:** its eval loop is now the library function `eval_xm ARM TAG`, so both queues share one
  tested path. The xm2 tests pass unchanged.
- **`scripts/queue_xm3.sh`:** library mode (queue_r4_seeds.sh + queue_xm2.sh). Two stages:
  - `eval_xm b2 D146`;
  - `xm_pilot analyze` (CPU, 1 thread, nice 19, timeout 7,200 s), once, when both evals have metrics.json.
- **Tests:** `tests/test_queue_xm3.py`, 3 tests; the queue, disk-guard and xm2 files together: 38 passed.
- **Commit:** `2975667`, pushed.
- **Pre-launch checks:** no B-v2 confirmatory dir yet; GPU 46.8 °C, 838 MiB, no compute process; xm3 lock free;
  47.17 GiB free.
- **The launch** (10:14:30):

  ```
  GPU_DEADLINE=13:00 setsid nohup scripts/queue_xm3.sh >> results/queue_xm3/nohup.log 2>&1 < /dev/null &
  ```

  - Chain PID 114795, 165 min left. Host 833 MiB, so the launch cap is 5.9 GiB.
  - `xm_eval_b2_a1` started 10:14:48 (timeout 9,913 s).
  - **The R2-1 reload check passed** ("unsloth_4bit fingerprint check: pass").
  - **The rate:** 48 / 2,040 after 189 s, ~3.9 s / problem, so generation ends at ~12:35.
- **The watcher:** `watch.sh 114795 results/queue_xm3 8 3600`. Its event pattern now matches any "chain done" and
  "xm_pilot not run".

### 10:15 D146 item 3: the MC-B M5 label

- **`physcal/eval/mc_b.py`:** `TEST_NAMES["M5"]` = "MC-B's {set} slope > 0", filled from `res["set"]` (a6b_pairs →
  A6b).
- **A6c is unchanged:** its stored results re-render byte-identically to the committed summary.md.
- **The test:** `test_the_m5_row_names_the_set_it_was_computed_on`. `tests/test_mc_b.py`: 24 passed (1 thread).
- **The A6b re-run** (`mc_b analyze --set a6b_pairs`, CPU, 1 thread, nice 19, 23 s, exit 0):
  - results.json is exactly equal to the copy saved before;
  - summary.md differs in one line, the M5 label.
- **Commit:** `f69af81`, pushed. It also corrects D146's time (~10:10, not ~09:10).

### 10:20 D146 item 5: the R5-6 estimate (`docs/r5_6_estimate.md`)

- **The CPU checks** (1 thread, nice 19):
  - the SFT rows contain 203 of the 204 formula-to-formula transitions in any valid chain;
  - the one unseen transition is in kinematics, in 3 chains;
  - many whole chains are unseen at depths 2–4; depth 5 is entirely unseen.
- **What follows:**
  - `generator_version` hashes `configs/env.yaml`'s generator section only, so new relations in an existing family
    would silently change old seeds' problems. New relations need a new family (optics; `heldout_families` already
    lists it).
  - The SFT rows store per-given sensitivities, so the Jacobian training support can be read directly.
- **The estimate:**
  - Part J (out-of-support Jacobians, a construction filter): ~0.5 day of code, ~3.2 GPU-h.
  - J + U (an optics family): ~1.5–2 days, ~5.6 GPU-h.
  - Both are for Qwen D-pos-v2 + B-v2 × 3 seeds, at the rates of the confirmatory evals.

### 12:13 B-v2 eval done, and the pilot analysis

- **`xm_eval_b2_a1`:** 10:14:48 → 12:12:41, exit 0, first attempt.
  - Generation 7,028 s (3.4 s / problem). 2 pauses of 30 s (11:42:27 at 87.2 °C; 11:59:59 at 87.4 °C).
  - The resume block: session 1, 0 read, 340 generated.
  - Memory: torch max reserved 5.20 GiB. The nvidia-smi peak is 7.46 GiB device-wide, above last night's 6.76; the
    host's own use was higher during the day. It stayed under 8 GiB.
  - **The new one-reading guard line in use:** "(GPU at 63.40C <= 75C)".
  - The queue's commit: `2892278`.
  - Headline: A6b 9.3 % accurate, A6c 10.3 %; 0 parse fails; median rel σ 0.023 / 0.035.
- **`xm_pilot`:** 12:12:47 → 12:13:21, exit 0, 34 s at 1 thread. The queue's commit: `ffc4ee4`.
  - **The go rule matches go.json:** K1 ✓, C2 ✓, C3 ✗ on the same numbers.
  - **The descriptive A6b slope on all answered pairs** (D146 item 1): Llama D-pos-v2 (a) −0.168 [−0.220, −0.119],
    510 pairs; Qwen s0 +0.010.
  - **Explicit vs implicit on Llama:**
    - A6c slope +0.785 against B-v2's +0.396;
    - expected crps_log D-pos-v2 − B-v2 = −0.0089 [−0.0188, −0.0009], so the explicit arm scores worse (Qwen s0:
      +0.0004);
    - the tail: 36 decimal shifts among the worst 102 (Qwen 17).
  - **Format:** ≥ 99.4 % on every measure for both arms. Computation-correct: 8.0 % in both arms (A6b 4.9 % for
    D-pos-v2).
- **The chain ended at 12:13:23.** No queue process left; the lock is free; no GPU compute process;
  `HEAD` = `origin/main` = `ffc4ee4`.
- **Full suite after the stages** (GPU idle): 705 passed, 0 failed (700 + 3 queue_xm3 + 1 disk-guard case + 1 mc_b).
- **FINDINGS #38 written.** `docs/decision_timeline.md` (D137–D139 rows, the summary list, the anchor) and
  `docs/paper_outline.md` (status, FINDINGS list) now cite it.

## 2026-10-30 afternoon: R5-6 part J (your decision at ~14:30 IST: J only; D147–D149)

### 14:30 The design, and a generator-only feasibility count (no model)
- **The training support** is per chain: both R4 arms trained on the same 3,000 R4-1 problems, SFT only.
  - **The catch:** chains are sparse. At depths 2–4 most chains occur 1–3 times (e.g. dynamics d3: 166 chains in 248
    rows; 13 with ≥ 3 rows).
- **Feasibility on development seeds (900,000 + …), per trained-chain draw with 24 / 96 worlds,** of a world with an
  exceedance ≥ 0.1, magnitudes in range and A6c's ratio ≥ 1.5:
  - kinematics 40–100%; energy 15–70%; dynamics 1–35%.
  - **Dynamics at depth 1 is ~4%:** most of its chains are power laws (constant Jacobian).
- **Two alternative supports, counted and rejected as the primary:**
  - **per (family, given):** 0–15% per cell;
  - **per (family, given, target):** dynamics d1 0%. It became the "strict" subgroup.
- **Invariance:** A6b's power-law chains can never be outside the range (constant Jacobian).
  - **The design:** dimensional scaling along directions that fix every constant (g):
    - kinematics: L, T (and M, unused);
    - dynamics / energy: (L², T), M.
  - Feasibility matched the sensitivity pairs.
- **The parameters set from the count:** margin 0.1, 96 candidate worlds. Recorded in D147.

### 14:39 D147 (pre-registration) and D148 (freeze), `8405f0f`, pushed before any J data; outline caveat `e4fd8fa`

### 14:40–14:46 Build
- **The support:** `jacobian_support build`: 1,149 chains, SHA-256 `eda3c3b2…`.
- **The smoke** (4 pair seeds × 24 cells, development seeds): every cell produced pairs except dynamics d1 (1–2 of 6),
  and every pair replays.
- **The J tests:** 14 at first.
  - **One fix:** the directions test asserted that every direction scales something. Kinematics' mass direction scales
    nothing (no mass quantity), so the test now asks for ≥ 2 active directions per family.
- **The frozen build:** `pairs export j_a6c` / `j_a6b`, 4 workers, nice 19, GPU idle, 18 s + 17 s.
  - All 24 cells full (34 pairs each). Sensitivity dynamics at d1: 1 / 0 / 3 pairs.
  - **Replay of all 1,632 problems** (4 workers): 816 / 816 and 816 / 816 identical.
  - `FROZEN.md` and `ada4ede`.
- **The distances (generator only):**
  - median exceedance 0.38 (sensitivity) / 0.34 (invariance); nearest training Jacobian (L∞) 0.41 / 0.38; chain
    training count 3;
  - **the clamped ln σ ratio (hi member) only 0.018 / 0.060:** in many pairs the out-of-range term contributes little
    to σ\*;
  - strict 27% / 35%; rank flip 11%.

### 14:48 Launch
- **The eval option** (`stage_r4 eval --r5-6`), the queue (`scripts/queue_r5_6.sh`), the .gitignore lines. `b31ef28`.
- **Chain 151740** with `GPU_DEADLINE=10:00`.
  - Host 1.4 GiB, so the cap is 5.9 GiB.
  - `r56_eval_dpos2_s0_a1` started 14:49:01.
  - ~2.1 s / problem at the start (longest batches first), 80–81 °C, 6.2 GiB device-wide.
- **The watcher:** `watch.sh 151740 results/queue_r5_6 8 3600`.

### 14:50 X1 on J (`x1_a6c r5_6`, 1 thread, nice 19, 118 s)
- Reproduction passed (max abs difference 0.0); development predictions identical; 1,632 predictions. `4fa3c69`.

### 14:55 The analysis (`physcal/eval/r5_6.py`)
- **The refactor:** `r4_confirm.load_data` factored out. R4-3's three-identical-seeds test passes (66 s).
- **`tests/test_r5_6.py`** (4): the development A6b / A6c with seed 0's tapes × 3 and injected J metadata.
  - J1 = R1, J2 = R1 − R4 (1e-12); P_J follows D147 §8; every split runs; 1,020 distance rows.
- `68449cb`.

### 15:13 Full suite (CPU, 1 thread, nice 19, during the first eval): 728 passed, 1 skipped, 0 failed (13 min 50 s)
- 705 before + 23 new (test_pairs_j 16, test_r5_6 4, test_queue_r5_6 3).
- D-pos-v2 s0 at 624 / 1,632.

### 15:49 J eval 1 of 6 done: D-pos-v2 s0 (operational counts only, D147 §11)
- `r56_eval_dpos2_s0_a1`: 14:49:01 → 15:48:53, exit 0, first attempt; the R2-1 fingerprint check passed.
  - Generation 3,534 s (2.17 s / problem); 102 / 102 batches in one session; 0 OOM backoffs.
  - Memory: torch max reserved 4.49 GiB; the nvidia-smi peak is 6.58 GiB device-wide.
  - **3 pauses of 30 s** (> 87 °C at 15:26, 15:44, 15:47; each resumed at ~64 °C).
- **Counts:** J invariance 816 problems, 814 answered, parse fail 0.2%; J sensitivity 816, 816 answered. Truncation 0,
  unit mismatch 0; mean 393 / 404 tokens.
- The queue's commit was pushed. `r56_eval_b2_s0_a1` started 15:49:00.

### 16:22 J eval 2 of 6 done: B-v2 s0 (operational counts only)
- b2/seed0/r5_6: gen 1909s, OOM backoffs 0, peak 6.02 GiB device-wide, resume 0+102/102
-   a6b_pairs: n 816, answered 816, parse fail 0.000, truncation 0.000, unit mismatch 0.000, mean tokens 219
-   a6c_pairs: n 816, answered 815, parse fail 0.001, truncation 0.000, unit mismatch 0.000, mean tokens 224
- 15:49:00 → 16:21:47, exit 0, first attempt; fingerprint check passed; 1 pause of 30 s. `r56_eval_dpos2_s1_a1` started 16:21:53.

### 17:21 J eval 3 of 6 done: D-pos-v2 s1 (operational counts only)
- dpos2/seed1/r5_6: gen 3465s, OOM backoffs 0, peak 5.61 GiB device-wide, resume 0+102/102
-   a6b_pairs: n 816, answered 815, parse fail 0.001, truncation 0.000, unit mismatch 0.000, mean tokens 392
-   a6c_pairs: n 816, answered 815, parse fail 0.000, truncation 0.000, unit mismatch 0.001, mean tokens 404
- 16:21:53 → 17:20:34, exit 0, first attempt; fingerprint check passed; 0 pauses. `r56_eval_b2_s1_a1` started 17:20:40.

### 17:54 J eval 4 of 6 done: B-v2 s1 (operational counts only)
- b2/seed1/r5_6: gen 1948s, OOM backoffs 0, peak 5.24 GiB device-wide, resume 0+102/102
-   a6b_pairs: n 816, answered 816, parse fail 0.000, truncation 0.000, unit mismatch 0.000, mean tokens 219
-   a6c_pairs: n 816, answered 816, parse fail 0.000, truncation 0.000, unit mismatch 0.000, mean tokens 224
- 17:20:40 → 17:54:03, exit 0, first attempt; fingerprint check passed; 1 pause of 30 s. `r56_eval_dpos2_s2_a1` started 17:54:09.

### 18:53 J eval 5 of 6 done: D-pos-v2 s2 (operational counts only)
- dpos2/seed2/r5_6: gen 3453s, OOM backoffs 0, peak 5.60 GiB device-wide, resume 0+102/102
-   a6b_pairs: n 816, answered 813, parse fail 0.004, truncation 0.000, unit mismatch 0.000, mean tokens 389
-   a6c_pairs: n 816, answered 809, parse fail 0.005, truncation 0.001, unit mismatch 0.004, mean tokens 405
- 17:54:09 → 18:52:38, exit 0, first attempt; fingerprint check passed; 0 pauses. `r56_eval_b2_s2_a1` started 18:52:44.

### 19:25 J eval 6 of 6 done: B-v2 s2 (operational counts only); the analysis starts
- b2/seed2/r5_6: gen 1892s, OOM backoffs 0, peak 5.27 GiB device-wide, resume 0+102/102
-   a6b_pairs: n 816, answered 816, parse fail 0.000, truncation 0.000, unit mismatch 0.000, mean tokens 218
-   a6c_pairs: n 816, answered 816, parse fail 0.000, truncation 0.000, unit mismatch 0.000, mean tokens 223
- 18:52:44 → 19:25:12, exit 0, first attempt; fingerprint check passed; 0 pauses.
- **All six evals ran on their first attempt.** In all: 5 pauses, 150 s of 16,552 s GPU-stage time (0.9%).
- `x1_r5_6` skipped (done at 14:50). `r5_6_analysis` started 19:25:24 (CPU, 1 thread, nice 19).

### 19:26 The analysis and the chain's end
- **`r5_6_analysis`:** 19:25:24 → 19:26:10, exit 0 (46 s, 1 thread). The queue's commit `f5835fc` was pushed.
  "R5-6 J chain done" at 19:26:12.
- **After the chain:** PID 151740 gone; the lock is free; no GPU compute process; `HEAD` = `origin/main` = `f5835fc`.
  - **The integrity check** on the 122 files written since 14:30 IST: 0 problems (`find -newermt` takes UTC: a first
    run with "14:30" matched no file).
  - Free space on MLStorage 47.17 GiB.
- **The result (FINDINGS #39): P_J confirmed.**
  - J1 +0.985 [+0.868, +1.103] (seeds 0.941 / 0.967 / 1.047; 403–407 answered pairs); C3 +0.026 [−0.055, +0.119] on
    72–86 computation-correct pairs per seed.
  - J2 +0.530 rejected. J3 +0.0011 (p 0.072), J4 0.0000, J5 −0.0005, J6 −0.0016: none rejected.
  - **The caveat:** the clamped comparator reaches +0.831 [+0.737, +0.919]. Rank-flip pairs (43): D-pos-v2 +1.123 vs
    clamped +0.357.
  - B-v2: J-sensitivity +0.455; J-invariance +0.184 [−0.032, +0.400], so C3 fails.
  - S lines within 10%: out-of-range terms 58–64%, in-range 69–75%.
- **Docs:** FINDINGS #39; `decision_timeline.md` (D147–D149 → #39; the anchor checked against #38's slug rule); the
  paper outline (status, claims to avoid, ledger); PROGRESS; MORNING.md.

## 2026-10-30 night: the write-up starts (your instruction at ~21:45 IST; D150)

### ~21:50–22:50 paper/ and scripts/paper/
- **Tooling:** no TeX was installed. tectonic 0.15.0 is the GitHub release binary, in `~/.local/bin` (SHA-256
  `4df19452…`). Its first compile fetched the packages.
- **`scripts/paper/`:**
  - `paperlib.py`: Sources (tracked and unchanged files only, with blob hashes), siunitx formatting, the Unicode map,
    an ASCII check;
  - `macros.py`: 114 values;
  - `tables.py`: 10 tables;
  - `figures.py`: the headline figure through the same source check;
  - `build.py`.
- **Fixed along the way:**
  - unescaped `_` in table notes;
  - markdown escapes (`\*`) leaking from the timeline;
  - `§` in macro comments;
  - table headers not listing their sources (the artifact name differed from the file name);
  - siunitx grouping decimals ("0.002,96"): now `group-digits=integer`;
  - floats drifting past the references: `placeins` [section];
  - three hand-typed numbers in the prose ("4-bit", "16-bit", "#1"): now a macro or reworded;
  - one wrong statement: exact givens in mixed-level problems are shown at 5 s.f., not 3 (D113). Rewritten.
- **The layout check:** pages rendered to PNG with a throwaway PyMuPDF (`uv run --no-project --with pymupdf`; no
  poppler on the machine).
  - The result: 16 pages, no overfull boxes, no undefined references.
  - The build is byte-reproducible: the same md5 for headline.pdf over two builds.
- **`tests/test_paper.py`:** 11 passed.

## 2026-10-31 night: the exploratory follow-up track (your instructions at ~01:00 IST; D151–D153; not the paper)

### ~01:00–01:15 the plan before any data
- **The pinned 7B mirror:** `unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit` @ `2ab7fb37…`, 5.55 GB. It was downloaded into
  the repo-local HF cache in 6 min. A download is not data.
- **D151** (the plan and every analysis rule for the three items) was pushed as `a8938c7` before anything ran.

### ~01:15–01:35 item 1 and item 2 (CPU, 1 thread, nice 19)
- **Item 1, X2 adaptive sampling** (`followup_ttc x2`, ~5 min):
  - majority@8 0.382 vs one sample 0.339;
  - D75 (the stated-σ gate) is the selection-half winner. Evaluation half: matched saving +0.21 [+0.12, +0.30], at
    −3.6 points vs majority@8.
- **Item 1, MC-A two-stage spread** (`followup_ttc mc_a`): it reproduces #37's 0.638 / 0.710 / 0.745. The two-stage
  rule loses AUROC on all sets.
- **Item 2** (`followup_arith`, ~6 min):
  - the anchors (a) and (c) reproduce #35 to 4 decimals;
  - E − (a) = −0.0016 [−0.0034, −0.0003];
  - E_chain ≈ (a).

### ~01:20–01:40 item 3 on the GPU
- **First feasibility, X2's decoding (T = 1):** dev 1188 took 140 s, with 7 of 8 samples at the 1,024 cap and a sample
  that switched to Chinese mid-solution. Stopped.
  - **A false alarm on the way:** display math looked missing from the decoded text. It was my own `grep -v '^ *\\'`
    filter on the debug output; the raw ids decode correctly.
- **D152:** the decoding becomes Qwen's documented sampling (T 0.7, top_p 0.8, top_k 20) for the 7B and the control.
- **Second feasibility:**
  - dev 1188: 139 s (one sample at the cap);
  - dev 1189: 155 s (7 of 8 at the cap);
  - a 2-sample check shows legitimate long derivations, still in phase 3 of 4 at 1,024 tokens.
- **D153:** D151's 120 s / problem rule fires, so there is no main stage. A development-only run at 2,048 tokens
  (12 problems) and the native 1.5B on the same problems size a future run.
- **Memory** fits under the 6.7 GiB cap. The rate is ~6.6 decoding steps / s at batch 8 (bitsandbytes dequantizes on
  every step).
