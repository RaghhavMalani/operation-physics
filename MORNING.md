# Report: 2026-10-22/23 night: the stage C pilot, Phase 12 stage D, and this morning's disk-full recovery

The previous report (2026-10-21/22 night) is in git history: `git show 54644f2:MORNING.md`.

**Short version.**
1. **The GPU ran from 23:28 to 07:12 IST** (deadline 09:00). Every stage exited 0 and nothing is BLOCKED. There was
   one 10-min temperature pause (00:05, 88 °C, GPU load alone).
2. **Stage C pilot (FINDINGS #28):**
   - **Tests:** C6 is supported (+0.045 over its SFT control); C1–C5 are not.
   - **What RL learned:** to state the smallest given percentage (1, 2 or 5%), σ ≈ 0.44·σ\*, which scores below RSS
     and X1.
   - **The policy turned deterministic per problem by step ~40** (98–100% zero-variance groups); steps 51–300
     changed nothing.
3. **Stage D (FINDINGS #29):**
   - **D-pos, the line's numbers trained:** it does not propagate (D1 / D2 not supported) and is below RSS and X1
     (D3 / D4 not supported).
   - **Two tests pass, both weakly:** D6, +0.001 over stage B; D7, +0.155 over D-fmt.
   - **The line locates the failure:**
     - the copy and the arithmetic are roughly right;
     - the sensitivities are exponent templates (77% exact on power-law terms, 5% on value-dependent ones);
     - it never reads per-given levels.
   - **D-fmt, numbers masked:** degenerate lines; "± 0" on 69% of answers.
4. **Across stages B, C and D, no trained σ beats X1 or the RSS heuristic** on the oracle-centred expected score
   (§4).
5. **A design limit found tonight:** none of the 2,250 noisy Phase 12 SFT problems mixes noise levels. So A6c (B2 /
   C2 / D2 / F2) tests transfer to an unseen level mix.
6. **This morning: the MLStorage volume had filled.**
   - **Integrity:** all 167 files from last night pass every check; nothing is damaged.
   - **Added:** a free-space guard in all 14 queue scripts (BLOCKED below 15 GiB), and stage D's ignore patterns.
   - **Space:** about 6.5 GB can be reclaimed safely, and up to ~16.4 GB more on your call (§5). Nothing was
     deleted.
7. **Please resend steps 3–6 of this morning's prompt.** That prompt is in no transcript on disk (§6.1).
8. **Decisions:** D108–D111. **Tests:** 584 passed, 0 failed, 0 skipped (GPU visible).

---

## 1. What ran (IST)

| time | step | result | commit |
|---|---|---|---|
| 23:28 | stage C pilot launched (`queue_stage_c_pilot.sh`, PID 6944) | OOM check 2 × 1,644 tokens, 2.0 GiB | – |
| 23:30–23:43 | CPU: D108 (your decisions); D109 (stage D) pre-registered; stage D built, 10 CPU tests | – | `ff59b4a` |
| 23:44 | stage D chain launched (`queue_stage_d.sh`, PID 18157), waiting on stage C's PID | – | – |
| 23:28–02:12 | GPU: stage C training, 300 / 300 steps (32 s / step; paused 10 min at 00:05, 88 °C) | 7.04 GiB peak, no OOM | (queue) |
| 02:12–02:51 | GPU: stage C σ suffixes on 3,060 prefixes: final (14 min), SFT control (12), step 150 (13) | 3,060 / 3,060 each | (queue) |
| 02:51 | CPU: stage C analysis | FINDINGS #28 | `e31997d`, `a9f145d` |
| 02:52–03:59 | GPU: D-pos SFT, 750 / 750 steps (67 min) | loss 0.125 | `06f710f` |
| 03:59–04:54 | GPU: D-pos eval (55 min) | 3,060 / 3,060 | `8203747` |
| 04:54–06:04 | GPU: D-fmt SFT, 750 / 750 steps (69 min) | loss 0.097 | `3320d85` |
| 06:04–07:11 | GPU: D-fmt eval (67 min) | 3,060 / 3,060 | `32b2eef` |
| 07:12 | CPU: stage D analysis | FINDINGS #29 | `f7caa40` |
| ~07:14 | FINDINGS #29, D110, PROGRESS written (not yet committed); the full test suite started; then the session ended (disk full) | – | – |
| 10:37 | your message: integrity checks, ignore patterns, free-space guard, space list (D111) | nothing damaged | (this commit) |

## 2. Stage C pilot (D106; FINDINGS #28)

| test | estimate [95% CI] | Holm |
|---|---|---|
| C1: A6 slope > 0 | −0.000 [−0.000, +0.000] | not supported |
| C2: A6c slope > 0 | +0.086 [−0.158, +0.324] | not supported |
| C3 / C4: expected crps_log − RSS / − X1 > 0 | −0.003 / −0.004 | not supported |
| C5: A6 slope − X1's > 0 | −0.063 [−0.103, −0.025] | not supported |
| **C6: RL − SFT-only control > 0** | **+0.045 [+0.029, +0.062]** | **supported** |

- **D106's signal rule fires on C6** ("3 seeds warranted"). But the control is the merged SFT model with prompt v5,
  which says "± 0" on 79% of these noisy problems. Most of C6 is RL removing that.
- **What the policy does (exploratory):** σ takes three values (1 / 2 / 5%). It equals the smallest stated level on
  93% of problems. On A6 / A6b, where every given shares one level, that makes the slope exactly 0. Coverage is
  36% / 59%.
- **Dynamics:**
  - **Steps 1–50:** reward −0.149 → −0.045, and σ = 0 went from 24% to 0%.
  - **From step ~40:** 98–100% of groups had zero reward variance, so TRL logged loss 0 and grad-norm 0 on most steps.
    E1d, with the realized-score reward, never went above 17%.
  - **[hypothesis]** score_sigma is nearly noise-free, so per-group std normalization turns tiny consistent
    differences into ±1 advantages and the policy sharpens fast.
- **D107.6 did nothing:** this TRL has no `max_prompt_length` (logged as a dropped key), so no prompt was cut either
  way.

## 3. Stage D (D109; FINDINGS #29)

| test | D-pos [95% CI] | Holm | D-fmt [95% CI] | Holm |
|---|---|---|---|---|
| 1: A6 slope > 0 | −0.18 [−1.13, +0.39] (50 pairs) | no | +2.04 [−2.05, +7.98] (55 pairs) | no |
| 2: A6c slope > 0 | −0.03 [−0.30, +0.23] | no | −0.26 [−1.77, +1.26] | no |
| 3: − RSS > 0 | −0.001 | no | −0.156 | no |
| 4: − X1 > 0 | −0.002 | no | −0.157 | no |
| 5: A6 slope − X1's | −0.30 | no | +1.89 | no |
| 6: − stage B > 0 | **+0.001 [+0.000, +0.002]** | **yes** | −0.154 | no |
| D7: D-pos − D-fmt > 0 | **+0.155 [+0.124, +0.186]** | **yes** | – | – |

| the line (vs the builder's own line for the same problem) | D-pos | D-fmt |
|---|---|---|
| present / parsed | 100% / 99.9% | 100% / 7.1% |
| levels right, in order | 67% (A6 / A6b 100%, A6c 1%) | 0% |
| every sensitivity exact (per term: power-law 77%, value-dependent 5%) | 31% | 6% |
| copy: σ = X·\|μ\| at 2 s.f. (within 6%) | 82% (97%) | 80% of 212 |

- **What D-pos learned:**
  - **The format, the copy step (R3-9) and approximate arithmetic:** yes.
  - **Value-dependent sensitivities** (e.g. u / v in a sum): no.
  - **Per-given levels:** no. It repeats one level on every term.
- **The consequence:** σ under-propagates with depth (median ln(σ / σ\*) −0.02 at depth 1 to −0.29 at depth 5).
  Stage B's error was a constant scale.
- **D-fmt:** masked numbers get no training signal. The model writes e.g.
  `sqrt((0*sqrt(1))**2 + (1*sqrt(1))**2) = 1` and answers "± 0". D84's format-only arm is a poor starting point for
  explicit-arm RL.
- **D109's seed rule fires on D6** (+0.001, ~20% of stage B's gap to σ\*). See §6.

## 4. Every Phase 12 σ source so far (oracle-centred, the same 3,060 problems)

| source | expected crps_log | median ln(σ / σ\*) | 1σ coverage | A6c slope |
|---|---|---|---|---|
| σ̂ = σ\* (ideal) | −0.0365 | 0 | 68% | +1.00 |
| **X1 primary (text model)** | **−0.0386** | −0.03 | 64% | −0.00 |
| **RSS of the stated levels** | **−0.0394** | −0.08 | 62% | 0 |
| D-pos (stage D, explicit, numbers trained) | −0.0405 | −0.11 | 54% | −0.03 |
| stage B (supervised σ\*, 3-seed mean) | −0.0415 | −0.31 to −0.45 | ~50% | +0.04 |
| stage C (σ-only RL, predictive reward) | −0.0427 | −0.81 | 36% | +0.09 |
| stage C's SFT-only control | −0.0874 | −10.5 (σ = 0 on 79%) | 9% | −1.03 |
| D-fmt (stage D, format only) | −0.1951 | −10.2 (σ = 0 on 69%) | 18% | −0.26 |

## 5. This morning: the disk-full crash (D111)

- **Cause (your account):** the MLStorage volume (`D:\MLStorage`) filled (STATUS_DISK_FULL). It holds the Ubuntu
  disk image `wsl_ubuntu\ext4.vhdx` (53.6 GB; 51.1 GB used inside). It has been extended (now 53 GiB free of
  104 GiB) and e2fsck re-run.
- **Integrity checks** (`scripts/integrity_check.py`, read-only) on the 167 files written since 23:20 IST last
  night, plus the memory notes. **Nothing is damaged:**
  - **`git fsck --full --strict`:** exit 0. The one dangling commit is a 2026-10-19 stash.
  - **Text / logs:** no NUL bytes, no truncated last line.
  - **JSONL:** 34,683 lines in 14 files, all parse.
  - **gzip (6):** CRCs pass, byte-identical to their uncompressed copies.
  - **safetensors (6 adapters and checkpoints):** header and offsets consistent; no all-zero tensor, NaN / Inf or
    all-zero 4 KiB block.
  - **Torch archives (9):** zip CRCs pass. **PNGs (8):** every chunk CRC matches.
  - **Tracked files:** the 77 that git calls unchanged hash to their HEAD blobs.
  - **Not checked:** `/lost+found` (needs root): `sudo ls -la /lost+found`.
- **`.gitignore`:** stage D now matches stage B. The large adapter files, `checkpoints/`, the uncompressed tapes and
  PID files are ignored. Configs, fingerprints, `.gz` tapes, summaries and plots are committed.
- **The free-space guard** (`scripts/disk_guard.sh`, sourced by all 14 `scripts/queue_*.sh`):
  - **The rule:** before every stage, it reads MLStorage's free space via `powershell.exe` (~1 s). Below 15 GiB it
    logs `BLOCKED <stage>: N GiB free on MLStorage < 15 GiB; queue stopped`; the queue exits 75 without touching
    running children.
  - **[REVIEW] it fails closed:** an unreadable volume (3 tries) is also BLOCKED.
  - **Overrides:** `DRY_RUN=1` only logs the reading; `DISK_GUARD=off` bypasses it (logged).
  - **Tests:** 20, including the real stage D queue stopping before its first stage when 1 GiB is reported.
- **What can be deleted** (nothing was deleted; sizes are of untracked files only, committed files stay; rows do not
  overlap):

| # | what | size | why it is safe | what you would lose |
|---|---|---|---|---|
| 1 | trainer `checkpoints/` (resume state) in 20 finished runs: `results/train/*`, `results/sft/qwen2.5-math-1.5b-instruct`, `results/stage_c/run` | 3.96 GB | final adapters and `permanent_checkpoints/` stay; no config or code reads these | resuming or extending those runs from their last save |
| 2 | 7 smoke runs: `resume-smoke-A/B`, `smoke-split-credit`, `smoke-overnight2-mb2`, `sft-continue-smoke`, `phase4-smoke-qwen3.5-2b`, `results/pinned_base_smoke` | 1.53 GB | finished machinery checks; their 44 committed files stay | nothing a rerun of the smoke test would not recreate |
| 3 | `results/discarded/` (the 2026-10-16 crashed split-credit seed 1) | 0.39 GB | discarded by name | – |
| 4 | 21 uncompressed `rollouts.jsonl` whose committed `.gz` is byte-identical (stage B / C / D evals, stage C, E1a–E1d, overnight2, split-credit training tapes, `e1d_x2`) | 0.66 GB | `gunzip -k` restores them exactly | `e1d_x2_analysis`, `posthoc_cond`, `e1.py`'s `read_meta` and tape replay read the uncompressed file: gunzip first |
| 5 | `unsloth_compiled_cache/`, `.hypothesis/`, `.pytest_cache/` | < 3 MB | regenerated | – |
| | **safe subtotal** | **≈ 6.5 GB** | | |
| 6 | Qwen3.5-2B: its HF cache (4.57 GB) + `results/sft/qwen3.5-2b` and the phase-4 run (4.08 GB) | 8.65 GB | re-downloadable; not used since Phase 4 | it is still named in `grpo_phase4` / `model_probe` / `sft` configs and one test |
| 7 | `results/x2/random_init_model` | 3.10 GB | seeded (seed 0), regenerable | X2's and X4's random-init controls load it; regeneration may not be bit-identical |
| 8 | `results/x2/hidden` (cached hidden states) | 1.09 GB | GPU re-extraction | X2 probe refits until re-extracted |
| 9 | `results/probe_sft` (probe caches) | 0.43 GB | GPU re-extraction | Phase 9.5 probe refits |
| 10 | `Qwen/Qwen2.5-1.5B-Instruct` HF cache | 3.10 GB | re-downloadable | CLAUDE.md's named model, in 8 configs |

  - **Not worth it:** `.cache/uv` looks like 6.7 GB, but only 0.05 GB of it is not hard-linked into `.venv`.
  - **Keep:**
    - the merged SFT model (3.1 GB: stage C's, E1d's, X2 / X4's base);
    - the pinned 4-bit base (1.15 GB);
    - the Qwen2.5-Math base (3.1 GB, for merges);
    - every final adapter and permanent checkpoint, `data/` and `.git`.
  - **Deleting inside WSL does not give the space back to MLStorage.** The freed blocks are reused, so the image stops
    growing, but `ext4.vhdx` does not shrink. To shrink it:
    - **Compact:** `wsl --shutdown`, then in an admin PowerShell
      `Optimize-VHD -Path D:\MLStorage\wsl_ubuntu\ext4.vhdx -Mode Full` (or diskpart `compact vdisk`).
    - **Or make it sparse:** `wsl --manage Ubuntu --set-sparse true`.

## 6. Decisions to review

1. **Steps 3–6 of this morning's prompt.**
   - **Where they are not:** this session's transcript has no entries between 07:14 and 10:37 IST, and no other
     local transcript contains a morning prompt. It was probably never written because the disk was full.
   - **So far:** only the steps in your 10:37 message were done. Please resend 3–6.
   - **Still open from your D108 approvals, if they were among them:** the 1.3.0 dispatch edit in `reward.reward` /
     `tape.replay`, approved for today.
2. **Two pre-registered 3-seed rules fire on effects I read as too weak.**
   - **Stage C (C6):** against a control that says "± 0".
   - **Stage D (D6):** +0.001, still below RSS.
   - **My recommendation:** no 3-seed replications of either configuration as they are. Each failure has a specific
     cause that more seeds would not change.
3. **Mixed-level SFT data** (a generator change: mixed per-given levels are representable but never generated).
   Without it, A6c cannot separate "no propagation" from "never saw a level mix".
4. **D109 [REVIEW] items:**
   - stage B's base and budget for stage D;
   - two Holm families;
   - the 10% lenient sensitivity match;
   - my reading of R3-4 D's seed rule.
5. **D111 [REVIEW]:** the free-space guard fails closed.
6. **D110.6:** stage C's 300 steps were kept after the policy froze at step ~40 (~2.2 GPU-h with no learning). The
   alternative was a mid-run protocol change.

## 7. BLOCKED / problems / tests

- **Nothing is BLOCKED.** One temperature pause (stage C, 00:05). No OOM, no spill; stage C peaked at 7.04 GiB
  reserved, stage D at 4.2 GiB device-wide.
- **The disk-full crash** happened after the GPU queue ended (07:12). No GPU stage was affected.
  - **The only work interrupted:** last night's uncommitted write-up, which is intact, and the full test run.
- **Tests:** 584 passed, 0 failed, 0 skipped (GPU visible, 10:44 IST): the full suite (583) plus the integrity
  checker's test, run on its own. 21 are new today: 20 for the guard, 1 for the checker.

## 8. Recommendation

- **Phase 12's σ line has now failed three ways on this base,** each now with a specific cause:
  - supervised imitation (B): a constant-scale surface reader;
  - σ-only RL (C): copies the smallest percentage, then stops exploring;
  - an explicit trained line (D): exponent templates, no per-given levels.
- **Before any more GPU nights, one change per arm, each with its own entry:**
  1. **Mixed-level SFT data** (generator change, your approval). It fixes A6c's distribution shift for every arm.
  2. **For explicit propagation, a chain-rule scratchpad:** each sensitivity computed from the solution's own
     intermediate values, with symbol names (D80's open [REVIEW]). D-pos shows the copy works and the derivatives
     don't.
  3. **For σ-RL, one exploration change** (a KL term, or batch-level reward scaling) against stage C's
     deterministic collapse.
- **Today (CPU), once you confirm steps 3–6:** the 1.3.0 dispatch edit, and the space clean-up you choose from §5.
