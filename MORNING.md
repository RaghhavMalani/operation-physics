# Report: 2026-10-19/20 night: X2 (all 1,200 candidates) and the X3 pilot, after a late start (23:06–06:08 IST)

The previous report (2026-10-19 day + afternoon addendum) is in git history: `git show 971ffc4:MORNING.md`.

**Short version.**
1. **Everything ran.**
   - **Timing:** GPU work ended at 05:54, inside your 09:00 deadline. Every stage was committed and pushed.
   - **Two stages failed first time** (logged BLOCKED by the queue, as designed). Both were fixed and rerun after the
     queue (D87, D88), so nothing is missing.
2. **X2 (FINDINGS #19): within a problem, no stated σ and no text model tells right answers from wrong.**
   - **Stated σ:** SFT 0.499; split credit 0.491.
   - **A5's text p:** 0.497.
   - **The hidden-state probe:** 0.547 (Holm supported), but a random-init model's probe reaches 0.530, and the
     difference is not significant.
   - **D74's label:** "another combination: reported as observed". No positive epistemic claim follows.
3. **X3 pilot (FINDINGS #20): "capacity likely".**
   - **Student (a):** the RL runs' own base + LoRA, SFT'd on the A5 teacher σ, reaches ρ = 0.575 [0.499, 0.643].
     Split credit's is 0.06. It is also calibrated (66% 1σ coverage).
   - **What it learned:** which problems are hard, not which answers are wrong. On X2's pairs it is at chance
     (0.513).
4. **D78's pinned-base GPU check: PASS.** **R2-1, now enforced:** both X3 students pass their reload checks.
5. **Launch:** your pre-launch checks were clean.
   - **One launch-time change (D86):** a host process held 1.65 GB of VRAM, so X2's allocator cap went 7.5 → 5.75 GiB.

---

## 1. What ran (IST)

| time | stage | result | commit |
|---|---|---|---|
| 23:04 | launch checks (after the WSL restart) | clean: git synced, no lock, inputs' hashes match | `efbbce4` (D86) |
| 23:06–23:07 | X2 batch probe (cap 5.75 GiB) | 128 / 96 / 64 / 48 OOM; **32 fits** (4.6 GiB, 35 s per worst-case batch) | queue |
| 23:07–01:11 | X2 sampling, 6 chunks × 200 (about 20 min each) | 9,600 completions, 99.6% parsed, 0 backoffs | queue (×6) |
| 01:11 | selection | **516** mixed problems, 5,390 pairs | queue |
| 01:11–02:18 | X2 method 2, split credit σ suffixes | **BLOCKED: timeout** (WSL spill, D87) | queue |
| 02:18–02:58 | hidden states, SFT and random init × fit and X2 | all 4 OK, Unsloth path (logit gap ≤ 0.5%) | queue |
| 02:58–03:18 | X2 analysis (CPU, without method 2) | superseded by 06:08 | queue |
| 02:58–03:45 | X3 student (a), 500 steps | exit 0, R2-1 fingerprint | queue |
| 03:45–03:49 | (a) on the frozen sets | exit 0, R2-1 **pass** | queue |
| 03:49–04:09 | (a) on the X2 answers | **BLOCKED: crash on a μ = 0 answer** (D88) | queue |
| 04:09–04:54 | X3 student (b), 500 steps | exit 0 | queue |
| 04:54–05:17 | (b) evals, the X3 analysis (without (a)'s X2) | exit 0 (the analysis was superseded) | queue |
| 05:17–05:18 | **D78 pinned-base GPU check** | **PASS** | queue |
| 05:18–05:37 | rerun: (a) on the X2 answers | exit 0 | reruns |
| 05:37–05:54 | rerun: X2 method 2 (batch 16, capped) | exit 0, 17 min | reruns |
| 05:37–06:08 | reruns: X3 analysis, X2 analysis (CPU) | exit 0 | reruns |

- **Temperatures:** stage starts were 55–67 °C; the peak seen was 85 °C, during X3 training. Never above 87 °C, so no
  pause.
- **The reruns** (`scripts/queue_x2_x3_reruns.sh`, same guards) were chained to start when the queue's PID exited.

## 2. Results

**X2 (D74; `results/x2/summary.md`; FINDINGS #19):**

| method | pairwise accuracy [95% CI] | Holm |
|---|---|---|
| 1. stated σ, SFT | 0.499 [0.474, 0.524] | no |
| 2. stated σ, split credit seed 0 @287 | 0.491 [0.461, 0.521] | no |
| 3. A5 p (4-bit) | 0.497 [0.481, 0.514] (ties on 64% of pairs) | no |
| 4. problem-text-only (check) | 0.500 exactly | – |
| 5. hidden-state probe, SFT | **0.547 [0.520, 0.574]** | **yes** |
| 6. hidden-state probe, random init | 0.530 [0.503, 0.557] | no (p 0.013 vs threshold 0.0125) |

- **Probe vs the other methods:**
  - probe − random init: +0.017 [−0.015, +0.050], so it is **not** "probe succeeds" under D74;
  - probe − SFT σ: +0.048 [+0.011, +0.086].
- **What the probe reads:** the cross-fitted probe picks layer 1 in four of five folds, which suggests surface features
  of the answer tokens.
- **Reading:** A5's between-problem advantage (ρ ≈ 0.65) is difficulty prediction. Neither the model's σ nor the text
  model knows which of its answers is right.

**X3 pilot (D79; `results/x3/pilot.md`; FINDINGS #20)**, on the 399 frozen-set answers with μ fixed:

| σ source | ρ(σ, err) [95% CI] | crps_log | 1σ cov | median σ | X2 pairwise |
|---|---|---|---|---|---|
| **student (a), A5 teacher** | **0.575 [0.499, 0.643]** | −0.306 | 66.4% | 10.8% | 0.513 [0.484, 0.543] |
| student (b), true-error labels | 0.583 [0.509, 0.650] | −0.348 | 30.8% | 0.3% | 0.488 [0.460, 0.517] |
| A5 teacher | 0.664 | −0.291 | 66.4% | 6.6% | – |
| SFT stated σ | 0.038 | −0.322 | 12.0% | 0.1% | – |

- **Label:** "capacity likely", i.e. a point estimate ≥ 0.5 with a CI lower end of 0.499. That is the pre-registered
  rule applied as written.
- **Fidelity to the teacher:** 0.72.
- **Student (b):** ranks as well as (a) but is badly overconfident. Greedy decoding takes the mode of its bimodal
  targets.

**Other checks:**
- **D78 (`results/pinned_base_smoke/report.json`): PASS on all five checks.** The snapshot loads 4-bit, the adapter
  names the snapshot dir, the reload gets the same base, and the provenance hash is `sha256:9939448b3a0fbe45`.
- **R2-1, enforced:** both students pass. Log-prob ratios are 0.030 / 0.027 and σ-token ratios 0.023 / 0.039, against
  a 0.25 limit. Split credit seed 0 (no fingerprint) got the "no_fingerprint" warning, as D77 specifies.

## 3. Decisions made tonight (please review)

1. **D86 (at launch):** X2's sampling cap 7.5 → 5.75 GiB, because a host-side process held 1.65 GB of VRAM. Deadline
   09:00 via `GPU_DEADLINE` (your message). The queue script was unchanged.
2. **D87: the method-2 stage spilled into WSL shared memory.**
   - **Why:** batch 32 × ~1,300-token prompts with no cap; about 60 s per batch.
   - **The fix:** σ-suffix evals now run at batch 16 under the 5.75 GiB cap, with Unsloth imported first and progress
     lines. The X3 evals inherited it (399 suffixes in about 2 min).
   - **The rerun:** after the queue.
3. **D88: a ZeroDivisionError on the one X2 answer with μ = 0.**
   - **The fix:** a direct comparison in the μ-changed check. μ = 0 has no relative σ, so it is a failed suffix, i.e.
     a tie.
   - **Effect elsewhere:** no existing E1 row is affected.
   - **The rerun:** student (a)'s X2 eval.
4. **Superseded outputs, committed and labelled:** the queue's first X2 analysis (method 2 as p = 1) and first X3
   analysis (without (a)'s X2 numbers). Holm's X2 verdicts were the same in both.

## 4. BLOCKED / housekeeping

- **BLOCKED and then fixed and rerun:** `x2_suffix` (D87) and `x3_eval_a_x2` (D88). **Nothing remains BLOCKED.**
- **Not committed (size):** the hidden-state arrays (`results/x2/hidden/**/*.npy`, about 1.4 GB), the X3 adapters and
  the random-init weights. Their hashes and info files are committed.
- **Tests:** 456 passed yesterday. Tonight's code changes (D87, D88) were covered by the targeted E1 / X3 / X2 tests
  (25–28 pass); the full suite was not rerun overnight.
- `PHASES_10-15.md` is still untracked.

## 5. Recommendation

- **For the paper (R2-5):** X2 and X3 together give a clean negative-plus-mechanism result.
  - **The model can be taught a calibrated, error-ranking σ** (X3: ρ 0.58 from the same base + LoRA). Split credit's
    failure to reach it is an optimization / credit problem, not capacity.
  - **But nothing in this setup carries answer-specific self-knowledge:** not stated σ, not text features, and not a
    hidden-state readout beyond what a random-init network gets (X2).
  - **This supports the R2-3 framing** (one σ asked to carry epistemic uncertainty fails), and it argues for judging
    the c channel against the X2 bar, not pooled ρ.
- **For Phase 12 (Round 3):** keep D83's τ question in view. At τ = 0.1% only 14.5% of answers count as
  computation-correct. X2 shows within-problem correctness is hard to read at all, so a c channel scored on a
  precision-heavy event would be harder still.
- **Before more GPU work:** a full X3 (3 seeds), if wanted, needs its own specification (D79). And the full test suite
  should be rerun once by day.
