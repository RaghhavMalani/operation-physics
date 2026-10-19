# Report: 2026-10-18/19 night: uniform-credit seeds, D69 re-fits, R2-1 on the GPU, D70 smoke test (06:30 IST)

The previous report (2026-10-18 day: D69–D75, X1, X2 setup) is in git history: `git show 8a01afd:MORNING.md`.

**Short version.**
1. **Everything ran.** The queue started at 22:50 and finished at 06:26, two hours inside the 08:30 cutoff. Every stage
   exited 0 and was committed and pushed. **BLOCKED: none.** No temperature pause was needed.
2. **The 3-vs-3 result agrees with FINDINGS #16 (FINDINGS #18, a robustness analysis).** With uniform credit also
   averaged over 3 seeds, S1a, S1b and S3 are supported and S2 is not (p = 0.24).
   - **Seed level:** split credit's 1σ coverage beats uniform's in 9 / 9 cross-seed pairs (+31 pts).
   - **ρ:** split's ρ also beats uniform's in 9 / 9 pairs, but it is tiny (0.03–0.08 against 0.01).
3. **R2-1 worked on its first GPU use.** The corrected loader reproduced the training-time fingerprint exactly on both
   new adapters (all differences 0.0, 64 / 64 sampled completions byte-identical).
   - **Labels:** seed 2's check is "pass". Seed 1's is "inconclusive", only because that adapter's log-prob effect
     (0.0047 nats/token) sits just under the pre-set floor of 0.005. See §4.1.
4. **D69 re-fits are done.** The global scale is now ×22.1 (it was ×18.1 on bf16). A5 on the 4-bit SFT model
   reaches 66.6% 1σ coverage and ρ 0.66.
   - **Sensitivity rerun of families N and E** with the re-fitted A5: **no verdict changes.**
5. **D70 smoke test: PASS.** The SFT adapter loads trainable on the 4-bit base and `get_peft_model` takes the skip
   path. Two GRPO steps moved all 392 LoRA tensors of the *same* adapter.

---

## 1. What ran (IST)

| time | stage | result | commit |
|---|---|---|---|
| 22:47 | **D76:** queue changes at launch (your instructions), loader hardening | – | `f2d4e9b` |
| 22:50–22:58 | 4-bit SFT calibration completions, 400 | 41.5% acc, 0 parse fails | `833e850` |
| 22:58–23:25 | the same, 1,600 | exit 0 | `dd969da` |
| 23:25–23:30 | D69 re-fits, global scale + A5 (CPU, background) | exit 0 / 0 | `864ed5f` |
| 23:30–23:33 | D69 sensitivity: families N / E with the re-fitted A5 (CPU) | no changes | `749ebfb` |
| 23:25–02:43 | uniform seed 1 → step 287 (40.8 s/step, 0 OOM) + fingerprint (79 s) | exit 0 | `71d74c3` |
| 02:43–02:54 | seed 1 corrected eval + R2-1 check + sanity | PASS; check inconclusive (exact) | `378f2a2` |
| 02:54–06:11 | uniform seed 2 → step 287 (40.7 s/step, 0 OOM) + fingerprint | exit 0 | `c1fb07b` |
| 06:11–06:22 | seed 2 corrected eval + R2-1 check + sanity | PASS; check **pass** (exact) | `969ba3a` |
| 06:22 | 3-vs-3 robustness analysis (D71) | §2 | `3be48c4` |
| 06:22–06:26 | **D70 smoke test** | **PASS** | `7e53323` |

- **Seed gate:** the gate for seed 2 (≥ 3.0 h left) passed with about 5 h to spare.
- **Temperatures:** about 63 °C at every stage start and up to 82 °C during training (spot checks). Never above
  87 °C, so no pause.

## 2. Results

### 3 vs 3 (FINDINGS #18; robustness, not the pre-registered verdict)

| test (both arms as 3-seed means) | split | control | difference [95% CI] | p | supported |
|---|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 41.8% | 12.0% | +29.8 pts [+25.5, +34.2] | < 1e-4 | yes |
| S1b: 1σ cov > uniform | 41.8% | 10.8% | +31.0 pts [+26.8, +35.3] | < 1e-4 | yes |
| S2: ρ > uniform | 0.061 | 0.012 | +0.048 [−0.085, +0.178] | 0.241 | **no** |
| S3: accuracy non-inferior | 37.7% | 37.2% | +0.5 pts [−0.8, +1.8] | < 1e-4 | yes |

**Seed level** (exact permutation; one-sided p ≥ 0.05 and two-sided p ≥ 0.10 are the floors with 3 vs 3):

| metric | Δ (split − uniform) | perm. p (one / two-sided) | cross-seed pairs with the expected sign |
|---|---|---|---|
| 1σ coverage | +31.0 pts | 0.05 / 0.10 | 9 / 9 |
| ρ(σ, err) | +0.048 | 0.05 / 0.10 | 9 / 9 |
| accuracy | +0.5 pts | 0.25 / 0.50 | 6 / 9 |
| σ = 0 share | −19.8 pts | – / 0.10 | – |
| median relative σ | 1.6% vs 0.05% | – / 0.10 | – |

- **The new uniform seeds:** coverage 11.5 / 10.5%, accuracy 37.5 / 36.8%, σ = 0 on 17 / 23% of answers. They look
  just like the seed-0 control.
- **Trajectories (`results/corrected/robustness_3v3/trajectories.png`):** the split / uniform separation in σ and
  σ = 0 starts at about step 100 and is identical across seeds. Reward and pass rate overlap.
- **Reading:** the S1 result is seed-robust. S2 still fails at the problem level. The seed-level ρ ordering is
  perfectly consistent but small: +0.05, against A5's ≈ 0.65.

### D69 re-fits and sensitivity

| baseline | bf16 fit (FINDINGS #10 / #16) | 4-bit fit (tonight) |
|---|---|---|
| global post-hoc scale (D40 fit, 400) | ×18.1 | **×22.1** (all-2,000 sensitivity: ×26.8) |
| scaled 4-bit SFT eval, 1σ cov ID / depth 5 | – | 35.0% / 37.2% |
| A5 strongest σ (same model class both times) | 66.9% cov, ρ 0.70 (bf16 SFT eval) | 66.6% cov, ρ 0.66 (4-bit SFT eval) |

- **Family N (3 seeds × 3 tests) and family E (E1a-N, E1b-N):** every verdict is identical with the re-fitted A5
  (`results/corrected/d69_sensitivity/report.md`). FINDINGS #16 stands.

### R2-1 regression check, first GPU use (D72)

| adapter | its own effect: log-prob (nats/token) / σ-token TV | reloaded vs training-time | samples identical | provenance | status |
|---|---|---|---|---|---|
| uniform seed 1 @287 | 0.0047 / 0.018 | 0.0 / 0.0 | 64 / 64 | match | inconclusive (under the 0.005 floor) |
| uniform seed 2 @287 | 0.0076 / 0.051 | 0.0 / 0.0 | 64 / 64 | match | **pass** |

- **Recording cost:** 79 s at the end of training. The training runs were otherwise unchanged.

### D70 smoke test (`results/sft_continue_smoke/report.json`)

- **Loading:**
  - `FastLanguageModel.from_pretrained(results/sft/…/adapter, load_in_4bit=True)` gives a PEFT model.
  - Its 392 trainable tensors are exactly the LoRA tensors, and they equal the saved adapter (max |Δ| = 0).
  - `get_peft_model` logs "Already have LoRA adapters! We shall skip this step." and returns the same object,
    untouched.
- **Training:** 2 real GRPO steps (grpo_overnight2 config, 41.8 s/step), with a finite loss.
  - **The final adapter:** the same 392 tensors and the same base (`unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit`). All
    392 tensors changed (max |Δ| 2.7e-5 after 2 steps at LR 5e-6). The fingerprint was recorded.
  - **Reading:** RL continues the SFT adapter; it does not start a new one. The Phase 12 plan (D70) works on this
    machine.

## 3. Code changes tonight

- **D76 (before launch):**
  - **Queue changes:** seed 2 is no longer gated on seed 1's sanity check (your instruction); BLOCKED logging; the
    D70 smoke test as stage 6.
  - **`run_eval.py`:** the R2-1 check and the provenance block are non-fatal while `enforce: false`.
- **`physcal/train/e1_compare.compare`:** optional `a5_tape` / `output_dir` for the D69 sensitivity. The defaults are
  unchanged.
- **New scripts:** `scripts/d69_sensitivity.py`, `scripts/sft_continue_smoke.py`.

## 4. Decisions to review

1. **R2-1's effect floor (D72).** Seed 1's check is "inconclusive" although the reload is bit-exact.
   - **Why:** the floor (0.005 nats/token on the fixed SFT-format continuations) sits right at the size of a
     287-step RL update. Seed 1 is 0.0047, seed 2 is 0.0076.
   - **What still discriminates:** the σ-token TV effect (0.018 / 0.051) does so clearly.
   - **Options:**
     - (a) keep the floor and accept "inconclusive" for small updates;
     - (b) judge discriminability by the σ-token criterion when the log-prob effect is small;
     - (c) lower the floor.
   - **Not changed:** D72 forbids changing it without a new entry. Tell me which.
2. **Enforcing R2-1.** Both real reloads were exact, so `enforce: true` for any future loader change looks safe.
   Approve it?
3. **The standing post-hoc baseline is now ×22.1** (D69's rule: the re-fits are the standing baselines from now on).
   The FINDINGS #16 tables keep ×18.1, labelled "not re-fitted". Shall I add the re-fitted row to #16, as D69 planned?
4. **The day-report items still open:** X2's 4-GPU-hour sampling (sample a prefix, or raise the batch size); pinning
   the 4-bit base revision for Phase 12 (D70); a "Read with" pointer on FINDINGS #4 / #5 (R2-2).

## 5. BLOCKED / housekeeping

- **BLOCKED:** none.
- **Tests:** 405 collected, 404 pass, 1 skipped, as of yesterday evening. Tonight's changes are the D76 hardening
  (eval, loader and fingerprint tests rerun: pass) and the `e1_compare` option (`test_e1` rerun: pass).
- `PHASES_10-15.md` is still untracked.

## 6. Recommendation

- **Paper framing:** the corrected split-credit story is now seed-robust on both sides. Split credit cures σ collapse
  and widens σ about 30×, with no accuracy cost. But its σ ranks errors barely better than uniform credit's, and far
  worse than a text-feature σ. That supports the R2-3 framing: a single σ asked to carry epistemic uncertainty fails
  to rank errors.
- **Next GPU nights:**
  - X2 (the within-problem test of whether *any* source carries answer-specific information);
  - then the Phase 12 SFT on one 4-bit adapter, which the smoke test shows works.
