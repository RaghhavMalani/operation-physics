# Report: 2026-10-18 night: corrected evals (D67 / D68), final 02:30 IST

The previous report (2026-10-16/17, with the 2026-10-17 evening update) is in git history: `git show cb75d82:MORNING.md`.

**Short version.**
1. **Corrected evals are the primary results now (D67).** Every RL adapter was evaluated unmerged on its 4-bit
   training base, with the SFT-only and uniform controls on the same base. The pre-registered analyses were then
   re-run (FINDINGS #16; side by side in `results/corrected/side_by_side.md`).
2. **Family S on the seed mean:**
   - **S1a, S1b and S3 are supported.** The coverage effects are about 3× the superseded ones: 1σ coverage is 41.8%
     vs 12.0% (SFT) and 10.5% (uniform).
   - **S2 is NOT supported:** p = 0.262, per seed 0.21 / 0.40 / 0.23. The superseded "narrowly supported" is
     withdrawn; it was an artifact of the merge.
3. **Family N fails on every seed** (all nine tests). **Family E is unchanged:** P supported, N not supported. On the
   policy as trained, σ widens a lot but still does not track the error (ρ ≈ 0.03–0.08 vs ≈ 0.67 for A5).
4. **Never-merge is the default, with a guard (D68).** The D66 resume path also passed its GPU check.
5. **Every stage ran,** 00:17–02:20, well inside the 08:30 cutoff. Every stage was committed and pushed. BLOCKED: none.

---

## 1. What ran (IST)

| time | stage | result | commit |
|---|---|---|---|
| 00:04 | **D67** recorded (before any corrected S/N/E result); D64 mapping confirmed | – | `bb7ff3c` |
| 00:15 | **D68**: never merge RL adapters (default + guard + tests; 380 tests pass) | – | `46eca0f` |
| 00:17 | queue + resume-check script (pass criteria fixed before the run) | – | `1c096d6` |
| 00:17–00:27 | SFT-only on the 4-bit base: frozen sets, E1 prefixes | 0 OOM | per stage |
| 00:27–00:54 | split credit seeds 0 (rerun), 1, 2 @287 | | per stage |
| 00:54–01:11 | uniform credit @287; E1a final (rerun) + step 150; E1b final + step 150 | | per stage |
| 00:20–01:13 | analysis code (`d80ee83`), then the tier-1 analyses | §2 | `ecd1f4d` |
| 01:20 | per-seed output-name bug fixed, seeds 0–2 rerun (§6) | | `20ec565` |
| 01:13–01:31 | FINDINGS #5 and #6 adapters | §3 | per stage |
| 01:31–02:10 | sampled k = 4 reruns: split seed 0, E1a | §3 | per stage |
| 02:10–02:18 | **D66 GPU resume check** | **PASS** | per stage |
| 02:20 | final analyses + side-by-side | | `c624656` |

The highest GPU temperature I saw was 80 °C (spot checks); no stage start was above 87 °C, so the queue never paused.

## 2. Primary results (corrected; pre-registered tests)

**Family S on the seed mean** (D62; split credit seeds 0–2 @287; 400 problems):

| test | corrected: split / control | diff [95% CI] | p | **corrected verdict** | superseded verdict |
|---|---|---|---|---|---|
| S1a: 1σ cov > SFT-only | 41.8% / 12.0% | +29.8 pts [+25.5, +34.2] | < 1e-4 | **supported** | supported (+8.4 pts) |
| S1b: 1σ cov > uniform | 41.8% / 10.5% | +31.3 pts [+27.0, +35.8] | < 1e-4 | **supported** | supported (+9.9 pts) |
| S2: ρ(σ, err) > uniform | 0.061 / 0.015 | +0.046 [−0.095, +0.184] | 0.262 | **not supported** | narrowly supported (0.040) |
| S3: accuracy non-inferior | 37.7% / 37.2% | +0.4 pts [−1.0, +1.8] | < 1e-4 | **supported** | supported |

- **Per-seed S2 p:** 0.207 / 0.396 / 0.225 (superseded 0.061 / 0.020 / 0.092).
- **Per-seed S1a / S1b / S3:** p < 1e-31 for S1a and S1b on every seed; S3 holds on every seed.

**Family N** (split credit's σ vs the A5 σ on the *same corrected answers*), per seed:

| seed | N1: \|cov − 68.3%\|, RL / A5 | N2: ρ, RL / A5 | N3: crps_log, RL / A5 |
|---|---|---|---|
| 0 | 0.265 / 0.005 | 0.077 / 0.676 | −0.316 / −0.289 |
| 1 | 0.270 / 0.002 | 0.034 / 0.674 | −0.317 / −0.290 |
| 2 | 0.258 / 0.003 | 0.071 / 0.644 | −0.307 / −0.283 |

All nine tests fail (p = 1). The superseded results also failed all nine.

**Family E** (μ fixed; SFT control = the SFT model's own σ suffix on the same base):

| test | corrected Δcrps_log [95% CI] | verdict | superseded |
|---|---|---|---|
| E1a-P | +0.0077 [+0.0062, +0.0092] | supported | +0.0024, supported |
| E1b-P | +0.0075 [+0.0061, +0.0090] | supported | +0.0020, supported |
| E1a-N | −0.0258 [−0.0342, −0.0181] | not supported | −0.0310, not supported |
| E1b-N | −0.0259 [−0.0345, −0.0181] | not supported | −0.0314, not supported |

**Headline, pooled, corrected (greedy):**

| arm | acc ≤ 1% | 1σ / 2σ cov | σ = 0 | median rel. σ | ρ(σ, err) | crps_log |
|---|---|---|---|---|---|---|
| SFT-only (4-bit) | 36.8% | 12.0 / 17.0% | 16.8% | 0.063% | 0.05 | −0.333 |
| SFT + post-hoc ×18.1 (standing baseline, not re-fitted) | 36.8% | 34.2 / 42.2% | 16.8% | 1.15% | 0.05 | −0.329 |
| SFT + A5 σ | 36.8% | 67.8 / 78.2% | 0.0% | 6.2% | 0.68 | −0.302 |
| uniform credit @287 | 37.2% | 10.5 / 15.0% | 19.5% | 0.052% | 0.01 | −0.324 |
| split credit @287 (seeds 0 / 1 / 2) | 37.8 / 36.8 / 38.5% | ≈ 42 / ≈ 51% | 0.0–0.3% | ≈ 1.6% | 0.08 / 0.03 / 0.07 | −0.316 / −0.317 / −0.307 |
| E1a final (μ fixed) | – | 48.5% / – | 0.0% | 3.2% | 0.01 | −0.331 |

- **Reading:**
  - Evaluated as trained, split credit roughly quadruples coverage over the controls.
  - But its σ ranks problems no better than uniform credit's, and a text-feature σ beats it on every test.
  - With μ fixed (E1), a σ can be learned, but it gets about 23% of the way from SFT to A5 in crps_log.
- **Accuracy:** the 4-bit SFT model is about 2 points more accurate than the bf16 merge (36.8% vs 34.5%). So most
  of the "as-trained accuracy" noted in FINDINGS #15 is the base, not RL.

## 3. Descriptive (tier 2)

- **Determinism:** all four reruns of D65's tapes are byte-identical, greedy and sampled (400 / 400 and
  1,600 / 1,600 completions; rewards identical).
- **σ instability (the within-problem SD of ln σ, k = 4):** split credit 0.85 (merged 1.01); E1a 0.35 (merged 0.94).
  Most of the "factor-of-e" instability reported yesterday came from the merge.
- **FINDINGS #6 adapter (uniform credit, step 759), as trained:** σ = 0 on **97.5%** of answers (merged: 43.2%),
  with 1σ coverage 2.8%. It is the collapse seen in training, at full strength.
- **FINDINGS #5 adapter, as trained:** SFT-like σ (median 0.057%, 1σ cov 12.8%), accuracy 39.0%.
- **E1 at step 150, as trained:** most of the final effect is already there (E1a σ 2.7%, coverage 46.5%).
- **D66 resume on the GPU** (Unsloth, 4-bit PEFT): **PASS**.
  - **Setup:** run B was killed during step 3 and resumed from checkpoint-2.
  - **Logs:** all four steps complete, and the tape replays.
  - **Curriculum:** the cells for steps 1–4 are identical to the uninterrupted run.
  - **Adapter:** reloaded (‖B4 − A4‖ = 0.0033 vs ‖B4 − A2‖ = 0.0089). 445 s in all.

## 4. Code changes tonight

- **D68:**
  - **`physcal/adapters.py`:** an adapter is treated as RL if its run's training meta has a GRPO config or its model
    card names GRPO. RL adapters are refused for merging, always. Any other adapter is merged only on
    `merge_adapter: true`.
  - **Where the guard runs:** in the eval backend, before anything is loaded, and in `train.merge`.
  - **Defaults:** `compare` (sft/rl parts) and `e1 eval` default to the 4-bit training loader.
  - **New:** `e1 eval --sft` for family E's SFT control.
- **D67 analyses:**
  - **`--corrected` modes:** in `phase95_compare.py`, `phase95_seed_mean.py` and `e1_compare.compare`. Only input and
    output paths change. Rerunning the superseded mode reproduces every existing number (checked, then restored).
  - **New scripts:** `scripts/corrected_vs_superseded.py`, `scripts/corrected_analyses.sh`.
- **The D66 GPU check:** `configs/grpo_resume_smoke.yaml` (a test pins it to grpo_split_credit apart from
  length and checkpointing) and `scripts/resume_gpu_check.py`.
- **Tests:** 383 pass.

## 5. Decisions to review

1. **D67's primary-tape rule:** D65's tapes are primary for split seed 0 and E1a final. This turned out not to
   matter, because the reruns are byte-identical.
2. **The family E SFT control:** it is now the SFT model's own σ suffix generated on the 4-bit base (`e1 eval --sft`).
   Before, it was the σ in the original bf16 SFT completions. They are nearly the same: crps_log −0.3386 in both.
3. **The standing post-hoc baseline** (×18.1) was applied to the 4-bit SFT answers without re-fitting. This is
   reported only, not tested. Re-fitting it on 4-bit SFT calibration completions would take one GPU eval (~10 min).
4. **The superseded FINDINGS (#3, #5, #6, #13, #14)** carry a pointer to #15 / #16. #13's headline (S2 narrowly
   supported) is withdrawn by #16. Tell me if you want #13's heading itself edited as well.

## 6. BLOCKED / housekeeping

- **BLOCKED:** none.
- **A bug in my own analysis code, found at 01:15 and fixed (`20ec565`).** The corrected per-seed S/N outputs were
  named after the run with `Path.with_suffix`. The run names contain dots (`v1.1.0`), so all three seeds wrote one
  file.
  - **Effect:** the tier-1 side-by-side showed the per-seed rows as missing. No number was wrong, and the seed-mean
    and family E files were not affected.
  - **Fix:** fixed, seeds 0–2 rerun, regression test added.
- `PHASES_10-15.md` is still untracked (its content is in CLAUDE.md).

## 7. Recommendation

- **Use the corrected results from here on.** The story is now cleaner: split credit cures σ collapse and widens σ
  about 25× (≈ 1.6% vs 0.06%), but at this scale RL does not make σ track the error. The surface-feature σ wins
  every N test, and S2 fails.
- **Credit assignment is not the bottleneck.** Both fixes (split credit, fixed-μ E1) give a wider but uninformative
  σ. The next informative experiment is Phase 11/12 (noise mode): there σ\* is known per problem, so A6/A6b can show
  directly whether any σ source propagates input noise.
- **Before Phase 12:** decide whether RL training should stay on the 4-bit base. All evals now use it, so train and
  eval agree. The alternative is a bf16 base with unmerged eval. Either is consistent now; mixing them is not.
