# Report: night of 2026-10-30/31, the exploratory follow-up track (D151–D154; not the paper)

Written on 2026-10-31 at ~02:50 IST, once the GPU work ended.
- **Sources:**
  - the full write-up: `docs/exploratory/followup_feasibility.md`;
  - the working log: `docs/lab_notebook.md`;
  - results in `results/followup/`.
- **The previous report** (R5-6 part J) is in git history: `git show 0e3e62f:MORNING.md`.
- **D148's freeze held.** Nothing tonight touched `paper/`, FINDINGS.md, the frozen sets or the paper's results. The
  frozen D-pos-v2 logs were only read.

**Short version.**
1. **The plan came first.** D151 (the plan and every analysis rule, marked exploratory) was pushed at 01:15 IST
   (`a8938c7`), before any data.
2. **Item 1, uncertainty-guided test-time compute:** a modest token saving, and voting itself buys little.
   - **Voting:** on X2's samples, majority@8 is only +4.3 points over one sample (0.382 vs 0.339) at 8× the tokens.
   - **The selected policy** (chosen on half the problems): "stop after one sample if its stated σ is small, else draw
     until two agree". On the other half it uses 26% of majority@8's tokens at −3.6 points.
     - At matched accuracy it saves **21% [12%, 30%]** of the fixed-k curve's tokens.
   - **Policies with no accuracy loss** ("stop at 3 agreeing", "stop when the majority is settled") save 10–21%.
   - **MC-A's raw spread as a two-stage stop rule fails** on every set (it loses AUROC).
3. **Item 2, a calculator on the model's own lines:** it hurts slightly.
   - **The anchors** reproduce #35 exactly (−0.0397, −0.0417).
   - **Re-evaluating every S formula** at the model's own values, with exact aggregation, scores **−0.0016 [−0.0034,
     −0.0003]** below the stated σ, and below RSS and X1.
   - **Re-deriving the intermediates too** recovers it to ≈ the stated σ. It is best at depths 1–2 and worse at 4–5.
   - **The bottleneck is the derivative and the chain, not the arithmetic** (#35's picture).
4. **Item 3, the 7B scale check: stopped by its own pre-stated rule (D153); there is no 7B result.**
   - **What went wrong:**
     - X2's T = 1 decoding degenerates on the 7B (D152 moved to Qwen's documented sampling);
     - the 7B's native chain-of-thought needs more than 1,024 tokens from depth 2 up;
     - at ~6.6 decoding steps / s under bitsandbytes, a problem takes 139–155 s at 1,024 tokens, against the rule's
       120 s.
   - **A development-only measurement at 2,048 tokens:**
     - 277 / 474 / 852 s per problem at depths 1–3;
     - batch 8 runs out of memory at that budget;
     - so about 50 problems per night locally, too few to measure.
   - **A real run needs** a cloud GPU or vLLM-class serving (estimated 1–2.5 h for 1,200 problems; an assumption). The
     code, the tested parser, the readouts and a queue script are ready.
5. **Tests** (§5): the new tests pass (15). One paper test broke on the decision count, and D155 fixed it with `paper/` untouched.
   No experiment needs your decision tonight; item 3's next step does (§6).

## 1. Item 1: uncertainty-guided test-time compute (`results/followup/ttc/`)

- **Data:** X2's 1,200 candidates × 8 samples of the SFT 1.5B; tokens are completion tokens.
- **Averaging:** each number is the mean over 200 random draw orders, which agrees with the fixed order.
- **The selection:** the policy was selected on a hash half (604 / 596) and is reported on the other half.

| evaluation half (604 problems) | accuracy | tokens | Δ vs maj@8 | tokens / maj@8 | matched-accuracy saving |
|---|---|---|---|---|---|
| **D75** (stated-σ gate, then 2 agreeing) | 0.353 | 402 | −0.036 [−0.049, −0.022] | 0.26 | **+0.21 [+0.12, +0.30]** |
| majority@8 | 0.388 | 1,563 | – | 1.00 | – |

- **In-sample, all problems:**
  - B3 (stop at 3 agreeing) loses 0.001 [−0.005, +0.003] at 0.79 of the tokens;
  - C (stop when settled) loses 0.000 at 0.90;
  - A (your example: 2 agree, else 8) loses 0.003 at 0.81.
- **The gate works between problems:** a small first stated σ means "more samples will not change the answer" often
  enough. That is not within-problem signal, which X2 (#19) showed this σ does not have.
- **MC-A** (B-v2, A6c): stopping after 2 re-asks when their spread is small gives AUROC 0.663 at 939 tokens, against
  fixed k = 3 at 0.700 with 751 tokens. A k = 2 and a k = 8 spread are not on one scale.
- **Caveats:**
  - early-stopped self-consistency is prior art (references to verify);
  - one model and T = 1;
  - the fixed-k curve is flat, so matched savings are noisy in-sample.

## 2. Item 2: exact arithmetic on D-pos-v2's own lines (`results/followup/arith/results.json`)

- **Coverage:**
  - 3,016–3,023 problems per seed are re-evaluated;
  - 95% of S formulas evaluate (the rest keep the written number);
  - σ moves by more than 25% on 10–14% of problems.

| variant (seed mean, oracle-centred expected crps_log) | value | − (a) [95% CI] | − RSS | − X1 | A6c slope |
|---|---|---|---|---|---|
| (a) as stated | −0.0397 | – | +0.0002 | −0.0005 | 0.837 |
| E: S re-evaluated, exact aggregation + copy | −0.0413 | **−0.0016 [−0.0034, −0.0003]** | −0.0014 | −0.0021 | 0.826 |
| E_S: S re-evaluated, own aggregation + copy | −0.0403 | −0.0006 [−0.0014, −0.0000] | −0.0004 | −0.0011 | 0.849 |
| E_chain: intermediates re-derived too | −0.0402 | −0.0005 [−0.0018, +0.0006] | −0.0003 | −0.0010 | 0.829 |

- **E_chain by depth:**
  - −0.0343 / −0.0368 at depths 1 / 2: the best non-oracle source there, ahead of RSS (−0.0371 / −0.0381) and X1;
    descriptive, seen after the fact;
  - −0.0476 at depth 5, against RSS's −0.0423.

## 3. Item 3: the 7B (`results/followup/scale/`, D152–D154)

| development problems only (kinematics, depths 1 / 2 / 3), 2,048 tokens | 7B 4-bit | native 1.5B 4-bit |
|---|---|---|
| s / problem | 277 / 474 / 852 | 23 / 67 / 46 |
| batch 8 | out of memory (halves to 2 × 4, then 2 × 2) | fits (peak 2.3 GiB) |
| answered correct | 13 / 20 (all 3 problems mixed) | 3 / 24 |
| P(yes) / hidden states, s per answer | 0.89 / 0.81 | 0.14 / 0.12 |

- **The parser** was checked on these answers only. One fix (D154): the unit written after the box.
- **No 7B answer to a non-development problem exists.**

## 4. Decisions made tonight (all logged; please review)

- **D152, decoding:**
  - **The change:** T 0.7, top_p 0.8, top_k 20 for both 7B and the 1.5B control, instead of X2's T = 1.
  - **The reason:** T = 1 makes the 7B ramble and switch language.
  - **The cost:** the comparison with X2's SFT numbers changes model, format and decoding at once.
- **D153, item 3 stopped by D151's own rule.**
  - **Not done:** a depth-1-only run, a short-answer prompt, or the 7B judging the 1.5B's answers. Each would have
    changed the design after the rule fired.
  - **Instead:** the GPU time went to a development-only measurement for planning.
- **D154:** the parser fix (development check), and the process fault below.
- **D155:** the one paper-side change, to tooling only.
  - **Why:** the full suite failed 1 test, because the paper's macro "The log has N entries" counts every D-entry, and
    D151–D154 moved it from 150 to 154.
  - **The change:** `scripts/paper/macros.py` now skips D-entries under a section headed as the follow-up project. The
    paper's value stays 150, and `paper/` is unchanged.
  - **The alternatives,** a one-line change either way: count them (rebuild to 154), or move the follow-up log to its
    own file.

## 5. Problems, and the state

- **A fault of mine, with no consequence:**
  - **The cause:** the hand-launched feasibility runs used `timeout … setsid uv run …`, which put Python in its own
    process group. The temperature and stall guards watched `timeout`'s group, so a pause would not have stopped the
    GPU process.
  - **Found and fixed** mid-run (02:15). The GPU peaked at 84.3 °C, so no pause was due.
  - **The queue scripts are unaffected** (`run_stage` uses plain `timeout`).
- **Hand-stopped runs:**
  - the first T = 1 feasibility (rule-relevant numbers in D152);
  - the 1,024-token feasibility after 2 problems (D153);
  - a 12-problem 2,048-token run after 1 problem, which would have hit its own timeout before writing anything. It
    was re-run on 3 problems.
  - Their logs are committed.
- **BLOCKED:** none.
- **Tests:**
  - **The full suite after the GPU work:** 755 passed, 1 failed (the paper count, fixed by D155).
  - **The re-run after D155:** 755 passed, 1 failed.
    - The failure was `test_temp_guard.py::test_an_unreadable_temperature_never_ends_a_pause_early`, a wall-clock
      test that measured a 3 s pause as 4 s under nice 19.
    - That file was not touched tonight. Alone it passes 3 of 3 times (6 / 6 each).
  - So every test passes, with one known timing flake.
  - `test_followup.py` adds 15.
- **The state:**
  - no GPU process;
  - every commit pushed;
  - free space on MLStorage ~42 GiB after the 5.5 GB model download.
  - The 7B weights stay in the repo-local HF cache for a future run (`.cache/` is gitignored).

## 6. Recommendation

- **Item 3 is the one worth continuing**, and it needs other hardware. My suggestion:
  - one cloud session with vLLM (AWQ 7B, or bf16 on a 40–80 GB GPU) running `followup_scale`'s prompts and parser on
    the first 600–1,200 X2 candidates;
  - the readouts (P(yes), hidden states) can run locally afterwards at ~1.7 s per kept answer, or there.
  - **Before that:** decide whether the native 1.5B control is worth it. It is cheap but rarely right natively.
- **Item 1:** publishable only as a small note unless a model with a larger voting gain is used.
- **Item 2:** answers its question (no). The interesting next step is symbolic differentiation of the model's own
  chain.
