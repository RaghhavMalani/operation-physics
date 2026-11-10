# Decisions log of the follow-up project (exploratory; not the paper)

The follow-up project's decisions (D151 on), moved here from DECISIONS.md by D156 so that the paper's log stays its
own (D148's freeze; the paper counts DECISIONS.md's entries). The numbering continues the main log's. Entries D151-D155
are moved verbatim; where they say "DECISIONS.md" or "this log", read this file. Every future follow-up entry is
appended here.

## 2026-10-31 night: an exploratory track for a follow-up project, not this paper (your instructions at ~01:00 IST; GPU deadline 11:00 IST)

### D151. Plan and analysis rules of the follow-up feasibility track (three items), recorded before any of its data exists. EXPLORATORY (2026-10-31, ~01:40 IST)
- **Status.** Exploratory, for a follow-up project. No hypothesis is pre-registered and nothing here is confirmatory.
  - **D148's freeze applies to the paper.** No result from this track enters `paper/`, FINDINGS.md or the paper's
    tables. The frozen sets and the confirmatory logs are read and never written.
  - **Outputs:** `results/followup/<item>/` and `docs/exploratory/followup_feasibility.md`.
  - **Rules stated here before any data.** Every later change gets its own D-entry, with the reason.
- **Resources.**
  - **CPU (items 1, 2):** 1 thread (OMP/OPENBLAS/MKL = 1), nice 19, while the GPU runs (D97).
  - **GPU (item 3):** `scripts/queue_followup.sh`, deadline 11:00 IST. It has:
    - D142-style resumable stages: every finished chunk is on disk, and a restart does only the missing ones;
    - the temperature guard (D143), the stall watchdog (D132) and the free-space guard (D111);
    - a launch cap of total − host − 0.5 GiB;
    - a commit + push after every stage.
- **Item 1: uncertainty-guided test-time compute** (offline, `physcal/eval/followup_ttc.py`; CPU).
  - **Data:** X2's k = 8 sampled completions (`results/x2/sft_k8/chunk_*`: 1,200 candidates, the corrected SFT 1.5B,
    T = 1). There is no new generation.
  - **Answers** are X2's (`x2_pairs.answer`): a parse failure or unit mismatch is not an answer. Correct means
    |relative error| ≤ 1%.
  - **Agreement:** |μ_j / μ_i − 1| ≤ 1% in SI. For μ_i = 0 it means μ_j = 0 exactly.
  - **The majority answer** of the drawn samples:
    - the parsed answer with the most agreeing parsed answers among those drawn, itself included;
    - ties go to the earliest drawn;
    - no parsed answer means no answer, which is scored wrong.
  - **Cost:** completion tokens of the drawn samples (`n_tokens`). The prompt is shared and is not counted. The mean
    number of samples is reported beside it.
  - **Draw order:** the k samples are exchangeable. Every quantity is the mean over R = 200 random orders per problem
    (numpy default_rng, seed 0 + pool index). The fixed order 0..7 is reported as a check.
  - **Fixed-k majority voting:** k = 1..8 on the same orders. That is the baseline curve.
  - **Adaptive policies** (the whole grid, fixed now):
    - **A** (your example): draw 2. If both parse and agree, stop with that answer. Otherwise draw the other 6 and take
      the majority of 8.
    - **B_c**, c ∈ {2, 3, 4}: draw 2, then one at a time, and stop as soon as some answer has ≥ c votes, or at 8. The
      answer is the majority.
    - **C** (no-loss stop, adaptive-consistency style): draw one at a time from 1. Stop once the leading answer's votes
      exceed the runner-up's votes plus the samples left (8 − drawn). The answer is the majority.
    - **D_q**, q ∈ {0.25, 0.5, 0.75}, the stated-σ gate:
      - draw 1, and stop if its stated relative σ ≤ the q-quantile of first-sample stated σ on the selection half;
      - otherwise continue as B_2;
      - a sample with no stated σ continues.
  - **The primary quantity:** the matched-accuracy token saving S_P = 1 − T_P / T_fix(acc_P).
    - T_fix(a) is the token cost at which the fixed-k curve reaches accuracy a.
      - The curve is made monotone first: the running maximum of accuracy over k.
      - T_fix is then a linear interpolation in (tokens, accuracy).
    - If acc_P is above the curve's maximum, T_fix is the cost of the first k that reaches the maximum. S_P is then
      flagged as a lower bound.
    - If acc_P is below acc(k = 1), S_P = 1 − T_P / T_1, flagged.
  - **Selection vs evaluation (one split, fixed now):** problem i is in the selection half iff the first byte of
    SHA-256("followup-ttc|" + seed) is even.
    - The headline is the policy with the largest S_P on the selection half.
    - It is reported on the evaluation half: S_P, accuracy minus majority@8, and tokens relative to majority@8.
    - The CIs are a bootstrap over the evaluation half's problems (B = 10,000, seed 0). Each replicate recomputes the
      fixed-k curve.
  - **Every policy is also reported on all 1,200 problems**, labelled in-sample: accuracy, tokens, samples, Δ accuracy
    and token ratio vs majority@8, and S_P. Bootstrap CIs are over problems.
  - **MC-A's raw spread as a stopping rule** (secondary, descriptive; the only sets where it exists):
    - **Data:** `results/mc_a/{a6c,a6,a6b}_pairs` (B-v2 seed 0; run 0 = its eval completion; re-asks j = 1..8 on
      perturbed inputs).
    - **There is no "majority answer" here:** each re-ask answers a perturbed problem. So the policy delivers run 0's μ
      and the raw spread as its uncertainty.
    - **The quantity:** MC-A's AUROC for "run 0 not computation-correct" (τ = 10⁻³, `mc_a.auroc`), against the mean
      re-ask tokens. The raw spread is SD (ddof 1) of ln μ over the valid re-asks drawn, and 0 (the floor) below 2.
    - **The baseline:** fixed k = 2..8 re-asks.
    - **The rule:** draw 2 re-asks. Stop if their raw spread ≤ θ, otherwise draw all 8. θ ∈ the {0.25, 0.5, 0.75}
      quantiles of the k = 2 spread on the selection half (the same hash split).
    - **Reported:** the matched-AUROC saving, as S_P with AUROC in place of accuracy.
    - **Sets:** A6c is the set the rule is selected and reported on; A6 and A6b get the same θ, descriptively.
    - **CIs:** a cluster bootstrap over pairs (B = 10,000, seed 0); the AUROC is recomputed in each replicate.
- **Item 2: exact arithmetic on the model's own lines** (offline, `physcal/eval/followup_arith.py`; CPU).
  - **Data:** D-pos-v2 seeds 0–2 on the frozen R4-5 set (`results/stage_r4/dpos2/seed*/confirmatory/`), read with
    `PHYSCAL_CONFIRMATORY=1`, and nothing written there.
  - **Kept apart from FINDINGS #35:** those were oracle swaps (the builder's exact values). Here nothing comes from the
    builder. It is a calculator on the model's own text.
  - **The model's values:** from the completion before `<steps>`.
    - Every `name = … = v unit` line counts: each `;`-part of the "SI values:" line and every derivation line. S_* and
      sigma_rel lines do not.
    - v is the leading number of the last `=`-part.
    - A later definition overrides an earlier one.
  - **Re-evaluation:**
    - **The formula:** each S line's formula (its first `=`-part) is evaluated with `r5_diag.safe_eval` (a restricted
      arithmetic AST, never Python eval; exponents ≤ 12) at the model's values.
    - **A constant S line** (one part) keeps its value.
    - **A formula that does not evaluate** (unknown name, syntax) keeps the model's written S value, and is counted.
    - **Matching:** the σ line's j-th term is matched to the j-th S line in order of appearance. If the counts
      differ, the problem keeps (a).
    - **Levels** are the σ line's own (exact levels change nothing, #35).
  - **Variants:**
    - **(a)** as stated. The anchor: it must reproduce #35's −0.0397.
    - **(c)** the model's written terms, exact aggregation and copy. The anchor: it must reproduce #35's −0.0417.
    - **E (primary):** re-evaluated S values, exact root-sum-square, exact copy: σ_rel = X′.
    - **E_S (secondary):** re-evaluated S values, with the model's own aggregation and copy factors kept:
      σ = σ_a · rss(S′) / rss(own).
    - **E_chain (secondary):**
      - the intermediates are also re-derived exactly, from the model's own derivation formulas in order, starting at
        its SI values of the givens;
      - then the S formulas are evaluated at those values;
      - aggregation and copy are exact.
  - **Problems that do not decompose** (no parsable σ line, zero line, no answer σ) keep (a) in every variant.
  - **Scores, as #35:**
    - the oracle-centred expected crps_log (problems with predictive draws), seed mean;
    - differences vs (a), RSS and X1, by `RC.seed_mean_diff` (the cluster bootstrap over pairs, B = 10,000);
    - the A6c slope by `RC.seed_mean_slope` (every pair, unanswered at the floor: R5-3's convention);
    - by depth;
    - the share of terms re-evaluated, and the share of problems whose σ moves by more than 25%.
  - **Reading (descriptive):**
    - "exact arithmetic helps" if E − (a) has a 95% CI above 0;
    - "beats RSS / X1" likewise for E − RSS and E − X1;
    - no Holm, and every variant is reported.
- **Item 3: scale check of the epistemic wall** (GPU, `physcal/eval/followup_scale.py`; inference only).
  - **The model:** `unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit` at revision `2ab7fb373c247ef5933d034f016efefc66608458`,
    in the repo-local HF cache, loaded offline.
    - The content hash of its weights (`fingerprint._dir_content_hash`) is recorded at the first load.
    - Nothing is trained.
  - **The native format:**
    - **System:** "Please reason step by step, and put your final answer within \\boxed{}." (Qwen2.5-Math's documented
      chain-of-thought prompt).
    - **User:** the X2 problem text with its `<steps>` sentence removed (`render.STEPS_TEMPLATES` /
      `NO_STEPS_TEMPLATES`). The question sentence, with its unit, stays.
    - **The model's chat template.**
  - **Decoding:** X2's (T = 1.0, top_p = 1.0, top_k = 0, k = 8 per problem), max_new_tokens 1,024 (native
    chain-of-thought is longer than our format), torch seed 0 + chunk index.
  - **a. Feasibility first** (`feasibility`):
    - **Problems:** the pool's last 12 candidates (indices 1,188–1,199, one per cell). They are used only for
      feasibility and for checking the parser, and are never analyzed.
    - **Measured:** peak reserved memory under the cap, a batch that fits (k = 8 sequences of one problem; 2 problems
      if it fits), and s / problem.
    - **If the model cannot run under the cap** (OOM at k = 8 with embeddings on the GPU and with them on the CPU), or
      a problem takes more than 120 s: stop and report. No main stage then.
  - **b. Sampling** (`sample`):
    - **The order:** X2's frozen pool, in chunks of 24 candidates (2 per cell), each chunk its own file.
    - **The subset** is the completed prefix, so it is stratified at chunk granularity.
    - **When sampling stops:** at T_stop = deadline − (estimated 7B readout time for the expected mixed answers × 1.5)
      − (the 1.5B control's estimate) − 20 min. The estimates come from feasibility and are logged.
  - **The answer:**
    - **What is read:** the content of the last `\boxed{…}` (balanced braces). A missing box or a truncation means no
      answer.
    - **The number:**
      - the first number, with an optional `\times 10^{n}` / `e` exponent;
      - `,` thousands separators are removed;
      - a bare `\frac{a}{b}` of numbers is evaluated.
    - **The unit:**
      - any remaining unit text that pint parses is converted to the requested unit;
      - a dimension mismatch means no answer;
      - with no unit, the requested unit is assumed.
    - **Correct:** within 1% of `target_value` in the requested unit.
    - **Before the main run:** the parser is checked only on the 12 feasibility problems' completions. Any fix gets a
      D-entry before the main stage starts.
  - **Kept:** problems with ≥ 1 correct and ≥ 1 incorrect answer (X2's rule).
  - **c. Readouts on the kept answers** (`readout`):
    - **(i) Self-consistency:** the share of the other 7 samples within 1% (`x2_robustness` definition). CPU.
    - **(ii) Direct P(yes),** X4's `direct` arm:
      - the conversation is system, problem, the answer's completion as the assistant turn, then a user turn;
      - the user turn is X4's `x4_v1` with "your <answer>" replaced by "your \\boxed{}" (SHA-256 recorded);
      - then the generation prompt + "Verdict:", with no generation;
      - the confidence is the yes-vs-no log-odds of the next token, from X4's token lists through a float32 head.
    - **(iii) Hidden-state probe, cross-fitted as X2's secondary probe:**
      - **The state:** every layer at the token holding the closing brace of the last `\boxed{}`. Its input holds the
        whole answer, the analogue of X2's "±" position.
      - **The fit:** 5 folds grouped by problem; (layer, C ∈ {0.001, 0.01, 0.1, 1}) chosen by inner grouped CV on the
        training part, then refit (`x2_analysis.crossfit_confidence`).
      - **When:** CPU, after the GPU queue, with X2's settings (4 jobs).
      - **If all 29 layers cannot finish by the report,** the even layers are used, and that is stated.
      - **No random-init control** (GPU time). This is a stated limit: on X2 the random-init probe was within noise of
        the real one.
    - **Length only** (shorter = more confident) as the surface reference (#22).
  - **The metric:** X2's within-problem pairwise accuracy (ties 0.5), mean over kept problems, with a bootstrap over
    problems (B = 10,000, seed 0).
  - **The 1.5B comparison:**
    - **(1) The published X2 / #22 numbers:** SC 0.643, X4 direct 0.526, cross-fitted probe 0.540, length 0.535. These
      are the SFT 1.5B in our format on its 516 kept problems.
    - **(2) The same, restricted to the same pool prefix:** SC, X4 direct and length are recomputed from the existing
      rows. The probe's per-answer values come from a re-run of X2's cross-fit, if X2's settings reproduce 0.540;
      otherwise the probe is missing.
    - **(3) The control (if time allows after the 7B stages):** the untrained Qwen2.5-Math-1.5B-Instruct (D78's pinned
      4-bit base) in the same native format, on the same prefix, with the same three readouts. It separates scale
      from format and SFT.
  - **Differences:**
    - 7B − 1.5B by an independent bootstrap over each model's kept problems (unpaired; the kept sets differ);
    - on the problems kept by both, a paired bootstrap (secondary).
  - **Reading (descriptive):**
    - **"The wall moves with scale"** if a single-sample readout of the 7B ((ii) or (iii)) exceeds the native 1.5B's,
      or failing that the SFT 1.5B's, with a CI excluding 0.
    - **Whether self-consistency still leads** is reported either way.
- **The write-up:** `docs/exploratory/followup_feasibility.md`, with honest caveats:
  - item 1 is in-sample on the full set, with prior art (adaptive-consistency / early-stopping self-consistency);
  - item 2 is one model;
  - item 3 has one prompt, no random-init control, a subset, and a different format from X2.
  - **MORNING.md** comes at the end.

### D152. Item 3: the 7B cannot use X2's decoding; it samples at Qwen2.5-Math's documented settings, and so does the 1.5B control. Feasibility uses 6 dev problems (2026-10-31, ~02:25 IST; recorded before the feasibility rerun, and before any non-development 7B completion exists)
- **What happened (development problems only, pool index 1188):**
  - **Under X2's decoding** (T = 1.0, top_p = 1.0, top_k = 0), 7 of 8 samples ran to the 1,024-token cap: 140 s for the
    problem, above D151's 120 s stop rule.
  - **A T = 1 sample degenerates mid-solution:** it switches to Chinese, then rambles without boxing.
  - **At T = 0.7, top_p = 0.8** (a 2-sample check), both samples finished in 576 and 671 tokens, boxed.
  - **Generation is memory-bound:** about 7 decoding steps / s at batch 8 (bitsandbytes dequantizes every weight on
    every step). So the cost per problem is set by the longest of its 8 samples.
- **The change (alternatives: keep T = 1 and stop per D151; or T = 1 with a top-p truncation):**
  - **Decoding** is T = 0.7, top_p = 0.8, top_k = 20 (Qwen2.5's documented sampling defaults).
  - **repetition_penalty stays 1.0,** as in X2: a penalty distorts repeated digits.
  - **The control is the same.** The native 1.5B control uses the identical decoding, so the scale comparison (7B vs
    native 1.5B) changes one variable.
  - **The comparison with X2's SFT 1.5B (T = 1) changes three:** model, format and decoding. That is a stated caveat. A
    lower temperature raises agreement among samples, so self-consistency and the mixed-problem rate are not directly
    comparable to X2's.
- **Why not stop:**
  - **The 120 s rule** was written for X2's decoding. Under it the model degenerates, which is a fact about T = 1, not
    about whether the 7B can be measured.
  - **The rule still applies under the new decoding:** above 120 s / problem, stop.
- **Feasibility:**
  - **Problems:** the first 6 dev problems (1188–1193), not 12. That keeps about 10 min of GPU time.
  - **Parser checks:** they use those 6 × 8 completions and the 2-problem batch-16 test.
  - **Analysis:** dev problems are still never analyzed.

### D153. Item 3's main stage does not run: D151's 120 s / problem rule fired. What runs instead is a longer feasibility measurement on the development problems only, to size a future run (2026-10-31, ~01:55 IST; no non-development 7B completion exists)
- **The measurement** (D152's decoding, k = 8, max_new_tokens 1,024, batch 8, cap 6.7 GiB):
  - dev 1188 (kinematics, depth 1): 139.1 s; 1 of 8 samples at the cap; 6 of the 7 answered are correct.
  - dev 1189 (kinematics, depth 2): 154.7 s; 7 of 8 at the cap.
  - **The rate:** about 6.6 decoding steps / s, set by bitsandbytes dequantizing every weight at every step. The time
    per problem is the longest sample's length / 6.6.
  - **Memory:** it fits. About 7.4 GB used on the card in all, with ~0.8 GB of host use.
- **Why the samples are long:** a 2-sample check on dev 1189 (both at the cap) shows legitimate step-by-step LaTeX
  derivations, not loops. At 1,024 tokens both were still in phase 3 of 4. So the model's native chain-of-thought needs
  more than 1,024 tokens from depth 2 up.
  - **At a sufficient budget (2,048)** a problem would take about 300 s. In the ~9 h left, minus the readouts, that is
    about 90 problems, and with an expected mixed share of 0.3–0.45, about 30–40 kept problems.
  - **The resulting CI** on a within-problem pairwise accuracy would be about ±0.09.
- **The rule:** D151 / D152 say: above 120 s / problem, stop and report; no main stage. It is applied as written.
  - **Not done:**
    - a depth-1-only run (it fits the budget, but changes the design after the rule fired);
    - a shorter-answer prompt (that changes the native format);
    - using the 7B to judge the SFT 1.5B's X2 answers (a different question: an external verifier, not the model's own
      wall).
  - All three are listed as options for the follow-up.
- **What runs instead (feasibility, development problems only, never analyzed):**
  - **The 7B on all 12 development problems** (1188–1199, one per cell): k = 8, max_new_tokens 2,048, batch 8. It
    measures:
    - s / problem at a budget that completes;
    - peak memory and OOM backoffs at the full budget;
    - lengths and truncation by depth;
    - the readout costs (direct P(yes), hidden states) on 8 answers.
  - **The native 1.5B control** on the same 12 problems, with the same settings: its rate and lengths, for the cost of
    the control.
  - **No batch-16 test:** at a 2,048 budget its KV cache cannot fit next to the 7B's 5.2 GiB of weights. The arithmetic
    is in the report.
  - **The outputs** go to `results/followup/scale/<model>/feasibility_2048.json` (plus the completions). The
    1,024-token run was stopped before it wrote its file; its numbers are the two above, from its log.
- **The report** says what a real run would need:
  - a cloud GPU or a faster inference stack (e.g. vLLM with AWQ / GPTQ kernels instead of bitsandbytes);
  - or a format with short answers (which would need SFT);
  - and the N reachable per night on the 4060.

### D154. Item 3's answer parser reads a unit written right after the box; found by D151's development-only parser check. Plus the night's GPU outcome and one process fault (2026-10-31, ~02:30 IST)
- **The gap:** the native 1.5B writes its unit after the box, e.g. `\(\boxed{9.55}\) seconds.`, often in another unit
  than the one requested. D151's rule only read units inside the box, so it assumed the requested unit (ms here).
  The 7B does the same occasionally (`\boxed{27.0} \text{ s} \]`, `\boxed{0.24} \] km.`).
- **The fix (`grade` → `unit_after_box`):**
  - with no unit inside the box, the text after it on the same line is cleaned (`\)`, `\]`, `\text{}`, trailing
    punctuation) and parsed by pint, whole or by its first word;
  - with matching dimensions, the value is converted;
  - anything else after the box is ignored, i.e. it never rejects an answer.
- **Tests:** `tests/test_followup.py` (15).
- **Re-graded development completions** (`feasibility_2048_regraded.json`; development only, never analyzed):
  - **7B:** unchanged. 13 of 20 answered correct; 4 answers had their unit after the box. All 3 development problems
    have right and wrong answers.
  - **Native 1.5B:** 0 → 3 of 24 correct. Its other answers are wrong in value, or a bare number in the wrong unit.
- **The night's GPU outcome** (D153's development-only measurement, 3 problems: kinematics at depths 1–3):
  - **7B at 2,048 tokens:**
    - 277 / 474 / 852 s per problem;
    - batch 8 runs out of memory at the full budget: at depth 2 it halves to 2 × 4, and at depth 3 the first half
      splits again into 2 × 2;
    - the peak reserved is 6.70 GiB at the 6.7 cap;
    - complete answers take 347–808 / 1,117–1,559 / 1,050–1,862 tokens, and 4 of 24 samples run to 2,048;
    - readouts take 0.89 s (P(yes)) and 0.81 s (29 layers) per answer;
    - both logit checks pass (gaps 0.0024, 0.0008).
  - **Native 1.5B at 2,048 tokens:** 23 / 67 / 46 s per problem, with no OOM (peak 2.33 GiB). Readouts take 0.14 /
    0.12 s per answer.
- **A process fault of mine (no consequence tonight):**
  - **The cause:** the hand-launched feasibility runs used `timeout … setsid uv run …`. `setsid` put Python in its
    own process group, while `gpu_temp_guard.sh` and `stall_guard.sh` watched `timeout`'s group. A pause would have
    stopped only `timeout`.
  - **Found** at 02:15 IST during the third 7B run; the guard was re-pointed at Python's group.
  - **The effect:** the GPU peaked at 84.3 °C, below 87 °C, so no pause was ever due.
  - **The queue scripts are not affected:** `run_stage` runs `timeout CMD`, which leads the group the guards watch.
  - **The last run (the 1.5B)** was launched without `setsid`, and the guards' group was checked.
- **There is no further GPU work tonight.** D153 stopped the main stage, the measurement is done, and the brief has no
  other GPU item. `scripts/queue_followup.sh` (tested, unused) is kept for a future run.

### D155. The paper's D-entry count excludes the follow-up project's entries, so the paper stays at 150 and its build test passes (2026-10-31, ~03:15 IST)
- **What failed:** after the night's work the full suite was 755 passed, 1 failed. The failure was
  `tests/test_paper.py::test_the_committed_generated_files_are_a_fresh_build`.
  - `scripts/paper/macros.py` counts every `### D<n>.` header for the paper's sentence "The log has
    \pv{record.dentries} entries".
  - D151–D154 moved that count from 150 to 154.
- **The choice:** the paper should not count another project's decisions, and the paper must stay untouched (your
  instruction).
  - **The change:** the macro skips D-entries under a `## …` section whose header names the "follow-up project" (this
    night's section does).
  - **The effect:** the value stays 150. `paper/generated/` is unchanged, and the test compares values only.
- **The alternatives (for your review):**
  - rebuild `paper/generated/` with 154 (the paper would count follow-up entries);
  - move the follow-up track's decisions to a separate log.
  - Either is a one-line change if you prefer it.
- **This entry sits under the follow-up section**, so it is not counted either.

## 2026-11-01: item 3 moves to Kaggle (your instructions, evening)

### D156. D155's alternative: the follow-up log moves to its own file, and the paper's macro counts DECISIONS.md as before. D152 is accepted (2026-11-01)
- **Your decision:** the second alternative D155 listed. Every follow-up D-entry, D151 on and all future ones, lives
  in this file.
- **The move:**
  - D151–D155 and their section header were moved verbatim from DECISIONS.md to this file;
  - DECISIONS.md now ends with a short pointer section, which has no `### D<n>.` header. So it holds D1–D150 again.
- **The revert:** D155's three lines in `scripts/paper/macros.py` are removed. The macro counts every `### D<n>.`
  header of DECISIONS.md, as at D150, and the value is 150.
  - `paper/` is untouched;
  - `tests/test_paper.py`'s fresh-build test passes. It runs once DECISIONS.md is committed, and skips while it has
    uncommitted changes.
- **D155 is superseded:** its macro change no longer exists; its entry stays above as the record.
- **D152 is accepted** by you: Qwen2.5's documented sampling (T 0.7, top_p 0.8, top_k 20, repetition penalty 1.0) for
  item 3.
- **Pointers updated:** `scripts/queue_followup.sh` and `docs/exploratory/followup_feasibility.md` now name this file.
  MORNING.md and the lab notebook are records of their nights and keep saying "DECISIONS.md".

### D157. Item 3 runs on Kaggle "GPU T4 x2" with vLLM: the official Qwen2.5-Math-7B-Instruct in fp16, D152's decoding, D151's prompts, parser and analysis; the readouts stay local. Every difference from D151's local plan (2026-11-01, ~23:00 IST; recorded before any Kaggle run, and before any non-development 7B completion exists)
- **Why:** D153 stopped the local run (139–155 s per problem at 1,024 tokens, against the 120 s rule). D153's
  development measurement put a sufficient budget at about 50 problems per night on the 4060. Your instruction is to
  sample on Kaggle and keep the readouts local.
- **What runs where:**
  - **On Kaggle:** `notebooks/kaggle_7b_scale.ipynb` → `physcal.eval.followup_kaggle` (`download`, `restore`,
    `sample`, `pack`). The repo is cloned at a pinned commit, the one that holds this entry's code. The commit is
    recorded in the notebook and in every session's meta.
  - **Locally:** `scripts/queue_followup_kaggle.sh` runs `ingest`, then followup_scale's `yes`, `hidden` and
    `analyze`.
  - **Tests:** `tests/test_followup_kaggle.py` (12), with a fake engine, clock and tokenizer. One test checks
    that the notebook's pinned commit holds the working tree's sampler, scale module and configs.
- **Unchanged from D151–D154:**
  - X2's frozen pool, in order; the development indices 1188–1199 are never sampled;
  - chunks of 24 candidates (2 per cell); the subset is the completed prefix;
  - k = 8;
  - the native prompt (Qwen2.5-Math's chain-of-thought system prompt; the problem text without its `<steps>`
    sentence; the model's chat template);
  - the parser (D151 + D154) and the 1% tolerance;
  - the kept rule (≥ 1 correct and ≥ 1 incorrect answer);
  - the readouts:
    - self-consistency;
    - X4-style direct P(yes) with the same verification turn;
    - the cross-fitted probe at the token of the last box's closing brace;
    - length;
  - the metric (X2's within-problem pairwise accuracy, bootstrap B = 10,000, seed 0);
  - the comparisons and reading rules;
  - no random-init control;
  - D148's freeze.
- **Differences from D151's local plan:**
  1. **Hardware and stack.**
     - **Kaggle:** "GPU T4 x2" (2 × T4, 15 GiB each, sm75), vLLM 0.29.0 (which pins torch 2.13.0, CUDA 13), in a uv
       0.13.1 venv with Python 3.11, isolated from Kaggle's preinstalled packages. pint 0.25.3, PyYAML 6.0.3 and sympy
       1.14.0 (the repo's lock) are added for the prompt and the parser.
     - **The notebook checks** for 2 GPUs and a driver ≥ 580 before installing. A public report has vLLM 0.29.0
       running in fp16 on one T4; tensor-parallel 2 on Kaggle is not verified beforehand.
     - **D151's plan:** the local 4060 with transformers + Unsloth + bitsandbytes (`HFBackend`).
  2. **Weights.**
     - **The model:** the official `Qwen/Qwen2.5-Math-7B-Instruct` at revision
       `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`.
     - **Precision:** `dtype half`, because T4 has no bf16; the bf16 checkpoint is cast to fp16. `tensor_parallel_size`
       is 2, and custom all-reduce is off (no P2P between the two T4s).
     - **Kernels:**
       - attention is vLLM's Triton backend, pinned (`attention_backend TRITON_ATTN`). This is what v0.29
         auto-selects on sm75, since its FlashInfer backend is disabled below sm80 (checked in the v0.29.0 source),
         and a public report runs it on a T4;
       - FlashInfer's sampler is off (`VLLM_USE_FLASHINFER_SAMPLER=0`), so nothing is compiled for the T4.
     - **Integrity:** each of the four safetensors files must match HF's LFS sha256 at that revision (in the config).
     - **D151's plan:** `unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit` at `2ab7fb37…`.
     - **The fp16 risk:** fp16's range is 65,504, against bf16's. Overflow is possible; the health gate (difference 7)
       is the guard.
  3. **The readouts read another copy than the sampler.**
     - **Where:** P(yes) and the hidden states run locally on the 4-bit copy, the only 7B that fits 8 GB. They read the
       fp16 model's answers through a 4-bit quantization of the same weights. In X2 the same model sampled and read.
       This is a stated caveat.
     - **Checked (2026-11-01):** the official and the 4-bit copies have the same chat template, vocabulary and merges.
       Unsloth adds one pad token, which is never in a prompt.
     - **At ingest:** every prompt is re-rendered with the local tokenizer and must match the Kaggle prompt's sha256
       and token count.
  4. **The sampler and its seeds.**
     - **The parameters** are D152's (accepted in D156): T 0.7, top_p 0.8, top_k 20, repetition penalty 1.0. The stop
       ids are the model's `generation_config` end tokens.
     - **One request per sample** (n = 1), with seed 0 + 8 × pool index + sample. D151 used one torch seed per chunk.
     - **Reproducibility:** a sample's random numbers do not depend on the batch. The numerics of batched kernels
       still can, so a rerun is reproducible in distribution, not bit for bit. vLLM's sampler is not HF's.
  5. **The budget.**
     - **max_new_tokens is 3,072**, against D151's 1,024. D153 found that more than 1,024 is needed from depth 2 up.
       At 2,048, 4 of 24 development samples at depths 1–3 still ran out, and complete depth-3 answers took up to
       1,862 tokens.
     - **The context:** the longest prompt among the 1,188 candidates is 363 tokens (measured on 2026-11-01), so a
       prompt plus its completion stays within `max_model_len` 4,096, the model's context.
     - **A truncated sample** is still no answer (D151).
  6. **The stop rule.** D151's 120 s / problem feasibility rule and its deadline-derived T_stop are replaced by:
     - **Timing:** the first 20 candidates (indices 0–19, part of chunk 0 and analyzed like the rest) are generated
       on their own and timed. N is sized from their timing only, never from correctness:
       - N = done + 24 × floor((session start + 9.5 h − now) / (1.25 × s per problem × 24));
       - capped at 1,176, the full chunks below the development indices.
     - **The chunk rule:** a chunk starts only if its projected end (1.25 × the session's mean chunk time) is before
       session start + 9.5 h. That leaves ≥ 2.5 h inside Kaggle's 12 h for the install, the 15 GB download, the load
       and the zip.
     - **A resumed session** re-plans with its own deadline and the stored timing.
  7. **The health gate (new; stated before any data).**
     - **The rule:** if fewer than 50% of the 160 timed samples are answered (boxed, parsed, not truncated), sampling
       stops and nothing more is drawn in any session. The result is reported.
     - **Why:** it guards against fp16 degeneration. The gate reads only the format, not correctness. For scale, the
       4-bit 7B answered 20 of 24 development samples at 2,048 tokens.
  8. **No development feasibility on Kaggle.**
     - The parser was checked on development answers (D154) and is unchanged.
     - **Failures are handled by the notebook:**
       - the sampler is killed if it prints nothing for 45 min (a chunk is ~3–10 min);
       - after any failure (the engine does not start, a crash, a stall) it is retried once with `enforce_eager` (no
         CUDA graphs) and `NCCL_P2P_DISABLE=1`. These change speed only, and are recorded in the meta;
       - the retry resumes from the chunks on disk.
  9. **The output format.**
     - **Each chunk row carries:**
       - the tape fields (`run_id … breakdown`); `reward` is null, because the native `\boxed{}` format has no
         verifier reward, and `breakdown` is the D151 / D154 grade;
       - item 3's chunk fields (index, key, sample, completion, n_tokens, truncated, the grade);
       - `sample_seed`, `finish_reason`, `prompt_sha256` and `prompt_tokens`.
     - **The reader:** `followup_scale.sampled_prefix` reads the rows unchanged.
     - **n_tokens** is vLLM's completion token count. Only differences within a problem enter the length readout.
     - **The meta records** the commit, the config sha256, the package versions, the GPUs and driver, the engine
       arguments, and the chat-template and tokenizer hashes.
  10. **Resuming.**
      - **Within a session:** chunks are written atomically, so a cutoff loses at most one chunk.
      - **Across sessions:** a Kaggle session starts empty. A re-run resumes from a previous version's output attached
        as input; `restore` copies its chunk prefix after `sampled_prefix`'s check and refuses another revision or
        decoding.
  11. **Grading.**
      - **On Kaggle** it is for progress only.
      - **At ingest** every row is re-graded with the local parser, which is authoritative (D151's analysis
        re-grades anyway). Any change is counted in `ingest.json`.
      - **Ingest also checks:**
        - the zip (allowed names only, sizes, the manifest's sha256);
        - every row's problem against the frozen pool;
        - the seeds;
        - every segment's revision and decoding.
  12. **The local readout context:** `scale.max_seq_length` goes from 3,072 to 4,096. A kept answer's P(yes) context
      is up to ~3,530 tokens: prompt + completion + verification turn.
  13. **The native 1.5B control is not in this run** (D151 (3): "if time allows").
      - Its columns stay missing unless a later entry adds it.
      - The 7B is compared with the SFT 1.5B's published X2 numbers, and with X2 restricted to the same prefix. Both
        comparisons change model, format and decoding at once (D152's caveat).
- **Expected cost (an assumption, not a measurement):**
  - a chunk is set by its longest sample: at ~15 decoding steps / s, up to 3,072 tokens is ~3.5 min;
  - so 1,176 candidates would take ~3–4 h;
  - the measured rate decides N.
  - **The local readouts:** about 0.45 × N × 8 kept answers at ~2–3 s each (D154's 0.9 + 0.8 s at ≤ 2,048 tokens,
    longer contexts here). At N = 1,176 that is ~3 h. The stages are resumable.

## 2026-11-02 morning: the Kaggle run is back (your instructions at ~07:10 IST; GPU deadline 13:00 IST)

### D158. The Kaggle samples are ingested; before the local readouts, the hidden-state batch goes from 2 to 1 and the readouts wait for enough free VRAM (2026-11-02, ~07:25 IST; before any readout of a Kaggle answer)
- **The Kaggle run (D157) finished cleanly:**
  - one session, on the first attempt: no stall, no retry, no `enforce_eager`;
  - the stop reason is "planned candidates done";
  - the timing on the first 20 candidates was 17.2 s / problem (568 tokens / s), so the plan was all 1,176;
  - sampling took 5.0 h (17:53–22:56 UTC on 2026-11-01), 15.4 s / problem;
  - the zip's sha256 is `10bdb2d9…` (in `results/followup/scale/7b/kaggle/ingest.json`).
- **The ingest passed every check:**
  - 1,176 candidates × 8 samples, all 49 chunks;
  - the manifest;
  - every row's problem against the frozen pool, and the seeds;
  - all 1,176 prompts re-render to the same sha256 and length with the local tokenizer;
  - the local re-grade changed 0 rows.
- **What was sampled (descriptive, no readout yet):**
  - answered 91.1%, truncated 8.7% at 3,072 tokens, correct per sample 51.4%;
  - by depth 1 → 4:
    - answered 96 / 93 / 89 / 86%;
    - truncated 4 / 7 / 10 / 14%;
    - correct 78 / 61 / 39 / 29%.
  - **Mixed (kept) problems:** 844 of 1,176 (71.8%). Of the rest, 216 have only correct answers and 116 only wrong
    ones.
  - **The readouts' items:** 6,133 answered answers of kept problems; completions average 1,003 tokens, at most 2,776.
- **Two changes before any readout. Both concern memory only, not what is measured.**
  1. **`scale.hidden_batch_size`: 2 → 1.**
     - **Why:** the hidden-state pass keeps every layer's full sequence (`output_hidden_states`), and it has no OOM
       backoff. At batch 2, the longest pair padded to ~3,100 tokens would need ~1.3 GB on top of the 5.23 GiB the
       loaded model holds. D153's development answers were ≤ 2,048 tokens.
     - **Effect:** a state differs only by padding numerics (right padding with a mask), and it is somewhat slower.
       Direct P(yes) already halves its batch on an OOM.
  2. **A launch gate in `scripts/queue_followup_kaggle.sh`.** The GPU stages start only when the launch cap is
     ≥ 6.5 GiB, the estimated peak of the longest readouts.
     - **The wait:** if the cap is lower, the chain re-reads every 60 s for up to 60 min. After that it stops as
       BLOCKED, with nothing started.
     - **Why:** at 07:15 IST, Windows apps held 1.9–2.2 GiB of VRAM, so the usual cap (total − host − 0.5 GiB) was
       ~5.4 GiB. A stage at that cap would run out of memory, and a larger cap would spill (the WSL rule).
- **The pin test is narrowed.** `test_followup_kaggle`'s check of the notebook's commit now compares only the sampling
  settings of `configs/followup.yaml`: `scale.kaggle`, generation, system prompt, tolerance, chunk size, development
  indices and pool. The sampler, scale module, render, units and x2 config are still compared byte for byte. A
  readout setting may change after the Kaggle run.
- **The readout estimate** uses D154's measured 0.89 s (P(yes)) and 0.81 s (hidden, batch 2) per answer, at similar
  lengths. With batch 1 the hidden pass is assumed at ~0.95 s.
  - **Total:** 6,133 × ~1.85 s ≈ 3.1 h, plus two loads ≈ 3.3 h.
  - **The deadline:** if started by ~09:00 IST, it ends before 13:00 IST.
  - **Otherwise:** the stages stop at the deadline and resume on a re-run (P(yes) appends rows; hidden states are
    saved in parts of 64).
- **The analysis** (CPU, after both readouts) is D151's, unchanged. The SFT 1.5B comparison uses X2 restricted to the
  same 1,176-candidate prefix, plus the published numbers.

## 2026-11-02 midday: the local readouts and the analysis (your instructions at ~10:00 IST: GPU deadline 14:00 IST)

### D159. Item 3's local readouts and D151's analysis ran on the Kaggle samples. The chain was restarted once for the later deadline. The SFT 1.5B's probe on the same prefix is missing, and the write-up says how it is compared instead (2026-11-02, ~14:00 IST; after the analysis, before the write-up)
- **The run, in order (`results/queue_followup_kaggle/chain.log`):**
  - 07:20: the first chain waited at D158's gate (cap 5.5–5.6 GiB < 6.5) and stopped as BLOCKED at 08:27, with
    nothing started.
  - 08:28: relaunched with a longer gate wait (`CAP_WAIT_S=17000`). The gate opened at 09:41 (cap 6.7 GiB) once
    Windows' GPU apps were closed, and P(yes) started.
  - 10:05: **restarted on your instruction**, to move the GPU deadline from 13:00 to 14:00. A running bash cannot re-read
    its deadline.
    - P(yes) was stopped right after a saved batch: 1,504 of 6,133 rows on disk, every row parses.
    - The copy of that attempt's log is `fs_yes_7b.attempt1.log`.
    - The cost was ~6 min (the chain's start-up and the model reload). Nothing was recomputed.
  - 10:11–11:22: P(yes), the other 4,629 answers (0.92 s / answer), with 9 temperature pauses (270 s).
  - 11:22–12:41: hidden states, all 6,133 answers at batch 1 (0.77 s / answer), with 7 temperature pauses (210 s).
  - 12:41–13:56: D151's analysis (CPU, 4 jobs, the GPU idle).
  - Every stage committed and pushed.
- **The analysis is D151's, unchanged.** Nothing in its rules or code changed after the readouts.
- **One comparison D151 lists is missing: the SFT 1.5B's cross-fitted probe on the same prefix** (D151 (2)).
  - **Why:** `followup_scale_analysis.x2_on_prefix` does not re-run X2's cross-fit; it reports the probe as missing.
    That was the implementation's choice before any data; D151 allowed a re-run.
  - **How it is compared instead:**
    - against the published X2 number, 0.540 [0.514, 0.565] on 516 kept problems (the full pool; the prefix keeps 504);
    - with an approximate unpaired difference: normal, from the two bootstrap CIs. It is labelled approximate, and it
      is not D151's bootstrap.
  - **Not re-run now:** the analysis saves no per-problem probe values, so a proper bootstrap would refit the 7B probe
    too (~75 min of CPU). The two CIs are ~0.17 apart, so the reading would not change. It can be run later.
- **The probe's selection,** a fact for the write-up: layer 19 of 29 and C = 0.001 in all five folds (inner-CV AUROC
  0.773–0.781). C = 0.001 is the edge of D151's grid. The grid is not extended after the fact.
- **The reading** is D151's, descriptive. The native 1.5B control did not run (D157, difference 13), so "the wall
  moves" is judged against the SFT 1.5B: D151's fallback. The write-up states every confound of that comparison.

## 2026-11-02 afternoon: the controls for D159 (your instructions at ~14:30 IST)

### D160. Three controls that could make D159 a scale result: the native 1.5B on Kaggle, a random-weights control for each model's probe, and D151 (2)'s refit of X2's probe on the same prefix. Plan and analysis rules (2026-11-02, ~14:45 IST; recorded before any of their data exists)
- **Why:** D159's 7B − SFT 1.5B comparison changes size, format, training, decoding, sampler precision and the
  problems at once. The 7B probe has no random-weights control. The SFT probe on the 7B's prefix is missing.
- **1. The native 1.5B on Kaggle** (D151 (3)'s control).
  - **The model:** the official `Qwen/Qwen2.5-Math-1.5B-Instruct`, revision `aafeb0fc6f22cbf0eaeed126eff8be45b0360a35`,
    one `model.safetensors` (3,087,467,144 bytes, LFS sha256 `08faf1e9…40d9`). No task training. 28 layers (29
    hidden-state arrays, as for the 7B); context 4,096.
  - **Identical to the 7B's run (D157):**
    - the prompts, checked on 2026-11-02 before any sample: the official 1.5B tokenizer, the local 4-bit 1.5B and the
      local 4-bit 7B render all 1,176 prompts to the same text and ids, and the official 1.5B matches the 7B's Kaggle
      rows' prompt sha256 and token count for all 1,176 (0 mismatches);
    - the format, D152's decoding (T 0.7, top_p 0.8, top_k 20, repetition penalty 1.0), k = 8, max_new_tokens 3,072;
    - the per-sample seeds (0 + 8 × index + sample), so each (candidate, sample) has the 7B's seed;
    - the 1,176 candidates, chunks of 24, the timing on 20 and N's sizing, the health gate (50% answered), the stall
      watchdog and the retry, the zip and every ingest check;
    - the stack: Kaggle "GPU T4 x2", vLLM 0.29.0, fp16, tensor-parallel 2, Triton attention, max_model_len 4,096,
      gpu_memory_utilization 0.90, no custom all-reduce. Tensor-parallel 2 is kept although the 1.5B fits on one T4:
      the same engine, at a cost in speed only. The 1.5B's 12 heads, 2 KV heads, 8,960 MLP width and vocabulary
      divide by 2.
  - **What differs:** the weights (repo, revision, sha256); the zip name `followup_1.5b_scale.zip`; the free-space check
    on /tmp (20 GB: weights 3.1 GB + the venv); the notebook `notebooks/kaggle_1.5b_scale.ipynb`, pinned to its own
    commit.
  - **The code change, which touches no 7B data:**
    - `followup_kaggle` takes `--model {7b, 1.5b}`;
    - a model's settings are `scale.kaggle` overlaid with `scale.kaggle_models[model]`. The 7B's stay in
      `scale.kaggle`, unchanged.
    - **The 7B notebook's pin test becomes a provenance test:** its commit is the one in the 7B's ingested meta, and
      that commit's sampling settings equal today's 7B settings. The 1.5B notebook gets D157's byte-for-byte test.
  - **The local readouts** run on the local 4-bit 1.5B (`unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit` at `e24e9530…`),
    as for the 7B: direct P(yes), the hidden states (29 layers, the same position), self-consistency and length.
- **2. A random-weights control for each probe** (the 7B tonight; the 1.5B after its zip).
  - **The weights:**
    - the readout copy's architecture (its `config.json` without the quantization block), with random weights;
    - every Linear and Embedding weight ~ N(0, 0.02²) (`initializer_range`); the embedding's padding row 0; biases 0;
      RMSNorm weights 1. That is transformers' Qwen2 initialization;
    - seed 0, one `torch.Generator`, tensors in parameter order;
    - saved in bf16 with the readout copy's tokenizer files. The 7B is ~15 GB, under
      `results/followup/scale/<model>/random_init_model/` (gitignored, regenerable).
    - **The one difference from X2's control (D74):** X2 used `from_config`, which cannot hold a 7B in this
      machine's RAM, so the weights are generated tensor by tensor. The distribution is transformers'; the random
      stream is not.
    - **Loaded the same way as the readout copy:** Unsloth, 4-bit, quantized at load, as X2's control was.
  - **The same answers and position:** the same kept answers (6,133 for the 7B), teacher-forced with the same prompt;
    the token of the last box's closing brace; batch 1; x2_hidden's logit check.
  - **The same probe protocol:** `crossfit_confidence` with the same groups, so the same five folds; the same
    (layer, C) grid; all 29 layers; the same thread settings as the real probe.
  - **Primary:** the random-weights probe's pairwise accuracy, and real − random-weights, **paired** over the same
    kept problems (bootstrap over problems, B = 10,000, seed 0).
  - **Reading:**
    - "the probe reads more than the text's surface" if real − random-weights > 0 with a CI excluding 0;
    - otherwise the probe's number is reported as a surface signal, not as knowledge.
- **3. X2's SFT probe on the same prefix** (D151 (2)).
  - **The refit:** X2's cross-fit (`x2_analysis.crossfit_confidence`) on X2's SFT hidden states at the "±" token,
    under X2's own settings: 4 jobs and the default thread environment, at nice 19.
  - **Not single-threaded** (your instruction), because of D97: under 1 thread the same refit gave 0.5390 against
    X2's 0.5396, since the inner (layer, C) choice depends on the BLAS thread count, and D92's check refused it.
    D97's single-thread rule is for while the GPU is busy, and it runs now, while the GPU is idle.
  - **The gate (D151 (2), D92):** the refit must reproduce X2's published 0.5396 on its 516 kept problems within
    D92's 1e-9. Otherwise the probe stays missing.
  - **Then:**
    - its per-answer values are restricted to the prefix's 504 kept problems;
    - the values come from a fit on all 516; the 12 kept problems beyond the prefix take part in the fit only;
    - 7B − SFT 1.5B for the probe (unpaired, same prefix) replaces D159's approximation.
- **4. The analysis** (`followup_scale analyze`, D151's metric and bootstrap; descriptive, no multiple-comparison
  correction).
  - **The readouts per model:** self-consistency, P(yes), the probe, the random-weights probe, length.
  - **D151's comparisons, now with the native 1.5B:**
    - 7B − native 1.5B, unpaired: primary, D151;
    - the same on the problems both keep, paired: secondary;
    - 7B − SFT 1.5B on the same prefix.
  - **D151's reading with its first comparison:** "the wall moves with scale" if a single-sample readout of the 7B,
    P(yes) or the probe, exceeds the native 1.5B's with a CI excluding 0.
    - The probe counts only if the 7B's real − random-weights difference also excludes 0.
    - **Secondary, the probe's scale effect net of the surface:** (7B real − random) − (1.5B real − random),
      unpaired.
  - **Also descriptive:**
    - native 1.5B − SFT 1.5B (unpaired, same prefix): format, training and decoding at a fixed size;
    - per model, P(yes) − length and probe − length, paired: is a readout more than length?
  - **The cache:** each model's per-answer probe values are saved with a hash of their inputs (the hidden-state
    parts, the answers, the probe settings), so later analyses do not refit.
  - **The 7B probe is refit once with the same code and threads.** It must reproduce D159's 0.7526 exactly (it is
    deterministic); otherwise the run stops and is reported.
- **Where and when:**
  - **Now (CPU, the GPU idle):** item 3's refit.
  - **Tonight (local GPU, `GPU_DEADLINE=09:00`):** `MODEL=7b scripts/queue_followup_kaggle.sh`:
    - D158's VRAM gate, the temperature and stall guards, the free-space guard, a commit + push after every stage;
    - the stages: the random-weights 7B (CPU; ~15 GB written), its hidden states (GPU, ~80 min), then the analysis
      (CPU, 4 jobs, after the GPU stage; D97).
  - **On Kaggle (you):** `notebooks/kaggle_1.5b_scale.ipynb` (`docs/exploratory/kaggle_instructions.md` §8).
  - **After its zip (prepared, not run):**
    - `MODEL=1.5b KAGGLE_ZIP=… scripts/queue_followup_kaggle.sh`: ingest, P(yes), hidden states, the random-weights
      1.5B and its hidden states, the analysis;
    - its launch gate is lower (the 4-bit 1.5B peaks at ~2.3 GiB; D153).
- **Done this afternoon, before any of the controls' data (code checks only):**
  - **Both random-weights copies are built:** the 1.5B (1.54B parameters, 2 shards, 2.9 GB) and the 7B (7.62B
    parameters, 8 shards, 15 GB). Tonight's chain finds them and moves on.
    - **The 7B's first attempt was stopped** at 12 of 15 GB of RAM: the maker held every shard until the end. It now
      writes each shard as soon as it fills; the grouping rule is unchanged, so the 1.5B copy is the same.
  - **A GPU smoke test** (2 kept 7B answers through the 1.5B copies; a code check, nothing saved or analyzed):
    - the random copy loads in 4-bit like the real one (196 4-bit linear modules each);
    - its states are finite and pass x2_hidden's logit check (gap 0.004 against 0.02);
    - their cosine with the real model's at layer 14 is −0.01.
  - **A plumbing dry run of the analysis** into the scratchpad: the real answers with a fast stand-in for the probe;
    the reproduction check fires as designed. No result was read from it.
  - **The full test suite** (CPU, 1 thread, nice 19): every test passes except D155's known wall-clock flake in
    `test_temp_guard.py`, which passes 3 of 3 alone.

### D161. Item 3 of D160: the X2 probe refit was refused by its gate; D160 had misread X2's thread settings. One retry with the documented settings, after tonight's GPU stage (2026-11-02, ~17:20 IST; recorded before the retry)
- **The first refit** (14:45–16:45 IST, 4 jobs, the default thread environment: loky gave each worker 24 / 4 = 6
  BLAS threads):
  - **The result:** 0.5397552 against X2's published 0.5396260, a gap of 1.3e-4 against D92's 1e-9. The gate refused
    it, and no per-answer value was kept (`results/followup/scale/x2_sft_probe_refit.json`).
  - **The selections are X2's in all five folds:** layer 1 with C 0.01 four times, layer 11 with C 0.001 once. The
    inner-CV AUROCs differ in the 5th decimal: a thread-count effect, as in D97.
- **The misreading:** D160 called X2's settings "4 jobs and the default thread environment". X2's analysis ran with
  `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4` and 4 jobs (`scripts/queue_x2_x3.sh:154`,
  `queue_x2_x3_reruns.sh:79`). D97 says the same: "4 jobs, 4 BLAS threads".
- **The retry, once:**
  - those settings, at nice 19, the GPU hidden;
  - it starts after tonight's GPU stage ends (D97: no multi-thread CPU job while the GPU is busy). It runs next to
    the analysis; thread counts, not load, set the numbers;
  - **the gate is unchanged:** exact reproduction within 1e-9;
  - **if it is refused again,** the SFT probe on the prefix stays missing, as D151 (2) says. There is no third
    setting.
- **After both finish:** the analysis is re-run. Its probes come from the cache, so this takes minutes, and it picks up
  the refit if it passed.
- **The outcome (18:40–19:11 IST):** the retry reproduces X2 exactly.
  - It gives 0.5396260151347361, the published value, with a gap of 0.0. The inner-CV AUROCs equal D92's to 7
    decimals.
  - Its 4,109 per-answer values are kept (`results/followup/scale/x2_sft_probe_refit.json`). The first attempt's
    record is beside it (`x2_sft_probe_refit_attempt1.json`).
  - The next analysis reads it for the SFT 1.5B's probe on the prefix.

## 2026-11-02 evening: the controls' data (your instructions at ~19:40 IST)

### D162. D160's controls ran. By D160's rules, scale moves direct P(yes), not the probe; both probes pass their random-weights control, but most of their signal is surface (2026-11-02, ~22:50 IST; after the analysis)
- **The 7B random-weights chain** (`MODEL=7b GPU_STAGES=hidden_random`):
  - the gate opened at 17:10;
  - its hidden states took 17:10–18:39 (18 temperature pauses);
  - the analysis took 18:39–20:42;
  - the 7B probe's refit reproduced D159's 0.7525589 exactly (D160's check).
- **The native 1.5B on Kaggle** (D160):
  - **The run:** 1,176 / 1,176 candidates in one clean session on the pinned commit `4039888`, at 6.2 s per problem
    (1.55 h).
  - **The ingest:** every check passed; the re-grade changed 0 rows.
  - **What was sampled:** answered 97.6%, truncated 1.0%, correct 41.4%; 766 kept (7B: 844); 599 kept by both.
- **The ingest order:** the report ingest ran into the scratchpad first, with the same code and every check. Tonight's
  7B analysis was still running and reads every model's samples, so it would have picked up partial 1.5B numbers. The
  repo ingest was the 1.5B chain's first stage.
- **The 1.5B chain** started at 20:42:49, after the 7B chain had exited and released its lock (a wrapper waited;
  never two GPU chains at once):
  - P(yes), the hidden states and the random-weights hidden states took 20:48–21:58 (20 temperature pauses);
  - the analysis took 21:58–22:42;
  - every stage committed and pushed.
- **The reading, D160's rules applied as written:**
  - **"The wall moves with scale": yes, through direct P(yes).** 7B − native 1.5B is +0.110 [+0.082, +0.138]
    unpaired, and +0.120 [+0.090, +0.150] paired on the 599 shared problems.
  - **The probe does not count:**
    - 7B − native 1.5B is −0.005 [−0.031, +0.020];
    - net of the random-weights control, (7B − random) − (1.5B − random) is +0.003 [−0.026, +0.032];
    - the 7B's own real − random is +0.088 [+0.068, +0.109], so the probe passes its control but shows no scale
      effect.
  - **The random-weights control:** both probes read more than the surface (7B +0.088, 1.5B +0.085). Random weights
    alone reach 0.664 / 0.673, above length (0.634 / 0.598).
  - **Native 1.5B − SFT 1.5B,** same prefix:
    - the probe +0.219 [+0.188, +0.249]: D159's 7B − SFT probe gap (+0.214) is format, training and decoding, not
      size;
    - P(yes) +0.047 [+0.014, +0.079].
  - **Self-consistency still leads at both sizes** (0.827 / 0.774), and 7B − 1.5B is +0.053 [+0.025, +0.081].
  - **P(yes) − length:** the 7B +0.048 [+0.027, +0.071]; the 1.5B −0.026 [−0.050, −0.002].
- **Nothing in D160's rules or code changed after its data.** The write-up is `docs/exploratory/followup_feasibility.md`
  §3.

## 2026-11-03 morning: cleanup (your instructions at ~06:40 IST)

### D163. The 7B's random-weights copy is deleted; the queue's PID and lock files are ignored (2026-11-03, ~06:45 IST)
- **Deleted:** `results/followup/scale/7b/random_init_model/`, D160's random-weights copy of the 7B (15.2 GB of bf16
  shards, gitignored).
  - **Why it is safe:** its only use was the `hidden_random` readout, which is done. The hidden states it produced
    (`hidden_random/`, local) and the probe values (`probe_hidden_random.json`, committed) stay, so the analysis
    re-runs without it.
  - **To rebuild it:** run `python -m physcal.eval.followup_scale random_init --model 7b` (CPU, ~10 min, seed 0 from
    `configs/followup.yaml`). Check the rebuild against content hash `773402619421ef14`, recorded in the committed
    `hidden_random_info.json`. The queue rebuilds it by itself if a `hidden_random` stage needs it.
  - **Kept:** the 1.5B's copy (2.9 GB). Deleting it was not asked for.
- **Free space after:**
  - **Windows, MLStorage:** unchanged at 21.6 GiB. Deleting inside WSL does not shrink the disk image, and the queue's
    disk guard reads only this figure.
  - **Inside the image:** ext4 use fell from 79.9 to 65.7 GiB, so 14.2 GiB is free for reuse. New WSL writes can grow
    about 16.6 GiB before the 82.3 GiB image grows again.
  - **To return the space to Windows:** compact the image (`wsl --shutdown`, then `Optimize-VHD` or diskpart's
    `compact vdisk`). Not done: it stops every WSL process.
- **The leftovers:** the follow-up Kaggle chain's `*.pid` and `queue.lock` are now gitignored, like every other queue's.
  - None of the PIDs is alive, and the lock is free.
- **One stale waiter was stopped:**
  - It waited for D161's X2 refit (started 2026-11-02 ~14:45 IST).
  - It never exited, because its `pgrep -f` pattern matched its own command line. It used no GPU and wrote nothing.
- **No GPU process is held,** and no queue lock is held.

## 2026-11-03 morning: a zero-GPU confound audit of D162 (your instructions at ~08:30 IST)

### D164. A confound audit of D162's table: truncation, the 599 shared problems by pass count, and an A5-style text classifier against the probe. Plan and analysis rules (2026-11-03, ~08:40 IST; recorded before any of its numbers exist)
- **Why:** three things besides size differ between D162's columns.
  - **Truncation:** 8.7% of the 7B's samples against 1.0%. A truncated sample is no answer, so it decides which wrong
    answers enter the pairs.
  - **The problem mix:** 844 against 766 kept problems, with different pass counts.
  - **The surface:** random weights alone reach 0.66–0.67, so a reader of the text alone may match the probe.
- **Status:** exploratory, under D151's rules and D148's freeze (nothing enters the paper).
  - CPU only, 1 thread (OMP / OPENBLAS / MKL = 1), nice 19. No GPU and no new model output.
- **Inputs, all unchanged from D162:**
  - each model's samples, re-graded by the D151 + D154 parser (`followup_scale_analysis.graded`);
  - the per-answer readouts: self-consistency, P(yes) (`yes.jsonl`), length, and the cross-fitted probe and its
    random-weights control. The probes come from the cached `probe_hidden*.json`, checked against their input
    hashes, and are not refit;
  - the SFT 1.5B's X2 rows on the same prefix, with D161's refit probe.
- **The metric and CIs:** X2's pairwise accuracy (ties 0.5), the mean over problems. Every CI is a bootstrap over
  problems (B = 10,000, seed 0), with D162's code (`x2_pairs.summarize`, `paired`, `unpaired`).
- **Facts checked before this plan (counts only, no readout):**
  - no truncated sample carries an answer in any model (7B 820 truncated samples, native 1.5B 96, SFT 5): the parser
    makes a truncation no answer (D151);
  - kept problems with ≥ 1 truncated sample: 7B 371 of 844, native 1.5B 55 of 766, SFT 0 of 504.
- **1. Truncation.**
  - **(a) As asked: truncated samples dropped.**
    - A truncated sample leaves its problem entirely: the answers (it was never one), the kept rule,
      self-consistency's denominator (now the problem's other untruncated samples) and the length readout.
    - Every row of D162's table, the paired controls and the SFT comparisons are recomputed.
    - **Expected, stated now:** identical to D162 in every digit. Truncated samples are already in no pair, and
      self-consistency's denominator is the same for every answer of a problem, so no ranking changes. Any difference
      is reported as a pipeline finding.
  - **(b) The informative variant: problems with any truncated sample dropped.**
    - Each model keeps its own kept problems without truncation: the 7B 473, the native 1.5B 711, the SFT 504.
    - This removes the selection D162 lists as a caveat: a problem whose runaway attempts were truncated keeps only its
      finished attempts.
    - Every row and comparison as in (a). The paired 7B − native column uses the shared problems with no truncated
      sample in either model.
  - **In both,** the per-answer confidences are D162's. Only the scored problems change.
- **2. The 599 problems kept by both.**
  - **Levels:** every readout for each model on the same 599, and 7B − native paired (D162's paired column, now with
    the levels beside it).
  - **The SFT 1.5B** on those of the 599 that it also keeps (the count reported), as context.
  - **By pass count:** c = the number of a model's 8 samples graded correct (1..7; a truncated or unparsed sample is not
    correct). Each model is stratified by its own c. Per stratum: n problems, and the pairwise accuracy with a
    bootstrap CI over the stratum's problems.
  - **One summary, added by this entry:** a pass-count-standardized 7B − native difference per readout.
    - The weights w_c are the pooled share of each c over both models' 1,198 (problem, model) units.
    - A model's standardized value is Σ_c w_c × its mean in stratum c.
    - The CI is a paired bootstrap over the 599 problems (B = 10,000, seed 0). A stratum that is empty in a replicate
      gives its weight to the others in proportion.
- **3. An A5-style text classifier against the probe.**
  - **Items:** the answers the probe reads (each model's answered samples of its kept problems: 6,133 / 6,000), in the
    probe's order. So the outer folds (5, GroupKFold by problem, as `crossfit_confidence`) are the probe's own folds.
  - **The text, A5's problem + answer + reasoning set:**
    - problem = the native user text (`followup_scale.native_text`);
    - answer = the content of the last `\boxed{}`;
    - reasoning = the completion before that box's `\boxed{`.
    - So the classifier sees what the probe's state can see: everything up to the box's closing brace. Text after the
      box is left out.
  - **Features, A5's** (`configs/nested_baselines_4bit.yaml`, `text`):
    - TF-IDF unigrams over whitespace tokens, case kept, min_df 3, sublinear tf;
    - one vocabulary per field, fitted on the training part only;
    - for the GBT, each field's 1,000 most frequent terms, dense;
    - built by `nested_baselines.Candidate`, not copied.
  - **Models, A5's p grid:** logistic regression with C ∈ {0.01, 0.1, 1, 10, 100}, and A5's 4 HistGradientBoosting
    classifiers (random_state 0).
  - **Selection, as the probe's:**
    - within each outer training part, an inner GroupKFold(5) by problem;
    - the candidate with the best pooled out-of-fold AUROC wins (ties: the earlier in the grid);
    - it is refit on the training part, and its P(correct) on the held-out answers is the confidence.
  - **Reported per model:**
    - the pairwise accuracy, overall and by depth;
    - **probe − text, paired: the primary quantity asked;**
    - text − random-weights probe, and text − length (paired);
    - 7B − native for the text classifier (unpaired);
    - the folds' choices.
  - **The length-matched comparison:**
    - a (correct, incorrect) pair is length-matched if the two completions' token counts (`n_tokens`, as the length
      readout) differ by ≤ 10% of the longer one;
    - per problem, the mean over its matched pairs. Problems with none drop out, and the headline is the mean over the
      rest;
    - every readout per model, with the numbers of pairs and problems. Length's own value on matched pairs shows how
      much length signal is left;
    - probe − text, paired over the problems with a matched pair; and 7B − native for each readout (unpaired).
  - **The text classifier also enters the tables of 1 and 2,** as a new row.
- **Reading (descriptive, no multiple-comparison correction):**
  - **"The probe reads more than the text"** if probe − text has a CI above 0, overall and length-matched.
  - **"A D162 conclusion is sensitive to truncation or to the problem mix"** if, in 1(b) or 2, a sign or a CI's
    exclusion of 0 changes for any of: 7B − native P(yes), 7B − native probe, the probe's net scale effect, or
    probe − random-weights.
- **Code and outputs:**
  - `physcal/eval/followup_audit.py`, with its settings in `configs/followup.yaml` (`audit`) and its tests in
    `tests/test_followup_audit.py`;
  - `results/followup/audit/`: `results.json`, `summary.md`, and the text classifier's per-answer values per model,
    under a hash of their inputs;
  - the write-up in `docs/exploratory/followup_feasibility.md` §3.
- **Cost (an estimate):** the text classifier is 2 models × 5 outer folds × (9 candidates × 5 inner folds + 1) fits,
  mostly GBT time. At one thread that is under an hour per model. The measured time is recorded with the results.
- **The outcome (2026-11-03, 08:44–09:57 IST; CPU only, 1 thread, nice 19; written after the analysis).**
  - **The run:**
    - D162's unchanged recompute reproduces `results/followup/scale/results.json` with a gap of exactly 0.0 (checked
      before the long fit, with the text readout stubbed, and again by `analyze`);
    - the text classifier took 16 min (native 1.5B) and 26 min (7B); the analysis 33 s;
    - outputs: `results/followup/audit/` (`results.json`, `summary.md`, `text_{7b,1.5b}.json`, the logs).
  - **One implementation detail, not a plan change:** in 1(a), a problem left with a single untruncated sample has no
    others for self-consistency, so it is left out of that readout (`followup_audit.with_others`). Such a problem
    cannot be kept, so no pair is affected. The first run stopped on it with a division by zero, before any number.
  - **1(a)** is identical to D162 in every digit, as stated above.
  - **The reading rules, applied as written:**
    - "the probe reads more than the text": **yes** at both sizes, on all pairs and length-matched;
    - "a D162 conclusion is sensitive": **2 of 20 checks flag**, both the probe's net scale effect,
      (probe − random) 7B − native. Its sign goes from +0.003 to −0.003 (the 599) and −0.001 (length-matched). Its
      CI contains 0 in D162 and in both variants, so the conclusion it carries (no net scale effect) does not change.
      The rule as written did not separate a sign change around 0 from a real one; that is recorded here, and the
      rule is not edited.
  - **The numbers and the write-up:** `docs/exploratory/followup_feasibility.md` §3, "The confound audit (D164)".

## 2026-11-03 midday: the round-6 review (your instructions at ~10:30 IST)

### D165. Round-6 review synthesis: R6-1 to R6-6 recorded (your decision, 2026-11-03; `REVIEW_SYNTHESIS_ROUND6_2026-11-03.md`, committed `e649ac6` before this entry)
- **Where both reviewers (Gemini, ChatGPT) agree:**
  1. Paper #1 goes to arXiv around 20 November. Its experiments stay frozen (D148).
  2. **"Fine-tuning removed the model's knowledge of its errors" is not supported.** The 0.758 → 0.539 probe gap may
     mean the information was:
     - (i) lost;
     - (ii) displaced to other tokens or layers;
     - (iii) present but not linearly readable;
     - (iv) never internal, only inferable from the text.
     The random-weights probe (0.66–0.67 on native traces) makes (iv) a serious candidate.
  3. **No scale-law claim from two models.** The wording is: "we observed no improvement in this probe metric between
     the two native sizes under these conditions".
  4. **Novelty:** hidden-vs-expressed knowledge, including pair ranking, is prior art (Gekhman et al., COLM 2025;
     Orgad et al., ICLR 2025). Paper #2's contribution must be the controlled effect of post-training, format and
     size on accessible answer-specific information, and whether it transfers into cheap verbalized confidence.
- **R6-1, paper #1's wording:** a paper D-entry, D166 in DECISIONS.md. It is the one exception to D151's rule that
  nothing from this track enters `paper/`: a single exploratory paragraph. FINDINGS.md and the paper's tables are
  unchanged.
- **R6-2, the zero-GPU audit:** this is D164 (`7872f87`), done. Against R6-2's list:
  - truncated samples dropped: yes, 1(a) and 1(b);
  - every row on the 599 problems mixed for both models: yes, item 2;
  - stratified by pass count: yes, with a standardized difference;
  - the A5-style text classifier and probe − text, paired: yes;
  - length-matched: yes;
  - **per-problem weighting:** D164's metric is already per-problem weighted (X2's: each problem weighs the same,
    whatever its number of pairs). D164 does not show the contrast with pair weighting. **It is added now, before it
    is computed:**
    - **the quantity:** the pair-weighted pairwise accuracy, where each pair weighs the same: Σ_p n_p · pa_p / Σ_p n_p;
    - **where:** per readout and model, on D162's kept sets and on the 599;
    - **the difference:** 7B − native, unpaired on the full sets and paired on the 599;
    - **the CI:** a bootstrap over problems, with the ratio recomputed in each replicate (B = 10,000, seed 0);
    - **the reading:** "the weighting matters" if, between the two weightings, 7B − native for P(yes) or the probe,
      or the probe's net scale effect, changes sign or CI side.
- **R6-3, R6-4, R6-5:** pre-registered together in D167, before any of their code is built.
- **R6-6, later, only after R6-2 to R6-5 have results:**
  - probe-to-verbal-confidence distillation, against direct supervised confidence and no training (pairwise accuracy,
    Brier, selective risk, held-out templates);
  - then one replication on a different model family;
  - paper #2's framing (representation change vs elicitation) is decided after R6-2 to R6-5.
  - Nothing of R6-6 is built now.
- **References to verify before citing** (the synthesis's list): Gekhman et al. (2503.15299), Orgad et al.
  (2410.02707), Ni et al. ReProbe (2511.06209), Kossen et al. (2406.15927), Lin, Hilton & Evans (2205.14334),
  Kadavath et al. (2207.05221), Tam et al. (2408.02442), Levinstein & Herrmann (2307.02477), Ni et al. (2408.09773).
- **R6-2's pair-weighting check, the outcome (2026-11-03, after it ran; CPU, 1 thread, 33 s):**
  - every number that D164 computed is unchanged in `results/followup/audit/results.json`; the new keys are
    `pair_weighted` and `weighting_sensitivity`;
  - 7B − native per pair: P(yes) +0.105 [+0.078, +0.133] (paired on the 599: +0.115); probe −0.012
    [−0.038, +0.013] (−0.013);
  - D165's rule flags one of six checks: the net scale effect on D162's sets, +0.003 → −0.001. Its CI contains 0
    both times, so no conclusion changes.
  - **R6-2 is complete.**

### D167. Pre-registration of R6-3 (a common candidate bank), R6-4 (a format × weights factorial) and R6-5 (probe controls), for paper #2. Recorded before any of their code, items or GPU output exists (2026-11-03, ~12:30 IST)
- **Status:** exploratory track, paper #2 (not paper #1; D148's freeze is unchanged). Every rule below is fixed now.
  - **Changes:** any later change gets its own D-entry, written before the data it affects.
  - **Correction:** none of the three is Holm-corrected. Each states its own primary quantity and reading.
- **Facts fixed before this entry** (counts and token lengths only, no readout; `results/followup/scale`, X2's tapes):
  - **the 599 problems kept by both Kaggle models:** 4,369 answered 7B samples (mean 998 tokens) and 4,683 answered
    native-1.5B samples (925 tokens). Truncated: 410 and 56;
  - **the problems kept by both the native 1.5B (Kaggle) and the SFT 1.5B (X2, the same 1,176 candidates):** 347,
    with 2,722 native traces (903 tokens) and 2,766 SFT terse completions (196 tokens), each with exactly one
    `<answer>…</answer>`;
  - **the prompts:** 225 tokens in the native frame, 971 in the SFT's (`v4_worked_example`, X2's prompt);
  - **R6-4's identical-candidate subset (rule below):** 93 problems, 323 candidates, 248 pairs.
- **What the audit (D164, D165) fixes here:**
  1. **R6-3's problems are the 599 that are mixed for both models** (D164 item 2). Every pair is then scored by both
     readers, so the pass-count mix is identical by construction and needs no standardization.
  2. **R6-3's primary pairs are within one source** (both answers by the same model). A text reader matches the
     random-weights probe at 0.66–0.67 (D164 §3), so style separates the sources, and the 7B is right more often
     (51% vs 41% per sample). Cross-source pairs would then reward a style detector. They are reported, as secondary.
  3. **Length-matched pairs (completion token counts within 10% of the longer, D164's rule) are a pre-specified
     secondary in all three.** Length's own readout falls to 0.51–0.54 there.
  4. **Per-problem weighting is primary everywhere** (X2's metric). Pair weighting is a secondary (D165: it does not
     change D162's reading).
  5. **Truncated samples have no box, so no probe position.** R6-3 reports them separately through P(yes) only
     (truncation covers 44% of the 7B's kept problems, D164 1(b)).
  6. **The surface reference is a text classifier.** D164's A5-style classifier equals the random-weights probe, so
     R6-5 adds a richer one.
- **Common to all three:**
  - **The metric:** X2's within-problem pairwise accuracy (ties and missing 0.5), the mean over problems.
  - **CIs:** a bootstrap over problems (B = 10,000, seed 0), paired whenever the same problems are compared.
  - **The probe:** `x2_analysis.crossfit_confidence` with `scale.probe`. That is 5 folds by `GroupKFold` over
    problems; the (layer, C ∈ {0.001, 0.01, 0.1, 1}) choice by inner grouped CV on the training folds only; all 29
    layers; refit, then score the held-out fold.
  - **The readers** are the local 4-bit copies, the same as every earlier readout. GPU readouts go through the
    follow-up queue (D157's guards: temperature, stall, free space, VRAM gate), batch 1 for hidden states.
  - **The CPU analysis** runs after the GPU stages, with the GPU idle: 4 jobs × 1 BLAS thread (as the cached
    probes, D159–D162).
  - **The candidates:** every item set is built on CPU, frozen as JSONL with a sha256 manifest, and committed
    before any GPU stage reads it.
- **R6-3. A common candidate bank: the 7B and the native 1.5B read the same answers.**
  - **The bank:** every answered sample of both models on the 599 problems (9,052 answers), with its source, label
    (correct within 1%) and the native frame (D151's prompt).
  - **The readers:** the 7B and the native 1.5B, each teacher-forced on every bank answer: its own (D162's readouts,
    reused) and the other model's (new).
  - **The readouts per reader:**
    - direct P(yes) (D151 / D157's verification turn; the log-odds);
    - the cross-fitted probe at the token holding the last box's closing brace, fit on the reader's states for the
      whole bank. Both readers use the same items in the same order, so the folds are identical;
    - reader-independent references: length, and D164's A5-style text classifier cross-fitted on the bank.
  - **Reused own-answer states and verdicts:** D162's, read by the same code and copies. A spot check re-extracts the
    first 16 own answers per reader; their states must match the stored ones within fp16 rounding (max relative
    difference ≤ 1e-2 at every layer), or the own answers are re-read too.
  - **No random-weights control here:** both readers see identical text, so the surface is held fixed. The 7B's
    random copy is also deleted (D163).
  - **The primary quantity:** reader 7B − reader 1.5B, paired over the 599, on the within-source pairs (the two
    sources' within-source pairs pooled per problem), for P(yes) and, separately, for the probe.
  - **Secondary:**
    - per source (7B answers only, 1.5B answers only), giving a 2 × 2 of reader × source and the interaction
      (own − other);
    - cross-source pairs; all pairs; length-matched within-source pairs; pair weighting;
    - **truncated samples, separately:** each reader's P(yes) on the 466 truncated samples, as within-problem
      pairwise accuracy of (correct answer, truncated) and (incorrect answer, truncated) pairs, and the mean log-odds
      per class.
  - **Reading:**
    - **"the verdict's size effect holds on identical inputs"** if P(yes) 7B − 1.5B has a CI above 0;
    - **for the probe:** "no reader difference on identical inputs" if its CI contains 0; "the 7B's states carry more
      / less" if its CI is above / below 0.
- **R6-4. A format × weights factorial at 1.5B.**
  - **The weights:** the native Qwen2.5-Math-1.5B-Instruct (D78's 4-bit copy) and the fine-tuned SFT model
    (`results/sft/qwen2.5-math-1.5b-instruct/merged`, 4-bit at load, X2's reader). They have the same architecture.
  - **The surface control:** D160's 1.5B random-weights copy, a third reader.
  - **The presentations, each with its own frame:**
    - **native:** the native 1.5B's chain-of-thought traces, with D151's CoT system prompt and the problem without its
      `<steps>` sentence;
    - **terse:** the SFT model's completions, with X2's prompt (`physcal.prompt.build_messages(text,
      "v4_worked_example")`) and the full problem text.
  - **The problems:** the 347 mixed for both the native and the SFT 1.5B. Every reader reads both banks.
  - **The positions,** one forward pass each, in the completion:
    - **P2 (primary), the first token after the final answer:** the token holding the closing brace of the last
      `\boxed{}` (native; D151's position), or the token holding the `<` of `</answer>` (terse);
    - **P1, the final answer's last token:** the token holding the last character of the box's content, or of the
      `<answer>` block's content. If P1 and P2 fall on one token, that is counted.
  - **The probes:** one per cell (3 readers × 2 presentations × 2 positions), cross-fitted on that cell's bank.
    Within a presentation, the readers share items and folds.
  - **The primary set, identical candidates in both presentations (93 problems, 323 candidates, 248 pairs):**
    - per problem, correct native traces are matched to correct terse traces in sample order (as many as the smaller
      count);
    - each incorrect native trace is matched to the first unmatched incorrect terse trace whose value agrees within
      1% (greedy, sample order);
    - a matched (native trace, terse trace) is one candidate, with one label;
    - a problem enters if it has ≥ 1 correct and ≥ 1 incorrect candidate. Pairs are (correct, incorrect) candidates;
    - each cell scores a candidate by its trace in that presentation (the out-of-fold probe value from the full-bank
      fit).
  - **Secondary:** the full banks (all pairs of each bank, 347 problems), P1, length-matched pairs, pair weighting.
  - **The effects (P2, the primary set; paired bootstrap over problems):**
    - the weights effect in each presentation: W_N = SFT/native − native/native, W_T = SFT/terse − native/terse;
    - the presentation effect for each weight set: P_native = native/terse − native/native, P_SFT = SFT/terse −
      SFT/native;
    - the main effects W = (W_N + W_T) / 2 and P = (P_native + P_SFT) / 2;
    - the interaction I = W_T − W_N;
    - the surface part of the presentation effect: random/terse − random/native.
  - **The interpretation table (fixed now):**

| W's CI excludes 0 | P's CI excludes 0 | reading |
|---|---|---|
| yes | no | **the weights factor dominates: fine-tuning is implicated** |
| no | yes | **the presentation factor dominates: format / readout** |
| yes | yes | both; the larger by \|W\| − \|P\|, with its bootstrap CI (descriptive) |
| no | no | neither factor is shown |

  - **Always reported:** the interaction ("the factors interact" if I's CI excludes 0), and P net of its surface
    part (descriptive).
  - **A plumbing anchor:** the native weights on the native bank at P2 re-extract D160's states for the same answers.
    The first 16 must match within fp16 rounding (as in R6-3).
- **R6-5. Probe controls on the native traces** (both Kaggle models, D162's kept answers, D162's probe as the reference).
  - **(a) A richer text classifier,** cross-fitted on the probe's folds with D164's selection:
    - **the features:**
      - TF-IDF word uni- and bigrams (whitespace tokens, min_df 3, sublinear tf) per field (problem, answer,
        reasoning; the text the probe can see, D164);
      - 15 hand features from the same text:
        - lengths: completion tokens; reasoning characters; lines;
        - numbers: all numbers; distinct numbers; the significant digits of the box's number; log10 of the box's
          |value| (0 → −12);
        - units: whether the box holds a unit; `\text{` occurrences;
        - expressions: `=` signs; LaTeX operators (`\frac \sqrt \cdot \times \left \pi`); `\boxed` occurrences;
        - hesitation markers: case-insensitive counts of "wait", "re-check", "recheck", "double-check", "let me",
          "however", "mistake", "actually", "verify", "again";
        - placement: the box's start over the completion's length; the problem's depth;
    - **the models:** A5's grid. The logistic models use TF-IDF plus the standardized hand features; the GBTs use
      each field's 1,000 most frequent terms plus the raw hand features;
    - **the quantity:** probe − text (rich), paired, on all pairs and length-matched.
  - **(b) The random-weights probe:** D160's, reported beside (no new run).
  - **(c) An answer-tokens-only probe:**
    - the same frame, with the completion replaced by `\boxed{<the box's content>}` alone;
    - the state at its closing brace; cross-fitted with the probe's protocol;
    - **the quantity:** full trace − answer only (paired);
    - beside it, D164's A5-style classifier on the problem and answer fields only (CPU).
  - **(d) A reasoning swap:**
    - **the donor:** for every kept answer a, the lowest-sample-index answered sample of the same problem with the
      opposite label;
    - **the swapped text:** the donor's completion up to its last `\boxed{`, then `\boxed{` + a's box content + `}`;
    - **the state:** at that closing brace;
    - **the score:** the probe fold model in which a's problem is held out, refit with the cached fold choices
      (layer, C) on the original states. The refit must reproduce the cached out-of-fold values within 1e-3 (max
      |Δp|); otherwise it is refit under the cached thread settings (4 jobs);
    - **reported:**
      - how much the probe moves: the median |p_swap − p_orig|, and the mean signed change for correct and for
        incorrect answers;
      - pairwise accuracy on swapped texts, labelled by a's (the box's) correctness. Below 0.5 means the probe follows
        the reasoning;
      - original − swapped (paired);
      - D164's text classifier on the swapped texts (its fold models, refit with its recorded choices) as the surface
        reference.
  - **Reading:**
    - **(a)** "the probe reads more than a rich text reader" if probe − rich text has a CI above 0;
    - **(c)** "the reasoning carries the signal" if full − answer only has a CI above 0; "the answer alone carries it"
      if answer only's CI is above 0.5 and full − answer only's contains 0;
    - **(d)** "the probe tracks the answer" if swapped pairwise accuracy has a CI above 0.5; "it tracks the
      reasoning" if below 0.5; otherwise neither.
- **The build (next, CPU only):**
  - `physcal/eval/followup_r6.py` (item sets, readout stages, analysis), `configs/followup.yaml` `r6`,
    `tests/test_followup_r6.py` and a queue script;
  - with GPU-time estimates, laptop against Kaggle. **Nothing runs on the GPU until you say so.**
- **After the build (2026-11-03, ~14:00 IST; CPU only, nothing on the GPU; appended, the plan above is unchanged):**
  - **The code:**
    - `physcal/eval/followup_r6.py`: item sets, readout stages, analysis, GPU estimate;
    - `scripts/queue_followup_r6.sh`: D157's guards and D158's VRAM gate per reader; **not run**;
    - `tests/test_followup_r6.py`: 17 tests, including the readout path on a tiny Qwen2 on CPU and the queue's dry
      run;
    - `configs/followup.yaml`: the `r6` block.
  - **The item sets are frozen** (`results/followup/r6/items/`, manifest with sha256). Their counts equal the facts
    above: bank 4,369 + 4,683, truncated 466, factorial 2,722 + 2,766, and identical candidates on 93 problems, 323
    candidates, 248 pairs. D162's kept-answer order is the probe's (asserted at the build).
  - **Storage, within the plan:** p1 is defined for R6-4 only, so only `fact_native` and `fact_terse` keep it. The
    other sets store p2 alone. States total ~7.9 GB, against ~13 GB with p1 everywhere.
  - **A real-data plumbing check (CPU):** the cached probes' five fold models, refit with their recorded (layer, C),
    reproduce the cached out-of-fold values exactly (max |Δp| = 0.0 for both models; the tolerance is 1e-3). So
    R6-5 (d) can score swapped states with them.
  - **One fix during the build, before any data:** hand feature 12 (`\boxed` occurrences) cut the text one character
    into the last box; its test found it.
  - **Laptop GPU estimate** (`results/followup/r6/gpu_estimate.json`): measured s / token at batch 1 with pauses
    included (D159, D162), applied to each set's real context lengths (the longest 3,576 tokens, under 4,096):

| | 1.5B readers | 7B reader | total | states on disk |
|---|---|---|---|---|
| R6-3 common bank | 0.80 h | 2.61 h | 3.41 h | 1.4 GB |
| R6-4 factorial | 1.19 h | – | 1.19 h | 2.9 GB |
| R6-5 controls | 0.51 h | 1.80 h | 2.32 h | 3.6 GB |
| **all** | 2.51 h | 4.40 h | **6.92 h** | **7.9 GB** |

  - **Kaggle instead (an estimate with assumptions, not measured):**
    - **The comparability rule:** R6-3 reuses D162's own-answer verdicts and states, and R6-5 scores swaps with fold
      models fit on D160's states. Both were read by the local 4-bit copies on the 4060. Read on a T4 (sm75 kernels),
      the new readouts would not be comparable unless the reused ones are re-read there too: about +4.6 h of
      laptop-equivalent work.
    - **The assumption:** one T4 runs 4-bit teacher forcing at 0.6–0.8× the 4060. Two T4s in parallel give 1.2–1.6×.
    - **So:** (6.9 + 4.6) / (1.2–1.6) ≈ 7–10 h of wall time. On top come a new Kaggle stack for teacher-forced
      readouts (transformers / bitsandbytes / Unsloth, not D157's vLLM), uploading the SFT model (3.1 GB) and the
      random copy (2.9 GB) as private datasets, and bringing ~11 GB of states back (7.9 GB new + 3.1 GB re-read). That is likely two 12 h
      sessions.
    - **Recommendation: the laptop,** one night (~7 GPU-h). Before launch, check free space: 14.2 GiB inside the
      image (D163), against 7.9 GB of states.
  - **CPU after the GPU (an estimate from D159–D162's probe times at 4 jobs; 7B 99 min / 6,133 answers, 1.5B
    29 min / 6,000):**
    - R6-3's two probes: ~3.2 h;
    - R6-4's 12 probes: ~2.7 h;
    - R6-5's answer-only probes: ~2.1 h;
    - the text classifiers: ~2–3 h;
    - **in all ≈ 10 h,** at nice 19 with the GPU idle.

## 2026-11-03 night: the R6 queue (your instructions at ~23:25 IST; GPU deadline 09:00 IST)

### D168. The R6 queue is launched with D157 / D158's guards. The 7B gate waits until the deadline, each stage reads the cap once, and the analysis caches every fit (2026-11-03, ~23:45 IST; after the first stage, before any result)
- **The launch:** `GPU_DEADLINE=09:00 CAP_WAIT_S=21600 HOST_TRIES=1 scripts/queue_followup_r6.sh`.
  - **The chain:** PID 503 at 23:31, relaunched as PID 2366 at 23:40 (below).
  - **The guards, unchanged:** temperature (D143), stall (D132), free space (D111), and D158's VRAM gate (7B 6.5 GiB,
    the 1.5B readers 3.5).
- **Before it:**
  - Windows held a steady ~1.05 GiB of VRAM, so the cap was 6.4 GiB, under the 7B's gate;
  - 21.6 GiB was free on MLStorage;
  - both extra readers were present (the SFT merge, D160's 1.5B random copy);
  - no other queue or GPU process was running.
- **Two settings, neither a guard:**
  1. **`CAP_WAIT_S=21600`:** the 7B gate waits until the deadline instead of 60 min, as D159 did (`CAP_WAIT_S=17000`).
     The 7B stages start after ~2.5 h of 1.5B stages, and Windows' share often drops at night (it was 802 MiB by
     23:36).
  2. **`HOST_TRIES=1`, after a restart:** the queue reads the cap before every stage. With 5 tries it re-read for up
     to 4 min whenever the cap was under 7.5 GiB. That is ~70 idle GPU minutes over 18 stages, enough to push the last
     7B stages past 09:00.
     - The chain was stopped between stages (no stage process alive) and relaunched with one reading per stage.
     - The cap formula and the gate are unchanged, and the 7B's gate loop re-reads every 60 s as before.
     - The finished spot stage was skipped in 2 s.
- **The first stage** (`spot_1.5b`, 16 own answers re-read by the 1.5B): 23:37, 26 s, cap 6.4 GiB.
  - **D167's spot check, run then (1 thread, seconds):** max |diff| / max |state| = 0.0. The states are bitwise D160's,
    so R6-3 reuses the stored ones as planned.
- **The analysis's robustness, an implementation change only (no rule):**
  - `followup_r6.analyze` now writes `results.json` after each experiment;
  - every probe and text-classifier fit is cached under the sha256 of its inputs (`results/followup/r6/cache/`), so a
    re-run after a failure keeps the finished fits;
  - the GPU-stage functions are byte-identical (checked by diff), and the module was swapped in by an atomic rename;
  - the tests: 18 pass (one new, for the cache).
- **The analysis starts only if every GPU stage finished:** 4 jobs × 1 BLAS thread, nice 19, the GPU hidden
  (`CPU_ENV`).

## 2026-11-04 midday: the analysis is paused for a shutdown (your instruction at ~11:38 IST)

### D169. D167's CPU analysis is stopped at the boundary between R6-4 and R6-5: R6-3 and R6-4 are done and saved, R6-5 is not started. The resume command (2026-11-04, ~11:50 IST)
- **The rule (your instruction):** let the running fit finish only if it ends within ~10 min, otherwise stop; the
  cached fits stay.
- **What happened:**
  - at 11:38 the running fit was R6-4's last cell (`fact_terse_random_1.5b_p1`, started 11:36). Its siblings took
    195–556 s;
  - it finished at 11:44 (483 s). R6-4's effects were computed and `results.json` was saved with R6-3 and R6-4;
  - R6-5 (7B) had just started its first fit, the rich text classifier, with nothing cached;
  - at 11:45 the chain (PID 2366) and the analysis's process group were stopped (SIGTERM): the analysis, its 4 loky
    workers and the resource tracker;
  - afterwards no R6, guard or queue process remained, no GPU process, and `/dev/shm` was empty.
- **Done and saved** (`results/followup/r6/results.json`, atomic writes; `results/followup/r6/cache/`, 15 fits):
  - **R6-3**, every quantity D167 lists:
    - the fits: the bank text classifier (1,300 s); the 7B reader's probe (9,778 s); the 1.5B reader's probe
      (3,043 s);
    - the spot checks; every pair filter; the reader × source table; pair weighting; the truncated readout.
  - **R6-4**, all 12 probe cells (3 readers × 2 presentations × 2 positions; 195–774 s each), with the effects, the
    interpretation table and the checks, on the primary set and the full banks.
- **Left:** R6-5 for the 7B and the 1.5B:
  - the rich text classifier, the answer-only probe, the answer-only text classifier and the reasoning swap. The swap
    needs no new fit beyond the five cached fold refits (exact, D167);
  - the estimate is ~4–5 h of CPU at the same settings (the 7B answer-only probe ~1.5 h, the rich classifiers ~1 h
    each);
  - **then:** the write-up and D167's readings (one D-entry), after which R6-6 is decided (D165).
- **The resume command** (no GPU stage):

  ```bash
  setsid nohup scripts/resume_followup_r6_analysis.sh >> results/queue_followup_r6/nohup.log 2>&1 < /dev/null &
  ```

  - It runs the queue's analysis stage alone, with the identical `CPU_ENV`: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= nice -n 19`. The probes' 4 jobs come from `configs/followup.yaml`
    (`scale.probe.n_jobs`), as in the first run.
  - It keeps the disk guard and the queue lock, and commits + pushes `results.json` and the cache at the end. The
    log is `results/queue_followup_r6/r6_analyze_resume.log`.
  - **What it reuses:** R6-3's and R6-4's fits come from the cache (checked by input hash), so those parts are rebuilt
    in minutes and the numbers are identical.
  - **Its dry run** (`DRY_RUN=1`) prints the same command as the first run's `start r6_analyze` line.
- **Also committed:** an ignore rule for the hidden-state parts (`results/followup/r6/states/*/*/`, 7.5 GB, local),
  which were untracked but not ignored. Their `*_info.json` files stay tracked.

## 2026-11-04 night: the resume is checked, fixed, and held (your instructions at ~21:30 and ~21:40 IST)

### D170. Before D169's resume: R6-3 and R6-4 are intact and R6-4 is complete, R6-5 would have crashed at its first fit (fixed, a rename), and the resume is held at your instruction. Recorded before any R6-5 number exists (2026-11-04, ~21:45 IST)
- **The state:** the analysis was not launched. Your pause (~21:40) came during the pre-launch checks, so no fit ran, no
  fit was interrupted, and `results.json` and `cache/` are unchanged. D169's resume command stands.
- **Integrity, read-only (your request):**
  - all 29 tracked files under `cache/`, `items/` and `results.json` equal their HEAD blobs;
  - all 15 cached fits: their `inputs_sha256`, recomputed from the states on disk, match. So the 7.5 GB of gitignored
    states are the fits' inputs, byte for byte;
  - R6-3 and R6-4, rebuilt from those fits through a `cached` that can neither fit nor write (30 s): 0 differences
    from `results.json`, compared exactly, every field but `seconds`.
- **R6-4 has no remaining cells.** All 12 are fitted and hash-matched, as D169 says. Only R6-5 is left.
- **The bug:** `analyze_controls` bound a local `cached` (the probe's JSON, since the D167 build). D168 then added calls
  to the module's `cached()` in the same function. Python treats the name as local throughout, so the first call raises
  `TypeError: 'dict' object is not callable`.
  - **Why it was not seen:** D169's stop came seconds after R6-5 started, before that line; no test ran the function.
  - **What the unfixed resume would have done:** rebuild R6-3 and R6-4, then stop at once as BLOCKED, with nothing lost.
- **The fix, an implementation change only (no rule, no number):** the local is renamed `probe_cache` (3 lines). Cache
  keys hash the inputs, not the code, so all 15 fits stay valid.
  - **A regression test:** `test_no_analysis_function_shadows_the_cache_helper`; it flags the pre-fix module. The R6
    tests: 19 pass.
- **A dry run of R6-5's whole path on the real inputs passed for both models (121 s + 99 s; no readout seen):**
  - **Run as they will be:** the states (all finite, the right shapes), the five fold refits and their gate, the
    text-swap models, the pairs, the bootstraps, the readings. `swap_refit_gap` is 0.0 for both (the gate is 1e-3).
  - **Reduced:** the three heavy fits ran through their real code on ~450 answers with a trimmed grid, padded with
    placeholders, and their outputs were serialized. So their full-size time and memory are not tested.
  - **Nothing was read:** it printed only counts, shapes, key names and the gate, and saved no readout. It wrote nothing
    under `results/` (a scratch script, not committed).
- **Two of the code's reading labels are looser or tighter than D167's text.** Fixed now, before the data: **the write-up
  applies D167's text**, not the labels, and the code is left as it is.
  - **(c):** the label says "the answer alone carries it" whenever answer-only's CI is above 0.5 and full − answer-only's
    is not above 0. D167 also needs that second CI to contain 0. If it lies below 0, D167's reading does not apply, and
    the write-up says so.
  - **(a):** the label needs the CI above 0 on both all pairs and length-matched. D167 names both as the quantity and
    gives one rule, so the write-up reports each separately and states the reading for each.
- **For the resume:**
  - **The command is D169's, unchanged,** with its `CPU_ENV`.
  - **Commits:** the script commits only at its end, so the session watching it must commit and push when `R6-5 7b` is
    saved. The script's own commit covers `R6-5 1.5b`.
  - **`results.json` is rewritten from the start:** for ~30 s it lacks R6-4. Git holds the saved copy.

## 2026-11-04 night: Kaggle-first (your instructions at ~23:10 IST)

### D171. From now on follow-up experiments are designed Kaggle-first: one self-contained notebook per experiment on "GPU T4 x2" that generates, reads out and analyzes, saves chunks so a session cutoff can resume, and returns only a small results zip. R6-6's draft plan follows it; nothing is built or run (2026-11-04, ~23:20 IST)
- **The decision is yours.** Below are the rules it implies, for your review. Numbered rules can be changed by a later
  entry.
- **Scope:**
  - **It applies to** every follow-up experiment designed from now on, starting with R6-6.
  - **It does not apply to** R6-3, R6-4 and R6-5: they finish locally, as launched (D169, D170).
  - **Paper #1** is untouched (D148).
  - **It replaces, going forward only:** D157's "the readouts stay local" and D167's recommendation of the laptop.
- **The rules:**
  1. **One notebook per experiment, all three parts in it:** generation, readouts, analysis. The laptop builds, tests on
     CPU and ingests the zip. There is no local GPU stage and no local analysis stage.
  2. **Self-contained means** that a run needs only Internet access, the repo at a pinned commit, pinned weights, and, for a
     resume, the previous version's output. This reading of "self-contained" is an assumption (the plan's decision 1).
  3. **Every stage resumes, not only sampling** (D157's item 10 applied throughout):
     - work is saved in atomic chunks, with a done marker and sha256 per stage;
     - a re-run skips what is finished, and refuses another commit or config;
     - a cutoff loses at most one chunk.
  4. **The notebook stops itself.** No chunk starts unless its projected end is before session start + 9.5 h (D157's
     rule, clean in both runs). Kaggle's 12 h limit is never the plan, because no source confirms that output survives it.
  5. **Two output tiers:**
     - **The resume tier** (`/kaggle/working`, under ~15 GB and ~300 files): chunks and states. It never enters the repo.
     - **The results zip** (under 50 MB, for the repo): `results.json`, a per-item score table, gzipped completions, the
       config, the environment, logs and a sha256 manifest.
     - **Hidden states, weights and adapters** do not come back. A rerun regenerates them.
     - **The completions come back** because every rollout is logged (CLAUDE.md); they are small (4.4 MB gzipped for
       1,176 problems × 8).
  6. **One stack per experiment.** Every number compared in an experiment comes from the same notebook. Baselines are
     measured there, not cited from local runs. Local numbers are context only (D167's comparability rule).
  7. **Pre-registration is unchanged.** A D-entry fixes the rules before any code. The notebook pins a commit that holds
     it. Stop rules read timing and format only, never correctness.
  8. **The ingest stays, as a check** (CPU, seconds): the zip's allow-list and manifest, a re-grade with the local parser,
     and the headline CIs recomputed from the per-item table, which must equal `results.json`.
  9. **Tests before any run:** the stage runner gets kill-and-resume tests on CPU, every stage runs on a tiny model on
     CPU, and a pin test ties the notebook's commit to the working tree.
  10. **Quota:** an experiment is sized for at most 2 sessions where possible. CPU work overlaps GPU work, because an idle
      GPU still spends the weekly quota.
- **What this costs, stated now:**
  - **Comparability:** R6-6's numbers cannot be read directly against the local R6-3 to R6-5 numbers (T4 fp16 against
    4-bit on the 4060).
  - **Quota:** ~30 h a week, floating, with R6-6a at about half of it.
  - **CPU:** probe fits run on a smaller machine than the laptop, and its size is not established (below).
  - **A new stack:** fp16 HF + PEFT on T4 has never run here.
  - **Your time:** each session and resume is started by hand.
- **What it gains:**
  - the readouts read the copy that sampled, which removes D157's difference 3;
  - no VRAM spill, heat or driver-reset handling (the WSL and laptop rules);
  - 7.5 GB-scale states stay off the laptop;
  - the laptop is free overnight.
- **What is not established (secondary sources, not Kaggle's docs; searched 2026-11-04):**
  - sessions of 12 h and ~20 GB of output are widely reported, and ~30 h a week varies;
  - sources conflict on 2 cores / 13 GB against 4 cores / ~30 GB, and vLLM logged 2.83 GiB of RAM available during the 7B
    load;
  - ~500 output files rests on forum reports;
  - **nothing confirms that `/kaggle/working` survives a crash or a timeout.** A smoke session settles this before any
    real run (the plan's §6).
- **R6-6's draft plan:** `docs/exploratory/r6_6_kaggle_plan.md`.
  - **It is a draft, not a pre-registration,** written before any R6-3, R6-4 or R6-5 number was read. D165's rule stands:
    the go decision comes after their readings.
  - **R6-6a** is distillation, **R6-6b** the replication, one notebook each.
  - **Two controls are proposed beyond D165's list:** a text-distilled arm and a shuffled-target arm. They need your
    decision, with the other decisions in the plan's §1.
  - **R6-6a's estimate:** ~13.5 h (12–18 h), 2 sessions. Only sampling's rate is measured; the rest are assumptions until
    the smoke session.
- **Nothing was built or run for this entry.** D165's "nothing of R6-6 is built" still holds.

## 2026-11-05 night: D167's readings (your instruction at ~23:08 IST on 2026-11-04)

### D172. D167's readings for R6-3, R6-4 and R6-5, and one pre-registered secondary the analysis had left out (2026-11-05, ~02:15 IST)
- **Status:** exploratory track, paper #2. D148's freeze holds. Nothing here is Holm-corrected (D167). The write-up is in
  `docs/exploratory/followup_feasibility.md`.
- **The run:** D169's resume, unchanged, from 23:08:29 to 01:49:34 IST (exit 0, `bb14f8f`), with D170's fix.
  - **Before it:** D170's checks (15 of 15 fits hash-match the states; R6-3 and R6-4 rebuild with 0 differences).
  - **After it:** R6-3 and R6-4 in `results.json` equal the saved copy, and R6-5's 7B block equals `4d1ae87`. No value is
    non-finite. The swap's refit gate gives max |Δp| = 0.0 for both models.
- **The readings, applied to D167's text:**

| experiment | quantity | result | reading |
|---|---|---|---|
| R6-3 | P(yes), reader 7B − 1.5B, within-source | +0.094 [+0.075, +0.112] | "the verdict's size effect holds on identical inputs" |
| R6-3 | probe, reader 7B − 1.5B, within-source | +0.022 [+0.011, +0.034] | "the 7B's states carry more" |
| R6-4 | W; P (P2, identical candidates) | +0.001 [−0.048, +0.052]; −0.029 [−0.148, +0.094] | "neither factor is shown" |
| R6-4 | interaction I | −0.047 [−0.138, +0.043] | "the factors interact" does not apply |
| R6-4 | P net of surface (descriptive) | −0.016 [−0.158, +0.126] | – |
| R6-5 (a), 7B | probe − rich text, all pairs; length-matched | +0.054 [+0.035, +0.072]; +0.060 [+0.027, +0.094] | "the probe reads more than a rich text reader" on both |
| R6-5 (a), 1.5B | the same | +0.059 [+0.040, +0.079]; +0.068 [+0.037, +0.101] | the same, on both |
| R6-5 (c), 7B | full − answer only, all pairs; length-matched | +0.098 [+0.074, +0.122]; +0.067 [+0.033, +0.101] | "the reasoning carries the signal" on both |
| R6-5 (c), 1.5B | the same | +0.111 [+0.088, +0.135]; +0.096 [+0.060, +0.130] | the same, on both |
| R6-5 (d), 7B | swapped pairwise accuracy, all pairs; length-matched | 0.480 [0.453, 0.506]; 0.467 [0.434, 0.5003] | neither, on both |
| R6-5 (d), 1.5B | the same | 0.391 [0.364, 0.420]; 0.403 [0.371, 0.436] | "it tracks the reasoning" on both |

- **D170's two notes on the code's labels changed nothing here:**
  - **(c):** the case D167 does not cover (answer only above 0.5, full − answer only below 0) did not occur.
  - **(a):** each variant is reported separately, and they agree.
- **The code's reading on R6-4's secondaries is not a reading.** D167 fixed the table for the primary set. The analysis
  also labels the full banks "the presentation factor dominates". There, the two banks hold different answers, so the
  label describes a secondary and P is not a clean presentation effect. W is clean, because within a presentation the
  readers read identical text.
- **A pre-registered secondary was left out by the analysis.**
  - **The fault:** D167 lists pair weighting among R6-4's secondaries. `analyze_factorial` never computed it, and I did
    not notice until reading the results.
  - **The fix:** `analyze_factorial_pair_weighted` (`python -m physcal.eval.followup_r6 r6_4_pair_weighted`), with two
    new tests. Each cell is `AU._pw`'s Σ n·pa / Σ n, as in R6-3, and each presentation uses its own pair counts on the
    full banks. Bootstrap: B = 10,000, seed 0.
  - **Its inputs:** the cached fits only. It writes `results/followup/r6/r6_4_pair_weighted.json`, so `results.json`
    stays as the registered run left it.
  - **The order is a deviation:** this was computed after R6-4's other numbers were read. D167 fixed the quantity, so no
    choice was open except the weights on the full banks, which follow the definition above.
  - **The checks:** the same fits, per-problem weighted, give back `results.json`'s cells exactly (a gate in the code). A
    separate plain-loop calculation gives the same primary cells and the same CI for P.
  - **The result:** no reading changes. Primary set (P2): W −0.004 [−0.047, +0.040], P −0.048 [−0.177, +0.080], "neither
    factor is shown". Full banks (P2): W −0.001 [−0.016, +0.014], P −0.239 [−0.277, −0.200].
- **Not computed: length-matched pairs on R6-4's 93-problem set.** D167 lists "length-matched pairs" without naming a
  set, and the analysis runs them on the full banks. On the matched set a candidate has two traces, so a pair would have
  to match in both presentations, leaving few of the 248 pairs. I did not add this.
- **What goes beyond the readings (interpretations, labelled as such in the write-up):**
  - R6-4's primary set keeps only wrong values both models reached, and that may be why every readout there is low.
  - The swap may partly measure the mismatch between box and reasoning: both classes fall in both models. No same-label
    swap was registered.
  - The answer-only probe's 0.65 has no random-weights reference, so it cannot be attributed to the weights yet.
- **R6-6 is not decided here.** The go decision stays yours (D165), with the six decisions in the plan's §1. Against the
  plan's §5:
  - **R6-5 (a):** C − D is the headline contrast.
  - **R6-5 (d):** R6-6a's student is the native 1.5B, so the row "tracks the reasoning" applies: add the swap to S5.
  - **R6-3:** a 7B student becomes an option, costed separately.
  - **R6-4:** "neither" has no row in §5. W ≈ 0 gives no reason to add the SFT weights as a second student.
- **The tests:** the full suite passes (814 passed, 1 skipped), with D170's fix and the two new tests.

## 2026-11-05 morning: the R6-6 decisions and two local controls (your instructions at ~09:00 IST)

### D173. Your six R6-6 decisions, and two controls for R6-5 at 1.5B, run locally as a recorded exception to D171. Plan and readings (2026-11-05, ~09:15 IST; recorded before any of their code or data exists)
- **Status:** exploratory track, paper #2. D148's freeze holds. Nothing here is Holm-corrected (D167).
- **Your decisions:**
  - **All six recommendations in `docs/exploratory/r6_6_kaggle_plan.md` §1 are accepted as written:**
    1. "self-contained" allows cloning the repo at a pinned commit and downloading pinned weights;
    2. the verbal confidence is P(yes) from D151's verification turn;
    3. held-out templates means both a problem question template (primary) and a verification-turn wording (secondary);
    4. arms D (text-distilled) and E (shuffled) are added;
    5. R6-6b's model is decided after R6-6a;
    6. one short Kaggle smoke session runs before R6-6a.
  - **R6-6a is pre-registered after these two controls,** using their readings (mapping below).
  - **The order after that:** build the R6-6a notebook and prepare the ~45-minute smoke test, with run instructions. **The
    full R6-6a is not run.**
- **The exception to D171:**
  - **It covers these two controls only.** Both need the local 4-bit states and the cached fold probes of R6-5, so a
    Kaggle version would have to re-read everything (D167's comparability rule).
  - D171 still governs every new experiment, including R6-6a and R6-6b.
  - Their numbers compare with R6-5's 1.5B numbers, not with anything R6-6a will measure.
  - **The 7B is not covered.** R6-5 (d)'s 7B reading ("neither") gets no control here.
- **What is known as this is written:** R6-5's numbers (D172), and item counts from the frozen answers. No state of either
  control exists.
  - R6-5 (d) at 1.5B: 0.391 [0.364, 0.420]; mean change −0.433 (correct boxes), −0.141 (incorrect boxes).
  - Of the 6,000 kept answers on 766 problems: 2,499 correct answers have another correct answer; of the 3,266 incorrect
    ones with a donor, 3,127 have a donor with a different value, 1,177 one with the same value, and 1,038 have both.
- **Why a plain same-label swap is not enough (found while counting):**
  - A correct box after another correct reasoning is nearly that donor's own text, so a pairwise accuracy on it says
    little.
  - **The clean contrast holds the reasoning fixed.** In a ← d (d's completion up to its last `\boxed{`, then a's box),
    only the box differs from d's own text. So e(a ← d) = p(a ← d) − p_orig(d) is the effect of replacing the box.
- **Control 1: the same-label swap at 1.5B** (set `swap_same_1.5b`, read by the local 1.5B, state at the closing brace):
  - **The texts, one per (a, kind), built exactly as R6-5 (d)'s:**
    - `correct`: a correct, d the lowest-sample other correct answer (a placebo: the value does not change);
    - `same_value`: a incorrect, d the lowest-sample other incorrect answer whose value is within 1% of a's (a placebo);
    - `diff_value`: a incorrect, d the lowest-sample other incorrect answer whose value differs by more than 1% (**the
      mismatch: the box disagrees with the reasoning, and neither is right, so the label does not change**).
    - 6,803 texts by the counts above.
  - **The score:** R6-5 (d)'s, unchanged: the probe fold model in which the problem is held out, refit with the cached
    (layer, C). **Two gates:** the refit reproduces the cached out-of-fold values within 1e-3, and R6-5's
    `swap_movement` for 1.5B is reproduced exactly.
  - **(e), the primary quantity: the mismatch cost M.** Over incorrect answers with both kinds:
    M_a = e(a ← d_diff) − e(a ← d_same); the mean per problem, then over problems; bootstrap over problems, B = 10,000,
    seed 0.
    - Beside it: the mean e per kind on all its texts, and M for D164's text classifier (its fold models, as in R6-5 (d))
      as the surface reference.
  - **(f), a secondary: the box with the reasoning fixed.**
    - Per problem, d0 is the lowest-sample incorrect answer. The pairs: a correct box after d0 (R6-5's swap texts)
      against a wrong box of a different value after d0 (`diff_value` with donor d0).
    - Both boxes disagree with one identical reasoning, so only the box's correctness differs.
    - Pairwise accuracy with X2's scoring (ties 0.5), per problem, CI as above.
  - **(g), descriptive:** the pairwise accuracy of `correct` against `diff_value` texts, with the original and R6-5's swap
    on the same pairs; and R6-5's mean changes split as p(a ← d) − p_orig(a) = e(a ← d) + [p_orig(d) − p_orig(a)].
  - **Readings:**
    - **(e):** "a mismatch alone lowers the probe" if M's CI is below 0; "no mismatch cost is shown" if it contains 0. Above
      0: no reading, reported and diagnosed.
    - **(f):** "with the reasoning fixed, the probe prefers the correct box" if the CI is above 0.5; "it prefers the wrong
      box" if below; otherwise neither.
  - **What this cannot do:** D167's reading of (d) stands as recorded. Both sides of an R6-5 pair are mismatched, so a
    uniform penalty cancels in pairwise accuracy. (e) only tests D172's interpretation that "both classes fall" is a
    penalty for the disagreement.
- **Control 2: the random-weights answer-only probe at 1.5B:**
  - **The states:** the frozen `answer_only_1.5b` set read by `random_1.5b` (D160's copy, 4-bit at load, as in R6-4), at p2.
  - **The fit:** `crossfit_items`, R6-5 (c)'s protocol (folds by problem, layer and C by inner CV).
  - **(h), the quantity:** trained answer-only − random answer-only, paired. All pairs is the reading's set;
    length-matched is reported separately.
  - **Beside it:** random answer-only's pairwise accuracy, and random full trace (D160's cached probe) − random answer-only.
  - **Reading:**
    - "the trained weights add to the box alone" if the CI is above 0;
    - "the box's signal is surface" if the CI contains 0 and random answer-only's CI is above 0.5;
    - otherwise no reading, reported.
- **What each reading changes in R6-6a (fixed now). Neither control can change its arms or its primary metric:**
  - **(e) "a mismatch alone lowers the probe":** S5's swap test on the students carries two sets, opposite-label and
    `diff_value`. **"No mismatch cost is shown":** opposite-label only.
  - **(h) "the trained weights add to the box alone":** R6-6a adds an answer-only probe as a reference (one short pass in
    S2). **"The box's signal is surface":** no answer-only reference; arm D and the rich-text reference cover the surface.
  - **(f):** goes to the write-up. It changes nothing in R6-6a.
- **The run:** two GPU stages under D157 / D158's guards, then the CPU analysis at D97's settings with the GPU idle. From
  R6-5's measured 1.5B stages (1,204 s and 440 s for 6,000 items): ~23 min and ~7 min of GPU, then ~30 min of CPU.
- **The code (next):**
  - `physcal/eval/followup_r6_controls.py`, tests, and `scripts/queue_r6_controls.sh`;
  - a second manifest for the new set, so D167's frozen manifest and `results.json` are not touched;
  - the results go to `results/followup/r6/controls_d173.json`.
- **Nothing of R6-6 is built yet.** Its pre-registration follows the readings, and its code follows that.

### D174. D173's readings: a mismatch alone lowers the probe, the probe prefers the correct box with the reasoning fixed, and the trained weights add to the box alone. Post-hoc: the mismatch cost looks like a copy cost, and the state at the brace sits on a different token in swapped texts (2026-11-05, ~10:40 IST)
- **Status:** exploratory track, paper #2. Nothing here is Holm-corrected. 1.5B only, local 4-bit copies, so these numbers
  compare with R6-5's 1.5B numbers and with nothing R6-6a will measure.
- **The run:** `scripts/queue_r6_controls.sh`, 09:06 to 10:12 IST, exit 0 (`adea2f9`).
  - **GPU:** 1,314 s (six 30-s temperature pauses, 87–89 °C) and 404 s. **CPU:** 1,591 s, of which the random-weights fit
    took 1,515 s.
  - **Both gates passed:** the refit reproduces the cached out-of-fold values exactly (max |Δp| = 0.0), and R6-5's
    `swap_movement` for 1.5B is reproduced exactly.
- **The readings, applied to D173's text:**

| quantity | result | reading |
|---|---|---|
| (e) mismatch cost M, 360 problems | −0.099 [−0.124, −0.075] | "a mismatch alone lowers the probe" |
| (f) correct box against wrong box, reasoning fixed, 648 problems | 0.608 [0.581, 0.635] | "with the reasoning fixed, the probe prefers the correct box" |
| (h) trained − random answer-only, all pairs, 766 problems | +0.113 [+0.082, +0.144] | "the trained weights add to the box alone" |

- **Beside them:**
  - **(e):** D164's text classifier gives M = −0.001 [−0.003, +0.000], so that surface reader sees no cost.
  - **(g), descriptive, on 489 problems:** `correct` against `diff_value` texts 0.912 [0.896, 0.927]; the original texts
    0.760 [0.741, 0.779]; R6-5's swap 0.402 [0.368, 0.438].
  - **(g), R6-5's mean changes split exactly:**

| | box effect | reasoning effect | R6-5's change |
|---|---|---|---|
| correct boxes | −0.178 | −0.256 | −0.433 |
| incorrect boxes | −0.363 | +0.222 | −0.141 |

  - **(h):** random answer-only 0.533 [0.510, 0.557], trained 0.647 [0.624, 0.668]; length-matched, the difference is
    +0.097 [+0.056, +0.138]. Random full trace − random answer-only: +0.139 [+0.109, +0.169].
- **What this settles:**
  - **D172's interpretation holds.** The incorrect boxes do rise with the correct reasoning (+0.222). The box's own
    disagreement (−0.363) outweighs it, so "both classes fall".
  - **D167's reading of R6-5 (d) at 1.5B stands.** Both sides of its pairs are mismatched, as D173 said.
  - **R6-5 (c)'s open question closes at 1.5B.** The 0.65 of the answer-only probe is mostly the trained weights, not the
    box's surface. The random-weights surface sits in the reasoning text (0.673), not in the box (0.533).
- **Post-hoc diagnostics (not registered, not readings; `scripts/diag_d173.py`, `results/followup/r6/controls_d173_diagnostics.json`).**
  The placebos were meant to stay near 0, and they did not: mean e is −0.119 (`correct`) and −0.071 (`same_value`). So I
  looked before writing this:
  - **It is the string, not the value.** Where a's box is the same string as d's, mean e is −0.006 and +0.009. Where the
    string differs but the value is within 1%, it is −0.180 and −0.140, close to `diff_value`'s −0.197.
  - **So the cost looks like a copy cost.** The probe reacts to a box that is not the reasoning's own string, by about as
    much for 287 against 289 as for a different wrong value. This is an interpretation.
  - **M is conservative.** With only identical-string placebos, M = −0.172 [−0.205, −0.140] (182 problems, 452 answers). In
    log-odds, M = −1.11 [−1.31, −0.91]. The same-value donors also start higher (0.38 against 0.30), so in p they have
    more room to fall.
- **A property of the texts I had not noticed, in R6-5 (c) and (d) as well as here:**
  - **The fault:** a swapped or answer-only text ends at `}`, but in 437 of 800 checked originals the brace shares a token
    with what follows (`}\`, `}\n`, `}.`). So the state "at the brace" is on a different token, and in the original it
    has seen one more character.
  - **Its size:** on identical strings with a plain `}` token, e is 0 up to numerics (n = 655, mean −8e-7, max 0.026).
    With a merged token it is noise of mean |e| 0.105 and mean −0.0001 (n = 760).
  - **What it changes:** no reading. M subtracts two swapped texts, and the mean is 0. It does add noise to (d), and no
    number recorded so far is changed by it.
  - **R6-6a's fix:** every swap comparison is made against the donor's own text cut at its box (D175).
- **What changes in R6-6a, by D173's mapping:**
  - **(e):** S5's swap test carries both sets, opposite-label and `diff_value`.
  - **(h):** an answer-only probe joins the references (one short pass in S2).
  - **(f):** the write-up only.
- **Limits:**
  - (f) has no random-weights reference of its own. The box's surface at random weights (0.533) suggests little, but
    that is a different quantity.
  - The 7B got neither control.
  - The placebo's failure was found after the data. The registered M stands as the reading, and the identical-string M
    is a diagnostic.

## 2026-11-05 midday: R6-6a's pre-registration

### D175. Pre-registration of R6-6a: can what a probe reads be moved into a cheap verbal confidence, beyond what direct labels and surface text give? Recorded before any of its code, pool or Kaggle output exists (2026-11-05, ~10:55 IST)
- **Status:** exploratory track, paper #2. D148's freeze holds. Nothing is Holm-corrected (D167). D171's rules apply. Your
  decisions are in D173, and D174's readings set the additions marked below. **Nothing here runs without your go.**
- **The model:** `Qwen/Qwen2.5-Math-1.5B-Instruct` at D160's pinned revision, fp16, on "GPU T4 x2", sampled with D157's
  decoding (vLLM, k = 8, 3,072 new tokens, the same seed rule on this pool's indices).
- **The pool (fresh; built and frozen on the laptop, CPU):**
  - 2,400 problems, generator 1.1.0, split train. Problem i is cell i mod 12 (kinematics, dynamics, energy × depths 1–4),
    seed 70,000,000 + i.
  - The build must show that no seed and no problem text is shared with X2's pool or any frozen set in `data/eval_sets`.
  - **Splits, fixed before sampling, by problem:**
    - **held-out template:** every problem whose question sentence is "Determine {sym}, the {desc}…" (~800). Test only;
    - the rest, permuted with seed 0: the first 1,000 **train**, the next 120 **validation**, all others **ID test**
      (~480).
- **Answers:** a sample is answered if D154's grade gives a value and it is not truncated. Truncated samples are counted
  per split, not read. All arms score the same frozen answers, so every comparison is paired.
- **The verbal confidence:** the yes-vs-no log-odds at D151's verification turn (X4's reading: system, problem, the
  completion, the verification text W0, the generation prompt, "Verdict:"). P(yes) is its sigmoid. The head is float32.
- **The arms:**

| arm | training target | seeds |
|---|---|---|
| A: no training | none: the base model's P(yes) | – |
| B: direct supervised | the correct / incorrect label | 0, 1, 2 |
| C: probe-distilled | the out-of-fold probe P(correct) | 0, 1, 2 |
| D: text-distilled | the out-of-fold rich-text P(correct) | 0, 1, 2 |
| E: shuffled | C's targets shuffled within each problem (seed 0) | 0 |

- **Training, identical in every arm, with no tuning:**
  - **Items:** the first 6,000 answered train samples in pool order (all of them if fewer).
  - **Loss:** binary cross-entropy of the log-odds against the target.
  - **Adapter:** LoRA r = 16, α = 32, dropout 0, on q, k, v, o, gate, up, down. Adapter weights fp32, the base fp16 and
    frozen, fp16 autocast with a gradient scaler.
  - **Optimizer:** AdamW, learning rate 1e-4, no weight decay, 5% linear warmup then linear decay, gradient norm clipped at
    1.0, batch 16, one epoch. A seed sets the LoRA init and the item order, so arms with one seed share an order.
- **The teachers (CPU, on train problems only; test problems never enter a target):**
  - **Probe:** logistic regression on every layer's state at the box's closing brace. 5 outer folds by problem, (layer, C)
    by inner CV, as `x2_analysis.crossfit_confidence`.
  - **Rich text:** `followup_r6.crossfit_rich`, D167's features and grid.
  - **For the references:** each is refit on all train answers, with its selection redone there, and applied to validation
    and test.
- **The metrics:**
  - **Primary:** X2's within-problem pairwise accuracy on the ID test, on problems with both classes. Per problem; an arm
    with seeds is the mean of its seeds' per-problem values; paired bootstrap over problems, B = 10,000, seed 0.
  - **Secondary, on all answered test items:** Brier, and selective risk (AURC: the mean over coverage of the error rate
    among the most confident answers), with the same bootstrap.
  - **Also:** all of the above on the held-out template set; the spread across seeds; truncated counts.
- **The readings (ID test, pairwise accuracy):**

| contrast | CI above 0 | CI contains 0 | CI below 0 |
|---|---|---|---|
| B − A | direct supervision helps | labels do not help | labels hurt |
| C − A | distillation helps | distillation does not help | distillation hurts |
| C − B | the probe's target adds to labels | labels are enough | labels beat the probe's target |
| **C − D (headline, R6-5 (a))** | it moves more than surface | distillation is surface transfer | the text target moves more |
| E − A (a check) | **the setup leaks: stop and diagnose** | as expected | as expected: shuffled targets cost the verdict's own signal |

  - B is trained on the truth and C on a noisy estimate of it, so C − B at or below 0 is a plausible, valid result.
  - The same table is reported on the held-out template set, as a second column of readings.
- **The references, not arms, on the same test items:** the probe and the rich-text classifier themselves; **the
  answer-only probe (added by D174 (h)):** D167's answer-only text, one short pass in S2, selected and fit on all train
  answers; self-consistency at k = 8 (the cost reference); length.
- **Held-out wording (decision 3, secondary):** arms A, B, C, D, seed 0, on the ID test, with the verification text W1:
  > Please review the work you just did. Does the number in your \boxed{} agree with the true answer to within 1%? Finish
  > with a single line: "Verdict: yes" if it does, or "Verdict: no" if it does not.

  Reported as pairwise accuracy and the same contrasts, with no reading of their own.
- **The swap test (secondary; both sets added by D174 (e)):** arms A, B, C, D, seed 0, W0, on the first 100 ID-test
  problems with both classes, in pool order:
  - `opposite`: R6-5 (d)'s text for each answer;
  - `diff_value`: D173's text, for incorrect answers with such a donor;
  - `own_cut`: each answer's own completion cut after its box. **D174's fix:** swaps are compared with this, not the
    original, so the tail and the brace's token match on both sides.
  - **Reported:** pairwise accuracy on `opposite`, labelled by the box, with D167's three-way wording (below 0.5 "tracks
    the reasoning", above "tracks the answer", otherwise neither); the box effect p(a ← d) − p(d cut) for `diff_value`.
- **Stop rules (D171: timing and format only, never correctness):**
  - D157's gate: at least 50% answered on the first timed candidates;
  - a non-finite loss, state or logit stops that stage;
  - no chunk starts unless its projected end is before session start + 9.5 h.
- **Validation is a report, not a gate.** Each run's cross-entropy against the labels on validation answers is recorded. A
  run above 1.0 (worse than any constant guess) is labelled diverged: it stays in, and the readings are also given
  without it. No run is retrained on this.
- **The stages (assumed rates are replaced by the smoke session's):**

| stage | what | GPU | hours |
|---|---|---|---|
| S0 | venv, repo, weights | – | 0.1 (D160: 2 min) |
| S1 | sampling, 2,400 × 8, vLLM | both | 4.2 (measured, D160) |
| S2 | P(yes) and states in one pass, ~18,700 answers; the short answer-only pass | both | 1.8 |
| S3 | teachers and references, CPU, during arm B | – | overlapped |
| S4 | 10 LoRA runs, 2 at a time | both | 4.2 |
| S5 | test scoring, W1, swaps | both | 3.8 |
| S6 | analysis, results zip | – | 0.2 |
| | **in all** | | **~14 (12–19), 2 sessions** |

- **What comes back (D171's zip):** `results.json`, a per-answer score table, gzipped completions, the config, the
  environment, logs, timings, a sha256 manifest. States and adapters stay in the resume tier.
- **The smoke session (decision 6), ~45 minutes, before any of this:**
  - **Its own 48 problems** (seed 70,100,000 + i, disjoint from the pool), so no R6-6a problem is touched before the run.
  - **It runs every stage's real code** on them, then answers the three open questions: does output survive a crash, what
    is the machine, and what are the real rates (fp16 HF + PEFT LoRA on T4, without overflow).
  - **What it may change:** rates, batch sizes, chunk sizes and token budgets. Anything else is a new D-entry before data.
  - **If the measured rates need more than 2 sessions,** that goes to you before the run.
  - Its numbers on 48 problems are plumbing, never results.
- **The build (next, CPU):** `physcal/eval/kaggle_stages.py`, `physcal/eval/r6_6a.py`, `configs/r6_6a.yaml`, the frozen pools,
  `notebooks/kaggle_r6_6a.ipynb`, `notebooks/kaggle_r6_6a_smoke.ipynb`, and tests: kill-and-resume, a tiny Qwen through
  every stage on CPU, and a pin test.
- **R6-6b is not decided here** (decision 5).

### D176. R6-6a is built and the smoke session is ready to run; nothing has run on Kaggle. A local pre-flight suggests D175's S4 estimate is too low (2026-11-05, ~12:00 IST)
- **Status:** exploratory track, paper #2. D175's rules are unchanged. **The smoke session and R6-6a both wait for your go.**
- **What is built:**
  - `physcal/eval/kaggle_stages.py`: D171's shared stage runner (atomic chunks, ledgers with sha256, done markers, pins,
    the self-deadline, restore, the zip and its checks);
  - `physcal/eval/r6_6a.py`, `configs/r6_6a.yaml`, `configs/r6_6a_smoke.yaml`: the pools, every stage, the driver, the
    rates report and the ingest. Sampling reuses `followup_kaggle.sample` unchanged, under a config overlay;
  - `notebooks/kaggle_r6_6a.ipynb` and `notebooks/kaggle_r6_6a_smoke.ipynb`, from `scripts/make_r6_6a_notebooks.py`;
  - the run steps: `docs/exploratory/kaggle_instructions.md` §9.
- **The pools are frozen.** No seed or text is shared with X2's pool or any frozen set (checked at the build and by a test).
  The splits are 1,000 train, 120 validation, 502 ID test and 778 held-out template (D175 said ~480 and ~800).
- **The tests (D171 rule 9):**
  - a real SIGKILL in the middle of a chunk, then a resume;
  - a tiny Qwen through every stage on CPU, including a deadline stop and the next session's resume;
  - the zip through the ingest, which refuses results that do not follow from their tables;
  - a pin test: the notebooks' commit holds the code, the configs, the pools and D175.
  - **What they do not cover:** vLLM, two real GPUs, and Kaggle itself. The smoke session does.
- **Two faults the tests and a re-read caught before any run:**
  - the full run's chunk estimates would have stopped S4 from ever starting inside the smoke's 45-minute deadline;
  - a smoke stopped before S6 would never have written its rates. The report now runs whatever the session reached.
- **A deviation from D171, for your review.** D171 says the laptop tests on CPU. I ran one local GPU pre-flight of ~5 min, a
  build check rather than an experiment:
  - **Why:** the tiny tests cannot exercise CUDA fp16, and a bug there would waste the smoke session.
  - **What:** X2's merged SFT weights as a stand-in for the architecture, on fake ~1,900-token completions of the smoke
    pool, through S2, S4 and S5's code. Nothing was kept, and no real sample or R6-6a problem's answer was read.
  - **The result:** no fault. Log-odds and states are finite (max |state| 158), the adapter is fp32 (74 MB), no scaler step
    was skipped, and the peak was 4.0 GiB.
- **The warning from that pre-flight.** The 4060 took 32 s / step (16 answers of ~1,900 tokens) and 0.47 s / answer in S2.
  - **Real lengths:** ~1,245 tokens per reading context (D160's 1.5B answers), and 97.6% answered.
  - **If a T4 matched the 4060,** S4 would be ~2.2 h per run and ~10.9 h for the 10 runs on 2 GPUs, against D175's 4.2 h. That
    gives ~21 h in all, more than 2 sessions (19 h).
  - **This is a rough projection.** A T4's speed here is unknown, the pre-flight used small micro-batches, and the laptop
    runs hot. The smoke session measures it. D175 already sends this to you if it holds.
  - **The size of S5 was also low:** ~133,000 items, not ~87,000, because 97.6% of samples are answered. The smoke's
    projection uses the new count.
- **Also noted from the pre-flight:** two Adam steps at 1e-4 moved the log-odds by up to 3.3 on these fake texts, with one
  warmup step. The full run has 19 warmup steps, and the smoke's ~11-step runs have one, so a smoke loss curve will look
  rougher than the real one. The learning rate is fixed by D175. Changing it needs a D-entry before data, for every arm.
- **The tests:** the full suite passes (873 passed, 1 skipped). One small change came after it: `spawn` now clears a finished
  stage's process group, so a leftover vLLM worker cannot hold a GPU. The five suites that touch it were re-run and pass
  (95 passed), including the new test for it.
- **The notebooks pin `5f685c71e59b`,** which is on GitHub and holds D175.

## 2026-11-05 afternoon: the smoke session is back (your instructions at ~16:25 IST)

### D177. The smoke session's answers: output survives a Python exception, version 2 resumed from version 1, and the machine is 4 cores / 31 GiB. At the measured rates R6-6a does not fit two sessions: ~24 h of work, 4 sessions as built, 3 with a scheduler fix (2026-11-05, ~16:45 IST)
- **Status:** exploratory track, paper #2. These are rates and plumbing on the smoke's own 48 problems, never results. You
  accepted D175's choices. **The size of R6-6a is your decision (D175), and nothing below is applied yet.**
- **What ran:** `notebooks/kaggle_r6_6a_smoke.ipynb` at `5f685c71e59b`, twice.
  - **Version 1:** started 05:34:04 UTC (11:04 IST); its zip was written 31.0 min later. Every stage exited 0.
  - **Version 2:** started 09:41:43 UTC (15:11 IST); its zip was written 5.0 min later.
  - The zips are `~/kaggle_downloads/r6_6a_smoke_v1.zip` and `_v2.zip`, byte-identical to your downloads. They are ingested to
    `results/followup/r6_6a/smoke/v1` and `v2`.
- **A fault in my ingest, found by this ingest.**
  - **What happened:** both zips were refused with "results.json does not follow from its tables".
  - **The cause:** two values, both in one secondary smoke number, differed by 4e-17. Every other value was identical. That is
    the last bit of a float, consistent with Kaggle's numpy 2.3.5 against the laptop's 2.4.6. D171 rule 8 said "must equal", which
    cannot hold across machines, so the full run would have failed the same way.
  - **The fix:** floats must agree within `analysis.recompute_atol` = 1e-12. NaN agrees with NaN. Strings, integers, keys and
    lengths must still be equal. A difference of 1e-9 is refused (a test). This relaxes D171 rule 8 from exact to 1e-12.
  - **After it:** both zips pass: 384 completions re-grade the same, and every number recomputes.
- **What survived version 1's exception: everything.**
  - Version 2 restored 472 files from version 1's attached output, including all 400 of the file-count test.
  - All 46 chunks passed their sha256 check.
  - **It is the state after the last step before the crash.** Version 2's copy of version 1's log holds the `pack` line, which
    is absent from version 1's own zip. That line's sha256 (`88397fdb…`) is the zip you downloaded.
  - So ~470 files came through. D171's ~500-file worry does not apply at R6-6a's ~200.
- **What survived version 2's hard kill: the zip, at least.** It holds `survival.json`, so it was written in version 2 and reached
  you. **The zips cannot show whether the rest of `/kaggle/working` came through.** No version 3 ran.
- **What neither zip can show: that the crashes happened.** The crash cell runs after the zip is written, and it does not log.
  The code leaves no other path, but the versions' Error badges are the direct evidence.
- **Version 2 resumed from version 1, for a finished run:**
  - the same pins were accepted, and `{"chunks_checked": 46}`;
  - `exit_codes` is empty: no stage was launched, and the log has no chunk lines;
  - S6 re-ran and `results.json`, `scores` and `rates.json` are byte-identical.
  - **Not tested on Kaggle: resuming a half-finished stage.** The CPU tests cover it. R6-6a's first resume will be the first real
    one.
- **The machine (both versions):** 4 × Xeon 2.0 GHz, 31 GiB (cgroup limit 30 GiB), no swap, 2 × T4 of 15,360 MiB, driver 580.178.04,
  20 GB in `/kaggle/working`. This settles D171's "2 cores / 13 GB against 4 cores / ~30 GB".
- **fp16 HF + PEFT on T4 works:** no skipped scaler step in any of the 4 runs, no non-finite value, no out-of-memory, no stall.
  Peak VRAM was not recorded. The longest context was 3,081 tokens against 3,460 in D160's answers, and the token budget bounds a
  batch's memory either way.
- **The measured rates:**

| stage | D175 assumed | measured | full run, h (D175's) |
|---|---|---|---|
| S1 sampling | 6.2 s / problem | 5.92 s / problem | 3.95 (4.2) |
| S2 reading | 0.3 s / pass | 0.37 s / answer per GPU | 0.95 (1.8) |
| S4 training | ~50 min / run | 16.6–20.6 s / step: 1.7–2.1 h / run | **10.6 (4.2)** |
| S5 scoring | 0.2 s / item | 0.45 s / item per GPU | **8.4 (3.8)** |
| **in all** | | | **~24 (~14)** |

  - **S5's rate is real, not a small-chunk artifact:** seconds fit 0.5 + 0.344 ms × tokens (R² 0.95), so 0.43 s at 1,245 tokens.
  - **The two T4s differ by ~20%:** GPU 0 ran 20.3 s / step and 0.51 s / item, GPU 1 16.6 and 0.41. One session, so it may not recur.
    `rates.json` uses GPU 0's step rate for both.
  - **Reading through an adapter costs ~50% more than the base** (0.51 against 0.34 s / item on GPU 0).
  - **S3 cannot be projected from 168 answers** (291 s). From the laptop's fits, scaled for size and the slower CPU, ~2.7 h, inside
    arm B's window. That is an estimate.
  - D176's warning held for S4 (10.9 h projected, 10.6 measured). Its ~21 h total was low, because S5 is slower than assumed too.
- **Two sessions: no. Packing is worse than hours ÷ 9.5 suggests.** A run is one ~2 h chunk, the deadline rule refuses a run near a
  session's end, and the driver holds C, D and E until arm B finishes. `scripts/sim_r6_6a_sessions.py` replays the driver's rules at these
  rates, for the T4s as measured and for both at GPU 0's rate:

| option | what changes | sessions | GPU hours | new smoke? |
|---|---|---|---|---|
| D175 as built | nothing | 4 / 4 | 27–28 | no |
| **scheduler fix** | a free GPU takes the next run; no arm-B wait | **3 / 3** | **24–25** | no |
| + merged adapters | how test scoring reads; rounds the adapter into fp16 | 3 / 3 | 21–22 | yes, ~15 min |
| + merged, 2 seeds for B, C, D | one seed fewer | 2 / 3 | 17–18 | yes |
| + merged, 4,000 training items | a third fewer items | 2 / 3 | 18–20 | yes |
| 2 seeds and 4,000 items, not merged | both cuts | 2 / 2 | 18–19 | no |
| + merged, 2 seeds and 4,000 items | both cuts | 2 / 2 | 15–16 | yes |

  - **Only cutting both seeds and items reaches 2 sessions in both cases.** A single cut does so only with uneven T4s, ending at
    9.1–9.3 h of 9.5 h. Unmerged, the double cut ends at 9.4 h with even T4s, so it has almost no margin.
- **My recommendation: the scheduler fix alone, three sessions.** It keeps every D175 number: the same arms, seeds, items and
  readings. It can be tested on CPU, so it needs no new smoke. It needs ~25 h of a ~30 h floating week. D171 rule 10 says 2 sessions
  "where possible", and that is not possible here without cutting the design.
- **Sizing applied now (D175 allows it):** `estimates_s.train_run` 3,600 → 7,650 s and `score_set` 900 → 2,800 s, so the first chunk of a
  session is projected from measured rates. With the ingest fix, this re-pins the notebooks.
- **Still to check on Kaggle, no quota needed:** version 2's Output tab (the folder, or only the zip?), both versions' status
  badges, and the quota used and left.
- **The notebooks now pin `6d856a7d261f`.** The five suites that touch this code pass. The full suite was not re-run for this entry.

## 2026-11-10: the size of R6-6a (your decision)

### D180. R6-6a runs as D175 built it, with D177's scheduler fix: three sessions, every D175 number kept. The fix is built and CPU-tested, and the notebooks are re-pinned; nothing has run (2026-11-10)
- **Status:** exploratory track, paper #2. Your choice, from D177's table: "scheduler fix (3 sessions), keeping every D175
  number". The arms, seeds, training items, learning rate, readings, sets and analysis are unchanged. So are the sizing
  values D177 set.
- **What changes: who runs what, and when.** The results cannot depend on it: every run and every scored set is
  determined by its own seed and inputs, whichever GPU takes it.
  - **S4, one launch, a shared queue.** Both GPU workers start with the teachers (S3, CPU). Each takes the next run nobody
    has, in D175's order (B_0 … E_0), through an atomic claim file. B trains on labels, so it starts at once. A C, D or E
    run is ready when S3's done marker exists. Before, the driver waited for all of B, then split C, D and E statically
    (`[w::2]`), which left a GPU idle.
  - **A worker with no ready run waits** (polling every `session.poll_seconds` = 60 s, a sizing value, with a log line
    every ten polls for the stall guard). It stops waiting when the driver marks S3 as stopped this session
    (`s3_teachers/STOPPED.json`), or when no run could still end before the deadline.
  - **S5, a shared queue, the largest set first.** A worker loads a model's adapter when its next chunk needs another one.
    Before, models were split statically and in config order.
  - **Claims:** the driver clears a stage's claims before launching it, so a claim left by a stopped or killed worker
    cannot hold its chunk into the next session. `finish` clears them, and `restore` does not copy them.
  - **S2 is unchanged** (static parts), as in D177's simulation.
- **The plan this gives** (`scripts/sim_r6_6a_sessions.py`, D177's rates; unchanged since D177):
  - T4s as measured: 3 sessions, 23.9 GPU-session hours, ending at 8.5 / 9.1 / 6.3 h.
  - Both at the slower T4's rate: 3 sessions, 25.4 h, ending at 7.1 / 8.6 / 9.6 h. The last includes S6 and the zip
    (~0.2 h), which run after the deadline check, inside Kaggle's 12 h limit.
  - **Session 1:** setup, S1 sampling (~4 h), S2 reading (~1 h), then S3 on the CPU while both GPUs train the three B
    runs. C cannot start before the deadline, so the session stops itself at ~8.5 h.
- **The tests (D171 rule 9, CPU):**
  - two real worker processes on one queue: every chunk is written exactly once, and both workers take some;
  - a worker waits while a chunk may become ready and stops when it cannot (a stopped marker, or the deadline);
  - a claim holds its chunk until cleared, and `restore` leaves claims behind;
  - S4 trains B at once and the rest only after the teachers, and stops instead of waiting when S3 is marked stopped;
  - S5's order is by set size;
  - the tiny Qwen still runs every stage through `drive()`, now with S3 and S4 started together.
  - `tests/test_kaggle_stages.py` and `tests/test_r6_6a.py` pass. Before the re-pin, only the two pin tests failed, as they
    must.
- **The Kaggle checks D177 left open** (version 2's Output tab, the versions' status badges, the GPU quota) are still to
  come from you. Session 1 needs about 10 h of quota left (§10 of `kaggle_instructions.md`).
- **Errata:** commit `4425478` (paper #1) says "887 passed"; the suite printed 888 passed.
