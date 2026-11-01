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
