# R6-6 on Kaggle: draft plan (D171; exploratory, paper #2; nothing built, nothing run)

**Status: a draft, not a pre-registration.**
- **The workflow** it follows is D171 in `docs/exploratory/DECISIONS_FOLLOWUP.md`.
- **Written before any R6-3, R6-4 or R6-5 number was read.** The go decision still waits for those results (D165). §5
  says how each reading changes this plan.
- **Before any code or run:** your decisions in §1, then a pre-registration D-entry that fixes the rules below.
- **Paper #1 is untouched** (D148's freeze).

R6-6 is two experiments (D165). They get one notebook each:
- **R6-6a:** probe-to-verbal-confidence distillation, against direct supervised confidence and no training, scored by
  pairwise accuracy, Brier and selective risk, with held-out templates;
- **R6-6b:** then one replication on a different model family.

## 1. Decisions I need from you

| # | Decision | Recommendation | Why it matters |
|---|---|---|---|
| 1 | Does "self-contained" allow cloning the repo at a pinned commit and downloading pinned weights? | **Yes**, as D157's notebooks do | The alternative is pasting ~2,000 lines of code into each notebook, which breaks the pin test |
| 2 | What is the "verbal confidence"? | **P(yes) from D151's verification turn**, one forward pass | It reuses a checked readout. A stated percentage is what a reader would see, but it needs its own format and decoder |
| 3 | What does "held-out templates" mean? | **Both:** a problem question template (primary) and a verification-turn wording (secondary) | D165 does not say which. The second costs one extra pass |
| 4 | May I add two controls D165 does not list? | **Yes:** arms D and E in §3 | Without D, a gain cannot be separated from surface text (random weights read 0.66–0.67) |
| 5 | Which model replicates in R6-6b? | **Decide after R6-6a**, between Phi-4-mini-instruct and Llama-3.2-3B-Instruct (§4) | Costs most of a week's quota |
| 6 | May I run one short Kaggle smoke session before R6-6a? | **Yes, when the build is ready** (§6) | It settles what no source confirms: whether output survives a crash |

## 2. What Kaggle-first changes here

- **The readouts read the copy that sampled.** D157's caveat (fp16 answers read through a 4-bit copy) goes away.
- **The cost:** nothing here compares directly with the local R6-3 to R6-5 numbers (D167's comparability rule). So
  R6-6a measures its own baselines, including the probe, in the same notebook. Local numbers are context only.
- **A new pool.** Fresh generator seeds, disjoint from X2's 1,200 and the frozen eval sets (checked at the build). It
  keeps test problems away from every set the probes were studied on.

## 3. R6-6a: distillation (`notebooks/kaggle_r6_6a.ipynb`, not built)

**The question:** can what a probe reads be moved into a cheap verbal confidence, beyond what direct labels and surface
text already give?

**The model:** the native Qwen2.5-Math-1.5B-Instruct, fp16, sampling at D157's settings. It is the smallest model with
the native-trace probe signal, and it fits one T4.

**The pool:** 2,400 problems, round-robin over the 12 family × depth cells, k = 8.

| split | problems | use |
|---|---|---|
| held-out template | ~800 (every problem using the held-out question template) | test only |
| train | 1,000 | targets, training |
| validation | 120 | divergence checks only |
| ID test | 480 | test only (primary) |

- **Splits are by problem** and fixed from seeds before sampling.
- **The held-out template needs no generator change.** `render.py` does not record template IDs, so a problem is
  classified by matching its text against the 3 question templates. The held-out one is "Determine {sym}, the {desc}…".
- **No held-out family:** `physcal/env/formulas/` has kinematics, dynamics and energy only. Nothing here claims transfer
  across families.

**The arms.** All arms score the same frozen answers, so every comparison is paired. A LoRA adapter acts only on the
verification turn, so answers and accuracy are identical across arms.

| arm | training target | seeds | source |
|---|---|---|---|
| A: no training | none (the base model's P(yes)) | – | D165 |
| B: direct supervised | the correct / incorrect label | 3 | D165 |
| C: probe-distilled | the out-of-fold probe P(correct) | 3 | D165 |
| D: text-distilled | the out-of-fold rich-text P(correct) | 3 | **added (decision 4)** |
| E: shuffled | C's targets shuffled within each problem | 1 | **added (decision 4)** |

- **References, not arms:** the probe and the rich-text classifier themselves on the test sets; self-consistency at
  k = 8 (the cost reference); length.
- **Training:** soft cross-entropy on the yes / no logits at the verification turn. Same items, steps, learning rate and
  LoRA shape in every arm. One fixed learning rate, so no arm is tuned more than another.
- **Targets:** `x2_analysis.crossfit_confidence` and `followup_r6.crossfit_rich`, on train problems only, folds by
  problem. Test problems never enter a target.
- **Note on B against C:** B is trained on the truth, and C on a noisy estimate of it. C − B at or below 0 is a plausible,
  valid result.

**The metrics:**
- **Primary:** X2's within-problem pairwise accuracy on the ID test (mixed problems; bootstrap over problems,
  B = 10,000, seed 0, paired).
- **Secondary:** Brier and selective risk (AURC) on **all** answered test items. Mixed problems alone would distort the
  base rate, so training and these two metrics use every answered sample, not only kept problems.
- **Also:** the same on the held-out template set; the spread across seeds; truncated samples, counted separately.
- **Rough precision:** D165's paired 7B − native CI had a half-width of ~0.03 on 599 problems. The ID test should have
  ~310 mixed problems, so about ±0.04.

**Draft readings** (fixed only in the pre-registration):

| contrast | CI above 0 | CI contains 0 |
|---|---|---|
| B − A | direct supervision helps | labels do not help |
| C − A | distillation helps | distillation does not help |
| C − B | the probe's target adds to labels | labels are enough |
| C − D | it moves more than surface | distillation is surface transfer |
| E − A (a check) | **the setup leaks; stop and diagnose** | as expected |

**The stages.** Each saves chunks and resumes (D171). Rates marked *assumed* are replaced by the first chunk's timing.

| stage | what | rate | hours |
|---|---|---|---|
| S0 | venvs, repo, weights | assumed | 0.4 |
| S1 | sampling, 2,400 × 8, vLLM | **measured**, 6.2 s / problem (D160) | 4.2 |
| S2 | P(yes) and states, ~18,700 answers, HF fp16, both GPUs | assumed 0.3 s / pass | 1.6 |
| S3 | teacher fits on CPU (probe, rich text), during S4's arm B | laptop 3,043 s; Kaggle 2–3× slower, assumed | overlapped |
| S4 | 10 LoRA runs, ≤ 6,000 answers each, 2 at a time | assumed ~50 min / run | 4.2 |
| S5 | test scoring, ~10,000 answers × 10 runs | assumed 0.2 s / item | 2.8 |
| S6 | analysis and the results zip | CPU | 0.2 |
| | **in all** | | **~13.5 (range 12–18)** |

- **Quota:** 2 sessions, about half of a typical week's ~30 h.
- **Why S3 overlaps:** an idle GPU still spends quota, and arm B does not need the teacher.
- **What comes back:** ~9 MB of gzipped completions (a full 1.5B run of 1,176 problems is 4.4 MB) plus small tables.
- **What stays:** ~1.7 GB of states, in the resume tier only.

## 4. R6-6b: one different family (`notebooks/kaggle_r6_6b.ipynb`, not built)

The same notebook with another model, after R6-6a. Its pool, arms and readings stay R6-6a's.
- **The rule for a candidate:** different family; weights reachable without a gated login; safe in fp16 (T4 has no bf16,
  which makes Gemma-family models a risk); produces `\boxed{}` answers.
- **The gate, on format and pass rate only, never on a readout:** the first chunk must show ≥ 50% answered (D157's gate)
  and a share of mixed problems high enough for the test set.
- **The candidates (licenses and sizes to re-verify before the entry):**
  - **Phi-4-mini-instruct:** the first choice; strong at math, so mixed problems are likely;
  - **Llama-3.2-3B-Instruct:** the fallback. FINDINGS #38's NO-GO came from too few computation-correct pairs on A6b, so
    this format's yield must be checked first.
- **The cost, assumed:** 20–28 h for the full design, about a week's quota. N and the seeds are set after R6-6a's
  measured rates.

## 5. How R6-3 to R6-5 change this

| result under D167 | change |
|---|---|
| R6-5 (a): the probe beats rich text | C − D is the headline contrast |
| R6-5 (a): it does not | C − D is expected at 0. Keep D; C can drop to 1 seed (saves ~0.8 h), and "elicitation" leads paper #2's framing |
| R6-5 (d): the probe tracks the reasoning | add the swap to S5, as a secondary test of the students |
| R6-5 (d): it tracks the answer | add an answer-only student context, a cheaper variant |
| R6-4: presentation dominates | as planned; the native frame is the right place to work |
| R6-4: weights dominate | add the SFT weights as a second student (secondary) |
| R6-3: the verdict's size effect holds | a 7B student becomes an option; costed separately, it is likely 25 h or more |

If R6-3 to R6-5 undercut the premise, the recommendation may be not to run R6-6a at all. That is yours to decide after
the readings.

## 6. Build order (each step waits for your go)

1. **Your decisions (§1),** D167's readings, then the pre-registration D-entry, before any code.
2. **A shared stage runner** (`physcal/eval/kaggle_stages.py`) with kill-and-resume tests on CPU.
3. **R6-6a's module, config and notebook,** with the pin test and a tiny-Qwen CPU run of every stage.
4. **One Kaggle smoke session (~45 min of quota):**
   - does `/kaggle/working` survive a deliberate crash;
   - the machine's real CPU count, RAM and disk;
   - fp16 HF + PEFT LoRA on a T4: a real step rate and no overflow.
5. **R6-6a,** session 1 then session 2; ingest; then the R6-6b decision.

## 7. Known risks

- **Crash survival is unknown.** No source confirms that `/kaggle/working` survives a crash or a timeout. The self-deadline
  avoids the timeout. If the smoke test shows a crash loses output, a mid-session copy to a private dataset is needed,
  which needs a token stored as a Kaggle secret.
- **Machine limits disagree across sources:** 2 cores / 13 GB against 4 cores / ~30 GB. vLLM logged only 2.83 GiB of RAM
  available during the 7B load (`results/followup/scale/7b/kaggle/logs/`). None of this is from Kaggle's own docs.
- **The quota floats** from week to week.
- **File limits:** ~20 GB and possibly ~500 files in `/kaggle/working` (the file count rests on forum reports).
- **fp16 training** can overflow: fp32 adapter weights, a gradient scaler, and a stop on a non-finite loss.
- **The HF + PEFT stack on T4 has never run here.** Every S2, S4 and S5 rate above is an assumption until step 4.
- **Each session needs you:** a resume means starting a new version with the previous output attached.
