# Item 3 on Kaggle: step by step (D157; exploratory, follow-up project, not the paper)

The notebook `notebooks/kaggle_7b_scale.ipynb` does everything on Kaggle:
- installs vLLM in its own venv;
- clones this repo at a pinned commit;
- downloads the pinned 7B weights and checks them;
- times the first 20 candidates, then samples as many chunks as fit;
- zips the result.

You start it, close the browser, and download one zip in the morning. The plan and every rule are in
`docs/exploratory/DECISIONS_FOLLOWUP.md` D157.

**Time and quota:** one run takes at most ~10 h of wall time. Sampling stops by 9.5 h after the session starts, and
Kaggle's limit is 12 h. It uses that much of your weekly GPU quota, typically 30 h; the editor's right panel shows
what is left.

## 1. Account (once)

1. Go to <https://www.kaggle.com> → **Register**. A Google account or e-mail both work.
2. **Phone verification**, which Kaggle needs for GPUs and for internet in notebooks:
   - click your profile picture (top right) → **Settings**;
   - in **Phone verification**, enter your number and type the SMS code.

## 2. Create the notebook (once)

1. Get the notebook file:
   - on GitHub: open `notebooks/kaggle_7b_scale.ipynb` → **Download raw file**;
   - or copy it out of your local checkout: `notebooks/kaggle_7b_scale.ipynb`.
2. On Kaggle: the left sidebar's **+ Create** → **New Notebook**. The editor opens.
3. In the editor's menu: **File** → **Import Notebook** → choose the `.ipynb` → **Import**. It replaces the empty cells.
4. Click the title (top left) and rename it, e.g. `physcal-7b-scale`.

## 3. Settings (check before every run)

In the editor's right panel (**Session options**; on a narrow window, the **Settings** menu):

1. **Accelerator** → **GPU T4 x2**. Confirm the dialog. The first cell refuses anything else.
2. **Internet** → **On**. If the switch is grey, the phone verification is missing.
3. **Persistence** and **Environment**: leave the defaults. The notebook brings its own Python environment.

Do not edit any cell. In particular `COMMIT` is the pinned code.

## 4. Run it in the background

1. Top right: **Save Version**.
2. **Version type:** **Save & Run All (Commit)**.
3. If **Advanced Settings** offers an output option, choose **Always save output for this version**.
4. **Save**. Kaggle runs the notebook on its own machine, so you can close the tab and the browser.
5. **Optional, recommended:** watch the start for ~30–40 min. Open the notebook → the version's **View Logs** (or
   **Versions** → the running version → **Logs**). These lines should appear in this order:
   - the GPU line, with 2 × Tesla T4 and the driver;
   - `repo <commit> | candidates ok | driver …`;
   - the torch / vLLM line, with `gpus 2`;
   - the weights' sha256 check;
   - vLLM's engine start (a few minutes);
   - `throughput on 20 candidates: … s / problem …; answered …`;
   - `plan: N candidates in all …`.

   After the `plan:` line nothing is needed until the end. A line per chunk follows every few minutes.

## 5. In the morning: download the output

1. Kaggle e-mails you when the version finishes. Or open the notebook: the version shows **Complete** (or **Error**).
2. Read the end of the log for the **`STATUS:`** line. It gives:
   - the sampler's exit code: 0 is done, and 3 means the health gate stopped it (D157 difference 7);
   - the number of chunks and candidates.
3. Open the notebook's page → the **Output** tab (on the version's page: **Output** / **Data** → output files) →
   `followup_7b_scale.zip` → **Download**. The other files in the output are the same data unzipped. They are only
   needed for a resume (§6).

**If the status is not 0:**
- **Exit 3:** the health gate stopped it. Do not re-run; tell me.
- **An error before sampling** (install, clone, download): nothing was sampled. Re-run (§4) or tell me.

## 6. Only if you want more candidates: resume

Use this when the zip has fewer than 1,176 candidates and you want the rest, or when a session ended early.

1. Open the notebook → **Edit**.
2. Right panel → **Input** → **+ Add Input** → **Your Work** (or **Notebook Output Files**) → this notebook → its
   latest version → **Add**.
3. Run it again as in §4.

The notebook copies the earlier chunks from `/kaggle/input`, after checking them, and continues from the first missing
one. Each new version's zip holds all chunks so far, so download the latest one.

## 7. Bring the zip into the repo

1. Copy it into WSL, outside the repo, e.g.:

   ```bash
   mkdir -p ~/kaggle_downloads
   ```

   ```bash
   cp /mnt/c/Users/<you>/Downloads/followup_7b_scale.zip ~/kaggle_downloads/
   ```

2. **Quick check (CPU, a minute; optional).** This only ingests:
   - it checks the zip, the manifest, the pool, the seeds and the prompts with the local tokenizer;
   - it re-grades every answer;
   - it writes `results/followup/scale/7b/samples/` and `results/followup/scale/7b/kaggle/ingest.json`.

   ```bash
   uv run python -m physcal.eval.followup_kaggle ingest --zip ~/kaggle_downloads/followup_7b_scale.zip
   ```

3. **The local chain** (when the GPU is free, e.g. overnight). It runs the ingest (again, idempotent), the P(yes)
   and hidden-state readouts on the GPU under the usual guards, then D151's analysis. It commits and pushes after
   every stage:

   ```bash
   mkdir -p results/queue_followup_kaggle
   ```

   ```bash
   KAGGLE_ZIP=~/kaggle_downloads/followup_7b_scale.zip GPU_DEADLINE=09:00 setsid nohup scripts/queue_followup_kaggle.sh >> results/queue_followup_kaggle/nohup.log 2>&1 < /dev/null &
   ```

   - **The chain's log:** `results/queue_followup_kaggle/chain.log`.
   - **A re-run resumes** an unfinished readout.
   - **The result:** `results/followup/scale/summary.md` and `results.json`.

The zip itself is not committed. The ingested chunks are, with the zip's sha256 in `ingest.json`.

## 8. The native 1.5B control (D160): the same steps, another notebook

The notebook is `notebooks/kaggle_1.5b_scale.ipynb`. It is the 7B's notebook with the official
`Qwen/Qwen2.5-Math-1.5B-Instruct`; everything else is the same: the prompts, decoding, k = 8, token limit, seeds and
candidates. It should take ~2–4 h, well under the 7B's 5 h. A 1.5B is faster, but the engine and the chunk rule are
the same.

1. **A new notebook,** not the 7B's: §2 with `kaggle_1.5b_scale.ipynb`; name it e.g. `physcal-1.5b-scale`.
   Importing it into the 7B's notebook would mix the two runs' versions.
2. **Settings:** §3 as before (**GPU T4 x2**, **Internet On**).
3. **Run:** §4 (**Save Version** → **Save & Run All (Commit)**). The log lines are the same; the `repo …` line also
   says `model 1.5b Qwen/Qwen2.5-Math-1.5B-Instruct@aafeb0fc`.
4. **Download:** §5. The zip is **`followup_1.5b_scale.zip`**, and the `STATUS:` line starts with `1.5b:`.
5. **A resume** (only if needed): §6, attaching this notebook's own latest version.
6. **Bring it back:** copy it next to the 7B's zip:

   ```bash
   cp /mnt/c/Users/<you>/Downloads/followup_1.5b_scale.zip ~/kaggle_downloads/
   ```

   The local chain then ingests it and runs:
   - P(yes);
   - the hidden states;
   - the random-weights copy's hidden states;
   - the analysis.

   It needs the GPU free; the 1.5B needs a 3.5 GiB cap, so most apps can stay open:

   ```bash
   MODEL=1.5b KAGGLE_ZIP=~/kaggle_downloads/followup_1.5b_scale.zip GPU_DEADLINE=09:00 setsid nohup scripts/queue_followup_kaggle.sh >> results/queue_followup_kaggle/nohup.log 2>&1 < /dev/null &
   ```

## 9. R6-6a's smoke session (D175): ~45 minutes of quota, two short versions

**What it is for.** Three questions, before the full R6-6a: does output survive a crash, what is the machine really, and
what are the real rates (fp16 HF + PEFT LoRA on a T4)? It runs every stage's real code on 48 problems of its own. Its numbers
are plumbing, never results.

**Both versions end in an error on purpose.** That is the test: after each error, look at the Output tab.

1. **A new notebook:** §2 with `notebooks/kaggle_r6_6a_smoke.ipynb`; name it e.g. `physcal-r6-6a-smoke`.
2. **Settings:** §3 (**GPU T4 x2**, **Internet On**).
3. **Version 1:** §4 (**Save Version** → **Save & Run All (Commit)**). In **Advanced Settings**, choose **Always save output
   for this version**. It takes ~35–40 min and stops itself by 45 min. The log should show, in order:
   - the GPU line, then `repo <commit> | mode smoke | driver …`;
   - the torch / vLLM / peft line, then the weights' sha256 check;
   - `throughput on 20 candidates: …`, then `S1: 384 samples, … answered`;
   - `s2_read: part_…`, then `s4_train` steps (`step n/m: loss …, … s / step`), then `s5_score: …`;
   - `S6: results.json written`, then `STATUS: {…}`;
   - the last cell's `RuntimeError: version 1: the deliberate crash …`. The version shows **Error**. This is expected.
4. **Check version 1's Output tab** (the notebook's page → the version → **Output**). Note three things for me:
   - are there files at all?
   - is `r6_6a_smoke_results.zip` there? **Download it.**
   - how many files are in `r6_6a_smoke/filecount/`? (400 were written.)

   **If the tab is empty,** stop here and tell me: a crash loses output, so R6-6a needs a mid-session copy to a private
   dataset (the plan's §7). Version 2 cannot run.
5. **Version 2:** open the notebook → **Edit** → right panel → **Input** → **+ Add Input** → **Your Work** (or **Notebook
   Output Files**) → this notebook → version 1 → **Add**. Then **Save & Run All (Commit)** again, with the same Advanced
   Setting. It takes ~8 min. The log should show:
   - `restore: … files from …` and a JSON block with `filecount_files`;
   - `{"chunks_checked": …}`: every chunk's sha256 came back intact;
   - no `s2_read`, `s4_train` or `s5_score` chunk lines: every stage is skipped;
   - `version 2: a hard kill of this kernel now`. The kernel dies and the version shows an error. This is expected.
6. **Check version 2's Output tab** the same way, and download its `r6_6a_smoke_results.zip` if it is there. It is the better
   zip, because it also holds `smoke/survival.json`.
7. **Bring the zip back** (the newest one you have):

   ```bash
   cp /mnt/c/Users/<you>/Downloads/r6_6a_smoke_results.zip ~/kaggle_downloads/
   ```

   ```bash
   uv run python -m physcal.eval.r6_6a ingest --mode smoke --zip ~/kaggle_downloads/r6_6a_smoke_results.zip
   ```

8. **Tell me:** the answers to step 4 for each version, and the GPU quota each used (the editor's right panel). I read
   `results/followup/r6_6a/smoke/smoke/rates.json`, record the measured rates in a D-entry, and re-pin if sizing changes. If
   the rates need more than 2 sessions, the choice comes to you before any run.

**The full run is §10** (D180). The smoke above is done (D177).

**If something goes wrong:**

| what you see | what it means | what to do |
|---|---|---|
| `STATUS` shows `stopped_at` and no `S6:` line | the 45-minute deadline came first, or a stage failed | the rates so far are still in the zip; carry on from step 4 and tell me |
| `STOP RULE: non-finite …` | fp16 overflow | a real finding; carry on from step 4 and tell me |
| `CUDA out of memory` | a batch budget is too large for a T4 | carry on from step 4 and tell me; budgets may change (D175) |
| `this run was started with other pins` | the attached output is from another commit | attach the output of this notebook's version 1 only |
| `attach one version only` | two versions are attached | remove one input |

## 10. R6-6a, the full run (D175, D180): three sessions of ~9 h, one after another

**What it is.** D175's design unchanged, with D180's scheduler: a free GPU takes the next run. Each session stops itself by
9.5 h, saves its chunks to the Output tab, and the next session resumes from them. D177's rates give three sessions:

| session | what it should do | ends at |
|---|---|---|
| 1 | setup; S1 sampling (~4 h); S2 reading (~1 h); S3 teachers (CPU) while both GPUs train B_0, B_1, B_2 | ~8.5 h |
| 2 | C, D, E (seven runs, ~2 h each, both GPUs) | ~9 h |
| 3 | the last runs if any, S5 scoring (~8 h of GPU work over 2 GPUs), S6 analysis, the zip | ~6–9.6 h |

In all, ~24–25 h of GPU quota (a T4 x2 session counts its wall time once).

**Before session 1: three checks, then tell me the answers** (they need no quota):
1. **Smoke version 2's Output tab:** is the `r6_6a_smoke` folder there, or only the zip?
2. **The status badges** of smoke versions 1 and 2: both **Error**, or something else?
3. **Your GPU quota** (kaggle.com → your profile → **Settings**, or the editor's right panel): hours used, hours left, and
   when it resets.

**Plan it around the quota.** Start a session only with **at least 10 h left**: a session can take 9.5 h plus ~15 min of
setup and packing. If fewer than 10 h are left, wait for the reset. The three sessions can span a reset; a later session
needs only the earlier version's output, which does not expire with the quota.

### Session 1

1. **A new notebook:** §2 with `notebooks/kaggle_r6_6a.ipynb` (pinned to `4cdd62c6e333`); name it e.g. `physcal-r6-6a`.
2. **Settings:** §3 (**GPU T4 x2**, **Internet On**). **No input attached:** session 1 starts fresh.
3. **Run:** §4 (**Save Version** → **Save & Run All (Commit)**). In **Advanced Settings**, choose **Always save output for
   this version**. You can close the browser: it runs in the background.
4. **What the log should show, in order:**
   - the GPU line, then `repo 4cdd62c… | mode full | driver …`;
   - `restore: no previous run under /kaggle/input`, then the weights' sha256 check;
   - sampling for ~4 h, then `S1: … samples, … answered`;
   - `s2_read: part_…` lines for ~1 h;
   - then the teachers and arm B together: `s4_train: … step n/375: loss …` from both GPUs, `s4_train: B_0.pt in … s`;
   - `s4_train: … chunks wait for an earlier stage` (C, D and E waiting for the teachers), then
     `… would end after the deadline; stopping here`;
   - `STATUS: {… "stopped_at": "s4_train" …}`. **This is the expected end of session 1**, at ~8.5 h. The version should
     show **Complete**, not Error.
5. **After it ends:** open the version's **Output** tab. The `r6_6a` folder should be there with `s1_sample`, `s2_read`,
   `s3_teachers` and `s4_train` (three `B_*.pt` files). **Download `r6_6a_results.zip`** as a backup (partial: samples and
   status only).
6. **Tell me:** the `STATUS` line, the session's run time, and the quota it used. I check the status and the rates before
   you start session 2.

### Session 2 and 3 (after I have read session 1)

Open the same notebook → **Edit** → right panel → **Input** → **+ Add Input** → **Your Work** → this notebook → **the
latest version** → **Add**, then remove any older version from the inputs, and **Save & Run All (Commit)** with the same
Advanced Setting. The log should show `restore: … files from …` and `{"chunks_checked": …}` before any new chunk. Session 3
ends with `S6: results.json written` and `stopped_at: null`; bring its zip back as in §7:

```bash
uv run python -m physcal.eval.r6_6a ingest --mode full --zip ~/kaggle_downloads/r6_6a_results.zip
```

**If something goes wrong:**

| what you see | what it means | what to do |
|---|---|---|
| the version shows **Error** or stops well before 8 h | a crash | download the zip if there is one, and tell me before session 2 |
| `chunks wait for an earlier stage that will not finish` early in the session | the teachers (S3) stopped | tell me; the next session resumes S3 |
| `STOP RULE: non-finite …` | fp16 overflow in a run | tell me; it is a finding, and the other runs continue |
| `this run was started with other pins` | the attached output is from another commit | attach only this notebook's latest version |
| `attach one version only` | two versions are attached | remove the older one |
| `STALL` then a retry | D157's automatic retry | nothing |

## Troubleshooting

| what you see | what it means | what to do |
|---|---|---|
| `needs Accelerator 'GPU T4 x2'` | another accelerator, or none | §3.1, then run again |
| the clone or the pip install fails at once | internet is off | §3.2 |
| `driver … < 580` | Kaggle's image changed its driver | tell me (the pinned vLLM needs CUDA 13) |
| `less than 35 GB free` | the machine has less disk than expected | tell me |
| `STALL` or `attempt 1: exit …`, then `attempt 2` | the automatic retry (D157) | nothing; it resumes from the chunks on disk |
| `STOP: the health gate failed` | the fp16 model is not answering | nothing more runs; download the zip and tell me |
