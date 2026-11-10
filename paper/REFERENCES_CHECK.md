# Reference check (D179; R5-10, R4-9)

Checked on 2026-11-10 against each paper's own record. The paper cites nothing that is not in this file. Every
sentence that describes a paper's content was written from its abstract (or, for DAR's reward, its arXiv HTML), read
in the same session.

**Sources used:**
- **arXiv:** the arXiv API (`export.arxiv.org/api/query`), which returns the abstract page's title, authors, first
  submission date and the authors' own comment. Where the comment names a venue, that is recorded as "venue per the
  arXiv comment".
- **Journals and books:** Crossref (`api.crossref.org/works/<DOI>`).
- **Venue pages:** opened directly where the arXiv record names no venue (ICLR 2026 poster and proceedings pages).
  Two venues were confirmed from search listings of the official page (PMLR for PAL, NeurIPS proceedings for QLoRA).
  OpenReview served a browser check instead of the page.

**Status:** verified = title, authors and year match the record, and the venue is confirmed as stated.

## Cited in Related work (new in D179)

| key | title | authors | year | venue | checked against | status |
|---|---|---|---|---|---|---|
| `kadavath2022know` | Language Models (Mostly) Know What They Know | Kadavath, Conerly, Askell, Henighan, Drain, Perez, … Kaplan (36 authors) | 2022 | arXiv preprint 2207.05221 (no venue in the record) | arXiv API | verified; cited as a preprint |
| `tian2023justask` | Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback | Tian, Mitchell, Zhou, Sharma, Rafailov, Yao, Finn, Manning | 2023 | EMNLP 2023 (venue per the arXiv comment, "EMNLP 2023 Camera Ready") | arXiv API, 2305.14975 | verified |
| `xiong2024express` | Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs | Xiong, Hu, Lu, Li, Fu, He, Hooi | 2024 | ICLR 2024 (venue per the arXiv comment) | arXiv API, 2306.13063 | verified |
| `groot2024overconfidence` | Overconfidence is Key: Verbalized Uncertainty Evaluation in Large Language and Vision-Language Models | Groot, Valdenegro-Toro | 2024 | arXiv preprint 2405.02917. The comment says "to appear in TrustNLP workshop @ NAACL 2024"; not confirmed on a venue page | arXiv API | verified; cited as a preprint |
| `wang2026pitfalls` | On the Pitfalls of Verbalized Confidence Priors for Calibrating Large Reasoning Models | S. Wang, Luo, Zeng, Yu, Zhang, Xie, Jing, Wei | 2026 | arXiv preprint 2609.32470 | arXiv API | verified; cited as a preprint |
| `epstein2025fermieval` | LLMs are Overconfident: Evaluating Confidence Interval Calibration with FermiEval | Epstein, Winnicki, Sornwanee, Dwaraknath | 2025 | arXiv preprint 2510.26995 | arXiv API | verified; cited as a preprint |
| `qin2026quantsight` | QuantSightBench: Evaluating LLM Quantitative Forecasting with Prediction Intervals | Qin, Andriushchenko | 2026 | arXiv preprint 2604.15859 | arXiv API | verified; cited as a preprint |
| `damani2026rlcr` | Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty | Damani, Puri, Slocum, Shenfeld, Choshen, Kim, Andreas | 2026 | ICLR 2026 | arXiv API (2507.16806); the ICLR 2026 poster page (iclr.cc/virtual/2026/poster/10011036) | verified |
| `baniharouni2026doubt` | Rewarding Doubt: A Reinforcement Learning Approach to Calibrated Confidence Expression of Large Language Models | Bani-Harouni, Pellegrini, Stangel, Özsoy, Zaripova, Navab, Keicher | 2026 | ICLR 2026 | arXiv API (2503.02623); ICLR 2026 proceedings page (proceedings.iclr.cc, hash bf806544…) | verified |
| `park2026dar` | Reinforcement Learning over Predictive Distributions for LLM Regression | Park, Chae, Mendes, DeYoung, Kishore, Xu, Ritter | 2026 | arXiv preprint 2605.20740 | arXiv API; the arXiv HTML confirms the reward is a leave-one-out empirical CRPS | verified; cited as a preprint |
| `wang2023selfconsistency` | Self-Consistency Improves Chain of Thought Reasoning in Language Models | X. Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou | 2023 | ICLR 2023 (venue per the arXiv comment) | arXiv API, 2203.11171 | verified (already in refs.bib; matches) |
| `huang2024selfcorrect` | Large Language Models Cannot Self-Correct Reasoning Yet | Huang, Chen, Mishra, Zheng, Yu, Song, Zhou | 2024 | ICLR 2024 (venue per the arXiv comment) | arXiv API, 2310.01798 | verified |
| `gao2023pal` | PAL: Program-aided Language Models | Gao, Madaan, Zhou, Alon, Liu, Yang, Callan, Neubig | 2023 | ICML 2023, PMLR 202:10764–10799 | arXiv API (2211.10435); PMLR listing (proceedings.mlr.press/v202/gao23f) from search | verified (already in refs.bib; matches) |
| `chen2023pot` | Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks | Chen, Ma, X. Wang, Cohen | 2023 | TMLR 2023 (venue per the arXiv comment) | arXiv API, 2211.12588 | verified (already in refs.bib; matches) |
| `jcgm101` | Evaluation of measurement data — Supplement 1 to the "Guide to the expression of uncertainty in measurement" — Propagation of distributions using a Monte Carlo method | Joint Committee for Guides in Metrology | 2008 | JCGM 101:2008 | BIPM's JCGM publications page | verified (already in refs.bib) |
| `gneiting2007strictly` | Strictly Proper Scoring Rules, Prediction, and Estimation | Gneiting, Raftery | 2007 | Journal of the American Statistical Association 102(477):359–378 | Crossref, DOI 10.1198/016214506000001437 | verified (already in refs.bib; matches) |

## Already cited in Setup, Methods and the protocol (checked in the same pass)

| key | checked against | result |
|---|---|---|
| `shao2024deepseekmath` | arXiv API, 2402.03300 | **corrected:** the entry left out Xiao Bi and Haowei Zhang (11 authors). The title and year match |
| `yang2024qwen25math` | arXiv API, 2409.12122 | matches (16 authors) |
| `grattafiori2024llama3` | arXiv API, 2407.21783 | matches (Grattafiori et al., 561 authors) |
| `hu2022lora` | arXiv API (2106.09685); the ICLR 2022 poster page (iclr.cc/virtual/2022/poster/6319) from search | matches; ICLR 2022 |
| `dettmers2023qlora` | arXiv API (2305.14314); NeurIPS 2023 proceedings listing from search | matches; NeurIPS 2023 |
| `gneiting2005emos` | Crossref, DOI 10.1175/MWR2904.1 | matches: Monthly Weather Review 133(5):1098–1118 |
| `kupiec1995` | Crossref, DOI 10.3905/jod.1995.407942 | matches: The Journal of Derivatives 3(2):73–84 |
| `schuirmann1987` | Crossref, DOI 10.1007/BF01068419 | matches: J. Pharmacokinetics and Biopharmaceutics 15(6):657–680 |
| `meurer2017sympy` | Crossref, DOI 10.7717/peerj-cs.103 | matches: PeerJ Computer Science 3:e103 |
| `davison1997bootstrap` | Crossref, DOI 10.1017/CBO9780511802843 | matches: Cambridge University Press, 1997 |
| `holm1979` | search listings of JSTOR stable 4615733 (no DOI in Crossref); the JSTOR page itself was not opened | matches: Scand. J. Statist. 6(2):65–70 |
| `vonwerra2020trl` | GitHub API: huggingface/trl, created 2020-03-27 | the repository exists; software citation |
| `unsloth` | GitHub API: unslothai/unsloth, created 2023-11-29 | the repository exists; software citation |

## Not cited (from the outline; not needed for this draft)

CARE-PPO, Turtel et al. (2505.17989), Dr.GRPO / DAPO (2607.00152, 2503.14476), Guo et al. (2604.05306), Grünefeld et
al. (2605.07776), Kapoor et al. (2406.08391), Kamath et al. (ACL 2020). None was checked in this session, so none is
cited.
