# Review synthesis, Round 6 — Gemini and ChatGPT, 2026-11-03

Follow-up track (paper #2) plus one wording change to paper #1. Commit this file.
Record R6-1 to R6-6 in docs/exploratory/DECISIONS_FOLLOWUP.md (R6-1 also gets a
paper #1 D-entry in DECISIONS.md, since it changes the paper's text).

## Where both reviewers agree

1. Paper #1 proceeds to arXiv on ~20 November; its experiments stay frozen.
2. The "fine-tuning removed the model's knowledge of its errors" reading is not
   supported. The 0.758 → 0.539 probe gap may mean the information was (i) lost,
   (ii) displaced to other tokens or layers, (iii) present but not linearly
   readable, or (iv) never internal, only inferable from the text. The strong
   random-weights probe (0.66–0.67 on native traces vs 0.53 on the fine-tuned
   format) makes (iv) a serious candidate: native reasoning exposes errors in the
   text (hesitation, re-checks, length); the terse format hides them.
3. No scale-law claim from two models: "we observed no improvement in this probe
   metric between the two native sizes under these conditions".
4. Novelty: hidden-vs-expressed knowledge, including pair ranking, is prior art
   (Gekhman et al., Inside-Out, COLM 2025; Orgad et al., ICLR 2025). Paper #2's
   contribution must be the controlled effect of post-training, format and size on
   accessible answer-specific information, and whether it transfers into cheap
   verbalized confidence.

## Decisions

**R6-1. Paper #1 wording (paper D-entry).** Replace the broad epistemic claim with:
"On the fine-tuned Qwen2.5-Math-1.5B model and its terse response format, the tested
single-completion uncertainty signals provide little within-problem discrimination
between correct and incorrect answers; sample agreement provides stronger
discrimination. This is specific to the evaluated checkpoint and elicitation
conditions and does not establish that small language models generally lack
accessible answer-correctness information." Add one short, explicitly exploratory
paragraph in Discussion/Limitations: a post-freeze comparison found stronger
hidden-state probe discrimination in the native 1.5B (0.758 vs 0.539), largely
matched by a random-weights probe on the same text (0.673), and a larger
verbalized-verdict gain from native 1.5B to 7B; conditions are not matched, so this
motivates separate investigation. The 7B comparison itself is reserved for paper #2.
Report R5-6 J with the clipping baseline (~0.83) and the 43-pair descriptive caveat;
Llama stays an exploratory NO-GO replication.

**R6-2. Zero-GPU audit (already requested; descriptive).** Drop truncated samples;
restrict every row to the 599 problems mixed for both models; stratify by pass count;
per-problem weighting; A5-style text classifier on native traces with paired
probe − text-classifier difference; length-matched comparison.

**R6-3. Common candidate bank (new pre-registration).** One shared bank of correct
and incorrect answers (drawn from both models' samples on shared problems), fed to
both models by teacher forcing. Compare P(yes), hidden-state probes and length on
identical question–answer pairs; truncated answers reported separately and excluded
from the primary comparison; problem-level bootstrap.

**R6-4. Format × weights factorial (new pre-registration).** 1.5B native vs fine-tuned
weights × native-style vs terse presentation, on identical problems, candidate answers
and labels. Hidden states at comparable positions (final answer tokens; first token
after the answer); layer selection on training folds only. Interpretation table:
fine-tuning implicated if the weights factor dominates; format/readout if the
presentation factor dominates; report the interaction.

**R6-5. Probe controls (new pre-registration).** Text-feature classifier (numbers,
units, lengths, expressions, n-grams); random-weights probe; answer-tokens-only probe
vs full-trace probe; reasoning-swap control (same problem and answer, another
completion's reasoning) measuring how much the probe moves and whether it still
discriminates.

**R6-6. Later, only after R6-2 to R6-5:** probe-to-verbal-confidence distillation vs
direct supervised confidence vs no training (pairwise accuracy, Brier, selective risk,
held-out templates); then one different-family replication. Decide paper #2's framing
(representation change vs elicitation) after R6-2 to R6-5.

## References to verify before citing

Gekhman et al., "Inside-Out: Hidden Factual Knowledge in LLMs" (COLM 2025, arXiv
2503.15299); Orgad et al., "LLMs Know More Than They Show" (ICLR 2025, arXiv
2410.02707); Ni et al., "ReProbe" (ACL 2026, arXiv 2511.06209); Kossen et al.,
"Semantic Entropy Probes" (arXiv 2406.15927); Lin, Hilton & Evans, "Teaching Models to
Express Their Uncertainty in Words" (arXiv 2205.14334); Kadavath et al. (arXiv
2207.05221); Tam et al., "Let Me Speak Freely?" (arXiv 2408.02442); Levinstein &
Herrmann, "Still No Lie Detector for Language Models" (arXiv 2307.02477); Ni et al.,
"Are LLMs More Honest in Their Probabilistic or Verbalized Confidence?" (arXiv
2408.09773).
