# Getting started with logprobs-mt-qe

This guide is a **canvas**, not a recipe. The repo ships a small set of cheap, model-internal signals extracted from per-token logprobs. What you turn them into — a quality score, a triage policy, a CAT-tool plug-in, a research dataset, a learned QE model — is open. The pages below lay out every signal, the formula behind it, what it cannot tell you, and a few research hooks worth pulling on.

## Minimal setup

Requires Python 3.10+. The CLIs use only the standard library.

```bash
git clone https://github.com/inacioinvita/logprobs-mt-qe.git
cd logprobs-mt-qe
```

Point any script at your OpenAI-compatible server with `--base-url` and `--model`:

| Mode | Endpoint |
|------|----------|
| MT | `http://localhost:8000/v1/chat/completions` |
| QE | `http://localhost:8000/v1/completions` |

## Two workflows

```bash
# MT — translate and read generation logprobs in one call
python3 mt/translate.py --base-url <…> --model <…> --lang English --source "..."

# QE — score a fixed hypothesis against the model (teacher-forced)
python3 qe/score_live.py --base-url <…> --model <…> --lang English \
    --source "..." --hypothesis "..."

# Rank candidates by mean logprob
python3 qe/compare_candidates.py --base-url <…> --model <…> --lang English \
    --source "..." --hypothesis "A" --hypothesis "B" --hypothesis "C"

# Batch — write raw signals to TSV
python3 mt/batch_translate.py --base-url <…> --model <…> --lang English \
    --input segments.txt --output results.tsv
```

The MT default report prints raw measurements only. Thresholds and rescale anchors are printed inline so you see them as knobs.

---

## The signal canvas

Each signal has the same shape: **formula → applications → caveats**.

### `mean_logprob`

- **Formula.** `sum(token_logprobs) / n_tokens`. Length-normalised mean over the generated (or teacher-forced) translation.
- **Applications.** Segment-level triage, candidate ranking, n-best reranking, large-batch sorting before human review.
- **Caveats.** Penalises rare-but-correct word choices; rewards short, generic translations; cannot tell you *why* the model is uncertain; comparable across segments only under the same model and prompt template.

### `min_logprob` / `min_lexical_prob`

- **Formula.** The single weakest token (`min_logprob`) or the lowest-probability lexical (non-punctuation, non-function-word) token (`min_lexical_prob`).
- **Applications.** Highlight a candidate position to a reviewer; gate a heavier QE model only on segments where `min_lexical_prob` falls below some cutoff.
- **Caveats.** A single weak token can be a rare name, an idiom, or just a tokeniser quirk; sensitive to tokenisation; one weak token in a long sentence may be benign.

### `top1 − top2` margin

- **Formula.** Per-token: `top1.logprob - top2.logprob`. Larger = more decisive.
- **Applications.** Flag near-synonym ambiguity; surface alternatives in a CAT tool; build a "second-best" suggestion stream alongside the MT.
- **Caveats.** Margin shrinks at any position with many valid synonyms; "decisive" is not the same as "correct"; depends on the top-k width the server returns.

### `entropy` (top-k Shannon entropy, nats)

- **Formula.** Per-token: `-Σ q_i · log(q_i)` over the top-k probability mass `q_i`, after normalising the top-k probabilities to sum to 1.
- **Applications.** Detect ambiguous positions where multiple valid options exist; drive an interactive "show alternatives" UI; feed entropy as a feature into a learned QE model.
- **Caveats.** Truncated to the top-k the server reports (default 5 here); high entropy on a function-word position is often noise; sensitive to temperature in the original decoding.

### `perplexity_proxy`

- **Formula.** `exp(-mean_logprob)`. Familiar perplexity scale.
- **Applications.** Cross-segment comparison on an intuitive scale; thresholding in a familiar unit; feature in dashboards.
- **Caveats.** Same caveats as `mean_logprob`; exponential rescale exaggerates differences in the tails.

### `display_score` (piecewise-linear rescale)

- **Formula.** Piecewise-linear map from `mean_logprob` onto `[0, 100]`. Default anchors `(0 → 100%, -1 → 75%, -3 → 25%, -6 → 0%)`. Override the anchors to fit your model.
- **Applications.** Human-readable dashboards, demos, screenshots — anywhere a percentage reads better than `-0.380`.
- **Caveats.** Purely cosmetic. It is *not* a calibrated quality score; it is a readability rescale. Two different models will produce two different rescales for the same underlying quality.

### `composite_score` (content-token weighted)

- **Formula.** `0.50 · mean_prob + 0.25 · min_lexical_prob + 0.15 · (1 − weak_share) + 0.10 · (1 − ambiguous_share)`. Range `[0, 1]`.
- **Applications.** A single number to sort by in a dashboard; default sort key for batch TSVs; baseline against which to compare a learned QE model.
- **Caveats.** The four weights are illustrative, not learned; sensitive to the function-word / punctuation filter; tells you nothing about adequacy.

### `weak_share` and `ambiguous_share`

- **Formula.** Share of content tokens flagged by `is_low_confidence_token` (`prob < 0.60` or `margin < 0.40` or `logprob < -0.75`) or by `is_ambiguous_token` (`prob < 0.80` or `margin < 1.00`).
- **Applications.** Spot-check segments dominated by weak content tokens; build cohort plots ("X % of segments have weak_share > 0.10"); feed both shares into a regression against human edit distance.
- **Caveats.** The thresholds are knobs, not constants of nature; the content-token filter (`is_punctuation_or_function_word`) is English-biased.

### Review spans (`find_uncertain_spans`)

- **Formula.** Contiguous runs of content tokens that are weak / ambiguous / below the seed logprob threshold (default `-0.75`), expanded forward by up to two trailing content tokens to capture context.
- **Applications.** Highlight a region for a reviewer; export to a CAT tool as a comment / annotation; drive a "fix this span" agent.
- **Caveats.** Spans bleed across punctuation; the seed threshold is hand-picked; long spans can swallow short, locally bad regions.

### Language-script drift (`detect_language_drift`)

- **Formula.** Any token whose top-k alternatives include candidates in a different Unicode script (Latin, Cyrillic, Arabic, CJK, Devanagari).
- **Applications.** Catch code-switch leaks (e.g. a Cyrillic alternative in a German translation); flag transliterated brand names; safety signal in multilingual deployments.
- **Caveats.** Cheap heuristic — flags transliteration and code-switching alike; needs an explicit `target_script` per language pair; says nothing about content fidelity.

### Agreement across samples (self-consistency)

- **Formula.** Sample N translations at `temperature > 0`; measure pairwise BLEU / chrF, content-token agreement, or simple n-gram overlap.
- **Applications.** Detect unstable translations before they reach a human; weight a candidate by self-agreement; trigger reruns at higher temperature on borderline segments.
- **Caveats.** Costs N × the inference budget; agreement on fluency is easier than agreement on adequacy; temperature is a confound.

### `n_tokens`, `n_content_tokens`, `n_lexical_tokens`

- **Formula.** Counts produced during signal extraction.
- **Applications.** Sanity-check segment length; normalise sums; debug the function-word filter.
- **Caveats.** Tokeniser-dependent; not comparable across models.

---

## How to use the canvas

A few paths through the same signals — none of them privileged by the repo.

- **Quick eyeball.** Run `mt/translate.py`; read the inline report; trust the `display_score` only as a rescale, not as a verdict.
- **Spreadsheet triage.** Run `mt/batch_translate.py`; load the TSV; sort or filter by whatever signals your team agrees on; revisit thresholds after each batch.
- **Calibration project.** Pick a labelled corpus (MQM, DA, post-edit time); regress each signal individually, then in combination, against human scores; publish your coefficients alongside the model id and prompt template.
- **Live CAT tool.** Stream signals into a sidebar — `mean_logprob` as a status, review spans as highlights, runner-up tokens as suggestions.
- **Research seed.** Use the raw extraction as a frozen dependency; build a separate package that owns the calibration, banding, or learned QE model.

---

## Surfacing your own rubric

If you want qualitative labels (`high`, `moderate`, `needs review`, MQM categories…) wire them at the edges of your own pipeline rather than in this repo. The shape used by `mt/presets/v01_heuristic.py` is illustrative:

```python
from mt.lib_mt_confidence import (
    content_token_summary, generation_tokens_from_response, mean_logprob,
)
from mt.presets.v01_heuristic import band, terminology_hint

# tokens = generation_tokens_from_response(api_response)
mean_lp = mean_logprob([t["logprob"] for t in tokens])
summary = content_token_summary(tokens)

print("band:", band(mean_lp))
print("terminology hint:", terminology_hint(summary))
```

To plug your own preset, copy `v01_heuristic.py`, swap the cuts, and import from there. Future contributors can ship calibrated, learned, or domain-specific presets without touching the core extractor.

A natural place to start: add a `--preset` argument to `mt/translate.py` that imports a preset module by name and prepends its output to the signal report. The repo deliberately does not ship that flag today — see if your application needs it before adding a knob.

---

## Offline demos

```bash
python3 demos/qe/rank_candidates.py
python3 demos/qe/agreement.py
python3 demos/mt/translate_with_confidence.py
```

The other QE demos (`average_logprob.py`, `weakest_span.py`, `entropy.py`, `margin.py`, `low_confidence_spans.py`) and the bundled `QElogprob.json` currently have prompt / hypothesis alignment gaps. Fixing them is a good first contribution.

---

## Where to look next

- [`docs/signals.md`](docs/signals.md) — one-page reference: signal → formula → applications → caveats.
- [`notebooks/calibrate_against_mqm.ipynb`](notebooks/calibrate_against_mqm.ipynb) — empty calibration notebook stub; bring your own labelled data.
- [`mt/presets/v01_heuristic.py`](mt/presets/v01_heuristic.py) — one illustrative qualitative mapping.
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — open directions and good-first-issue list.

```bash
python3 mt/translate.py --help
python3 qe/score_live.py --help
python3 qe/compare_candidates.py --help
python3 mt/batch_translate.py --help
```
