# Signals reference

One-page summary of every signal `logprobs-mt-qe` exposes. For longer discussion (research hooks, applications, caveats) see [`../GETTING_STARTED.md`](../GETTING_STARTED.md).

| Signal | Formula | Defined in | Known caveats | Research hook |
|--------|---------|-----------|---------------|---------------|
| `mean_logprob` | `sum(token_logprobs) / n_tokens` | `mt.lib_mt_confidence`, `qe.lib_prompt_logprobs` | Length-normalised; comparable only under same model + prompt | Correlate vs MQM / DA / post-edit time |
| `sum_logprob` | `sum(token_logprobs)` | `qe.lib_prompt_logprobs.aggregate_scores` | Penalises long translations | Use only with explicit length normalisation |
| `min_logprob` | `min(token_logprobs)` | `qe.lib_prompt_logprobs.aggregate_scores` | One token may be a rare name / idiom | Confusion matrix vs error spans |
| `max_logprob` | `max(token_logprobs)` | `qe.lib_prompt_logprobs.aggregate_scores` | Almost always punctuation; sanity check only | Outlier debugging |
| `geometric_mean_prob` | `exp(mean_logprob)` | `mt.lib_mt_confidence`, `qe.lib_prompt_logprobs` | Same caveats as `mean_logprob`; values saturate near high confidence | Compare with Park et al. (2025) geometric token confidence |
| `perplexity_proxy` | `exp(-mean_logprob)` | both | Same caveats as `mean_logprob`; exaggerates tails | Familiar-scale dashboards |
| `display_score` | Piecewise-linear rescale of `mean_logprob` (default anchors `0→100%, -1→75%, -3→25%, -6→0%`) | `mt.lib_mt_confidence.plausibility_score` | Cosmetic; not calibrated quality | Replace with learned isotonic regression |
| `n_tokens` | Number of scored tokens | both | Tokeniser-dependent | Feature for length-bias correction |
| Per-token `prob` | `exp(logprob)` | `mt.lib_mt_confidence` | Tokenisation artefacts | Token-level error localisation |
| `top1 − top2` margin | `top1.logprob - top2.logprob` | `mt.lib_mt_confidence.token_margin`, `qe.lib_prompt_logprobs.margin_top1_top2` | Shrinks at every synonym-rich position | Predict reviewer rewrites |
| `entropy` (nats) | `-Σ q_i log q_i` over normalised top-k probs | `mt.lib_mt_confidence.token_entropy`, `qe.lib_prompt_logprobs.token_entropy` | Truncated to top-k; sensitive to decoding temperature | Feature for learned QE |
| `top-k kurtosis` | `E[(p−mean)^4] / variance^2` over returned top-k probs | `mt.lib_mt_confidence.token_topk_kurtosis`, `qe.lib_prompt_logprobs.token_topk_kurtosis` | Only as good as returned top-k; QE needs `--hypothesis` alignment for per-position top-k | Token-distribution peakedness from uncertainty visualisation work |
| `runner_up` | Token at rank 2 | `mt.lib_mt_confidence.runner_up` | Empty when `top_logprobs < 2` | Surface as suggestion in CAT tool |
| `composite_score` | `0.50·mean_prob + 0.25·min_lexical_prob + 0.15·(1−weak_share) + 0.10·(1−ambig_share)` | `mt.lib_mt_confidence.content_token_summary` | Weights are illustrative, not learned | Fit weights against human labels |
| `mean_prob_all` | Mean `prob` over all scored tokens | `mt.lib_mt_confidence`, `qe.lib_prompt_logprobs.aggregate_scores` | Includes punctuation and function words | Compare with Park et al. (2025) arithmetic token confidence |
| `mean_prob` | Mean `prob` over content tokens | `mt.lib_mt_confidence.content_token_summary` | Function-word filter is English-biased | — |
| `min_lexical_prob` | Lowest `prob` over lexical tokens | `mt.lib_mt_confidence.content_token_summary` | Single rare token can dominate | Spot rare-word errors |
| `mean_top_entropy` | Mean per-token entropy over content tokens | `mt.lib_mt_confidence.content_token_summary` | Top-k truncated | Combine with margin into composite |
| `mean_topk_kurtosis` | Mean top-k kurtosis over scored tokens / positions | `mt.lib_mt_confidence.content_token_summary`, `qe.lib_prompt_logprobs.aggregate_position_scores` | QE: requires `--hypothesis` for aligned positions; not comparable across top-k widths | Validate vs BLEU / chrF / MQM before treating as confidence |
| `mean_margin` | Mean `top1 − top2` over content tokens | `mt.lib_mt_confidence.content_token_summary` | Same caveats as per-token margin | — |
| `weak_share` | Share of content tokens with `prob < 0.60` or `margin < 0.40` or `logprob < -0.75` | `mt.lib_mt_confidence.content_token_summary` | Thresholds are knobs | Sweep cuts vs human labels |
| `ambiguous_share` | Share of content tokens with `prob < 0.80` or `margin < 1.00` | `mt.lib_mt_confidence.content_token_summary` | Same | Same |
| `n_content_tokens`, `n_lexical_tokens` | Counts under the function-word / punctuation filters | `mt.lib_mt_confidence.content_token_summary` | Filter is English-biased | Localise to your target language |
| Review spans (MT) | Contiguous content tokens flagged as weak / ambiguous / `logprob < -0.75`, expanded by up to 2 trailing content tokens | `mt.lib_mt_confidence.find_uncertain_spans` | Bleeds across punctuation | Map onto MQM severities |
| Low-confidence spans (QE) | Contiguous tokens with `logprob < -2.0` | `qe.lib_prompt_logprobs.find_low_confidence_spans` | Threshold is a knob | Learn per-pair threshold |
| Language-script drift | Top-k alternatives in a different Unicode script | `mt.lib_mt_confidence.detect_language_drift` | Says nothing about content fidelity | Hybrid with language-id classifiers |
| Candidate ranking | Sort hypotheses by `mean_logprob` | `qe/compare_candidates.py` | Length bias; no length penalty | Per-pair tuned scoring (e.g. min logprob tiebreaker) |
| Agreement across samples | Sample N at `temperature > 0`, measure pairwise agreement | `demos/qe/agreement.py` | Costs N × inference; temperature confound | Disagreement on content tokens only |

## Thresholds and knobs

All thresholds the extractor uses internally are exposed as module-level constants in `mt.lib_mt_confidence`:

```python
AMBIGUOUS_PROB_CUT       = 0.80
AMBIGUOUS_MARGIN_CUT     = 1.00
WEAK_PROB_CUT            = 0.60
WEAK_MARGIN_CUT          = 0.40
WEAK_LOGPROB_CUT         = -0.75
SPAN_LOGPROB_THRESHOLD   = -0.75
PLAUSIBILITY_ANCHORS     = [(0, 100), (-1, 75), (-3, 25), (-6, 0)]
```

These are illustrative defaults. Override them by passing arguments to the classifier functions, or fork them in your own preset module under `mt/presets/`.

## Recommended default QE score

Use **`mean_logprob`** as the one-stop segment score for QE (`score_live.py`, `compare_candidates.py`). Access it via `segment_qe_score(agg)` in `qe/lib_prompt_logprobs.py`.

| If you need… | Use |
|--------------|-----|
| Rank candidates / triage segments | `mean_logprob` (higher = model finds hypothesis more plausible) |
| Break ties between similar means | `min_logprob` (penalise one bad token) |
| Human-readable 0–1 scale | `geometric_mean_prob` or `mean_prob_all` — same ranking as `mean_logprob` on one segment; values saturate near 1.0 |
| Familiar perplexity scale | `perplexity_proxy` — monotone transform of `mean_logprob` |
| Ambiguity / peakedness (supplementary) | `mean_topk_kurtosis` — not calibrated adequacy; needs `--hypothesis` alignment |

Do **not** use `mean_topk_kurtosis` or Park metrics alone as a quality verdict until calibrated on your data. For production gates, learn cutoffs from MQM / post-edit labels (`notebooks/calibrate_against_mqm.ipynb`).

## What this reference does **not** cover

- How to choose thresholds for your domain — see `notebooks/calibrate_against_mqm.ipynb` for a starting point.
- Mapping signals onto a learned quality score — open research.
- Vendor-specific server quirks — see the README "Compatible servers" section.

## Related uncertainty visualisation work

Park et al. (2025), ["Visualizing Uncertainty in Translation Tasks: An Evaluation of LLM Performance and Confidence Metrics"](https://arxiv.org/abs/2501.17187), evaluates geometric token probability, arithmetic token probability, and top-k kurtosis for MT uncertainty, then visualises token confidence with a colour gradient. This repo exposes the first two directly (`geometric_mean_prob`, `mean_prob_all`) and a truncated OpenAI-compatible version of top-k kurtosis (`top-k kurtosis`, `mean_topk_kurtosis`). The main caveat is API width: their kurtosis experiments use much wider distributions than most OpenAI-compatible `top_logprobs` responses expose.
