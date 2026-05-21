# Contributing

This repo is a small extractor for logprob-based MT / QE signals. It is meant as a seed: cheap measurements that other projects can calibrate, learn from, or wrap into application-specific tools. PRs are very welcome — especially ones that push the open research questions below rather than freeze a particular interpretation of the signals.

## Open directions (good first issues)

Each item below is roughly issue-sized. Pick one, open a draft PR with your approach, and iterate.

### Calibration

1. **Correlate signals against MQM / DA.** Pick a public dataset (WMT MQM, MLQE-PE, …); plot every signal in `docs/signals.md` against the human label; report Pearson / Spearman / Kendall.
2. **Fit `composite_score` weights against human labels.** Replace the four hand-picked weights with a regression learned on a held-out set; ship the coefficients as a new preset.
3. **Per-model and per-language priors.** Compute z-scores or percentile rescales of `mean_logprob` per (model, language pair); document how much it shifts the band cutoffs.
4. **Length-bias correction.** Compare `mean_logprob`, `mean_logprob / log(n_tokens)`, content-token-only means, and a learned length regression on the same labelled set.

### Extension signals

5. **Self-consistency at scale.** Wire the `demos/qe/agreement.py` idea into a CLI; measure pairwise BLEU / chrF / content-token agreement across N samples; report cost vs signal quality.
6. **Span-level severity.** Annotate a small set of `find_uncertain_spans` outputs against MQM severities (minor / major / critical); learn a span-severity classifier.
7. **Drift heuristics beyond script.** Extend `detect_language_drift` to register shifts, code-switching markers, or named-entity transliteration.
8. **Hybrid pipelines.** Use the cheap signals to gate a heavier QE model (CometKiwi, MetricX, xCOMET) only on uncertain spans; report cost / accuracy trade-off.

### Engineering

9. **Fix offline-demo alignment.** `demos/qe/{average_logprob,weakest_span,entropy,margin,low_confidence_spans}.py` and `QElogprob.json` have prompt / hypothesis alignment gaps. The recipe is in `qe/lib_prompt_logprobs.py`.
10. **Add a `--preset` flag** to `mt/translate.py` that imports a preset module by name and prepends its output to the signal report. Keep the default empty so the CLI stays neutral.
11. **Vendor compatibility matrix.** Test against vLLM, llama.cpp, Ollama, Together, Anyscale — document JSON shape differences, prompt-template quirks, and any signal that breaks.
12. **Cross-tokeniser stability.** How much do `min_lexical_prob` and `weak_share` shift when the same translation is scored under two tokenisers? Build a small benchmark.

## How to ship a new preset

Presets are the right place for opinions. The default CLIs deliberately stay neutral.

1. Copy `mt/presets/v01_heuristic.py` to a new file, e.g. `mt/presets/wmt_mqm_calibrated.py`.
2. Replace the cuts, weights, or rescale with whatever your data supports.
3. Add a paragraph at the top of the file describing the calibration set, the model id, and the prompt template the preset was fitted on.
4. Open a PR. Add a row to `docs/signals.md` if you introduce a new signal.

## Style

- No new runtime dependencies for the extractor — keep `mt/` and `qe/` standard-library only.
- Use module-level constants for any numeric threshold so callers can override.
- Print thresholds inline anywhere a CLI emits them; never hide a magic number in a report.
- Document each new signal in `docs/signals.md` with formula, caveats, and a research hook.

## What does **not** belong here

- Hard-coded quality verdicts in the default CLIs.
- Per-domain or per-customer thresholds in the extractor.
- Calibration data, model weights, or proprietary corpora.

Those live in downstream packages, presets, or your own training repo.
