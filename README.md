# mt-qe-logprobs

Logprob-based confidence scoring for machine translation. Works with any OpenAI-compatible LLM server (vLLM, llama.cpp, Ollama).

Two modes:
- **MT** — translate and get per-token confidence in one call
- **QE** — score existing translations without regenerating them

---

## What you get

### MT: translate with confidence

```
$ python3 mt/translate.py --model gemma-4 --lang German \
    --source "The wizard casts a powerful spell."

Translation: Der Zauberer wirkt einen mächtigen Zauberspruch.
Confidence:  0.72 (high)

  token              logprob      prob   margin
  ──────────────────────────────────────────────
  Der                 -0.050    0.951    3.050
   Za                 -0.010    0.990    5.190
  uber                -0.002    0.998    6.998
  er                  -0.003    0.997    8.197
   wirkt              -0.420    0.657    0.820
   einen              -0.080    0.923    2.820
   mächt              -0.030    0.970    4.070
  igen                -0.001    0.999    7.799
   Zauber             -0.020    0.980    4.480
  spruch              -0.002    0.998    8.118
  .                   -0.010    0.990    4.890

⚠ Ambiguous: " wirkt" (margin 0.82, runner-up: " spricht")
✓ No low-confidence spans detected
```

### QE: score an existing translation

```
$ python3 qe/score_live.py --model gemma-4 --lang German \
    --source "The wizard casts a powerful spell." \
    --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch."

=== live score ===
hypothesis: ' Der Zauberer wirkt einen mächtigen Zauberspruch.'
mean_logprob=-2.190664  perplexity_proxy=8.941148  n_tokens=12

 idx  token                      logprob          prob
────────────────────────────────────────────────────────
   0   Der                    -16.196169      0.000000
   1   Za                      -7.125683      0.000804
   2  uber                     -0.001937      0.998064
   ...
  11  .                        -0.000005      0.999995
```

### QE: rank competing translations

```
$ python3 qe/compare_candidates.py --model gemma-4 --lang German \
    --source "The wizard casts a powerful spell." \
    --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch." \
    --hypothesis "Der Zauberer wirft einen schwachen Zauber."

rank  label            mean_lp   ppl_proxy  hypothesis
──────────────────────────────────────────────────────────
   1  candidate_1     -2.1907      8.9411  Der Zauberer wirkt einen mächtigen...
   2  candidate_2     -5.8660    352.8429  Der Zauberer wirft einen schwachen...
```

### Batch: triage a file

```
$ python3 mt/batch_translate.py --model gemma-4 --lang French \
    --input segments.txt --output results.tsv \
    --flag-for-review needs_review.txt

Processed 150 segments
Mean confidence: 0.71
Flagged for review: 23 (15.3%)
```

---

## Metrics

| Metric | What it measures | Use for |
|--------|-----------------|---------|
| **Mean logprob** | Overall token-level confidence | Segment-level QE score |
| **Min logprob** | Weakest single token | Spot mistranslations, rare words |
| **Entropy** | Ambiguity across top-k alternatives | Find positions with multiple valid options |
| **Margin (top1 − top2)** | How decisive the model was | Flag near-synonyms, close alternatives |
| **Low-confidence spans** | Contiguous weak regions | Target human review efficiently |
| **Agreement (n samples)** | Consistency across N generations | Detect unstable translations |
| **Confidence score** | Sigmoid-mapped mean logprob (0–1) | Quick pass/fail threshold |
| **Perplexity proxy** | exp(−mean logprob) | Compare across segments |

---

## Demos

Runnable offline against included sample data — no server needed.

**MT demos** (`demos/mt/`):

| Script | Shows |
|--------|-------|
| `translate_with_confidence.py` | Full confidence report from generation logprobs |

**QE demos** (`demos/qe/`):

| Script | Shows |
|--------|-------|
| `average_logprob.py` | Mean logprob as segment-level score |
| `weakest_span.py` | Finding the worst-scoring token |
| `entropy.py` | Per-token ambiguity from top-k |
| `margin.py` | Decisiveness at each position |
| `low_confidence_spans.py` | Contiguous weak regions |
| `agreement.py` | Self-consistency across samples |

```bash
python3 demos/qe/average_logprob.py
python3 demos/mt/translate_with_confidence.py
```

---

## Structure

```
mt-qe-logprobs/
├── mt/                     # Translate + confidence
│   ├── translate.py
│   ├── batch_translate.py
│   └── lib_mt_confidence.py
├── qe/                     # Score existing translations
│   ├── score_json.py
│   ├── score_live.py
│   ├── compare_candidates.py
│   └── lib_prompt_logprobs.py
├── demos/                  # Offline-runnable demos
│   ├── mt/
│   └── qe/
├── LICENSE                 # MIT
└── pyproject.toml
```

---

## Compatible servers

Any OpenAI-compatible endpoint:
- **vLLM** — full support (`prompt_logprobs` + generation `logprobs`)
- **llama.cpp server** — OpenAI-compatible endpoint
- **Ollama** — check version for `prompt_logprobs` support

---

## Limitations

- **Model plausibility ≠ human adequacy** — fluent but unfaithful translations can outscore correct ones.
- **Prompt format matters** — only compare scores from the same prompt template.
- **vLLM `prompt_logprobs` ≠ generation `logprobs`** — different JSON structures; this repo handles both.
- **Forced tokens may not appear in top-k** — always pass `--hypothesis` for accurate QE alignment.
