# logprob-qe

Small scripts for **hypothesis-level quality estimation** of machine translations using `prompt_logprobs` from any OpenAI-compatible LLM server (vLLM, llama.cpp, Ollama, etc.).

The idea: teacher-force a translation hypothesis into the prompt and read the token logprobs assigned by the model. Higher mean logprob → the model finds the hypothesis more plausible → lightweight reference-free QE proxy.

## Quick start

```bash
# Score a hypothesis from a saved API response
python3 score_json.py QElogprob.json \
  --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch." \
  --marker "German translation:"

# Live score — teacher-force a hypothesis against a running vLLM server
python3 score_live.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <your-model-id> \
  --lang German \
  --source "The wizard casts a powerful spell." \
  --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch."

# Rank multiple candidates (higher mean_logprob wins)
python3 compare_candidates.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <your-model-id> \
  --lang German \
  --source "The wizard casts a powerful spell." \
  --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch." \
  --hypothesis "Der Zauberer wirft einen schwachen Zauber."

# Good vs bad demo (reads QElogprob.json + calls live API for the bad hypothesis)
python3 example_qe.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <your-model-id>
```

## How it works

1. Build a prompt: `Translate the following <source-lang> text to <target-lang>:\n\n<source>\n\n<lang> translation: <hypothesis>`
2. Send to the LLM with `prompt_logprobs: N` and `max_tokens: 1`.
3. Walk the returned `prompt_logprobs` array; locate the marker (`<lang> translation:`); extract per-token logprobs after it.
4. Aggregate: mean, sum, perplexity proxy.

No generation happens — the hypothesis is forced as prompt text and scored in one forward pass.

## Metrics

| Field              | Meaning                                                    |
| ------------------ | ---------------------------------------------------------- |
| `mean_logprob`     | Average log P(token \| context) — general confidence       |
| `min_logprob`      | Worst single-token logprob — weakest span                  |
| `entropy`          | Shannon entropy over top-k — ambiguity                     |
| `margin`           | Gap between top-1 and top-2 logprob — decisiveness         |
| `low_conf_spans`   | Contiguous tokens below threshold — review targeting       |
| `agreement`        | Variance of scores across N samples — self-consistency     |
| `sum_logprob`      | Sum of token logprobs                                      |
| `perplexity_proxy` | `exp(-mean_logprob)` — lower is better                     |
| `n_tokens`         | Hypothesis token count (after marker)                      |

## Examples — didactic metric demos

The `examples/` directory contains one script per QE metric, each runnable offline against the included sample data:

| Script | Metric | What it shows |
|--------|--------|---------------|
| `01_average_logprob.py` | Average token logprob | General translation confidence |
| `02_weakest_span.py` | Minimum token logprob | Worst-scoring token detection |
| `03_entropy.py` | Entropy | Ambiguity at each token position |
| `04_margin.py` | Margin (top1 − top2) | Decoder decisiveness |
| `05_low_confidence_spans.py` | Low-confidence spans | Contiguous weak regions for review |
| `06_agreement.py` | Agreement across samples | Self-consistency across N generations |

```bash
cd examples
python3 01_average_logprob.py
python3 02_weakest_span.py
# ... etc
```

## Curl example

```bash
curl -sS -X POST 'http://localhost:8000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<your-model-id>",
    "messages": [{"role": "user", "content": "Translate the following text to German:\n\nThe wizard casts a powerful spell.\n\nGerman translation: Der Zauberer wirkt einen mächtigen Zauberspruch."}],
    "temperature": 0,
    "max_tokens": 1,
    "logprobs": true,
    "top_logprobs": 5,
    "prompt_logprobs": 5
  }'
```

Save the response to a `.json` file and pass it to `score_json.py` with `--hypothesis`.

**Response structure (truncated)** — full example: `QElogprob.json`.

```json
{
  "prompt_logprobs": [
    null,
    { "138932": { "rank": 1, "logprob": -0.30, "decoded_token": " wirkt" },
      "10175":  { "rank": 3, "logprob": -2.61, "decoded_token": " einen" } }
  ]
}
```

## Files

| File                     | Role                                                              |
| ------------------------ | ----------------------------------------------------------------- |
| `lib_prompt_logprobs.py` | Core library: parse `prompt_logprobs`, aggregate scores           |
| `score_json.py`          | CLI — score a saved JSON response                                 |
| `score_live.py`          | CLI — teacher-force a hypothesis against a live API               |
| `compare_candidates.py`  | CLI — rank multiple hypotheses on one source                      |
| `example_qe.py`          | Demo — good vs bad German hypothesis comparison                   |
| `QElogprob.json`         | Saved vLLM response for the wizard example (good hypothesis)      |

## Compatible servers

Any server that returns `prompt_logprobs` in the OpenAI chat completions format:

- **vLLM** — native support, pass `"prompt_logprobs": N` in the request body
- **llama.cpp server** — supports `prompt_logprobs` via OpenAI-compatible endpoint
- **Ollama** — check your version; `prompt_logprobs` support varies
- Any other OpenAI-compatible endpoint that honours the `prompt_logprobs` field

## Notes

- Uses **stdlib only** — no `pip install` required.
- Scoring window starts after `--marker` (defaults to `<lang> translation:` in live scripts).
- When the forced prompt token is not top-1, vLLM returns it with a high `rank` value; the library selects that entry over rank-1.
- Stops at chat-template tokens (`<|channel>`, `thought`, double newline after sentence end).
- `compare_candidates.py --from-json DIR` scores pre-saved `*.json` responses without calling the API.
- Always pass `--hypothesis` (or use `score_live.py` / `compare_candidates.py`) so tokens align to the forced target text; naive rank==1-only parsing can mis-read vLLM output.

## Limitations

- **Model plausibility, not human adequacy** — a fluent but unfaithful translation can outscore a correct one if the model finds it more "natural".
- **Prompt format matters** — logprob scores are relative to the exact prompt template used. Compare only hypotheses scored with the same prompt.
- **vLLM `prompt_logprobs` differs from generation logprobs** — the JSON structure uses token-id keys, not a flat list; this library handles that format.
- **Forced tokens may not appear in top-k** — always pass `--hypothesis` so the library can locate the exact forced-token entry regardless of rank.
