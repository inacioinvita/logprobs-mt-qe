# Gemma-4 logprob tooling (MT/QE)

Small scripts to score machine translation hypotheses with **prompt logprobs** from **local vLLM** on spark-240.

**Not a cloud API** — OpenAI-compatible JSON over HTTP on the LAN (`http://10.0.1.240:8001`), model `google/gemma-4-26B-A4B-it`, weights on GPU. No API key. For production LoRA MT use DGX-05 (`TRANSLATION_API.md`); this folder is for Gemma-4 logprob/QE experiments only.

## Quick start

```bash
cd /home/admin/logprob

# Score a saved API response (pass --hypothesis for aligned token matching)
python3 score_json.py QElogprob.json \
  --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch."

# Good vs bad comparison (QElogprob.json + live API)
python3 analyse_qe_sample.py

# Live score one hypothesis
python3 score_live.py \
  --source "The wizard casts a powerful spell." \
  --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch."

# Rank candidates (higher mean_logprob = more plausible under the model)
python3 compare_candidates.py \
  --source "The wizard casts a powerful spell." \
  --hypothesis "Der Zauberer wirkt einen mächtigen Zauberspruch." \
  --hypothesis "Der Zauberer wirft einen schwachen Zauber."
```

## Metrics


| Field              | Meaning                                                       |
| ------------------ | ------------------------------------------------------------- |
| `mean_logprob`     | Average log P(token | context) over hypothesis tokens         |
| `sum_logprob`      | Sum of token logprobs                                         |
| `perplexity_proxy` | `exp(-mean_logprob)` — lower is better                        |
| `n_tokens`         | Hypothesis token count (after marker, default `translation:`) |


## Local vLLM request (OpenAI-compatible JSON)

POST to `http://10.0.1.240:8001/v1/chat/completions` — same field names as OpenAI Chat Completions, executed on spark-240.

```bash
curl -sS -X POST 'http://10.0.1.240:8001/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-26B-A4B-it",
    "messages": [{"role": "user", "content": "Translate ...\n\nGerman translation: <hypothesis>"}],
    "temperature": 0,
    "max_tokens": 1,
    "logprobs": true,
    "top_logprobs": 5,
    "prompt_logprobs": 5
  }'
```

**Response (truncated)** — full example: `QElogprob.json`.

```json
{
  "model": "google/gemma-4-26B-A4B-it",
  "choices": [{
    "message": { "content": "..." },
    "logprobs": {
      "content": [{ "token": "...", "logprob": -0.26, "top_logprobs": [...] }]
    }
  }],
  "prompt_logprobs": [
    null,
    { "138932": { "rank": 1, "logprob": -0.30, "decoded_token": " wirkt" },
      "10175":  { "rank": 3, "logprob": -2.61, "decoded_token": " einen" } }
  ]
}
```

Parse with `score_json.py` + `--hypothesis`; see `~/knowledge-base/private_docs/GEMMA4_LOGPROB_QE_2026-05-19.md`.

## Files


| File                     | Role                                           |
| ------------------------ | ---------------------------------------------- |
| `lib_prompt_logprobs.py` | Parse vLLM `prompt_logprobs`, aggregate scores |
| `score_json.py`          | CLI for saved JSON responses                   |
| `score_live.py`          | CLI for live API scoring                       |
| `compare_candidates.py`  | Rank multiple hypotheses on one source         |


## Notes

- Uses **stdlib only** (no pip install on host).
- Scoring window starts after `--marker` (default `translation:`; matches `German translation:`).
- When the forced prompt token is not top-1, vLLM returns it with a high `rank` value; the library prefers that entry over rank-1.
- Stops at chat-template tokens (`<|channel>`, `thought`, double newline after sentence end).
- `compare_candidates.py --from-json DIR` scores pre-saved `*.json` responses without calling the API.
- Pass **`--hypothesis`** (or use `score_live.py` / `compare_candidates.py`) so tokens align to the forced target text; naive rank==1-only parsing mis-reads vLLM output.
- Logprob QE measures **model plausibility**, not human adequacy — see `~/knowledge-base/private_docs/GEMMA4_LOGPROB_QE_2026-05-19.md`.

