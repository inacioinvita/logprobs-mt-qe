# logprobs-mt-qe

Logprob-based review signals for machine translation. Works with any OpenAI-compatible LLM server (vLLM, llama.cpp, Ollama).

Two modes:

- **MT** — translate and get per-token logprobs in one call (`/v1/chat/completions`)
- **QE** — score existing translations without regenerating them (`/v1/completions` with `prompt_logprobs`)

The main value is **terminology-first review**: logprobs localise uncertainty around domain terms a human reviewer would check — not a calibrated adequacy score.

---

## What you get (live PT-BR → EN)

Examples use a local OpenAI-compatible server. Replace `http://localhost:8000` and `<model-id>` with your own endpoint and model.

### Example 1 — customer service / boleto

**Source:**

```
A equipe de atendimento abriu um registro de reclamação, mas o pedido ficou parado pois o fornecedor ainda não enviou a confirmação de inscrição estadual corrigida nem confirmou se a cobrança do boleto seria estornada em tempo.
```

**Generated MT:**

```
The customer service team opened a complaint ticket, but the request has stalled because the supplier has not yet sent the corrected state registration confirmation nor confirmed whether the bank slip charge would be refunded in time.
```

**Output highlights:**

```
Mean logprob: -0.38
Plausibility band: high (heuristic)
Terminology review signal: medium-high uncertainty
Review recommendation: review recommended
```

Review spans flag wording choices such as `customer` vs `support`, `ticket` vs `record`, and Brazilian payment/admin terms (`inscrição estadual`, `boleto`, `estornada`).

**Candidate ranking (same source):** careful domain translation wins on `mean_logprob`:

| Rank | Style | mean_lp |
|------|-------|---------|
| 1 | Careful (`complaint record`, `on hold`, `boleto`) | -2.139 |
| 2 | Generated MT | -2.436 |
| 3 | Generic / smoothed | -2.524 |
| 4 | Literal / wrong | -3.421 |

Logprobs rank plausible phrasing; they do not guarantee adequacy. A fluent-but-wrong translation can still beat a clumsier correct one.

### Example 2 — fiscal / CFOP / DIFAL

**Source:**

```
Antes de escriturar a entrada, o fiscal pediu o XML autorizado, o manifesto do destinatário e a carta de correção, pois a nota veio com CFOP de devolução simbólica, sem destaque de DIFAL, embora a mercadoria tivesse sido remetida para uso e consumo.
```

**Generated MT:**

```
Before recording the entry, the tax inspector requested the authorized XML, the recipient's manifesto, and the correction letter, as the invoice was issued with a symbolic return CFOP without highlighting the DIFAL, even though the goods had been sent for use and consumption.
```

**Output highlights:**

```
Mean logprob: -0.34
Plausibility band: high (heuristic)
Terminology review signal: medium-high uncertainty
Review recommendation: review recommended
```

Flags include `manifesto` vs `manifest`, `as` vs `because`, and `highlighting the DIFAL`.

---

## Commands

### MT: translate with review signals

```bash
python3 mt/translate.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <model-id> \
  --lang English \
  --source "A equipe de atendimento abriu um registro de reclamação, mas o pedido ficou parado pois o fornecedor ainda não enviou a confirmação de inscrição estadual corrigida nem confirmou se a cobrança do boleto seria estornada em tempo."
```

### QE: score an existing translation

```bash
python3 qe/score_live.py \
  --base-url http://localhost:8000/v1/completions \
  --model <model-id> \
  --lang English \
  --source "A equipe de atendimento abriu um registro de reclamação, mas o pedido ficou parado pois o fornecedor ainda não enviou a confirmação de inscrição estadual corrigida nem confirmou se a cobrança do boleto seria estornada em tempo." \
  --hypothesis "The customer service team opened a complaint ticket, but the request has stalled because the supplier has not yet sent the corrected state registration confirmation nor confirmed whether the bank slip charge would be refunded in time."
```

### QE: rank competing translations

```bash
python3 qe/compare_candidates.py \
  --base-url http://localhost:8000/v1/completions \
  --model <model-id> \
  --lang English \
  --source "A equipe de atendimento abriu um registro de reclamação, mas o pedido ficou parado pois o fornecedor ainda não enviou a confirmação de inscrição estadual corrigida nem confirmou se a cobrança do boleto seria estornada em tempo." \
  --hypothesis "The customer service team opened a complaint record, but the request is on hold because the supplier has not yet sent the corrected state registration confirmation nor confirmed whether the boleto charge would be refunded in time." \
  --hypothesis "The customer service team opened a complaint ticket, but the request has stalled because the supplier has not yet sent the corrected state registration confirmation nor confirmed whether the bank slip charge would be refunded in time." \
  --hypothesis "The support team opened a ticket, but the request was delayed because the supplier had not yet sent the corrected registration or confirmed whether the payment would be refunded."
```

### Batch: triage a file

```bash
python3 mt/batch_translate.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <model-id> \
  --lang English \
  --input demos/data/ptbr_single.txt \
  --output results.tsv \
  --flag-for-review needs_review.txt
```

The batch path writes raw mean logprob, a heuristic plausibility band, terminology review signal, review recommendation, ambiguous-token count, and review-span count.

---

## Scoring standard

Token-level logprobs are the source of truth. They explain *why* a segment needs review: weak tokens, ambiguous alternatives, low margins, high entropy, or review spans.

For segment-level decisions, use **`mean_logprob`** as the default raw score. It is length-normalised and safer than `sum_logprob` when comparing translations with different token counts.

**Plausibility band** is a coarse heuristic over raw `mean_logprob` (`> -1` high, `-1` to `-3` moderate, `< -3` low). It is not calibrated and should only be compared under the same model and prompt template.

**Review recommendation** should come from **terminology review signal** and **review spans** — not from a 0-1 display score.

For research or production, store raw fields separately: `mean_logprob`, `min_logprob`, `margin`, `entropy`, `perplexity_proxy`, and `n_tokens`.

## Metrics

| Metric | What it measures | Use for |
|--------|-----------------|---------|
| **Mean logprob** | Overall token-level plausibility | Segment-level raw score |
| **Min logprob** | Weakest single token | Spot mistranslations, rare words |
| **Entropy** | Ambiguity across top-k alternatives | Find positions with multiple valid options |
| **Margin (top1 − top2)** | How decisive the model was | Flag near-synonyms, close alternatives |
| **Review spans** | Contiguous weak or ambiguous regions | Target human review efficiently |
| **Terminology review signal** | Lexical uncertainty in content tokens | Drive review recommendation |
| **Agreement (n samples)** | Consistency across N generations | Detect unstable translations |
| **Plausibility band** | Coarse mean-logprob band | Quick heuristic only — not calibrated |
| **Perplexity proxy** | exp(−mean logprob) | Compare across segments |
| **n_tokens** | Number of scored target tokens | Interpret score stability and length effects |

---

## Demos

Runnable offline against included sample data — no server needed.

**MT demos** (`demos/mt/`):

| Script | Shows |
|--------|-------|
| `translate_with_confidence.py` | Full review-signal report from generation logprobs |

**QE demos** (`demos/qe/`):

| Script | Status | Shows |
|--------|--------|-------|
| `rank_candidates.py` | Works | PT-BR → EN candidate ranking and limitation case |
| `agreement.py` | Works | Self-consistency across N generations |
| `average_logprob.py` | Alignment gaps | Mean logprob as segment-level score |
| `weakest_span.py` | Alignment gaps | Finding the worst-scoring token |
| `entropy.py` | Alignment gaps | Per-token ambiguity from top-k |
| `margin.py` | Alignment gaps | Decisiveness at each position |
| `low_confidence_spans.py` | Alignment gaps | Contiguous weak regions |

Offline QE demos (`average_logprob`, `margin`, `entropy`, `weakest_span`, `low_confidence_spans`) and bundled `QElogprob.json` currently have **prompt/hypothesis alignment gaps**. Do not treat their numeric output as validated QE until fixed. Use live `qe/score_live.py` and `qe/compare_candidates.py` for reliable scoring.

```bash
python3 demos/qe/rank_candidates.py      # ranking + limitation case
python3 demos/qe/agreement.py            # self-consistency
python3 demos/mt/translate_with_confidence.py
```

---

## Structure

```
logprobs-mt-qe/
├── mt/                     # Translate + review signals
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

QE scripts default to `/v1/completions` so prompt scoring uses exactly the text sent in `prompt`. Chat completions can apply a chat template with hidden role/special tokens, making marker alignment more fragile. MT generation defaults to `/v1/chat/completions` because it asks the model to produce a translation.

---

## Limitations

- **Plausibility ≠ human adequacy** — fluent but unfaithful translations can outscore correct ones.
- **Terminology review signals drive triage** — use spans and terminology labels, not a global score alone.
- **Prompt format matters** — only compare scores from the same prompt template.
- **vLLM `prompt_logprobs` ≠ generation `logprobs`** — different JSON structures; this repo handles both.
- **Forced tokens may not appear in top-k** — always pass `--hypothesis` for accurate QE alignment.
- **Offline QE demos** — several bundled demos and `QElogprob.json` need alignment fixes before their numbers can be trusted.
