# Getting started with logprobs-mt-qe

Logprob-based review signals for machine translation over any OpenAI-compatible LLM server (vLLM, llama.cpp, Ollama). Two workflows: translate with per-token logprobs (MT), or score existing translations without regenerating them (QE).

## Minimal setup

Requires Python 3.10+. No extra dependencies beyond the standard library for the CLI scripts.

Clone the repo and run from the project root:

```bash
git clone https://github.com/inacioinvita/logprobs-mt-qe.git
cd logprobs-mt-qe
```

Point scripts at your server with `--base-url` and `--model`. Live validation used:

| Mode | Endpoint |
|------|----------|
| MT | `http://localhost:8000/v1/chat/completions` |
| QE | `http://localhost:8000/v1/completions` |
| Model | `<model-id>` |

Replace host, port, and model id for your environment.

## What's inside

```
logprobs-mt-qe/
├── mt/                     # Translation + review signals
│   ├── translate.py        # Translate one segment live
│   ├── batch_translate.py  # Batch processing for files
│   └── lib_mt_confidence.py
├── qe/                     # Quality estimation
│   ├── score_live.py       # Score a single translation
│   ├── compare_candidates.py  # Rank multiple translations
│   ├── score_json.py       # Batch scoring from JSON
│   └── lib_prompt_logprobs.py
├── demos/                  # Offline examples (no server)
│   ├── mt/translate_with_confidence.py
│   └── qe/                 # Metric demos + rank/agreement
└── README.md, pyproject.toml, LICENSE
```

- **mt/** — generate translations and inspect logprob review signals
- **qe/** — teacher-force an existing hypothesis and read prompt logprobs
- **demos/** — offline concept demos; see [known gaps](#known-demo-gaps) below

---

## 1. Live MT — translate with review signals

```bash
python3 mt/translate.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <model-id> \
  --lang English \
  --source "A equipe de atendimento abriu um registro de reclamação, mas o pedido ficou parado pois o fornecedor ainda não enviou a confirmação de inscrição estadual corrigida nem confirmou se a cobrança do boleto seria estornada em tempo."
```

Example output:

```
Translation: The customer service team opened a complaint ticket, but the request has stalled because the supplier has not yet sent the corrected state registration confirmation nor confirmed whether the bank slip charge would be refunded in time.
Mean logprob: -0.38
Plausibility band: high (heuristic)
Terminology review signal: medium-high uncertainty
Review recommendation: review recommended

Review spans:
  "state registration confirmation"
  "bank slip charge"
  ...
```

**How to read it:** mean logprob is the raw segment-level plausibility signal. The band is a coarse heuristic, not a calibrated quality score. Use **terminology review signal**, **review recommendation**, and **review spans** to decide what a human should check.

---

## 2. Live QE — score an existing translation

QE uses `/v1/completions` with `prompt_logprobs` so the scored text matches your prompt exactly.

```bash
python3 qe/score_live.py \
  --base-url http://localhost:8000/v1/completions \
  --model <model-id> \
  --lang English \
  --source "Antes de escriturar a entrada, o fiscal pediu o XML autorizado, o manifesto do destinatário e a carta de correção, pois a nota veio com CFOP de devolução simbólica, sem destaque de DIFAL, embora a mercadoria tivesse sido remetida para uso e consumo." \
  --hypothesis "Before recording the entry, the tax inspector requested the authorized XML, the recipient's manifesto, and the correction letter, as the invoice was issued with a symbolic return CFOP without highlighting the DIFAL, even though the goods had been sent for use and consumption."
```

Look at `mean_logprob`, weakest spans, and per-token logprobs. Flags on this example include `manifesto` vs `manifest`, `as` vs `because`, and phrasing around DIFAL.

For segment-level decisions, prefer raw **`mean_logprob`** over the displayed plausibility number.

---

## 3. Compare candidates — rank translations

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

On the customer-service example, a careful domain translation (`complaint record`, `on hold`, `boleto`) ranked first at **mean_lp = -2.139**, ahead of the generated MT (**-2.436**), a generic/smoothed variant (**-2.524**), and a literal/wrong variant (**-3.421**).

Higher mean logprob means the model finds the wording more plausible — not necessarily more faithful.

---

## 4. Batch translate (experimental triage)

```bash
python3 mt/batch_translate.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <model-id> \
  --lang English \
  --input demos/data/ptbr_single.txt \
  --output results.tsv \
  --flag-for-review needs_review.txt
```

Batch translation writes raw mean logprob, a heuristic plausibility band, terminology review signal, review recommendation, ambiguous-token count, and review-span count.

---

## Offline demos (no server)

Start with demos that are known to work:

```bash
python3 demos/qe/rank_candidates.py      # candidate ranking + limitation case
python3 demos/qe/agreement.py            # self-consistency across samples
python3 demos/mt/translate_with_confidence.py
```

Other QE demos (`average_logprob.py`, `weakest_span.py`, `entropy.py`, `margin.py`, `low_confidence_spans.py`) illustrate metrics but currently have alignment gaps against bundled mock data. See below.

### Known demo gaps

| Demo / asset | Status |
|--------------|--------|
| `demos/qe/rank_candidates.py` | Works |
| `demos/qe/agreement.py` | Works |
| `demos/qe/average_logprob.py`, `margin.py`, `entropy.py`, `weakest_span.py`, `low_confidence_spans.py` | Prompt/hypothesis alignment gaps — do not trust numeric output yet |
| `QElogprob.json` | Same alignment gaps as offline QE demos |

Use live `qe/score_live.py` and `qe/compare_candidates.py` for reliable QE numbers.

---

## Key metrics

| Metric | What it measures | When to use |
|--------|------------------|-------------|
| **Mean logprob** | Overall model plausibility | Segment-level raw score |
| **Terminology review signal** | Lexical uncertainty in content tokens | Review recommendation |
| **Review spans** | Contiguous weak or ambiguous regions | Target human review |
| **Min logprob** | Weakest single token | Spot rare words / mistranslations |
| **Margin (top1 − top2)** | Decisiveness at each position | Near-synonym ambiguity |
| **Entropy** | Ambiguity across top-k | Uncertain positions |
| **Plausibility band** | Coarse mean-logprob band | Quick read only — not calibrated |
| **Perplexity proxy** | exp(−mean logprob) | Compare segments on a familiar scale |

Store raw `mean_logprob`, `min_logprob`, `margin`, `entropy`, `perplexity_proxy`, and `n_tokens` for serious evaluation.

---

## Next steps

```bash
python3 mt/translate.py --help
python3 qe/score_live.py --help
python3 qe/compare_candidates.py --help
python3 mt/batch_translate.py --help
```

**Server notes:** MT defaults to chat completions; QE defaults to completions for stable prompt alignment. Both need an endpoint that exposes logprobs.

Read the full [README](README.md) for metric detail, compatibility, and limitations (fluent ≠ faithful).
