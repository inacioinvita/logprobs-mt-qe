# logprobs-mt-qe

Logprob-based review signals for machine translation. Works with any OpenAI-compatible LLM server (vLLM, llama.cpp, Ollama).

Two modes:

- **MT** — translate and read per-token generation logprobs in one call (`/v1/chat/completions` with `logprobs: true`).
- **QE** — score an already-produced hypothesis without regenerating it (`/v1/completions` with `prompt_logprobs`).

The repo exposes per-token logprobs and a small set of derived signals (`mean_logprob`, `min_logprob`, `margin`, `entropy`, `perplexity_proxy`, low-confidence spans, language-script drift, and a few distributional summaries). Qualitative bands, review thresholds, and quality labels are deliberately left to a calibration layer the caller chooses — see [`mt/presets/v01_heuristic.py`](mt/presets/v01_heuristic.py) for one illustrative example.

The core idea: model-internal logprobs are cheap, reference-free, and localise *where* a translation needs a second look. Mapping them onto a calibrated quality score is an open research question, not something this repo prescribes.

---

## What you get (live PT-BR → EN)

Examples assume an OpenAI-compatible server at `http://localhost:8000` and a model id of `<model-id>`.

### Example 1 — customer service / boleto

**Source:**

```
A equipe de atendimento abriu um registro de reclamação, mas o pedido ficou parado pois o fornecedor ainda não enviou a confirmação de inscrição estadual corrigida nem confirmou se a cobrança do boleto seria estornada em tempo.
```

**Generated MT:**

```
The customer service team opened a complaint ticket, but the request has stalled because the supplier has not yet sent the corrected state registration confirmation nor confirmed whether the bank slip charge would be refunded in time.
```

**Signals (excerpt):**

```
Aggregate signals
  mean_logprob          -0.380
  perplexity_proxy       1.462      (exp(-mean_logprob))
  display_score          90.5%      (piecewise rescale; anchors: 0→100%, -1→75%, -3→25%, -6→0%)

Distribution signals (content tokens)
  composite_score        0.840
  min_lexical_prob       0.510
  mean_top_entropy       0.342 nats
  mean_margin            2.100
  weak_share             0.040      (prob<0.6, or margin<0.4, or logprob<-0.75)
  ambiguous_share        0.090      (prob<0.8 or margin<1.0)

Review spans (n=2, seed logprob threshold -0.75)
  ... "state registration confirmation" ...
  ... "bank slip charge" ...
```

Span flags pick up wording choices such as `customer` vs `support`, `ticket` vs `record`, and Brazilian payment / admin terms (`inscrição estadual`, `boleto`, `estornada`).

**Candidate ranking on the same source:**

| Rank | Style | mean_lp |
|------|-------|---------|
| 1 | Careful (`complaint record`, `on hold`, `boleto`) | -2.139 |
| 2 | Generated MT | -2.436 |
| 3 | Generic / smoothed | -2.524 |
| 4 | Literal / wrong | -3.421 |

Logprobs rank plausible phrasing; they do not guarantee adequacy. Fluent-but-wrong can still beat clumsier-but-correct.

### Example 2 — fiscal / CFOP / DIFAL

**Source:**

```
Antes de escriturar a entrada, o fiscal pediu o XML autorizado, o manifesto do destinatário e a carta de correção, pois a nota veio com CFOP de devolução simbólica, sem destaque de DIFAL, embora a mercadoria tivesse sido remetida para uso e consumo.
```

**Generated MT:**

```
Before recording the entry, the tax inspector requested the authorized XML, the recipient's manifesto, and the correction letter, as the invoice was issued with a symbolic return CFOP without highlighting the DIFAL, even though the goods had been sent for use and consumption.
```

Spans surface `manifesto` vs `manifest`, `as` vs `because`, and the phrase around `highlighting the DIFAL`.

---

## Commands

### MT: translate with signals

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
  --source "..." \
  --hypothesis "..."
```

### QE: rank competing translations

```bash
python3 qe/compare_candidates.py \
  --base-url http://localhost:8000/v1/completions \
  --model <model-id> \
  --lang English \
  --source "..." \
  --hypothesis "candidate A" \
  --hypothesis "candidate B" \
  --hypothesis "candidate C"
```

### Batch: write raw signals to TSV

```bash
python3 mt/batch_translate.py \
  --base-url http://localhost:8000/v1/chat/completions \
  --model <model-id> \
  --lang English \
  --input demos/data/ptbr_single.txt \
  --output results.tsv
```

The TSV holds the raw measurements only (see [`docs/signals.md`](docs/signals.md)). Triage cutoffs — which rows count as "needs review" — are intentionally left to the caller; filter the TSV with awk / pandas / SQL using whatever cutoffs make sense for your domain.

---

## 

| Layer | Where | Notes |
|-------|-------|-------|
| Per-token logprobs | `mt/lib_mt_confidence.py`, `qe/lib_prompt_logprobs.py` | Source of truth |
| Per-token derived signals | same files | top1−top2 margin, top-k entropy, runner-up |
| Aggregates | same files | `mean_logprob`, `sum_logprob`, `min_logprob`, `max_logprob`, `perplexity_proxy`, `n_tokens` |
| Distributional summary | `content_token_summary` | mean/min lexical prob, weak/ambiguous shares, mean entropy, mean margin, composite score |
| Span detector | `find_uncertain_spans`, `find_low_confidence_spans` | Contiguous weak / ambiguous regions |
| Language-script drift | `detect_language_drift` | Top-k alternatives in a different Unicode script |
| Candidate ranker | `qe/compare_candidates.py` | Order by `mean_logprob` |
| Optional preset | `mt/presets/v01_heuristic.py` | One illustrative mapping from numbers onto bands; **opt-in, not wired into the CLIs** |

All thresholds the code uses internally are exposed as module constants in `mt.lib_mt_confidence` and printed inline by `format_signal_report` so they are use as knobs rather than laws.

---

## Open directions

This repo is intentionally a seed. The signals it exposes are the input to a much larger question: *how do we turn cheap model-internal uncertainty into helpful QE signal?* Concrete extension paths:

1. **Calibration against human judgement.** Correlate `mean_logprob`, `min_logprob`, `mean_top_entropy`, and `composite_score` against MQM, DA, or post-edit effort on a held-out set. Learn band cutoffs instead of hand-picking them.
2. **Per-model and per-language priors.** Logprob scales drift across models and language pairs; a per-pair recalibration (z-score or percentile mapping) is an obvious next step.
3. **Length-bias correction.** `mean_logprob` rewards short translations. Try `mean_logprob / log(n_tokens)`, content-token-only means, or a length-conditioned regression.
4. **Self-consistency / agreement.** Sample N translations at `temperature > 0`; measure agreement, BLEU/chrF among samples, or content-token disagreement. The agreement demo is wired but unused at the CLI surface.


PRs and issues exploring any of these are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## Demos

Runnable offline against included sample data — no server needed.

**MT demos** (`demos/mt/`):

| Script | Shows |
|--------|-------|
| `translate_with_confidence.py` | Full signal report from generation logprobs |

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

The five offline QE demos with the "alignment gaps" label, plus the bundled `QElogprob.json`, currently have prompt / hypothesis alignment issues. Fixing them is a good first contribution; the alignment recipe is in `qe/lib_prompt_logprobs.py`.

```bash
python3 demos/qe/rank_candidates.py
python3 demos/qe/agreement.py
python3 demos/mt/translate_with_confidence.py
```

---

## Structure

```
logprobs-mt-qe/
├── mt/
│   ├── translate.py
│   ├── batch_translate.py
│   ├── lib_mt_confidence.py
│   └── presets/
│       └── v01_heuristic.py
├── qe/
│   ├── score_json.py
│   ├── score_live.py
│   ├── compare_candidates.py
│   └── lib_prompt_logprobs.py
├── demos/
│   ├── mt/
│   └── qe/
├── docs/
│   └── signals.md
├── notebooks/
│   └── calibrate_against_mqm.ipynb
├── CONTRIBUTING.md
├── GETTING_STARTED.md
├── LICENSE
└── pyproject.toml
```

---

## Compatible servers

Any OpenAI-compatible endpoint:

- **vLLM** — full support (`prompt_logprobs` + generation `logprobs`).
- **llama.cpp server** — OpenAI-compatible endpoint.
- **Ollama** — check version for `prompt_logprobs` support.

QE scripts default to `/v1/completions` so prompt scoring uses exactly the text sent in `prompt`. Chat completions can apply a chat template with hidden role / special tokens, making marker alignment more fragile. MT generation defaults to `/v1/chat/completions` because it asks the model to produce a translation.

---

## Limitations

- **Plausibility ≠ human adequacy.** Fluent-but-unfaithful translations can outscore correct ones.
- **Prompt format matters.** Only compare scores from the same prompt template.
- **vLLM `prompt_logprobs` ≠ generation `logprobs`.** Different JSON structures; this repo handles both.
- **Forced tokens may not appear in top-k.** Pass `--hypothesis` for accurate QE alignment.
- **Offline QE demos.** Several bundled demos and `QElogprob.json` need alignment fixes before their numbers can be trusted.
