#!/usr/bin/env python3
"""Translate with review signals: generation logprob scoring.

Generation logprobs score the model's OWN output — no hypothesis needed.
See where the model's token probabilities are weakest.

Unlike prompt_logprobs (QE demos), which teacher-force a *fixed* hypothesis
and measure how plausible the model finds it, generation logprobs come free
with every translation call.

Key concepts:
  - Mean logprob: raw length-normalised generation plausibility.
  - Ambiguous tokens: close margin between top-1 and top-2.
  - Review spans: contiguous regions where the model struggled.

Offline demo
------------
Runs against a hard-coded mock response (no server needed)::

    python3 demos/mt/translate_with_confidence.py

Live demo
---------
Calls a real API endpoint::

    python3 demos/mt/translate_with_confidence.py --live \\
      --base-url http://localhost:8000/v1/chat/completions \\
      --model <model-id>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)

from mt.lib_mt_confidence import (
    find_uncertain_spans,
    flag_ambiguous_tokens,
    format_review_report,
    generation_tokens_from_response,
    mean_logprob,
    terminology_assessment,
)

MOCK_RESPONSE = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": (
                "The customer service team opened a support ticket, but the request "
                "stalled because the supplier had still not sent the corrected state "
                "registration details or confirmed whether the boleto charge would be "
                "reversed."
            ),
        },
        "logprobs": {
            "content": [
                {
                    "token": "The customer service team",
                    "logprob": -0.34,
                    "top_logprobs": [
                        {"token": "The customer service team", "logprob": -0.34},
                        {"token": "The support team", "logprob": -1.16},
                        {"token": "The customer care team", "logprob": -1.90},
                    ],
                },
                {
                    "token": " opened a",
                    "logprob": -0.28,
                    "top_logprobs": [
                        {"token": " opened a", "logprob": -0.28},
                        {"token": " created a", "logprob": -0.90},
                        {"token": " filed a", "logprob": -1.60},
                    ],
                },
                {
                    "token": " support ticket",
                    "logprob": -0.52,
                    "top_logprobs": [
                        {"token": " support ticket", "logprob": -0.52},
                        {"token": " ticket", "logprob": -0.95},
                        {"token": " case", "logprob": -1.70},
                    ],
                },
                {
                    "token": ", but the",
                    "logprob": -0.10,
                    "top_logprobs": [
                        {"token": ", but the", "logprob": -0.10},
                        {"token": ", however the", "logprob": -1.60},
                        {"token": " and the", "logprob": -2.20},
                    ],
                },
                {
                    "token": " request stalled",
                    "logprob": -0.86,
                    "top_logprobs": [
                        {"token": " request stalled", "logprob": -0.86},
                        {"token": " request was delayed", "logprob": -1.03},
                        {"token": " case remained pending", "logprob": -1.18},
                    ],
                },
                {
                    "token": " because the",
                    "logprob": -0.08,
                    "top_logprobs": [
                        {"token": " because the", "logprob": -0.08},
                        {"token": " since the", "logprob": -1.80},
                        {"token": " as the", "logprob": -2.10},
                    ],
                },
                {
                    "token": " supplier",
                    "logprob": -0.21,
                    "top_logprobs": [
                        {"token": " supplier", "logprob": -0.21},
                        {"token": " vendor", "logprob": -1.12},
                        {"token": " provider", "logprob": -1.70},
                    ],
                },
                {
                    "token": " had still not sent the corrected",
                    "logprob": -0.24,
                    "top_logprobs": [
                        {"token": " had still not sent the corrected", "logprob": -0.24},
                        {"token": " had not yet sent the corrected", "logprob": -0.70},
                        {"token": " still had not provided the corrected", "logprob": -1.20},
                    ],
                },
                {
                    "token": " state registration details",
                    "logprob": -1.32,
                    "top_logprobs": [
                        {"token": " state registration details", "logprob": -1.32},
                        {"token": " state tax registration details", "logprob": -1.38},
                        {"token": " corrected registration", "logprob": -1.55},
                    ],
                },
                {
                    "token": " or confirmed whether the",
                    "logprob": -0.35,
                    "top_logprobs": [
                        {"token": " or confirmed whether the", "logprob": -0.35},
                        {"token": " nor confirmed if the", "logprob": -0.90},
                        {"token": " or said whether the", "logprob": -2.10},
                    ],
                },
                {
                    "token": " boleto charge",
                    "logprob": -1.58,
                    "top_logprobs": [
                        {"token": " boleto charge", "logprob": -1.58},
                        {"token": " payment slip charge", "logprob": -1.62},
                        {"token": " payment", "logprob": -1.75},
                    ],
                },
                {
                    "token": " would be reversed.",
                    "logprob": -0.93,
                    "top_logprobs": [
                        {"token": " would be reversed.", "logprob": -0.93},
                        {"token": " would be refunded.", "logprob": -1.06},
                        {"token": " would be cancelled.", "logprob": -1.30},
                    ],
                },
            ],
        },
    }],
}


def run_offline_demo() -> None:
    """Process the mock response through review-signal functions."""
    print("=" * 60)
    print("OFFLINE DEMO — mock generation logprobs")
    print("=" * 60)
    print()

    gen_tokens = generation_tokens_from_response(MOCK_RESPONSE)
    translation = MOCK_RESPONSE["choices"][0]["message"]["content"]
    logprobs = [t["logprob"] for t in gen_tokens]

    mean_lp = mean_logprob(logprobs)
    ambiguous = flag_ambiguous_tokens(gen_tokens, margin_threshold=1.0)
    uncertain = find_uncertain_spans(gen_tokens)

    report = format_review_report(
        translation, gen_tokens, mean_lp, ambiguous, uncertain,
    )
    print(report)
    print()

    terminology = terminology_assessment(gen_tokens)
    if terminology["review_recommendation"] == "no terminology review triggered":
        print("\u2713 No terminology review triggered")
    else:
        print(f"\u26a0 {terminology['review_recommendation'].capitalize()}")


def run_live_demo(
    base_url: str,
    model: str,
    source: str = (
        "A equipe de atendimento abriu um tíquete, mas o pedido ficou parado porque "
        "o fornecedor ainda não enviou a inscrição estadual corrigida nem confirmou "
        "se a cobrança do boleto seria estornada."
    ),
    lang: str = "English",
) -> None:
    """Call a real API and display the review-signal report."""
    from mt.translate import translate as api_translate

    print("=" * 60)
    print("LIVE DEMO — calling API")
    print("=" * 60)
    print()

    data = api_translate(source, lang, base_url=base_url, model=model)
    translation = data["choices"][0]["message"]["content"]
    gen_tokens = generation_tokens_from_response(data)

    if not gen_tokens:
        print(f"Translation: {translation}")
        print("(No generation logprobs returned.)")
        return

    logprobs = [t["logprob"] for t in gen_tokens]
    mean_lp = mean_logprob(logprobs)
    ambiguous = flag_ambiguous_tokens(gen_tokens)
    uncertain = find_uncertain_spans(gen_tokens)

    report = format_review_report(
        translation, gen_tokens, mean_lp, ambiguous, uncertain,
    )
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate with logprob review signals demo")
    ap.add_argument("--live", action="store_true", help="Call a real API instead of using mock data")
    ap.add_argument("--base-url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="")
    ap.add_argument("--source", default=(
        "A equipe de atendimento abriu um tíquete, mas o pedido ficou parado porque "
        "o fornecedor ainda não enviou a inscrição estadual corrigida nem confirmou "
        "se a cobrança do boleto seria estornada."
    ))
    ap.add_argument("--lang", default="English")
    args = ap.parse_args()

    if args.live:
        run_live_demo(args.base_url, args.model, args.source, args.lang)
    else:
        run_offline_demo()


if __name__ == "__main__":
    main()
