#!/usr/bin/env python3
"""Translate with confidence: generation logprob scoring.

Generation logprobs score the model's OWN output — no hypothesis needed.
Translate and immediately know how confident the model is.

Unlike prompt_logprobs (QE demos), which teacher-force a *fixed* hypothesis
and measure how plausible the model finds it, generation logprobs come free
with every translation call.

Key concepts:
  - Confidence score: sigmoid mapping from mean logprob to 0–1.
  - Ambiguous tokens: close margin between top-1 and top-2.
  - Low-confidence spans: contiguous regions where the model struggled.

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
sys.path.insert(0, str(Path(_ROOT) / "mt"))

from lib_mt_confidence import (
    confidence_score,
    find_uncertain_spans,
    flag_ambiguous_tokens,
    format_confidence_report,
    generation_tokens_from_response,
)

MOCK_RESPONSE = {
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "Der Zauberer wirkt einen mächtigen Zauberspruch.",
        },
        "logprobs": {
            "content": [
                {
                    "token": "Der",
                    "logprob": -0.05,
                    "top_logprobs": [
                        {"token": "Der", "logprob": -0.05},
                        {"token": "Ein", "logprob": -3.1},
                        {"token": "The", "logprob": -4.8},
                        {"token": "A", "logprob": -5.9},
                        {"token": "Die", "logprob": -6.2},
                    ],
                },
                {
                    "token": " Za",
                    "logprob": -0.01,
                    "top_logprobs": [
                        {"token": " Za", "logprob": -0.01},
                        {"token": " Mag", "logprob": -5.2},
                        {"token": " Wiz", "logprob": -6.8},
                        {"token": " He", "logprob": -7.1},
                        {"token": " Zau", "logprob": -7.5},
                    ],
                },
                {
                    "token": "uber",
                    "logprob": -0.002,
                    "top_logprobs": [
                        {"token": "uber", "logprob": -0.002},
                        {"token": "uber", "logprob": -7.0},
                        {"token": "ub", "logprob": -8.5},
                        {"token": "auber", "logprob": -9.1},
                        {"token": "UBER", "logprob": -10.0},
                    ],
                },
                {
                    "token": "er",
                    "logprob": -0.003,
                    "top_logprobs": [
                        {"token": "er", "logprob": -0.003},
                        {"token": "erer", "logprob": -8.2},
                        {"token": "ers", "logprob": -9.0},
                        {"token": "erin", "logprob": -9.5},
                        {"token": "ern", "logprob": -10.1},
                    ],
                },
                {
                    "token": " wirkt",
                    "logprob": -0.42,
                    "top_logprobs": [
                        {"token": " wirkt", "logprob": -0.42},
                        {"token": " spricht", "logprob": -1.24},
                        {"token": " zaubert", "logprob": -2.90},
                        {"token": " casts", "logprob": -5.1},
                        {"token": " macht", "logprob": -5.3},
                    ],
                },
                {
                    "token": " einen",
                    "logprob": -0.08,
                    "top_logprobs": [
                        {"token": " einen", "logprob": -0.08},
                        {"token": " ein", "logprob": -2.9},
                        {"token": " a", "logprob": -5.5},
                        {"token": " den", "logprob": -5.8},
                        {"token": " seine", "logprob": -6.1},
                    ],
                },
                {
                    "token": " m\u00e4cht",
                    "logprob": -0.03,
                    "top_logprobs": [
                        {"token": " m\u00e4cht", "logprob": -0.03},
                        {"token": " kraft", "logprob": -4.1},
                        {"token": " stark", "logprob": -4.8},
                        {"token": " power", "logprob": -6.2},
                        {"token": " gew", "logprob": -6.5},
                    ],
                },
                {
                    "token": "igen",
                    "logprob": -0.001,
                    "top_logprobs": [
                        {"token": "igen", "logprob": -0.001},
                        {"token": "ige", "logprob": -7.8},
                        {"token": "ig", "logprob": -9.2},
                        {"token": "iger", "logprob": -9.5},
                        {"token": "igem", "logprob": -10.0},
                    ],
                },
                {
                    "token": " Zauber",
                    "logprob": -0.02,
                    "top_logprobs": [
                        {"token": " Zauber", "logprob": -0.02},
                        {"token": " Zau", "logprob": -4.5},
                        {"token": " Spell", "logprob": -6.8},
                        {"token": " Fluch", "logprob": -7.1},
                        {"token": " Ban", "logprob": -8.0},
                    ],
                },
                {
                    "token": "spruch",
                    "logprob": -0.002,
                    "top_logprobs": [
                        {"token": "spruch", "logprob": -0.002},
                        {"token": "sp", "logprob": -8.12},
                        {"token": "kraft", "logprob": -9.0},
                        {"token": "trank", "logprob": -9.5},
                        {"token": "stab", "logprob": -10.2},
                    ],
                },
                {
                    "token": ".",
                    "logprob": -0.01,
                    "top_logprobs": [
                        {"token": ".", "logprob": -0.01},
                        {"token": "!", "logprob": -4.9},
                        {"token": ",", "logprob": -6.3},
                        {"token": ".\n", "logprob": -7.0},
                        {"token": ".\"", "logprob": -8.1},
                    ],
                },
            ],
        },
    }],
}


def run_offline_demo() -> None:
    """Process the mock response through all confidence functions."""
    print("=" * 60)
    print("OFFLINE DEMO — mock generation logprobs")
    print("=" * 60)
    print()

    gen_tokens = generation_tokens_from_response(MOCK_RESPONSE)
    translation = MOCK_RESPONSE["choices"][0]["message"]["content"]
    logprobs = [t["logprob"] for t in gen_tokens]

    conf = confidence_score(logprobs)
    ambiguous = flag_ambiguous_tokens(gen_tokens, margin_threshold=1.0)
    uncertain = find_uncertain_spans(gen_tokens)

    report = format_confidence_report(
        translation, gen_tokens, conf, ambiguous, uncertain,
    )
    print(report)
    print()

    if conf >= 0.5:
        print("\u2713 High confidence")
    else:
        print("\u26a0 Review recommended")


def run_live_demo(
    base_url: str,
    model: str,
    source: str = "The wizard casts a powerful spell.",
    lang: str = "German",
) -> None:
    """Call a real API and display the confidence report."""
    from translate import translate as api_translate

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
    conf = confidence_score(logprobs)
    ambiguous = flag_ambiguous_tokens(gen_tokens)
    uncertain = find_uncertain_spans(gen_tokens)

    report = format_confidence_report(
        translation, gen_tokens, conf, ambiguous, uncertain,
    )
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate with confidence scoring demo")
    ap.add_argument("--live", action="store_true", help="Call a real API instead of using mock data")
    ap.add_argument("--base-url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="")
    ap.add_argument("--source", default="The wizard casts a powerful spell.")
    ap.add_argument("--lang", default="German")
    args = ap.parse_args()

    if args.live:
        run_live_demo(args.base_url, args.model, args.source, args.lang)
    else:
        run_offline_demo()


if __name__ == "__main__":
    main()
