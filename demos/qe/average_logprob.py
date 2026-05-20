#!/usr/bin/env python3
"""
Average Token Logprob — General Confidence
===========================================

The simplest QE metric: average the log-probability the model assigns to each
token in the hypothesis.  Higher (closer to 0) means the model finds the
translation more plausible overall.  Useful as a single-number segment-level
QE score that correlates with human judgements of fluency.

Interpretation guide:
    mean > -1.0   →  high confidence (model strongly agrees)
    -1 to -3      →  moderate confidence
    < -3          →  low confidence (review recommended)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / "qe"))

from lib_prompt_logprobs import extract_hypothesis_tokens, aggregate_scores

DATA = Path(_ROOT) / "QElogprob.json"

MARKER = "German translation:"
HYPOTHESIS = "Der Zauberer wirkt einen mächtigen Zauberspruch."


def main() -> None:
    data = json.loads(DATA.read_text())
    prompt_logprobs = data["prompt_logprobs"]

    tokens = extract_hypothesis_tokens(
        prompt_logprobs, marker=MARKER, hypothesis=HYPOTHESIS,
    )
    if not tokens:
        print("ERROR: no hypothesis tokens found — check marker / hypothesis.")
        sys.exit(1)

    logprobs = [lp for _, lp in tokens]
    agg = aggregate_scores(logprobs)

    print("Per-token logprobs")
    print(f"{'idx':>4}  {'token':<20}  {'logprob':>10}  {'prob':>10}")
    print("-" * 50)
    for i, (tok, lp) in enumerate(tokens):
        prob = math.exp(lp)
        print(f"{i:4d}  {tok!r:<20}  {lp:10.4f}  {prob:10.6f}")

    print()
    print(f"  mean_logprob      = {agg['mean_logprob']:.6f}")
    print(f"  perplexity_proxy  = {agg['perplexity_proxy']:.6f}")
    print(f"  n_tokens          = {agg['n_tokens']}")

    m = agg["mean_logprob"]
    if m > -1.0:
        band = "HIGH confidence"
    elif m > -3.0:
        band = "MODERATE confidence"
    else:
        band = "LOW confidence"
    print(f"\n→ mean_logprob {m:.4f} → {band}")
    print(
        "\nHigher mean → model finds the translation more plausible overall.\n"
        "Useful as a single-number segment-level QE score."
    )


if __name__ == "__main__":
    main()
