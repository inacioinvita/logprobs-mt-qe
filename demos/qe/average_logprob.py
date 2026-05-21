#!/usr/bin/env python3
"""
Average Token Logprob — General Confidence
===========================================

The simplest QE metric: average the log-probability the model assigns to each
token in the hypothesis.  Higher (closer to 0) means the model finds the
translation more plausible overall.  Useful as a single-number segment-level
QE score that correlates with human judgements of fluency.

Interpretation guide:
    mean > -1.0   ->  high plausibility
    -1 to -3      ->  moderate plausibility
    < -3          ->  low plausibility (review recommended)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)

from demos.qe.sample_ptbr_en import CANDIDATE_A, MARKER, PROMPT_LOGPROBS
from qe.lib_prompt_logprobs import aggregate_scores, extract_hypothesis_tokens

HYPOTHESIS = CANDIDATE_A


def main() -> None:
    tokens = extract_hypothesis_tokens(
        PROMPT_LOGPROBS, marker=MARKER, hypothesis=HYPOTHESIS,
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
        band = "HIGH plausibility"
    elif m > -3.0:
        band = "MODERATE plausibility"
    else:
        band = "LOW plausibility"
    print(f"\n-> mean_logprob {m:.4f} -> {band}")
    print(
        "\nHigher mean -> model finds the translation more plausible overall.\n"
        "Useful as a single-number segment-level QE score."
    )


if __name__ == "__main__":
    main()
