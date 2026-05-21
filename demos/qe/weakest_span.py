#!/usr/bin/env python3
"""
Minimum Token Logprob — Weakest Span Detection
================================================

The minimum logprob reveals the single most surprising token in the
hypothesis.  In MT this often points to mistranslations, rare vocabulary,
or hallucinations.

Caveat: the *first* target token after the marker typically has a very low
logprob because the decoder hasn't built context yet.  The second-worst
token is usually more informative for actual translation quality issues.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)

from demos.qe.sample_ptbr_en import CANDIDATE_A, MARKER, PROMPT_LOGPROBS
from qe.lib_prompt_logprobs import extract_hypothesis_tokens

HYPOTHESIS = CANDIDATE_A


def main() -> None:
    tokens = extract_hypothesis_tokens(
        PROMPT_LOGPROBS, marker=MARKER, hypothesis=HYPOTHESIS,
    )
    if not tokens:
        print("ERROR: no hypothesis tokens found.")
        sys.exit(1)

    ranked = sorted(enumerate(tokens), key=lambda x: x[1][1])

    print("All tokens sorted by logprob (worst -> best)")
    print(f"{'pos':>4}  {'token':<20}  {'logprob':>10}  {'prob':>10}")
    print("-" * 50)
    for pos, (tok, lp) in ranked:
        prob = math.exp(lp)
        print(f"{pos:4d}  {tok!r:<20}  {lp:10.4f}  {prob:10.6f}")

    worst_pos, (worst_tok, worst_lp) = ranked[0]
    print(f"\n★ Worst token: {worst_tok!r} at position {worst_pos}"
          f" (logprob {worst_lp:.4f})")

    if worst_pos == 0 and len(ranked) > 1:
        pos2, (tok2, lp2) = ranked[1]
        print(
            f"\n  Note: position 0 is the first target token after the marker —\n"
            f"  context hasn't built up yet, so low logprob is expected.\n"
            f"  Second worst: {tok2!r} at position {pos2}"
            f" (logprob {lp2:.4f}) - often more informative."
        )

    print(
        "\nThe minimum logprob reveals the single most surprising token.\n"
        "In MT, this often points to mistranslations, rare vocabulary,\n"
        "or hallucinations."
    )


if __name__ == "__main__":
    main()
