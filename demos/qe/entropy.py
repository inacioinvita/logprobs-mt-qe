#!/usr/bin/env python3
"""
Entropy — Ambiguity Estimation
===============================

Shannon entropy over the top-k token probabilities at each hypothesis
position.  High entropy means the model is uncertain between several
plausible continuations; low entropy means one token dominates.

For MT, high-entropy positions indicate where the model sees multiple
valid translations — potential review points where a human translator
might choose differently.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)

from demos.qe.sample_ptbr_en import CANDIDATE_A, MARKER, PROMPT_LOGPROBS
from qe.lib_prompt_logprobs import (
    _find_marker_end_index,
    rank1_from_position,
    token_entropy,
    top_logprobs_from_position,
)

HYPOTHESIS = CANDIDATE_A


def main() -> None:
    prompt_logprobs = PROMPT_LOGPROBS

    marker_end = _find_marker_end_index(prompt_logprobs, MARKER)
    if marker_end < 0:
        print("ERROR: marker not found.")
        sys.exit(1)

    remaining = HYPOTHESIS if HYPOTHESIS[:1].isspace() else f" {HYPOTHESIS.lstrip()}"
    results: list[tuple[str, float]] = []

    print(f"{'pos':>4}  {'token':<20}  {'entropy':>8}  {'top-k tokens'}")
    print("-" * 75)

    for i in range(marker_end + 1, len(prompt_logprobs)):
        if not remaining:
            break
        pos = prompt_logprobs[i]
        if not pos or not isinstance(pos, dict):
            continue

        r1 = rank1_from_position(pos)
        tops = top_logprobs_from_position(pos)
        matched_tok = None
        for tok, lp, rank in tops:
            if remaining.startswith(tok):
                matched_tok = tok
                break
        if matched_tok is None:
            break

        ent = token_entropy(pos)
        top_str = "  ".join(f"{t}({lp:.2f})" for t, lp, _ in tops[:4])
        print(f"{i:4d}  {matched_tok!r:<20}  {ent:8.4f}  {top_str}")

        results.append((matched_tok, ent))
        remaining = remaining[len(matched_tok):]

    if results:
        mean_ent = sum(e for _, e in results) / len(results)
        max_tok, max_ent = max(results, key=lambda x: x[1])
        print(f"\n  Mean entropy  = {mean_ent:.4f} nats")
        print(f"  Max entropy   = {max_ent:.4f} nats at token {max_tok!r}")
        print(
            "\nHigh entropy = model uncertain between several plausible tokens.\n"
            "Low entropy = one token dominates.\n"
            "High-entropy positions are where the model sees multiple valid\n"
            "translations — potential review points."
        )


if __name__ == "__main__":
    main()
