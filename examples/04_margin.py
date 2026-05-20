#!/usr/bin/env python3
"""
Margin (top-1 − top-2) — Decoder Decisiveness
===============================================

The margin is the logprob gap between the model's first and second choice
at each position.  A large margin means the decoder strongly prefers one
token; a small margin means a close race between alternatives.

In MT, a small margin at a content word suggests the model sees a
near-synonym or alternative phrasing — worth reviewing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib_prompt_logprobs import (
    _find_marker_end_index,
    margin_top1_top2,
    top_logprobs_from_position,
)

DATA = Path(__file__).resolve().parent.parent / "QElogprob.json"

MARKER = "German translation:"
HYPOTHESIS = "Der Zauberer wirkt einen mächtigen Zauberspruch."


def main() -> None:
    data = json.loads(DATA.read_text())
    prompt_logprobs = data["prompt_logprobs"]

    marker_end = _find_marker_end_index(prompt_logprobs, MARKER)
    if marker_end < 0:
        print("ERROR: marker not found.")
        sys.exit(1)

    remaining = HYPOTHESIS if HYPOTHESIS[:1].isspace() else f" {HYPOTHESIS.lstrip()}"

    print(f"{'pos':>4}  {'token':<16}  {'rank-1':<16}  {'rank-2':<16}  {'margin':>8}")
    print("-" * 70)

    results: list[tuple[str, float | None]] = []

    for i in range(marker_end + 1, len(prompt_logprobs)):
        if not remaining:
            break
        pos = prompt_logprobs[i]
        if not pos or not isinstance(pos, dict):
            continue

        tops = top_logprobs_from_position(pos)
        matched_tok = None
        for tok, lp, rank in tops:
            if remaining.startswith(tok):
                matched_tok = tok
                break
        if matched_tok is None:
            break

        m = margin_top1_top2(pos)
        r1 = tops[0] if len(tops) > 0 else ("?", 0.0, 0)
        r2 = tops[1] if len(tops) > 1 else ("?", 0.0, 0)
        m_str = f"{m:.4f}" if m is not None else "N/A"
        print(f"{i:4d}  {matched_tok!r:<16}  {r1[0]!r:<16}  {r2[0]!r:<16}  {m_str:>8}")

        results.append((matched_tok, m))
        remaining = remaining[len(matched_tok):]

    # Identify tightest races
    with_margin = [(tok, m) for tok, m in results if m is not None]
    if with_margin:
        tightest = sorted(with_margin, key=lambda x: x[1])
        print("\nTightest races (smallest margin):")
        for tok, m in tightest[:3]:
            print(f"  {tok!r:>20}  margin = {m:.4f}")

    print(
        "\nLarge margin = decoder strongly prefers one token.\n"
        "Small margin = close race between alternatives.\n"
        "Small margins at content words suggest near-synonyms or\n"
        "alternative phrasings worth reviewing."
    )


if __name__ == "__main__":
    main()
