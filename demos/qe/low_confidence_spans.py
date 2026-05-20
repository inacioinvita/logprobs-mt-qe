#!/usr/bin/env python3
"""
Low-Confidence Spans — Human Review Targeting
===============================================

Rather than flagging individual tokens, grouping contiguous low-confidence
tokens into spans gives reviewers actionable segments to check.

Adjust the threshold based on your quality bar:
    -2.0  →  catches most uncertain regions (sensitive)
    -5.0  →  only flags severely surprising tokens (strict)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(_ROOT) / "qe"))

from lib_prompt_logprobs import (
    extract_hypothesis_tokens,
    find_low_confidence_spans,
    hypothesis_text,
)

DATA = Path(_ROOT) / "QElogprob.json"

MARKER = "German translation:"
HYPOTHESIS = "Der Zauberer wirkt einen mächtigen Zauberspruch."
THRESHOLD = -2.0


def main() -> None:
    data = json.loads(DATA.read_text())
    tokens = extract_hypothesis_tokens(
        data["prompt_logprobs"], marker=MARKER, hypothesis=HYPOTHESIS,
    )
    if not tokens:
        print("ERROR: no hypothesis tokens found.")
        sys.exit(1)

    full_text = hypothesis_text(tokens)
    spans = find_low_confidence_spans(tokens, threshold=THRESHOLD)

    print(f"Hypothesis: {full_text!r}")
    print(f"Threshold:  {THRESHOLD}")
    print(f"Spans found: {len(spans)}\n")

    if not spans:
        print("No low-confidence spans detected — translation looks confident.")
    else:
        for j, span in enumerate(spans):
            start, end = span["start"], span["end"]
            before = "".join(t for t, _ in tokens[max(0, start - 2):start])
            after = "".join(t for t, _ in tokens[end:end + 2])
            print(f"  Span {j + 1}: positions {start}–{end - 1}")
            print(f"    Text:       ...{before}[{span['text']}]{after}...")
            print(f"    Mean logprob: {span['mean_logprob']:.4f}")
            print(f"    Tokens:     {span['tokens']}")
            print()

    print(
        "Rather than flagging individual tokens, grouping contiguous\n"
        "low-confidence tokens into spans gives reviewers actionable\n"
        "segments to check.  Adjust the threshold based on your quality bar."
    )


if __name__ == "__main__":
    main()
