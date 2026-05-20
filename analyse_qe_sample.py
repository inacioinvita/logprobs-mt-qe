#!/usr/bin/env python3
"""
Analyse QElogprob.json (good) vs live bad hypothesis on the same source.

Default: teacher-forced prompt tokens (lib_prompt_logprobs.prompt_token_from_position).
Use --strict-rank1 for naive rank==1 only (often mis-aligns on Gemma-4 vLLM output).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib_prompt_logprobs import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    aggregate_scores,
    build_scoring_prompt,
    format_aggregate_line,
    hypothesis_text,
    print_score_report,
    rank1_from_position,
    scores_from_response,
)
from score_live import request_prompt_logprobs

SOURCE = "The wizard casts a powerful spell."
GOOD = "Der Zauberer wirkt einen mächtigen Zauberspruch."
BAD = "Der Zauberer wirft einen schwachen Zauber."
MARKER = "German translation:"
JSON_PATH = Path(__file__).resolve().parent / "QElogprob.json"


def rank1_strict_extract(prompt_logprobs, marker: str) -> list[tuple[str, float]]:
    """For each position after marker, take rank == 1 → logprob (team spec)."""
    ranked: list[tuple[str, float]] = []
    for pos in prompt_logprobs or []:
        r1 = rank1_from_position(pos)
        if r1:
            ranked.append(r1)
    full = "".join(t for t, _ in ranked)
    idx = full.lower().find(marker.lower())
    if idx < 0:
        return []
    end = idx + len(marker)
    pos_c = 0
    start = len(ranked)
    for i, (tok, _) in enumerate(ranked):
        pos_c += len(tok)
        if pos_c >= end:
            start = i + 1
            break
    return ranked[start:]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strict-rank1", action="store_true")
    p.add_argument("--marker", default=MARKER)
    args = p.parse_args()

    good_data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    prompt = build_scoring_prompt(SOURCE, BAD, "German")
    bad_data = request_prompt_logprobs(
        base_url=DEFAULT_BASE_URL,
        model=DEFAULT_MODEL,
        prompt=prompt,
        api_key=None,
        timeout=120,
    )

    if args.strict_rank1:
        good_toks = rank1_strict_extract(good_data.get("prompt_logprobs"), args.marker)
        bad_toks = rank1_strict_extract(bad_data.get("prompt_logprobs"), args.marker)
        good_agg = aggregate_scores([lp for _, lp in good_toks])
        bad_agg = aggregate_scores([lp for _, lp in bad_toks])
    else:
        good_toks, good_agg = scores_from_response(
            good_data, marker=args.marker, hypothesis=GOOD
        )
        bad_toks, bad_agg = scores_from_response(
            bad_data, marker=args.marker, hypothesis=BAD
        )

    print(f"Source: {SOURCE}")
    print(f"Mode: {'rank==1' if args.strict_rank1 else 'teacher-forced'}\n")
    print_score_report(good_toks, good_agg, label=f"Good — {GOOD!r}")
    print(f"Decoded: {hypothesis_text(good_toks)!r}\n")
    print_score_report(bad_toks, bad_agg, label=f"Bad — {BAD!r}")
    print(f"Decoded: {hypothesis_text(bad_toks)!r}\n")

    delta = good_agg["mean_logprob"] - bad_agg["mean_logprob"]
    print("## Comparison")
    print(f"delta mean_logprob (good - bad): {delta:.6f}")
    print(f"Winner: {'good' if delta > 0 else 'bad'}")
    return 0 if good_toks and bad_toks else 1


if __name__ == "__main__":
    sys.exit(main())
