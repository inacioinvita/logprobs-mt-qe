#!/usr/bin/env python3
"""Score hypothesis tokens from a saved vLLM chat completion JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lib_prompt_logprobs import print_score_report, scores_from_response


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract prompt_logprobs hypothesis scores from a vLLM response JSON.",
    )
    parser.add_argument("response_json", type=Path, help="Path to chat completion JSON")
    parser.add_argument(
        "--marker",
        default="translation:",
        help='Substring after which hypothesis tokens begin (default: "translation:")',
    )
    parser.add_argument(
        "--hypothesis",
        default=None,
        help="Target text for token alignment (recommended for accurate QE scores)",
    )
    args = parser.parse_args()

    if not args.response_json.is_file():
        print(f"ERROR: file not found: {args.response_json}", file=sys.stderr)
        return 1

    try:
        data = json.loads(args.response_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"ERROR: invalid JSON: {exc}", file=sys.stderr)
        return 1

    hypothesis, agg = scores_from_response(
        data, marker=args.marker, hypothesis=args.hypothesis
    )
    if not hypothesis:
        print(
            f"WARNING: no hypothesis tokens found (marker={args.marker!r})",
            file=sys.stderr,
        )

    print_score_report(hypothesis, agg, label=args.response_json.name)
    return 0 if hypothesis else 1


if __name__ == "__main__":
    sys.exit(main())
