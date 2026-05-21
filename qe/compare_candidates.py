#!/usr/bin/env python3
"""Compare multiple translation hypotheses on the same source by mean logprob."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qe.lib_prompt_logprobs import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    format_aggregate_line,
    hypothesis_text,
    scores_from_response,
)
from qe.score_live import score_hypothesis


def load_candidates(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Return list of (label, hypothesis_text)."""
    if args.candidates:
        path = Path(args.candidates)
        if not path.is_file():
            print(f"ERROR: candidates file not found: {path}", file=sys.stderr)
            sys.exit(1)
        lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
        hyps = [ln for ln in lines if ln and not ln.startswith("#")]
        return [(f"candidate_{i + 1}", h) for i, h in enumerate(hyps)]

    if args.hypothesis:
        return [(f"candidate_{i + 1}", h) for i, h in enumerate(args.hypothesis)]

    print("ERROR: provide --candidates FILE or one or more --hypothesis flags", file=sys.stderr)
    sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rank translation candidates by mean prompt logprob on the same source.",
    )
    parser.add_argument("--source", help="Source segment (required for live API mode)")
    parser.add_argument(
        "--candidates",
        metavar="FILE",
        help="Text file with one hypothesis per line",
    )
    parser.add_argument(
        "--hypothesis",
        action="append",
        default=None,
        help="Hypothesis string (repeatable)",
    )
    parser.add_argument("--lang", default=None, required=True, help="Target language in prompt")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--marker", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions to generate",
    )
    parser.add_argument(
        "--from-json",
        metavar="DIR",
        help="Directory of saved vLLM JSON responses (one per candidate, *.json)",
    )
    args = parser.parse_args()
    if args.marker is None:
        args.marker = f"{args.lang} translation:"

    rows: list[tuple[str, dict, str]] = []

    if args.from_json:
        json_dir = Path(args.from_json)
        if not json_dir.is_dir():
            print(f"ERROR: not a directory: {json_dir}", file=sys.stderr)
            return 1
        for path in sorted(json_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            hypothesis, agg = scores_from_response(data, marker=args.marker)
            rows.append((path.stem, agg, hypothesis_text(hypothesis)))
    else:
        if not args.source:
            print("ERROR: --source is required for live API mode", file=sys.stderr)
            return 1
        for label, hyp in load_candidates(args):
            try:
                result = score_hypothesis(
                    base_url=args.base_url,
                    model=args.model,
                    source=args.source,
                    hypothesis=hyp,
                    lang=args.lang,
                    marker=args.marker,
                    api_key=args.api_key,
                    timeout=args.timeout,
                    n=args.n,
                )
            except ConnectionError as exc:
                print(f"ERROR scoring {label}: {exc}", file=sys.stderr)
                return 1
            agg = result["aggregates"]
            text = hypothesis_text(result["tokens"]) or hyp
            rows.append((label, agg, text))

    if not rows:
        print("ERROR: no candidates to compare", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r[1]["mean_logprob"], reverse=True)

    print(f"{'rank':>4}  {'label':<20}  {'mean_lp':>10}  {'min_lp':>10}  {'ppl_proxy':>10}  {'n_tok':>6}  hypothesis")
    print("-" * 112)
    for rank, (label, agg, hyp) in enumerate(rows, start=1):
        preview = hyp.replace("\n", "\\n")
        if len(preview) > 48:
            preview = preview[:45] + "..."
        mean_lp = agg["mean_logprob"]
        min_lp = agg["min_logprob"]
        ppl = agg["perplexity_proxy"]
        print(
            f"{rank:4d}  {label:<20}  {mean_lp:10.6f}  {min_lp:10.6f}  {ppl:10.6f}  {agg['n_tokens']:6d}  {preview}"
        )

    print()
    print("Top by mean_logprob:", rows[0][0], "\u2014", format_aggregate_line(rows[0][1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
