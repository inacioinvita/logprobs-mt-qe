#!/usr/bin/env python3
"""Batch-translate segments with confidence scoring.

Usage
-----
python3 -m mt.batch_translate \\
  --base-url http://localhost:8000/v1/chat/completions \\
  --model <model-id> \\
  --lang English \\
  --input segments.txt \\
  --output results.tsv \\
  --flag-for-review review.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt.lib_mt_confidence import (
    confidence_score,
    find_uncertain_spans,
    flag_ambiguous_tokens,
    generation_tokens_from_response,
)
from mt.translate import DEFAULT_BASE_URL, DEFAULT_MODEL, translate


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Batch-translate segments with confidence scoring.",
    )
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API endpoint URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier")
    ap.add_argument("--lang", required=True, help="Target language (e.g. English)")
    ap.add_argument("--input", required=True, help="Input file (one source segment per line)")
    ap.add_argument("--output", required=True, help="Output TSV file")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0)")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens (default 256)")
    ap.add_argument("--api-key", default=None, help="API key (optional)")
    ap.add_argument("--margin-threshold", type=float, default=1.0, help="Ambiguity margin threshold")
    ap.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold for flagging")
    ap.add_argument("--flag-for-review", default=None, metavar="PATH", help="Write flagged segments to file")
    args = ap.parse_args(argv)

    with open(args.input) as f:
        segments = [line.rstrip("\n") for line in f if line.strip()]

    if not segments:
        print("No segments found in input file.", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = []
    for i, source in enumerate(segments):
        print(f"[{i + 1}/{len(segments)}] {source[:60]}...", file=sys.stderr)
        data = translate(
            source,
            args.lang,
            base_url=args.base_url,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key=args.api_key,
        )
        translation = data["choices"][0]["message"]["content"]
        gen_tokens = generation_tokens_from_response(data)
        logprobs = [t["logprob"] for t in gen_tokens]
        conf = confidence_score(logprobs)
        ambiguous = flag_ambiguous_tokens(gen_tokens, margin_threshold=args.margin_threshold)
        uncertain = find_uncertain_spans(gen_tokens)
        results.append({
            "source": source,
            "translation": translation,
            "confidence": conf,
            "n_ambiguous": len(ambiguous),
            "n_uncertain_spans": len(uncertain),
        })

    with open(args.output, "w") as f:
        f.write("source\ttranslation\tconfidence\tn_ambiguous\tn_uncertain_spans\n")
        for r in results:
            cols = [
                r["source"],
                r["translation"],
                f"{r['confidence']:.4f}",
                str(r["n_ambiguous"]),
                str(r["n_uncertain_spans"]),
            ]
            f.write("\t".join(cols) + "\n")

    flagged = [r for r in results if r["confidence"] < args.confidence_threshold]
    if args.flag_for_review and flagged:
        with open(args.flag_for_review, "w") as f:
            for r in flagged:
                f.write(f"{r['source']}\t{r['translation']}\t{r['confidence']:.4f}\n")

    total = len(results)
    mean_conf = sum(r["confidence"] for r in results) / total if total else 0.0
    pct_flagged = len(flagged) / total * 100 if total else 0.0

    print(file=sys.stderr)
    print(f"Total segments:  {total}", file=sys.stderr)
    print(f"Mean model plausibility: {mean_conf:.4f}", file=sys.stderr)
    print(f"Flagged:         {len(flagged)} ({pct_flagged:.1f}%)", file=sys.stderr)
    print(f"Output:          {args.output}", file=sys.stderr)
    if args.flag_for_review and flagged:
        print(f"Review file:     {args.flag_for_review}", file=sys.stderr)


if __name__ == "__main__":
    main()
