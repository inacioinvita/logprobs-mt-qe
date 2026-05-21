#!/usr/bin/env python3
"""Batch-translate segments and write raw logprob signals as TSV.

Usage
-----
python3 -m mt.batch_translate \\
  --base-url http://localhost:8000/v1/chat/completions \\
  --model <model-id> \\
  --lang English \\
  --input segments.txt \\
  --output results.tsv

The TSV columns are raw measurements only. Triage rules (which rows count as
"needs review") are left to the caller — filter the TSV with awk / pandas /
SQL using whatever cutoffs make sense for your domain. See
``docs/signals.md`` for definitions and ``mt/presets/v01_heuristic.py`` for
one illustrative mapping.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt.lib_mt_confidence import (
    AMBIGUOUS_MARGIN_CUT,
    content_token_summary,
    detect_language_drift,
    find_uncertain_spans,
    flag_ambiguous_tokens,
    generation_tokens_from_response,
    mean_logprob,
    perplexity_proxy,
    plausibility_score,
)
from mt.translate import DEFAULT_BASE_URL, DEFAULT_MODEL, translate

TSV_HEADER = (
    "source\ttranslation\t"
    "mean_logprob\tperplexity_proxy\tdisplay_score\t"
    "composite_score\tmean_prob\tmin_lexical_prob\t"
    "mean_top_entropy\tmean_margin\t"
    "weak_share\tambiguous_share\t"
    "n_content_tokens\tn_lexical_tokens\t"
    "n_ambiguous\tn_review_spans\tn_drift_tokens"
)


def _fmt(v: float, digits: int = 4) -> str:
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{v:.{digits}f}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Batch-translate segments and write raw logprob signals as TSV.",
    )
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API endpoint URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier")
    ap.add_argument("--lang", required=True, help="Target language (e.g. English)")
    ap.add_argument("--input", required=True, help="Input file (one source segment per line)")
    ap.add_argument("--output", required=True, help="Output TSV file")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--margin-threshold", type=float, default=AMBIGUOUS_MARGIN_CUT,
                    help=f"Ambiguity margin threshold (default {AMBIGUOUS_MARGIN_CUT})")
    ap.add_argument("--target-script", default="latin")
    args = ap.parse_args(argv)

    with open(args.input) as f:
        segments = [line.rstrip("\n") for line in f if line.strip()]

    if not segments:
        print("No segments found in input file.", file=sys.stderr)
        sys.exit(1)

    rows: list[list[str]] = []
    mean_lp_total = 0.0
    n_with_signal = 0

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
        mean_lp = mean_logprob(logprobs)
        summary = content_token_summary(gen_tokens)
        ambiguous = flag_ambiguous_tokens(gen_tokens, margin_threshold=args.margin_threshold)
        uncertain = find_uncertain_spans(gen_tokens)
        drift = detect_language_drift(gen_tokens, target_lang_script=args.target_script)

        if not math.isnan(mean_lp):
            mean_lp_total += mean_lp
            n_with_signal += 1

        rows.append([
            source,
            translation,
            _fmt(mean_lp),
            _fmt(perplexity_proxy(mean_lp)),
            _fmt(plausibility_score(mean_lp), 1),
            _fmt(summary["composite_score"], 3),
            _fmt(summary["mean_prob"], 3),
            _fmt(summary["min_lexical_prob"], 3),
            _fmt(summary["mean_top_entropy"], 3),
            _fmt(summary["mean_margin"], 3),
            _fmt(summary["weak_share"], 3),
            _fmt(summary["ambiguous_share"], 3),
            str(summary["n_content_tokens"]),
            str(summary["n_lexical_tokens"]),
            str(len(ambiguous)),
            str(len(uncertain)),
            str(len(drift)),
        ])

    with open(args.output, "w") as f:
        f.write(TSV_HEADER + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

    total = len(rows)
    mean_lp = mean_lp_total / n_with_signal if n_with_signal else 0.0

    print(file=sys.stderr)
    print(f"Total segments:   {total}", file=sys.stderr)
    print(f"Mean logprob:     {mean_lp:.4f}", file=sys.stderr)
    print(f"Output:           {args.output}", file=sys.stderr)
    print(
        "Triage hint:      filter the TSV by mean_logprob, weak_share, "
        "n_review_spans, or n_drift_tokens to suit your domain.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
