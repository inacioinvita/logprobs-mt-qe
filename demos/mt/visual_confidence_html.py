#!/usr/bin/env python3
"""Render generation-logprob confidence as a static HTML token heatmap.

Offline demo::

    python3 demos/mt/visual_confidence_html.py --output confidence.html

From a saved API response produced by ``mt/translate.py --save-json``::

    python3 demos/mt/visual_confidence_html.py --input response.json --output confidence.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)

from demos.mt.translate_with_confidence import MOCK_RESPONSE
from mt.lib_mt_confidence import (
    content_token_summary,
    generation_tokens_from_response,
    is_low_confidence_token,
    mean_logprob,
    token_entropy,
    token_margin,
    token_topk_kurtosis,
)


def _colour_for_prob(prob: float) -> str:
    """Map token probability onto a red-to-green confidence gradient."""
    prob = max(0.0, min(1.0, prob))
    hue = 120.0 * prob
    return f"hsl({hue:.1f} 78% 82%)"


def _token_title(tok: dict) -> str:
    margin = token_margin(tok)
    kurtosis = token_topk_kurtosis(tok)
    margin_s = f"{margin:.3f}" if math.isfinite(margin) else "n/a"
    kurtosis_s = f"{kurtosis:.3f}" if not math.isnan(kurtosis) else "n/a"
    return (
        f"token={tok['token']!r}\n"
        f"logprob={tok['logprob']:.3f}\n"
        f"prob={tok['prob']:.3f}\n"
        f"margin={margin_s}\n"
        f"entropy={token_entropy(tok):.3f}\n"
        f"top-k kurtosis={kurtosis_s}"
    )


def render_html(data: dict) -> str:
    choice = data["choices"][0]
    translation = choice.get("message", {}).get("content", choice.get("text", "")).strip()
    tokens = generation_tokens_from_response(data)
    summary = content_token_summary(tokens)
    mean_lp = mean_logprob([t["logprob"] for t in tokens])

    token_spans = []
    for tok in tokens:
        classes = ["token"]
        if is_low_confidence_token(tok):
            classes.append("weak")
        token_spans.append(
            '<span class="{classes}" style="background:{colour}" title="{title}">{text}</span>'.format(
                classes=" ".join(classes),
                colour=_colour_for_prob(tok["prob"]),
                title=html.escape(_token_title(tok), quote=True),
                text=html.escape(tok["token"]),
            )
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MT Token Confidence</title>
  <style>
    body {{
      color: #1f2933;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
      margin: 2rem auto;
      max-width: 980px;
      padding: 0 1rem;
    }}
    .translation {{
      border: 1px solid #d9e2ec;
      border-radius: 10px;
      font-size: 1.15rem;
      padding: 1rem;
    }}
    .token {{
      border-radius: 4px;
      box-decoration-break: clone;
      -webkit-box-decoration-break: clone;
      margin: 0 1px;
      padding: 0.12rem 0.18rem;
    }}
    .weak {{
      box-shadow: inset 0 -2px 0 #c2410c;
    }}
    .metrics {{
      display: grid;
      gap: 0.5rem 1rem;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      margin: 1.25rem 0;
    }}
    .metric {{
      background: #f7fafc;
      border: 1px solid #d9e2ec;
      border-radius: 8px;
      padding: 0.75rem;
    }}
    .legend span {{
      border-radius: 4px;
      display: inline-block;
      margin-right: 0.35rem;
      padding: 0.1rem 0.45rem;
    }}
  </style>
</head>
<body>
  <h1>MT Token Confidence</h1>
  <p>
    Token colour uses generation probability: red = lower confidence, green = higher confidence.
    Underline marks tokens that cross the repo's weak-token heuristic. Hover for logprob,
    margin, entropy, and truncated top-k kurtosis.
  </p>
  <div class="legend">
    <span style="background:{_colour_for_prob(0.2)}">low</span>
    <span style="background:{_colour_for_prob(0.6)}">mid</span>
    <span style="background:{_colour_for_prob(0.95)}">high</span>
  </div>
  <div class="metrics">
    <div class="metric"><strong>mean_logprob</strong><br>{mean_lp:.3f}</div>
    <div class="metric"><strong>geometric_mean_prob</strong><br>{summary["geometric_mean_prob"]:.3f}</div>
    <div class="metric"><strong>mean_prob_all</strong><br>{summary["mean_prob_all"]:.3f}</div>
    <div class="metric"><strong>mean_topk_kurtosis</strong><br>{summary["mean_topk_kurtosis"]:.3f}</div>
    <div class="metric"><strong>weak_share</strong><br>{summary["weak_share"]:.3f}</div>
    <div class="metric"><strong>ambiguous_share</strong><br>{summary["ambiguous_share"]:.3f}</div>
  </div>
  <h2>Translation</h2>
  <p class="translation">{"".join(token_spans) or html.escape(translation)}</p>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Render MT logprob confidence as static HTML")
    ap.add_argument("--input", default=None, help="Saved OpenAI-compatible JSON response")
    ap.add_argument("--output", default="confidence.html", help="HTML output path")
    args = ap.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = MOCK_RESPONSE

    Path(args.output).write_text(render_html(data), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
