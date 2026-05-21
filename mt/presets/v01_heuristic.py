#!/usr/bin/env python3
"""Illustrative qualitative preset over the raw signals.

This is one possible mapping from the numeric signals in
``mt.lib_mt_confidence`` onto qualitative bands and review hints. It is **not**
imported by the default CLIs. It exists as a reference for developers who want
to layer their own calibration on top of the raw measurements.

Suggested ways to use this file:

* Import directly: ``from mt.presets.v01_heuristic import band, terminology_hint``
  and call from your own pipeline.
* Copy as a starting point for a domain- or model-specific preset.
* Replace the thresholds with values learned against MQM / DA / post-edit data.

Nothing here is calibrated against a quality benchmark. Treat every threshold
as a hypothesis to test.
"""

from __future__ import annotations

import math
from typing import Iterable


def band(mean_logprob: float) -> str:
    """Coarse mean-logprob band: ``high`` / ``moderate`` / ``low`` / ``unknown``."""
    if math.isnan(mean_logprob):
        return "unknown"
    if mean_logprob > -1.0:
        return "high"
    if mean_logprob > -3.0:
        return "moderate"
    return "low"


def terminology_hint(summary: dict) -> dict:
    """Map a ``content_token_summary`` dict onto a qualitative hint.

    Returns a dict with two keys:

    * ``label``     — short qualitative tag
    * ``rationale`` — which numeric condition triggered it

    The four branches below are illustrative cuts (not learned). Tune,
    replace, or delete them for your own use case.
    """
    weak_share = summary.get("weak_share", 0.0)
    ambiguous_share = summary.get("ambiguous_share", 0.0)
    min_prob = summary.get("min_lexical_prob", 0.0)
    weak_lex = summary.get("weak_lexical_count", 0)
    ambig = summary.get("ambiguous_count", 0)
    composite = summary.get("composite_score", 0.0)
    if summary.get("n_content_tokens", 0) == 0:
        return {"label": "no content tokens", "rationale": "n_content_tokens=0"}

    if weak_share >= 0.15 or min_prob < 0.50 or (weak_lex >= 1 and ambig >= 1):
        return {
            "label": "elevated-uncertainty terms",
            "rationale": (
                f"weak_share={weak_share:.2f}\u22650.15 OR "
                f"min_lexical_prob={min_prob:.2f}<0.50 OR "
                f"(weak_lex={weak_lex} and ambig={ambig})"
            ),
        }
    if ambiguous_share >= 0.08 or ambig >= 2:
        return {
            "label": "ambiguous terms",
            "rationale": (
                f"ambiguous_share={ambiguous_share:.2f}\u22650.08 "
                f"OR ambig={ambig}\u22652"
            ),
        }
    if composite >= 0.75:
        return {
            "label": "settled terms",
            "rationale": f"composite_score={composite:.2f}\u22650.75",
        }
    return {
        "label": "mixed terms",
        "rationale": "no other branch matched",
    }


def header_lines() -> Iterable[str]:
    """Banner the caller can prepend to flag preset use."""
    yield "preset: v01_heuristic \u2014 illustrative cuts, not calibrated"
