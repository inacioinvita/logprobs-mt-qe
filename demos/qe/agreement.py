#!/usr/bin/env python3
"""
Agreement Across Samples — Self-Consistency
=============================================

Generate N translations (temperature > 0, n=N), then score each with
prompt_logprobs.  If all N get similar logprob scores the model is
consistent.  High variance signals the model is uncertain about the
whole segment.

Formula:
    agreement = 1 - (std_dev(mean_logprobs) / abs(mean(mean_logprobs)))

This is a normalised consistency measure: 1.0 = perfect agreement,
values near 0 or negative = high variance relative to score magnitude.

This demo runs **offline** with hard-coded scores to illustrate the
concept.  For live usage, generate N hypotheses via the API with
temperature > 0, score each with prompt_logprobs, then apply this formula.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)


def agreement_score(mean_logprobs: list[float]) -> float:
    """Normalised self-consistency: 1.0 = all samples score identically."""
    if len(mean_logprobs) < 2:
        return 1.0
    mu = sum(mean_logprobs) / len(mean_logprobs)
    if abs(mu) < 1e-9:
        return 1.0
    variance = sum((x - mu) ** 2 for x in mean_logprobs) / len(mean_logprobs)
    std = math.sqrt(variance)
    return 1.0 - (std / abs(mu))


def main() -> None:
    scores_consistent = [-1.25, -1.30, -1.22]
    scores_inconsistent = [-0.80, -3.50, -1.90]

    print("=" * 60)
    print("Example A — Consistent translations")
    print("=" * 60)
    print(f"  Hypothesis scores: {scores_consistent}")
    mu_a = sum(scores_consistent) / len(scores_consistent)
    ag_a = agreement_score(scores_consistent)
    print(f"  Mean of means:     {mu_a:.4f}")
    print(f"  Agreement:         {ag_a:.4f}")
    print(f"  → {'High' if ag_a > 0.8 else 'Low'} consistency"
          " — model produces similar translations.\n")

    print("=" * 60)
    print("Example B — Inconsistent translations")
    print("=" * 60)
    print(f"  Hypothesis scores: {scores_inconsistent}")
    mu_b = sum(scores_inconsistent) / len(scores_inconsistent)
    ag_b = agreement_score(scores_inconsistent)
    print(f"  Mean of means:     {mu_b:.4f}")
    print(f"  Agreement:         {ag_b:.4f}")
    print(f"  → {'High' if ag_b > 0.8 else 'Low'} consistency"
          " — model is uncertain about this segment.\n")

    print(
        "Agreement formula:\n"
        "  agreement = 1 - (std_dev(mean_logprobs) / abs(mean(mean_logprobs)))\n\n"
        "To use live:\n"
        "  1. Generate N translations with temperature > 0\n"
        "  2. Score each with prompt_logprobs (teacher-forced)\n"
        "  3. Collect mean_logprob per hypothesis\n"
        "  4. Compute agreement_score(list_of_means)\n"
    )


if __name__ == "__main__":
    main()
