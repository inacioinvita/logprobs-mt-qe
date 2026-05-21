#!/usr/bin/env python3
"""Rank the PT-BR -> EN candidate translations used in the README."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, _ROOT)

from demos.qe.sample_ptbr_en import CANDIDATE_SCORES, SOURCE


def main() -> None:
    print("Source:")
    print(SOURCE)
    print()
    print(f"{'rank':>4}  {'candidate':<9}  {'mean_lp':>8}  {'min_lp':>8}  {'ppl_proxy':>10}  judgement")
    print("-" * 72)
    for rank, (label, _hyp, mean_lp, min_lp, ppl_proxy, judgement) in enumerate(
        CANDIDATE_SCORES, start=1,
    ):
        print(
            f"{rank:4d}  {label:<9}  {mean_lp:8.2f}  {min_lp:8.2f}  "
            f"{ppl_proxy:10.2f}  {judgement}"
        )

    print()
    print(
        "Note: Candidate D can beat Candidate C because it is fluent, even though "
        "it is materially wrong. Logprobs are plausibility signals, not full "
        "adequacy metrics."
    )


if __name__ == "__main__":
    main()
