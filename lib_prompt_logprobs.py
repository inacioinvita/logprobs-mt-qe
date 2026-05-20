#!/usr/bin/env python3
"""Shared helpers for vLLM prompt_logprobs-based MT / QE scoring."""

from __future__ import annotations

import math
import re
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = ""

_TEMPLATE_STOP = re.compile(
    r"^(\s*|\n+|<\|[^>]*\|>|<channel\|?>?|thought|model|user|never)$",
    re.IGNORECASE,
)

_LANG_LABEL = re.compile(
    r"^\s*(German|English|French|Spanish|Italian|Portuguese|Dutch|Polish|"
    r"Japanese|Chinese|Korean|Arabic|Russian)\s*$",
    re.IGNORECASE,
)


def rank1_from_position(pos: dict[str, Any] | None) -> tuple[str, float] | None:
    """Return (decoded_token, logprob) for rank-1 entry in one prompt position."""
    if not pos or not isinstance(pos, dict):
        return None
    for info in pos.values():
        if isinstance(info, dict) and info.get("rank") == 1:
            tok = info.get("decoded_token", "")
            lp = info.get("logprob")
            if lp is not None:
                return tok, float(lp)
    return None


def prompt_token_from_position(pos: dict[str, Any] | None) -> tuple[str, float] | None:
    """Fallback: rank-1 at this position (see align_hypothesis_tokens)."""
    return rank1_from_position(pos)


def align_hypothesis_tokens(
    prompt_logprobs: list[Any],
    hypothesis: str,
    marker_end: int,
) -> list[tuple[str, float]]:
    """
    Walk prompt positions after *marker_end*; at each step pick the candidate
    decoded_token that matches the longest prefix of the remaining hypothesis.
    """
    # vLLM often tokenises " translation: Der" with a leading space on the first target token.
    remaining = hypothesis if hypothesis[:1].isspace() else f" {hypothesis.lstrip()}"
    out: list[tuple[str, float]] = []

    for i in range(marker_end + 1, len(prompt_logprobs)):
        if not remaining:
            break
        pos = prompt_logprobs[i]
        if not pos or not isinstance(pos, dict):
            continue

        candidates: list[tuple[str, float]] = []
        for info in pos.values():
            if not isinstance(info, dict):
                continue
            tok = info.get("decoded_token", "")
            lp = info.get("logprob")
            if tok and lp is not None:
                candidates.append((tok, float(lp)))

        if not candidates:
            continue

        match_tok: str | None = None
        match_lp: float | None = None
        for tok, lp in sorted(candidates, key=lambda x: len(x[0]), reverse=True):
            if remaining.startswith(tok):
                match_tok, match_lp = tok, lp
                break

        if match_tok is None:
            break

        out.append((match_tok, match_lp))
        remaining = remaining[len(match_tok) :]

    return out


def iter_prompt_tokens(
    prompt_logprobs: list[Any] | None,
    *,
    use_surprising: bool = True,
) -> list[tuple[str, float]]:
    if not prompt_logprobs:
        return []
    picker = prompt_token_from_position if use_surprising else rank1_from_position
    out: list[tuple[str, float]] = []
    for pos in prompt_logprobs:
        if pos is None:
            continue
        picked = picker(pos)
        if picked is not None:
            out.append(picked)
    return out


def _find_marker_end_index(prompt_logprobs: list[Any], marker: str) -> int:
    """
    Index in ``prompt_logprobs`` of the last position that is still part of the marker.

    Returns -1 if not found.
    """
    marker_lower = marker.lower()

    # 1) Rank-1 concatenation (works when alignment is clean).
    rank1_buf = ""
    last_rank1_end = -1
    rank1_index_map: list[int] = []
    for i, pos in enumerate(prompt_logprobs):
        if pos is None:
            continue
        r1 = rank1_from_position(pos)
        if r1 is None:
            continue
        rank1_index_map.append(i)
        rank1_buf += r1[0]
        if marker_lower in rank1_buf.lower():
            last_rank1_end = i

    if last_rank1_end >= 0:
        return last_rank1_end

    # 2) Scan for first word of marker + "translation" + ":" across nearby positions (any rank).
    first_word = marker.lower().split()[0]
    last_end = -1
    for i, pos in enumerate(prompt_logprobs):
        if not pos:
            continue
        decoded = [
            info.get("decoded_token", "")
            for info in pos.values()
            if isinstance(info, dict)
        ]
        has_german = any(first_word in d.lower() for d in decoded)
        has_trans = any("translation" in d.lower() for d in decoded)
        if not (has_german or has_trans):
            continue
        for j in range(i, min(i + 4, len(prompt_logprobs))):
            pos2 = prompt_logprobs[j]
            if not pos2:
                continue
            dec2 = [
                info.get("decoded_token", "")
                for info in pos2.values()
                if isinstance(info, dict)
            ]
            if any(d.strip() == ":" for d in dec2) and (
                has_trans or any("translation" in d.lower() for d in dec2)
            ):
                last_end = j
    return last_end


def is_whitespace_only(token: str) -> bool:
    return token.strip() == ""


def _should_stop(hypothesis: list[tuple[str, float]], token: str) -> bool:
    if not hypothesis:
        return False
    prev = "".join(t[0] for t in hypothesis)
    trial = prev + token
    if any(junk in trial for junk in ("<|channel>", "<|turn>")):
        return True
    if token.strip().lower() == "thought":
        return True
    if _TEMPLATE_STOP.match(token):
        return True
    stripped = prev.rstrip()
    if stripped.endswith((".", "!", "?")) and ("\n\n" in token or trial.endswith("\n\n")):
        return True
    return False


def extract_hypothesis_tokens(
    prompt_logprobs: list[Any] | None,
    marker: str = "German translation:",
    *,
    hypothesis: str | None = None,
) -> list[tuple[str, float]]:
    """
    Tokens after *marker* (case-insensitive).

    If *hypothesis* is given, align tokens by matching decoded fragments (best for QE).
    Otherwise fall back to rank-1 walk with template stop rules.
    """
    if not prompt_logprobs:
        return []

    marker_end = _find_marker_end_index(prompt_logprobs, marker)
    if marker_end < 0:
        return []

    if hypothesis is not None:
        return align_hypothesis_tokens(prompt_logprobs, hypothesis, marker_end)

    collecting = False
    out: list[tuple[str, float]] = []

    for i, pos in enumerate(prompt_logprobs):
        if pos is None:
            continue
        if i <= marker_end:
            continue
        tok, lp = rank1_from_position(pos)
        if tok is None or lp is None:
            continue
        if not collecting:
            if is_whitespace_only(tok) or _LANG_LABEL.match(tok):
                continue
            collecting = True
        if _should_stop(out, tok):
            break
        out.append((tok, lp))
        text = "".join(t[0] for t in out)
        if any(junk in text for junk in ("<|channel>", "<|turn>")):
            return _trim_at_junk(out)

    return out


def _trim_at_junk(hypothesis: list[tuple[str, float]]) -> list[tuple[str, float]]:
    text = ""
    trimmed: list[tuple[str, float]] = []
    for tok, lp in hypothesis:
        if any(j in text + tok for j in ("<|channel>", "<|turn>")):
            break
        trimmed.append((tok, lp))
        text += tok
    return trimmed


def aggregate_scores(logprobs: list[float]) -> dict[str, float | int]:
    """Summary statistics over hypothesis token logprobs."""
    if not logprobs:
        return {
            "mean_logprob": float("nan"),
            "sum_logprob": float("nan"),
            "perplexity_proxy": float("nan"),
            "n_tokens": 0,
            "min_logprob": float("nan"),
            "max_logprob": float("nan"),
        }
    n = len(logprobs)
    mean = sum(logprobs) / n
    return {
        "mean_logprob": mean,
        "sum_logprob": sum(logprobs),
        "perplexity_proxy": math.exp(-mean),
        "n_tokens": n,
        "min_logprob": min(logprobs),
        "max_logprob": max(logprobs),
    }


def hypothesis_text(hypothesis: list[tuple[str, float]]) -> str:
    return "".join(t[0] for t in hypothesis)


def build_scoring_prompt(source: str, hypothesis: str, lang: str) -> str:
    """Build the scoring prompt with source and hypothesis for teacher-forced logprob extraction."""
    return (
        f"Translate the following English text to {lang}:\n\n"
        f"{source.strip()}\n\n"
        f"{lang} translation: {hypothesis.strip()}"
    )


def scores_from_response(
    data: dict[str, Any],
    marker: str,
    *,
    hypothesis: str | None = None,
) -> tuple[list[tuple[str, float]], dict[str, float | int]]:
    hypothesis_tokens = extract_hypothesis_tokens(
        data.get("prompt_logprobs"),
        marker=marker,
        hypothesis=hypothesis,
    )
    logprobs = [lp for _, lp in hypothesis_tokens]
    return hypothesis_tokens, aggregate_scores(logprobs)


def format_aggregate_line(agg: dict[str, float | int]) -> str:
    return (
        f"mean_logprob={agg['mean_logprob']:.6f}  "
        f"sum_logprob={agg['sum_logprob']:.6f}  "
        f"perplexity_proxy={agg['perplexity_proxy']:.6f}  "
        f"n_tokens={agg['n_tokens']}  "
        f"min_logprob={agg['min_logprob']:.6f}  "
        f"max_logprob={agg['max_logprob']:.6f}"
    )


def print_score_report(
    hypothesis: list[tuple[str, float]],
    agg: dict[str, float | int],
    *,
    label: str | None = None,
) -> None:
    if label:
        print(f"=== {label} ===")
    if hypothesis:
        print(f"hypothesis: {hypothesis_text(hypothesis)!r}")
    print(format_aggregate_line(agg))
    print()
    print(f"{'idx':>4}  {'token':<24}  {'logprob':>12}  {'prob':>12}")
    print("-" * 56)
    for i, (tok, lp) in enumerate(hypothesis):
        prob = math.exp(lp)
        display = repr(tok)[1:-1]
        print(f"{i:4d}  {display:<24}  {lp:12.6f}  {prob:12.6f}")
