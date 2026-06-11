#!/usr/bin/env python3
"""Shared helpers for vLLM prompt_logprobs-based MT / QE scoring."""

from __future__ import annotations

import math
import re
from typing import Any

DEFAULT_BASE_URL = "http://localhost:8000/v1/completions"
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


def _align_hypothesis_walk(
    prompt_logprobs: list[Any],
    hypothesis: str,
    marker_end: int,
) -> list[tuple[str, float, dict[str, Any] | None]]:
    """
    Walk prompt positions after *marker_end*; at each step pick the candidate
    decoded_token that matches the longest prefix of the remaining hypothesis.

    Returns (token, logprob, position_dict) triples. *position_dict* retains the
    full vLLM top-k mass at that step for entropy / kurtosis signals.
    """
    # vLLM often tokenises " translation: Der" with a leading space on the first target token.
    remaining = hypothesis if hypothesis[:1].isspace() else f" {hypothesis.lstrip()}"
    out: list[tuple[str, float, dict[str, Any] | None]] = []

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

        out.append((match_tok, match_lp, pos))
        remaining = remaining[len(match_tok) :]

    return out


def align_hypothesis_tokens(
    prompt_logprobs: list[Any],
    hypothesis: str,
    marker_end: int,
) -> list[tuple[str, float]]:
    """Aligned hypothesis tokens as (token, logprob) pairs."""
    return [(tok, lp) for tok, lp, _ in _align_hypothesis_walk(prompt_logprobs, hypothesis, marker_end)]


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
    marker: str = "translation:",
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


def geometric_mean_prob(mean_lp: float) -> float:
    """exp(mean_logprob), equivalent to geometric mean token probability."""
    if math.isnan(mean_lp):
        return float("nan")
    return math.exp(mean_lp)


def aggregate_scores(logprobs: list[float]) -> dict[str, float | int]:
    """Summary statistics over hypothesis token logprobs."""
    if not logprobs:
        return {
            "mean_logprob": float("nan"),
            "sum_logprob": float("nan"),
            "perplexity_proxy": float("nan"),
            "geometric_mean_prob": float("nan"),
            "mean_prob_all": float("nan"),
            "mean_topk_kurtosis": float("nan"),
            "n_tokens": 0,
            "min_logprob": float("nan"),
            "max_logprob": float("nan"),
        }
    n = len(logprobs)
    mean = sum(logprobs) / n
    probs = [math.exp(lp) for lp in logprobs]
    return {
        "mean_logprob": mean,
        "sum_logprob": sum(logprobs),
        "perplexity_proxy": math.exp(-mean),
        "geometric_mean_prob": geometric_mean_prob(mean),
        "mean_prob_all": sum(probs) / n,
        "mean_topk_kurtosis": float("nan"),
        "n_tokens": n,
        "min_logprob": min(logprobs),
        "max_logprob": max(logprobs),
    }


def segment_qe_score(agg: dict[str, float | int]) -> float:
    """Canonical one-number QE score for segment triage and candidate ranking.

    Returns ``mean_logprob`` over aligned hypothesis tokens. ``geometric_mean_prob``
    and ``mean_prob_all`` are monotone transforms of the same underlying signal on
    a fixed segment; use them for dashboards, not as a different ranking axis.
    Break ties with ``min_logprob`` (see ``compare_candidates.py``).
    """
    return float(agg["mean_logprob"])


def top_logprobs_from_position(pos: dict[str, Any] | None) -> list[tuple[str, float, int]]:
    """Return all (decoded_token, logprob, rank) entries sorted by rank."""
    if not pos or not isinstance(pos, dict):
        return []
    entries = []
    for info in pos.values():
        if not isinstance(info, dict):
            continue
        tok = info.get("decoded_token", "")
        lp = info.get("logprob")
        rank = info.get("rank")
        if tok and lp is not None and rank is not None:
            entries.append((tok, float(lp), int(rank)))
    return sorted(entries, key=lambda e: e[2])


def token_entropy(pos: dict[str, Any] | None) -> float:
    """Shannon entropy over top-k probabilities at one position (nats)."""
    tops = top_logprobs_from_position(pos)
    if not tops:
        return 0.0
    probs = [math.exp(lp) for _, lp, _ in tops]
    total = sum(probs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for p in probs:
        p_norm = p / total
        if p_norm > 0:
            entropy -= p_norm * math.log(p_norm)
    return entropy


def margin_top1_top2(pos: dict[str, Any] | None) -> float | None:
    """Log-probability gap between rank-1 and rank-2 (larger = more decisive)."""
    tops = top_logprobs_from_position(pos)
    if len(tops) < 2:
        return None
    return tops[0][1] - tops[1][1]


def token_topk_kurtosis(pos: dict[str, Any] | None) -> float:
    """Kurtosis over available top-k probabilities at one prompt position.

    Computed only over the top-k alternatives returned by the server (truncated
    compared with full-vocabulary kurtosis in uncertainty-visualisation papers).
    """
    tops = top_logprobs_from_position(pos)
    if len(tops) < 2:
        return float("nan")
    probs = [math.exp(lp) for _, lp, _ in tops]
    mean_p = sum(probs) / len(probs)
    variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
    if variance <= 0:
        return 0.0
    fourth = sum((p - mean_p) ** 4 for p in probs) / len(probs)
    return fourth / (variance ** 2)


def aggregate_position_scores(
    positions: list[dict[str, Any] | None],
) -> dict[str, float]:
    """Top-k distributional summaries over aligned hypothesis positions."""
    kurtoses = [
        token_topk_kurtosis(pos)
        for pos in positions
        if pos is not None
    ]
    kurtoses = [k for k in kurtoses if not math.isnan(k)]
    return {
        "mean_topk_kurtosis": sum(kurtoses) / len(kurtoses) if kurtoses else float("nan"),
    }


def find_low_confidence_spans(
    tokens: list[tuple[str, float]],
    threshold: float = -2.0,
) -> list[dict]:
    """Find contiguous runs of tokens below a logprob threshold."""
    spans = []
    current_span = None
    for i, (tok, lp) in enumerate(tokens):
        if lp < threshold:
            if current_span is None:
                current_span = {"start": i, "tokens": [], "logprobs": []}
            current_span["tokens"].append(tok)
            current_span["logprobs"].append(lp)
        else:
            if current_span is not None:
                current_span["end"] = i
                current_span["text"] = "".join(current_span["tokens"])
                current_span["mean_logprob"] = sum(current_span["logprobs"]) / len(current_span["logprobs"])
                spans.append(current_span)
                current_span = None
    if current_span is not None:
        current_span["end"] = len(tokens)
        current_span["text"] = "".join(current_span["tokens"])
        current_span["mean_logprob"] = sum(current_span["logprobs"]) / len(current_span["logprobs"])
        spans.append(current_span)
    return spans


def hypothesis_text(hypothesis: list[tuple[str, float]]) -> str:
    return "".join(t[0] for t in hypothesis)


def build_scoring_prompt(source: str, hypothesis: str, lang: str) -> str:
    """Build the scoring prompt with source and hypothesis for teacher-forced logprob extraction."""
    return (
        f"Translate the following text to {lang}:\n\n"
        f"{source.strip()}\n\n"
        f"{lang} translation: {hypothesis.strip()}"
    )


def prompt_logprobs_from_response(data: dict[str, Any]) -> list[Any] | None:
    """Return vLLM prompt_logprobs from either OpenAI-compatible response location."""
    prompt_logprobs = data.get("prompt_logprobs")
    if prompt_logprobs is not None:
        return prompt_logprobs
    return data.get("choices", [{}])[0].get("prompt_logprobs")


def scores_from_response(
    data: dict[str, Any],
    marker: str,
    *,
    hypothesis: str | None = None,
) -> tuple[list[tuple[str, float]], dict[str, float | int], list[dict[str, Any] | None]]:
    prompt_logprobs = prompt_logprobs_from_response(data)
    if hypothesis is not None and prompt_logprobs:
        marker_end = _find_marker_end_index(prompt_logprobs, marker)
        if marker_end >= 0:
            aligned = _align_hypothesis_walk(prompt_logprobs, hypothesis, marker_end)
            hypothesis_tokens = [(tok, lp) for tok, lp, _ in aligned]
            positions = [pos for _, _, pos in aligned]
        else:
            hypothesis_tokens = []
            positions = []
    else:
        hypothesis_tokens = extract_hypothesis_tokens(
            prompt_logprobs,
            marker=marker,
            hypothesis=hypothesis,
        )
        positions = []

    logprobs = [lp for _, lp in hypothesis_tokens]
    agg = aggregate_scores(logprobs)
    if positions:
        agg.update(aggregate_position_scores(positions))
    return hypothesis_tokens, agg, positions


def format_aggregate_line(agg: dict[str, float | int]) -> str:
    parts = [
        f"mean_logprob={agg['mean_logprob']:.6f}",
        f"geometric_mean_prob={agg['geometric_mean_prob']:.6f}",
        f"mean_prob_all={agg['mean_prob_all']:.6f}",
        f"sum_logprob={agg['sum_logprob']:.6f}",
        f"perplexity_proxy={agg['perplexity_proxy']:.6f}",
        f"n_tokens={agg['n_tokens']}",
        f"min_logprob={agg['min_logprob']:.6f}",
        f"max_logprob={agg['max_logprob']:.6f}",
    ]
    if "mean_topk_kurtosis" in agg and not math.isnan(float(agg["mean_topk_kurtosis"])):
        parts.insert(4, f"mean_topk_kurtosis={agg['mean_topk_kurtosis']:.6f}")
    return "  ".join(parts)


def print_score_report(
    hypothesis: list[tuple[str, float]],
    agg: dict[str, float | int],
    *,
    label: str | None = None,
    positions: list[dict[str, Any] | None] | None = None,
) -> None:
    if label:
        print(f"=== {label} ===")
    if hypothesis:
        print(f"hypothesis: {hypothesis_text(hypothesis)!r}")
    print(format_aggregate_line(agg))
    print()
    print(
        f"{'idx':>4}  {'token':<24}  {'logprob':>12}  {'prob':>12}  "
        f"{'margin':>8}  {'entropy':>9}  {'kurtosis':>9}"
    )
    print("-" * 88)
    for i, (tok, lp) in enumerate(hypothesis):
        prob = math.exp(lp)
        display = repr(tok)[1:-1]
        pos = positions[i] if positions and i < len(positions) else None
        margin = margin_top1_top2(pos)
        margin_disp = f"{margin:.3f}" if margin is not None else "\u2014"
        ent = token_entropy(pos)
        kurt = token_topk_kurtosis(pos)
        kurt_disp = f"{kurt:.3f}" if not math.isnan(kurt) else "\u2014"
        print(
            f"{i:4d}  {display:<24}  {lp:12.6f}  {prob:12.6f}  "
            f"{margin_disp:>8}  {ent:9.3f}  {kurt_disp:>9}"
        )
