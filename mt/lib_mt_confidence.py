#!/usr/bin/env python3
"""Generation-logprob review signals for machine translation.

Unlike prompt_logprobs (which score a *fixed* hypothesis), generation logprobs
score the model's **own** output — no hypothesis needed.  This module works
with the ``choices[0].logprobs.content`` structure returned by OpenAI-compatible
APIs when ``logprobs: true, top_logprobs: N`` are set.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

_SPECIAL_TOKEN = re.compile(r"^<[^>]+>$")
_PUNCT_OR_SPACE = re.compile(r"^[\s\W_]+$")
_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "if", "in",
    "is", "it", "its", "not", "of", "on", "or", "she", "that", "the",
    "their", "there", "they", "this", "to", "was", "were", "whether", "which",
    "who", "will", "with", "would",
}


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def generation_tokens_from_response(data: dict[str, Any]) -> list[dict]:
    """Extract generation token entries from ``choices[0].logprobs.content``.

    Each returned dict has keys: *token*, *logprob*, *prob*, *top_logprobs*
    (list of ``{"token": str, "logprob": float}``).
    """
    out: list[dict] = []
    content = (
        data.get("choices", [{}])[0]
        .get("logprobs", {})
        .get("content", [])
    )
    if not content:
        logprobs = data.get("choices", [{}])[0].get("logprobs", {})
        tokens = logprobs.get("tokens", [])
        token_logprobs = logprobs.get("token_logprobs", [])
        top_logprobs = logprobs.get("top_logprobs", [])
        for tok, lp, tops in zip(tokens, token_logprobs, top_logprobs):
            if lp is None:
                continue
            if _SPECIAL_TOKEN.match(tok.strip()):
                continue
            out_top = [
                {"token": alt_tok, "logprob": float(alt_lp)}
                for alt_tok, alt_lp in (tops or {}).items()
            ]
            out.append({
                "token": tok,
                "logprob": float(lp),
                "prob": math.exp(float(lp)),
                "top_logprobs": out_top,
            })
        return out
    for entry in content:
        lp = entry.get("logprob", 0.0)
        token = entry.get("token", "")
        if _SPECIAL_TOKEN.match(token.strip()):
            continue
        out.append({
            "token": token,
            "logprob": float(lp),
            "prob": math.exp(float(lp)),
            "top_logprobs": [
                {"token": t["token"], "logprob": float(t["logprob"])}
                for t in entry.get("top_logprobs", [])
            ],
        })
    return out


# ---------------------------------------------------------------------------
# Plausibility signals
# ---------------------------------------------------------------------------

def mean_logprob(logprobs: list[float]) -> float:
    """Length-normalised raw generation logprob."""
    if not logprobs:
        return float("nan")
    return sum(logprobs) / len(logprobs)


def plausibility_band(mean_lp: float) -> str:
    """Coarse heuristic band for raw mean logprob.

    These thresholds are demo defaults, not calibrated quality labels.
    """
    if math.isnan(mean_lp):
        return "unknown"
    if mean_lp > -1.0:
        return "high"
    if mean_lp > -3.0:
        return "moderate"
    return "low"


def token_margin(tok: dict) -> float:
    tops = tok.get("top_logprobs", [])
    if len(tops) < 2:
        return float("inf")
    return tops[0]["logprob"] - tops[1]["logprob"]


def runner_up(tok: dict) -> str | None:
    tops = tok.get("top_logprobs", [])
    if len(tops) < 2:
        return None
    return tops[1]["token"]


def is_punctuation_or_function_word(tok: dict) -> bool:
    text = tok.get("token", "").strip().lower()
    if not text:
        return True
    if _PUNCT_OR_SPACE.match(text):
        return True
    return text in _FUNCTION_WORDS


def is_ambiguous_token(tok: dict) -> bool:
    return tok["prob"] < 0.80 or token_margin(tok) < 1.00


def is_low_confidence_token(tok: dict) -> bool:
    return (
        tok["prob"] < 0.60
        or token_margin(tok) < 0.40
        or tok["logprob"] < -0.75
    )


def terminology_assessment(gen_tokens: list[dict]) -> dict[str, float | str | int]:
    """Composite content-token score so easy punctuation cannot hide weak terms."""
    content_tokens = [t for t in gen_tokens if not is_punctuation_or_function_word(t)]
    lexical_tokens = [t for t in gen_tokens if t.get("token", "").strip() and not _PUNCT_OR_SPACE.match(t["token"].strip())]
    if not content_tokens:
        return {
            "score": 0.0,
            "mean_prob": 0.0,
            "min_prob": 0.0,
            "weak_share": 0.0,
            "ambiguous_share": 0.0,
            "n_content_tokens": 0,
            "label": "high uncertainty",
            "review_recommendation": "review recommended",
        }

    probs = [t["prob"] for t in content_tokens]
    lexical_probs = [t["prob"] for t in lexical_tokens]
    weak_lexical_count = sum(is_low_confidence_token(t) for t in lexical_tokens)
    ambiguous_count = sum(is_ambiguous_token(t) for t in content_tokens)
    weak_share = sum(is_low_confidence_token(t) for t in content_tokens) / len(content_tokens)
    ambiguous_share = ambiguous_count / len(content_tokens)
    mean_prob = sum(probs) / len(probs)
    min_prob = min(lexical_probs or probs)
    score = (
        0.50 * mean_prob
        + 0.25 * min_prob
        + 0.15 * (1 - weak_share)
        + 0.10 * (1 - ambiguous_share)
    )

    if weak_share >= 0.15 or min_prob < 0.50 or (weak_lexical_count >= 1 and ambiguous_count >= 1):
        label = "medium-high uncertainty"
        review_recommendation = "review recommended"
    elif ambiguous_share >= 0.08 or ambiguous_count >= 2:
        label = "medium uncertainty"
        review_recommendation = "terminology review recommended"
    elif score >= 0.75:
        label = "low uncertainty"
        review_recommendation = "no terminology review triggered"
    else:
        label = "medium uncertainty"
        review_recommendation = "review terminology"

    return {
        "score": score,
        "mean_prob": mean_prob,
        "min_prob": min_prob,
        "weak_share": weak_share,
        "ambiguous_share": ambiguous_share,
        "n_content_tokens": len(content_tokens),
        "weak_lexical_count": weak_lexical_count,
        "ambiguous_count": ambiguous_count,
        "label": label,
        "review_recommendation": review_recommendation,
        "qe_proxy": review_recommendation,
    }


# ---------------------------------------------------------------------------
# Ambiguity / margin
# ---------------------------------------------------------------------------

def flag_ambiguous_tokens(
    gen_tokens: list[dict],
    margin_threshold: float = 1.0,
) -> list[dict]:
    """Tokens where the top-1 → top-2 log-probability margin < *margin_threshold*.

    Returns list of ``{index, token, logprob, runner_up, margin}``.
    """
    flagged: list[dict] = []
    for i, tok in enumerate(gen_tokens):
        tops = tok.get("top_logprobs", [])
        if len(tops) < 2:
            continue
        margin = token_margin(tok)
        if is_ambiguous_token(tok) or margin < margin_threshold:
            flagged.append({
                "index": i,
                "token": tok["token"],
                "logprob": tok["logprob"],
                "runner_up": runner_up(tok),
                "margin": margin,
            })
    return flagged


# ---------------------------------------------------------------------------
# Language drift
# ---------------------------------------------------------------------------

_SCRIPT_RANGES: dict[str, set[str]] = {
    "latin": {"LATIN"},
    "cyrillic": {"CYRILLIC"},
    "arabic": {"ARABIC"},
    "cjk": {"CJK", "HIRAGANA", "KATAKANA", "HANGUL"},
    "devanagari": {"DEVANAGARI"},
}


def _dominant_script(text: str) -> str | None:
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            for script, keywords in _SCRIPT_RANGES.items():
                if any(kw in name for kw in keywords):
                    return script
    return None


def detect_language_drift(
    gen_tokens: list[dict],
    target_lang_script: str = "latin",
) -> list[dict]:
    """Tokens where a top-k alternative appears to be in a different script.

    Simple heuristic: if any ``top_logprobs`` token's dominant Unicode script
    differs from *target_lang_script*, flag it.
    """
    flagged: list[dict] = []
    for i, tok in enumerate(gen_tokens):
        for alt in tok.get("top_logprobs", [])[1:]:
            alt_script = _dominant_script(alt["token"])
            if alt_script and alt_script != target_lang_script:
                flagged.append({
                    "index": i,
                    "token": tok["token"],
                    "suspect_alternative": alt["token"],
                    "suspect_logprob": alt["logprob"],
                })
                break
    return flagged


# ---------------------------------------------------------------------------
# Uncertain spans
# ---------------------------------------------------------------------------

def find_uncertain_spans(
    gen_tokens: list[dict],
    threshold: float = -0.75,
) -> list[dict]:
    """Contiguous runs of tokens with ``logprob < threshold``.

    Returns list of ``{start, end, text, tokens, mean_logprob}``.
    """
    spans: list[dict] = []
    current: dict | None = None
    weak_indexes = {
        i
        for i, tok in enumerate(gen_tokens)
        if tok.get("token", "").strip()
        and not _PUNCT_OR_SPACE.match(tok["token"].strip())
        and (is_low_confidence_token(tok) or tok["logprob"] < threshold)
    }
    review_indexes = {
        i
        for i, tok in enumerate(gen_tokens)
        if not is_punctuation_or_function_word(tok)
        and (is_low_confidence_token(tok) or is_ambiguous_token(tok) or tok["logprob"] < threshold)
    }
    expanded_indexes = set(review_indexes | weak_indexes)
    for i in review_indexes | weak_indexes:
        added = 0
        for j in range(i + 1, len(gen_tokens)):
            if _PUNCT_OR_SPACE.match(gen_tokens[j].get("token", "").strip()):
                break
            if is_punctuation_or_function_word(gen_tokens[j]):
                expanded_indexes.add(j)
                continue
            expanded_indexes.add(j)
            added += 1
            if added >= 2:
                break

    for i, tok in enumerate(gen_tokens):
        if i in expanded_indexes:
            if current is None:
                current = {"start": i, "tokens": [], "logprobs": [], "reasons": []}
            current["tokens"].append(tok["token"])
            current["logprobs"].append(tok["logprob"])
            if i in weak_indexes or (i in review_indexes and is_ambiguous_token(tok)):
                margin = token_margin(tok)
                kind = "weak token" if i in weak_indexes else "ambiguous token"
                reason = f"{kind} {tok['token']!r}: prob={tok['prob']:.3f}, margin={margin:.3f}"
                alt = runner_up(tok)
                if alt:
                    reason += f", runner-up={alt!r}"
                current["reasons"].append(reason)
        else:
            if current is not None:
                current["end"] = i
                current["text"] = "".join(current["tokens"])
                current["mean_logprob"] = (
                    sum(current["logprobs"]) / len(current["logprobs"])
                )
                del current["logprobs"]
                spans.append(current)
                current = None
    if current is not None:
        current["end"] = len(gen_tokens)
        current["text"] = "".join(current["tokens"])
        current["mean_logprob"] = (
            sum(current["logprobs"]) / len(current["logprobs"])
        )
        del current["logprobs"]
        spans.append(current)
    return spans


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_review_report(
    translation: str,
    gen_tokens: list[dict],
    mean_lp: float,
    ambiguous: list[dict],
    uncertain_spans: list[dict],
) -> str:
    """Format a human-readable review-signal report."""
    lines: list[str] = []

    terminology = terminology_assessment(gen_tokens)
    lines.append(f"Translation: {translation}")
    lines.append(f"Mean logprob: {mean_lp:.3f}")
    lines.append(f"Plausibility band: {plausibility_band(mean_lp)} (heuristic)")
    lines.append(f"Terminology review signal: {terminology['label']}")
    lines.append(f"Review recommendation: {terminology['review_recommendation']}")
    lines.append(
        "Terminology detail: "
        f"score={terminology['score']:.2f}, "
        f"min_prob={terminology['min_prob']:.2f}, "
        f"weak_share={terminology['weak_share']:.0%}, "
        f"ambiguous_share={terminology['ambiguous_share']:.0%}"
    )
    lines.append("")

    lines.append(f"  {'token':<20} {'logprob':>10} {'prob':>8} {'margin':>8}")
    lines.append("  " + "-" * 50)
    for tok in gen_tokens:
        tops = tok.get("top_logprobs", [])
        margin = token_margin(tok)
        margin_s = f"{margin:.3f}" if margin != float("inf") else "—"
        lines.append(
            f"  {tok['token']:<20} {tok['logprob']:>10.3f} {tok['prob']:>8.3f} {margin_s:>8}"
        )

    content_ambiguous = [
        a for a in ambiguous
        if not is_punctuation_or_function_word({"token": a["token"]})
    ]
    if content_ambiguous:
        lines.append("")
        for a in content_ambiguous:
            lines.append(
                f"\u26a0 Ambiguous: \"{a['token']}\" "
                f"(margin {a['margin']:.2f}, runner-up: \"{a['runner_up']}\")"
            )
    else:
        lines.append("")
        lines.append("\u2713 No ambiguous tokens detected.")

    if uncertain_spans:
        lines.append("")
        for s in uncertain_spans:
            lines.append(
                f"\u26a0 Review span [{s['start']}:{s['end']}]: "
                f"\"{s['text']}\" (mean logprob {s['mean_logprob']:.3f})"
            )
            if s.get("reasons"):
                lines.append(f"  Reason: {s['reasons'][0]}")
    else:
        lines.append("\u2713 No review spans flagged.")

    return "\n".join(lines)
