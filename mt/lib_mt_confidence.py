#!/usr/bin/env python3
"""Generation-logprob review signals for machine translation.

Generation logprobs score the model's own output (no hypothesis needed) via
the ``choices[0].logprobs.content`` structure returned by OpenAI-compatible
APIs when ``logprobs: true, top_logprobs: N`` are set.

This module ships **measurements only** — raw aggregates, per-token signals,
distributional summaries, and span detectors. It does not commit to bands,
labels, or review recommendations. A v0.1 illustrative preset that maps these
numbers onto qualitative labels lives in ``mt/presets/v01_heuristic.py`` and
is opt-in for any caller that wants it.

Thresholds used in classifiers (``is_ambiguous_token``, ``is_low_confidence_token``,
``find_uncertain_spans``) are exposed as module constants so callers can
override them, and are printed inline by ``format_signal_report`` so a
reader sees them as knobs rather than laws.
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
# Knob defaults (illustrative; tune per model / domain)
# ---------------------------------------------------------------------------

AMBIGUOUS_PROB_CUT = 0.80
AMBIGUOUS_MARGIN_CUT = 1.00
WEAK_PROB_CUT = 0.60
WEAK_MARGIN_CUT = 0.40
WEAK_LOGPROB_CUT = -0.75
SPAN_LOGPROB_THRESHOLD = -0.75

# Piecewise-linear display rescale anchors: (mean_logprob, display_percent)
PLAUSIBILITY_ANCHORS: list[tuple[float, float]] = [
    (0.0, 100.0),
    (-1.0, 75.0),
    (-3.0, 25.0),
    (-6.0, 0.0),
]


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
# Aggregate signals
# ---------------------------------------------------------------------------

def mean_logprob(logprobs: list[float]) -> float:
    """Length-normalised raw generation logprob."""
    if not logprobs:
        return float("nan")
    return sum(logprobs) / len(logprobs)


def perplexity_proxy(mean_lp: float) -> float:
    """exp(-mean_logprob). Familiar perplexity scale; not calibrated quality."""
    if math.isnan(mean_lp):
        return float("nan")
    return math.exp(-mean_lp)


def plausibility_score(
    mean_lp: float,
    anchors: list[tuple[float, float]] | None = None,
) -> float:
    """Piecewise-linear 0-100 display rescale of mean logprob.

    Default anchors (illustrative): ``0 -> 100%, -1 -> 75%, -3 -> 25%, -6 -> 0%``.
    Pass a different list of (mean_logprob, percent) anchors to override.
    """
    if math.isnan(mean_lp):
        return float("nan")
    pts = sorted(anchors or PLAUSIBILITY_ANCHORS, key=lambda p: p[0], reverse=True)
    if mean_lp >= pts[0][0]:
        return pts[0][1]
    for (x1, y1), (x0, y0) in zip(pts, pts[1:]):
        if mean_lp >= x0:
            t = (mean_lp - x0) / (x1 - x0)
            return max(0.0, min(100.0, y0 + t * (y1 - y0)))
    return pts[-1][1]


# ---------------------------------------------------------------------------
# Per-token signals
# ---------------------------------------------------------------------------

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


def token_entropy(tok: dict) -> float:
    """Shannon entropy (nats) over the available top-k alternatives.

    A higher value means the model spreads probability across more candidates
    at this position.
    """
    tops = tok.get("top_logprobs", [])
    if not tops:
        return 0.0
    probs = [math.exp(t["logprob"]) for t in tops]
    total = sum(probs)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for p in probs:
        q = p / total
        if q > 0:
            entropy -= q * math.log(q)
    return entropy


def is_punctuation_or_function_word(tok: dict) -> bool:
    text = tok.get("token", "").strip().lower()
    if not text:
        return True
    if _PUNCT_OR_SPACE.match(text):
        return True
    return text in _FUNCTION_WORDS


def is_ambiguous_token(
    tok: dict,
    prob_cut: float = AMBIGUOUS_PROB_CUT,
    margin_cut: float = AMBIGUOUS_MARGIN_CUT,
) -> bool:
    return tok["prob"] < prob_cut or token_margin(tok) < margin_cut


def is_low_confidence_token(
    tok: dict,
    prob_cut: float = WEAK_PROB_CUT,
    margin_cut: float = WEAK_MARGIN_CUT,
    logprob_cut: float = WEAK_LOGPROB_CUT,
) -> bool:
    return (
        tok["prob"] < prob_cut
        or token_margin(tok) < margin_cut
        or tok["logprob"] < logprob_cut
    )


# ---------------------------------------------------------------------------
# Content-token summary (numbers only)
# ---------------------------------------------------------------------------

def content_token_summary(gen_tokens: list[dict]) -> dict[str, float | int]:
    """Distributional summary over content tokens.

    Returns only numbers. Mapping any of these onto qualitative bands or
    review recommendations is left to the caller (see ``mt/presets/`` for one
    illustrative example).
    """
    content_tokens = [t for t in gen_tokens if not is_punctuation_or_function_word(t)]
    lexical_tokens = [
        t for t in gen_tokens
        if t.get("token", "").strip()
        and not _PUNCT_OR_SPACE.match(t["token"].strip())
    ]
    if not content_tokens:
        return {
            "composite_score": 0.0,
            "mean_prob": 0.0,
            "min_lexical_prob": 0.0,
            "mean_top_entropy": 0.0,
            "mean_margin": 0.0,
            "weak_share": 0.0,
            "ambiguous_share": 0.0,
            "n_content_tokens": 0,
            "n_lexical_tokens": 0,
            "weak_lexical_count": 0,
            "ambiguous_count": 0,
        }

    probs = [t["prob"] for t in content_tokens]
    lexical_probs = [t["prob"] for t in lexical_tokens]
    entropies = [token_entropy(t) for t in content_tokens]
    margins = [token_margin(t) for t in content_tokens if math.isfinite(token_margin(t))]
    weak_lexical_count = sum(is_low_confidence_token(t) for t in lexical_tokens)
    ambiguous_count = sum(is_ambiguous_token(t) for t in content_tokens)
    weak_share = sum(is_low_confidence_token(t) for t in content_tokens) / len(content_tokens)
    ambiguous_share = ambiguous_count / len(content_tokens)
    mean_prob = sum(probs) / len(probs)
    min_prob = min(lexical_probs or probs)
    composite = (
        0.50 * mean_prob
        + 0.25 * min_prob
        + 0.15 * (1 - weak_share)
        + 0.10 * (1 - ambiguous_share)
    )
    return {
        "composite_score": composite,
        "mean_prob": mean_prob,
        "min_lexical_prob": min_prob,
        "mean_top_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        "mean_margin": sum(margins) / len(margins) if margins else float("inf"),
        "weak_share": weak_share,
        "ambiguous_share": ambiguous_share,
        "n_content_tokens": len(content_tokens),
        "n_lexical_tokens": len(lexical_tokens),
        "weak_lexical_count": weak_lexical_count,
        "ambiguous_count": ambiguous_count,
    }


# ---------------------------------------------------------------------------
# Ambiguity / margin
# ---------------------------------------------------------------------------

def flag_ambiguous_tokens(
    gen_tokens: list[dict],
    margin_threshold: float = AMBIGUOUS_MARGIN_CUT,
) -> list[dict]:
    """Tokens where ``top1 - top2`` logprob margin < *margin_threshold*."""
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
# Language-script drift
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
    """Tokens where a top-k alternative is written in a different Unicode script.

    Cheap heuristic. Useful when the source mixes scripts (e.g. a Latin-script
    target where the model momentarily considers Cyrillic / CJK alternatives).
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
    threshold: float = SPAN_LOGPROB_THRESHOLD,
) -> list[dict]:
    """Contiguous runs of weak / ambiguous content tokens.

    A token seeds a span if it satisfies ``is_low_confidence_token``,
    ``is_ambiguous_token``, or has ``logprob < threshold``. Each seed extends
    forward by up to two trailing content tokens to capture surrounding context.
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
# Report formatting (numbers only; thresholds printed inline)
# ---------------------------------------------------------------------------

def _fmt_anchors(anchors: list[tuple[float, float]]) -> str:
    return ", ".join(f"{x:g}\u2192{y:g}%" for x, y in anchors)


def format_signal_report(
    translation: str,
    gen_tokens: list[dict],
    mean_lp: float,
    ambiguous: list[dict],
    uncertain_spans: list[dict],
    drift: list[dict] | None = None,
    anchors: list[tuple[float, float]] | None = None,
) -> str:
    """Human-readable signal block.

    Emits raw aggregates, distributional summaries, a token table, and three
    span/flag lists. Thresholds and rescale anchors are printed inline so the
    reader sees them as knobs.

    No bands, no labels, no review recommendations are produced here. To layer
    qualitative bands on top, see ``mt/presets/v01_heuristic.py`` or write
    your own preset that consumes this dict.
    """
    drift = drift or []
    anchors = anchors or PLAUSIBILITY_ANCHORS
    summary = content_token_summary(gen_tokens)
    ppl = perplexity_proxy(mean_lp)
    score = plausibility_score(mean_lp, anchors)
    margin_s = (
        f"{summary['mean_margin']:.3f}"
        if math.isfinite(summary["mean_margin"])
        else "\u2014"
    )

    lines: list[str] = []
    lines.append(f"Translation: {translation}")
    lines.append("")
    lines.append("Aggregate signals")
    lines.append(f"  mean_logprob        {mean_lp:>8.3f}")
    lines.append(f"  perplexity_proxy    {ppl:>8.3f}      (exp(-mean_logprob))")
    lines.append(
        f"  display_score       {score:>7.1f}%      "
        f"(piecewise rescale; anchors: {_fmt_anchors(anchors)})"
    )
    lines.append(f"  n_content_tokens    {summary['n_content_tokens']:>8d}")
    lines.append(f"  n_lexical_tokens    {summary['n_lexical_tokens']:>8d}")
    lines.append("")
    lines.append("Distribution signals (content tokens)")
    lines.append(
        f"  composite_score     {summary['composite_score']:>8.3f}      "
        "(0.50\u00b7mean_prob + 0.25\u00b7min_lexical_prob "
        "+ 0.15\u00b7(1\u2212weak_share) + 0.10\u00b7(1\u2212ambig_share))"
    )
    lines.append(f"  mean_prob           {summary['mean_prob']:>8.3f}")
    lines.append(f"  min_lexical_prob    {summary['min_lexical_prob']:>8.3f}")
    lines.append(f"  mean_top_entropy    {summary['mean_top_entropy']:>8.3f} nats")
    lines.append(f"  mean_margin         {margin_s:>8}      (top1\u2212top2 logprob)")
    lines.append(
        f"  weak_share          {summary['weak_share']:>8.3f}      "
        f"(prob<{WEAK_PROB_CUT}, or margin<{WEAK_MARGIN_CUT}, "
        f"or logprob<{WEAK_LOGPROB_CUT})"
    )
    lines.append(
        f"  ambiguous_share     {summary['ambiguous_share']:>8.3f}      "
        f"(prob<{AMBIGUOUS_PROB_CUT} or margin<{AMBIGUOUS_MARGIN_CUT})"
    )
    lines.append("")

    lines.append(f"  {'token':<20} {'logprob':>10} {'prob':>8} {'margin':>8} {'entropy':>9}")
    lines.append("  " + "-" * 60)
    for tok in gen_tokens:
        margin = token_margin(tok)
        margin_disp = f"{margin:.3f}" if math.isfinite(margin) else "\u2014"
        ent = token_entropy(tok)
        lines.append(
            f"  {tok['token']:<20} {tok['logprob']:>10.3f} {tok['prob']:>8.3f} "
            f"{margin_disp:>8} {ent:>9.3f}"
        )

    content_ambiguous = [
        a for a in ambiguous
        if not is_punctuation_or_function_word({"token": a["token"]})
    ]
    lines.append("")
    if content_ambiguous:
        lines.append(f"Ambiguous tokens (n={len(content_ambiguous)})")
        for a in content_ambiguous:
            lines.append(
                f"  [{a['index']}] \"{a['token']}\" "
                f"margin={a['margin']:.2f}  runner-up=\"{a['runner_up']}\""
            )
    else:
        lines.append("Ambiguous tokens (n=0)")

    lines.append("")
    if uncertain_spans:
        lines.append(f"Review spans (n={len(uncertain_spans)}, "
                     f"seed logprob threshold {SPAN_LOGPROB_THRESHOLD})")
        for s in uncertain_spans:
            lines.append(
                f"  [{s['start']}:{s['end']}] \"{s['text']}\"  "
                f"mean_logprob={s['mean_logprob']:.3f}"
            )
            if s.get("reasons"):
                lines.append(f"    reason: {s['reasons'][0]}")
    else:
        lines.append(f"Review spans (n=0, seed logprob threshold {SPAN_LOGPROB_THRESHOLD})")

    lines.append("")
    if drift:
        lines.append(f"Language-script drift signals (n={len(drift)})")
        for d in drift:
            lines.append(
                f"  [{d['index']}] \"{d['token']}\"  "
                f"alt=\"{d['suspect_alternative']}\"  logprob={d['suspect_logprob']:.3f}"
            )
    else:
        lines.append("Language-script drift signals (n=0)")

    return "\n".join(lines)
