#!/usr/bin/env python3
"""Generation-logprob confidence metrics for machine translation.

Unlike prompt_logprobs (which score a *fixed* hypothesis), generation logprobs
score the model's **own** output — no hypothesis needed.  This module works
with the ``choices[0].logprobs.content`` structure returned by OpenAI-compatible
APIs when ``logprobs: true, top_logprobs: N`` are set.
"""

from __future__ import annotations

import math
import unicodedata
from typing import Any


# ---------------------------------------------------------------------------
# Token extraction
# ---------------------------------------------------------------------------

def generation_tokens_from_response(data: dict[str, Any]) -> list[dict]:
    """Extract generation token entries from ``choices[0].logprobs.content``.

    Each returned dict has keys: *token*, *logprob*, *prob*, *top_logprobs*
    (list of ``{"token": str, "logprob": float}``).
    """
    content = (
        data.get("choices", [{}])[0]
        .get("logprobs", {})
        .get("content", [])
    )
    out: list[dict] = []
    for entry in content:
        lp = entry.get("logprob", 0.0)
        out.append({
            "token": entry.get("token", ""),
            "logprob": float(lp),
            "prob": math.exp(float(lp)),
            "top_logprobs": [
                {"token": t["token"], "logprob": float(t["logprob"])}
                for t in entry.get("top_logprobs", [])
            ],
        })
    return out


# ---------------------------------------------------------------------------
# Confidence score
# ---------------------------------------------------------------------------

def confidence_score(logprobs: list[float]) -> float:
    """Map mean logprob to a 0–1 confidence score (sigmoid-style).

    ``score = 1 / (1 + exp(-mean_logprob - 1))``

    Landmarks: mean_lp=0 → ~0.73, mean_lp=-1 → 0.5, mean_lp=-3 → ~0.12.
    """
    if not logprobs:
        return 0.0
    mean_lp = sum(logprobs) / len(logprobs)
    return 1.0 / (1.0 + math.exp(-mean_lp - 1.0))


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
        margin = tops[0]["logprob"] - tops[1]["logprob"]
        if margin < margin_threshold:
            flagged.append({
                "index": i,
                "token": tok["token"],
                "logprob": tok["logprob"],
                "runner_up": tops[1]["token"],
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
    threshold: float = -2.0,
) -> list[dict]:
    """Contiguous runs of tokens with ``logprob < threshold``.

    Returns list of ``{start, end, text, tokens, mean_logprob}``.
    """
    spans: list[dict] = []
    current: dict | None = None
    for i, tok in enumerate(gen_tokens):
        if tok["logprob"] < threshold:
            if current is None:
                current = {"start": i, "tokens": [], "logprobs": []}
            current["tokens"].append(tok["token"])
            current["logprobs"].append(tok["logprob"])
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

def format_confidence_report(
    translation: str,
    gen_tokens: list[dict],
    confidence: float,
    ambiguous: list[dict],
    uncertain_spans: list[dict],
) -> str:
    """Format a human-readable confidence report string."""
    lines: list[str] = []

    level = "high" if confidence >= 0.6 else "medium" if confidence >= 0.4 else "low"
    lines.append(f"Translation: {translation}")
    lines.append(f"Confidence:  {confidence:.2f} ({level})")
    lines.append("")

    lines.append(f"  {'token':<20} {'logprob':>10} {'prob':>8} {'margin':>8}")
    lines.append("  " + "-" * 50)
    for tok in gen_tokens:
        tops = tok.get("top_logprobs", [])
        margin = tops[0]["logprob"] - tops[1]["logprob"] if len(tops) >= 2 else float("inf")
        margin_s = f"{margin:.3f}" if margin != float("inf") else "—"
        lines.append(
            f"  {tok['token']:<20} {tok['logprob']:>10.3f} {tok['prob']:>8.3f} {margin_s:>8}"
        )

    if ambiguous:
        lines.append("")
        for a in ambiguous:
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
                f"\u2717 Low-confidence span [{s['start']}:{s['end']}]: "
                f"\"{s['text']}\" (mean logprob {s['mean_logprob']:.3f})"
            )
    else:
        lines.append("\u2713 No low-confidence spans detected.")

    return "\n".join(lines)
