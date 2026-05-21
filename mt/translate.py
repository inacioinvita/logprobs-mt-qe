#!/usr/bin/env python3
"""Translate a segment and show generation-logprob review signals.

Usage
-----
python3 -m mt.translate \\
  --base-url http://localhost:8000/v1/chat/completions \\
  --model <model-id> \\
  --lang English \\
  --source "A equipe de atendimento abriu um tíquete, mas o pedido ficou parado..."
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mt.lib_mt_confidence import (
    find_uncertain_spans,
    flag_ambiguous_tokens,
    format_review_report,
    generation_tokens_from_response,
    mean_logprob,
    terminology_assessment,
)

DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"
DEFAULT_MODEL = ""


def _post_json(url: str, payload: dict, api_key: str | None = None) -> dict:
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def translate(
    source: str,
    lang: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 256,
    api_key: str | None = None,
) -> dict:
    """Call the API and return the raw response dict."""
    prompt = (
        f"Translate the following text to {lang}. "
        "Output translation only, with no notes, alternatives, quotation marks, or explanation.\n\n"
        f"{source.strip()}\n\n"
        f"{lang}:"
    )
    if "/chat/completions" in base_url:
        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": 5,
        }
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": 5,
            "top_logprobs": 5,
            "stop": ["\n"],
        }
    return _post_json(base_url, payload, api_key)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Translate a segment and show generation-logprob review signals.",
    )
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API endpoint URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Model identifier")
    ap.add_argument("--lang", required=True, help="Target language (e.g. English)")
    ap.add_argument("--source", required=True, help="Source text to translate")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0)")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens (default 256)")
    ap.add_argument("--api-key", default=None, help="API key (optional)")
    ap.add_argument("--margin-threshold", type=float, default=1.0, help="Ambiguity margin threshold (default 1.0)")
    ap.add_argument("--save-json", default=None, metavar="PATH", help="Save raw API response to JSON file")
    args = ap.parse_args(argv)

    data = translate(
        args.source,
        args.lang,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_key=args.api_key,
    )

    if args.save_json:
        with open(args.save_json, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    choice = data["choices"][0]
    translation = choice.get("message", {}).get("content", choice.get("text", "")).strip()
    gen_tokens = generation_tokens_from_response(data)

    if not gen_tokens:
        print(f"Translation: {translation}")
        print("(No generation logprobs returned — server may not support logprobs.)")
        sys.exit(0)

    logprobs = [t["logprob"] for t in gen_tokens]
    mean_lp = mean_logprob(logprobs)
    ambiguous = flag_ambiguous_tokens(gen_tokens, margin_threshold=args.margin_threshold)
    uncertain = find_uncertain_spans(gen_tokens)

    report = format_review_report(translation, gen_tokens, mean_lp, ambiguous, uncertain)
    print(report)

    print()
    terminology = terminology_assessment(gen_tokens)
    if terminology["review_recommendation"] == "no terminology review triggered":
        print("\u2713 No terminology review triggered")
    else:
        print(f"\u26a0 {terminology['review_recommendation'].capitalize()}")


if __name__ == "__main__":
    main()
