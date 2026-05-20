#!/usr/bin/env python3
"""Score a fixed hypothesis via live vLLM chat/completions (prompt_logprobs)."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from lib_prompt_logprobs import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    build_scoring_prompt,
    print_score_report,
    scores_from_response,
)


def request_prompt_logprobs(
    *,
    base_url: str,
    model: str,
    prompt: str,
    api_key: str | None,
    timeout: int,
    n: int = 1,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 5,
        "prompt_logprobs": 5,
        "n": n,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(base_url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")[:800]
        raise ConnectionError(f"HTTP {exc.code} from {base_url}: {err_body}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Cannot reach {base_url}: {exc.reason}") from exc


def score_hypothesis(
    *,
    base_url: str,
    model: str,
    source: str,
    hypothesis: str,
    lang: str,
    marker: str,
    api_key: str | None = None,
    timeout: int = 300,
    n: int = 1,
) -> dict:
    """
    Call API and return {response, tokens, aggregates}.

    Currently uses first completion when n > 1; use --n 1 for deterministic scoring.
    """
    prompt = build_scoring_prompt(source, hypothesis, lang)
    data = request_prompt_logprobs(
        base_url=base_url,
        model=model,
        prompt=prompt,
        api_key=api_key,
        timeout=timeout,
        n=n,
    )
    tokens, agg = scores_from_response(
        data, marker=marker, hypothesis=hypothesis
    )
    return {"response": data, "tokens": tokens, "aggregates": agg}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Teacher-force a hypothesis and score prompt token logprobs.",
    )
    parser.add_argument("--source", required=True, help="Source segment")
    parser.add_argument("--hypothesis", required=True, help="Target hypothesis to score")
    parser.add_argument("--lang", default=None, help="Target language name in prompt (required)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Chat completions URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id")
    parser.add_argument("--api-key", default=None, help="Optional Bearer token")
    parser.add_argument(
        "--marker",
        default=None,
        help='Marker before hypothesis in prompt (default: "<lang> translation:")',
    )
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions to generate",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        metavar="PATH",
        help="Optional path to write raw API response",
    )
    args = parser.parse_args()
    if args.lang is None:
        parser.error("--lang is required")
    if args.marker is None:
        args.marker = f"{args.lang} translation:"

    try:
        result = score_hypothesis(
            base_url=args.base_url,
            model=args.model,
            source=args.source,
            hypothesis=args.hypothesis,
            lang=args.lang,
            marker=args.marker,
            api_key=args.api_key,
            timeout=args.timeout,
            n=args.n,
        )
    except ConnectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.save_json:
        Path(args.save_json).write_text(
            json.dumps(result["response"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    tokens = result["tokens"]
    agg = result["aggregates"]
    if not tokens:
        print(
            f"WARNING: no hypothesis tokens extracted (marker={args.marker!r})",
            file=sys.stderr,
        )

    print_score_report(tokens, agg, label="live score")
    return 0 if tokens else 1


if __name__ == "__main__":
    sys.exit(main())
