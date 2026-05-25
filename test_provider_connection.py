#!/usr/bin/env python3
"""Send one small request to validate an AI provider configuration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llm_provider import ChatClient, SUPPORTED_PROVIDERS, default_model, prepare_provider


def main() -> int:
    parser = argparse.ArgumentParser(description="Test configured translation provider access.")
    parser.add_argument("--provider", choices=SUPPORTED_PROVIDERS, required=True)
    parser.add_argument("--model", help="Model to test; defaults to the provider default.")
    parser.add_argument("--env-file", default=".env", help="Path to the environment file to load.")
    args = parser.parse_args()

    try:
        provider = prepare_provider(args.provider, Path(args.env_file))
        model = args.model or default_model(provider)
        client = ChatClient(provider, model)
        result = client.complete(
            "Return exactly the requested text, with no explanation.",
            "Return exactly: connection-ok",
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Connection failed: {exc}", file=sys.stderr)
        return 1
    print(f"Connected to {provider} using model {model}.")
    print(f"Response: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
