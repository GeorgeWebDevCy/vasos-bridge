"""Provider-neutral chat completion clients for translation tools."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional


SUPPORTED_PROVIDERS = ("openai", "anthropic", "ollama")
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-20250514",
    "ollama": "llama3.2",
}
API_KEY_ENV_VARS: Dict[str, Optional[str]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def normalize_provider(provider: str) -> str:
    normalized = (provider or "openai").strip().lower()
    if normalized == "claude":
        normalized = "anthropic"
    if normalized not in SUPPORTED_PROVIDERS:
        choices = ", ".join(SUPPORTED_PROVIDERS)
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {choices}.")
    return normalized


def default_model(provider: str) -> str:
    return DEFAULT_MODELS[normalize_provider(provider)]


def _load_dotenv_key(var_name: str, dotenv_path: Path) -> Optional[str]:
    if not dotenv_path.exists():
        return None
    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == var_name:
            return value.strip().strip('"').strip("'")
    return None


def prepare_provider(provider: str, dotenv_path: Optional[Path] = None) -> str:
    """Load provider credentials from .env and verify optional SDK dependencies."""
    normalized = normalize_provider(provider)
    env_var = API_KEY_ENV_VARS[normalized]
    if normalized == "ollama" and dotenv_path:
        for optional_var in ("OLLAMA_HOST", "OLLAMA_API_KEY"):
            dotenv_value = _load_dotenv_key(optional_var, dotenv_path)
            if dotenv_value:
                os.environ[optional_var] = dotenv_value
    if env_var and dotenv_path:
        dotenv_value = _load_dotenv_key(env_var, dotenv_path)
        if dotenv_value:
            os.environ[env_var] = dotenv_value
    if env_var and not os.getenv(env_var):
        raise RuntimeError(f"Missing {env_var} (set it in your environment or .env).")
    if normalized == "openai":
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install openai (`pip install openai`).") from exc
    if normalized == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Missing dependency: install anthropic (`pip install anthropic`).") from exc
    if normalized == "ollama":
        hostname = urllib.parse.urlparse(os.getenv("OLLAMA_HOST", "")).hostname
        if hostname == "ollama.com" and not os.getenv("OLLAMA_API_KEY"):
            raise RuntimeError(
                "Missing OLLAMA_API_KEY for direct Ollama Cloud access "
                "(set it in your environment or .env)."
            )
    return normalized


class ChatClient:
    """Small provider adapter returning only assistant text."""

    def __init__(self, provider: str, model: str):
        self.provider = normalize_provider(provider)
        self.model = model.strip() or default_model(self.provider)
        self._client = None
        if self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI()
        elif self.provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic()

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0) -> str:
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return (response.choices[0].message.content or "").strip()
        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
            )
            return "".join(block.text for block in response.content if block.type == "text").strip()
        return self._complete_ollama(system_prompt, user_prompt, temperature)

    def _complete_ollama(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        endpoint = f"{host}/chat" if host.endswith("/api") else f"{host}/api/chat"
        hostname = urllib.parse.urlparse(host).hostname
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": temperature},
                "stream": False,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key and hostname not in ("localhost", "127.0.0.1", "::1"):
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {endpoint}. Start Ollama or set OLLAMA_HOST."
            ) from exc
        try:
            return decoded["message"]["content"].strip()
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"Unexpected Ollama response: {decoded!r}") from exc
