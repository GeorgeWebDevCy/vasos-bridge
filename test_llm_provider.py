import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from llm_provider import ChatClient, default_model, normalize_provider, prepare_provider


class _HTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class TestLLMProvider(unittest.TestCase):
    def test_claude_alias_maps_to_anthropic(self):
        self.assertEqual(normalize_provider("claude"), "anthropic")
        self.assertTrue(default_model("anthropic").startswith("claude-"))

    def test_local_ollama_needs_no_api_key_and_uses_chat_endpoint(self):
        client = ChatClient("ollama", "gemma3")
        response = _HTTPResponse({"message": {"content": "translated"}})

        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "http://test-host:11434/api"},
            clear=True,
        ):
            prepare_provider("ollama")
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                result = client.complete("instructions", "text", temperature=0)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "http://test-host:11434/api/chat")
        self.assertEqual(payload["model"], "gemma3")
        self.assertFalse(payload["stream"])
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(result, "translated")

    def test_local_ollama_does_not_receive_cloud_key(self):
        client = ChatClient("ollama", "gemma3")
        response = _HTTPResponse({"message": {"content": "translated"}})
        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "http://localhost:11434", "OLLAMA_API_KEY": "cloud-key"},
            clear=True,
        ):
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                client.complete("instructions", "text", temperature=0)
        request = urlopen.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))

    def test_ollama_cloud_sends_bearer_key(self):
        client = ChatClient("ollama", "gpt-oss:120b")
        response = _HTTPResponse({"message": {"content": "translated"}})
        with patch.dict(
            os.environ,
            {"OLLAMA_HOST": "https://ollama.com", "OLLAMA_API_KEY": "secret"},
            clear=True,
        ):
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                client.complete("instructions", "text", temperature=0)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://ollama.com/api/chat")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret")

    def test_ollama_lists_available_models(self):
        client = ChatClient("ollama", "gemma3")
        response = _HTTPResponse({"models": [{"model": "gemma3"}, {"name": "llama3.2"}]})
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://localhost:11434"}, clear=True):
            with patch("urllib.request.urlopen", return_value=response) as urlopen:
                models = client.list_models()
        self.assertEqual(models, ["gemma3", "llama3.2"])
        self.assertEqual(urlopen.call_args.args[0].full_url, "http://localhost:11434/api/tags")

    def test_direct_ollama_cloud_loads_required_key_from_dotenv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "OLLAMA_HOST=https://ollama.com\nOLLAMA_API_KEY=cloud-key\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(prepare_provider("ollama", env_file), "ollama")
                self.assertEqual(os.environ["OLLAMA_API_KEY"], "cloud-key")

    def test_direct_ollama_cloud_requires_key(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "https://ollama.com"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Missing OLLAMA_API_KEY"):
                prepare_provider("ollama")

    def test_anthropic_uses_system_parameter_and_text_content(self):
        messages = Mock()
        messages.create.return_value = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="Bonjour")]
        )
        anthropic_module = types.SimpleNamespace(
            Anthropic=lambda: types.SimpleNamespace(messages=messages)
        )

        with patch.dict("sys.modules", {"anthropic": anthropic_module}):
            client = ChatClient("anthropic", "claude-test")
            result = client.complete("system text", "user text")

        self.assertEqual(result, "Bonjour")
        self.assertEqual(messages.create.call_args.kwargs["system"], "system text")

    def test_anthropic_lists_accessible_models(self):
        models = Mock()
        models.list.return_value = types.SimpleNamespace(
            data=[types.SimpleNamespace(id="claude-sonnet-test")]
        )
        anthropic_module = types.SimpleNamespace(
            Anthropic=lambda: types.SimpleNamespace(models=models)
        )
        with patch.dict("sys.modules", {"anthropic": anthropic_module}):
            client = ChatClient("anthropic", "claude-test")
            self.assertEqual(client.list_models(), ["claude-sonnet-test"])

    def test_openai_keeps_chat_completion_message_shape(self):
        create = Mock()
        create.return_value = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="Hola"))]
        )
        openai_module = types.SimpleNamespace(
            OpenAI=lambda: types.SimpleNamespace(
                chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
            )
        )

        with patch.dict("sys.modules", {"openai": openai_module}):
            client = ChatClient("openai", "gpt-test")
            result = client.complete("system text", "user text")

        self.assertEqual(result, "Hola")
        messages = create.call_args.kwargs["messages"]
        self.assertEqual(messages[0], {"role": "system", "content": "system text"})

    def test_openai_lists_text_models_without_specialized_models(self):
        models = Mock()
        models.list.return_value = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(id="gpt-4.1"),
                types.SimpleNamespace(id="text-embedding-3-small"),
                types.SimpleNamespace(id="gpt-image-1"),
            ]
        )
        openai_module = types.SimpleNamespace(
            OpenAI=lambda: types.SimpleNamespace(models=models)
        )
        with patch.dict("sys.modules", {"openai": openai_module}):
            client = ChatClient("openai", "gpt-4.1")
            self.assertEqual(client.list_models(), ["gpt-4.1"])


if __name__ == "__main__":
    unittest.main()
