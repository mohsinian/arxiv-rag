"""OpenAI-compatible chat client (works with OpenAI, OpenRouter, Groq, Ollama, ...)."""

import logging

from openai import OpenAI

from arxiv_rag.rag.answer import LLMNotConfiguredError

logger = logging.getLogger(__name__)


class OpenAICompatibleChat:
    """Thin adapter: system+user prompt → completion text.

    The underlying client is created lazily on first use, so the app boots
    (health, stats, retrieval) without an LLM key configured; only /ask
    needs one, and a call without it raises `LLMNotConfiguredError` (503)
    instead of surfacing the SDK's credential error as a provider failure.
    Keyless local servers (e.g. Ollama) accept any placeholder key.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._client: OpenAI | None = None

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> OpenAI:
        if not self.configured:
            raise LLMNotConfiguredError(
                "no LLM credentials configured: set LLM_API_KEY (plus LLM_BASE_URL and "
                "LLM_MODEL for non-OpenAI providers) to enable answers"
            )
        if self._client is None:
            self._client = OpenAI(
                base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        response = self._get_client().chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"model {self._model} returned an empty completion")
        return content
