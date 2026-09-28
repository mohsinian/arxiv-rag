"""OpenAI-compatible chat client (works with OpenAI, OpenRouter, Groq, Ollama, ...)."""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAICompatibleChat:
    """Thin adapter: system+user prompt → completion text."""

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.2) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
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
