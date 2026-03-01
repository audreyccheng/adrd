"""
Gemini LLM client via OpenAI-compatible API.
"""

import os
from typing import Optional

import openai

from .config import LLMConfig


class GeminiClient:
    """Wrapper around OpenAI-compatible API for Gemini."""

    def __init__(self, config: LLMConfig):
        self.config = config
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key not found. Set {config.api_key_env} environment variable."
            )
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=config.api_base,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: The user message with current context.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            The LLM's response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
        )

        return response.choices[0].message.content
