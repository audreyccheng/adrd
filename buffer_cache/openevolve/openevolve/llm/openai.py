"""
OpenAI API interface for LLMs
"""

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
import os

import openai
from openai import AzureOpenAI

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Set up API client
        # OpenAI client requires max_retries to be int, not None
        max_retries = self.retries if self.retries is not None else 0
        
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        
        # Check if we are using Azure OpenAI
        is_azure = self.api_base and "azure.com" in self.api_base.lower()
        
        if is_azure:
            # Azure OpenAI - get API key from environment
            api_key = (
                self.api_key 
                or os.getenv("AZURE_OPENAI_API_KEY") 
                or os.getenv("OPENAI_API_KEY")
            )
            
            # Parse Azure endpoint and API version from api_base
            # Example: https://east-docetl.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-12-01-preview
            azure_endpoint = self.api_base.split("/openai/")[0] if "/openai/" in self.api_base else self.api_base
            
            # Extract API version from URL if present
            api_version_match = re.search(r'api-version=([^&]+)', self.api_base)
            api_version = api_version_match.group(1) if api_version_match else "2024-12-01-preview"
            
            # Extract deployment name from URL if present, otherwise use model name
            deployment_match = re.search(r'/deployments/([^/]+)/', self.api_base)
            if deployment_match:
                # Use deployment from URL (the actual deployed model)
                self.azure_deployment = deployment_match.group(1)
            else:
                # Fall back to model name as deployment
                self.azure_deployment = self.model
            
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                timeout=self.timeout,
                max_retries=max_retries,
            )
            self.is_azure = True
            logger.info(f"Initialized AzureOpenAI client - endpoint: {azure_endpoint}, deployment: {self.azure_deployment}")
        else:
            # Standard OpenAI or compatible API
            self.is_azure = False
        
            if self.model.startswith("gpt"):
                api_key = os.getenv("OPENAI_API_KEY")
            elif self.model.startswith("claude"):
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.model.startswith("gemini"):
                api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("OPENAI_API_KEY")
            else:
                # Try generic API key as fallback
                api_key = self.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(f"No API key found for model: {self.model}")
            
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=max_retries,
            )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is a reasoning model (works for both OpenAI and Azure)
        model_lower = str(self.model).lower()
        is_reasoning_model = model_lower.startswith(REASONING_MODEL_PREFIXES)
        
        # For Azure, use deployment name instead of model name
        model_or_deployment = getattr(self, 'azure_deployment', self.model) if getattr(self, 'is_azure', False) else self.model

        if is_reasoning_model:
            # For reasoning models (OpenAI or Azure)
            params = {
                "model": model_or_deployment,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": model_or_deployment,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        if model_lower.startswith("claude"):
            # Claude models do not support top_p parameter
            params.pop("top_p", None)

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content
