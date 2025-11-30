# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

import os
from typing import Any

from datus.configuration.agent_config import ModelConfig
from datus.models.openai_compatible import OpenAICompatibleModel
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ZhipuModel(OpenAICompatibleModel):
    """
    OpenAI-compatible wrapper for Zhipu BigModel (GLM) chat completions API.

    Uses the OpenAI-compatible endpoint:
    https://open.bigmodel.cn/api/paas/v4/chat/completions
    """

    def __init__(self, model_config: ModelConfig):
        super().__init__(model_config)
        logger.debug(f"Using Zhipu model: {self.model_name} base Url: {self.base_url}")

    def _get_api_key(self) -> str:
        """Get Zhipu API key from config or environment."""
        api_key = self.model_config.api_key or os.environ.get("ZHIPU_API_KEY") or os.environ.get("GLM_API_KEY")
        if not api_key:
            raise ValueError(
                "Zhipu API key must be provided or set as ZHIPU_API_KEY (or GLM_API_KEY) environment variable"
            )
        return api_key

    def _get_base_url(self) -> str:
        """Get Zhipu base URL from config or environment."""
        return self.model_config.base_url or os.environ.get(
            "ZHIPU_API_BASE", "https://open.bigmodel.cn/api/paas/v4"
        )

    def generate(self, prompt: Any, enable_thinking: bool = False, **kwargs) -> str:
        """
                Generate a response from Zhipu GLM model.

                Args:
                    prompt: User prompt
                    enable_thinking: Reserved for compatibility; Zhipu currently ignores this flag
                    **kwargs: Additional OpenAI-compatible parameters
        Returns:
                    Model response content
        """
        return super().generate(prompt, enable_thinking, **kwargs)