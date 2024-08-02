import os
from typing import List

import backoff
import groq
from blade_bench.llms.base import TextGenerator
from blade_bench.llms.datamodel import TextGenResponse
from mistralai.exceptions import MistralAPIException
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from blade_bench.llms.datamodel.gen_config import MistralGenConfig
from blade_bench.llms.datamodel.usage import UsageData

from .datamodel import Message


class MistralTextGenerator(TextGenerator):

    def __init__(self, config: MistralGenConfig, cache_dir: str = None):
        super().__init__(config, cache_dir=cache_dir)
        self.config: MistralGenConfig
        if self.config.api_key_env_name is None:
            raise ValueError("Mistral API key envrionment is not set")
        self.api_key = os.environ.get(self.config.api_key_env_name, None)
        if self.api_key is None:
            raise ValueError(
                f"Mistral API key from the environment varible `{self.config.api_key_env_name}` is not set"
            )
        self.client = MistralClient(api_key=self.api_key)

    # @backoff.on_exception(backoff.expo, (MistralAPIException))
    def generate_core(self, messages: List[dict] | str, **kwargs) -> TextGenResponse:
        chat_messages = []
        for message in messages:
            chat_messages.append(
                ChatMessage(role=message["role"], content=message["content"])
            )

        chat_completion = self.client.chat(
            messages=messages,
            model=self.config.model,
            max_tokens=self.config.textgen_config.max_tokens,
            temperature=self.config.textgen_config.temperature,
            top_p=self.config.textgen_config.top_p,
            # random_seed=self.config.textgen_config.run_config.get("run_num"),
        )

        response = TextGenResponse(
            text=[Message(**x.message.model_dump()) for x in chat_completion.choices],
            config=self.config.textgen_config,
            usage=UsageData(**dict(chat_completion.usage)),
            response=chat_completion,
        )
        return response
