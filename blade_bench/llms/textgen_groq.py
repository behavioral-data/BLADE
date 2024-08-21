import os
from typing import List

import backoff
from groq import Groq
import groq
from blade_bench.llms.base import TextGenerator
from blade_bench.llms.datamodel import TextGenResponse
from blade_bench.llms.datamodel.gen_config import GroqGenConfig
from blade_bench.llms.datamodel.usage import UsageData
from blade_bench.llms.utils import backoff_hdlr

from .datamodel import Message


class GroqTextGenerator(TextGenerator):

    def __init__(self, config: GroqGenConfig, cache_dir: str = None):
        super().__init__(config, cache_dir=cache_dir)
        self.config: GroqGenConfig
        if self.config.api_key_env_name is None:
            raise ValueError("Groq API key envrionment is not set")
        self.api_key = os.environ.get(self.config.api_key_env_name, None)
        if self.api_key is None:
            raise ValueError(
                f"Groq API key from the environment varible `{self.config.api_key_env_name}` is not set"
            )
        self.client = Groq(
            api_key=self.api_key,
        )

    @backoff.on_exception(
        backoff.expo,
        (groq.RateLimitError),
        on_backoff=backoff_hdlr,
    )
    def generate_core(self, messages: List[dict] | str, **kwargs) -> TextGenResponse:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.config.model,
            max_tokens=self.config.textgen_config.max_tokens,
            temperature=self.config.textgen_config.temperature,
            top_p=self.config.textgen_config.top_p,
            n=self.config.textgen_config.n,
            frequency_penalty=self.config.textgen_config.frequency_penalty,
            presence_penalty=self.config.textgen_config.presence_penalty,
            stop=self.config.textgen_config.stop_sequences or None,
            stream=False,
        )

        response = TextGenResponse(
            text=[Message(**x.message.model_dump()) for x in chat_completion.choices],
            config=self.config.textgen_config,
            usage=UsageData(**dict(chat_completion.usage)),
            response=chat_completion,
        )
        return response
