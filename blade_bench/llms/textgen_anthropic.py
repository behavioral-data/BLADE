import os
from typing import List
import backoff
import anthropic

from blade_bench.llms.utils import backoff_hdlr

from .datamodel import AnthropicGenConfig, TextGenResponse, Message, UsageData
from .base import TextGenerator


class AnthropicTextGenerator(TextGenerator):
    def __init__(self, config: AnthropicGenConfig, cache_dir: str = None):
        super().__init__(config, cache_dir=cache_dir)
        self.config: AnthropicGenConfig
        if self.config.api_key_env_name is None:
            raise ValueError("Anthropic API key envrionment is not set")
        self.api_key = os.environ.get(self.config.api_key_env_name, None)
        if self.api_key is None:
            raise ValueError(
                "Anthropic API key from the environment variable `{}` is not set".format(
                    self.config.api_key_env_name
                )
            )
        self.client = anthropic.Anthropic(
            api_key=self.api_key,
        )
        self.max_tokens = config.max_tokens if config.max_tokens else 4096

    @backoff.on_exception(
        backoff.expo,
        (anthropic.RateLimitError, anthropic.APITimeoutError),
        on_backoff=backoff_hdlr,
    )
    def generate_core(self, messages: List[dict], **kwargs) -> TextGenResponse:

        if isinstance(messages, list) and messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        else:
            system_prompt = ""
            messages = messages

        api_response = self.client.messages.create(
            messages=messages,
            system=system_prompt,
            model=self.config.model,
            max_tokens=(
                min(self.max_tokens, self.config.textgen_config.max_tokens)
                if self.config.textgen_config.max_tokens
                else self.max_tokens
            ),
            temperature=self.config.textgen_config.temperature,
            stop_sequences=self.config.textgen_config.stop_sequences or None,
        )

        usage = {
            "prompt_tokens": api_response.usage.input_tokens,
            "completion_tokens": api_response.usage.output_tokens,
            "total_tokens": api_response.usage.input_tokens
            + api_response.usage.output_tokens,
        }
        response = TextGenResponse(
            text=[
                Message(content=x.text, role="assistant") for x in api_response.content
            ],
            config=self.config.textgen_config,
            usage=UsageData(**usage),
            response=api_response,
        )
        return response
