import os
from typing import List, Union
import backoff
import openai
from openai import AzureOpenAI, OpenAI


from .utils import backoff_hdlr, num_tokens_from_messages
from .datamodel import OpenAIGenConfig, TextGenResponse, Message, UsageData
from .base import TextGenerator


class OpenAITextGenerator(TextGenerator):
    def __init__(self, config: OpenAIGenConfig, cache_dir: str = None):
        super().__init__(config, cache_dir=cache_dir)
        self.config: OpenAIGenConfig
        if self.config.api_key_env_name is None:
            raise ValueError("OpenAI API key envrionment is not set")
        self.api_key = os.environ.get(self.config.api_key_env_name, None)
        if self.api_key is None:
            raise ValueError(
                f"OpenAI API key from the environment varible `{self.config.api_key_env_name}` is not set"
            )

        if config.provider == "openai":
            self.client = OpenAI(
                api_key=self.api_key,
                organization=self.config.organization,
            )
        else:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.config.api_version,
                azure_endpoint=self.config.api_base,
                organization=self.config.organization,
                azure_deployment=self.config.deployment,
            )

    @backoff.on_exception(
        backoff.expo,
        (openai.RateLimitError, openai.APITimeoutError),
        on_backoff=backoff_hdlr,
    )
    def generate_core(
        self, messages: Union[List[dict], str], **kwargs
    ) -> TextGenResponse:
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            (
                self.config.textgen_config.max_tokens
                if self.config.textgen_config.max_tokens
                else 4096 - prompt_tokens - 10
            ),
            200,
        )

        oai_response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            n=self.config.textgen_config.n,
            max_tokens=max_tokens,
            temperature=self.config.textgen_config.temperature,
            top_p=self.config.textgen_config.top_p,
            frequency_penalty=self.config.textgen_config.frequency_penalty,
            presence_penalty=self.config.textgen_config.presence_penalty,
            stop=self.config.textgen_config.stop_sequences or None,
        )
        response = TextGenResponse(
            text=[Message(**x.message.model_dump()) for x in oai_response.choices],
            config=self.config.textgen_config,
            usage=UsageData(**dict(oai_response.usage)),
            response=oai_response,
        )
        return response
