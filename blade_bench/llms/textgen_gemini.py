import os
from typing import List, Union
import backoff
import openai
import google.generativeai as genai


from .utils import num_tokens_from_messages
from .datamodel import GeminiGenConfig, TextGenResponse, Message, UsageData
from .base import TextGenerator


class GeminiTextGenerator(TextGenerator):
    ROLES_MAP = {"user": "USER", "assistant": "MODEL"}
    ROLES_MAP_REVERSE = {"user": "user", "model": "assistant"}

    def __init__(self, config: GeminiGenConfig, cache_dir: str = None):
        super().__init__(config, cache_dir=cache_dir)
        self.config: GeminiGenConfig
        if self.config.api_key_env_name is None:
            raise ValueError("Gemini API key envrionment is not set")
        self.api_key = os.environ.get(self.config.api_key_env_name, None)
        if self.api_key is None:
            raise ValueError(
                f"Gemini API key from the environment varible `{self.config.api_key_env_name}` is not set"
            )
        genai.configure(api_key=self.api_key)

    def convert_messages_to_gemini(self, messages: List[dict]):
        ret = []
        for message in messages:
            ret.append(
                {
                    "role": self.ROLES_MAP[message["role"]],
                    "parts": [{"text": message["content"]}],
                }
            )
        return ret

    def generate_core(self, messages: List[dict], **kwargs) -> TextGenResponse:
        prompt_tokens = num_tokens_from_messages(messages)
        max_tokens = max(
            (
                self.config.textgen_config.max_tokens
                if self.config.textgen_config.max_tokens
                else 4096 - prompt_tokens - 10
            ),
            200,
        )

        if messages[0]["role"] == "system":
            system_prompt = messages[0]["content"]
            messages = messages[1:]
        else:
            system_prompt = None
            messages = messages

        model = genai.GenerativeModel(
            self.config.model,
            system_instruction=system_prompt,
        )

        api_response = model.generate_content(
            contents=self.convert_messages_to_gemini(messages),
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=self.config.textgen_config.temperature,
                stop_sequences=self.config.textgen_config.stop_sequences or None,
            ),
        )

        usage_metadata = api_response.usage_metadata
        usage = UsageData(
            prompt_tokens=usage_metadata.prompt_token_count,
            completion_tokens=usage_metadata.candidates_token_count,
            total_tokens=usage_metadata.total_token_count,
        )

        response = TextGenResponse(
            text=[
                Message(
                    role=(
                        self.ROLES_MAP_REVERSE[x.content.role]
                        if x.content.role in self.ROLES_MAP_REVERSE
                        else "model"
                    ),
                    content=x.content.parts[0].text,
                )
                for x in api_response.candidates
            ],
            config=self.config.textgen_config,
            usage=usage,
            response=api_response,
        )
        return response
