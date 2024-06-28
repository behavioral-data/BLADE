from typing import List
from blade_bench.llms.datamodel.gen_config import HuggingFaceGenConfig
from blade_bench.llms.datamodel.local import LocalResponse
from blade_bench.llms.local.local_client import LocalLLMClient
from .base import TextGenerator
from .datamodel import Message, TextGenResponse, UsageData


class HuggingFaceTextGenerator(TextGenerator):
    def __init__(self, config: HuggingFaceGenConfig, cache_dir: str = None):
        super().__init__(config, cache_dir=cache_dir)
        self.config: HuggingFaceGenConfig
        if self.config.llm_client == "default":
            self.llm_client = LocalLLMClient(
                self.config.api_base
                if self.config.api_base is not None
                else "http://localhost:8000"
            )
        else:
            raise ValueError(f"Unknown LLM client: {self.config.llm_client}")

        server_model_name = self.llm_client.get_model_name()
        assert (
            server_model_name == self.config.model
        ), f"Model name mismatch. Config model `{self.config.model}`, server model `{server_model_name}`"

        self.max_tokens = (
            self.config.textgen_config.max_tokens
            if self.config.textgen_config.max_tokens
            else 2048
        )

    def generate_core(self, messages: List[dict], **kwargs) -> TextGenResponse:
        self.llm_client.base_url = self.config.api_base
        resp: LocalResponse = self.llm_client.generate_chat(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.config.textgen_config.temperature,
            stop_sequences=self.config.textgen_config.stop_sequences,
        )

        response = TextGenResponse(
            text=[
                Message(
                    role="assistant",
                    content=resp.response,
                )
            ],
            config=self.config.textgen_config,
            usage=resp.usage,
            response=resp,
        )
        return response
