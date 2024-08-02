from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator
from pathlib import Path

from blade_bench.llms.datamodel.usage import UsageData
from blade_bench.llms.local.local_client import LocalLLMClient


class Message(BaseModel):
    role: str
    content: str


class TextGenConfig(BaseModel):
    model: Optional[str] = None
    n: int = 1
    temperature: float = 0.8
    max_tokens: Union[int, None] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    run_config: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ProviderModelConfig(BaseModel):
    provider: str
    model: str
    api_key_env_name: str
    max_tokens: Optional[int] = None
    use_cache: bool = Field(default=True, exclude=True)
    textgen_config: Optional[TextGenConfig] = None
    log_file: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.textgen_config is None:
            self.textgen_config = TextGenConfig()
        self.textgen_config.model = self.model
        if self.textgen_config.max_tokens is None:
            self.textgen_config.max_tokens = self.max_tokens
        if self.log_file is not None:
            file = Path(self.log_file)
            file.parent.mkdir(parents=True, exist_ok=True)


class OpenAIGenConfig(ProviderModelConfig):
    provider: Literal["openai", "azureopenai"] = "openai"
    deployment: Optional[str] = Field(None, exclude=True)
    organization: Optional[str] = Field(None, exclude=True)
    api_base: Optional[str] = Field(None, exclude=True)
    api_version: Optional[str] = Field(None, exclude=True)

    @model_validator(mode="before")
    def validate_deployment(cls, data):
        deployment = data.get("deployment")
        model = data.get("model")
        if data.get("provider") == "azureopenai":
            if deployment and model is None:
                data["model"] = deployment
            elif model and deployment is None:
                data["deployment"] = model

        return data


class AnthropicGenConfig(ProviderModelConfig):
    provider: Literal["anthropic"] = "anthropic"


class GeminiGenConfig(ProviderModelConfig):
    provider: Literal["gemini"] = "gemini"


class GroqGenConfig(ProviderModelConfig):
    provider: Literal["groq"] = "groq"


class MistralGenConfig(ProviderModelConfig):
    provider: Literal["mistral"] = "mistral"


class TogetherGenConfig(ProviderModelConfig):
    provider: Literal["together"] = "together"


class HuggingFaceGenConfig(ProviderModelConfig):
    llm_client: Literal["default"] = "default"
    provider: Literal["huggingface"] = "huggingface"
    api_base: Optional[str] = Field(None, exclude=True)
    api_key_env_name: Optional[str] = Field(None, exclude=True)


class TextGenResponse(BaseModel):
    text: List[Message]
    config: Any
    api_elapsed_time: Optional[float] = None
    cache_elapsed_time: Optional[float] = None
    from_cache: Optional[bool] = False
    response: Optional[Any] = Field(None, exclude=True)
    usage: Optional[UsageData] = None


class PromptHistoryEntry(BaseModel):
    messages: List[dict]
    response: TextGenResponse
    tags: List[str] = []


class LLMHistory(BaseModel):
    token_usage_history: Optional[List[UsageData]] = []
    prompt_history: Optional[List[PromptHistoryEntry]] = Field(
        default_factory=list, exclude=True
    )

    @property
    def prompts_by_tags(self) -> Dict[str, List[PromptHistoryEntry]]:
        prompts_by_tags = defaultdict(list)
        for entry in self.prompt_history:
            for tag in entry.tags:
                prompts_by_tags[tag].append(entry)
        return prompts_by_tags

    @property
    def total_calls(self):
        return len(self.token_usage_history)

    @property
    def total_tokens_used(self):
        return sum([x.total_tokens for x in self.token_usage_history])

    @property
    def total_prompt_tokens_used(self):
        return sum([x.prompt_tokens for x in self.token_usage_history])

    @property
    def total_completion_tokens_used(self):
        return sum([x.completion_tokens for x in self.token_usage_history])


GenConfig = Union[
    OpenAIGenConfig,
    GeminiGenConfig,
    AnthropicGenConfig,
    HuggingFaceGenConfig,
    GroqGenConfig,
    MistralGenConfig,
    TogetherGenConfig,
]


if __name__ == "__main__":
    conf = OpenAIGenConfig(
        provider="azureopenai",
        api_key_env_name="OPENAI_API",
        deployment="text-davinci-003",
    )
    print(conf)
