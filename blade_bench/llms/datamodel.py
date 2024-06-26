from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path


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
    max_tokens: Optional[int] = None
    model: Optional[str] = None
    api_key_env_name: Optional[str] = Field(None, exclude=True)
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


class AnthropicGenConfig(ProviderModelConfig):
    provider: Literal["anthropic"] = "anthropic"


class GeminiGenConfig(ProviderModelConfig):
    provider: Literal["gemini"] = "gemini"


class UsageData(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


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


class LLMHistory(BaseModel):
    token_usage_history: Optional[List[UsageData]] = []
    prompt_history: Optional[List[PromptHistoryEntry]] = []
