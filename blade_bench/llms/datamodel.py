from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path


class Message(BaseModel):
    role: str
    content: str


class TextGenConfig(BaseModel):
    provider: str
    model: str
    temperature: float = 0.8
    max_tokens: Union[int, None] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    run_config: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    api_key_env_name: Optional[str] = Field(None, exclude=True)
    use_cache: bool = Field(default=True, exclude=True)
    log_file: Optional[str] = None

    def __post_init__(self):
        if self.log_file is not None:
            file = Path(self.log_file)
            file.parent.mkdir(parents=True, exist_ok=True)


class OpenAIGenConfig(TextGenConfig):
    provider: Literal["openai"] = "openai"
    n: int = 1
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    deployment: Optional[str] = Field(None, exclude=True)
    organization: Optional[str] = Field(None, exclude=True)
    api_type: Optional[str] = Field(None, exclude=True)
    api_base: Optional[str] = Field(None, exclude=True)
    api_version: Optional[str] = Field(None, exclude=True)


class AnthropicGenConfig(TextGenConfig):
    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-3-opus-20240229"
    max_tokens: int = 4096


class GeminiGenConfig(TextGenConfig):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-1.5-pro-latest"
    max_tokens: int = 4096


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
