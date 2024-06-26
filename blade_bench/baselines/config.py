from typing import Union
from pydantic import BaseModel

from blade_bench.llms.datamodel import (
    AnthropicGenConfig,
    GeminiGenConfig,
    OpenAIGenConfig,
)


class BenchmarkConfig(BaseModel):
    llm: Union[OpenAIGenConfig, GeminiGenConfig, AnthropicGenConfig]
    llm_eval: Union[OpenAIGenConfig, GeminiGenConfig, AnthropicGenConfig]
    output_dir: str
    run_dataset: str
    use_agent: bool = False
    use_data_desc: bool = False
    use_code_cache: bool = True


def load_config(llm_dict: dict):
    if llm_dict.get("provider") == "openai":
        return OpenAIGenConfig(**llm_dict)
    elif llm_dict.get("provider") == "gemini":
        return GeminiGenConfig(**llm_dict)
    elif llm_dict.get("provider") == "anthropic":
        return AnthropicGenConfig(**llm_dict)

    raise ValueError(f"Invalid LLM provider: {llm_dict.get('provider')}")


def load_benchmark_config(config_dict: dict):
    llm = load_config(config_dict["llm"])
    llm_eval = load_config(config_dict["llm_eval"])

    config_dict["llm"] = llm
    config_dict["llm_eval"] = llm_eval

    return BenchmarkConfig(**config_dict)
