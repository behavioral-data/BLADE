from dotenv import load_dotenv


from blade_bench.llms.config_load import load_config
from blade_bench.llms.datamodel.gen_config import (
    GenConfig,
    MistralGenConfig,
    OpenAIGenConfig,
    AnthropicGenConfig,
    GeminiGenConfig,
    HuggingFaceGenConfig,
    GroqGenConfig,
    TextGenConfig,
    TogetherGenConfig,
)
from blade_bench.llms.textgen_groq import GroqTextGenerator
from blade_bench.llms.textgen_mistral import MistralTextGenerator
from blade_bench.llms.textgen_together import TogetherTextGenerator
from blade_bench.logger import logger
from blade_bench.llms.textgen_huggingface import HuggingFaceTextGenerator
from blade_bench.llms.textgen_openai import OpenAITextGenerator
from blade_bench.llms.textgen_gemini import GeminiTextGenerator
from blade_bench.llms.textgen_anthropic import AnthropicTextGenerator
from blade_bench.llms.base import TextGenerator

from blade_bench.utils import get_conf_dir

load_dotenv()


def get_text_gen(llm_config: GenConfig, cache_dir: str = None) -> TextGenerator:
    if isinstance(llm_config, OpenAIGenConfig):
        text_gen = OpenAITextGenerator(llm_config, cache_dir=cache_dir)
    elif isinstance(llm_config, GeminiGenConfig):
        text_gen = GeminiTextGenerator(llm_config, cache_dir=cache_dir)
    elif isinstance(llm_config, AnthropicGenConfig):
        text_gen = AnthropicTextGenerator(llm_config, cache_dir=cache_dir)
    elif isinstance(llm_config, HuggingFaceGenConfig):
        text_gen = HuggingFaceTextGenerator(llm_config, cache_dir=cache_dir)
    elif isinstance(llm_config, GroqGenConfig):
        text_gen = GroqTextGenerator(llm_config, cache_dir=cache_dir)
    elif isinstance(llm_config, MistralGenConfig):
        text_gen = MistralTextGenerator(llm_config, cache_dir=cache_dir)
    elif isinstance(llm_config, TogetherGenConfig):
        text_gen = TogetherTextGenerator(llm_config, cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown LLM config type: {llm_config}")
    return text_gen


def llm(
    provider: str = None,
    model: str = None,
    textgen_config: TextGenConfig = None,
    cache_dir: str = None,
    **kwargs,
):
    config = get_llm_config(
        provider=provider, model=model, textgen_config=textgen_config, **kwargs
    )
    return get_text_gen(config, cache_dir=cache_dir)


def get_llm_config_from_conf_dict(conf_dict: dict, provider: str, **kwargs):
    conf_dict.update(kwargs)
    conf_dict["provider"] = provider
    if (
        conf_dict.get("provider") == "openai"
        or conf_dict.get("provider") == "azureopenai"
    ):
        return OpenAIGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "anthropic":
        return AnthropicGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "gemini":
        return GeminiGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "huggingface":
        return HuggingFaceGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "groq":
        return GroqGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "mistral":
        return MistralGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "together":
        return TogetherGenConfig(**conf_dict)
    else:
        raise ValueError(f"Unknown provider {conf_dict.get('provider')}")


def sanitize_provider(provider: str):
    if provider.lower() == "openai" or provider.lower() == "default":
        return "openai"
    elif provider.lower() == "azureopenai" or provider.lower() == "azureoai":
        return "azureopenai"
    elif provider.lower() == "gemini" or provider.lower() == "google":
        return "gemini"
    elif provider.lower() == "anthropic":
        return "anthropic"
    elif provider.lower() == "huggingface":
        return "huggingface"
    elif provider.lower() == "groq":
        return "groq"
    elif provider.lower() == "mistral":
        return "mistral"
    elif provider.lower() == "together":
        return "together"
    else:
        raise ValueError(
            f"Invalid provider '{provider}'.  Supported providers are 'gemini', 'openai', 'anthropic', 'azureopenai', 'huggingface'."
        )


def get_llm_config(
    provider: str = None,
    model: str = None,
    textgen_config: TextGenConfig = None,
    **kwargs,
):
    config = load_config()
    if provider is None:
        provider = (
            config["model"]["provider"] if "provider" in config["model"] else None
        )
    if provider is None:
        raise ValueError("Provider not specified in config file")
    provider = sanitize_provider(provider)
    models = (
        config["providers"][provider]["models"]
        if "providers" in config and provider in config["providers"]
        else []
    )
    if model is None:
        model_name, model_conf = models[0]["model"]
        model_conf["max_tokens"] = models[0]["max_tokens"]
        logger.info("Using default model '%s' for provider '%s'.", model_name, provider)
    else:
        model_name = model
        model_conf = None
        for model in models:
            if model["name"] == model_name:
                model_conf = model["model"]
                model_conf["max_tokens"] = model["max_tokens"]
                break
        if model_conf is None:
            raise ValueError(
                f"Model '{model_name}' not found in config file. Options are: {[m['name']for m in models]}"
            )
    model_conf["textgen_config"] = textgen_config
    config = get_llm_config_from_conf_dict(model_conf, provider, **kwargs)
    return config


if __name__ == "__main__":
    from blade_bench.llms import get_llm_config

    gen = llm(provider="azureopenai", model="gpt-4o-azure")
    print("here")
