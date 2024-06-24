import os.path as osp
from dotenv import load_dotenv
from typing import Union
import hydra
from hydra import compose, initialize
import yaml


from blade_bench.llms import OpenAIGenConfig, AnthropicGenConfig, GeminiGenConfig
from .utils import get_conf_dir

load_dotenv()


def get_llm_config_from_conf_dict(conf_dict: dict):
    if conf_dict.get("provider") == "openai":
        return OpenAIGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "anthropic":
        return AnthropicGenConfig(**conf_dict)
    elif conf_dict.get("provider") == "gemini":
        return GeminiGenConfig(**conf_dict)
    else:
        raise ValueError(f"Unknown provider {conf_dict.get('provider')}")


def get_llm_config(
    cfg_name: str,
    conf_path: str = None,
) -> Union[OpenAIGenConfig, AnthropicGenConfig, GeminiGenConfig]:
    if conf_path is None:
        conf_path = get_conf_dir()

    cfg_path = osp.join(conf_path, "llm", cfg_name + ".yaml")
    res = yaml.safe_load(open(cfg_path, "r"))
    return get_llm_config_from_conf_dict(res)
