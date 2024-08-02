import os
import os.path as osp
from blade_bench.logger import logger
from blade_bench.utils import get_conf_dir
import yaml


def get_providers():
    config = load_config()
    return config["providers"].keys()


def get_models():
    config = load_config()
    return [
        f"{model['name']}"
        for provider in config["providers"].keys()
        for model in config["providers"][provider]["models"]
    ]


def load_config():
    try:
        config_path = os.environ.get("LLM_CONFIG_PATH", None)
        if config_path is None or os.path.exists(config_path) is False:
            config_path = os.path.join(get_conf_dir(), "llm_config.yml")
            logger.info(
                f"Info: LLM_CONFIG_PATH environment variable is not set to a valid config file. Using default config file at '{config_path}'."
            )
        if config_path is not None:
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded config from '{config_path}'.")
                    return config
            except FileNotFoundError as file_not_found:
                logger.info(
                    "Error: Config file not found at '{}'. Please check the LLMX_CONFIG_PATH environment variable. {}".format(
                        config_path,
                        str(file_not_found),
                    )
                )
            except IOError as io_error:
                logger.info(
                    "Error: Could not read the config file at '{}'. {}".format(
                        config_path, str(io_error)
                    )
                )
            except yaml.YAMLError as yaml_error:
                logger.info(
                    "Error: Malformed YAML in config file at '{}'. {}".format(
                        config_path, str(yaml_error)
                    )
                )
        else:
            logger.info(
                "Info:LLM_CONFIG_PATH environment variable is not set. Please set it to the path of your config file to setup your default model."
            )
    except Exception as error:
        logger.info("Error: An unexpected error occurred: %s", str(error))
    return None


if __name__ == "__main__":
    print(get_providers())
    print(get_models())
    print(load_config())
