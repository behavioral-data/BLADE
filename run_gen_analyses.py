import datetime
import os
import os.path as osp
import time

import click
import yaml

from blade_bench.llms.config_load import get_models, get_providers

start_time = time.time()
from blade_bench.baselines.config import (
    MultiRunConfig,
)
from blade_bench.baselines.multirun import multirun_llm

start_time = time.time()
from blade_bench.data.datamodel.transforms import (
    TransformDataReturn,
)  # ❗️ this import needs to be kept here

from blade_bench.data.dataset import list_datasets
from blade_bench.logger import logger, formatter
from blade_bench.utils import get_absolute_dir


def run_agent(
    run_dataset: str,
    num_runs: int,
    use_agent: bool,
    cache_code_results: bool,
    use_data_desc: bool,
    llm_config_path: str,
    llm_eval_config_path: str,
    output_dir: str = None,
    llm_provider: str = None,
    llm_model: str = None,
):
    llm_config = yaml.safe_load(open(llm_config_path))
    if llm_provider:
        llm_config["provider"] = llm_provider
    if llm_model:
        llm_config["model"] = llm_model

    llm_eval_config = yaml.safe_load(open(llm_eval_config_path))

    if not output_dir:
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"./outputs/multirun/{run_dataset}_{llm_config['provider']}-{llm_config['model']}_{time_str}"
        if not osp.exists(output_dir):
            os.makedirs(output_dir)
    # remove old file
    if osp.exists(osp.join(output_dir, "run.log")):
        os.remove(osp.join(output_dir, "run.log"))
    logger.add(osp.join(output_dir, f"run.log"), format=formatter.format)

    # Building the command string
    command_string = f"python {__file__} "
    command_string += f"\\\n\t--run_dataset {run_dataset} "
    command_string += f"\\\n\t-n {num_runs} "
    if use_agent:
        command_string += "\\\n\t--use_agent "
    if not cache_code_results:
        command_string += "\\\n\t--no_cache_code_reuslts "
    if not use_data_desc:
        command_string += "\\\n\t--no_use_data_desc "
    command_string += f"\\\n\t--llm_config_path {get_absolute_dir(llm_config_path)} "
    command_string += (
        f"\\\n\t--llm_eval_config_path {get_absolute_dir(llm_eval_config_path)} "
    )
    command_string += f"\\\n\t--output_dir {get_absolute_dir(output_dir)} "
    if llm_provider:
        command_string += f"\\\n\t--llm_provider {llm_provider} "
    if llm_model:
        command_string += f"\\\n\t--llm_model {llm_model} "

    logger.info(f"Running command: \n{command_string}")
    with open(osp.join(output_dir, "command.sh"), "w") as f:
        f.write("""#!/bin/bash\n""")
        f.write(command_string)

    config = MultiRunConfig(
        llm=llm_config,
        llm_eval=llm_eval_config,
        output_dir=output_dir,
        run_dataset=run_dataset,
        use_agent=use_agent,
        use_data_desc=use_data_desc,
        num_runs=num_runs,
        use_code_cache=cache_code_results,
    )
    logger.info(config.model_dump_json(indent=2))
    multirun_llm(config)


@click.command()
@click.option(
    "--run_dataset",
    type=click.Choice(list_datasets()),
    default="hurricane",
    help="Dataset to run",
    required=True,
)
@click.option(
    "-n",
    "--num_runs",
    type=int,
    default=10,
    help="Number of runs to perform",
    show_default=True,
)
@click.option(
    "--use_agent",
    is_flag=True,
    default=False,
    help="Whether to use agent or just the base LM",
    show_default=True,
)
@click.option(
    "--no_cache_code_reuslts",
    "cache_code_results",
    is_flag=True,
    default=True,
    help="[ONLY used when use_agent=True] Whether to cache code results when running code.",
)
@click.option(
    "--no_use_data_desc",
    "use_data_desc",
    is_flag=True,
    default=True,
    help="Whether to use data description in the prompts for the LM",
    show_default=True,
)
@click.option(
    "--llm_config_path",
    type=click.Path(exists=True, file_okay=True),
    default="./conf/llm.yaml",
    help="Path to the LLM config file",
    show_default=True,
)
@click.option(
    "--llm_provider",
    type=click.Choice(get_providers()),
    default=None,
    help="Provider for the LLM to override the config file at llm_config_path",
)
@click.option(
    "--llm_model",
    type=str,
    default=None,
    help=f"Model for the LLM to override the config file at llm_config_path. Options are {get_models()}",
)
@click.option(
    "--llm_eval_config_path",
    type=click.Path(exists=True, file_okay=True),
    default="./conf/llm_eval.yaml",
    help="Path to the LLM eval config file",
    show_default=True,
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=None,
    help="output directory to store saved analyses",
)
def run_agent_click(
    run_dataset: str,
    num_runs: int,
    use_agent: bool,
    cache_code_results: bool,
    use_data_desc: bool,
    llm_config_path: str,
    llm_eval_config_path: str,
    output_dir: str,
    llm_provider: str,
    llm_model: str,
):

    run_agent(
        run_dataset=run_dataset,
        num_runs=num_runs,
        use_agent=use_agent,
        cache_code_results=cache_code_results,
        use_data_desc=use_data_desc,
        llm_config_path=llm_config_path,
        llm_eval_config_path=llm_eval_config_path,
        output_dir=output_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


if __name__ == "__main__":
    run_agent_click()
