import ast
import datetime
import json
import os
import os.path as osp
from typing import List
import click
import yaml
from blade_bench.logger import logger, formatter
from blade_bench.baselines.config import EvalConfig
from blade_bench.eval.datamodel.run import MultiRunResults
from blade_bench.eval.evaluator import run_eval_on_analyses
from blade_bench.data.datamodel.transforms import (
    TransformDataReturn,
)  # ❗️ this import needs to be kept here for the eval code to work
from blade_bench.utils import get_absolute_dir


def load_list_int(value: str) -> List[int]:
    v = ast.literal_eval(value)
    v = [int(i) for i in v]
    assert isinstance(v, list), "Value must be a list"
    assert all(isinstance(i, int) for i in v), "All elements must be integers"
    return v


def run_eval(
    multirun_load_path: str,
    llm_eval_config_path: str,
    cache_code_results: bool,
    output_dir: str,
    diversity_ks: str,
    diversity_n_samples: int,
):
    diversity_ks = load_list_int(diversity_ks)
    llm_eval_config = yaml.safe_load(open(llm_eval_config_path))
    with open(multirun_load_path, "r") as f:
        multirun_results = MultiRunResults(**json.load(f))
    if not output_dir:
        current_time = datetime.datetime.now()
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"./outputs/eval/{multirun_results.dataset_name}_nruns={multirun_results.n}_{time_str}"
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # remove old file
    if osp.exists(osp.join(output_dir, "run_eval.log")):
        os.remove(osp.join(output_dir, "run_eval.log"))
    logger.add(osp.join(output_dir, f"run_eval.log"), format=formatter.format)

    command_string = f"python {__file__} "
    command_string += (
        f"\\\n\t--multirun_load_path {get_absolute_dir(multirun_load_path)} "
    )
    command_string += (
        f"\\\n\t--llm_eval_config_path {get_absolute_dir(llm_eval_config_path)} "
    )
    if not cache_code_results:
        command_string += "\\\n\t--no_cache_code_reuslts "
    command_string += f"\\\n\t--output_dir {get_absolute_dir(output_dir)} "
    command_string += f"\\\n\t--ks '{diversity_ks}' "
    command_string += f"\\\n\t--diversity_n_samples {diversity_n_samples}"
    logger.info(f"Running command: \n{command_string}")
    with open(osp.join(output_dir, "command.sh"), "w") as f:
        f.write("""#!/bin/bash\n""")
        f.write(command_string)

    logger.info(f"Loaded multirun results from: {multirun_load_path}")
    logger.info(
        f"Running evaluation for dataset {multirun_results.dataset_name} with nruns={multirun_results.n}"
    )

    eval_config = EvalConfig(
        multirun_load_path=multirun_load_path,
        llm_eval=llm_eval_config,
        output_dir=output_dir,
        use_code_cache=cache_code_results,
        diversity_ks=diversity_ks,
        diversity_n_samples=diversity_n_samples,
    )

    logger.info(
        f"Running evaluation with config: {eval_config.model_dump_json(indent=2)}"
    )
    # run_eval_on_analyses(eval_config)


@click.command()
@click.option(
    "--multirun_load_path",
    type=str,
    required=True,
    default="./example/multirun_analyses.json",
    help="Path to load the multirun analyses",
)
@click.option(
    "--llm_eval_config_path",
    type=click.Path(exists=True, file_okay=True),
    default="./conf/llm_eval.yaml",
    help="Path to the LLM eval config file",
)
@click.option(
    "--no_cache_code_reuslts",
    "cache_code_results",
    is_flag=True,
    default=True,
    help="Whether to cache code results when running code for the evaluation",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=None,
    help="output directory to store saved eval results",
)
@click.option(
    "--ks",
    "diversity_ks",
    type=str,
    default="[1, 5, 10]",
    help="List of k values for diversity metrics. Default is [1, 5, 10]",
)
@click.option(
    "--diversity_n_samples",
    type=int,
    default=10000,
    help="Number of samples to use for diversity metrics",
)
def run_eval_click(
    multirun_load_path: str,
    llm_eval_config_path: str,
    cache_code_results: bool,
    output_dir: str,
    diversity_ks: str,
    diversity_n_samples: int,
):
    run_eval(
        multirun_load_path=multirun_load_path,
        llm_eval_config_path=llm_eval_config_path,
        cache_code_results=cache_code_results,
        output_dir=output_dir,
        diversity_ks=diversity_ks,
        diversity_n_samples=diversity_n_samples,
    )


if __name__ == "__main__":
    run_eval_click()
