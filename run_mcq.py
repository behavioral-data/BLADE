import os
import click
import yaml

from blade_bench.baselines.run_mcq import RunMCQ
from blade_bench.eval.datamodel.run_mcq import MCQMetrics, MCQMetricsAcrossDatasets
from blade_bench.llms.config_load import get_models, get_providers
from blade_bench.logger import logger
from blade_bench.baselines.config import BenchmarkMCQConfig
from blade_bench.data.dataset import list_datasets_mcq


@click.command()
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
    "--llm_eval_config_path",
    type=click.Path(exists=True, file_okay=True),
    default="./conf/llm_eval.yaml",
    help="Path to the LLM eval config file",
    show_default=True,
)
@click.option(
    "--output_file",
    type=click.Path(exists=False, file_okay=False, dir_okay=True),
    default=None,
    help="output json file to store saved analyses",
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
def run_mcq(
    use_data_desc: bool,
    llm_config_path: str,
    llm_eval_config_path: str,
    output_file: str,
    llm_provider: str,
    llm_model: str,
):

    llm_config = yaml.safe_load(open(llm_config_path))
    if llm_provider:
        llm_config["provider"] = llm_provider
    if llm_model:
        llm_config["model"] = llm_model

    llm_eval_config = yaml.safe_load(open(llm_eval_config_path))
    datasets = list_datasets_mcq()
    if not output_file:
        output_file = f"./outputs/mcq/{llm_config['provider']}-{llm_config['model']}_ndatasets={len(datasets)}.json"
    # create directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    logger.info(f"MCQ Datasets: {datasets}")
    res = {}
    for i, run_dataset in enumerate(datasets):
        config = BenchmarkMCQConfig(
            llm=llm_config,
            llm_eval=llm_eval_config,
            run_dataset=run_dataset,
            use_data_desc=use_data_desc,
        )
        run_mcq = RunMCQ(config)
        results: MCQMetrics = run_mcq.run()
        logger.info(f"MCQ metrics: {results.model_dump_json(indent=2)}")
        res[run_dataset] = results

    all_res = MCQMetricsAcrossDatasets(dataset_metrics=res)
    with open(output_file, "w") as f:
        f.write(all_res.model_dump_json(indent=2))
    logger.success(f"Saved MCQ results to {output_file}")


if __name__ == "__main__":
    run_mcq()
