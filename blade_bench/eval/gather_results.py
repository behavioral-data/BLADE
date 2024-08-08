from collections import defaultdict
import json
import os
from typing import Dict, Tuple

import pandas as pd

from blade_bench.eval.datamodel.run import MetricsAcrossRuns


def get_metrics_from_dict(data):
    d = MetricsAcrossRuns(**data)
    metrics_dict = d.get_metrics()
    for div in d.diversity_metrics:
        metrics_dict.update(div.get_metrics())
    return metrics_dict


def add_individual_run_metrics_for_bootstrap(
    bootstrap_metrics_dict: Dict[str, list],
    metrics_obj: MetricsAcrossRuns,
    k=10,
    num_samples=100,
    include_gen_errors=True,
):
    """
    Add to bootstrap_metrics_dict the metrics for the individual runs for bootstrapping
    """
    match_cvars = []
    match_models = []
    match_models_cvars = []
    match_gspecs = []
    match_vspecs = []

    for i, n_match in enumerate(metrics_obj.num_match_cvar2):
        match_cvars.append(n_match / metrics_obj.num_cvars2[i])

    for i, n_match in enumerate(metrics_obj.num_match_model):
        match_models.append(n_match / 1)

    for i, n_match in enumerate(metrics_obj.num_match_model_cvar):
        match_models_cvars.append(n_match / 1)

    for i, n_match in enumerate(metrics_obj.num_match_gspec2):
        match_gspecs.append(n_match / max(1, metrics_obj.num_tspecs2[i]))

    for i, n_match in enumerate(metrics_obj.num_match_vspec2):
        match_vspecs.append(n_match / max(metrics_obj.num_tspecs2[i], 1))

    bootstrap_metrics_dict["hit_rate_cvars-bootstrap"].extend(
        match_cvars + [0] * (metrics_obj.count_generation_failed)
    )
    bootstrap_metrics_dict["hit_rate_models-bootstrap"].extend(
        match_models + [0] * (metrics_obj.count_generation_failed)
    )
    bootstrap_metrics_dict["hit_rate_models_cvars-bootstrap"].extend(
        match_models_cvars + [0] * (metrics_obj.count_generation_failed)
    )
    bootstrap_metrics_dict["hit_rate_transforms_graph-bootstrap"].extend(
        match_gspecs + [0] * (metrics_obj.count_generation_failed)
    )
    bootstrap_metrics_dict["hit_rate_transforms_value-bootstrap"].extend(
        match_vspecs + [0] * (metrics_obj.count_generation_failed)
    )

    bootstrap_metrics_dict["coverage_cvars@{}-bootstrap".format(k)].extend(
        metrics_obj.average_coverage_cvars_k(
            k,
            use_combinations=True,
            num_samples=num_samples,
            include_gen_errors=include_gen_errors,
        )
    )

    bootstrap_metrics_dict["coverage_models@{}-bootstrap".format(k)].extend(
        metrics_obj.average_coverage_models_k(
            k,
            use_combinations=True,
            num_samples=num_samples,
            include_gen_errors=include_gen_errors,
        )
    )

    bootstrap_metrics_dict["coverage_models_cvars@{}-bootstrap".format(k)].extend(
        metrics_obj.average_coverage_cvar_models_k(
            k,
            use_combinations=True,
            num_samples=num_samples,
            include_gen_errors=include_gen_errors,
        )
    )

    bootstrap_metrics_dict["coverage_transforms_graph@{}-bootstrap".format(k)].extend(
        metrics_obj.average_coverage_gspecs_k(
            k,
            use_combinations=True,
            num_samples=num_samples,
            include_gen_errors=include_gen_errors,
        )
    )

    bootstrap_metrics_dict["coverage_transforms_value@{}-bootstrap".format(k)].extend(
        metrics_obj.average_coverage_vspecs_k(
            k,
            use_combinations=True,
            num_samples=num_samples,
            include_gen_errors=include_gen_errors,
        )
    )


def get_metrics_for_datasets(
    dataset_to_result_path: Dict[str, str], model_name: str = None, **kwargs
) -> Tuple[pd.DataFrame, dict, dict]:
    """
    params:
    dataset_to_result_path: Dict[str, str]
        A dictionary mapping dataset name to the path of the result directory
    model_name: str
        The name of the model for which the metrics are being collected

    returns:
    pd.DataFrame
        A DataFrame with the metrics for each dataset
    dict
        A dictionary of metric name to the list of metrics for each dataset and each run for bootstrapping
    dict
        A dictionary with the MetricsAcrossRuns object for each dataset
    """

    bootstrap_metrics_dict = defaultdict(list)
    df = pd.DataFrame()
    metrics_objs = {}
    for dataset, dataset_res_older in dataset_to_result_path.items():
        eval_metrics_file = os.path.join(dataset_res_older, "eval_metrics.json")
        with open(eval_metrics_file, "r") as f:
            data = json.load(f)
        metrics_obj = MetricsAcrossRuns(**data)
        metrics_objs[dataset] = metrics_obj
        d = get_metrics_from_dict(data)
        add_individual_run_metrics_for_bootstrap(
            bootstrap_metrics_dict, metrics_obj, **kwargs
        )
        info_d = {"model": model_name, "dataset": dataset}
        df_d = {**d, **info_d}
        col_order = ["model", "dataset"] + list(sorted(d.keys()))
        df_add = pd.DataFrame.from_dict(df_d, orient="index").T[col_order]
        df = pd.concat([df, df_add], axis=0)
        df[list(d.keys())] = df[list(d.keys())].apply(
            lambda x: pd.to_numeric(x, errors="ignore")
        )
    return df, bootstrap_metrics_dict, metrics_objs
