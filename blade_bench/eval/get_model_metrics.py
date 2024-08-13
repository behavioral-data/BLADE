from collections import Counter, defaultdict
import json
import os
import random
from typing import Dict, List, Literal, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from blade_bench.eval.datamodel.run import MetricsAcrossRuns, SingleRunMetrics
from blade_bench.eval.metrics.final_metrics import calculate_analysis_scores
from blade_bench.eval.utils import bootstrapped_mean


class AnalysisScore(BaseModel):
    mean_weighted: float
    mean_unweighted: float
    mean_bootstrapped_weighted: float
    mean_bootstrapped_unweighted: float
    ci_weighted: List[float]
    ci_unweighted: List[float]


class GetFinalMetricsForModel:
    CONVERT_MAP = {
        "RunResultModes.FINISHED_SUCCESSFULLY": "No Execution Errors",
        "RunResultModes.LM_GENERATION_FAILED": "No Generation",
        "RunResultModes.LM_SUBMISSION_CONVERSION_FAILED": "Submission Conversion Failed",
        "RunResultModes.LM_SUBMISSION_TRANSFORM_CODE_EMPTY": "Empty Transform",
        "RunResultModes.LM_SUBMISSION_EXECUTION_FAILED": "Transform Execution Errors",
        "RunResultModes.GETTING_METRICS_FAILED": "Metrics Calculation Failed",
    }

    def __init__(self, datasets_to_path: Dict[str, str], model_name: str):
        self.datasets_to_path = datasets_to_path
        self.model_name = model_name
        self.df, self.metrics_objs = self.get_metrics_for_datasets(
            self.datasets_to_path, self.model_name
        )

    def get_metrics_for_datasets(
        self,
        dataset_to_result_path: Dict[str, str],
        model_name: str = None,
    ) -> Tuple[pd.DataFrame, Dict[str, MetricsAcrossRuns]]:
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
            A dictionary with the MetricsAcrossRuns object for each dataset
        """

        df = pd.DataFrame()
        metrics_objs = {}
        for dataset, dataset_res_older in dataset_to_result_path.items():
            eval_metrics_file = os.path.join(dataset_res_older, "eval_metrics.json")
            with open(eval_metrics_file, "r") as f:
                data = json.load(f)
            metrics_obj = MetricsAcrossRuns(**data)
            metrics_objs[dataset] = metrics_obj
            info_d = {"model": model_name, "dataset": dataset}
            d = metrics_obj.get_metrics()
            df_d = {**d, **info_d}
            col_order = ["model", "dataset"] + list(sorted(d.keys()))
            df_add = pd.DataFrame.from_dict(df_d, orient="index").T[col_order]
            df = pd.concat([df, df_add], axis=0)
        df[list(d.keys())] = df[list(d.keys())].apply(
            lambda x: pd.to_numeric(x, errors="ignore")
        )
        to_round = [c for c in df.columns if "hit_rate" in c or "average_coverage" in c]
        df[to_round] = (df[to_round].replace(-1, 0) * 100).round(2).replace(-1, 0)
        return df, metrics_objs

    def get_bootstrap_metrics_dict(
        self, metrics_objs: Dict[str, MetricsAcrossRuns], k=10, num_samples=1000
    ):
        """

        returns:
        dict
            A dictionary of metric name to the list of metrics for each run across datasets for bootstrapping
        """
        bootstrap_metrics_dict = defaultdict(list)
        for dataset, metrics_obj in metrics_objs.items():
            self._add_individual_run_metrics_for_bootstrap(
                bootstrap_metrics_dict,
                metrics_obj,
                k=k,
                num_samples=num_samples,
            )
        return bootstrap_metrics_dict

    def analysis_metric_bootstrap_calculation(
        self,
        coverage_k: int = 10,
        num_samples=1000,
        cis=[0.025, 0.975],
        to_str=False,
        precision=3,
    ):
        """
        Calculate the metrics for the runs
        """
        metrics_objs = self.metrics_objs
        dataset_runs = {k: v.get_single_run_metrics() for k, v in metrics_objs.items()}
        bootstrap_statistics_weighted = []
        bootstrap_statistics_unweighted = []
        for i in tqdm(range(num_samples)):
            new_dataset_runs = {}
            for k, runs in dataset_runs.items():
                assert len(runs) > coverage_k, "Number of runs should be greater than coverage k"
                # Step 2: Sample answers within the sampled run
                new_dataset_runs[k] = random.choices(runs, k=len(runs))
            weighted_stat, unweighted_stat = calculate_analysis_scores(
                coverage_k, new_dataset_runs
            )
            bootstrap_statistics_weighted.append(weighted_stat)
            bootstrap_statistics_unweighted.append(unweighted_stat)

        weighted_mean_stat, unweighted_mean_stat = calculate_analysis_scores(
            coverage_k, dataset_runs
        )
        cis_weighted = np.quantile(bootstrap_statistics_weighted, cis)
        cis_unweighted = np.quantile(bootstrap_statistics_unweighted, cis)
        btstrap_mean = np.mean(bootstrap_statistics_weighted)
        btstrap_mean_unweighted = np.mean(bootstrap_statistics_unweighted)

        if to_str:
            mean = "{0:.{1}f}".format(weighted_mean_stat, precision)
            btstrap_mean = "{0:.{1}f}".format(btstrap_mean, precision)
            ci_0 = "{0:.{1}f}".format(cis_weighted[0], precision)
            ci_1 = "{0:.{1}f}".format(cis_unweighted[1], precision)

            btstrap_mean_unweighted = "{0:.{1}f}".format(
                btstrap_mean_unweighted, precision
            )
            mean2 = "{0:.{1}f}".format(unweighted_mean_stat, precision)
            ci_02 = "{0:.{1}f}".format(cis_unweighted[0], precision)
            ci_12 = "{0:.{1}f}".format(cis_unweighted[1], precision)
            return (
                f"{mean} {btstrap_mean} ({ci_0}, {ci_1})",
                f"{mean2} {btstrap_mean_unweighted} ({ci_02}, {ci_12})",
            )

        analysis_score = AnalysisScore(
            mean_weighted=weighted_mean_stat,
            mean_unweighted=unweighted_mean_stat,
            mean_bootstrapped_weighted=btstrap_mean,
            mean_bootstrapped_unweighted=btstrap_mean_unweighted,
            ci_weighted=list(cis_weighted),
            ci_unweighted=list(cis_unweighted),
        )

        return analysis_score

    def get_status_count(self) -> Dict[str, int]:
        ret = dict(
            Counter(
                [
                    self.CONVERT_MAP[s]
                    for statuses in self.df["status"].values
                    for s in statuses
                ]
            )
        )
        return ret

    def _add_individual_run_metrics_for_bootstrap(
        self,
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

        bootstrap_metrics_dict[
            "coverage_transforms_graph@{}-bootstrap".format(k)
        ].extend(
            metrics_obj.average_coverage_gspecs_k(
                k,
                use_combinations=True,
                num_samples=num_samples,
                include_gen_errors=include_gen_errors,
            )
        )

        bootstrap_metrics_dict[
            "coverage_transforms_value@{}-bootstrap".format(k)
        ].extend(
            metrics_obj.average_coverage_vspecs_k(
                k,
                use_combinations=True,
                num_samples=num_samples,
                include_gen_errors=include_gen_errors,
            )
        )

    def get_bootstrap_scores(self) -> Dict[str, float]:
        ret_dict = {}
        bootstrap_metrics_dict = self.get_bootstrap_metrics_dict(self.metrics_objs)
        hit_rate_keys = [k for k in bootstrap_metrics_dict.keys() if "hit_rate" in k]
        coverage_keys = [k for k in bootstrap_metrics_dict.keys() if "coverage" in k]

        hit_rate_bootsraps = [
            i for k in hit_rate_keys for i in bootstrap_metrics_dict[k]
        ]
        coverage_bootsraps = [
            i for k in coverage_keys for i in bootstrap_metrics_dict[k]
        ]

        ret_dict["hit_rate_mean"] = bootstrapped_mean(
            pd.Series(hit_rate_bootsraps), n_samples=1000, to_str=True
        )

        ret_dict["coverage_mean"] = bootstrapped_mean(
            pd.Series(coverage_bootsraps), n_samples=1000, to_str=True
        )

        for k, v in bootstrap_metrics_dict.items():
            ret_dict[k] = bootstrapped_mean(pd.Series(v), n_samples=1000, to_str=True)
        return ret_dict
