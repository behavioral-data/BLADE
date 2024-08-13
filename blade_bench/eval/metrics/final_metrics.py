from collections import defaultdict
from typing import Dict, List, Literal
import numpy as np
from blade_bench.eval.datamodel.run import SingleRunMetrics


def calculate_coverage(
    dataset_runs: Dict[str, List[SingleRunMetrics]], k: int = 10
) -> Dict[
    Literal["cvar", "model", "model_cvar", "transform_graph", "transform_value"],
    float,
]:

    def calculate_coverage_helper(runs: List[SingleRunMetrics]):
        coverage_cvar = (
            len(set([match for run in runs for match in run.matched_cvar]))
            / runs[0].num_cvars1
        )
        coverage_model = len(
            set([match for run in runs for match in run.matched_model])
        ) / min(runs[0].num_mspecs1_unique, len(runs))
        coverage_model_cvar = len(
            set([match for run in runs for match in run.matched_model_cvar])
        ) / min(runs[0].num_mspecs1, len(runs))

        coverage_transforms_graph = (
            len(set([match for run in runs for match in run.matched_gspec]))
            / runs[0].num_tspecs1
        )
        coverage_transforms_value = (
            len(set([match for run in runs for match in run.matched_vspec]))
            / runs[0].num_tspecs1
        )
        return {
            "cvar": coverage_cvar,
            "model": coverage_model,
            "model_cvar": coverage_model_cvar,
            "transform_graph": coverage_transforms_graph,
            "transform_value": coverage_transforms_value,
        }

    coverage_metrics = defaultdict(list)
    for _, runs in dataset_runs.items():
        for metric, v in calculate_coverage_helper(runs[:k]).items():
            coverage_metrics[metric].append(v)
    coverage_metrics = {
        metric_name: np.mean(v) for metric_name, v in coverage_metrics.items()
    }
    return coverage_metrics


def calculate_hit_rate(dataset_runs: Dict[str, List[SingleRunMetrics]]) -> Dict[
    Literal["cvar", "model", "model_cvar", "transform_graph", "transform_value"],
    float,
]:
    def calculate_hit_rate_helper(dataset_runs: List[SingleRunMetrics]):
        hit_rate_cvars = np.asarray([run.hit_rate_cvars for run in dataset_runs]).mean()
        hit_rate_models = np.asarray(
            [run.hit_rate_models for run in dataset_runs]
        ).mean()
        hit_rate_models_cvars = np.asarray(
            [run.hit_rate_models_cvar for run in dataset_runs]
        ).mean()
        hit_rate_vspecs = np.asarray(
            [run.hit_rate_vspecs for run in dataset_runs]
        ).mean()
        hit_rate_gspecs = np.asarray(
            [run.hit_rate_gspecs for run in dataset_runs]
        ).mean()

        return {
            "cvar": hit_rate_cvars,
            "model": hit_rate_models,
            "model_cvar": hit_rate_models_cvars,
            "transform_graph": hit_rate_gspecs,
            "transform_value": hit_rate_vspecs,
        }

    hit_rate_metrics = defaultdict(list)
    for dataset, runs in dataset_runs.items():
        for metric, v in calculate_hit_rate_helper(runs).items():
            hit_rate_metrics[metric].append(v)
    hit_rate_metrics = {
        metric_name: np.mean(v) for metric_name, v in hit_rate_metrics.items()
    }
    return hit_rate_metrics


def calculate_analysis_scores(
    coverage_k: int,
    dataset_runs: Dict[str, List[SingleRunMetrics]],
):
    hit_rate_metrics = calculate_hit_rate(dataset_runs)
    coverage_metrics = calculate_coverage(dataset_runs, k=coverage_k)

    def harmonic_mean(a, b):
        return 2 * a * b / (a + b)

    def get_weights():
        # Calculate the totals
        totals = {
            "tspecs": sum(v[0].num_tspecs1 for v in dataset_runs.values()),
            "cvars": sum(v[0].num_cvars1 for v in dataset_runs.values()),
            "mspecs": sum(
                min(v[0].num_mspecs1, coverage_k) for v in dataset_runs.values()
            ),
            "mspecs_unique": sum(
                min(v[0].num_mspecs1_unique, coverage_k) for v in dataset_runs.values()
            ),
        }
        # Calculate the total sum
        total_sum = sum(totals.values())
        # Calculate the weights
        weights = {
            "cvar": totals["cvars"] / total_sum,
            "model": totals["mspecs_unique"] / total_sum,
            "model_cvar": totals["mspecs"] / total_sum,
            "gspec": totals["tspecs"] / (2 * total_sum),
            "vspec": totals["tspecs"] / (2 * total_sum),
        }
        return weights

    f1_cvars = harmonic_mean(hit_rate_metrics["cvar"], coverage_metrics["cvar"])
    f1_models = harmonic_mean(hit_rate_metrics["model"], coverage_metrics["model"])
    f1_models_cvars = harmonic_mean(
        hit_rate_metrics["model_cvar"], coverage_metrics["model_cvar"]
    )
    f1_gspecs = harmonic_mean(
        hit_rate_metrics["transform_graph"], coverage_metrics["transform_graph"]
    )
    f1_vspecs = harmonic_mean(
        hit_rate_metrics["transform_value"], coverage_metrics["transform_value"]
    )

    unweighted_score = (
        f1_cvars + f1_models + f1_models_cvars + 0.5 * f1_gspecs + 0.5 + f1_vspecs
    ) / 4

    weights = get_weights()
    weighted_score = (
        weights["cvar"] * f1_cvars
        + weights["model"] * f1_models
        + weights["model_cvar"] * f1_models_cvars
        + weights["gspec"] * f1_gspecs
        + weights["vspec"] * f1_vspecs
    )
    return weighted_score, unweighted_score
