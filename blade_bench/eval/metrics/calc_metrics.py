from collections import Counter
from typing import List
from blade_bench.data.datamodel.specs import ROOT_SPEC_ID, ROOT_SPEC_NAME
from blade_bench.eval.datamodel.match import MatchedAnnotations
from blade_bench.eval.datamodel.result import EvalResults, SubmissionRuns
from blade_bench.eval.datamodel.run import MetricsAcrossRuns, RunResultModes
from blade_bench.eval.metrics.all_metrics import AllMetrics, DiversityMetric
import random


def sample_if_greater_than_k(lst, sample_size):
    """
    Samples elements from the list if its length is greater than the sample size.

    Parameters:
    lst (list): The list to sample from.
    sample_size (int): The number of elements to sample.

    Returns:
    list: A sampled list if the original list length is greater than k; otherwise, the original list.
    """
    if len(lst) > sample_size:
        return random.sample(lst, min(sample_size, len(lst)))
    return lst


def simpson_di(data):
    """Given a hash { 'species': count } , returns the Simpson Diversity Index

    >>> simpson_di({'a': 10, 'b': 20, 'c': 30,})
    0.3888888888888889
    """

    def p(n, N):
        """Relative abundance"""
        if n == 0:
            return 0
        else:
            return float(n) / N

    N = sum(data.values())

    return 1 - sum(p(n, N) ** 2 for n in data.values() if n != 0)


class CalcSubmissionMetrics:
    def __init__(
        self, submission: EvalResults, ks: List[int] = [5], num_samples: int = 1000
    ):
        self.submission = submission
        self.ks = ks
        self.num_samples = num_samples

    def get_submission_matched_items(self):
        all_submission_matched_transforms_value = []
        all_submission_matched_transforms_graph = []
        all_submission_matched_model = []
        all_submission_matched_conceptual_vars = []

        for i, submission_res in enumerate(self.submission.results):

            matched_annotations = submission_res.matched_annotations
            submission_matched_model = []
            submission_matched_conceptual_vars = []
            run_res = submission_res.eval_run_result
            if run_res.res_type != RunResultModes.FINISHED_SUCCESSFULLY.value:
                all_submission_matched_conceptual_vars.append([])
                all_submission_matched_model.append([])
                all_submission_matched_transforms_value.append([])
                all_submission_matched_transforms_graph.append([])
                continue

            for k, matched_cvars in matched_annotations.matched_cvars.items():
                for matched in matched_cvars.matched.values():
                    if matched.similarity >= 8:
                        submission_matched_conceptual_vars.append(matched.var1)
            for (
                k,
                matched_model,
            ) in matched_annotations.matched_models.matched_unique.items():
                submission_matched_model.append(matched_model.model1.spec_name)

            all_submission_matched_transforms_value.append(
                [
                    spec.spec_name
                    for spec in matched_annotations.matched_transforms.matched_tspecs.vspecs1
                    if spec.spec_name != ROOT_SPEC_NAME and spec.spec_id != ROOT_SPEC_ID
                ]
            )
            all_submission_matched_transforms_graph.append(
                [
                    spec.spec_name
                    for spec in matched_annotations.matched_transforms.matched_tspecs.gspecs1
                    if spec.spec_name != ROOT_SPEC_NAME and spec.spec_id != ROOT_SPEC_ID
                ]
            )
            all_submission_matched_model.append(submission_matched_model)
            all_submission_matched_conceptual_vars.append(
                submission_matched_conceptual_vars
            )
        return SubmissionRuns(
            all_submission_matched_cvars=all_submission_matched_conceptual_vars,
            all_submission_matched_models=all_submission_matched_model,
            all_submission_matched_transforms_value=all_submission_matched_transforms_value,
            all_submission_matched_transforms_graph=all_submission_matched_transforms_graph,
        )

    def get_average_diversity(self, items):
        item_counts = [Counter(item) for item in items]
        diversity = [simpson_di(item) for item in item_counts if len(item) > 0]
        return sum(diversity) / max(len(diversity), 1)

    def calculate_diversity(self, ks: List[int] = [5], num_samples: int = 1000):
        submission_runs = self.get_submission_matched_items()
        diversity_l = []

        for k in ks:
            transforms_value = submission_runs.transforms_value_k(
                min(len(self.submission.results), k), num_samples
            )
            transforms_graph = submission_runs.transforms_graph_k(
                min(len(self.submission.results), k), num_samples
            )
            conceptual_vars = submission_runs.cvars_k(
                min(len(self.submission.results), k), num_samples
            )
            models = submission_runs.models_k(
                min(len(self.submission.results), k), num_samples
            )

            avg_diversity_transforms_value = self.get_average_diversity(
                transforms_value
            )
            avg_diversity_transforms_graph = self.get_average_diversity(
                transforms_graph
            )
            avg_diversity_conceptual_vars = self.get_average_diversity(conceptual_vars)
            avg_diversity_models = self.get_average_diversity(models)
            diversity_l.append(
                DiversityMetric(
                    k=k,
                    transforms_graph=avg_diversity_transforms_graph,
                    transforms_value=avg_diversity_transforms_value,
                    cvars=avg_diversity_conceptual_vars,
                    models=avg_diversity_models,
                )
            )
        return diversity_l

    def get_error_classes(self):
        single_run_res = [res.eval_run_result for res in self.submission.results]
        # TODO calculate this

    def __update_transform_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        a = matched_annotations.matched_transforms.matched_tspecs
        matched_vspec_ids = [
            spec.spec_name
            for spec in a.vspecs1
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]
        matched_gspec_ids = [
            spec.spec_name
            for spec in a.gspecs1
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]

        metrics.matched_vspecs.append(matched_vspec_ids)
        metrics.matched_gspecs.append(matched_gspec_ids)

        matched_vspec_ids2 = [
            spec.spec_name
            for spec in a.vspecs2
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]
        matched_gspec_ids2 = [
            spec.spec_name
            for spec in a.gspecs2
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]

        metrics.matched_vspecs2.append(matched_vspec_ids2)
        metrics.matched_gspecs2.append(matched_gspec_ids2)

        metrics.num_tspecs2.append(
            len(
                matched_annotations.matched_transforms.transform_state2.expanded_id_to_spec
            )
            - 1
        )

    def __update_model_match_cvar_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        match_model_cvar = False
        match_spec_name = None
        for match in matched_annotations.matched_models.matched_unique.values():
            if match_model_cvar:
                break
            for match_non_unique in matched_annotations.matched_models.matched.values():
                if match_non_unique.model1.spec_name == match.model1.spec_name:
                    if match_non_unique.is_cvar_all_matched():
                        match_model_cvar = True
                        match_spec_name = (
                            match_non_unique.model1.spec_name
                            + " | "
                            + ", ".join(match_non_unique.cvars1)
                        )
                        break

        if match_spec_name:
            metrics.matched_models_cvar.append([match_spec_name])
        else:
            metrics.matched_models_cvar.append([])

    def __update_model_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        match_unique = list(matched_annotations.matched_models.matched_unique.values())
        if match_unique:
            matched_unique_spec_name = match_unique[0].model1.specification
            metrics.matched_models.append([matched_unique_spec_name])
        else:
            metrics.matched_models.append([])

    def __update_cvar_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        matched_cvars = [
            matched
            for k, v in matched_annotations.matched_cvars.items()
            for matched in v.matched.values()
            if matched.similarity >= 8
        ]
        metrics.matched_cvars.append(list(set([m.var1 for m in matched_cvars])))
        metrics.matched_cvars2.append(list(set([m.var2 for m in matched_cvars])))
        metrics.num_cvars2.append(len(matched_annotations.cvars2))

    def calculate_metrics(self) -> MetricsAcrossRuns:
        metrics = MetricsAcrossRuns()
        for i, submission_res in enumerate(self.submission.results):
            matched_annotations = submission_res.matched_annotations
            run_res = submission_res.eval_run_result
            analysis = submission_res.analysis
            metrics.analyses.append(analysis)

            if run_res.res_type != RunResultModes.FINISHED_SUCCESSFULLY.value:
                metrics.status.append(RunResultModes(run_res.res_type))
                continue
            if run_res.res_type_transform is None:
                self.__update_transform_metrics_from_matched(
                    matched_annotations, metrics
                )
                metrics.status.append(RunResultModes.FINISHED_SUCCESSFULLY)
                metrics.num_tspecs1 = matched_annotations.tspecs1
                metrics.num_mspecs1 = matched_annotations.mspecs1
                metrics.num_mspecs1_unique = matched_annotations.mspecs_unique
                metrics.converted_code.append(
                    submission_res.analysis_processed.transform_state.converted_code
                )
            else:
                metrics.status.append(RunResultModes(run_res.res_type_transform))
                metrics.matched_vspecs.append([])
                metrics.matched_gspecs.append([])
                metrics.matched_vspecs2.append([])
                metrics.matched_gspecs2.append([])
                metrics.converted_code.append("")
                metrics.num_tspecs2.append(0)

            metrics.num_cvars1 = len(matched_annotations.cvars1)

            self.__update_model_match_cvar_metrics_from_matched(
                matched_annotations, metrics
            )
            self.__update_model_metrics_from_matched(matched_annotations, metrics)
            self.__update_cvar_metrics_from_matched(matched_annotations, metrics)
        metrics.diversity_metrics = self.calculate_diversity(self.ks, self.num_samples)
        return metrics


if __name__ == "__main__":
    pass
