from typing import List
from blade_bench.data.datamodel.specs import ROOT_SPEC_ID, ROOT_SPEC_NAME
from blade_bench.eval.datamodel.match import MatchedAnnotations
from blade_bench.eval.datamodel.result import EvalResults
from blade_bench.eval.datamodel.run import MetricsAcrossRuns, RunResultModes
from blade_bench.eval.metrics.all_metrics import AllMetrics


class CalcSubmissionMetrics:
    def __init__(self, submission: EvalResults):
        self.submission = submission

    def calculate_diversity(self):
        single_run_res = [res.eval_run_result for res in self.submission.results]
        # TODO calculate this

    def get_error_classes(self):
        single_run_res = [res.eval_run_result for res in self.submission.results]
        # TODO calculate this

    def __update_transform_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        a = matched_annotations.matched_transforms.matched_tspecs
        matched_vspec_ids = [
            spec.spec_id
            for spec in a.vspecs1
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]
        matched_gspec_ids = [
            spec.spec_id
            for spec in a.gspecs1
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]
        metrics.all_transforms_match_vspecs.update(matched_vspec_ids)
        metrics.all_transforms_match_gspecs.update(matched_gspec_ids)

        metrics.num_match_vspec.append(len(matched_vspec_ids))
        metrics.num_match_gspec.append(len(matched_gspec_ids))

        matched_vspec_ids2 = [
            spec.spec_id
            for spec in a.vspecs2
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]
        matched_gspec_ids2 = [
            spec.spec_id
            for spec in a.gspecs2
            if spec.spec_id != ROOT_SPEC_ID and spec.spec_name != ROOT_SPEC_NAME
        ]

        metrics.all_transforms_match_vspecs2.update(matched_vspec_ids2)
        metrics.all_transforms_match_gspecs2.update(matched_gspec_ids2)
        metrics.num_match_vspec2.append(len(matched_vspec_ids2))
        metrics.num_match_gspec2.append(len(matched_gspec_ids2))

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
        match_spec_id = None
        for match in matched_annotations.matched_models.matched_unique.values():
            if match_model_cvar:
                break
            for match_non_unique in matched_annotations.matched_models.matched.values():
                if match_non_unique.model1.spec_name == match.model1.spec_name:
                    if match_non_unique.is_cvar_all_matched():
                        match_model_cvar = True
                        match_spec_id = match_non_unique.model1.spec_id
                        break
        metrics.num_match_model_cvar.append(int(match_model_cvar))
        if match_spec_id:
            metrics.all_models_match_cvar.add(match_spec_id)

    def __update_model_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        match_unique = list(matched_annotations.matched_models.matched_unique.values())
        if match_unique:
            matched_unique_spec_name = match_unique[0].model1.specification
            metrics.all_models_match.add(matched_unique_spec_name)
            metrics.num_match_model.append(1)
        else:
            metrics.num_match_model.append(0)

    def __update_cvar_metrics_from_matched(
        self, matched_annotations: MatchedAnnotations, metrics: MetricsAcrossRuns
    ):
        matched_cvars = [
            matched
            for k, v in matched_annotations.matched_cvars.items()
            for matched in v.matched.values()
            if matched.similarity >= 8
        ]
        metrics.num_match_cvar.append(len(matched_cvars))
        metrics.num_cvars2.append(len(matched_annotations.cvars2))
        for m in matched_cvars:
            metrics.all_cvars_match.add(m.var1)

    def calculate_metrics(self) -> MetricsAcrossRuns:
        metrics = MetricsAcrossRuns()
        for i, submission_res in enumerate(self.submission.results):
            matched_annotations = submission_res.matched_annotations
            run_res = submission_res.eval_run_result
            analysis = submission_res.analysis
            metrics.analyses.append(analysis)
            metrics.converted_code.append(
                submission_res.analysis_processed.transform_state.converted_code
            )
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
                metrics.num_cvars1 = len(matched_annotations.cvars1)
            else:
                metrics.status.append(RunResultModes(run_res.res_type_transform))
                metrics.num_match_gspec.append(0)
                metrics.num_match_vspec.append(0)
                metrics.num_tspecs2.append(0)

            self.__update_model_match_cvar_metrics_from_matched(
                matched_annotations, metrics
            )
            self.__update_model_metrics_from_matched(matched_annotations, metrics)
            self.__update_cvar_metrics_from_matched(matched_annotations, metrics)
        return metrics
