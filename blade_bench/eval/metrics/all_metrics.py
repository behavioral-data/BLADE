from pydantic import BaseModel

from blade_bench.eval.datamodel.match import MatchedAnnotations
from blade_bench.eval.metrics.cvar import CVMatch, get_cv_metrics
from blade_bench.eval.metrics.model import ModelMatch, get_model_match
from blade_bench.eval.metrics.transform import TransformMatch, get_transform_metrics


class AllMetrics(BaseModel):
    smodel_match: ModelMatch
    cv_match: CVMatch
    transform_match: TransformMatch


def get_metrics_from_match_obj(matched_annotations: MatchedAnnotations):
    model_metrics = get_model_match(matched_annotations.matched_models)
    cv_metrics = get_cv_metrics(matched_annotations.matched_cvars)
    transform_metrics = get_transform_metrics(matched_annotations.matched_transforms)
    return AllMetrics(
        smodel_match=model_metrics,
        cv_match=cv_metrics,
        transform_match=transform_metrics,
    )
