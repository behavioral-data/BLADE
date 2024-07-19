from typing import Optional
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


class DiversityMetric(BaseModel):
    k: Optional[int] = None
    transforms_value: float
    transforms_graph: float
    cvars: float
    models: float

    def get_metrics(self):
        post_fix = "" if self.k is None else f"_k={self.k:02}"
        return {
            f"div_transforms_value{post_fix}": self.transforms_value,
            f"div_transforms_graph{post_fix}": self.transforms_graph,
            f"div_cvars{post_fix}": self.cvars,
            f"div_models{post_fix}": self.models,
        }
