from typing import List
from pydantic import BaseModel

from blade_bench.eval.datamodel.match import MatchedModels
from blade_bench.eval.metrics.cvar import CVMatch, get_cv_metrics


class ModelMatchMetrics(BaseModel):
    num_match1: int
    num_match2: int
    num_total1: int
    num_total2: int
    num_unqiue1: int
    num_match_unique: int


class ModelMatch(BaseModel):
    match_semantics: ModelMatchMetrics
    match_conceptual_model: List[CVMatch]
    match_conceptual_model_unique: List[CVMatch]


def get_model_match(matched_models: MatchedModels):
    num_total1 = len(matched_models.input_models1)
    num_total2 = len(matched_models.input_models2)
    num_match1 = 0
    num_match2 = 0
    match_cvs = []
    for key, value in matched_models.matched.items():
        num_match1 += 1
        num_match2 += 1
        match_cvs.append(
            get_cv_metrics(value.matched_cvars),  # just a placehodler for this to work
        )

    match_cvs_unqiue = []
    for key, value in matched_models.matched_unique.items():
        if value.matched_cvars:
            match_cvs_unqiue.append(get_cv_metrics(value.matched_cvars))
    return ModelMatch(
        match_semantics=ModelMatchMetrics(
            num_match1=num_match1,
            num_match2=num_match2,
            num_total1=num_total1,
            num_total2=num_total2,
            num_unqiue1=len(matched_models.input_models1_unique),
            num_match_unique=len(matched_models.matched_unique),
        ),
        match_conceptual_model=match_cvs,
        match_conceptual_model_unique=match_cvs_unqiue,
    )
