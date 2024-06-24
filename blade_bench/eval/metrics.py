from typing import Dict, List, Literal, Set
from pydantic import BaseModel

from .datamodel import (
    MatchTransforms,
    MatchedAnnotations,
    MatchedCvars,
    MatchedModels,
)


class BaseMatchMetrics(BaseModel):
    num_match1: int
    num_match2: int
    num_total1: int
    num_total2: int

    @property
    def match_rate1(self):
        return self.num_match1 / self.num_total1

    @property
    def match_rate2(self):
        return self.num_match2 / self.num_total2


class ModelMatchMetrics(BaseModel):
    num_match1: int
    num_match2: int
    num_total1: int
    num_total2: int
    num_unqiue1: int
    num_match_unique: int


class CVMatch(BaseMatchMetrics):
    pass


class ModelMatch(BaseModel):
    match_semantics: ModelMatchMetrics
    match_conceptual_model: List[CVMatch]
    match_conceptual_model_unique: List[CVMatch]


class TransformMatch(BaseModel):
    match_graph: BaseMatchMetrics
    match_value: BaseMatchMetrics
    match_categorical: BaseMatchMetrics
    match_value_and_graph: BaseMatchMetrics

    @property
    def match_rate1_graph(self):
        return self.match_graph.match_rate1

    @property
    def match_rate2_graph(self):
        return self.match_graph.match_rate2

    @property
    def match_rate1_value(self):
        return self.match_value.match_rate1

    @property
    def match_rate2_value(self):
        return self.match_value.match_rate2


class AllMetrics(BaseModel):
    model_match: ModelMatch
    cv_match: CVMatch
    transform_match: TransformMatch


def get_transform_metrics(matched_transforms: MatchTransforms):
    if (
        matched_transforms.transform_state1 is None
        and matched_transforms.transform_state2 is None
    ):
        return TransformMatch(
            match_graph=BaseMatchMetrics(
                num_match1=0, num_match2=0, num_total1=0, num_total2=0
            ),
            match_value=BaseMatchMetrics(
                num_match1=0, num_match2=0, num_total1=0, num_total2=0
            ),
            match_categorical=BaseMatchMetrics(
                num_match1=0, num_match2=0, num_total1=0, num_total2=0
            ),
            match_value_and_graph=BaseMatchMetrics(
                num_match1=0, num_match2=0, num_total1=0, num_total2=0
            ),
        )

    num_value1 = max(0, len(matched_transforms.matched_tspecs.vspecs1) - 1)
    total_value1 = len(matched_transforms.transform_state1.expanded_id_to_spec) - 1
    num_value2 = max(0, len(matched_transforms.matched_tspecs.vspecs2) - 1)
    total_value2 = len(matched_transforms.transform_state2.expanded_id_to_spec) - 1
    match_value = BaseMatchMetrics(
        num_match1=num_value1,
        num_match2=num_value2,
        num_total1=total_value1,
        num_total2=total_value2,
    )

    num_graph1 = max(0, len(matched_transforms.matched_tspecs.gspecs1) - 1)
    total_graph1 = len(matched_transforms.transform_state1.expanded_id_to_spec) - 1
    num_graph2 = max(0, len(matched_transforms.matched_tspecs.gspecs2) - 1)
    total_graph2 = len(matched_transforms.transform_state2.expanded_id_to_spec) - 1
    match_graph = BaseMatchMetrics(
        num_match1=num_graph1,
        num_match2=num_graph2,
        num_total1=total_graph1,
        num_total2=total_graph2,
    )

    num_categorical1 = max(0, len(matched_transforms.matched_tspecs.cat_specs1) - 1)
    total_categorical1 = (
        len(matched_transforms.transform_state1.expanded_id_to_spec) - 1
    )
    num_categorical2 = max(0, len(matched_transforms.matched_tspecs.cat_specs2) - 1)
    total_categorical2 = (
        len(matched_transforms.transform_state2.expanded_id_to_spec) - 1
    )
    match_categorical = BaseMatchMetrics(
        num_match1=num_categorical1,
        num_match2=num_categorical2,
        num_total1=total_categorical1,
        num_total2=total_categorical2,
    )

    num_value_and_graph1 = max(0, len(matched_transforms.matched_tspecs.vspecs1) - 1)
    total_value_and_graph1 = (
        len(matched_transforms.transform_state1.expanded_id_to_spec) - 1
    )
    num_value_and_graph2 = max(0, len(matched_transforms.matched_tspecs.vspecs2) - 1)
    total_value_and_graph2 = (
        len(matched_transforms.transform_state2.expanded_id_to_spec) - 1
    )

    match_value_and_graph = BaseMatchMetrics(
        num_match1=num_value_and_graph1,
        num_match2=num_value_and_graph2,
        num_total1=total_value_and_graph1,
        num_total2=total_value_and_graph2,
    )

    return TransformMatch(
        match_graph=match_graph,
        match_value=match_value,
        match_categorical=match_categorical,
        match_value_and_graph=match_value_and_graph,
    )


def get_cv_metrics(
    matched_cvars: Dict[Literal["Control", "IV", "DV", "Moderator"], MatchedCvars]
):
    num_total1 = 0
    num_total2 = 0
    num_match1 = 0
    num_match2 = 0
    for key, value in matched_cvars.items():
        num_total1 += len(value.input_vars1)
        num_total2 += len(value.input_vars2)
        for m in value.matched.values():
            num_match1 += 1
            num_match2 += 1
    return CVMatch(
        num_match1=num_match1,
        num_match2=num_match2,
        num_total1=num_total1,
        num_total2=num_total2,
    )


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


def get_metrics_from_match_obj(matched_annotations: MatchedAnnotations):
    model_metrics = get_model_match(matched_annotations.matched_models)
    cv_metrics = get_cv_metrics(matched_annotations.matched_cvars)
    transform_metrics = get_transform_metrics(matched_annotations.matched_transforms)
    return AllMetrics(
        model_match=model_metrics,
        cv_match=cv_metrics,
        transform_match=transform_metrics,
    )
