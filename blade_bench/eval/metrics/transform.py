from pydantic import BaseModel

from .base import BaseMatchMetrics
from blade_bench.eval.datamodel import MatchTransforms


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
