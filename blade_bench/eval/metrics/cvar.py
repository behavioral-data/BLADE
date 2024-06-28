from typing import Dict, Literal
from blade_bench.eval.datamodel.match import MatchedCvars
from blade_bench.eval.metrics.base import BaseMatchMetrics


class CVMatch(BaseMatchMetrics):
    pass


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
