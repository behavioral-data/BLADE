from typing import List

from pydantic import BaseModel
from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis


class DatasetSubmission(BaseModel):
    dataset_name: str
    analyses: List[EntireAnalysis]
