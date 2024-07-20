from typing import Dict, Tuple, Union

from pydantic import BaseModel

from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis
from blade_bench.eval.datamodel.run import RunResultModes
from blade_bench.eval.datamodel.submission import DatasetSubmission


class MultiRunResults(BaseModel):
    dataset_name: str
    n: int
    analyses: Dict[int, Union[EntireAnalysis, Tuple[RunResultModes, str]]]

    def to_dataset_submission(self) -> DatasetSubmission:
        return DatasetSubmission(
            dataset_name=self.dataset_name,
            analyses=[
                a for a in self.analyses.values() if isinstance(a, EntireAnalysis)
            ],
        )
