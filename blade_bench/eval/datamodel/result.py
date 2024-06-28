import os
import os.path as osp
from typing import List, Optional
from pydantic import BaseModel

from blade_bench.data.datamodel.transforms import TransformDatasetState
from blade_bench.eval.datamodel.lm_analysis import (
    EntireAnalysis,
    EntireAnalysisProcessed,
)
from blade_bench.eval.datamodel.match import MatchedAnnotations
from blade_bench.eval.datamodel.run import EvalRunResults
from blade_bench.eval.metrics.all_metrics import AllMetrics
from blade_bench.eval.utils import SAVE_CONVERTED_CODE_TEMPLATE
from blade_bench.utils import get_dataset_csv_path


class EvalResult(BaseModel):
    dataset_name: str
    analysis: EntireAnalysis
    eval_run_result: EvalRunResults
    analysis_processed: Optional[EntireAnalysisProcessed] = None
    matched_annotations: Optional[MatchedAnnotations] = None
    metrics: Optional[AllMetrics] = None

    def save_analysis_processed(self, path, save_pkl=True):
        if self.analysis_processed is not None:
            if save_pkl:
                self.analysis_processed.save(path)
            else:
                with open(path, "w") as f:
                    f.write(self.analysis_processed.model_dump_json(indent=2))

    def save_converted_code(
        self,
        python_path,
    ):
        if self.analysis_processed is not None:
            os.makedirs(python_path, exist_ok=True)
            save_converted_path = osp.join(python_path, "transforms_converted.py")
            with open(save_converted_path, "w") as f:
                if isinstance(
                    self.analysis_processed.transform_state, TransformDatasetState
                ):
                    code = SAVE_CONVERTED_CODE_TEMPLATE.format(
                        data_path=get_dataset_csv_path(self.dataset_name),
                        transform_code=self.analysis.transform_code,
                        converted_code=self.analysis_processed.transform_state.converted_code,
                    )
                    f.write(code)

    def save_matched_annotations(self, path, save_pkl=True):
        if self.matched_annotations is not None:
            if save_pkl:
                self.matched_annotations.save(path)
            else:
                with open(path, "w") as f:
                    f.write(self.matched_annotations.model_dump_json(indent=2))

    def save_metrics(self, path):
        if self.metrics is not None:
            with open(path, "w") as f:
                f.write(self.metrics.model_dump_json())


class EvalResults(BaseModel):
    dataset_name: str
    results: List[EvalResult]
