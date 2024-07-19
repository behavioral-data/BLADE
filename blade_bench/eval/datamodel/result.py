import math
import os
import os.path as osp
import itertools
import random
from typing import List, Optional
from pydantic import BaseModel, model_validator

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


class SubmissionRuns(BaseModel):
    all_submission_matched_transforms_value: List[List[str]]
    all_submission_matched_transforms_graph: List[List[str]]
    all_submission_matched_cvars: List[List[str]]
    all_submission_matched_models: List[List[str]]

    @model_validator(mode="after")
    def test_len_equal(cls, data):
        assert len(data.all_submission_matched_transforms_value) == len(
            data.all_submission_matched_cvars
        )
        assert len(data.all_submission_matched_transforms_value) == len(
            data.all_submission_matched_models
        )
        assert len(data.all_submission_matched_transforms_value) == len(
            data.all_submission_matched_transforms_graph
        )

    @property
    def n(self):
        return len(self.all_submission_matched_transforms_value)

    def __get_combinations(self, k, data, num_samples=1000):
        if k > self.n:
            k = self.n

        ntotal = math.comb(self.n, k)
        if ntotal > num_samples:
            combinations = self.__reservoir_sample_combinations(data, k, num_samples)
        else:
            combinations = list(itertools.combinations(data, k))
        combinations = [
            [item for items in combination for item in items]
            for combination in combinations
        ]
        return combinations

    @staticmethod
    def __reservoir_sample_combinations(iterable, r, sample_size):
        """
        Takes a random sample of combinations from an iterable using reservoir sampling.

        Parameters:
        iterable (iterable): The input iterable to generate combinations from.
        r (int): The length of each combination.
        sample_size (int): The number of combinations to sample.

        Returns:
        list: A list of sampled combinations.
        """
        iterator = itertools.combinations(iterable, r)
        reservoir = []

        # Fill the reservoir with the first sample_size elements
        for i, combination in enumerate(iterator):
            if i < sample_size:
                reservoir.append(combination)
            else:
                # Randomly replace elements in the reservoir with a decreasing probability
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = combination

        return reservoir

    def transforms_value_k(self, k, num_samples: int = 10000) -> List[List[str]]:
        return self.__get_combinations(
            k, self.all_submission_matched_transforms_value, num_samples=num_samples
        )

    def transforms_graph_k(self, k, num_samples: int = 10000) -> List[List[str]]:
        return self.__get_combinations(
            k, self.all_submission_matched_transforms_graph, num_samples=num_samples
        )

    def cvars_k(self, k, num_samples: int = 10000) -> List[List[str]]:
        return self.__get_combinations(
            k, self.all_submission_matched_cvars, num_samples=num_samples
        )

    def models_k(self, k, num_samples: int = 10000) -> List[List[str]]:
        return self.__get_combinations(
            k, self.all_submission_matched_models, num_samples=num_samples
        )

    @property
    def transforms_value(self):
        return [
            item.spec_id
            for l in self.all_submission_matched_transforms_value
            for item in l
        ]

    @property
    def transforms_graph(self):
        return [
            item.spec_id
            for l in self.all_submission_matched_transforms_graph
            for item in l
        ]

    @property
    def cvars(self):
        return [item for l in self.all_submission_matched_cvars for item in l]

    @property
    def models(self):
        return [item for l in self.all_submission_matched_models for item in l]
