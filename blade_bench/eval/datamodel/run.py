from enum import Enum
import itertools
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis
from blade_bench.eval.datamodel.submission import DatasetSubmission
from blade_bench.eval.metrics.all_metrics import DiversityMetric
from blade_bench.llms.datamodel import LLMHistory
from blade_bench.eval.metrics import AllMetrics


class RunResultModes(Enum):
    LM_SUBMISSION_EXECUTION_FAILED = 0
    LM_SUBMISSION_CONVERSION_FAILED = 1
    LM_GENERATION_FAILED = 2
    LOAD_GND_TRUTH_FAILED = 3
    LM_SUBMISSION_TRANSFORM_CODE_EMPTY = 4
    FINISHED_SUCCESSFULLY = 5
    MATCHING_FAILED = 6
    GETTING_METRICS_FAILED = 7


class SingleRunMetrics(BaseModel):
    dataset_name: str = None
    matched_model: List[str] = Field(default_factory=list)
    matched_model_cvar: List[str] = Field(default_factory=list)
    matched_vspec: List[str] = Field(default_factory=list)
    matched_gspec: List[str] = Field(default_factory=list)
    matched_vspec2: List[str] = Field(default_factory=list)
    matched_gspec2: List[str] = Field(default_factory=list)
    matched_cvar: List[str] = Field(default_factory=list)
    matched_cvar2: List[str] = Field(default_factory=list)
    num_tspecs2: int = 0
    num_cvars2: int = 0
    num_tspecs1: int = 0
    num_mspecs1: int = 0
    num_mspecs1_unique: int = 0
    num_cvars1: int = 0

    @computed_field
    @property
    def num_match_model(self) -> int:
        return len(self.matched_model)

    @computed_field
    @property
    def num_match_model_cvar(self) -> int:
        return len(self.matched_model_cvar)

    @computed_field
    @property
    def num_match_vspec(self) -> int:
        return len(self.matched_vspec)

    @computed_field
    @property
    def num_match_gspec(self) -> int:
        return len(self.matched_gspec)

    @computed_field
    @property
    def num_match_vspec2(self) -> int:
        return len(self.matched_vspec2)

    @computed_field
    @property
    def num_match_gspec2(self) -> int:
        return len(self.matched_gspec2)

    @computed_field
    @property
    def num_match_cvar(self) -> int:
        return len(self.matched_cvar)

    @computed_field
    @property
    def hit_rate_models(self) -> float:
        return len(self.matched_model) / 1

    @computed_field
    @property
    def hit_rate_models_cvar(self) -> float:
        return len(self.matched_model_cvar) / 1

    @computed_field
    @property
    def hit_rate_vspecs(self) -> float:
        return len(self.matched_vspec2) / max(self.num_tspecs2, 1)

    @computed_field
    @property
    def hit_rate_gspecs(self) -> float:
        return len(self.matched_gspec2) / max(self.num_tspecs2, 1)

    @computed_field
    @property
    def hit_rate_cvars(self) -> float:
        return len(self.matched_cvar2) / max(self.num_cvars2, 1)


class MetricsAcrossRuns(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    matched_models: List[List[str]] = Field(default_factory=list)
    matched_models_cvar: List[List[str]] = Field(default_factory=list)
    matched_vspecs: List[List[str]] = Field(default_factory=list)
    matched_gspecs: List[List[str]] = Field(default_factory=list)
    matched_vspecs2: List[List[str]] = Field(default_factory=list)
    matched_gspecs2: List[List[str]] = Field(default_factory=list)
    matched_cvars: List[List[str]] = Field(default_factory=list)
    matched_cvars2: List[List[str]] = Field(default_factory=list)
    num_tspecs2: List[int] = Field(default_factory=list)
    num_cvars2: List[int] = Field(default_factory=list)
    status: List[RunResultModes] = Field(default_factory=list)
    num_tspecs1: int = -1
    num_mspecs1: int = -1
    num_mspecs1_unique: int = -1
    num_cvars1: int = -1
    analyses: Optional[List[EntireAnalysis]] = Field(default_factory=list)
    converted_code: Optional[List[str]] = Field(default_factory=list)
    diversity_metrics: Optional[List[DiversityMetric]] = Field(default_factory=list)

    @computed_field
    @property
    def total_unique_match_model(self) -> int:
        return len(self.all_models_match)

    @computed_field
    @property
    def total_unique_match_model_cvar(self) -> int:
        return len(self.all_models_match_cvar)

    @computed_field
    @property
    def total_unique_match_vspec(self) -> int:
        return len(self.all_transforms_match_vspecs)

    @computed_field
    @property
    def total_unique_match_gspec(self) -> int:
        return len(self.all_transforms_match_gspecs)

    @computed_field
    @property
    def total_unique_match_cvar(self) -> int:
        return len(self.all_cvars_match)

    @computed_field
    @property
    def avg_num_tspecs2(self) -> float:
        arr = [x for x in self.num_tspecs2 if x > 0]
        if len(arr) == 0:
            return np.nan
        return sum(arr) / len(arr)

    @computed_field
    @property
    def avg_num_cvars2(self) -> float:
        arr = [x for x in self.num_cvars2 if x > 0]
        if len(arr) == 0:
            return np.nan
        return sum(arr) / len(arr)

    @computed_field
    @property
    def all_models_match(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_models])

    @computed_field
    @property
    def all_models_match_cvar(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_models_cvar])

    @computed_field
    @property
    def all_transforms_match_vspecs(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_vspecs])

    @computed_field
    @property
    def all_transforms_match_gspecs(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_gspecs])

    @computed_field
    @property
    def all_transforms_match_vspecs2(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_vspecs2])

    @computed_field
    @property
    def all_transforms_match_gspecs2(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_gspecs2])

    @computed_field
    @property
    def all_cvars_match(self) -> Set[str]:
        return set.union(*[set(m) for m in self.matched_cvars])

    @computed_field
    @property
    def num_match_model(self) -> List[int]:
        return [len(m) for m in self.matched_models]

    @computed_field
    @property
    def num_match_model_cvar(self) -> List[int]:
        return [len(m) for m in self.matched_models_cvar]

    @computed_field
    @property
    def num_match_vspec(self) -> List[int]:
        return [len(m) for m in self.matched_vspecs]

    @computed_field
    @property
    def num_match_gspec(self) -> List[int]:
        return [len(m) for m in self.matched_gspecs]

    @computed_field
    @property
    def num_match_vspec2(self) -> List[int]:
        return [len(m) for m in self.matched_vspecs2]

    @computed_field
    @property
    def num_match_gspec2(self) -> List[int]:
        return [len(m) for m in self.matched_gspecs2]

    @computed_field
    @property
    def num_match_cvar(self) -> List[int]:
        return [len(m) for m in self.matched_cvars]

    @computed_field
    @property
    def num_match_cvar2(self) -> List[int]:
        return [len(m) for m in self.matched_cvars2]

    @computed_field
    @property
    def y_pred_cvars(self) -> List[int]:
        a = [
            [1] * len(m) + [0] * (self.num_cvars2[i] - len(m))
            for i, m in enumerate(self.matched_cvars)
        ]
        return [item for sublist in a for item in sublist]

    @computed_field
    @property
    def y_pred_models(self) -> List[int]:
        return [len(m) for i, m in enumerate(self.matched_models)]

    @computed_field
    @property
    def y_pred_models_cvar(self) -> List[int]:
        return [len(m) for i, m in enumerate(self.matched_models_cvar)]

    @computed_field
    @property
    def y_pred_vspecs(self) -> List[int]:
        a = [
            [1] * len(m) + [0] * (self.num_tspecs2[i] - len(m))
            for i, m in enumerate(self.matched_vspecs)
        ]
        return [item for sublist in a for item in sublist]

    @computed_field
    @property
    def y_pred_gspecs(self) -> List[int]:
        a = [
            [1] * len(m) + [0] * (self.num_tspecs2[i] - len(m))
            for i, m in enumerate(self.matched_gspecs)
        ]
        return [item for sublist in a for item in sublist]

    def __getitem__(self, n: int):
        return SingleRunMetrics(
            matched_model=self.matched_models[n],
            matched_model_cvar=self.matched_models_cvar[n],
            matched_vspec=self.matched_vspecs[n],
            matched_gspec=self.matched_gspecs[n],
            matched_vspec2=self.matched_vspecs2[n],
            matched_gspec2=self.matched_gspecs2[n],
            matched_cvar=self.matched_cvars[n],
            matched_cvar2=self.matched_cvars2[n],
            num_tspecs2=self.num_tspecs2[n],
            num_cvars2=self.num_cvars2[n],
            num_tspecs1=self.num_tspecs1,
            num_mspecs1=self.num_mspecs1,
            num_mspecs1_unique=self.num_mspecs1_unique,
            num_cvars1=self.num_cvars1,
        )

    def __len__(self):
        return len(self.num_cvars2)

    def get_single_run_metrics(self) -> List[SingleRunMetrics]:
        ret = [self[i] for i in range(len(self))]
        empty = [
            SingleRunMetrics(
                num_tspecs1=self.num_tspecs1,
                num_cvars1=self.num_cvars1,
                num_mspecs1=self.num_mspecs1,
                num_mspecs1_unique=self.num_mspecs1_unique,
            )
            for _ in range(self.count_generation_failed)
        ]
        return ret + empty

    def get_nth_analysis(self, n: int):
        # TODO
        return self.analyses[n], self.converted_code[n]

    def get_metrics(self):
        return {
            "coverage_num_models_matched": self.coverage_num_models_matched,
            "coverage_num_models_matched_cvar": self.coverage_num_models_matched_cvar,
            "coverage_num_transform_matched_vspecs": self.coverage_num_transform_matched_vspecs,
            "coverage_num_transform_matched_gspecs": self.coverage_num_transform_matched_gspecs,
            "coverage_num_cvars_matched": self.coverage_num_cvars_matched,
            "hit_rate_models_matched": self.hit_rate_models_matched,
            "hit_rate_models_matched_cvar": self.hit_rate_models_matched_cvar,
            "hit_rate_transform_vspecs": self.hit_rate_transforms_matched_vspecs,
            "hit_rate_transforms_matched_gspecs": self.hit_rate_transforms_matched_gspecs,
            "hit_rate_cvars_matched": self.hit_rate_cvars_matched,
            "average_coverage_models": self.average_coverage_models,
            "average_coverage_cvar_models": self.average_coverage_cvar_models,
            "average_coverage_vspecs": self.average_coverage_vspecs,
            "average_coverage_gspecs": self.average_coverage_gspecs,
            "average_coverage_cvars": self.average_coverage_cvars,
            "status": [str(RunResultModes(s)) for s in self.status],
            "matched_models": self.matched_models,
            "matched_models_cvar": self.matched_models_cvar,
            "matched_cvars": self.matched_cvars,
            "matched_vspecs": self.matched_vspecs,
            "matched_gspecs": self.matched_gspecs,
            "avg_num_tspecs2": self.avg_num_tspecs2,
            "avg_num_cvars2": self.avg_num_cvars2,
            "total_unique_match_model": self.total_unique_match_model,
            "total_unique_match_model_cvar": self.total_unique_match_model_cvar,
            "total_unique_match_vspec": self.total_unique_match_vspec,
            "total_unique_match_gspec": self.total_unique_match_gspec,
            "total_unique_match_cvar": self.total_unique_match_cvar,
        }

    @property
    def count_generation_failed(self) -> int:
        return sum(
            [
                s
                in [
                    RunResultModes.LM_GENERATION_FAILED.value,
                ]
                for s in self.status
            ]
        )

    @computed_field
    @property
    def coverage_num_models_matched(self) -> float:
        return len(self.all_models_match)

    @computed_field
    @property
    def coverage_num_models_matched_cvar(self) -> float:
        return len(self.all_models_match_cvar)

    @computed_field
    @property
    def coverage_num_transform_matched_vspecs(self) -> float:
        return len(self.all_transforms_match_vspecs)

    @computed_field
    @property
    def coverage_num_transform_matched_gspecs(self) -> float:
        return len(self.all_transforms_match_gspecs)

    @computed_field
    @property
    def coverage_num_cvars_matched(self) -> float:
        return len(self.all_cvars_match)

    @computed_field
    @property
    def hit_rate_models_matched(self) -> float:
        return sum(self.num_match_model) / max(len(self.num_match_model), 1)

    @computed_field
    @property
    def hit_rate_models_matched_cvar(self) -> float:
        return sum(self.num_match_model_cvar) / max(len(self.num_match_model_cvar), 1)

    @computed_field
    @property
    def hit_rate_transforms_matched_vspecs(self) -> float:
        return sum(self.num_match_vspec2) / max(sum(self.num_tspecs2), 1)

    @computed_field
    @property
    def hit_rate_transforms_matched_gspecs(self) -> float:
        return sum(self.num_match_gspec2) / max(sum(self.num_tspecs2), 1)

    @computed_field
    @property
    def hit_rate_cvars_matched(self) -> float:
        return sum(self.num_match_cvar2) / max(sum(self.num_cvars2), 1)

    def __sample_until_length(self, items: List[Any], length: int) -> List[Any]:
        if len(items) >= length:
            return items

        new_items = (
            random.sample(items, length - len(items))
            if length - len(items) < len(items)
            else random.choices(items, k=length - len(items))
        )
        return items + new_items

    def __get_combinations(self, k, data, num_samples=1000):
        if k > len(data):
            raise ValueError("k must be less than or equal to the length of the data")

        ntotal = math.comb(len(data), k)
        if ntotal > num_samples:
            combinations = self.__direct_sample_combinations(data, k, num_samples)
        else:
            combinations = list(itertools.combinations(data, k))
        return combinations

    @staticmethod
    def __direct_sample_combinations(iterable, r, sample_size):
        """
        Takes a random sample of combinations from an iterable using direct sampling.

        Parameters:
        iterable (iterable): The input iterable to generate combinations from.
        r (int): The length of each combination.
        sample_size (int): The number of combinations to sample.

        Returns:
        list: A list of sampled combinations.
        """
        # Convert the iterable to a list
        iterable = list(iterable)
        n = len(iterable)

        sampled_combinations_inds = set()
        sampled_combinations = []
        while len(sampled_combinations) < sample_size:
            # Sample a random combination
            indices = np.random.choice(n, r, replace=False)
            if frozenset(indices) in sampled_combinations_inds:
                continue
            sampled_combinations_inds.add(frozenset(indices))
            combination = [iterable[i] for i in indices]
            sampled_combinations.append(combination)

        return sampled_combinations

    def __calcualte_coverage(
        self,
        items: List[List[str]],
        k: int,
        denominator: int,
        use_combinations=False,
        num_samples=100,
        include_gen_errors=False,
    ) -> float:
        if include_gen_errors:
            items = items + [[] for _ in range(self.count_generation_failed)]

        if use_combinations:
            lists = self.__get_combinations(k, items, num_samples=num_samples)
        else:
            assert (
                k <= len(self.status) and len(self.status) % k == 0
            ), f"k must be a factor of the number of runs, k={k}, len(status)={len(self.status)}"
            matched_specs = self.__sample_until_length(items, len(self.status))
            lists = [matched_specs[i : i + k] for i in range(0, len(matched_specs), k)]
        return [len(set.union(*[set(m) for m in l])) / denominator for l in lists]

    def average_coverage_models_k(self, k: int, **kwargs) -> List[float]:
        return self.__calcualte_coverage(
            self.matched_models, k, min(self.num_mspecs1_unique, k), **kwargs
        )

    def average_coverage_cvar_models_k(self, k: int, **kwargs) -> List[float]:
        return self.__calcualte_coverage(
            self.matched_models_cvar, k, min(self.num_mspecs1, k), **kwargs
        )

    def average_coverage_vspecs_k(self, k: int, **kwargs) -> List[float]:
        return self.__calcualte_coverage(
            self.matched_vspecs, k, self.num_tspecs1, **kwargs
        )

    def average_coverage_gspecs_k(self, k: int, **kwargs) -> List[float]:
        return self.__calcualte_coverage(
            self.matched_gspecs, k, self.num_tspecs1, **kwargs
        )

    def average_coverage_cvars_k(self, k: int, **kwargs) -> List[float]:
        return self.__calcualte_coverage(
            self.matched_cvars, k, self.num_cvars1, **kwargs
        )

    @computed_field
    @property
    def average_coverage_models(self) -> float:
        if self.num_mspecs1_unique == -1:
            return -1
        return len(self.all_models_match) / self.num_mspecs1_unique

    @computed_field
    @property
    def average_coverage_cvar_models(self) -> float:
        if self.num_mspecs1 == -1:
            return -1
        return len(self.all_models_match_cvar) / self.num_mspecs1

    @computed_field
    @property
    def average_coverage_vspecs(self) -> float:
        if self.num_tspecs1 == -1:
            return -1
        return len(self.all_transforms_match_vspecs) / self.num_tspecs1

    @computed_field
    @property
    def average_coverage_gspecs(self) -> float:
        if self.num_tspecs1 == -1:
            return -1
        return len(self.all_transforms_match_gspecs) / self.num_tspecs1

    @computed_field
    @property
    def average_coverage_cvars(self) -> float:
        if self.num_cvars1 == -1:
            return -1
        return len(self.all_cvars_match) / self.num_cvars1


class EvalRunResults(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    info: str
    res_type: RunResultModes
    info_transform: Optional[str] = None
    res_type_transform: Optional[RunResultModes] = None
    eval_lm_history: Optional[LLMHistory] = None
    eval_metrics: Optional[AllMetrics] = None

    @property
    def res_type_combined(self) -> str:
        if self.res_type_transform is None:
            return str(RunResultModes(self.res_type))
        else:
            return str(RunResultModes(self.res_type_transform))


class RunResults(EvalRunResults):
    model_config = ConfigDict(use_enum_values=True)
    lm_history: Optional[LLMHistory] = None
    agent_steps: Optional[int] = 0

    def __str__(self):
        return f"""
INFO: {self.info}
RES_TYPE: {self.res_type}
"""
