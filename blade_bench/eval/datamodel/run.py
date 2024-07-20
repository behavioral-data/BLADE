from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

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
    status: RunResultModes
    matched_model: List[str] = Field(default_factory=list)
    matched_model_cvar: List[str] = Field(default_factory=list)
    matched_vspec: List[str] = Field(default_factory=list)
    matched_gspec: List[str] = Field(default_factory=list)
    matched_vspec2: List[str] = Field(default_factory=list)
    matched_gspec2: List[str] = Field(default_factory=list)
    matched_cvar: List[str] = Field(default_factory=list)
    matched_cvar2: List[str] = Field(default_factory=list)
    num_tspecs2: int
    num_cvars2: int
    analysis: Optional[EntireAnalysis] = None
    converted_code: Optional[str] = None

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

    def __getitem__(self, n: int):
        return SingleRunMetrics(
            status=self.status[n],
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
            analysis=self.analyses[n] if self.analyses else None,
            converted_code=self.converted_code[n] if self.converted_code else None,
        )

    def __len__(self):
        return len(self.status)

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
            "average_models_matched": self.average_models_matched,
            "average_models_matched_cvar": self.average_models_matched_cvar,
            "average_transforms_matched_vspecs": self.average_transforms_matched_vspecs,
            "average_transforms_matched_gspecs": self.average_transforms_matched_gspecs,
            "average_cvars_matched": self.average_cvars_matched,
            "average_coverage_models": self.average_coverage_models,
            "average_coverage_cvar_models": self.average_coverage_cvar_models,
            "average_coverage_vspecs": self.average_coverage_vspecs,
            "average_coverage_gspecs": self.average_coverage_gspecs,
            "average_coverage_cvars": self.average_coverage_cvars,
            "status": [str(RunResultModes(s)) for s in self.status],
        }

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
    def average_models_matched(self) -> float:
        return sum(self.num_match_model) / max(len(self.num_match_model), 1)

    @computed_field
    @property
    def average_models_matched_cvar(self) -> float:
        return sum(self.num_match_model_cvar) / max(len(self.num_match_model_cvar), 1)

    @computed_field
    @property
    def average_transforms_matched_vspecs(self) -> float:
        return sum(self.num_match_vspec2) / max(sum(self.num_tspecs2), 1)

    @computed_field
    @property
    def average_transforms_matched_gspecs(self) -> float:
        return sum(self.num_match_gspec2) / max(sum(self.num_tspecs2), 1)

    @computed_field
    @property
    def average_cvars_matched(self) -> float:
        return sum(self.num_match_cvar2) / max(sum(self.num_cvars2), 1)

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
