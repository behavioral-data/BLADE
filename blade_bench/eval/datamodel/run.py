from enum import Enum
from typing import List, Optional, Set

from pydantic import BaseModel, ConfigDict

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


class MetricsAcrossRuns(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    all_models_match: Set[str]
    all_models_match_cvar: Set[str]
    all_transforms_match_vspecs: Set[str]
    all_transforms_match_gspecs: Set[str]
    all_transforms_match_vspecs2: Set[str]
    all_transforms_match_gspecs2: Set[str]
    all_cvars_match: Set[str]
    status: List[RunResultModes]
    # value for each run
    num_match_model: List[int]
    num_match_model_cvar: List[int]
    num_match_vspec: List[int]
    num_match_gspec: List[int]
    num_match_vspec2: List[int]
    num_match_gspec2: List[int]
    num_match_cvar: List[int]
    num_tspec2: List[int]
    num_cvars2: List[int]
    num_tspecs1: int = -1
    num_mspecs1: int = -1
    num_mspecs1_unique: int = -1
    num_cvars1: int = -1

    @property
    def coverage_num_models_matched(self):
        return len(self.all_models_match)

    @property
    def coverage_num_models_matched_cvar(self):
        return len(self.all_models_match_cvar)

    @property
    def coverage_num_transform_matched_vspecs(self):
        return len(self.all_transforms_match_vspecs)

    @property
    def coverage_num_transform_matched_gspecs(self):
        return len(self.all_transforms_match_gspecs)

    @property
    def coverage_num_cvars_matched(self):
        return len(self.all_cvars_match)

    @property
    def average_models_matched(self):
        return sum(self.num_match_model) / max(len(self.num_match_model), 1)

    @property
    def average_models_matched_cvar(self):
        return sum(self.num_match_model_cvar) / max(len(self.num_match_model_cvar), 1)

    @property
    def average_transforms_matched_vspecs(self):
        return sum(self.num_match_vspec) / max(sum(self.num_tspec2), 1)

    @property
    def average_transforms_matched_gspecs(self):
        return sum(self.num_match_gspec) / max(sum(self.num_tspec2), 1)

    @property
    def average_cvars_matched(self):
        return sum(self.num_match_cvar) / max(sum(self.num_cvars2), 1)


class EvalRunResults(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    info: str
    res_type: RunResultModes
    res_type_transform: Optional[RunResultModes] = None
    eval_lm_history: Optional[LLMHistory] = None
    eval_metrics: Optional[AllMetrics] = None


class RunResults(EvalRunResults):
    model_config = ConfigDict(use_enum_values=True)
    lm_history: Optional[LLMHistory] = None
    agent_steps: Optional[int] = 0

    def __str__(self):
        return f"""
INFO: {self.info}
RES_TYPE: {self.res_type}
"""
