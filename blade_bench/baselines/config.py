import os
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator
import os.path as osp
import time
from blade_bench.llms.config import llm
from blade_bench.llms.base import TextGenerator
from blade_bench.llms.datamodel import TextGenConfig
from functools import cached_property


class LLMConfig(BaseModel):
    provider: str
    model: str
    textgen_config: TextGenConfig = None
    log_file: str = None
    use_cache: bool = True

    @cached_property
    def texgt_gen(self) -> TextGenerator:
        return llm(
            provider=self.provider,
            model=self.model,
            textgen_config=self.textgen_config,
            log_file=self.log_file,
            use_cache=self.use_cache,
        )


class SingleRunConfig(BaseModel):
    llm: LLMConfig
    llm_eval: LLMConfig
    output_dir: str
    run_dataset: str
    use_agent: bool = False
    use_data_desc: bool = False
    use_code_cache: bool = True

    @model_validator(mode="before")
    def add_log_file(cls, data):
        data["llm"] = dict(data["llm"])
        data["llm_eval"] = dict(data["llm_eval"])
        data["llm"]["log_file"] = osp.join(data["output_dir"], "llm.log")
        data["llm_eval"]["log_file"] = osp.join(data["output_dir"], "llm.log")

        # remove old file
        if osp.exists(data["llm"]["log_file"]):
            os.remove(data["llm"]["log_file"])
        if osp.exists(data["llm_eval"]["log_file"]):
            os.remove(data["llm_eval"]["log_file"])
        return data


class EvalConfig(BaseModel):
    glob_str: Optional[str] = None
    multirun_load_path: Optional[str] = None
    llm_eval: LLMConfig
    output_dir: str
    run_dataset: Optional[str] = None
    use_code_cache: bool = True
    diversity_ks: Optional[List[int]] = Field(default_factory=list)
    diversity_n_samples: int = 10000

    @model_validator(mode="before")
    def glob_or_save_path_is_not_none(cls, data):
        if data.get("glob_str") is None and data.get("multirun_load_path") is None:
            raise ValueError("Either glob_str or multirun_load_path must be provided")
        return data


class MultiRunConfig(SingleRunConfig):
    num_runs: int = 10
    save_results: bool = True


class BenchmarkMCQConfig(BaseModel):
    llm: LLMConfig
    llm_eval: LLMConfig
    run_dataset: str
    use_data_desc: bool = False
