from pydantic import BaseModel, computed_field, model_validator
import os.path as osp
from blade_bench.llms import llm, TextGenerator, TextGenConfig
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


class BenchmarkConfig(BaseModel):
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
        return data
