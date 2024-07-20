import ast
import os
import os.path as osp

import traceback
from typing import Union

from langchain.output_parsers import PydanticOutputParser

from blade_bench.eval.exceptions import (
    LMGenerationError,
)
from blade_bench.llms.datamodel import LLMHistory
from blade_bench.llms.llm import LLMBase
from blade_bench.baselines.config import SingleRunConfig

from blade_bench.eval.utils import (
    normalize_code_string,
    SAVE_CODE_TEMPLATE,
)

from blade_bench.eval.datamodel import (
    EntireAnalysis,
    RunResultModes,
    RunResults,
)
from blade_bench.baselines.agent import ReActAgent
from blade_bench.baselines.lm import GenAnalysisLM

from blade_bench.data.dataset import load_dataset_info
from blade_bench.utils import (
    get_dataset_csv_path,
)


class SingleRunExperiment:
    GEN_ANALYSIS_FNAME = "llm_analysis.json"

    def __init__(self, config: SingleRunConfig):
        self.config = config
        self.dinfo = load_dataset_info(config.run_dataset)
        self.llm_history = LLMHistory()
        self.format_lm = LLMBase(config.llm_eval.texgt_gen)
        self.eval_text_gen = config.llm_eval.texgt_gen
        self.gen_analysis_lm = GenAnalysisLM(
            config.llm.texgt_gen, history=self.llm_history
        )
        if config.use_agent:
            self.agent = ReActAgent(
                self.gen_analysis_lm,
                dinfo=self.dinfo,
                data_path=get_dataset_csv_path(config.run_dataset),
                use_data_desc=config.use_data_desc,
                use_code_cache=config.use_code_cache,
            )
        else:
            self.agent = None

    async def __process_generated_analysis(
        self, llm: LLMBase, parser: PydanticOutputParser, response: str
    ):
        if response == "":
            raise LMGenerationError("Empty response from agent")
        try:
            resp: EntireAnalysis = llm.get_pydantic_obj_w_retires(
                parser, response, retries=3
            )
            if not resp:
                raise LMGenerationError(f"No valid response given: {response}")

        except Exception as e:
            raise LMGenerationError(
                f"Failed to parse response: {traceback.format_exc()}"
            )
        try:
            ast.parse(resp.transform_code)
        except Exception as e:
            resp.transform_code = normalize_code_string(resp.transform_code)
        try:
            ast.parse(resp.m_code)
        except Exception:
            resp.m_code = normalize_code_string(resp.m_code)
        return resp

    def save_lm_analysis(self, analysis: EntireAnalysis):
        python_path = osp.join(self.config.output_dir, "python_scripts")
        os.makedirs(python_path, exist_ok=True)
        save_transform_path = osp.join(python_path, "lm_analysis.py")
        with open(save_transform_path, "w") as f:
            code = SAVE_CODE_TEMPLATE.format(
                data_path=f"{get_dataset_csv_path(self.config.run_dataset)}",
                transform_code=analysis.transform_code,
                model_code=analysis.m_code,
            )
            f.write(code)

    async def get_lm_analysis(self) -> Union[EntireAnalysis, RunResults]:
        if self.config.use_agent:
            try:
                resp = self.agent.run()
            except Exception as e:
                raise LMGenerationError(
                    f"Failed to run agent: {traceback.format_exc()}"
                )
        else:
            resp = self.gen_analysis_lm.gen_analysis_example(
                self.dinfo, use_data_desc=self.config.use_data_desc
            )
        analysis = await self.__process_generated_analysis(
            self.format_lm,
            PydanticOutputParser(pydantic_object=EntireAnalysis),
            resp,
        )
        return analysis
