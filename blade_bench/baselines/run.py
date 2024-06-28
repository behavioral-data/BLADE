import ast
import os
import os.path as osp

import traceback
from typing import Union

from hydra.types import RunMode
from langchain.output_parsers import PydanticOutputParser

from blade_bench.eval.datamodel.result import EvalResult
from blade_bench.eval.datamodel.submission import DatasetSubmission
from blade_bench.eval.evaluator import Evaluator
from blade_bench.eval.exceptions import (
    LMGenerationError,
)
from blade_bench.llms.config import get_text_gen
from blade_bench.llms.datamodel import LLMHistory
from blade_bench.llms.llm import LLMBase
from blade_bench.logger import logger, formatter
from blade_bench.baselines.config import BenchmarkConfig

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

from blade_bench.data.dataset import get_dataset_info
from blade_bench.utils import (
    get_dataset_csv_path,
)


class RunLLMAndEval:
    GND_TRUTH_FNAME = "gnd_truth.pkl"
    GEN_ANALYSIS_FNAME = "llm_analysis.json"
    GEN_ANALYSIS_PROCESSED_FNAME = "llm_analysis_processed.pkl"
    MATCHED_ANNOTATIONS_FNAME = "matched_annotations.pkl"
    MATCH_METRICS_FNAME = "match_metrics.json"
    EVAL_RESULTS_FNAME = "eval_results.json"

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        self.dinfo = get_dataset_info(config.run_dataset)

        config.llm.log_file = osp.join(config.output_dir, "llm.log")
        config.llm_eval.log_file = osp.join(config.output_dir, "llm.log")
        self.llm_config = config.llm
        self.llm_history = LLMHistory()
        self.eval_text_gen = get_text_gen(config.llm_eval)
        self.gen_analysis_lm = GenAnalysisLM.init_from_llm_config(
            self.llm_config, history=self.llm_history
        )
        self.format_lm = LLMBase(self.eval_text_gen)
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

    def __save_lm_analysis(self, analysis: EntireAnalysis):
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
            resp = self.agent.run()
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

    async def run(self):
        try:
            analysis = await self.get_lm_analysis()
            self.__save_lm_analysis(analysis)
            evaluator = Evaluator(
                submission=DatasetSubmission(
                    dataset_name=self.config.run_dataset, analyses=[analysis]
                ),
                text_gen=self.eval_text_gen,
                use_code_cache=self.config.use_code_cache,
                output_dir=self.config.output_dir,
            )
            eval_result: EvalResult = await evaluator.run_eval(analysis)
        except LMGenerationError as e:
            logger.error(f"Error: {e.message}")

            return RunResults(
                e.res_type,
                e.message,
                is_error=True,
                lm_history=self.llm_history,
                agent_steps=self.agent.steps if self.agent is not None else 0,
            )
        finally:
            if self.agent is not None:
                await self.agent.code_executor.nb_executor.terminate()

        path = osp.join(self.config.output_dir, self.GEN_ANALYSIS_PROCESSED_FNAME)
        eval_result.save_analysis_processed(path)

        path = osp.join(self.config.output_dir, self.MATCHED_ANNOTATIONS_FNAME)
        eval_result.save_matched_annotations(path)

        python_path = osp.join(self.config.output_dir, "python_scripts")
        eval_result.save_converted_code(python_path)

        path = osp.join(self.config.output_dir, self.MATCH_METRICS_FNAME)
        eval_result.save_metrics(path)

        with open(osp.join(self.config.output_dir, self.EVAL_RESULTS_FNAME), "w") as f:
            f.write(eval_result.model_dump_json(indent=2))

        return RunResults(
            res_type=eval_result.eval_run_result.res_type,
            res_type_transform=eval_result.eval_run_result.res_type_transform,
            info=eval_result.eval_run_result.info,
            eval_lm_history=eval_result.eval_run_result.eval_lm_history,
            lm_history=self.llm_history,
            agent_steps=self.agent.steps if self.agent is not None else 0,
        )


async def run_main(agent: RunLLMAndEval, run_mode: RunMode, run_name: str):
    res: RunResults = await agent.run()
    with open(osp.join(agent.config.output_dir, "run_report.json"), "w") as f:
        f.write(res.model_dump_json())
    if res.res_type == RunResultModes.FINISHED_SUCCESSFULLY.value:
        logger.success("Completed, everything is logged at: " + agent.config.output_dir)
    else:
        logger.error(
            "Failed to complete, everything is logged at: " + agent.config.output_dir
        )
