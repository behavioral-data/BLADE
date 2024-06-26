import ast
import json
import os
import os.path as osp
import shutil
import sys
import traceback
from typing import Union

import astor
import pandas as pd
from hydra.types import RunMode
from langchain.schema.output_parser import OutputParserException
from langchain.output_parsers import PydanticOutputParser

from blade_bench.llms.datamodel import LLMHistory
from blade_bench.llms.llm import LLMBase
from blade_bench.logger import logger, formatter
from blade_bench.baselines.config import BenchmarkConfig
from blade_bench.baselines.convert import Convert
from blade_bench.data.annotation import AnnotationDBData, get_annotation_data_from_df
from blade_bench.data.datamodel.transforms import TransformDatasetState
from blade_bench.data.load_annotation import load_ground_truth
from blade_bench.eval.datamodel.match import MatchedAnnotations
from blade_bench.eval.match.match_submission import SubmissionMatch
from blade_bench.eval.metrics import get_metrics_from_match_obj
from .utils import (
    normalize_code_string,
    SAVE_CODE_TEMPLATE,
    SAVE_CONVERTED_CODE_TEMPLATE,
)
from blade_bench.baselines.datamodel import RunResultModes, RunResults
from blade_bench.eval.datamodel import EntireAnalysis, EntireAnalysisProcessed
from blade_bench.baselines.agent import ReActAgent
from blade_bench.baselines.lm import GenAnalysisLM

from blade_bench.data.dataset import get_dataset_info
from blade_bench.utils import (
    get_dataset_csv_path,
    get_dataset_dir,
    get_dataset_annotations_path,
)


class RunLLMAndEval:
    GND_TRUTH_FNAME = "gnd_truth.pkl"
    GEN_ANALYSIS_FNAME = "llm_analysis.json"
    GEN_ANALYSIS_PROCESSED_FNAME = "llm_analysis_processed.pkl"
    MATCHED_ANNOTATIONS_FNAME = "matched_annotations.pkl"
    MATCH_METRICS_FNAME = "match_metrics.json"

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        self.dinfo = get_dataset_info(config.run_dataset)
        self.convert = Convert(config)

        config.llm.log_file = osp.join(config.output_dir, "llm.log")
        config.llm_eval.log_file = osp.join(config.output_dir, "llm.log")
        self.llm_config = config.llm
        self.eval_llm_config = config.llm_eval
        self.llm_history = LLMHistory()
        self.eval_llm_history = LLMHistory()

        self.gen_analysis_lm = GenAnalysisLM.init_from_llm_config(
            self.llm_config, history=self.llm_history
        )
        self.format_lm = LLMBase.init_from_llm_config(
            self.eval_llm_config, history=self.eval_llm_history
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

        self.matcher: SubmissionMatch = SubmissionMatch(
            config.run_dataset, llm_config=self.eval_llm_config
        )
        self.run_result = None

    async def get_run_results(
        self, res_type: RunResultModes, info: str, is_error=True, eval_res=None
    ):
        if is_error:
            logger.error(info)
        else:
            logger.info(info)

        await self.convert.transform_executor.nb_executor.terminate()
        await self.convert.code_executor.nb_executor.terminate()
        if self.convert.annotation.nb_executor is not None:
            await self.convert.annotation.nb_executor.nb_executor.terminate()
        if self.agent is not None:
            await self.agent.code_executor.nb_executor.terminate()

        return RunResults(
            res_type=res_type,
            res_type2=self.run_result.res_type if self.run_result is not None else None,
            info=info,
            lm_token_usage_history=self.llm_history,
            eval_token_usage_history=self.eval_llm_history,
            agent_steps=self.agent.steps if self.agent is not None else 0,
            eval_res=eval_res,
        )

    async def __process_generated_analysis(
        self, llm: LLMBase, parser: PydanticOutputParser, response: str
    ):
        if response == "":
            return await self.get_run_results(
                RunResultModes.LM_GENERATION_FAILED,
                f"Empty response from agent",
            )
        try:
            resp: EntireAnalysis = llm.get_pydantic_obj_w_retires(
                parser, response, retries=3
            )
            if not resp:
                return await self.get_run_results(
                    RunResultModes.LM_GENERATION_FAILED,
                    f"No valid response given: {str(e)}",
                )
        except Exception as e:
            return await self.get_run_results(
                RunResultModes.LM_GENERATION_FAILED,
                f"Failed to parse response: {traceback.format_exc()}",
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
        try:
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
            if isinstance(analysis, RunResults):
                return analysis

        except Exception as e:
            return await self.get_run_results(
                RunResultModes.LM_GENERATION_FAILED,
                f"Failed to generate response: {traceback.format_exc()}",
            )
        save_path = osp.join(self.config.output_dir, self.GEN_ANALYSIS_FNAME)
        with open(save_path, "w") as f:
            f.write(analysis.model_dump_json())
        self.__save_lm_analysis(analysis)
        return analysis

    async def process_analysis(
        self, analysis: EntireAnalysis
    ) -> Union[EntireAnalysisProcessed, RunResults]:
        save_path = osp.join(self.config.output_dir, self.GEN_ANALYSIS_PROCESSED_FNAME)
        try:
            analysis_processed, run_result = await self.convert.convert_entire_analysis(
                analysis
            )
            if run_result is not None:
                logger.error(run_result.info)
                logger.info("Continuing with the next step, skipping transformations.")
                self.run_result = run_result
            analysis_processed.save(save_path)
        except Exception as e:
            return await self.get_run_results(
                RunResultModes.LM_SUBMISSION_CONVERSION_FAILED,
                f"Failed to convert submission: {traceback.format_exc()}",
            )
        python_path = osp.join(self.config.output_dir, "python_scripts")
        save_converted_path = osp.join(python_path, "transforms_converted.py")
        with open(save_converted_path, "w") as f:
            if isinstance(analysis_processed.transform_state, TransformDatasetState):
                code = SAVE_CONVERTED_CODE_TEMPLATE.format(
                    data_path=get_dataset_csv_path(self.config.run_dataset),
                    transform_code=analysis.transform_code,
                    converted_code=analysis_processed.transform_state.converted_code,
                )
                f.write(code)
        return analysis_processed

    async def load_ground_truth(self) -> Union[AnnotationDBData, RunResults]:
        try:
            gnd_truth = load_ground_truth(
                self.config.run_dataset, self.config.output_dir
            )
        except Exception as e:
            return await self.get_run_results(
                RunResultModes.LOAD_GND_TRUTH_FAILED,
                f"Failed to load ground truth: {traceback.format_exc()}",
            )
        return gnd_truth

    async def match_annotations(
        self, gnd_truth: AnnotationDBData, analysis_processed: EntireAnalysisProcessed
    ):
        try:
            matched_annotations: MatchedAnnotations = await self.matcher.match_all(
                gnd_truth, analysis_processed
            )
        except Exception as e:
            return await self.get_run_results(
                RunResultModes.MATCHING_FAILED,
                f"Failed to match submission: {traceback.format_exc()}",
            )
        save_path = osp.join(self.config.output_dir, self.MATCHED_ANNOTATIONS_FNAME)
        matched_annotations.save(save_path)
        return matched_annotations

    async def get_metrics(self, matched_annotations: MatchedAnnotations):
        try:
            match_metrics = get_metrics_from_match_obj(matched_annotations)
        except Exception as e:
            return await self.get_run_results(
                RunResultModes.GETTING_METRICS_FAILED,
                f"Failed to get match metrics: {traceback.format_exc()}",
            )
        save_path = osp.join(self.config.output_dir, self.MATCH_METRICS_FNAME)
        with open(save_path, "w") as f:
            f.write(match_metrics.model_dump_json())
        return match_metrics

    async def run(self):
        analysis = await self.get_lm_analysis()
        if isinstance(analysis, RunResults):
            return analysis
        analysis_processed = await self.process_analysis(analysis)
        if isinstance(analysis_processed, RunResults):
            return analysis_processed
        gnd_truth = await self.load_ground_truth()
        if isinstance(gnd_truth, RunResults):
            return gnd_truth
        matched_annotations = await self.match_annotations(
            gnd_truth, analysis_processed
        )
        if isinstance(matched_annotations, RunResults):
            return matched_annotations
        logger.success(f"Got matched annotations")

        match_metrics = await self.get_metrics(matched_annotations)
        if isinstance(match_metrics, RunResults):
            return match_metrics

        logger.success(f"Got match metrics")
        return await self.get_run_results(
            RunResultModes.FINISHED_SUCCESSFULLY,
            f"Completed successfully",
            is_error=False,
            eval_res=match_metrics,
        )


async def run_main(agent: RunLLMAndEval, run_mode: RunMode, run_name: str):
    res = await agent.run()
    with open(osp.join(agent.config.output_dir, "run_report.json"), "w") as f:
        f.write(res.model_dump_json())
    if res.res_type == RunResultModes.FINISHED_SUCCESSFULLY:
        logger.success("Completed, everything is logged at: " + agent.config.output_dir)
    else:
        logger.error(
            "Failed to complete, everything is logged at: " + agent.config.output_dir
        )
