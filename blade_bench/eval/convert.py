import ast
import json
import time
import astor
import os.path as osp

from typing import List, Literal, Tuple, Union

from blade_bench.data.dataset import DatasetInfo
from blade_bench.eval.datamodel import (
    ModelAndColumns,
    EntireAnalysis,
    EntireAnalysisProcessed,
    RunResultModes,
    EvalRunResults,
)
from blade_bench.eval.exceptions import (
    LMSubmissionExecutionError,
    LMSubmissionEmptyError,
    LMSubmissionExecutionError,
    RunError,
)
from blade_bench.eval.llm.code_to_model import CodeToModelLLM
from blade_bench.eval.llm.code_to_transforms import CodeToTransformsLLM

from blade_bench.llms.base import TextGenerator
from blade_bench.llms.datamodel.gen_config import GenConfig, LLMHistory
from blade_bench.llms.utils import cache_request
from blade_bench.nb.simple_code_executor import ExecutorReturn


from blade_bench.nb import TransformCodeExecutor, TransformObjExecutor
from blade_bench.data.datamodel import TransformDataReturn, TransformDatasetState
from blade_bench.data.annotation import AnnotationDataTransforms
from blade_bench.eval.llm import DebugCodeLM
from blade_bench.parse_code import extract_code_inside_functions_and_func_names
from blade_bench.utils import get_dataset_csv_path
from blade_bench.data.dataset import load_dataset_info

from blade_bench.logger import (
    CODE_ENV_QUERY,
    CODE_ENV_RESP,
    TS_STATE_QUERY,
    TS_STATE_RESP,
    logger,
)
from diskcache import Cache


class Convert:
    def __init__(
        self,
        run_dataset: str,
        llm_config: GenConfig = None,
        llm_history: LLMHistory = None,
        text_gen: TextGenerator = None,
        use_code_cache: bool = True,
        output_dir: str = ".",
        timeout: int = 10,
    ):
        self.data_path = get_dataset_csv_path(run_dataset)
        self.run_dataset = run_dataset
        self.use_code_cache = use_code_cache
        assert osp.exists(
            self.data_path
        ), f"Dataset path {self.data_path} does not exist"
        self.dinfo: DatasetInfo = load_dataset_info(run_dataset)
        self.__init_llms(llm_config, llm_history, text_gen)
        self.annotation = AnnotationDataTransforms(
            dataset_path=self.data_path,
            save_path=output_dir,
            run_nb=False,
            timeout=timeout,
        )
        self.transform_executor = TransformObjExecutor(
            data_path=self.data_path,
            save_path=output_dir,
            run_init_once=True,
        )
        self.code_executor = TransformCodeExecutor(
            data_path=self.data_path,
            save_path=output_dir,
            run_init_once=True,
        )

        self.cache_dir = osp.join(osp.dirname(osp.abspath(__file__)), "code_cache")
        self.cache = Cache(self.cache_dir, size_limit=2**38)

    def __init_llms(
        self, llm_config: GenConfig, llm_history: LLMHistory, text_gen: TextGenerator
    ):
        if llm_config is None and text_gen is None:
            raise ValueError("llm_config or text_gen must be provided")
        if text_gen is not None:
            self.code_to_transform_llm = CodeToTransformsLLM(
                text_gen, history=llm_history
            )
            self.code_to_model_llm = CodeToModelLLM(text_gen, history=llm_history)
            self.debug_llm = DebugCodeLM(text_gen, history=llm_history)

    async def convert_entire_analysis(
        self, entire_analysis: EntireAnalysis
    ) -> Tuple[EntireAnalysisProcessed, Union[None, EvalRunResults]]:
        try:
            state_data: Union[TransformDatasetState] = (
                await self.get_state_data_from_code(
                    entire_analysis.transform_code_inside_func
                )
            )
            run_results = None
        except RunError as e:
            state_data = None
            run_results = EvalRunResults(
                info=e.message,
                res_type=e.res_type,
            )

        model_code: ModelAndColumns = self.code_to_model_llm.code_to_model_obj(
            entire_analysis.model_code_inside_func
        )
        cvar_specs = entire_analysis.cvars.to_cvar_specs()
        model_spec = model_code.to_model_spec()

        analysis = EntireAnalysisProcessed(
            cv_specs=cvar_specs,
            m_specs={model_spec.spec_id: model_spec},
            transform_state=state_data,
            agent_cvars=entire_analysis.cvars,
            m_code_and_cols=model_code,
        )
        return analysis, run_results

    async def get_transform_state_with_cache(
        self, transform_ret: List[TransformDataReturn]
    ):
        inp = {
            "codes": [ts.code for ts in transform_ret],
            "tverbs": [ts.transform_verb for ts in transform_ret],
            "col_mappings": [
                {
                    str(tuple(sorted(key))): value
                    for key, value in ts.column_mapping.items()
                }
                for ts in transform_ret
            ],
            "groupby_cols": [
                str(tuple(sorted(ts.groupby_cols))) for ts in transform_ret
            ],
            "dataset": self.run_dataset,
        }

        logger.bind(message=json.dumps(inp, indent=2)).log(
            TS_STATE_QUERY, "Getting transform state"
        )
        if self.use_code_cache:
            start_time = time.time()
            state_data: TransformDatasetState = cache_request(self.cache, params=inp)
            if state_data:
                logger.bind(
                    from_cache=True,
                    cache_elapsed_time=time.time() - start_time,
                    message=json.dumps(state_data.summary_dict, indent=2),
                ).log(TS_STATE_RESP, "Finished getting transform state")
                return state_data
        start_time = time.time()
        state_data = await self.annotation.build_state_data_from_transform_return(
            transform_ret
        )
        elapsed_time = time.time() - start_time
        logger.bind(
            from_cache=False,
            api_elapsed_time=elapsed_time,
            message=json.dumps(state_data.summary_dict, indent=2),
        ).log(TS_STATE_RESP, "Finished getting transform state")
        cache_request(self.cache, params=inp, values=state_data)

        return state_data

    async def run_code_with_cache(
        self, code: str, executor: Literal["code_executor", "transform_executor"]
    ) -> ExecutorReturn:
        inp = {
            "executor": executor,
            "code": code,
            "dataset": self.run_dataset,
        }
        logger.bind(message=code).log(CODE_ENV_QUERY, f"Running code with {executor}")
        if self.use_code_cache:
            start_time = time.time()
            code_res: ExecutorReturn = cache_request(self.cache, params=inp)
            if code_res:
                logger.bind(
                    from_cache=True,
                    message=(
                        f"[Success]\n" + code_res.output
                        if code_res.success
                        else "[Error]\n" + code_res.output
                    ),
                    cache_elapsed_time=time.time() - start_time,
                ).log(CODE_ENV_RESP, f"Finished running code with {executor}")
                return code_res
        start_time = time.time()
        if executor == "code_executor":
            code_res = await self.code_executor.run_code(code)
        elif executor == "transform_executor":
            code_res = await self.transform_executor.run_code(code)

        elapsed_time = time.time() - start_time
        logger.bind(
            from_cache=False,
            api_elapsed_time=elapsed_time,
            message=(
                f"[Success]\n" + code_res.output
                if code_res.success
                else "[Error]\n" + code_res.output
            ),
        ).log(CODE_ENV_RESP, f"Finished running code with {executor}")
        cache_request(self.cache, params=inp, values=code_res)
        return code_res

    async def get_state_data_from_code(
        self, code, max_retries: int = 3
    ) -> Union[TransformDatasetState, EvalRunResults]:
        try:
            ast.parse(code)
        except Exception as e:
            raise LMSubmissionExecutionError(
                message=f"Code has syntax error, cannot parse with AST: {e}",
                res_type=RunResultModes.LM_SUBMISSION_EXECUTION_FAILED,
            )

        code_res: ExecutorReturn = await self.run_code_with_cache(code, "code_executor")
        if not code_res.success:
            raise LMSubmissionExecutionError(
                message=code_res.output,
                res_type=RunResultModes.LM_SUBMISSION_EXECUTION_FAILED,
            )

        if not astor.to_source(ast.parse(code)):
            raise LMSubmissionEmptyError(
                message="Code is empty after parsing",
                res_type=RunResultModes.LM_SUBMISSION_TRANSFORM_CODE_EMPTY,
            )

        retries = 0
        converted_code = self.code_to_transform_llm.convert_code(code)
        while retries < max_retries:
            code_res: ExecutorReturn = await self.run_code_with_cache(
                converted_code, "transform_executor"
            )
            if code_res.success:
                break
            else:
                retries += 1
                if retries < max_retries:
                    logger.warning(f"Code conversion failed, retrying")
                    background_code = self.transform_executor.init_code.format(
                        data_path="data.csv"
                    )
                    postfix_code = self.transform_executor.postfix_code.format(
                        save_path="save_path.pkl"
                    )
                    _, function_names = extract_code_inside_functions_and_func_names(
                        converted_code
                    )
                    code_and_reflection = self.debug_llm.debug_code(
                        background_code=background_code,
                        function_code=converted_code,
                        curr_code=postfix_code,
                        runtime_error=code_res.output,
                        func_names=function_names,
                    )
                    converted_code = code_and_reflection.code
                else:
                    logger.error(f"Code conversion failed after {max_retries} retries")
        if not code_res.success:
            raise LMSubmissionExecutionError(
                message=f"Code conversion failed after {max_retries} retries\n{code_res.output}",
                res_type=RunResultModes.LM_SUBMISSION_CONVERSION_FAILED,
            )

        transform_ret: List[TransformDataReturn] = code_res.value
        codes, _ = extract_code_inside_functions_and_func_names(converted_code)
        for i, ts in enumerate(transform_ret):
            ts.code = codes[i]
        if not transform_ret:
            raise LMSubmissionExecutionError(
                message="No transforms were returned",
                res_type=RunResultModes.LM_SUBMISSION_CONVERSION_FAILED,
            )

        try:
            llm_state_data = await self.get_transform_state_with_cache(transform_ret)
        except Exception as e:
            raise LMSubmissionExecutionError(
                message=f"Error getting transform state: {e}",
                res_type=RunResultModes.LM_SUBMISSION_CONVERSION_FAILED,
            )

        llm_state_data.converted_code = converted_code
        return llm_state_data
