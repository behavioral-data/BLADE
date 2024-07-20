import asyncio
import json
import os
from typing import Dict, Tuple, Union
import os.path as osp

from blade_bench.baselines.config import MultiRunConfig
from blade_bench.baselines.run import SingleRunExperiment
from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis
from blade_bench.eval.datamodel.run import RunResultModes
from blade_bench.eval.datamodel.multirun import MultiRunResults
from blade_bench.eval.exceptions import LMGenerationError
from blade_bench.utils import get_dataset_csv_path

from blade_bench.eval.utils import (
    SAVE_CODE_TEMPLATE,
)
from blade_bench.logger import logger


class RunLLMMultiRun(SingleRunExperiment):
    def __init__(self, config: MultiRunConfig):
        super().__init__(config)
        self.config = config

    def __save_results(self, results: MultiRunResults):
        os.makedirs(self.config.output_dir, exist_ok=True)
        for i, analysis in results.analyses.items():
            if isinstance(analysis, EntireAnalysis):
                save_path = osp.join(self.config.output_dir, f"llm_analysis_{i}.py")
                with open(save_path, "w") as f:
                    code = SAVE_CODE_TEMPLATE.format(
                        data_path=f"{get_dataset_csv_path(self.config.run_dataset)}",
                        transform_code=analysis.transform_code,
                        model_code=analysis.m_code,
                    )
                    f.write(code)
        with open(osp.join(self.config.output_dir, "multirun_analyses.json"), "w") as f:
            f.write(results.model_dump_json(indent=2))
        with open(osp.join(self.config.output_dir, "config.json"), "w") as f:
            f.write(self.config.model_dump_json(indent=2))

        with open(osp.join(self.config.output_dir, "propmts.json"), "w") as f:
            f.write(
                json.dumps(
                    {
                        "prompts": [
                            b.model_dump() for b in self.llm_history.prompt_history
                        ]
                    }
                )
            )

    async def run(self, save_results=True) -> MultiRunResults:
        results: Dict[int, Union[EntireAnalysis, Tuple[RunResultModes, str]]] = {}
        for i in range(self.config.num_runs):
            try:
                self.config.llm.texgt_gen.config.textgen_config.run_config = {
                    "run_num": i
                }
                analysis = await self.get_lm_analysis()
                results[i] = analysis
            except LMGenerationError as e:
                results[i] = (e.res_type, e.message)
        result = MultiRunResults(
            dataset_name=self.config.run_dataset,
            n=self.config.num_runs,
            analyses=results,
        )
        if save_results:
            self.__save_results(result)
        return result


def multirun_llm(config: MultiRunConfig):
    runner = RunLLMMultiRun(config)
    results = asyncio.run(runner.run(config.save_results))
    logger.success("Completed, everything is logged at: " + config.output_dir)
