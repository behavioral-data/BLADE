import json
from blade_bench.eval.datamodel.run_mcq import MCQMetrics
from blade_bench.logger import logger
from blade_bench.baselines.config import BenchmarkMCQConfig
from langchain.output_parsers import PydanticOutputParser


from blade_bench.baselines.lm.mcq import AnsMCQLM
from blade_bench.data.datamodel.mcq import MCQDatasetSimple, MCQResponse
from blade_bench.data.dataset import load_dataset_info
from blade_bench.llms.datamodel.gen_config import LLMHistory
from blade_bench.llms.llm import LLMBase
from blade_bench.utils import get_dataset_mcq_path


class RunMCQ:
    def __init__(self, config: BenchmarkMCQConfig):
        self.config = config
        self.llm_history = LLMHistory()
        self.format_lm = LLMBase(config.llm_eval.texgt_gen)
        self.mcq_llm = AnsMCQLM(config.llm.texgt_gen, history=self.llm_history)
        self.dinfo = load_dataset_info(config.run_dataset)
        mcq_path = get_dataset_mcq_path(config.run_dataset)
        self.dataset_mcq = MCQDatasetSimple(**json.load(open(mcq_path)))

    def process_mcq_response(self, response: str) -> MCQResponse:
        parser = PydanticOutputParser(pydantic_object=MCQResponse)
        try:
            mcq_response = self.format_lm.get_pydantic_obj_w_retires(
                parser=parser, response=response, retries=0
            )
            return mcq_response
        except Exception as e:

            def check_letters(input_string):
                # Convert the string to uppercase to make the check case-insensitive
                # Define the set of letters to check
                letters_to_check = {"A", "B", "C", "D"}

                # Check if any of the letters are in the input string
                for letter in letters_to_check:
                    if letter in input_string:
                        return MCQResponse(answer=letter, rationale="")

            return check_letters(response.split("\n")[0])

    def run(self):
        count_correct_cvars = 0
        total_cvars = len(self.dataset_mcq.mcqs_cvar)
        for mcq in self.dataset_mcq.mcqs_cvar:
            resp = self.mcq_llm.answer_multiple_choice(
                mcq,
                self.dinfo,
                use_data_desc=self.config.use_data_desc,
            )

            res = self.process_mcq_response(resp)
            if res is None:
                logger.warning(f"LM did not answer the question")
                continue
            logger.info(f"LM answered with {res.answer_index}")
            logger.info(
                f"The correct answer is {mcq.options.index(mcq.correct_answer)}"
            )
            if res.answer_index == mcq.options.index(mcq.correct_answer):
                count_correct_cvars += 1

        count_correct_transforms = 0
        total_transforms = self.dataset_mcq.num_transforms
        for cvar, mcqs in self.dataset_mcq.mcqs_transform.items():
            for mcq in mcqs:
                resp = self.mcq_llm.answer_multiple_choice(
                    mcq,
                    self.dinfo,
                    use_data_desc=self.config.use_data_desc,
                )
                res = self.process_mcq_response(resp)
                if res is None:
                    logger.warning(f"LM did not answer the question")
                    continue

                logger.info(f"LM answered with {res.answer_index}")
                logger.info(
                    f"The correct answer is {mcq.options.index(mcq.correct_answer)}"
                )
                if res.answer_index == mcq.options.index(mcq.correct_answer):
                    count_correct_transforms += 1

        result = MCQMetrics(
            num_correct_transforms=count_correct_transforms,
            num_correct_cvars=count_correct_cvars,
            num_total_transforms=total_transforms,
            num_total_cvars=total_cvars,
        )
        return result
