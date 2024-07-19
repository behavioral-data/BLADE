import json
from typing import Union


from langchain.output_parsers import PydanticOutputParser

from blade_bench.data.datamodel.mcq import (
    MCQResponse,
    MCQSimpleCvar,
    MCQSimpleTransform,
)
from blade_bench.data.dataset import DatasetInfo
from blade_bench.llms.llm import LLMBase


SYSTEM_PROMPT = """You are an AI Data Analysis Assistant who is an expert at \
writing an end-to-end scientific analysis given a research question and a dataset. \
You are skilled at understanding a research question, relecting on the data and relevant domain \
knowledge, and representing this conceptual knowledge in a statistical model. \
Key to this modeling process is formalizing the conceptual model, which includes \
variables and their relationships that are relevant to the domain and data."""

INSTRUCTION_PROMPT = """<Instruction>
{task_instruction}
In addition to the answer please also include a rationale.
Return your answer in the format specified below:
{format_instructions}
</Instruction> 

Research Question: {research_question}
Dataset: {dataset}

{mcq_choices}
The valid values are: {valid_values}
Answer: """


class AnsMCQLM(LLMBase):
    prompt_templates = {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION_PROMPT,
    }

    def answer_multiple_choice(
        self,
        mcq: Union[MCQSimpleTransform, MCQSimpleCvar],
        dinfo: DatasetInfo,
        use_data_desc=True,
    ) -> str:
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction"]},
        ]
        parser = PydanticOutputParser(pydantic_object=MCQResponse)

        data_desc = (
            json.dumps(dinfo.data_desc, indent=2)
            if use_data_desc
            else json.dumps(dinfo.data_desc_no_desc_no_semantic_type, indent=2)
        )
        valid_values = mcq.valid_values

        prompt_variables = {
            "research_question": dinfo.research_question,
            "dataset": data_desc,
            "task_instruction": mcq.task_instruction,
            "mcq_choices": mcq.choices,
            "format_instructions": parser.get_format_instructions(),
            "valid_values": ", ".join(valid_values),
        }

        response = self.generate(
            prompt_template,
            prompt_variables,
            tags=["ans_multiple_choice"],
            metadata={"prompt_name": "ans_multiple_choice"},
        )
        return response
