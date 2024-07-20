import ast
import json
from langchain.output_parsers import PydanticOutputParser

from blade_bench.eval.datamodel import EntireAnalysis
from blade_bench.eval.llm.examples.fertility import FERTILITY_ANALYSIS, FERTILITY_DINFO
from blade_bench.llms import LLMBase
from blade_bench.data.dataset import DatasetInfo


SYSTEM_PROMPT = """You are an AI Data Analysis Assistant who is an expert at \
writing an end-to-end scientific analysis given a research question and a dataset. \
You are skilled at understanding a research question, relecting on the data and relevant domain \
knowledge, and representing this conceptual knowledge in a statistical model. \
Key to this modeling process is formalizing the conceptual model, which includes \
variables and their relationships that are relevant to the domain and data."""


INSTRUCTION_PROMPT = """<Instruction> 
Given the research question, dataset \
formulate the conceptual model and write an analysis including all necessary \
data transformations and a statistical model to answer the research question. 
</Instruction>

<Format Instructions>
You will return 3 things:
1. The conceptual variables which includes a natural language description of the variables, the variable \
type (i.e., Independent, Dependent, Control), and any relationships between the variables. Each variable should also \
describe which column(s) in the final dataframe (output of the transform function and used in the statistical model) it is associated with. \
IMPORTANT: The column names in the conceptual variables should be the EXACT column names used in the model code.
    
2. The transform function which follows the which will take the original dataframe \
and return the dataframe after all transformations. \
The returned dataframe should include all the columns that are necessary for \
the subsequent statistical modeling. \
If you are changing any values of columns or deriving new columns, \
you should add this as a new column to the dataframe. \
    
3. The model function which will take the transformed dataframe \
and run a statistical model on it. The model function should return the results of the model.

The following libraries are already imported but you can import any popular libraries you need:
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt

Here is the code template for the transform function:
```python
def transform(df: pd.DataFrame) -> pd.DataFrame:
    # Your code here
    return df
```
Here is the code template for the model function:
```python
def model(df: pd.DataFrame) -> Any:
    # Your code here
    return results
```

Please return the conceptual variables, the transform function, and the model function in the format specified below:
{format_instructions}
</Format Instructions>
"""

EXAMPLE = """<Example>
Research Question: {research_question_ex}
Dataset Schema: {dataset_schema_ex}
Result: {result_ex}
</Example>
"""

POST_FIX = """Research Question: {research_question}
Dataset Schema: {dataset_schema}
Result: """


class GenAnalysisLM(LLMBase):
    prompt_templates = {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION_PROMPT + POST_FIX,
        "instruction_example": INSTRUCTION_PROMPT + EXAMPLE + POST_FIX,
    }

    def gen_analysis_example(
        self,
        dinfo: DatasetInfo,
        use_data_desc=True,
    ) -> str:
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction_example"]},
        ]

        parser = PydanticOutputParser(pydantic_object=EntireAnalysis)

        data_desc = (
            json.dumps(dinfo.data_desc, indent=2)
            if use_data_desc
            else json.dumps(dinfo.data_desc_no_desc_no_semantic_type, indent=2)
        )

        data_desc_ex = (
            json.dumps(FERTILITY_DINFO.data_desc_no_semantic_type, indent=2)
            if use_data_desc
            else json.dumps(
                FERTILITY_DINFO.data_desc_no_desc_no_semantic_type, indent=2
            )
        )

        prompt_variables = {
            "research_question_ex": FERTILITY_DINFO.research_question,
            "dataset_schema_ex": data_desc_ex,
            "result_ex": FERTILITY_ANALYSIS.model_dump_json(indent=2),
            "research_question": dinfo.research_question,
            "dataset_schema": data_desc,
            "format_instructions": parser.get_format_instructions(),
        }

        response = self.generate(
            prompt_template,
            prompt_variables,
            tags=["gen_entire_analysis_1shot"],
            metadata={"prompt_name": "gen_entire_analysis_1shot"},
        )
        return response
