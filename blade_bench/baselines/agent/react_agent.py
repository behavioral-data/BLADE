import json
import asyncio
import time
from typing import Optional
import os.path as osp

from diskcache import Cache
from langchain.output_parsers import PydanticOutputParser
from langchain.tools import tool

from blade_bench.data.dataset import DatasetInfo
from blade_bench.llms.utils import cache_request
from blade_bench.llms.llm import LLMBase
from blade_bench.eval.datamodel import EntireAnalysis
from blade_bench.nb import SimpleCodeExecutor
from blade_bench.eval.llm.examples.fertility import FERTILITY_ANALYSIS, FERTILITY_DINFO
from blade_bench.baselines.agent.examples import FERTILITY_REACT_TRAJECTORY

from blade_bench.logger import CODE_ENV_QUERY, CODE_ENV_RESP, logger

import nest_asyncio

nest_asyncio.apply()

ACTION_TAG = "[Action]:"
OBSERVATION_TAG = "[Observation]:"
THOUGHT_TAG = "[Thought]:"
FINISH_TAG = "[Finish]:"


SYSTEM_PROMPT = """You are an AI Data Analysis Assistant who is an expert at \
writing an end-to-end scientific analysis given a research question and a dataset. \
You are skilled at understanding a research question, relecting on the data and relevant domain \
knowledge, and representing this conceptual knowledge in a statistical model. \
Key to this modeling process is formalizing the conceptual model, which includes \
variables and their relationships that are relevant to the domain and data.

Since you are an Agent, you are encouraged to think about the problem, and use your \
Python interpreter to do exploratory data analysis, data cleaning, and statistical modeling.\
"""


INSTRUCTION_PROMPT = f"""<Agent Behavior> 
You operate under the ReAct framework. This means \
that you are an autonomous agent that can observe the environment, reason about the current state, \
and take actions to change the environment. Your goal is to perform actions that will help you \
with your objectives. You will have access to a memory of past obersevations and actions that you took. \

There are five tags that you should be aware of:
1. {THOUGHT_TAG} This is where you reason about the current state of the environment and decide on the best course of action.
    <example>
    {THOUGHT_TAG} Let's start by examing the data and understanding the relationships between the variables.
    </example>
2. {ACTION_TAG}: This is where you perform an action that will change the environment. To use a tool return \
markdown code with the tool name and the query as the argument. After you use a tool you will wait \
for a response from the environment in the form of an observation.
    <example>
    {ACTION_TAG}
    ```python
    import pandas as pd
    df.head()
    ```
    </example>
3. {OBSERVATION_TAG} This is where information about your actions will be returned to you. The results will \
be returned in markdown code.
    <example>
    {OBSERVATION_TAG}
    ```
        WorkerID  Rel1  Rel2  Rel3  Sure1    
    0         1     6   5.0   6.0      9     
    1         2     1   2.0   1.0      4     
    2         3     7   8.0   8.0      5      
    3         4     2   1.0   1.0      8     
    4         5     5   5.0   5.0      5      
    ```
    </example>
4. {FINISH_TAG} When you generate this tag you should output your final answer. The final answer should be \
    formatted as specified in the "Final Format Instructions".
    

**IMPORTANT** Always respond in this format exactly: 
```
{THOUGHT_TAG} Your reasoning here.
{ACTION_TAG} The code you want to execute in the environment.
{OBSERVATION_TAG}
```

The {THOUGHT_TAG[:-1]}, {ACTION_TAG[:-1]}, {OBSERVATION_TAG[:-1]} tags can repeat N times \
until you generate the {FINISH_TAG[:-1]} tag with the final result. 

Once and ONLY when you are done with your analysis and want to output the final result, you should respond in this format:
```
{THOUGHT_TAG} Your reasoning here.
{FINISH_TAG} The final result here as specified in the "Final Format Instructions" below.
```
</Agent Behavior>

<Instruction> 
Given the research question, dataset \
formulate the conceptual model and write an analysis including all necessary \
data transformations and a statistical model to answer the research question. \
    
Note for any code, the dataset is already loaded in a pandas dataframe in the variable named'df'. 
</Instruction>
"""


FORMAT_INSTRUCTIONS = """<Final Format Instructions>
Your final goal is to return 3 things:
1. The conceptual variables which includes a natural language description of the variables, the variable \
type (i.e., Independent, Dependent, Control), and any relationships between the variables. Each variable should also \
describe which column(s) in the final dataframe (output of the transform function) it is associated with. \
    
2. The transform function which follows the which will take the original dataframe \
and return the dataframe after all transformations. \
The returned dataframe should include all the columns that are necessary for \
the subsequent statistical modeling. \
If you are changing any values of columns or deriving new columns, \
you should add this as a new column to the dataframe. \
    
3. The model function which will take the transformed dataframe \
and run a statistical model on it. The model function should return the results of the model.

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

Please return the conceptual variables, the transform function, and the model function in the format specified below.
IT IS VERY IMPORTANT THAT THIS SHOULD COME AFTER [Finish]:
{format_instructions}
</Final Format Instructions>
"""

EXAMPLE = """<Example with Multiple Steps>
Research Question: {research_question_ex}
Dataset Schema: {dataset_schema_ex}
{react_trajectory}
[Finish]:
{result_ex}
</Example with Multiple Steps>
"""

POST_FIX = """Research Question: {research_question}
Dataset Schema: {dataset_schema}
"""


@tool
def identity(query: str) -> str:
    """Just return the input string as is"""
    return query


class ReActAgent:
    def __init__(
        self,
        llm: LLMBase,
        dinfo: DatasetInfo,
        data_path: str,
        use_data_desc: bool = True,
        use_code_cache: bool = True,
    ):
        self.llm = llm
        self.use_data_desc = use_data_desc
        self.data_path = data_path
        self.code_executor = SimpleCodeExecutor(
            data_path=self.data_path, run_init_once=True
        )
        self.steps = 0
        self.memory = ""
        self.dinfo = dinfo
        self.parser = PydanticOutputParser(pydantic_object=EntireAnalysis)

        self.cache_dir = osp.join(osp.dirname(osp.abspath(__file__)), "code_cache")
        self.cache = Cache(self.cache_dir, size_limit=2**34)
        self.code_and_observation_history = ""

        @tool
        def python(query: str) -> str:
            """Evaluate the input string as a Python program"""
            inp = {
                "memory": self.code_and_observation_history,
                "query": query,
                "dataset": self.data_path,
            }
            logger.bind(message=query).log(CODE_ENV_QUERY, "")
            if use_code_cache:
                start_time = time.time()
                response = cache_request(cache=self.cache, params=inp)
                if response:
                    elapsed_time = time.time() - start_time
                    logger.bind(
                        from_cache=True,
                        cache_elapsed_time="{:.3f}ms".format(elapsed_time * 1000),
                        message=response,
                    ).log(CODE_ENV_RESP, "")
                    return response
            start_time = time.time()
            result = asyncio.run(self.code_executor.run_code(query))
            if result.value is not None:
                output = str(result.value)
            else:
                output = str(result.output)
            elapsed_time = time.time() - start_time
            logger.bind(
                from_cache=False,
                api_elapsed_time="{:.3f}s".format(elapsed_time),
                message=output,
            ).log(CODE_ENV_RESP, "")
            cache_request(cache=self.cache, params=inp, values=output)
            return output

        self.python_tool = python

    def reset(self):
        self.steps = 0
        self.memory = ""
        self.code_and_observation_history = ""
        asyncio.run(self.code_executor.reset_nb_executor())

    def get_prompt_and_vars(self, add_thought_tag: bool = True):
        # Combine memory and the instructions to create a prompt for the LLM
        data_desc = (
            json.dumps(self.dinfo.data_desc, indent=2)
            if self.use_data_desc
            else json.dumps(self.dinfo.data_desc_no_desc_no_semantic_type, indent=2)
        )

        example_desc = (
            json.dumps(FERTILITY_DINFO.data_desc, indent=2)
            if self.use_data_desc
            else json.dumps(
                FERTILITY_DINFO.data_desc_no_desc_no_semantic_type, indent=2
            )
        )
        prompts = [
            SYSTEM_PROMPT,
            INSTRUCTION_PROMPT,
            FORMAT_INSTRUCTIONS.format(
                format_instructions=self.parser.get_format_instructions()
            ),
            EXAMPLE.format(
                research_question_ex=FERTILITY_DINFO.research_question,
                dataset_schema_ex=example_desc,
                react_trajectory=FERTILITY_REACT_TRAJECTORY,
                result_ex=FERTILITY_ANALYSIS.model_dump_json(indent=2),
            ),
            POST_FIX.format(
                research_question=self.dinfo.research_question,
                dataset_schema=data_desc,
            ),
            self.memory,
        ]
        if add_thought_tag:
            prompts.append(THOUGHT_TAG)
        prompt = "\n".join(prompts)
        return prompt, {}

    def perform_action(self, response: str):
        # Simulate performing the action and updating the environment state
        lines = response.split("\n")
        line_w_action = 0
        for i, line in enumerate(lines):
            if ACTION_TAG in line or ACTION_TAG[:-1] in line:
                line_w_action = i

        tool_call = lines[line_w_action + 1 :]
        if tool_call:
            tool = tool_call[0]
        else:
            return "Tool not found"
        tool_args = "\n".join(tool_call[1:-1])
        if "python" in tool:

            tool_args = tool_args.split("```python")[-1].split("```")[0]
            resp = self.python_tool(tool_args)
            self.code_and_observation_history += (
                f"\n{ACTION_TAG}\n{tool_args}\n{OBSERVATION_TAG}\n{resp}"
            )
            return resp

        else:
            return "Tool not found"

    def take_step(self, add_thought_tag: bool = True):
        prompt, prompt_vars = self.get_prompt_and_vars(add_thought_tag)

        model_response = self.llm.generate(
            prompt,
            prompt_vars,
            tags=[f"ReAct Agent Step {self.steps + 1}"],
            stop_sequences=[OBSERVATION_TAG],
        )
        self.steps += 1
        return model_response.strip()

    def run(self, query: Optional[str] = "", max_turns: int = 10):
        self.reset()
        self.memory = query
        count_no_tag = 0
        while self.steps < max_turns:
            response = self.take_step(add_thought_tag=True)
            if THOUGHT_TAG in response:
                self.memory += "\n" + response
            else:
                self.memory += "\n" + THOUGHT_TAG + "\n" + response

            if FINISH_TAG in response:
                final_answer = response.split(FINISH_TAG)[-1]
                if final_answer.strip() != "":
                    return final_answer
                else:
                    return self.take_step(add_thought_tag=False)

            if FINISH_TAG[:-1] in response:
                final_answer = response.split(FINISH_TAG[:-1])[-1]
                if final_answer.strip() != "":
                    return final_answer
                else:
                    return self.take_step(add_thought_tag=False)

            if ACTION_TAG in response or ACTION_TAG[:-1] in response:
                count_no_tag = 0
                observation = self.perform_action(response)
                self.memory += (
                    "\n\n" + OBSERVATION_TAG + "\n" + f"```\n{observation}\n```\n"
                )
                continue

            # default to just taking the last response
            count_no_tag += 1
            if count_no_tag > 2:
                return response

        if self.steps >= max_turns:
            self.memory += "\nWe have reached the maximum number of steps. Please finish the analysis based on the final format instructions.\n[Finish]:\n"
            return self.take_step(add_thought_tag=False)
