import json
from typing import Dict

from blade_bench.llms import LLMBase
from blade_bench.nb import SimpleCodeExecutor, ExecutorReturn
from blade_bench.eval.datamodel import CodeAndReflection
from blade_bench.eval.llm.examples import (
    BACKGROUND_CODE_EX,
    CURR_CODE_EX,
    FERTILITY_DATA_EX,
    ERROR_MSG,
    REFLECTION,
    THOUGHT,
    RESULT,
)

SYSTEM_PROMPT = "You are an AI Python Data Science assistant. You will be given a previous implementation code, runtime error results, and the task."

DEBUG_CODE_PROMPT = """<Instruction> Given the original dataframe schema, the background code and function definition code that is \
already executed and the current code that raised an error, \
change the current code to fix the error. \
ONLY update the code inside Current Code. \
If we need to import any packages, include it as well. 
    
Respond in this format exactly INCLUDING the Result:
Reflection: What is the error? Why did it happen?
Thought: What is the best way to fix it without changing the original intention of the code? 
Result: Write the improved code for **Current Code** ONLY. The Background Code and Postfix Code will remain unchanged.
</Instruction>
{example}
Dataframe: {dataframe_schema}
Background Code: ```python 
{background_code} 
```
Current Code: ```python
{curr_code}
```
Error: {error}
Reflection: """


EXAMPLE = """<Example>
Dataframe: {dataframe_schema_ex}
Background Code: ```python
{background_code_ex}
```
Current Code: ```python
{curr_code_ex}
```
Error: {error_ex}
Reflection: {reflection_ex}
Thought: {thought_ex}
Result: {result_ex}
</Example>
"""


class DebugCodeLM(LLMBase):
    prompt_templates = {
        "debug_code": DEBUG_CODE_PROMPT,
        "debug_code_example": EXAMPLE,
        "system": SYSTEM_PROMPT,
    }

    def debug_code(
        self,
        cur_code: str,
        data_desc: Dict[str, str],
        code_executor: SimpleCodeExecutor,
        result: ExecutorReturn,
        one_shot: bool = True,
    ):
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["debug_code"]},
        ]
        if code_executor.run_init_once:
            background_code = code_executor.init_code.format(data_path="data.csv")
        else:
            background_code = (
                code_executor.init_code.format(data_path="data.csv")
                + code_executor.code_history
            )

        if one_shot:
            example_code = EXAMPLE.format(
                dataframe_schema_ex=json.dumps(FERTILITY_DATA_EX, indent=2),
                background_code_ex=BACKGROUND_CODE_EX,
                curr_code_ex=CURR_CODE_EX,
                error_ex=ERROR_MSG,
                reflection_ex=REFLECTION,
                thought_ex=THOUGHT,
                result_ex=RESULT,
            )
        else:
            example_code = ""

        prompt_variables = {
            "dataframe_schema": json.dumps(data_desc, indent=2),
            "background_code": background_code,
            "example": example_code,
            "curr_code": cur_code,
            "error": result.output,
        }

        raw_response = self.generate(
            prompt_template,
            prompt_variables,
            tags=["debug_code"],
            metadata={"prompt_name": "debug_code"},
        )

        reflection, code = self._parse_reflection_and_code(raw_response, "Result: ")

        return CodeAndReflection(code=code, reflection=reflection)
