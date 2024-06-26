from langchain.output_parsers import PydanticOutputParser
from blade_bench.llms import LLMBase
from blade_bench.eval.datamodel import ModelAndColumns

SYSTEM_PROMPT = """You are an AI Python Data Science assistant \
who is an expert at understanding statistical modeling code."""

INSTRUCTION_PROMPT = """<Instruction> 
Given the code snippet, carefully analyze and accurately \
determine the statistical model specification. Return the model specification that best \
matches the code snippet. 
</Instruction> 

<Example 1>
Code Snippet1: {code_snippet_ex}
Model Specification: {model_spec_ex}
</Example 1>
<Example 2>
Code Snippet: {code_snippet_2_ex}
Model Specification: {model_spec_2_ex}
</Example 2>
<Example 3>
Code Snippet: {code_snippet_3_ex}
Model Specification: {model_spec_3_ex}
</Example 3>

Code Snippet: {code_snippet}
Model Specification:
"""

INSTRUCTION_PROMPT_OBJ = """<Instruction> Given the code snippet, carefully analyze and accurately \
determine the statistical model specification. Return the model specification that best \
matches the code snippet. Please respond in the format specified by "Format Instructions".

IMPORTANT: the column names in the model specification should be the EXACT column names used in the model code. 
</Instruction> 

<Format Instructions>
{format_instructions}
</Format Instructions>

<Example 1>
Code Snippet: {code_snippet_ex}
Model Specification: {model_spec_ex}
</Example 1>
<Example 2>
Code Snippet: {code_snippet_2_ex}
Model Specification: {model_spec_2_ex}
</Example 2>
<Example 3>
Code Snippet: {code_snippet_3_ex}
Model Specification: {model_spec_3_ex}
</Example 3>

Code Snippet: {code_snippet}
Model Specification:
"""


CODE_SNIPPET_EX = """
X_df = pd.DataFrame(X, columns=["age", "education", "income"])
y_df = pd.Series(y, name='target')

X_df = sm.add_constant(X_df)

logit_model = sm.Logit(y_df, X_df)
"""

MODEL_SPEC_EX = "Logistic Regression"
MODEL_SPEC_EX_OBJ = ModelAndColumns(
    m_spec=MODEL_SPEC_EX, m_columns=["age", "education", "income", "target"]
)


CODE_SNIPPET_EX_2 = """
model = GLmer("y ~ X + Z + (1 | group)", data=df, family='poisson')
result = model.fit()
"""

MODEL_SPEC_EX_2 = (
    "Generalized linear mixed model with Poisson family and log link function"
)

MODEL_SPEC_EX_2_OBJ = ModelAndColumns(
    m_spec=MODEL_SPEC_EX_2, m_columns=["y", "X", "Z", "group"]
)

CODE_SNIPPET_EX_3 = """
lm = smf.ols('redCards ~ skin_tone_category * meanIAT', data=df).fit()
print(lm.summary(), '\n')
table = sm.stats.anova_lm(lm, typ=2)
print(table)
"""
MODEL_SPEC_EX_3 = "Linear Regression"
MODEL_SPEC_EX_3_OBJ = ModelAndColumns(
    m_spec=MODEL_SPEC_EX_3,
    m_columns=["redCards", "skin_tone_category", "meanIAT"],
)


class CodeToModelLLM(LLMBase):
    prompt_templates = {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION_PROMPT,
        "instruction_obj": INSTRUCTION_PROMPT_OBJ,
    }

    def code_to_model(
        self,
        code_snippet: str,
        code_snippet_ex: str = CODE_SNIPPET_EX,
        model_spec_ex: str = MODEL_SPEC_EX,
        code_snippet_2_ex: str = CODE_SNIPPET_EX_2,
        model_spec_2_ex: str = MODEL_SPEC_EX_2,
        code_snippet_3_ex: str = CODE_SNIPPET_EX_3,
        model_spec_3_ex: str = MODEL_SPEC_EX_3,
    ) -> str:
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction"]},
        ]
        prompt_variables = {
            "code_snippet": code_snippet,
            "code_snippet_ex": code_snippet_ex,
            "model_spec_ex": model_spec_ex,
            "code_snippet_2_ex": code_snippet_2_ex,
            "model_spec_2_ex": model_spec_2_ex,
            "code_snippet_3_ex": code_snippet_3_ex,
            "model_spec_3_ex": model_spec_3_ex,
        }
        raw_response = self.generate(
            prompt_template,
            prompt_variables,
            tags=["code_to_model"],
            metadata={"prompt_name": "code_to_model"},
        )
        resp = raw_response.strip()
        return resp

    def code_to_model_obj(
        self,
        code_snippet: str,
        code_snippet_ex: str = CODE_SNIPPET_EX,
        model_spec_ex: ModelAndColumns = MODEL_SPEC_EX_OBJ,
        code_snippet_2_ex: str = CODE_SNIPPET_EX_2,
        model_spec_2_ex: ModelAndColumns = MODEL_SPEC_EX_2_OBJ,
        code_snippet_3_ex: str = CODE_SNIPPET_EX_3,
        model_spec_3_ex: ModelAndColumns = MODEL_SPEC_EX_3_OBJ,
    ) -> ModelAndColumns:

        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction_obj"]},
        ]
        parser = PydanticOutputParser(pydantic_object=ModelAndColumns)
        prompt_variables = {
            "code_snippet": code_snippet,
            "code_snippet_ex": code_snippet_ex,
            "model_spec_ex": model_spec_ex.model_dump(),
            "code_snippet_2_ex": code_snippet_2_ex,
            "model_spec_2_ex": model_spec_2_ex.model_dump(),
            "code_snippet_3_ex": code_snippet_3_ex,
            "model_spec_3_ex": model_spec_3_ex.model_dump(),
            "format_instructions": parser.get_format_instructions(),
        }
        resp = self.generate_with_pydantic_parser(
            parser,
            prompt_template,
            prompt_variables,
            tags=["code_to_model"],
            metadata={"prompt_name": "code_to_model"},
        )
        return resp


if __name__ == "__main__":
    pass
