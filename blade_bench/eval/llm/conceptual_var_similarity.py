import re
from typing import List, Union
import json

from blade_bench.data.dataset import DatasetInfo
from blade_bench.data.datamodel import ConceptualVarSpec
from blade_bench.eval.datamodel import MatchCvar, MatchedCvars

from blade_bench.llms import LLMBase
from blade_bench.eval.llm.examples import (
    FERTILITY_DINFO,
    FERTILITY_VARIABLES_A,
    FERTILITY_VARIABLES_B,
    FERTILITY_CVAR_SIMILARITY_RESULT,
)
from blade_bench.logger import logger

SYSTEM_PROMPT = """You are an AI Data Analysis assistant who is an expert at \
understanding a research question, relecting on the data and relevant domain \
knowledge, and representing this conceptual knowledge in a statistical model. \
Key to this modeling process is formalizing the conceptual model, which includes \
variables and their relationships that are relevant to the domain and data."""


INSTRUCTION_PROMPT = """<Instruction> Given the research question, dataset, and existing conceptual variables already \
carefully analyze and accurately match the conceptual variables specified ensuring a \
strong correspondence between the matched points. Examine the verbatim closely.

Please follow the example JSON format below for matching decisions. For instance, if variable 1 from \
variables A is nearly identical to variable 2 from variables B, it should look like this:
{{
"A1-B2": {{"rationale": "<explain why A1 and B2 are nearly identical>", "similarity":
    "<5-10, only an integer>"}},
...
}}
Do not match a variable with itself. Note that you should only match variables with a significant degree of similarity for conducting the analysis. \
Also pay attention to the type of variable it is (e.g., independent, dependent, control) and how the \
variable fits with the research question, dataset, and the analysis.

Specifically, two variables would be similar if they were to be operationlized in the same way, \
would be used in the same way in a statistical model, and lead to measurements of the same concept. \
Refrain from matching points with only superficial similarities or weak connections. \
For each matched pair, rate the similarity on a scale of 5-10.
5. Somewhat Related: Variables address a similar concept for the analysis but from different angles and would be operationalized differently.
6. Moderately Related: Variables address a similar concept and but might be operationalized differently.
7. Strongly Related: Variables are largely aligned but differ in some details or nuances that can impact the analysis differently.
8. Very Strongly Related: Variables offer similar concepts or concerns, and would be operationalized in a similar way.
9. Almost Identical: Variables are nearly the same, operationalized the same way, with minor differences in wording or
    presentation.
10. Identical: Variables are exactly the same in terms of the concept, operationalization, and impact on the analysis.
If no match is found, output an empty JSON object. Provide your output as JSON only.
</Instruction>

<Example>
Research Question: {research_question_ex}
Dataset Schema: {dataset_schema_ex}
Variables to Match:
======Conceptual Variables A:
```
{variables_a_ex}
```
======Conceptual Variables B:
```
{variables_b_ex}
```
Result: 
```json
{result_ex}
```
</Example>

Research Question: {research_question}
Dataset Schema: {dataset_schema}
Variables to Match:
======Conceptual Variables A:
```
{variables_a}
```
======Conceptual Variables B:
```
{variables_b}
```
Result: 
"""


class ConceptualVarSimilarity(LLMBase):
    prompt_templates = {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION_PROMPT,
    }

    def match_variables_str(
        self,
        dataset_info: DatasetInfo,
        var_list1: List[str],
        var_list2: List[str],
    ):
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction"]},
        ]
        d1 = {f"{i}": var for i, var in enumerate(var_list1, 1)}
        d2 = {f"{i}": var for i, var in enumerate(var_list2, 1)}

        prompt_variables = {
            "research_question": dataset_info.research_question,
            "dataset_schema": json.dumps(
                dataset_info.data_desc_no_semantic_type, indent=2
            ),
            "variables_a": json.dumps(d1, indent=2),
            "variables_b": json.dumps(d2, indent=2),
            "variables_a_ex": json.dumps(FERTILITY_VARIABLES_A, indent=2),
            "variables_b_ex": json.dumps(FERTILITY_VARIABLES_B, indent=2),
            "result_ex": json.dumps(FERTILITY_CVAR_SIMILARITY_RESULT, indent=2),
            "research_question_ex": FERTILITY_DINFO.research_question,
            "dataset_schema_ex": json.dumps(
                FERTILITY_DINFO.data_desc_no_semantic_type, indent=2
            ),
        }

        ret = self.generate(
            prompt_template,
            prompt_variables,
            tags=["cvar_match_similarities"],
            metadata={"prompt_name": "cvar_match_similarities"},
        )

        json_resp = self.match_json_catch_error(ret)
        matched = {}
        for k, v in json_resp.items():
            numbers = re.findall(r"\d+", k)
            result_tuple = tuple(map(int, numbers))

            try:
                matched[result_tuple] = MatchCvar(
                    var1=var_list1[result_tuple[0] - 1],
                    var2=var_list2[result_tuple[1] - 1],
                    rationale=v["rationale"],
                    similarity=int(v.get("similarity", 0)),
                )
            except Exception as e:
                logger.error(f"Error in matching variables: {e}")
                continue
        return MatchedCvars(
            input_vars1=var_list1,
            input_vars2=var_list2,
            matched=matched,
        )

    def match_variables_cvarspec(
        self,
        dataset_info: DatasetInfo,
        var_list1: List[ConceptualVarSpec],
        var_list2: List[ConceptualVarSpec],
    ):
        v1 = [str(var) for var in var_list1]
        v2 = [str(var) for var in var_list2]
        return self.match_variables_str(dataset_info, v1, v2)

    def match_variables(
        self,
        rq: str,
        dataset_schema: str,
        cvar_list1: List[ConceptualVarSpec],
        cvar_list2: List[ConceptualVarSpec],
    ):
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction"]},
        ]
        d1 = {f"{i}": str(cvar) for i, cvar in enumerate(cvar_list1, 1)}
        d2 = {f"{i}": str(cvar) for i, cvar in enumerate(cvar_list2, 1)}
        inp1 = cvar_list1
        inp2 = cvar_list2

        prompt_variables = {
            "research_question": rq,
            "dataset_schema": dataset_schema,
            "variables_a": json.dumps(d1, indent=2),
            "variables_b": json.dumps(d2, indent=2),
        }

        ret = self.generate(
            prompt_template,
            prompt_variables,
            tags=["cvar_match_similarities"],
            metadata={"prompt_name": "cvar_match_similarities"},
        )

        json_resp = self.match_json_catch_error(ret)

        matched = {}
        for k, v in json_resp.items():
            numbers = re.findall(r"\d+", k)
            result_tuple = tuple(map(int, numbers))
            matched[result_tuple] = MatchCvar(
                var1=inp1[result_tuple[0] - 1],
                var2=inp2[result_tuple[1] - 1],
                rationale=v["rationale"],
            )
        return MatchedCvars(
            input_vars1=inp1,
            input_vars2=inp2,
            matched=matched,
        )


if __name__ == "__main__":
    from blade_bench.eval.llm.examples import FERTILITY_CVARS

    cspecs = [cvar for cvar in FERTILITY_CVARS.to_cvar_specs().values()]

    llm = ConceptualVarSimilarity.init_from_base_llm_config()
    matched = llm.match_variables_cvarspec(FERTILITY_DINFO, cspecs, cspecs)
    print("Done!")
