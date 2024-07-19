import json
import re
from typing import Dict, List, Tuple
from blade_bench.data.datamodel import ModelSpec
from blade_bench.eval.datamodel import MatchModel
from blade_bench.llms import LLMBase
from blade_bench.eval.llm.examples import (
    FERTILITY_MODELS_A,
    FERTILITY_MODELS_B,
    FERTILITY_MODELS_SIMILARITY_RESULT,
)


SYSTEM_PROMPT = """You are an AI Data Analysis assistant who is an expert at \
statistical modeling and understands the similarities, nuances, and differences \
between statistical models."""

INSTRUCTION_PROMPT = """<Instruction> Give different statistical model specifications \
written in natural language applied to the same analysis carefully analyze and accurately determine the \
matched model specifciations. Return ALL COMBINATIONS of matching pairs between the two sets of models.\
    
Please follow the example JSON format below for matching model specifications. \
For instance, if model 1 from \
models A is identical to model 2 and model 4 from models B, and model 2 from \
models A is identical to model 2 and model 3 from models B it should look like this:
{{
"A1-B2": {{"rationale": "<explain why A1 and B2 are identical>"}},
"A1-B4": {{"rationale": "<explain why A1 and B4 are identical>"}},
"A2-B2": {{"rationale": "<explain why A2 and B2 are identical>"}},
"A2-B3": {{"rationale": "<explain why A2 and B3 are identical>"}},
...
}} 

Do not match a variable with itself. Note that you should only match models that are \
the exact same for conducting the analysis. \
    
Specifically, two models are the same if give the same data, the model specification would \
lead to the same results. For example "logisistic regression" is equivalent to \
a "binomial GLM with a logit link function" or "binary classification using logistic regression". \
However, "logistic regression" is not equivalent to "linear regression". \
If the specification or code references specific data or variables, IGNORE it for this task. \
</Instruction>

<Example>
Models to Match:
======Statistical Model Specification A:
```
{models_a_ex}
```
======Statistical Model Specification B:
```
{models_b_ex}
Result:
{result_ex}
</Example>

Models to Match:
======Statistical Model Specification A:
```
{models_a}
```
======Statistical Model Specification B:
```
{models_b}
```
Result:
"""


class StatsModelSimilarity(LLMBase):
    prompt_templates = {
        "system": SYSTEM_PROMPT,
        "instruction": INSTRUCTION_PROMPT,
    }

    def match_models_str(
        self,
        models_a: List[str],
        models_b: List[str],
    ) -> Dict[Tuple[int, int], MatchModel]:
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction"]},
        ]

        d1 = {f"{i}": ma for i, ma in enumerate(models_a, 1)}

        d2 = {f"{i}": mb for i, mb in enumerate(models_b, 1)}

        prompt_variables = {
            "models_a": json.dumps(d1, indent=2),
            "models_b": json.dumps(d2, indent=2),
            "models_a_ex": json.dumps(FERTILITY_MODELS_A, indent=2),
            "models_b_ex": json.dumps(FERTILITY_MODELS_B, indent=2),
            "result_ex": json.dumps(FERTILITY_MODELS_SIMILARITY_RESULT, indent=2),
        }

        ret = self.generate(
            prompt_template,
            prompt_variables,
            tags=["model_match_similarities"],
            metadata={"prompt_name": "model_match_similarities"},
        )
        json_resp = self.match_json_catch_error(ret)

        return json_resp

    def match_models(
        self,
        models_a: List[ModelSpec],
        models_b: List[ModelSpec],
    ) -> Dict[Tuple[int, int], MatchModel]:
        model_specs_a = [m.specification.strip().lower() for m in models_a]
        model_specs_b = [m.specification.strip().lower() for m in models_b]

        json_resp = self.match_models_str(model_specs_a, model_specs_b)
        matched = {}

        for k, v in json_resp.items():
            numbers = re.findall(r"\d+", k)
            result_tuple = tuple(map(int, numbers))
            try:
                matched[result_tuple] = MatchModel(
                    model1=models_a[result_tuple[0] - 1],
                    model2=models_b[result_tuple[1] - 1],
                    rationale=v["rationale"],
                )
            except Exception as e:
                print(f"Error: {e}")
                continue
        return matched


if __name__ == "__main__":
    pass
