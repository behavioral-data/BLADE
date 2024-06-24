from blade_bench.llms import LLMBase
from blade_bench.eval.llm.examples import (
    EX_ORIG_CODE,
    EX_CONVERTED_CODE,
    EX_ORIG_CODE_2,
    EX_CONVERTED_CODE_2,
    EX_ORIG_CODE_3,
    EX_CONVERTED_CODE_3,
    EX_ORIG_CODE_4,
    EX_CONVERTED_CODE_4,
    EX_ORIG_CODE_5,
    EX_CONVERTED_CODE_5,
    EX_ORIG_CODE_6,
    EX_CONVERTED_CODE_6,
    EX_ORIG_CODE_7,
    EX_CONVERTED_CODE_7,
)

SYSTEM_PROMPT = """You are an AI Python Data Science assistant \
who is an expert at understanding data transformation code."""

TRANSFORMATION_API_DESCRIPTION = """
Each transformation verb transforms the function in some way. In addition to the verb, \
depending on the code and transformation, there can be an input columns to output column \
mapping that specifies which input columns are used in the transformation and how \
it affects each output column specified or the whole datframe. \
Here is the description of each transformation verb, example code, \
and any input and output columns specification for each transformation:

**Derive**: Derive new column values based on the provided expressions. Input is a dataframe or groupby object.
    ```python
    import pandas as pd
    import numpy as np

    # create a dataframe
    df = pd.DataFrame({{
        'x': [1, 2, 3, 4, 5],
        'y': [5, 4, 3, 2, 1]
    }})
    # derive a new column 'sumXY' by adding 'x' and 'y'
    df['sumXY'] = df['x'] + df['y']
    ```
    column_mapping: {{frozenset(['x', 'y']): 'sumXY'}}
**Filter**: Filter a table to a subset of rows based on the input criteria. 
    ```python
    # filter the dataframe to include only rows where 'x' is greater than 2
    df = df[df['x'] > 2]
    ```
    column_mapping: {{frozenset(['x']): 'ALL'}}
**Groupby**: Group table rows based on a set of column values. Returns a groupby object
    ```python
    # group the dataframe by 'x'
    grouped = df.groupby('x')
    ```
    groupby_cols: set(['x'])
**De-duplicate**: De-duplicate table rows by removing repeated row values.
    ```python
    # remove duplicate rows in the dataframe
    df = df.drop_duplicates()
    ```
    column_mapping: {{}}
**Impute**: Impute missing values or rows.
    ```python
    # replace NaN values in column 'x' with the mean of 'x'
    df['x'] = df['x'].fillna(df['x'].mean())
    ```
    column_mapping: {{frozenset(['x']): 'x'}}
**Rollup**: Rollup a table to produce an aggregate summary. Input is a grouby object. This is used in conjunction with groupby when we aggregate a group.
    ```python
    grouped_df = df_grped.agg({'y': 'mean'})
    ```
    column_mapping: {{frozenset(['y']): 'y'}}
**Orderby**: Order table rows based on a set of column values. Returns a dataframe.**
    ```python
    # order the dataframe by 'x'
    df = df.sort_values(by='x')
    ```
    column_mapping: {{frozenset(['x']): 'ALL'}}
    
IMPORTANT:
Only "Filter" and "Orderby" transformations can have column_mapping with 'ALL' as the output column value. \
For other transformers you need be specific about the exact column mapping. 
IMPORTANT:
We do not consider any operation that selects a subset of columns as a transformation. \
The code for this should be included with another transformation. 
"""

INSTRUCTION_PROMPT = """<Instruction> Given a description of the transformation API, and Python code, \
convert the code to the sequence of unit transformations that are applied to the dataframe.
Each unit function will take in and return a TransformDataReturn objet defined below:
```python
class TransformDataReturn(BaseModel):
    df: pd.DataFrame
    column_mapping: Dict[FrozenSet[str], str] # we can specify multiple input and output column mappings
    groupby_cols: Set[str] # only for groupby verb
    transform_verb: Literal['derive', 'filter', 'groupby', 'deduplicate', 'impute', 'rollup', 'orderby']
```
These transform functions will be called by the main "transform" function that will apply the transformations in sequence. 
IMPORTANT: The transform functions (i.e., transform_funcs) cannot call each other or reference any variables outside of the function.
```python
def transform(df: pd.DataFrame,
              transform_funcs: List[Callable[[pd.DataFrame], TransformDataReturn]]):
    td_objs: List[TransformDataReturn] = []
    for func in transform_funcs:
        td_obj = func(df)
        df = td_obj.df
        td_objs.append(td_obj)
    return df
```
Your answer should be a list of transform functions of type \
Callable[[pd.DataFrame], TransformDataReturn] \
stored in the variable "transform_funcs". \
    
DO NOT ADD ANY NEW CODE THAT WAS NOT SPECIFIED IN THE ORIGINAL CODE.
For example, if the original code is empty then return empty.
If there is any statistical modeling code, do not include it in the transformation functions. 
We are only interested in the code that transforms the dataframe.
</Instruction>


<TransformationAPI> {transformation_api_description}
</TransformationAPI>

<Example 1> 
original code: ```python
{ex_code}
```
result: ```python
{ex_converted_code}
```
</Example 1>
<Example 2>
original code: ```python
{ex2_code}
```
result: ```python
{ex2_converted_code}
```
</Example 2>
<Example 3>
original code: ```python
{ex3_code}
```
result: ```python
{ex3_converted_code}
```
</Example 3>
<Example 4>
original code: ```python
{ex4_code}
```
result: ```python
{ex4_converted_code}
```
</Example 4>
<Example 5>
original code: ```python
{ex5_code}
```
result: ```python
{ex5_converted_code}
```
</Example 5>
<Example 6>
original code: ```python
{ex6_code}
```
result: ```python
{ex6_converted_code}
```
</Example 6>
<Example 7>
original code: ```python
{ex7_code}
```
result: ```python
{ex7_converted_code}
```
</Example 7>
 
original code: ```python
{code}
```
result:
"""


class CodeToTransformsLLM(LLMBase):
    prompt_templates = {
        "instruction": INSTRUCTION_PROMPT,
        "system": SYSTEM_PROMPT,
    }

    def convert_code(self, code: str) -> str:
        prompt_template = [
            {"role": "system", "content": self.prompt_templates["system"]},
            {"role": "user", "content": self.prompt_templates["instruction"]},
        ]
        prompt_variables = {
            "code": code,
            "ex_code": EX_ORIG_CODE,
            "ex_converted_code": EX_CONVERTED_CODE,
            "ex2_code": EX_ORIG_CODE_2,
            "ex2_converted_code": EX_CONVERTED_CODE_2,
            "ex3_code": EX_ORIG_CODE_3,
            "ex3_converted_code": EX_CONVERTED_CODE_3,
            "ex4_code": EX_ORIG_CODE_4,
            "ex4_converted_code": EX_CONVERTED_CODE_4,
            "ex5_code": EX_ORIG_CODE_5,
            "ex5_converted_code": EX_CONVERTED_CODE_5,
            "ex6_code": EX_ORIG_CODE_6,
            "ex6_converted_code": EX_CONVERTED_CODE_6,
            "ex7_code": EX_ORIG_CODE_7,
            "ex7_converted_code": EX_CONVERTED_CODE_7,
            "transformation_api_description": TRANSFORMATION_API_DESCRIPTION,
        }
        raw_response = self.generate(
            prompt_template,
            prompt_variables,
            tags=["gen_fuzzy_labels"],
            metadata={"prompt_name": "gen_fuzzy_labels"},
        )
        code = self._parse_code(raw_response)

        return code


if __name__ == "__main__":
    llm = CodeToTransformsLLM.init_from_base_llm_config()
    code_example = "df[df['Female_Contributions'] == df['UniqueFemaleContributors']].groupby('ThreadId').apply(lambda grp: grp.assign(WC_s=scale(grp['WC'])))"
    llm.convert_code(code_example)
    s = (
        code_example
    ) = """
df = df[df['Female_Contributions'] == df['UniqueFemaleContributors']]
df = df.groupby('ThreadId').apply(lambda group: group.assign(row_number=range(1, len(group) + 1))).reset_index(drop=True)
df = df.groupby(['Title']).apply(lambda group: group.iloc[0]).reset_index(drop=True)
df = df[df['Female'] == 1]
df = df.groupby(['Title', 'Id']).agg(
    co        prev_threads_mean=('PreviousThreads', 'mean'),
    threads_this_year_mean=('ThreadsThisYear'mments=('Id', 'count'),
, 'mean'),
    cont_this_year_mean=('ContributionsThisYear', 'mean'),
    prev_contributions_mean=('PreviousContributions', 'mean')
).reset_index()
df['total_prev_threads'] = df['prev_threads_mean'] + df['threads_this_year_mean']
df['total_prev_contributions'] = df['cont_this_year_mean'] + df['prev_contributions_mean']
df['cont_per_thread'] = df['total_prev_contributions'] / df['total_prev_threads']
df['comments_now_percent_change'] = (df['comments'] - df['cont_per_thread']) * 100 / df['cont_per_thread']
    """
    s2 = llm.convert_code(code_example)

    code_example = """
df = (df[df['Female_Contributions'] == df['UniqueFemaleContributors']]
          .groupby('ThreadId')
          .apply(lambda x: x.sort_values(by='Order'))
          .reset_index(drop=True))
df['next_female'] = df['Female'].shift(-1)
    """
    s3 = llm.convert_code(code_example)
    print("Done!")
