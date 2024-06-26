from .nb_executor import (
    ExecutorReturn,
    NotebookExecutorBasic,
)
from blade_bench.data.datamodel import TransformDataReturn


INIT_CODE = """from pydantic import BaseModel, ConfigDict
from typing import List, Optional, Callable, Union, Dict, FrozenSet, Set, Literal, Optional
from pydantic import BaseModel, field_validator
import copy
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('{data_path}')

class TransformDataReturn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame
    column_mapping: Dict[FrozenSet[str], str]
    transform_verb: Literal[
        "derive", "filter", "groupby", "deduplicate", "impute", "rollup", "orderby"
    ]
    groupby_cols: Optional[Set[str]] = set()  # only for groupby verb
    code: str = ""
    @field_validator("df")
    def check_df_is_df(cls, v):
        assert isinstance(v, pd.DataFrame), f"df is of type {{type(v)}} and not a pd.DataFrame object"
        return v    

def transform(df: pd.DataFrame,
              transform_funcs: List[Callable[[pd.DataFrame], TransformDataReturn]]):
    td_objs: List[TransformDataReturn] = []
    for func in transform_funcs:
        td_obj = func(df)
        df = copy.deepcopy(td_obj.df)
        td_objs.append(td_obj)
    return td_objs
"""

CODE_TEMPLATE = """
{code}
"""

POSTFIX_CODE = """
result = transform(df, transform_funcs)

assert isinstance(result, list), f"Expected result to be a list, but got {{type(result)}}"
assert all(isinstance(td_obj, TransformDataReturn) for td_obj in result), f"Expected all elements in the list to be of type TransformDataReturn, but got {{[type(td_obj) for td_obj in result]}}"
for td_obj in result:
    for col_set, col_name in td_obj.column_mapping.items():
        assert isinstance(col_set, frozenset), f"Expected column set to be of type frozenset, but got {{type(col_set)}}"
        assert isinstance(col_name, str), f"Expected column name to be of type str, but got{{type(col_name)}}"
        if col_name != 'ALL':
            assert col_name in td_obj.df.columns, f"Expected column name {{col_name}} to be in the DataFrame columns, but it is not"

with open("{save_path}", "wb") as f:
    pickle.dump(result, f)
"""


class TransformObjExecutor(NotebookExecutorBasic):
    init_code: str = INIT_CODE
    code_template: str = CODE_TEMPLATE
    code_template_run: str = CODE_TEMPLATE
    postfix_code: str = POSTFIX_CODE
    add_tabs: bool = False

    async def run_code(self, code: str, **kwargs) -> ExecutorReturn:
        """
        Run the code and return the output
        """
        output, success, var = await super().run_code(code, **kwargs)
        return ExecutorReturn(output=output, success=success, value=var)


if __name__ == "__main__":
    pass
