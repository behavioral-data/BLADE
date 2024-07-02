import os
import os.path as osp
from typing import Any, Dict, FrozenSet, Literal, Optional, Set
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
import pickle
from blade_bench.nb.base import ExecuteNbCode
from blade_bench.data.datamodel import TransformDataReturn

TAB_STR = "    "
INIT_CODE = """from pydantic import BaseModel
from typing import List, Optional, Callable, Union
from pydantic import BaseModel, validator
from pandas.core.groupby import DataFrameGroupBy
import pandas as pd
import numpy as np
import scipy

import pickle
    
df = pd.read_csv('{data_path}')
"""

POSTFIX_CODE = """
result = transform(df)
with open("{save_path}", "wb") as f:
    pickle.dump(result, f)
"""
CODE_TEMPLATE_RUN = """def transform(df: Union[pd.DataFrame, DataFrameGroupBy]):
    df = df.copy(deep=True)
{code}
    return df
"""

CODE_TEMPLATE = """def transform(df: Union[pd.DataFrame, DataFrameGroupBy]):
{code}
    return df
"""

CODE_TEMPLATE_W_NEW_CODE = """def transform(df: Union[pd.DataFrame, DataFrameGroupBy]):
{prev_code}
    # your code added here 👇
{new_code}
    return df
"""


CODE_TEMPLATE_INSTRUCTION = """def transform(df: Union[pd.DataFrame, DataFrameGroupBy]):
{prev_code}
    # add your code here 👇
    return df
"""

CODE_TEMPLATE_FOR_LLM = """def transform(df: Union[pd.DataFrame, DataFrameGroupBy]):
{prev_code}
    # ADD YOUR CODE HERE, SET THE RETURN VALUE AFTER TRANSFORMING THE DATAFRAME TO "df"
    return df
"""


class ExecutorReturn(BaseModel):
    output: str  # any output strings from the code or the error message
    success: bool
    value: Any


class NotebookExecutorBasic(BaseModel):
    data_path: str
    save_path: str = "."
    nb_executor: ExecuteNbCode = Field(
        default_factory=ExecuteNbCode,
        exclude=True,
    )
    init_code: str = INIT_CODE
    postfix_code: str = POSTFIX_CODE
    ran_init_code: bool = False
    run_init_once: bool = False
    code_template: str = CODE_TEMPLATE
    code_template_run: str = CODE_TEMPLATE_RUN
    add_tabs: bool = True

    async def reset_nb_executor(self, data_path: str = None):
        await self.nb_executor.build()
        await self.nb_executor.reset()
        if data_path:
            self.data_path = data_path
        await self.run_init_code()

    async def terminate(self):
        await self.nb_executor.terminate()
        self.ran_init_code = False

    @property
    def cur_code(self):
        return self.nb_executor.nb.cells[-1].source

    async def run_init_code(self):
        output_str, success = await self.nb_executor.run(
            self.init_code.format(data_path=self.data_path)
        )
        if not success:
            raise Exception(f"Failed to run init code: {output_str}")
        self.ran_init_code = True

    async def run_code(
        self,
        code: str,
        save_obj_name: str = "saved_obj.pkl",
    ) -> pd.DataFrame:
        if save_obj_name is None:
            save_obj_name = "saved_obj.pkl"
        save_path = osp.join(self.save_path, save_obj_name)
        if self.save_path is not None and not osp.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        myvar = None
        if not self.run_init_once or not self.ran_init_code:
            await self.run_init_code()
        template = self.code_template_run if self.run_init_once else self.code_template
        if self.add_tabs:
            code = TAB_STR + code.replace("\n", f"\n{TAB_STR}")
        output, success = await self.nb_executor.run(
            template.format(code=code) + self.postfix_code.format(save_path=save_path)
        )
        if success:
            with open(save_path, "rb") as f:
                myvar = pickle.load(f)
        # remove the file
        if osp.exists(save_path):
            os.remove(save_path)
        return output, success, myvar


async def run_code(sheets_path, code):
    if osp.exists(sheets_path):
        path = sheets_path
    else:
        path = sheets_path.split("/edit")[0] + "/export?format=csv"
    nbexec = NotebookExecutorBasic(data_path=path, save_path=".")
    outputs, success, my_var = await nbexec.run_code(code)
    return outputs, success, my_var


def add_tabs_to_code(code_str: str):
    return TAB_STR + code_str.replace("\n", f"\n{TAB_STR}")


# Example usage

if __name__ == "__main__":
    import pickle
    import os.path as osp
    import asyncio
    from blade_bench.utils import get_dataset_csv_path

    dataset_csv_path = get_dataset_csv_path("fertility")
    TRY_CODE = "df = df.head()"
    nbexec = NotebookExecutorBasic(
        data_path=dataset_csv_path, save_path=".", run_init_once=True
    )

    async def main():
        outputs, sucess, my_var = await nbexec.run_code(TRY_CODE)
        print(outputs)
        print("here")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
