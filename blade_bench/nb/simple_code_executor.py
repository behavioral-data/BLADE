import traceback
from typing import Any, List
from pydantic import BaseModel
from .code_formatter import CodeFormatter
from blade_bench.logger import logger

from .nb_executor import NotebookExecutorBasic

INIT_CODE = """from pydantic import BaseModel
from typing import List, Optional, Callable, Union, Dict, FrozenSet, Set, Literal, Optional, Any
from pydantic import BaseModel, validator
import pandas as pd
import sklearn
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import pickle
  
df = pd.read_csv('{data_path}')
"""

CODE_TEMPLATE = """
{code}
"""

CODE_TEMPLATE_FUNC = """def transform(df: pd.DataFrame):
{code}
    return df
"""

POST_FIX_CODE = """
with open("{save_path}", "wb") as f:
    pickle.dump(__output__, f)
"""

POSTFIX_CODE_FUNC = """
result = transform(df)
with open("{save_path}", "wb") as f:
    pickle.dump(result, f)
"""


class ExecutorReturn(BaseModel):
    output: str  # any output strings from the code or the error message
    success: bool
    value: Any


class SimpleCodeExecutor(NotebookExecutorBasic):
    init_code: str = INIT_CODE
    postfix_code: str = POST_FIX_CODE
    add_tabs: bool = False
    code_template: str = CODE_TEMPLATE
    code_template_run: str = CODE_TEMPLATE
    run_init_once: bool = True  # run the init code only once and the context is kept
    _code_history: str = []

    async def reset_nb_executor(self):
        logger.debug("Resetting nb and clearing history...")
        self._code_history = []
        await super().reset_nb_executor()

    async def reset_state_before_error(self, num_resets, **kwargs):
        logger.debug("Resetting the state before the error...")
        self._code_history.pop()
        await super().reset_nb_executor()
        if len(self._code_history) == 0:
            return
        await self.run_code("\n".join(self._code_history), num_resets + 1, **kwargs)

    async def run_code(self, code: str, num_resets=0, **kwargs) -> ExecutorReturn:
        """
        Run the code and return the output
        """
        if num_resets >= 2:
            return ExecutorReturn(output="", success=False, value=None)

        self._code_history.append(code)
        # adds ___output__ =  to the last line of code
        # see CodeFormatter for details
        try:
            code_formatted = CodeFormatter.preprocess_code_for_tracing(code.strip())
        except SyntaxError as e:
            s = traceback.format_exc()
            std_err = "\n".join([s.split("\n")[0]] + s.split("\n")[-5:-1])
            self._code_history.pop()
            return ExecutorReturn(output=std_err, success=False, value=None)

        if "__output__" not in code_formatted:  # assume a function and return nothing
            return ExecutorReturn(
                output="",
                success=True,
                value=None,
            )
        output, success, myvar = await super().run_code(code_formatted, **kwargs)
        if not success:  # clean up to remove file path in the traceback error message
            # await self.reset_state_before_error(num_resets)
            output = output.split("nbclient.exceptions.CellExecutionError: ")[-1]

        return ExecutorReturn(output=output, success=success, value=myvar)

    @property
    def code_history(self):
        if len(self._code_history) == 0:
            return ""
        if self.run_init_once:
            return "\n".join(self._code_history)
        else:
            return "\n".join(self._code_history[-1])


class TransformCodeExecutor(NotebookExecutorBasic):
    init_code: str = INIT_CODE
    code_template: str = CODE_TEMPLATE_FUNC
    code_template_run: str = CODE_TEMPLATE_FUNC
    postfix_code: str = POSTFIX_CODE_FUNC
    add_tabs: bool = True

    async def run_code(self, code: str, **kwargs) -> ExecutorReturn:
        """
        Run the code and return the output
        """
        output, success, var = await super().run_code(code, **kwargs)
        return ExecutorReturn(output=output, success=success, value=var)


if __name__ == "__main__":
    pass
