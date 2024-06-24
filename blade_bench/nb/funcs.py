from typing import List
import pandas as pd

from blade_bench.data.datamodel.transform_state import TransformState
from .df_hash import DFValueHasher
from .nb_executor import NotebookExecutorBasic


async def get_code_execeution_output(
    code: str,
    spec_id: str,
    nb_executor: NotebookExecutorBasic = None,
    columns_to_hash: List[str] = None,
    spec_name: str = None,
):
    outputs, success, my_var = await nb_executor.run_code(code)
    assert success, f"Code execution failed:\n{outputs}"
    if not isinstance(my_var, pd.DataFrame) or isinstance(my_var, pd.Series):
        if isinstance(my_var, pd.core.groupby.DataFrameGroupBy):
            columns_to_hash = []
            transform_state = get_transform_state(
                spec_id, my_var.any(), columns_to_hash, spec_name
            )
            transform_state.columns = list(my_var.any().columns)
            transform_state.df = None
            return transform_state
        else:
            return None
    else:
        return get_transform_state(spec_id, my_var, columns_to_hash, spec_name)


def get_transform_state(
    spec_id: str,
    df: pd.DataFrame,
    columns_to_hash: List[str],
    spec_name: str = None,
) -> TransformState:
    value_hasher = DFValueHasher(
        df=df,
        columns_to_hash=columns_to_hash,
    )
    hashes = value_hasher.hash()
    hashes_categorical = value_hasher.hash_categorical()
    return TransformState(
        spec_id=spec_id,
        df_value_hash=hashes,
        df_categorical_value_hash=hashes_categorical,
        df=df[value_hasher.columns_to_hash],
        spec_name=spec_name,
        columns=value_hasher.columns_to_hash,
    )
