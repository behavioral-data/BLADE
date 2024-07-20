from absl.testing import absltest
import pandas as pd
import pytest
import unittest
from blade_bench.data.datamodel import TransformDataReturn
from blade_bench.nb.transform_unit_executor import TransformObjExecutor
from blade_bench.utils import get_dataset_csv_path
from blade_bench.nb.nb_executor import NotebookExecutorBasic

FERTILITY_PATH = get_dataset_csv_path("fertility")
CONVERSATION_PATH = get_dataset_csv_path("conversation")
TRANSFORM_CODE = """
def t1(df: pd.DataFrame):
    df = df.dropna(subset=['ReportedCycleLength'])
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['ReportedCycleLength']): 'ALL'},
            transform_verb='filter'
        )

def t2(df: pd.DataFrame):
    df = df.dropna(subset=['Rel1', 'Rel2', 'Rel3'])
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['Rel1', 'Rel2', 'Rel3']): 'ALL'},
            transform_verb='filter'
        )

transform_funcs = [t1, t2]
"""

TRANSFORM_CODE_INVALID = """
def t1(df: pd.DataFrame):
    df = df[df['Female_Contributions'] == df['UniqueFemaleContributors']]
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['Female_Contributions', 'UniqueFemaleContributors']): 'ALL'},
            transform_verb='filter'
        )

def t2(df: pd.DataFrame):
    df = df.head()
    return TransformDataReturn(
            df=df, 
            groupby_cols=frozenset(['ThreadId']),
            column_mapping={frozenset(['WC']): 'WC_s'},
            transform_verb='groupby'
        )

transform_funcs = [t1, t2]
"""


@pytest.mark.asyncio
async def test_nb_executor():
    try_code = "df = df.head()"
    nbexec = NotebookExecutorBasic(
        data_path=FERTILITY_PATH, save_path=".", run_init_once=True
    )
    outputs, sucess, my_var = await nbexec.run_code(try_code)

    df_head = pd.read_csv(FERTILITY_PATH).head()
    assert my_var.shape == df_head.shape
    assert my_var.columns.tolist() == df_head.columns.tolist()
    assert sucess


@pytest.mark.asyncio
async def test_nb_executor_invalid_code():
    invalid_code = "df = df.head(unknown_arg)"
    nbexec = NotebookExecutorBasic(
        data_path=FERTILITY_PATH, save_path=".", run_init_once=True
    )
    outputs, success, my_var = await nbexec.run_code(invalid_code)
    assert not success
    assert my_var is None


@pytest.mark.asyncio
async def test_nb_executor_transform_invalid():
    nbexec = TransformObjExecutor(data_path=CONVERSATION_PATH, save_path=".")
    res = await nbexec.run_code(TRANSFORM_CODE_INVALID)
    assert not res.success


@pytest.mark.asyncio
async def test_nb_executor_transform():
    nbexec = TransformObjExecutor(data_path=FERTILITY_PATH, save_path=".")
    res = await nbexec.run_code(TRANSFORM_CODE)
    assert isinstance(res.value, list)
    assert len(res.value) == 2
    assert all(isinstance(td_obj, TransformDataReturn) for td_obj in res.value)
    assert len(res.value[-1].df) == 206


if __name__ == "__main__":
    from blade_bench.data.datamodel import TransformDataReturn

    pytest.main([__file__])
