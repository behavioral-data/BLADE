from typing import List
import pandas as pd
import pytest
from unittest import TestCase

import networkx as nx
from blade_bench.data.datamodel.transforms import (
    TransformDataReturn,
    TransformDatasetState,
)
from blade_bench.data.process.transforms.dataflow import AnnotationDataTransforms
from blade_bench.eval.match.transform import TransformMatcher

df = pd.DataFrame(
    {
        "o1": [1, 2, 3, 4],
        "o2": [5, 6, 7, 8],
        "o3": [9, 10, 11, 12],
        "o4": [13, 14, 15, 16],
    }
)


@pytest.fixture
def transform_returns_gnd_truth():
    transform_rets = [
        TransformDataReturn(
            df=df[(df.o1 >= 2) & (df.o2 <= 7) & (df.o3 <= 13) & (df.o4 <= 16)],
            column_mapping={frozenset(["o1", "o2", "o3", "o4"]): "ALL"},
            transform_verb="filter",
            code="df = df[df.o1 >= 2 & df.o2 <= 7 & df.o3 <= 13 & df.o4 <= 16]",
        ),
    ]
    return transform_rets


@pytest.fixture
def transform_returns_analysis():
    transform_rets = [
        TransformDataReturn(
            df=df[(df.o1 >= 2) & (df.o2 <= 7)],
            column_mapping={frozenset(["o1", "o2"]): "ALL"},
            transform_verb="filter",
            code="df = df[df.o1 >= 2 & df.o2 <= 7]",
        )
    ]
    return transform_rets


@pytest.fixture
def dataflow_annotation():
    return AnnotationDataTransforms(data_columns=list(df.columns), run_nb=False)


@pytest.mark.asyncio
async def test_match_transform_eval_input(
    transform_returns_gnd_truth: List[TransformDataReturn],
    transform_returns_analysis: List[TransformDataReturn],
    dataflow_annotation: AnnotationDataTransforms,
):
    gnd_truth: TransformDatasetState = (
        await dataflow_annotation.build_state_data_from_transform_return(
            transform_returns_gnd_truth
        )
    )
    llm_state: TransformDatasetState = (
        await dataflow_annotation.build_state_data_from_transform_return(
            transform_returns_analysis
        )
    )

    expected_llm_dict = {
        "o2__ROOT_SPEC_ID": {
            "o2__node_0_filter": {"type": "input + data"},
            "o3__node_0_filter": {"type": "input"},
            "o1__node_0_filter": {"type": "input"},
            "o4__node_0_filter": {"type": "input"},
        },
        "o3__ROOT_SPEC_ID": {"o3__node_0_filter": {"type": "data"}},
        "o1__ROOT_SPEC_ID": {
            "o1__node_0_filter": {"type": "input + data"},
            "o2__node_0_filter": {"type": "input"},
            "o3__node_0_filter": {"type": "input"},
            "o4__node_0_filter": {"type": "input"},
        },
        "o4__ROOT_SPEC_ID": {"o4__node_0_filter": {"type": "data"}},
        "o2__node_0_filter": {},
        "o3__node_0_filter": {},
        "o1__node_0_filter": {},
        "o4__node_0_filter": {},
    }

    expected_gnd_truth_dict = {
        "o2__ROOT_SPEC_ID": {
            "o2__node_0_filter": {"type": "input + data"},
            "o3__node_0_filter": {"type": "input"},
            "o1__node_0_filter": {"type": "input"},
            "o4__node_0_filter": {"type": "input"},
        },
        "o3__ROOT_SPEC_ID": {
            "o3__node_0_filter": {"type": "input + data"},
            "o2__node_0_filter": {"type": "input"},
            "o1__node_0_filter": {"type": "input"},
            "o4__node_0_filter": {"type": "input"},
        },
        "o1__ROOT_SPEC_ID": {
            "o1__node_0_filter": {"type": "input + data"},
            "o2__node_0_filter": {"type": "input"},
            "o3__node_0_filter": {"type": "input"},
            "o4__node_0_filter": {"type": "input"},
        },
        "o4__ROOT_SPEC_ID": {
            "o4__node_0_filter": {"type": "input + data"},
            "o2__node_0_filter": {"type": "input"},
            "o3__node_0_filter": {"type": "input"},
            "o1__node_0_filter": {"type": "input"},
        },
        "o2__node_0_filter": {},
        "o3__node_0_filter": {},
        "o1__node_0_filter": {},
        "o4__node_0_filter": {},
    }

    TestCase().assertDictEqual(
        expected_gnd_truth_dict,
        nx.to_dict_of_dicts(gnd_truth.graphs[0].col_graph_w_input.nx_g),
    )
    TestCase().assertDictEqual(
        expected_llm_dict,
        nx.to_dict_of_dicts(llm_state.graphs[0].col_graph_w_input.nx_g),
    )
    # the values are the same but the graph hashes are different
    transform_matcher = TransformMatcher(dataset_name="none")
    matched_specs = transform_matcher.match_w_tsdata(ts1=gnd_truth, ts2=llm_state)
    assert len(matched_specs.vspecs1) == 2  # including ROOT
    assert len(matched_specs.vspecs2) == 2
    assert len(matched_specs.gspecs1) == 0
    assert len(matched_specs.gspecs2) == 0


if __name__ == "__main__":
    pytest.main([__file__])
