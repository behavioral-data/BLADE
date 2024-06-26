import json
from typing import Any, Dict, List, Literal, Tuple, Union
import networkx as nx
import pandas as pd

from ..datamodel import (
    Branch,
    ConceptualVarSpec,
    ModelSpec,
    TransformSpec,
    TRANSFORM_SPEC_COLUMN_NAME,
    MODEL_SPEC_COLUMN_NAME,
    CONCEPTUAL_VAR_SPEC_COLUMN_NAME,
)

COL_NAME_TO_CLASS = {
    TRANSFORM_SPEC_COLUMN_NAME: TransformSpec,
    MODEL_SPEC_COLUMN_NAME: ModelSpec,
    CONCEPTUAL_VAR_SPEC_COLUMN_NAME: ConceptualVarSpec,
}


def get_graph_from_df(df: pd.DataFrame) -> Tuple[nx.DiGraph, Dict[str, Any], str]:
    row = df[df.dependency_graph.isna() == False]
    assert len(row) == 1, "Only one dependency graph should be saved"
    g_json = json.loads(row["dependency_graph"].values[0])
    nx_g = nx.node_link_graph(g_json)
    return nx_g, g_json, row["spec_id"].values[0]


def get_saved_specs_from_df(
    df: pd.DataFrame,
    spec_col: Literal[
        "transform_spec_json",
        "model_spec_json",
        "conceptual_spec_json",
    ],
):
    if spec_col not in df.columns:
        return []
    df = df[df[spec_col].isna() == False]
    specs: List[Union[TransformSpec, ModelSpec, ConceptualVarSpec]] = []
    if "spec_id" in df.columns:
        for spec_json_str, spec_id in zip(df[spec_col].values, df["spec_id"].values):
            spec_kwargs = json.loads(spec_json_str)
            if spec_id:
                spec_kwargs["spec_id"] = spec_id
            if spec_col == "transform_spec_json":
                if "branches" in spec_kwargs and isinstance(
                    spec_kwargs["branches"][0], list
                ):
                    spec_kwargs["branches"] = [
                        Branch(dependencies=b).model_dump()
                        for b in spec_kwargs["branches"]
                    ]

            specs.append(COL_NAME_TO_CLASS[spec_col](**spec_kwargs))
    else:
        for spec_json_str in df[spec_col].values:
            specs.append(COL_NAME_TO_CLASS[spec_col](**json.loads(spec_json_str)))
    return specs
