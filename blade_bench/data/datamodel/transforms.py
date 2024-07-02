import copy
import pickle
from typing import Dict, FrozenSet, List, Literal, Optional, Set, Tuple
import networkx as nx

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from blade_bench.data.datamodel import TransformSpec
from .transform_graph import ColsGraph
from .transform_state import SingleColState, TransformState


class PathInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    path_graph: List[str]
    col_graph: ColsGraph
    col_graph_w_input: ColsGraph
    nid_to_col_state: Dict[str, SingleColState]
    cols_gid: int = None
    spec_id_to_ts: Dict[str, TransformState] = {}


class GraphHashInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    col_states: List[SingleColState]
    graph: Optional[nx.DiGraph] = Field(default=None, exclude=True)


class TransformDatasetState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id_to_spec: Dict[str, TransformSpec]
    expanded_id_to_spec: Dict[str, TransformSpec]
    graph_hashes: Dict[
        str, List[GraphHashInfo]
    ]  # hash val to graph and its single col states
    value_hashes: Dict[str, List[SingleColState]]  # hash val to single col states
    categorical_value_hashes: Dict[
        str, List[SingleColState]
    ]  # hash val to single col states
    graphs: Dict[int, PathInfo]  # graph id to path graphs
    converted_code: str = ""

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def summary_dict(self):
        return {
            "num_graphs": len(self.graphs),
            "num_graph_hashes": len(self.graph_hashes),
            "num_value_hashes": len(self.value_hashes),
            "num_tspec_expanded": len(self.expanded_id_to_spec),
        }


class TransformDataReturn(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    df: pd.DataFrame
    column_mapping: Dict[FrozenSet[str], str]
    transform_verb: Literal[
        "derive", "filter", "groupby", "deduplicate", "impute", "rollup", "orderby"
    ]
    groupby_cols: Optional[Set[str]] = set()  # only for groupby verb
    code: str = ""
