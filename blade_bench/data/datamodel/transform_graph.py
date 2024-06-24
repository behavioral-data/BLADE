from typing import Dict, Optional

import networkx as nx
from pydantic import BaseModel

from blade_bench.data.datamodel import TransformSpec, ROOT_SPEC_ID
from .transform_state import SingleColState, TransformState


class ExpandedGraph(BaseModel):
    nx_g: nx.DiGraph
    id_to_spec: Dict[str, TransformSpec]
    id_to_ts: Optional[Dict[str, TransformState]] = {}

    class Config:
        arbitrary_types_allowed = True


class ColsGraph(BaseModel):
    nx_g: nx.DiGraph
    nid_to_state: Dict[str, SingleColState]

    class Config:
        arbitrary_types_allowed = True

    def get_parent_cols_of_col(
        self, col_name: str, exclude_og_cols: bool = False, derived_only: bool = False
    ):
        """
        derived_only: bool. If True, only return the columns that are derived.
        """
        node = next(
            n  # Select each node
            for n in self.nx_g.nodes  # Iterate over all nodes
            if (
                self.nx_g.out_degree(n) == 0  # Check if out degree is 0
                and self.nx_g.nodes[n]["col"]
                == col_name  # Check if column matches col_name
            )
        )
        ancestors = nx.ancestors(self.nx_g, node)
        if exclude_og_cols:
            ancestors = [
                n for n in ancestors if self.nx_g.nodes[n]["expanded"] == ROOT_SPEC_ID
            ]
        if derived_only:
            ancestors = [
                n
                for n in ancestors
                if self.nx_g.nodes[n]["val"] not in ["groupby", "filter"]
            ]
        cols = [self.nx_g.nodes[n]["col"] for n in ancestors]
        return cols
