from typing import Dict, Optional

from blade_bench.parse_code import process_groupby_code
from .specs import ROOT_SPEC_ID, LEAF_SPEC_ID, Branch, TransformSpec
from pydantic import BaseModel, ConfigDict
import networkx as nx


class SerialGraphCodeRunInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    branch: Branch
    serial_nx_g: nx.DiGraph
    leaf_node_spec_id: Optional[str] = None
    unserial_nx_g: Optional[nx.DiGraph] = None

    def build_code_str(self, specs: Dict[str, TransformSpec]):
        code_str = ""
        output_cols = set()
        for n in nx.topological_sort(self.serial_nx_g):
            if n == ROOT_SPEC_ID:
                continue
            if n == LEAF_SPEC_ID:
                continue
            code_str += f"{process_groupby_code(self.serial_nx_g.nodes[n]['code'])}\n"
            output_cols.update(specs[n].output_cols)
        return code_str.strip(), output_cols
