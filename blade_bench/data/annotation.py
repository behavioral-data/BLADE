from collections import defaultdict
import pickle
from typing import Dict, List, Literal, Optional, Set
import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field


from blade_bench.data.datamodel import (
    ROOT_SPEC_ID,
    ConceptualVarSpec,
    ModelSpec,
    TransformSpec,
)
from blade_bench.parse_code import process_groupby_code
from .datamodel.transforms import TransformDatasetState
from .process import AnnotationDataTransforms


class AnnotationDBData(BaseModel):
    transform_specs: Dict[str, TransformSpec]
    cv_specs: Dict[str, ConceptualVarSpec]
    model_specs: Dict[str, ModelSpec]
    nx_g: nx.DiGraph
    df: Optional[pd.DataFrame] = None  # analysis dataset_df
    annotator: Optional[str] = None
    _state_data: Optional[TransformDatasetState] = None

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_model_associated_cvars(self) -> Dict[str, List[ConceptualVarSpec]]:
        mspec_id_to_cvars = defaultdict(list)
        for model_spec in self.model_specs.values():
            for col in model_spec.associated_columns_orig:
                for cvar in self.cv_specs.values():
                    if col in cvar.final_columns_orig:
                        mspec_id_to_cvars[model_spec.spec_id].append(cvar)
            for col in model_spec.associated_columns_derived:
                for cvar in self.cv_specs.values():
                    if col in cvar.final_columns_derived:
                        mspec_id_to_cvars[model_spec.spec_id].append(cvar)
        return mspec_id_to_cvars

    def build_code_str(self, g: nx.DiGraph):
        code_str = ""
        for n in nx.topological_sort(g):
            if n == ROOT_SPEC_ID:
                continue
            code_str += f"{process_groupby_code(g.nodes[n]['code'])}\n"
        return code_str.strip()

    def get_cvars_by_type(
        self,
    ) -> Dict[
        Literal["IV", "DV", "Control", "Moderator"],
        Dict[str, ConceptualVarSpec],
    ]:
        cvars_by_type = defaultdict(dict)
        for cvar in self.cv_specs.values():
            if cvar.variable_type != "Control":
                cvars_by_type[cvar.variable_type][cvar.spec_id] = cvar
            else:
                if not cvar.interaction:
                    cvars_by_type["Control"][cvar.spec_id] = cvar
                else:
                    cvars_by_type["Moderator"][cvar.spec_id] = cvar
        return cvars_by_type

    def build_graph_from_specs(self):
        nx_g = nx.DiGraph()
        for spec in self.transform_specs.values():
            nx_g.add_node(spec.spec_id)
            for branch in spec.branches:
                for dep in branch.dependencies:
                    nx_g.add_edge(dep, spec.spec_id)
        return nx_g

    async def get_state_data(
        self, data_path: str, save_path: str = "."
    ) -> TransformDatasetState:
        # TODO need to test this method
        if self._state_data is None:
            nx_g = self.build_graph_from_specs()

            annotate = AnnotationDataTransforms(
                data_path,
                id_to_spec=self.transform_specs,
                nx_g=nx_g,
                save_path=save_path,
            )
            state_data: TransformDatasetState = await annotate.build_state_data(
                save_ts=True
            )
            self._state_data = state_data
            await annotate.nb_executor.nb_executor.terminate()
        return self._state_data

    def summary_stats(self):
        return {
            "num_transform_specs": len(self.transform_specs),
            "num_code_paths": len(self.get_serial_graphs_run_info()),
            "num_branches": sum(
                [
                    len(spec.branches)
                    for spec in self.transform_specs.values()
                    if len(spec.branches) > 1
                ]
            ),
            "num_cv_specs": len(self.cv_specs),
            "num_model_specs": len(self.model_specs),
        }
