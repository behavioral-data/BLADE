from collections import defaultdict
import json
import pickle
from typing import Any, Dict, List, Literal, Optional, Set, Union
import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, computed_field


from blade_bench.data.datamodel import (
    ROOT_SPEC_ID,
    CONCEPTUAL_VAR_SPEC_COLUMN_NAME,
    MODEL_SPEC_COLUMN_NAME,
    TRANSFORM_SPEC_COLUMN_NAME,
    ConceptualVarSpec,
    ModelSpec,
    TransformSpec,
)
from blade_bench.data.datamodel.graph import SerialGraphCodeRunInfo
from blade_bench.data.datamodel.specs import BaseSpec, Branch
from blade_bench.data.process.transforms.graph_paths import GraphPaths
from blade_bench.parse_code import process_groupby_code
from .datamodel.transforms import TransformDatasetState
from .process import AnnotationDataTransforms


COL_NAME_TO_CLASS = {
    TRANSFORM_SPEC_COLUMN_NAME: TransformSpec,
    MODEL_SPEC_COLUMN_NAME: ModelSpec,
    CONCEPTUAL_VAR_SPEC_COLUMN_NAME: ConceptualVarSpec,
}


# TODO pretty view the different specs
class AnnotationDBData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    transform_specs: Dict[str, TransformSpec]
    cv_specs: Dict[str, ConceptualVarSpec]
    m_specs: Dict[str, ModelSpec]
    nx_g: nx.DiGraph = Field(default=None, exclude=True)
    df: Optional[pd.DataFrame] = None  # analysis dataset_df
    annotator: Optional[str] = None
    _state_data: Optional[TransformDatasetState] = None

    def model_post_init(self, __context: Any) -> None:
        if self.nx_g is None:
            self.nx_g = self.build_graph_from_specs()

    @property
    def num_unique_code_paths(self):
        def min_path_cover_dag(dag: nx.DiGraph) -> int:
            n = len(dag.nodes)

            # Create a bipartite graph
            bipartite_graph = nx.DiGraph()

            for u in dag.nodes:
                bipartite_graph.add_node(f"{u}_1")
                bipartite_graph.add_node(f"{u}_2")

            for u, v in dag.edges:
                bipartite_graph.add_edge(f"{u}_1", f"{v}_2")

            # Find maximum matching
            max_matching = nx.bipartite.maximum_matching(
                bipartite_graph, top_nodes=[f"{u}_1" for u in dag.nodes]
            )

            # Calculate minimum path cover
            min_path_cover = n - len(max_matching) // 2
            return min_path_cover

        leaf_specs = [
            self.transform_specs[node]
            for node in self.nx_g.nodes
            if self.nx_g.out_degree(node) == 0
            and node in self.transform_specs
            and node != ROOT_SPEC_ID
        ]

        gp = GraphPaths(self.transform_specs)
        serial_graphs_run_info: List[SerialGraphCodeRunInfo] = []
        for leaf_spec in leaf_specs:
            serial_graphs_run_info.extend(
                gp.get_graphs_from_current_spec(
                    leaf_spec, remove_leaf_node=False, ret_partial=True
                )
            )
        return len(serial_graphs_run_info), min_path_cover_dag(self.nx_g)

    @computed_field
    @property
    def num_model_specs(self) -> int:
        return len(self.m_specs)

    def max_cols_for_cvar(self, k=10) -> List[str]:
        ret = []
        for cvar in self.cv_specs.values():
            if (len(cvar.final_columns_orig) + len(cvar.final_columns_derived)) > k:
                ret.append(str(cvar))
        return ret

    @computed_field
    @property
    def cv_spec_names(self) -> List[str]:
        return [str(cvar) for cvar in self.cv_specs.values()]

    @computed_field
    @property
    def transform_spec_names(self) -> List[str]:
        return [
            tspec.specification
            for tspec in self.transform_specs.values()
            if tspec.spec_id != ROOT_SPEC_ID
        ]

    @computed_field
    @property
    def model_spec_names(self) -> List[str]:
        return list(set([mspec.specification for mspec in self.m_specs.values()]))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_model_associated_cvars(self) -> Dict[str, List[ConceptualVarSpec]]:
        mspec_id_to_cvars = defaultdict(list)
        for model_spec in self.m_specs.values():
            for col in model_spec.associated_columns_orig:
                for cvar in self.cv_specs.values():
                    if (
                        col in cvar.final_columns_orig
                        and cvar not in mspec_id_to_cvars[model_spec.spec_id]
                    ):
                        mspec_id_to_cvars[model_spec.spec_id].append(cvar)
            for col in model_spec.associated_columns_derived:
                for cvar in self.cv_specs.values():
                    if (
                        col in cvar.final_columns_derived
                        and cvar not in mspec_id_to_cvars[model_spec.spec_id]
                    ):
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
        self, data_path: str, save_path: str = ".", timeout: int = 10
    ) -> TransformDatasetState:
        # TODO need to test this method
        if self._state_data is None:
            nx_g = self.build_graph_from_specs()

            annotate = AnnotationDataTransforms(
                data_path,
                id_to_spec=self.transform_specs,
                nx_g=nx_g,
                save_path=save_path,
                timeout=timeout,
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
            "num_model_specs": len(self.m_specs),
        }


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
                        Branch(dependencies=b).dict() for b in spec_kwargs["branches"]
                    ]

            specs.append(COL_NAME_TO_CLASS[spec_col](**spec_kwargs))
    else:
        for spec_json_str in df[spec_col].values:
            specs.append(COL_NAME_TO_CLASS[spec_col](**json.loads(spec_json_str)))
    return specs


def get_annotation_data_from_df(df: pd.DataFrame) -> AnnotationDBData:
    df = df.replace(r"^\s*$", np.nan, regex=True)
    transform_specs = get_saved_specs_from_df(df, TRANSFORM_SPEC_COLUMN_NAME)
    model_specs = get_saved_specs_from_df(df, MODEL_SPEC_COLUMN_NAME)
    conceptual_specs = get_saved_specs_from_df(df, CONCEPTUAL_VAR_SPEC_COLUMN_NAME)

    def specs_to_id_to_spec(specs: List[BaseSpec]):
        return {spec.spec_id: spec for spec in specs}

    transform_specs = specs_to_id_to_spec(transform_specs)
    model_specs = specs_to_id_to_spec(model_specs)
    conceptual_specs = specs_to_id_to_spec(conceptual_specs)

    return AnnotationDBData(
        transform_specs=transform_specs,
        m_specs=model_specs,
        cv_specs=conceptual_specs,
    )
