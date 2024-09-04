import asyncio
from collections import defaultdict
from typing import Dict, List, Tuple
import pandas as pd
import networkx as nx
from blade_bench.data.datamodel.specs import POST_GROUPBY_TRANS_VERB
from blade_bench.data.datamodel.transforms import GraphHashInfo
from blade_bench.parse_code import process_groupby_code
from blade_bench.utils import timeout
from .graph_paths import GraphPaths


from blade_bench.nb import (
    NotebookExecutorBasic,
)
from blade_bench.data.datamodel import (
    Branch,
    TransformSpec,
    ROOT_SPEC_ID,
    ROOT_SPEC_NAME,
    TransformDataReturn,
    TransformDatasetState,
    SingleColState,
    TransformState,
    PathInfo,
    ExpandedGraph,
)
from ..load import get_saved_specs_from_df

from .expand_graph import ProcessGraph

from blade_bench.nb.funcs import (
    get_code_execeution_output,
    get_transform_state,
)


class AnnotationDataTransforms:
    def __init__(
        self,
        dataset_path: str = None,
        id_to_spec: Dict[str, TransformSpec] = None,
        nx_g: nx.DiGraph = None,
        run_nb: bool = True,
        save_path: str = ".",
        data_columns: List[str] = None,
        timeout: int = 10,
    ):
        if dataset_path is not None:
            df = pd.read_csv(dataset_path)
            self.data_columns = list(df.columns)
        elif data_columns is not None:
            self.data_columns = data_columns
        else:
            raise ValueError("Either dataset_path or data_columns must be provided")

        self.timeout = timeout
        self.id_to_spec = id_to_spec
        self.nx_g = nx_g if nx_g is not None else self.build_graph_from_specs()
        if run_nb:
            self.nb_executor: NotebookExecutorBasic = NotebookExecutorBasic(
                data_path=dataset_path, save_path=save_path, run_init_once=True
            )
        else:
            self.nb_executor = None
        self.dataset_path = dataset_path

    def build_graph_from_specs(self):
        if self.id_to_spec is None:
            return None
        nx_g = nx.DiGraph()
        for spec in self.id_to_spec.values():
            nx_g.add_node(spec.spec_id)
            for branch in spec.branches:
                for dep in branch.dependencies:
                    nx_g.add_edge(dep, spec.spec_id)
        return nx_g

    def get_leaf_specs(
        self, nx_g: nx.DiGraph, id_to_spec: Dict[str, TransformSpec]
    ) -> List[TransformSpec]:
        leaf_specs = [
            id_to_spec[spec_id]
            for spec_id in nx_g.nodes
            if nx_g.out_degree(spec_id) == 0 and nx_g.in_degree(spec_id) != 0
        ]
        return leaf_specs

    async def build_state_data(
        self,
        id_to_spec: Dict[str, TransformSpec] = None,
        leaf_specs: List[TransformSpec] = None,
        id_to_ts: Dict[str, TransformState] = None,
        save_ts: bool = False,
    ) -> TransformDatasetState:
        if id_to_spec is None:
            id_to_spec = self.id_to_spec
        if leaf_specs is None:
            leaf_specs = self.get_leaf_specs(self.nx_g, self.id_to_spec)

        all_expanded_id_to_spec = {}
        all_graph_hashes = defaultdict(list)
        all_value_hashes = defaultdict(list)
        all_categorical_hashes = defaultdict(list)
        col_graphs = {}
        all_graphs = self.get_path_graphs_from_leaf_specs(leaf_specs, id_to_spec)
        # we get graph hashes and values for every unique graph
        for g in all_graphs:
            if len(all_graphs) == 1 and id_to_ts is not None:
                expanded_id_to_spec, path_info = self.process_path_g_with_states(
                    g, id_to_spec, id_to_ts, all_graph_hashes, save_ts
                )
            else:
                expanded_id_to_spec, path_info = await self.process_path_g(
                    g, id_to_spec, all_graph_hashes, save_ts
                )
            all_expanded_id_to_spec.update(expanded_id_to_spec)
            ind = len(col_graphs)
            for k, v in path_info.nid_to_col_state.items():
                v.cols_graph_id = ind
                all_value_hashes[v.value_hash].append(v)
                all_categorical_hashes[v.categorical_value_hash].append(v)

            path_info.cols_gid = ind
            col_graphs[ind] = path_info

        graph_hashes = {
            k: [GraphHashInfo(col_states=s[1], graph=s[0])]
            for k, v in all_graph_hashes.items()
            for s in v
        }

        return TransformDatasetState(
            id_to_spec=id_to_spec,
            expanded_id_to_spec=all_expanded_id_to_spec,
            graph_hashes=graph_hashes,
            value_hashes=all_value_hashes,
            categorical_value_hashes=all_categorical_hashes,
            graphs=col_graphs,
        )

    async def build_state_data_from_transform_return(
        self, transform_returns: List[TransformDataReturn]
    ) -> TransformDatasetState:
        ts_specs = [
            TransformSpec(
                spec_id=ROOT_SPEC_ID,
                spec_name=ROOT_SPEC_NAME,
            )
        ]
        id_to_ts: Dict[str, TransformState] = {}

        for i, transform_return in enumerate(transform_returns):
            if (
                transform_return.transform_verb == "groupby"
                or transform_return.groupby_cols
            ):
                transform_return.transform_verb = "groupby"
                ts_spec = TransformSpec(
                    spec_id=f"node_{i}_{transform_return.transform_verb}",
                    spec_name=f"node_{i}_{transform_return.transform_verb}",
                    trans_verb=[transform_return.transform_verb],
                    input_cols_to_output_col_mapping=[
                        (list(transform_return.groupby_cols), "")
                    ],
                    code="",
                    branches=[Branch(dependencies=[ts_specs[-1].spec_id])],
                )
                ts_specs.append(ts_spec)
                ts_spec_post_grpby = TransformSpec(
                    spec_id=f"node_{i}_{POST_GROUPBY_TRANS_VERB}",
                    spec_name=f"node_{i}_{POST_GROUPBY_TRANS_VERB}",
                    trans_verb=[POST_GROUPBY_TRANS_VERB],
                    input_cols_to_output_col_mapping=[
                        (list(inp_cols), output_col if output_col != "ALL" else "")
                        for inp_cols, output_col in transform_return.column_mapping.items()
                    ],
                    branches=[Branch(dependencies=[ts_specs[-1].spec_id])],
                    code=transform_return.code,
                )
                ts_specs.append(ts_spec_post_grpby)
                id_to_ts[ts_spec_post_grpby.spec_id] = get_transform_state(
                    ts_spec_post_grpby.spec_id,
                    transform_return.df,
                    list(transform_return.df.columns),
                    ts_spec_post_grpby.spec_name,
                )
            else:
                if transform_return.transform_verb != "filter":
                    input_output_col_map_set = set()
                    for inp_cols, output_col in transform_return.column_mapping.items():
                        if inp_cols == "ALL":
                            continue
                        if output_col == "ALL":
                            for col in transform_return.df.columns:
                                input_output_col_map_set.add((frozenset(inp_cols), col))
                        else:
                            input_output_col_map_set.add(
                                (frozenset(inp_cols), output_col)
                            )
                    input_output_col_map = [
                        (list(inp_cols), output_col)
                        for inp_cols, output_col in input_output_col_map_set
                    ]
                else:
                    input_output_col_map = [
                        (list(inp_cols), output_col if output_col != "ALL" else "")
                        for inp_cols, output_col in transform_return.column_mapping.items()
                        if inp_cols != "ALL"
                    ]

                ts_spec = TransformSpec(
                    spec_id=f"node_{i}_{transform_return.transform_verb}",
                    spec_name=f"node_{i}_{transform_return.transform_verb}",
                    trans_verb=[transform_return.transform_verb],
                    input_cols_to_output_col_mapping=input_output_col_map,
                    branches=[Branch(dependencies=[ts_specs[-1].spec_id])],
                    code=transform_return.code,
                )
                ts_specs.append(ts_spec)
                id_to_ts[ts_spec.spec_id] = get_transform_state(
                    ts_spec.spec_id,
                    transform_return.df,
                    list(transform_return.df.columns),
                    ts_spec.spec_name,
                )

        id_to_spec = {spec.spec_id: spec for spec in ts_specs}
        leaf_specs = [ts_specs[-1]]

        return await self.build_state_data(
            id_to_spec, leaf_specs, id_to_ts=id_to_ts, save_ts=True
        )

    def get_path_graphs_from_leaf_specs(
        self,
        leaf_specs: List[TransformSpec],
        id_to_spec: Dict[str, TransformSpec],
    ) -> List[nx.DiGraph]:
        graph_paths = GraphPaths(id_to_spec)
        all_graphs = []
        for leaf_spec in leaf_specs:
            graphs_info = graph_paths.get_graphs_from_current_spec(
                leaf_spec, remove_leaf_node=False, ret_partial=True
            )
            all_graphs.extend([g_info.unserial_nx_g for g_info in graphs_info])
        return all_graphs

    async def build_state_data_from_path_specs(
        self, path_specs: List[TransformSpec], save_ts: bool = False
    ) -> TransformDatasetState:
        leaf_specs = [path_specs[-1]]
        id_to_spec = {spec.spec_id: spec for spec in path_specs}
        return await self.build_state_data(
            id_to_spec=id_to_spec, leaf_specs=leaf_specs, save_ts=save_ts
        )

    async def relabel_graph(
        self, nx_g: nx.DiGraph, expanded_id_to_spec: Dict[str, TransformSpec]
    ):
        mapping = {}
        for node in nx_g.nodes:
            spec_id = nx_g.nodes[node]["expanded_spec_id"]
            mapping[node] = (
                node.split("__")[0] + "__" + expanded_id_to_spec[spec_id].spec_name
            )
        return nx.relabel_nodes(nx_g, mapping)

    async def process_path_g(
        self,
        g: nx.DiGraph,
        id_to_spec: Dict[str, TransformSpec],
        graph_hash_and_graphs: Dict[str, List[Tuple[nx.DiGraph, List[SingleColState]]]],
        save_ts: bool = False,
    ):

        spec_id_to_ts = await self.__get_transform_states(g, id_to_spec)
        return self.process_path_g_with_states(
            g, id_to_spec, spec_id_to_ts, graph_hash_and_graphs, save_ts
        )

    def process_path_g_with_states(
        self,
        g: nx.DiGraph,
        id_to_spec: Dict[str, TransformSpec],
        spec_id_to_ts: Dict[str, TransformState],
        graph_hash_and_graphs: Dict[str, List[Tuple[nx.DiGraph, List[SingleColState]]]],
        save_ts: bool = False,
    ):
        process_g = ProcessGraph()
        ex_g: ExpandedGraph = process_g.get_expand_g(
            nx_g=g,
            id_to_spec=id_to_spec,
            orig_data_cols=self.data_columns,
            id_to_ts=spec_id_to_ts,
        )

        cols_g = process_g.get_col_g(
            ex_g
        )  # can search on this graph for a particular expanded_spec_id,

        cols_g_w_input = process_g.get_col_g(ex_g, add_input_edges=True)

        path_info = PathInfo(
            path_graph=list(nx.topological_sort(g)),
            col_graph=cols_g,
            col_graph_w_input=cols_g_w_input,
            nid_to_col_state=cols_g.nid_to_state,
            spec_id_to_ts=spec_id_to_ts if save_ts else {},
        )

        self.__graph_hash_from_cols_g(
            cols_g_w_input.nx_g,
            cols_g.nid_to_state,
            graph_hash_and_graphs,
        )

        return ex_g.id_to_spec, path_info

    async def __get_transform_states(
        self,
        nx_g: nx.DiGraph,
        id_to_spec: Dict[str, TransformSpec],
    ) -> Dict[str, TransformState]:
        """
        nx_g: unserialized graph representing one alternative path of transforms
        """
        cur_code = ""
        ProcessGraph().update_post_groupby_transverb(nx_g, id_to_spec)
        id_to_ts = {}
        path = nx.topological_sort(nx_g)
        self.nb_executor.run_init_once = True
        self.nb_executor.data_path = self.dataset_path
        for node in path:
            cur_spec = id_to_spec[node]
            if not cur_spec.code:
                continue
            code = cur_code + cur_spec.code
            task = asyncio.create_task(
                get_code_execeution_output(
                    code, node, self.nb_executor, spec_name=cur_spec.spec_name
                )
            )
            transform_state: TransformState = await task
            if transform_state is not None:
                id_to_ts[node] = transform_state
            cur_code += f"{process_groupby_code(cur_spec.code)}\n"
        return id_to_ts

    def __graph_hash_from_cols_g(
        self,
        cols_g: nx.DiGraph,
        nid_to_col_state: Dict[str, SingleColState],
        graph_hash_and_graphs: Dict[
            str, List[Tuple[nx.DiGraph, List[SingleColState]]]
        ] = None,
    ):
        """
        Add graph hashes to the graph_hash_and_graphs dict
        """

        @timeout(seconds=self.timeout, default=False)
        def is_isomorphic(g, subgraph):
            return nx.is_isomorphic(
                g,
                subgraph,
                edge_match=lambda x, y: x["type"] == y["type"],
                node_match=lambda x, y: x["val"] == y["val"],
            )

        for node in cols_g.nodes:
            if (
                node not in nid_to_col_state
            ):  # does not get states for groupby for example
                continue
            nodes = nx.ancestors(cols_g, node)
            all_nodes = nodes.union({node})
            subgraph = nx.subgraph(cols_g, all_nodes).copy()
            graph_hash = nx.weisfeiler_lehman_graph_hash(
                subgraph, edge_attr="type", node_attr="val"
            )
            col_state = nid_to_col_state[node]
            col_state.graph_hash = graph_hash
            if graph_hash not in graph_hash_and_graphs:
                graph_hash_and_graphs[graph_hash].append((subgraph, [col_state]))
            else:
                matched = False
                for g, col_states in graph_hash_and_graphs[graph_hash]:
                    if is_isomorphic(g, subgraph):
                        col_states.append(col_state)
                        matched = True
                        break
                if not matched:
                    graph_hash_and_graphs[graph_hash].append((subgraph, [col_state]))

    @classmethod
    def init_from_csv(cls, annotation_path: str, dataset_path: str = None):
        df = pd.read_csv(annotation_path)
        transform_specs = get_saved_specs_from_df(df, "transform_spec_json")
        id_to_spec = {spec.spec_id: spec for spec in transform_specs}
        return cls(dataset_path=dataset_path, id_to_spec=id_to_spec)


if __name__ == "__main__":
    pass
