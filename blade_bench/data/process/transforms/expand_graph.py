from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import networkx as nx


from ...datamodel import (
    ROOT_SPEC_ID,
    POST_GROUPBY_TRANS_VERB,
    VERBS_AFFECTING_WHOLE_DF,
    SingleColState,
    TransformState,
    ColsGraph,
    ExpandedGraph,
    TransformSpec,
    Branch,
)


class ProcessGraph:
    def get_expand_g(
        self,
        nx_g: nx.DiGraph,
        id_to_spec: Dict[str, TransformSpec],
        orig_data_cols: List[str],
        id_to_ts: Dict[str, TransformState] = {},
    ) -> ExpandedGraph:
        """
        Expand specs (nodes) with multiple input column and output column mappings into
        a graph with one column mapping for each spec
        """
        nodes = list(nx.topological_sort(nx_g))
        expanded_g = nx.DiGraph()
        expanded_g.add_node(nodes[0])
        col_to_spec_mapping = {col: nodes[0] for col in orig_data_cols}
        expanded_id_to_spec = {nodes[0]: id_to_spec[nodes[0]]}
        expanded_id_to_ts = {}
        expanded_g.nodes[nodes[0]]["data_output_cols"] = set(orig_data_cols)

        for n in nodes[1:]:
            spec = id_to_spec[n]
            verb = spec.trans_verb[0]
            ts_state: TransformState = id_to_ts.get(n)
            # when the operation opereates on the entire columns of the graph
            if verb == "groupby" or verb == "filter":
                leaf_nodes = [
                    new_n
                    for new_n in expanded_g.nodes
                    if expanded_g.out_degree(new_n) == 0
                ]
                # incoming edges for each leaf node
                data_out_cols = set(orig_data_cols)
                for leaf_node in leaf_nodes:
                    expanded_g.add_edge(leaf_node, n)
                    data_out_cols.update(
                        expanded_g.nodes[leaf_node]["data_output_cols"]
                    )

                if verb == "groupby":
                    for col in spec.input_cols:
                        data_out_cols.remove(col)
                else:
                    for col in data_out_cols:
                        col_to_spec_mapping[col] = n
                if ts_state:
                    data_out_cols = {c for c in data_out_cols if c in ts_state.columns}
                    expanded_id_to_ts[n] = ts_state.get_transform_state_from_cols(
                        spec.spec_id, list(data_out_cols)
                    )
                expanded_id_to_spec[n] = spec
                expanded_g.nodes[n]["data_output_cols"] = data_out_cols

            elif verb == POST_GROUPBY_TRANS_VERB:
                parents = nx_g.predecessors(n)
                input_output_col_mapping = spec.input_cols_to_output_col_mapping
                for parent in parents:
                    for input_cols, output_col in input_output_col_mapping:
                        cur_node = f"{POST_GROUPBY_TRANS_VERB}_{output_col}"
                        new_spec = TransformSpec(
                            spec_id=f"{spec.spec_id}__{cur_node}",
                            spec_name=cur_node,
                            branches=[Branch(dependencies=[parent])],
                            input_cols_to_output_col_mapping=[(input_cols, output_col)],
                            trans_verb=[POST_GROUPBY_TRANS_VERB],
                            code=spec.code,
                        )
                        expanded_g.add_edge(parent, new_spec.spec_id)
                        col_to_spec_mapping[output_col] = new_spec.spec_id
                        expanded_id_to_spec[new_spec.spec_id] = new_spec
                        expanded_g.nodes[new_spec.spec_id]["data_output_cols"] = set(
                            [output_col]
                        )
                        if ts_state:
                            expanded_id_to_ts[new_spec.spec_id] = (
                                ts_state.get_transform_state_from_cols(
                                    new_spec.spec_id, [output_col]
                                )
                            )
            else:
                parents = list(nx_g.predecessors(n))
                if (
                    len(parents) == 1
                    and parents[0] != ROOT_SPEC_ID
                    and id_to_spec[parents[0]].trans_verb[0] == "filter"
                ):
                    parent = parents[0]
                    for (
                        input_cols,
                        output_col,
                    ) in spec.input_cols_to_output_col_mapping:
                        cur_node = f"{verb}_{output_col}"
                        new_spec = TransformSpec(
                            spec_id=f"{spec.spec_id}__{cur_node}",
                            spec_name=cur_node,
                            branches=[Branch(dependencies=[parent])],
                            input_cols_to_output_col_mapping=[(input_cols, output_col)],
                            trans_verb=[verb],
                            code=spec.code,
                        )
                        expanded_g.add_edge(
                            parent, new_spec.spec_id
                        )  # add an edge from the filter to the current node rather than the parent specified
                        col_to_spec_mapping[output_col] = new_spec.spec_id
                        expanded_id_to_spec[new_spec.spec_id] = new_spec

                        data_out_cols = expanded_g.nodes[parent]["data_output_cols"]
                        if ts_state:
                            data_out_cols = {
                                c for c in data_out_cols if c in ts_state.columns
                            }
                            expanded_id_to_ts[new_spec.spec_id] = (
                                ts_state.get_transform_state_from_cols(
                                    new_spec.spec_id, [output_col]
                                )
                            )
                        expanded_g.nodes[new_spec.spec_id]["data_output_cols"] = set(
                            [output_col]
                        ).union(data_out_cols)
                else:
                    for input_cols, output_col in spec.input_cols_to_output_col_mapping:
                        cur_node = f"{verb}_{output_col}"
                        if output_col == "":
                            output_col = input_cols[0]
                        deps = [
                            col_to_spec_mapping[input_col] for input_col in input_cols
                        ]
                        new_spec = TransformSpec(
                            spec_id=f"{spec.spec_id}__{cur_node}",
                            spec_name=cur_node,
                            branches=[Branch(dependencies=deps)],
                            input_cols_to_output_col_mapping=[(input_cols, output_col)],
                            trans_verb=[verb],
                            code=spec.code,
                        )
                        data_out_cols = set()
                        for input_col in input_cols:
                            parent = col_to_spec_mapping[input_col]
                            expanded_g.add_edge(parent, new_spec.spec_id)
                            data_out_cols.update(
                                expanded_g.nodes[parent]["data_output_cols"]
                            )
                        data_out_cols.add(output_col)
                        col_to_spec_mapping[output_col] = new_spec.spec_id
                        expanded_id_to_spec[new_spec.spec_id] = new_spec

                        if ts_state:
                            data_out_cols = {
                                c for c in data_out_cols if c in ts_state.columns
                            }
                            expanded_id_to_ts[new_spec.spec_id] = (
                                ts_state.get_transform_state_from_cols(
                                    new_spec.spec_id, [output_col]
                                )
                            )
                        expanded_g.nodes[new_spec.spec_id][
                            "data_output_cols"
                        ] = data_out_cols

        return ExpandedGraph(
            nx_g=expanded_g, id_to_spec=expanded_id_to_spec, id_to_ts=expanded_id_to_ts
        )

    def _get_parents(self, ex_g: ExpandedGraph, n: str):
        return list(
            reversed(
                list(
                    nx.topological_sort(ex_g.nx_g.subgraph(nx.ancestors(ex_g.nx_g, n)))
                )
            )
        )

    def _add_node_to_cols_g_new(
        self,
        ex_g: ExpandedGraph,
        cols_g: nx.DiGraph,
        ex_g_to_cols_g_nodes: Dict[
            str, Dict[str, str]
        ],  # expanded node id to dict of column name to column graph node
        nid_to_state: Dict[str, SingleColState],
        parent_nodes: List[str],
        spec: TransformSpec,
        col: str,
    ):
        cols_added = set()
        for parent_node in parent_nodes:
            if (
                ex_g.id_to_spec[parent_node].trans_verb
                and ex_g.id_to_spec[parent_node].trans_verb[0] == "groupby"
            ):
                break
            if col in ex_g_to_cols_g_nodes[parent_node].keys():
                new_nid = f"{col}__{spec.spec_id}"
                if new_nid in cols_g.nodes:
                    continue
                if col in cols_added:
                    continue
                cols_added.add(col)
                cols_g.add_node(
                    new_nid,
                    col=col,
                    expanded_spec_id=spec.spec_id,
                    expanded_spec_name=spec.spec_name,
                    val=spec.trans_verb[0],
                )
                cols_g.add_edge(
                    ex_g_to_cols_g_nodes[parent_node][col], new_nid, type="data"
                )
                ex_g_to_cols_g_nodes[spec.spec_id][col] = new_nid
                if ex_g.id_to_ts and spec.spec_id in ex_g.id_to_ts:
                    ts_state = ex_g.id_to_ts[spec.spec_id]
                    if ts_state.df is not None:
                        single_col_state = ts_state.get_single_state_from_col(
                            col, new_nid, spec.code
                        )
                        nid_to_state[new_nid] = single_col_state

            if (
                ex_g.id_to_spec[parent_node].trans_verb
                and ex_g.id_to_spec[parent_node].trans_verb[0]
                in VERBS_AFFECTING_WHOLE_DF
            ):
                break

    def get_col_g(
        self,
        ex_g: ExpandedGraph,
        add_input_edges: bool = False,
    ) -> ColsGraph:
        """
        Convert the expanded unserialized graph to a column graph

        add_input_edges: if True, add edges for the input columns to filter and groupby nodes
        """
        nodes = list(nx.topological_sort(ex_g.nx_g))
        node_order = {node: i for i, node in enumerate(nx.topological_sort(ex_g.nx_g))}
        cols_g = nx.DiGraph()
        root_node_spec_id = nodes[0]

        cols_last_node = {}
        nid_to_state: Dict[str, SingleColState] = {}
        ex_g_to_cols_g_nodes = defaultdict(dict)
        for data_col in ex_g.nx_g.nodes[root_node_spec_id]["data_output_cols"]:
            nid = f"{data_col}__{root_node_spec_id}"
            cols_g.add_node(
                nid,
                col=data_col,
                expanded_spec_id=root_node_spec_id,
                expanded_spec_name=ex_g.id_to_spec[root_node_spec_id].spec_name,
                val=nid,
            )
            ex_g_to_cols_g_nodes[root_node_spec_id][data_col] = nid
            cols_last_node[data_col] = nid

        for n in nodes[1:]:
            spec = ex_g.id_to_spec[n]
            verb = spec.trans_verb[0]
            if verb == "groupby" or verb == "filter":
                # new_cols_last_node = {**cols_last_node}
                parents = self._get_parents(ex_g, n)
                for data_col in ex_g.nx_g.nodes[n]["data_output_cols"]:
                    self._add_node_to_cols_g_new(
                        ex_g,
                        cols_g,
                        ex_g_to_cols_g_nodes,
                        nid_to_state,
                        parents,
                        spec,
                        data_col,
                    )
                if add_input_edges:
                    all_cols = {
                        col
                        for parent in ex_g.nx_g.predecessors(n)
                        for col in ex_g_to_cols_g_nodes[parent]
                    }
                    for parent in ex_g.nx_g.predecessors(n):
                        for col, nid in ex_g_to_cols_g_nodes[parent].items():
                            if col in spec.input_cols:
                                for out_col in all_cols:
                                    if (
                                        nid,
                                        f"{out_col}__{spec.spec_id}",
                                    ) not in cols_g.edges:
                                        cols_g.add_edge(
                                            nid,
                                            f"{out_col}__{spec.spec_id}",
                                            type="input",
                                        )
                                    else:
                                        cols_g.add_edge(
                                            nid,
                                            f"{out_col}__{spec.spec_id}",
                                            type="input + data",
                                        )
            else:
                inp_cols, output_col = spec.input_cols_to_output_col_mapping[0]
                nid = f"{output_col}__{spec.spec_id}"
                cols_g.add_node(
                    nid,
                    col=output_col,
                    expanded_spec_id=spec.spec_id,
                    expanded_spec_name=spec.spec_name,
                    val=verb,
                )
                parents = list(
                    sorted(ex_g.nx_g.predecessors(n), key=lambda x: -node_order[x])
                )
                for col in inp_cols:
                    for parent in parents:
                        if col in ex_g_to_cols_g_nodes[parent]:
                            parent_nid = ex_g_to_cols_g_nodes[parent][col]
                            cols_g.add_edge(parent_nid, nid, type="data")
                            break

                ex_g_to_cols_g_nodes[spec.spec_id][output_col] = nid
                if ex_g.id_to_ts:
                    single_col_state = ex_g.id_to_ts[
                        spec.spec_id
                    ].get_single_state_from_col(output_col, nid, spec.code)
                    nid_to_state[nid] = single_col_state
        return ColsGraph(nx_g=cols_g, nid_to_state=nid_to_state)

    def update_post_groupby_transverb(
        self, nx_g: nx.DiGraph, id_to_spec: Dict[str, TransformSpec]
    ):
        """
        updates the transform verb of any spec after a groupby to "post_groupby"
        updates id_to_spec in place
        nx_g: unserialized graph with conditions and branch spec to all valid paths
        """
        for n in nx_g.nodes:
            if any(
                [
                    id_to_spec[parent].trans_verb[0] == "groupby"
                    for parent in nx_g.predecessors(n)
                    if id_to_spec[parent].trans_verb
                ]
            ):
                id_to_spec[n].trans_verb = [POST_GROUPBY_TRANS_VERB]

    def get_cols_g_from_unexpanded_g(
        self,
        unexpanded_g: nx.DiGraph,
        orig_data_cols: List[str],
        id_to_spec: Dict[str, TransformSpec],
        id_to_ts: Dict[str, TransformState] = {},
    ):
        ex_g: ExpandedGraph = self.get_expand_g(
            nx_g=unexpanded_g,
            id_to_spec=id_to_spec,
            orig_data_cols=orig_data_cols,
            id_to_ts=id_to_ts,
        )

        cols_g = self.get_col_g(ex_g)
        return cols_g
