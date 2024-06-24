from collections import defaultdict
import copy
from typing import Dict, FrozenSet, List, Set, Tuple
import networkx as nx
from blade_bench.data.datamodel.graph import SerialGraphCodeRunInfo
from blade_bench.data.datamodel.specs import (
    ROOT_SPEC_ID,
    LEAF_SPEC_ID,
    LEAF_SPEC_NAME,
    Branch,
    TransformSpec,
)
from blade_bench.data.process.transforms.parser.condition import ConditionParser


def replace_spec_id_w_spec_name_in_condition(
    condition: str, id_to_spec: Dict[str, TransformSpec]
):
    for spec_id, spec in id_to_spec.items():
        condition = condition.replace(spec_id, spec.spec_name)
    return condition


class GraphPaths:
    """
    Convert branch dependency specifications into all unique paths of transformations
    """

    def __init__(self, spec_id_to_spec: Dict[str, TransformSpec]):
        self.spec_id_to_spec = spec_id_to_spec

    def get_graphs_from_branches(
        self,
        branches: List[Branch] = [],
    ) -> List[SerialGraphCodeRunInfo]:
        if len(branches) == 0:
            branches = [Branch(dependencies=[ROOT_SPEC_ID])]
        leaf_node = TransformSpec(
            spec_id=LEAF_SPEC_ID,
            spec_name=LEAF_SPEC_NAME,
            branches=branches,
        )
        self.spec_id_to_spec[leaf_node.spec_id] = leaf_node  # add leaf node
        serial_graphs_and_branches = self.get_graphs_from_current_spec(leaf_node)
        self.spec_id_to_spec.pop(LEAF_SPEC_ID)  # remove leaf node
        return serial_graphs_and_branches

    def get_graphs_from_current_spec(
        self,
        leaf_spec: TransformSpec,
        remove_leaf_node: bool = True,
        ret_partial=False,
    ) -> List[SerialGraphCodeRunInfo]:
        graphs = self.get_unserialized_graphs_from_leaf_spec(
            leaf_spec, self.spec_id_to_spec
        )
        serial_graphs: List[SerialGraphCodeRunInfo] = []
        for g in graphs:
            sg, leaf_node = self.serialize_graph(g, ret_partial=ret_partial)
            if sg is not None:
                ind: int = sg.nodes[leaf_node].get("branch_ind", 0)
                if not ret_partial:
                    branch = leaf_spec.branches[ind]
                else:
                    branch = self.spec_id_to_spec[leaf_node].branches[ind]
                    if self.spec_id_to_spec[leaf_node].trans_verb[0] in [
                        "groupby",
                        "join",
                    ]:  # verbs in which we should not have leaf nodes
                        continue

                unserial_nx_g = copy.deepcopy(g).subgraph(sg.nodes)
                if remove_leaf_node:
                    sg.remove_node(leaf_node)
                    sg_code_info = SerialGraphCodeRunInfo(
                        branch=branch, serial_nx_g=sg, unserial_nx_g=unserial_nx_g
                    )
                else:
                    sg_code_info = SerialGraphCodeRunInfo(
                        branch=branch,
                        serial_nx_g=sg,
                        leaf_node_spec_id=leaf_node,
                        unserial_nx_g=unserial_nx_g,
                    )
                serial_graphs.append(sg_code_info)
        return serial_graphs

    def get_valid_graph_and_branches(
        self,
        transform_spec: TransformSpec,
        spec_deps_of_interest: List[str] = [],
    ) -> Tuple[nx.DiGraph, Dict[str, Set[FrozenSet[str]]], Dict[str, str]]:
        """ """

        leaf_node = transform_spec
        graphs = self.get_unserialized_graphs_from_leaf_spec(
            leaf_node, self.spec_id_to_spec
        )
        graphs_ok = [g for g in graphs if self.check_graph(g)]
        conditions = {}
        edge_set = set()
        valid_deps = defaultdict(set)
        for g in graphs_ok:
            edge_set.update(g.edges)
            for n in g.nodes:
                condition = g.nodes[n].get("condition")
                if condition:
                    conditions[n] = condition

            for spec_id in spec_deps_of_interest:
                if spec_id in g.nodes:
                    valid_deps[spec_id].add(frozenset(list(g.predecessors(spec_id))))

        nx_g = nx.DiGraph()
        nx_g.add_edges_from(edge_set)
        return nx_g, valid_deps, conditions

    def get_unserialized_graphs_from_leaf_spec(
        self,
        leaf_node: TransformSpec,
        spec_id_to_spec: Dict[str, TransformSpec] = None,
    ) -> List[nx.DiGraph]:
        """
        Build the (unserialized) graphs from bottom up recursively based on one leaf spec node.
        If encounter a node with multiple alternative depdency branches -> split into new graphs.
        Each graph represents one possible path of transformations.
        """
        # get leaf node
        stack = [leaf_node.spec_id]
        graph = nx.DiGraph()
        graphs: List[nx.DiGraph] = []
        self.__build_graph_recur(
            graph=graph,
            stack=stack,
            spec_id_to_spec=(
                spec_id_to_spec if spec_id_to_spec else self.spec_id_to_spec
            ),
            graphs=graphs,
        )
        return graphs

    def __build_graph_recur(
        self,
        graph: nx.DiGraph,
        stack: List[str],
        spec_id_to_spec: Dict[str, TransformSpec],
        graphs: List[nx.DiGraph],
    ):

        def update_stack_and_graph(
            branch: List[str], graph: nx.DiGraph, stack: List[str]
        ):
            for spec_id in branch:
                if spec_id not in graph.nodes:
                    stack.append(spec_id)
                graph.add_edge(spec_id, node)

        while stack:
            node = stack.pop()
            spec = spec_id_to_spec[node]
            graph.add_node(node)
            if len(spec.branches) > 1:
                cur_graph = graph.copy()
                cur_stack = [*stack]
            for i, branch in enumerate(spec.branches):
                if i == 0:
                    update_stack_and_graph(branch.dependencies, graph, stack)
                    if len(spec.branches) > 1:  # record the branch_ind for later
                        graph.nodes[node]["branch_ind"] = i
                    graph.nodes[node][
                        "condition"
                    ] = branch.condition  # record the condition for later
                else:
                    branch_graph = cur_graph.copy()
                    branch_stack = [*cur_stack]
                    branch_graph.nodes[node]["branch_ind"] = i
                    branch_graph.nodes[node]["condition"] = branch.condition
                    update_stack_and_graph(
                        branch.dependencies, branch_graph, branch_stack
                    )
                    self.__build_graph_recur(
                        branch_graph,
                        branch_stack,
                        spec_id_to_spec,
                        graphs,
                    )

        graphs.append(graph)

    def __check_condition(
        self,
        condition: str,
        history: Dict[
            str, Tuple[List[str], int]
        ],  # value is a tuple of dependencies and branch_ind
    ):

        cond_w_col_name = replace_spec_id_w_spec_name_in_condition(
            condition, self.spec_id_to_spec
        )
        condition_for_eval, msg_parse = ConditionParser(
            cond_w_col_name, history.keys()
        ).parse_and_recon()
        is_ok, msg = ConditionParser.eval_constraints_on_branch(
            condition_for_eval, history
        )
        if msg_parse and "not a valid transform spec name" in msg_parse:
            is_ok = True
        return is_ok

    def check_graph(self, nx_g: nx.DiGraph, ret_partial=False):
        """
        Check if the graph is valid
        """
        id_to_spec = self.spec_id_to_spec
        nodes = nx.topological_sort(nx_g)
        history = {}
        for n in nodes:
            condition = nx_g.nodes[n].get("condition")
            if condition and id_to_spec is not None:
                if not self.__check_condition(condition, history):
                    return False
            branch_ind = nx_g.nodes[n].get("branch_ind")
            if branch_ind is not None and id_to_spec is not None:
                branch_ind: int
                history[id_to_spec[n].spec_name] = (
                    [
                        id_to_spec[d].spec_name
                        for d in id_to_spec[n].branches[branch_ind].dependencies
                    ],
                    branch_ind,
                )
        return True

    def get_node_below_shared_ancestor(
        self, shared_ancestor: str, next_parent_node: str, nx_g: nx.DiGraph
    ):
        node_below_ancestor = [
            succ
            for succ in nx_g.successors(shared_ancestor)
            if nx.has_path(nx_g, succ, next_parent_node)
        ]
        assert len(node_below_ancestor) == 1
        return node_below_ancestor[0]

    def reconnect_nodes_for_serilization(
        self, cur_node: str, nx_g: nx.DiGraph, in_edges: List[Tuple[str, str]]
    ):
        id_to_spec = self.spec_id_to_spec
        for i, edge in enumerate(in_edges):
            cur_parent_node = edge[0]
            if id_to_spec is not None:
                nx_g.nodes[cur_parent_node]["code"] = id_to_spec[cur_parent_node].code
                # convert_last_line(
                # id_to_spec[cur_parent_node].code,
                # suffix=f"_{id_to_spec[cur_parent_node].spec_name}",
                # )
            if i + 1 == len(in_edges):
                break

            next_parent_node = in_edges[i + 1][
                0
            ]  # next node that is also a parent of n
            nx_g.remove_edge(cur_parent_node, cur_node)
            shared_ancestor = nx.lowest_common_ancestor(
                nx_g, cur_parent_node, next_parent_node
            )
            # shared_ancestor_spec = self.spec_id_to_spec[shared_ancestor]
            node_below_ancestor = self.get_node_below_shared_ancestor(
                shared_ancestor, next_parent_node, nx_g
            )
            nx_g.remove_edge(shared_ancestor, node_below_ancestor)
            nx_g.add_edge(cur_parent_node, node_below_ancestor)
        return nx_g

    def serialize_graph_all_permutations(self, nx_g: nx.DiGraph):
        """
        gives all valid permutations of paths in the graph
        nx_g: unserialized graph with conditions and branch spec to all valid paths
        """
        all_valid_paths = list(nx.all_topological_sorts(nx_g))
        ret = []
        id_to_spec = self.spec_id_to_spec
        for path in all_valid_paths:
            history = {}
            satisfy = True
            for n in path:
                # if a parent is groupby

                condition = nx_g.nodes[n].get("condition")
                if condition and self.spec_id_to_spec is not None:
                    if not self.__check_condition(condition, history):
                        satisfy = False
                        break

                branch_ind = nx_g.nodes[n].get("branch_ind")
                if branch_ind is not None and id_to_spec is not None:
                    history[id_to_spec[n].spec_name] = (
                        [
                            id_to_spec[d].spec_name
                            for d in id_to_spec[n].branches[branch_ind].dependencies
                        ],
                        branch_ind,
                    )
            if satisfy:
                ret.append(path)
        return ret

        # get all possible permutations of incoming edges

    def serialize_graph(self, nx_g: nx.DiGraph, ret_partial=False):
        """
        Given a graph processed by get_graphs_from_branch_specs, serialize the graph into one single path.
        Sometimes a node may have multiple incoming edges as it depends on multiple transforms. =
        Returns:
            nx.DiGraph: serialized graph
            str: leaf node
        """

        id_to_spec = self.spec_id_to_spec
        nodes = list(nx.topological_sort(nx_g))
        history = {}
        leaf_node = None
        for i, n in enumerate(nodes):
            nx_g.nodes[n]["code"] = id_to_spec[n].code
            nx_g.nodes[n]["spec_name"] = id_to_spec[n].spec_name

            condition = nx_g.nodes[n].get("condition")
            if condition and id_to_spec is not None:
                if not self.__check_condition(condition, history):
                    if ret_partial:
                        subgraph = copy.deepcopy(nx_g).subgraph(nodes[:i])
                        return self.serialize_graph(subgraph, ret_partial=True)
                    else:
                        return None, None
            branch_ind = nx_g.nodes[n].get("branch_ind")
            if branch_ind is not None and id_to_spec is not None:
                history[id_to_spec[n].spec_name] = (
                    [
                        id_to_spec[d].spec_name
                        for d in id_to_spec[n].branches[branch_ind].dependencies
                    ],
                    branch_ind,
                )
            if nx_g.in_degree(n) > 1:  # when there are multiple incoming edges
                in_edges = list(sorted(nx_g.in_edges(n)))
                nx_g = self.reconnect_nodes_for_serilization(n, nx_g, in_edges)

            if nx_g.out_degree(n) == 0:
                leaf_node = n
        return nx_g, leaf_node
