from typing import Any, Callable, Set, List, Tuple
import networkx as nx

from blade_bench.data.datamodel import (
    TransformSpec,
    TransformDatasetState,
    SingleColState,
)
from blade_bench.data.annotation import AnnotationDBData
from .base import BaseMatcher
from ..datamodel import (
    MatchTransforms,
    MatchedTSpecs,
)


class TransformMatcher(BaseMatcher):

    def __init__(self, data_path: str = "", dataset_name: str = ""):
        if dataset_name:
            super().__init__(dataset_name)
        else:
            assert data_path, "Either dataset_path or dataset_name must be provided"
            self.dataset_path = data_path
            self.dataset_name = "none"

    async def match_annotator_data(
        self, adata1: AnnotationDBData, adata2: AnnotationDBData
    ):
        ts1 = await adata1.get_state_data(self.dataset_path)
        ts2 = await adata2.get_state_data(self.dataset_path)
        matched_tspecs: MatchedTSpecs = self.match_w_tsdata(ts1, ts2)
        return MatchTransforms(
            transform_state1=ts1,
            transform_state2=ts2,
            matched_tspecs=matched_tspecs,
        )

    async def match_with_llm(
        self, adata: AnnotationDBData, llm: TransformDatasetState
    ) -> MatchTransforms:
        gnd_truth = await adata.get_state_data(self.dataset_path)
        matched_tspecs: MatchedTSpecs = self.match_w_tsdata(gnd_truth, llm)
        return MatchTransforms(
            transform_state1=gnd_truth,
            transform_state2=llm,
            matched_tspecs=matched_tspecs,
        )

    def match_w_tsdata(
        self, ts1: TransformDatasetState, ts2: TransformDatasetState
    ) -> MatchedTSpecs:
        matching_value_hashes, matching_graph_hashes, matching_categorical_hashes = (
            self.find_matching_hashes(ts1, ts2)
        )
        matched1_vspec_ids, matched2_vspec_ids = self.match_specs_from_v_hashes(
            matching_value_hashes, ts1, ts2
        )
        vspecs1 = self.get_specs_from_ids(matched1_vspec_ids, ts1)
        vspecs2 = self.get_specs_from_ids(matched2_vspec_ids, ts2)

        matched1_cat_spec_ids, matched2_cat_spec_ids = self.match_specs_from_v_hashes(
            matching_categorical_hashes, ts1, ts2, get_categorical=True
        )
        cat_specs1 = self.get_specs_from_ids(matched1_cat_spec_ids, ts1)
        cat_specs2 = self.get_specs_from_ids(matched2_cat_spec_ids, ts2)

        matched1_gspec_ids, matched2_gspec_ids = self.match_specs_from_g_hashes(
            matching_graph_hashes, ts1, ts2
        )

        gspecs1 = self.get_specs_from_ids(matched1_gspec_ids, ts1)
        gspecs2 = self.get_specs_from_ids(matched2_gspec_ids, ts2)
        return MatchedTSpecs(
            vspecs1=vspecs1,
            vspecs2=vspecs2,
            gspecs1=gspecs1,
            gspecs2=gspecs2,
            cat_specs1=cat_specs1,
            cat_specs2=cat_specs2,
        )

    def find_matching_hashes(
        self, ts1: TransformDatasetState, ts2: TransformDatasetState
    ):
        """
        Find the matching hashes between two transform datasets.
        """

        value_hash1 = set(ts1.value_hashes.keys())
        value_hash2 = set(ts2.value_hashes.keys())
        matching_value_hashes = value_hash1.intersection(value_hash2)

        graph_hash1 = set(ts1.graph_hashes.keys())
        graph_hash2 = set(ts2.graph_hashes.keys())
        matching_graph_hashes = graph_hash1.intersection(graph_hash2)

        category_hash1 = set(ts1.categorical_value_hashes.keys())
        category_hash2 = set(ts2.categorical_value_hashes.keys())
        matching_categorical_hashes = category_hash1.intersection(category_hash2)
        return matching_value_hashes, matching_graph_hashes, matching_categorical_hashes

    def get_specs_from_ids(
        self, spec_ids, state_data: TransformDatasetState
    ) -> List[TransformSpec]:
        return [state_data.expanded_id_to_spec[spec_id] for spec_id in spec_ids]

    def match_specs_from_v_hashes(
        self,
        matching_value_hashes: Set[str],
        ts1: TransformDatasetState,
        ts2: TransformDatasetState,
        get_categorical: bool = False,
    ) -> Tuple[Set[str], Set[str]]:
        matched1_vspec_ids = set()
        matched2_vspec_ids = set()
        for vhash in matching_value_hashes:
            if get_categorical:
                col_states = ts1.categorical_value_hashes.get(vhash)
            else:
                col_states = ts1.value_hashes.get(vhash)
            if col_states:
                matched1_vspec_ids.update(self.__get_matched_spec_ids(col_states, ts1))

            if get_categorical:
                col_states = ts2.categorical_value_hashes.get(vhash)
            else:
                col_states = ts2.value_hashes.get(vhash)
            if col_states:
                matched2_vspec_ids.update(self.__get_matched_spec_ids(col_states, ts2))
        return matched1_vspec_ids, matched2_vspec_ids

    def match_specs_from_g_hashes(
        self,
        matched_g_hashes: Set[str],
        ts1: TransformDatasetState,
        ts2: TransformDatasetState,
    ):
        matched1_gspec_ids = set()
        matched2_gspec_ids = set()

        def helper_func(
            col_states1: List[SingleColState], col_states2: List[SingleColState]
        ):
            matched1_gspec_ids.update(
                self.__get_matched_spec_ids(col_states1, ts1, graph_w_input=True)
            )
            matched2_gspec_ids.update(
                self.__get_matched_spec_ids(col_states2, ts2, graph_w_input=True)
            )

        self._match_g_hashes_helper(matched_g_hashes, ts1, ts2, helper_func)
        return matched1_gspec_ids, matched2_gspec_ids

    def _match_g_hashes_helper(
        self,
        matched_g_hashes: Set[str],
        ts1: TransformDatasetState,
        ts2: TransformDatasetState,
        helper_func: Callable[[List[SingleColState], List[SingleColState]], None],
    ):
        for graph_hash in matched_g_hashes:
            graph_states1 = ts1.graph_hashes[graph_hash]
            graph_states2 = ts2.graph_hashes[graph_hash]
            matched = False
            for gstate in graph_states1:
                sub_g1, col_states1 = gstate.graph, gstate.col_states
                if matched:
                    break
                for gstate2 in graph_states2:
                    sub_g2, col_states2 = gstate2.graph, gstate2.col_states
                    if matched:
                        break
                    if nx.is_isomorphic(
                        sub_g1,
                        sub_g2,
                        edge_match=lambda x, y: x["type"] == y["type"],
                        node_match=lambda x, y: x["val"] == y["val"],
                    ):
                        helper_func(col_states1, col_states2)
                        matched = True

    def __get_matched_spec_ids(
        self,
        col_states: List[SingleColState],
        state_data: TransformDatasetState,
        graph_w_input: bool = False,
    ) -> Set[str]:
        matched_spec_ids_ret = set()
        for col_state in col_states:
            col_state: SingleColState
            gid = col_state.cols_graph_id
            path_graphs = state_data.graphs[gid]
            g = (
                path_graphs.col_graph.nx_g
                if not graph_w_input
                else path_graphs.col_graph_w_input.nx_g
            )
            matched_spec_ids = self.__get_matched_col_nids_from_nid(
                g, col_state.cols_nid
            )
            matched_spec_ids_ret.update(matched_spec_ids)
        return matched_spec_ids_ret

    def __get_matched_col_nids_from_nid(self, g: nx.DiGraph, nid: str):
        """
        col_g node data:
        {
            "col": "Income",  # associated column name
            "expanded_spec_id": "GROUPBY", # reference to expanded spec id
            "expanded_spec_name": "GROUPBY",
            "val": "groupby" # the transform verb
        }
        """
        ancestors = list(nx.ancestors(g, nid))
        matched_spec_ids = set()
        for ancestor in ancestors:
            matched_spec_ids.add(g.nodes[ancestor]["expanded_spec_id"])
        matched_spec_ids.add(g.nodes[nid]["expanded_spec_id"])
        return matched_spec_ids


if __name__ == "__main__":
    # TODO implement test cases for matching on a toy example here
    pass
