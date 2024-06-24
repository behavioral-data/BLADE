from typing import Dict, List, Tuple
import os.path as osp
import networkx as nx
from absl.testing import absltest
import asynctest

from blade_bench.eval.datamodel import MatchedTSpecs
from blade_bench.eval.match import TransformMatcher
from blade_bench.utils import get_dataset_csv_path, get_dataset_dir
from blade_bench.data.datamodel import (
    POST_GROUPBY_TRANS_VERB,
    Branch,
    TransformSpec,
    TransformDatasetState,
    TransformDataReturn,  # ❗️import to include since we load a pickle that has this class
)
from blade_bench.data.process import GraphPaths, ProcessGraph, AnnotationDataTransforms


from blade_bench.tests.mock_data import (
    GND_TRUTH_SPECS,
    GND_TRUTH_SPECS_W_BRANCH,
    LLM_PATH_SPECS,
    LLM_CODE,
)


SAVE_PATH = osp.join(get_dataset_dir("toy"), "gnd_truth_data_from_GND_TRUTH_SPECS.pkl")
SAVE_PATH_BRANCHES = osp.join(
    get_dataset_dir("toy"), "gnd_truth_data_from_GND_TRUTH_SPECS_W_BRANCH.pkl"
)


def get_parent_transform_specs_from_node_id(
    cols_g: nx.DiGraph, node_id: str, id_to_spec: Dict[str, TransformSpec]
):
    ancestors = nx.ancestors(cols_g, node_id)
    ancestors.add(node_id)
    spec_ids = [cols_g.nodes[ancestor]["expanded_spec_id"] for ancestor in ancestors]
    return [id_to_spec[spec_id] for spec_id in set(spec_ids)]


def get_paret_specs_from_spec_name(
    cols_g: nx.DiGraph,
    spec_name: str,
    ex_id_to_spec: Dict[str, TransformSpec],
):
    col_node_id = [
        node
        for node in cols_g.nodes
        if cols_g.nodes[node]["expanded_spec_name"] == spec_name
    ][0]

    parent_specs = get_parent_transform_specs_from_node_id(
        cols_g, col_node_id, ex_id_to_spec
    )
    return parent_specs


class TestProcessGraph(absltest.TestCase, asynctest.TestCase):
    TEST_SPECS = specs = [
        TransformSpec(spec_id="ROOT", spec_name="ROOT"),
        TransformSpec(
            spec_id="GROUPBY",
            spec_name="GROUPBY",
            trans_verb=["groupby"],
            branches=[Branch(dependencies=["ROOT"])],
        ),
        TransformSpec(
            spec_id="POST_GROUPBY",
            spec_name="POST_GROUPBY",
            trans_verb=[POST_GROUPBY_TRANS_VERB],
            branches=[Branch(dependencies=["GROUPBY"])],
            input_cols_to_output_col_mapping=[
                (["O1"], "col1"),
                (["O2"], "col2"),
                (["O3"], "col3"),
                (["O4"], "col4"),
            ],
        ),
        TransformSpec(
            spec_id="F1",
            spec_name="Filter1",
            trans_verb=["filter"],
            branches=[Branch(dependencies=["POST_GROUPBY"])],
        ),
        TransformSpec(
            spec_id="C",
            spec_name="C",
            trans_verb=["derive"],
            branches=[Branch(dependencies=["F1"])],
            input_cols_to_output_col_mapping=[(["col1", "col2"], "col5")],
        ),
        TransformSpec(
            spec_id="D",
            spec_name="D",
            trans_verb=["derive"],
            branches=[Branch(dependencies=["F1"])],
            input_cols_to_output_col_mapping=[(["col3", "col4"], "col6")],
        ),
        TransformSpec(
            spec_id="E",
            spec_name="E",
            trans_verb=["derive"],
            branches=[Branch(dependencies=["C", "D"])],
            input_cols_to_output_col_mapping=[
                (["col5", "col6"], "col7"),
                (["col6"], "col8"),
            ],
        ),
        TransformSpec(
            spec_id="F2",
            spec_name="Filter2",
            trans_verb=["filter"],
            branches=[Branch(dependencies=["E"])],
        ),
        TransformSpec(
            spec_id="G",
            spec_name="G",
            trans_verb=["derive"],
            branches=[Branch(dependencies=["F2"])],
            input_cols_to_output_col_mapping=[(["col5", "col8"], "col9")],
        ),
    ]

    TEST_SPECS2 = specs = [
        TransformSpec(spec_id="ROOT", spec_name="ROOT"),
        TransformSpec(
            spec_id="GROUPBY",
            spec_name="GROUPBY",
            trans_verb=["groupby"],
            branches=[Branch(dependencies=["ROOT"])],
        ),
        TransformSpec(
            spec_id="POST_GROUPBY",
            spec_name="POST_GROUPBY",
            trans_verb=[POST_GROUPBY_TRANS_VERB],
            branches=[Branch(dependencies=["GROUPBY"])],
            input_cols_to_output_col_mapping=[
                (["O1"], "col1"),
                (["O2"], "col2"),
                (["O3"], "col3"),
                (["O4"], "col4"),
            ],
        ),
        TransformSpec(
            spec_id="F1",
            spec_name="Filter1",
            trans_verb=["filter"],
            input_cols_to_output_col_mapping=[(["col2"], "")],
            branches=[Branch(dependencies=["POST_GROUPBY"])],
        ),
        TransformSpec(
            spec_id="F2",
            spec_name="Filter2",
            trans_verb=["filter"],
            input_cols_to_output_col_mapping=[(["col2"], "")],
            branches=[Branch(dependencies=["POST_GROUPBY"])],
        ),
        TransformSpec(
            spec_id="C",
            spec_name="C",
            trans_verb=["derive"],
            branches=[
                Branch(dependencies=["F1", "F2"]),
            ],
            input_cols_to_output_col_mapping=[(["col1", "col2"], "col5")],
        ),
    ]
    RESULT2_G = {
        "C": {},
        "F1": {"C": {}},
        "F2": {"C": {}},
        "POST_GROUPBY": {"F2": {}, "F1": {}},
        "GROUPBY": {"POST_GROUPBY": {}},
        "ROOT": {"GROUPBY": {}},
    }

    RESULT2_EX_G = {
        "ROOT": {"GROUPBY": {}},
        "GROUPBY": {
            "POST_GROUPBY__post_groupby_col1": {},
            "POST_GROUPBY__post_groupby_col2": {},
            "POST_GROUPBY__post_groupby_col3": {},
            "POST_GROUPBY__post_groupby_col4": {},
        },
        "POST_GROUPBY__post_groupby_col1": {"F2": {}},
        "POST_GROUPBY__post_groupby_col2": {"F2": {}},
        "POST_GROUPBY__post_groupby_col3": {"F2": {}},
        "POST_GROUPBY__post_groupby_col4": {"F2": {}},
        "F2": {"F1": {}},
        "F1": {"C__derive_col5": {}},
        "C__derive_col5": {},
    }
    RESULT2_COLSG = {
        "O3__ROOT": {"O3__GROUPBY": {"type": "data"}},
        "O1__ROOT": {"O1__GROUPBY": {"type": "data"}},
        "O2__ROOT": {"O2__GROUPBY": {"type": "data"}},
        "O4__ROOT": {"O4__GROUPBY": {"type": "data"}},
        "O3__GROUPBY": {"col3__POST_GROUPBY__post_groupby_col3": {"type": "data"}},
        "O1__GROUPBY": {"col1__POST_GROUPBY__post_groupby_col1": {"type": "data"}},
        "O2__GROUPBY": {"col2__POST_GROUPBY__post_groupby_col2": {"type": "data"}},
        "O4__GROUPBY": {"col4__POST_GROUPBY__post_groupby_col4": {"type": "data"}},
        "col1__POST_GROUPBY__post_groupby_col1": {"col1__F2": {"type": "data"}},
        "col2__POST_GROUPBY__post_groupby_col2": {"col2__F2": {"type": "input + data"}},
        "col3__POST_GROUPBY__post_groupby_col3": {"col3__F2": {"type": "data"}},
        "col4__POST_GROUPBY__post_groupby_col4": {"col4__F2": {"type": "data"}},
        "col4__F2": {"col4__F1": {"type": "data"}},
        "col2__F2": {"col2__F1": {"type": "input + data"}},
        "col3__F2": {"col3__F1": {"type": "data"}},
        "col1__F2": {"col1__F1": {"type": "data"}},
        "col4__F1": {},
        "col2__F1": {"col5__C__derive_col5": {"type": "data"}},
        "col3__F1": {},
        "col1__F1": {"col5__C__derive_col5": {"type": "data"}},
        "col5__C__derive_col5": {},
    }

    RESULT3_EX_G = {
        "ROOT": {"F1": {}},
        "F1": {"Impute__impute_O2": {}, "C__derive_col5": {}},
        "Impute__impute_O2": {"C__derive_col5": {}},
        "C__derive_col5": {},
    }

    TEST_SPECS3 = specs = [
        TransformSpec(spec_id="ROOT", spec_name="ROOT"),
        TransformSpec(
            spec_id="F1",
            spec_name="Filter1",
            trans_verb=["filter"],
            input_cols_to_output_col_mapping=[(["O1"], "")],
            branches=[Branch(dependencies=["ROOT"])],
        ),
        TransformSpec(
            spec_id="Impute",
            spec_name="impute",
            trans_verb=["impute"],
            input_cols_to_output_col_mapping=[(["O2"], "O2")],
            branches=[Branch(dependencies=["F1"])],
        ),
        TransformSpec(
            spec_id="C",
            spec_name="C",
            trans_verb=["derive"],
            branches=[
                Branch(dependencies=["Impute"]),
            ],
            input_cols_to_output_col_mapping=[(["O1", "O2"], "col5")],
        ),
    ]

    RESULT3_COLSG = {
        "O4__ROOT": {"O4__F1": {"type": "data"}},
        "O1__ROOT": {"O1__F1": {"type": "input + data"}},
        "O2__ROOT": {"O2__F1": {"type": "data"}},
        "O3__ROOT": {"O3__F1": {"type": "data"}},
        "O4__F1": {},
        "O1__F1": {"col5__C__derive_col5": {"type": "data"}},
        "O2__F1": {"O2__Impute__impute_O2": {"type": "data"}},
        "O3__F1": {},
        "O2__Impute__impute_O2": {"col5__C__derive_col5": {"type": "data"}},
        "col5__C__derive_col5": {},
    }

    RESULT3_G = {"C": {}, "Impute": {"C": {}}, "F1": {"Impute": {}}, "ROOT": {"F1": {}}}

    ORIG_COLS = ["O1", "O2", "O3", "O4"]

    # def test_graph_hash_on_same_graph_but_diff_transform(self):
    #     id_to_spec = {spec.spec_id: spec for spec in self.TEST_SPECS2}
    #     graph_paths = GraphPaths(id_to_spec)
    #     process_graph = ProcessGraph()
    #     gs = graph_paths.get_unserialized_graphs_from_leaf_spec(id_to_spec["C"])
    #     ex_g_obj = process_graph.get_expand_g(gs[0], id_to_spec, self.ORIG_COLS)
    #     expanded_g = ex_g_obj.nx_g
    #     ex_id_to_spec = ex_g_obj.id_to_spec

    #     cols_g_obj = process_graph.get_col_g(ex_g_obj, add_input_edges=True)
    #     cols_g = cols_g_obj.nx_g
    #     self.assertDictEqual(nx.to_dict_of_dicts(cols_g), self.RESULT2_COLSG)
    #     self.assertDictEqual(nx.to_dict_of_dicts(expanded_g), self.RESULT2_EX_G)
    #     self.assertDictEqual(nx.to_dict_of_dicts(gs[0]), self.RESULT2_G)

    def test_impute(self):
        id_to_spec = {spec.spec_id: spec for spec in self.TEST_SPECS3}
        graph_paths = GraphPaths(id_to_spec)
        process_graph = ProcessGraph()
        gs = graph_paths.get_unserialized_graphs_from_leaf_spec(id_to_spec["C"])
        ex_g_obj = process_graph.get_expand_g(gs[0], id_to_spec, self.ORIG_COLS)
        expanded_g = ex_g_obj.nx_g
        self.assertDictEqual(nx.to_dict_of_dicts(expanded_g), self.RESULT3_EX_G)

        ex_id_to_spec = ex_g_obj.id_to_spec
        cols_g_obj = process_graph.get_col_g(ex_g_obj, add_input_edges=True)
        cols_g = cols_g_obj.nx_g
        self.assertDictEqual(nx.to_dict_of_dicts(cols_g), self.RESULT3_COLSG)
        self.assertDictEqual(nx.to_dict_of_dicts(gs[0]), self.RESULT3_G)

    def test_ex_g_and_col_g(self):
        id_to_spec = {spec.spec_id: spec for spec in self.TEST_SPECS}
        graph_paths = GraphPaths(id_to_spec)
        process_graph = ProcessGraph()
        gs = graph_paths.get_unserialized_graphs_from_leaf_spec(id_to_spec["G"])
        ex_g_obj = process_graph.get_expand_g(gs[0], id_to_spec, self.ORIG_COLS)
        expanded_g = ex_g_obj.nx_g
        ex_id_to_spec = ex_g_obj.id_to_spec
        self.assertLen(expanded_g.nodes, 13)
        self.assertLen(expanded_g.edges, 17)

        cols_g_obj = process_graph.get_col_g(ex_g_obj)
        cols_g = cols_g_obj.nx_g
        self.assertLen(cols_g.nodes, 29)
        self.assertLen(cols_g.edges, 29)

        parent_specs = get_paret_specs_from_spec_name(
            cols_g, "derive_col5", ex_id_to_spec
        )
        self.assertLen(parent_specs, 6)

        parent_specs = get_paret_specs_from_spec_name(
            cols_g, "derive_col8", ex_id_to_spec
        )
        self.assertLen(parent_specs, 7)

        parent_specs = get_paret_specs_from_spec_name(
            cols_g, "derive_col7", ex_id_to_spec
        )
        self.assertLen(parent_specs, 10)

    async def __get_gnd_truth_data(self):
        id_to_spec = {spec.spec_id: spec for spec in GND_TRUTH_SPECS}
        annotation = AnnotationDataTransforms(
            dataset_path=get_dataset_csv_path("toy"), id_to_spec=id_to_spec
        )

        # if not osp.exists(SAVE_PATH):
        gnd_truth_state_data = await annotation.build_state_data(
            leaf_specs=[GND_TRUTH_SPECS[-1]]
        )
        gnd_truth_state_data.save(SAVE_PATH)

        gnd_truth_state_data = TransformDatasetState.load(SAVE_PATH)

        return gnd_truth_state_data, annotation

    async def __get_gnd_truth_data_w_branches(self):
        id_to_spec = {spec.spec_id: spec for spec in GND_TRUTH_SPECS_W_BRANCH}
        annotation = AnnotationDataTransforms(
            dataset_path=get_dataset_csv_path("toy"), id_to_spec=id_to_spec
        )

        # if not osp.exists(SAVE_PATH_BRANCHES):
        gnd_truth_state_data = await annotation.build_state_data(
            leaf_specs=[GND_TRUTH_SPECS_W_BRANCH[-1]]
        )
        gnd_truth_state_data.save(SAVE_PATH_BRANCHES)

        gnd_truth_state_data = TransformDatasetState.load(SAVE_PATH_BRANCHES)

        return gnd_truth_state_data, annotation

    async def test_specs_w_runs(self):
        gnd_truth_state_data_w_branches, annotation_w_branches = (
            await self.__get_gnd_truth_data_w_branches()
        )
        self.assertLen(gnd_truth_state_data_w_branches.expanded_id_to_spec, 9)
        self.assertLen(gnd_truth_state_data_w_branches.graphs, 4)

        gnd_truth_state_data, annotation = await self.__get_gnd_truth_data()
        self.assertLen(gnd_truth_state_data.expanded_id_to_spec, 8)
        self.assertLen(gnd_truth_state_data.graphs, 2)
        self.assertLen(gnd_truth_state_data.value_hashes, 14)  # one value is duplicated
        self.assertLen(gnd_truth_state_data.graph_hashes, 15)

        # best way to do this might be to save the csv file then run the init code again
        llm_state_data = await annotation.build_state_data_from_path_specs(
            LLM_PATH_SPECS, save_ts=True
        )
        self.assertLen(llm_state_data.expanded_id_to_spec, 6)
        self.assertLen(llm_state_data.graphs, 1)
        self.assertLen(llm_state_data.value_hashes, 8)
        self.assertLen(llm_state_data.graph_hashes, 8)

        matcher = TransformMatcher(data_path=annotation.dataset_path)
        matching_value_hashes, matching_graph_hashes, matching_categorical_hashes = (
            matcher.find_matching_hashes(gnd_truth_state_data, llm_state_data)
        )
        self.assertLen(matching_value_hashes, 2)
        self.assertLen(matching_graph_hashes, 4)

        mtspecs: MatchedTSpecs = matcher.match_w_tsdata(
            gnd_truth_state_data, llm_state_data
        )
        self.assertLen(mtspecs.vspecs1, 4)
        self.assertLen(mtspecs.vspecs2, 4)
        self.assertLen(mtspecs.gspecs1, 6)
        self.assertLen(mtspecs.gspecs2, 6)

        self.assertGreaterEqual(len(mtspecs.cat_specs1), len(mtspecs.vspecs1))
        self.assertGreaterEqual(len(mtspecs.cat_specs2), len(mtspecs.vspecs2))
        # evalulator = TransformEvaluator(gnd_truth_state_data, annotation.nb_executor)
        # updated_v_specs, updated_v_specs_llm = (
        #     await evalulator.match_based_on_matched_g_hash(
        #         matching_graph_hashes, llm_state_data
        #     )
        # )
        # unique_v_specs = {}
        # for spec in updated_v_specs + mtspecs.vspecs1:
        #     unique_v_specs[spec.spec_id] = spec

        # unique_v_specs_llm = {}
        # for spec in updated_v_specs_llm + mtspecs.vspecs2:
        #     unique_v_specs_llm[spec.spec_id] = spec

        # self.assertLen(updated_v_specs, 1)
        # self.assertLen(updated_v_specs_llm, 1)
        # self.assertLen(unique_v_specs, 5)
        # self.assertLen(unique_v_specs_llm, 5)

        # res = await evalulator.main_match_against_gnd_truth(llm_state_data)
        # self.__compare_res(res)

    def __compare_res(self, res: Dict[str, float]):
        compare_res = {
            "num_gnd_truth_specs": 7,
            "num_llm_specs": 5,
            "matches_value": 3,
            "matches_graph": 5,
            "matches_value_g_hash": 4,
            "match_v_gnd_truth_rate": 0.4285,
            "match_g_gnd_truth_rate": 0.7142,
            "match_v_g_hash_gnd_truth_rate": 0.5714,
            "match_v_llm_rate": 0.6,
            "match_g_llm_rate": 1.0,
            "match_v_g_hash_llm_rate": 0.8,
        }
        self.assertAlmostEqual(
            res["match_v_gnd_truth_rate"],
            compare_res["match_v_gnd_truth_rate"],
            places=3,
        )
        self.assertAlmostEqual(
            res["match_g_gnd_truth_rate"],
            compare_res["match_g_gnd_truth_rate"],
            places=3,
        )
        self.assertAlmostEqual(
            res["match_v_g_hash_gnd_truth_rate"],
            compare_res["match_v_g_hash_gnd_truth_rate"],
            places=3,
        )
        self.assertAlmostEqual(
            res["match_v_llm_rate"], compare_res["match_v_llm_rate"], places=3
        )
        self.assertAlmostEqual(
            res["match_g_llm_rate"], compare_res["match_g_llm_rate"], places=3
        )
        self.assertAlmostEqual(
            res["match_v_g_hash_llm_rate"],
            compare_res["match_v_g_hash_llm_rate"],
            places=3,
        )
        self.assertEqual(res["num_gnd_truth_specs"], compare_res["num_gnd_truth_specs"])
        self.assertEqual(res["num_llm_specs"], compare_res["num_llm_specs"])
        self.assertEqual(res["matches_value"], compare_res["matches_value"])
        self.assertEqual(res["matches_graph"], compare_res["matches_graph"])
        self.assertEqual(
            res["matches_value_g_hash"], compare_res["matches_value_g_hash"]
        )

    async def test_run_eval(self):
        pass


if __name__ == "__main__":
    absltest.main()
