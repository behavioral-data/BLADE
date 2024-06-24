from typing import List
from absl.testing import absltest
import ast
import astor

from blade_bench.parse_code import (
    extract_code_inside_functions_and_func_names,
    get_function_arg_name,
    ast_equal,
    replace_variable_name,
)

FUNCTION_CODE = """
def t1(df: pd.DataFrame):
    df = df[df['Female_Contributions'] == df['UniqueFemaleContributors']]
    print("hello")
    return TransformDataReturn(
            df=df, 
            column_mapping={frozenset(['Female_Contributions', 'UniqueFemaleContributors']): 'ALL'},
            transform_verb='filter'
        )

def t2(df: pd.DataFrame):
    df = df.groupby('ThreadId').apply(lambda grp: grp.assign(WC_s=scale(grp['WC'])))
    return TransformDataReturn(
            df=df, 
            groupby_cols=frozenset(['ThreadId']),
            column_mapping={frozenset(['WC']): 'WC_s'},
            transform_verb='groupby'
        )


transform_funcs = [t1, t2]
"""

CODE_EXTRACTED_1 = """df = df[df[\'Female_Contributions\'] == df[\'UniqueFemaleContributors\']]\nprint("hello")"""

CODE_EXTRACTED_2 = """df = df.groupby('ThreadId').apply(lambda grp: grp.assign(WC_s=scale(grp['WC'])))"""

EXPECTED_FUNCTION_CODES = [
    CODE_EXTRACTED_1,
    CODE_EXTRACTED_2,
]

FUNC_CODE2 = """
def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.UniqueContributors > 1) & (df.UniqueFemaleContributors >= 1)]
    df_grp = df.groupby('ThreadId')
    def calculate_running_stats(group):
        group['running_female_count'] = group['Female'].cumsum()
        return group
    df = df_grp.apply(calculate_running_stats)
    return df, ['running_female_count']
"""

CODE2_EXTRACTED = """
df = df[(df.UniqueContributors > 1) & (df.UniqueFemaleContributors >= 1)]
df_grp = df.groupby('ThreadId')
def calculate_running_stats(group):
    group['running_female_count'] = group['Female'].cumsum()
    return group
df = df_grp.apply(calculate_running_stats)"""

EXPECTED_FUNCTION_CODES2 = [CODE2_EXTRACTED]


class TestGenerators(absltest.TestCase):

    def test_ast_equal(self):
        # Test equality for simple nodes
        node1 = ast.parse("a = 1")
        node2 = ast.parse("a = 1")
        self.assertTrue(ast_equal(node1, node2))

        # Test inequality for different nodes
        node3 = ast.parse("a = 2")
        self.assertFalse(ast_equal(node1, node3))

        # Test list nodes equality
        list_node1 = [ast.parse("a = 1")]
        list_node2 = [ast.parse("a = 1")]
        self.assertTrue(ast_equal(list_node1, list_node2))

        # Test list nodes inequality
        list_node3 = [ast.parse("a = 2")]
        self.assertFalse(ast_equal(list_node1, list_node3))

    def equal_codes(self, codes1: List[str], codes2: List[str]):
        for code1, code2 in zip(codes1, codes2):
            tree1 = ast.parse(code1)
            tree2 = ast.parse(code2)
            self.assertTrue(ast_equal(tree1, tree2))

    def test_extract_code_inside_functions_and_func_names(self):
        function_codes, function_names = extract_code_inside_functions_and_func_names(
            FUNCTION_CODE
        )

        self.equal_codes(function_codes, EXPECTED_FUNCTION_CODES)
        self.assertEqual(function_names, ["t1", "t2"])

        function_codes2, function_names2 = extract_code_inside_functions_and_func_names(
            FUNC_CODE2
        )
        self.equal_codes(function_codes2, EXPECTED_FUNCTION_CODES2)
        self.assertEqual(function_names2, ["transform"])

    def test_get_function_arg_name(self):
        code = """
def foo(a, b):
    return a + b

def bar(x, y):
    return x * y
"""
        self.assertEqual(get_function_arg_name(code, "foo"), "a")
        self.assertEqual(get_function_arg_name(code, "bar"), "x")
        self.assertIsNone(get_function_arg_name(code, "baz"))

    def test_replace_variable_name(self):
        code = """
def foo(a, b):
    return a + b
"""
        expected_code = """
def foo(cd, b):
    return cd + b
"""
        self.assertEqual(
            replace_variable_name(code, "a", "cd").strip(), expected_code.strip()
        )


if __name__ == "__main__":
    absltest.main()
