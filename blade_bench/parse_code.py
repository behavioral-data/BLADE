import astor
import ast
from typing import List


def ast_equal(node1, node2):
    if type(node1) != type(node2):
        return False

    if isinstance(node1, ast.AST):
        for attr in node1._fields:
            if not ast_equal(getattr(node1, attr), getattr(node2, attr)):
                return False
        return True

    if isinstance(node1, list):
        if len(node1) != len(node2):
            return False
        for n1, n2 in zip(node1, node2):
            if not ast_equal(n1, n2):
                return False
        return True

    return node1 == node2


def extract_code_inside_functions_and_func_names(code):
    function_codes = []
    function_names = []

    tree = ast.parse(code)
    visited = []
    for node in tree.body:
        if any(ast_equal(node, visited_node) for visited_node in visited):
            continue
        if isinstance(node, ast.FunctionDef):
            function_names.append(node.name)
            code_str = ""
            for child in node.body:
                if isinstance(child, ast.Return):
                    break
                elif isinstance(child, ast.FunctionDef):
                    visited.append(child)
                    code_str += astor.to_source(child)
                else:
                    code_str += astor.to_source(child)
            function_codes.append(code_str)
    return function_codes, function_names


def get_function_arg_name(source_code, function_name):
    class FunctionArgVisitor(ast.NodeVisitor):
        def __init__(self):
            self.arg_names = []

        def visit_FunctionDef(self, node):
            if node.name == function_name:
                for arg in node.args.args:
                    self.arg_names.append(arg.arg)
            self.generic_visit(node)

    tree = ast.parse(source_code)
    visitor = FunctionArgVisitor()
    visitor.visit(tree)
    return visitor.arg_names[0] if visitor.arg_names else None


def replace_variable_name(source_code, old_name, new_name):
    class VariableNameReplacer(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == old_name:
                node.id = new_name
            return node

        def visit_FunctionDef(self, node):
            for i, arg in enumerate(node.args.args):
                if arg.arg == old_name:
                    arg.arg = new_name
            self.generic_visit(node)
            return node

    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Create an instance of the transformer and apply it to the AST
    replacer = VariableNameReplacer()
    new_tree = replacer.visit(tree)

    # Convert the modified AST back to source code
    new_source_code = astor.to_source(new_tree)
    return new_source_code


def process_groupby_code(code):
    # Parse the code into an abstract syntax tree (AST)
    tree = ast.parse(code)

    # Define a transformer to modify the AST
    class GroupByTransformer(ast.NodeTransformer):
        def visit_Assign(self, node):
            if (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "groupby"
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.startswith("df") and not target.id.endswith(
                            "_grp"
                        ):
                            target.id += "_grp"
            return node

    # Apply the transformer to the AST
    transformer = GroupByTransformer()
    transformed_tree = transformer.visit(tree)

    # Generate code from the modified AST
    modified_code = ast.unparse(transformed_tree)
    return modified_code


if __name__ == "__main__":
    code_test = """
def transform(data):
    # Drop the rows with missing values in the ReportedCycleLength, Rel1, Rel2, and Rel3 columns.
    data = data.dropna(subset=['ReportedCycleLength', 'Rel1', 'Rel2', 'Rel3'])
    # Calculate the cycle length based on provided dates.
    data['CycleLength'] = data['EndDate'] - data['StartDate']
    # Calculate the expected start date of the next menstrual period.
    data['NextPeriodStart'] = data['StartDate'] + data['CycleLength']
    # Calculate the day of ovulation by subtracting 14 days from the expected start date of the next period.
    data['OvulationDay'] = data['NextPeriodStart'] - pd.Timedelta(days=14)
    # Define high-fertility (cycle days 6-14) and low-fertility (cycle days 17-27) groups.
    data['FertilityGroup'] = data['CycleDay'].apply(lambda x: 'High' if 6 <= x <= 14 else ('Low' if 17 <= x <= 27 else 'Other'))
    # Filter out the 'Other' group to focus on the high and low fertility groups.
    data = data[data['FertilityGroup'] != 'Other']
    return data
    """
    a = get_function_arg_name(code_test, "transform")
    b = replace_variable_name(code_test, a, "df")
    print(b)
