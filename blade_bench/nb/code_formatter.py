import ast
from io import BytesIO
import tokenize
import astor
from typing import Text
import os
import inspect
from functools import wraps

IMG_BUFFER_VAR_NAME = "image_buffer"
SNS_BUFFER_VAR_NAME = "sns_buffer"


class CodeFormatterSnsSave(ast.NodeTransformer):
    """Traverses the AST and applies the following modifications.
    1. if there is a sns.histplot or sns.scatterplot call, save the figure to a buffer to the variable in SNS_BUFFER_VAR_NAME.

    For example:
    ```python
    sns.histplot(data=df, x='masfem', kde=True)
    ```
    becomes
    ```python
    sns_img = sns.histplot(data=df, x='masfem', kde=True)
    {SNS_BUFFER_VAR_NAME} = BytesIO()
    sns_img.figure.savefig({SNS_BUFFER_VAR_NAME}, format='png')
    """

    def __init__(self):
        super(CodeFormatterSnsSave, self).__init__()
        self.counter = 0

    @property
    def image_buffer_name(self):
        return f"{SNS_BUFFER_VAR_NAME}_{self.counter}"

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        """
        Modify an AST for expressions.

        If the expression is a call to sns.histplot or sns.scatterplot, add a variable
        assignment to the left side.

        Args:
            node: original AST node.

        Returns:
            the modified AST node.
        """
        if isinstance(node.value, ast.Call):
            call_node = node.value
            if (
                isinstance(call_node.func, ast.Attribute)
                and isinstance(call_node.func.value, ast.Name)
                and call_node.func.value.id == "sns"
            ):
                # Check if the function being called is sns.histplot or sns.scatterplot
                if call_node.func.attr in ["histplot", "scatterplot", "heatmap"]:
                    # Create a new variable name based on the function name
                    var_name = f"sns_img"

                    # Create a variable assignment statement
                    var_assign = ast.Assign(
                        targets=[ast.Name(id=var_name, ctx=ast.Store())],
                        value=call_node,
                        ctx=ast.Store(),
                    )

                    new_code = (
                        astor.to_source(var_assign)
                        + f"{self.image_buffer_name} = BytesIO()\nsns_img.figure.savefig({self.image_buffer_name}, format='png')"
                    )
                    new_node = ast.parse(new_code).body
                    self.counter += 1
                    # Return the modified AST with the variable assignment
                    return new_node

        # If the expression is not a call to sns.histplot or sns.scatterplot, return the original AST node
        return node

    @classmethod
    def preprocess_code_sns_save(cls, source_code: Text, **kwargs) -> Text:
        """Use the formatter to preprocess a given code snippet."""
        inst = cls(**kwargs)

        code_ast: ast.Module = ast.parse(source_code)
        modified_ast = inst.visit(code_ast)
        return astor.to_source(modified_ast)


class CodeFormatterPltShow(ast.NodeTransformer):
    """Traverses the AST and applies the following modifications.
    1. if there is a plt.show(), save the figure to a buffer to the variable in IMG_BUFFER_VAR_NAME.

    For example:
    ```python
    plt.show()
    ```
    becomes
    ```python
    {IMG_BUFFER_VAR_NAME} = BytesIO()
    plt.savefig({IMG_BUFFER_VAR_NAME}, format='png')
    {IMG_BUFFER_VAR_NAME}
    """

    def __init__(self):
        super(CodeFormatterPltShow, self).__init__()
        self.counter = 0

    @property
    def image_buffer_name(self):
        return f"{IMG_BUFFER_VAR_NAME}_{self.counter}"

    def visit_Call(self, node):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "plt"
            and node.func.attr == "show"
        ):
            # Replace plt.show() with the new code
            new_code = f"{self.image_buffer_name} = BytesIO()\nplt.savefig({self.image_buffer_name}, format='png')\n{self.image_buffer_name}"
            new_node = ast.parse(new_code).body
            self.counter += 1
            return new_node

        return node

    @classmethod
    def preprocess_code_plt_show(cls, source_code: Text, **kwargs) -> Text:
        """Use the formatter to preprocess a given code snippet."""
        inst = cls(**kwargs)

        code_ast: ast.Module = ast.parse(source_code)
        modified_ast = inst.visit(code_ast)
        return astor.to_source(modified_ast)


class CodeFormatter(ast.NodeTransformer):
    """Traverse an AST and applies the following modifications.

    1. unwrap print statements:
    print(df.head()) --> df.head()

    2. save the return value of the last function call to a reserved
    variable `__output__`
    """

    def __init__(self, capture_all_variables: bool = False):
        super(CodeFormatter, self).__init__()

        self.capture_all_variables = capture_all_variables

    def visit_Module(
        self, node: ast.Module
    ) -> ast.Module:  # pylint: disable=invalid-name
        """Modify an AST.

        Save the output of the last statement to varibale `output`. If it is
        a print function like `print(df.std())`, change it to `__output__=df.std()`.
        `result = df.call()` will be changed to `__output__ = result = df.call()`

        Args:
        node: original AST node.

        Returns:
        the modified AST node.
        """
        new_body = []

        for stmt_id, stmt in enumerate(node.body):
            # Find the last statement.
            if stmt_id == len(node.body) - 1 or self.capture_all_variables:
                expr_val = None
                if isinstance(stmt, ast.Expr):
                    # and isinstance(stmt.value, (ast.Call, ast.Attribute, ast.Subscript))
                    # Unwrap print statements.
                    if (
                        isinstance(stmt.value, ast.Call)
                        and isinstance(stmt.value.func, ast.Name)
                        and stmt.value.func.id == "print"
                        and stmt.value.args
                    ):
                        if len(stmt.value.args) == 1:
                            expr_val = stmt.value.args[0]
                        else:
                            sep = " "
                            for kw in stmt.value.keywords:
                                if kw.arg == "sep" and isinstance(
                                    kw.value, ast.Constant
                                ):
                                    sep = kw.value.value
                            args = [
                                f"str({astor.to_source(arg)[:-1]})"
                                for arg in stmt.value.args
                            ]
                            new_code = f"'{sep}'.join([{', '.join(args)}])"
                            expr_val = ast.parse(new_code).body[0].value
                    else:
                        expr_val = stmt.value

                    existing_names = []
                    captured_var_name = (
                        "__output__"
                        if stmt_id == len(node.body) - 1
                        else f"__tmp_{stmt_id}"
                    )
                elif stmt_id == len(node.body) - 1 and isinstance(stmt, ast.Assign):
                    existing_names = list(stmt.targets)
                    expr_val = stmt.value
                    captured_var_name = "__output__"

                if expr_val:
                    new_stmt = ast.Assign(
                        targets=(
                            [ast.Name(id=captured_var_name, ctx=ast.Load())]
                            + existing_names
                        ),
                        value=expr_val,
                        ctx=ast.Load(),
                    )
                else:
                    new_stmt = stmt

                new_body.append(new_stmt)
            else:
                new_body.append(stmt)

        return ast.Module(body=new_body)

    @classmethod
    def preprocess_code_for_tracing(cls, source_code: Text, **kwargs) -> Text:
        """Use the formatter to preprocess a given code snippet."""
        inst = cls(**kwargs)

        code_ast: ast.Module = ast.parse(source_code)
        modified_ast = inst.visit(code_ast)
        modified_code = astor.to_source(modified_ast)

        return modified_code


def has_variable_assignment(code: str):
    try:
        parsed_code = ast.parse(code)
        for node in ast.walk(parsed_code):
            if isinstance(node, ast.Assign):
                return True
        return False
    except SyntaxError:
        return False


class EnvException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


def normalize_args_kwargs(f, *args, **kwargs):
    """This function takes a function and its arguments and returns a dictionary of the arguments, with the keys being the argument names."""
    sig = inspect.signature(f)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()  # This line is optional, it fills in any omitted arguments that have default values
    return bound.arguments


def check_file_in_work_dir(arg_names, **kwargs):
    """This decorator checks if the file is in the work directory."""

    def inner(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            new_kwargs = normalize_args_kwargs(func, *args, **kwargs)
            work_dir = new_kwargs["work_dir"]
            for arg_name in arg_names:
                file_name = new_kwargs[arg_name]
                if not os.path.abspath(os.path.join(work_dir, file_name)).startswith(
                    os.path.abspath(work_dir)
                ):
                    raise EnvException(
                        f"cannot access file {file_name} because it is not in the work directory."
                    )
            return func(*args, **kwargs)

        return wrapper

    return inner


def get_last_code_line(code):
    codes = [
        tok.string
        for tok in list(tokenize.tokenize(BytesIO(code.encode("utf-8")).readline))
        if tok.type != tokenize.COMMENT
    ]
    return "".join(codes).strip().split("\n")[-1]


if __name__ == "__main__":
    code = "print(1, 'Python Tricks', 'Dan Bader', sep=',')"
    print(CodeFormatter.preprocess_code_for_tracing(code))

    code = """
import seaborn as sns

# Generate descriptive statistics for 'masfem' column
print(df['masfem'].describe())

# Visualize distribution of 'masfem' scores
sns.histplot(data=df, x='masfem', kde=True)

# Examine relationship between 'masfem' and 'death'
sns.scatterplot(data=df, x='masfem', y='death')

# Examine relationship between 'masfem' and 'dam'
sns.scatterplot(data=df, x='masfem', y='dam')
""".lstrip()
    print(CodeFormatterSnsSave.preprocess_code_sns_save(code))
