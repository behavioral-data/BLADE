from copy import deepcopy
from blade_bench.data.datamodel import (
    Branch,
    TransformSpec,
    ROOT_SPEC_ID,
    ROOT_SPEC_NAME,
)

LLM_CODE = """
df = (
    df.groupby('Education')
      .agg(income_mean_ai=('Income', 'mean'),
           age_mean_ai=('Age', 'mean'),
           children_mean_ai=('Has_Children', 'mean'))
      .reset_index()
)

df['income_divide_half_if_children_ai'] = df.apply(lambda x: x['income_mean_ai'] / 2 if x['children_mean_ai'] else x['income_mean_ai'], axis=1)
"""

LLM_MODEL_CODE = """
"""

LLM_PATH_SPECS = [
    TransformSpec(
        spec_id=ROOT_SPEC_ID,
        spec_name=ROOT_SPEC_NAME,
    ),
    TransformSpec(
        spec_id="GROUPBY",
        spec_name="GROUPBY",
        code="""df = df""",
        trans_verb=["groupby"],
        branches=[Branch(dependencies=[ROOT_SPEC_ID])],
        input_cols_to_output_col_mapping=[
            [["Education"], ""],
        ],
    ),
    TransformSpec(
        spec_id="D1",
        spec_name="D1",
        code="""
df_grp = df.groupby(['Education'])
# derive_after_group
df = df_grp.agg(income_mean_ai=('Income', 'mean'), age_mean_ai=('Age', 'mean'), children_mean_ai=('Has_Children', 'mean')).reset_index()
""",
        trans_verb=["derive"],
        branches=[Branch(dependencies=["GROUPBY"])],
        input_cols_to_output_col_mapping=[
            [["Income"], "income_mean_ai"],
            [["Age"], "age_mean_ai"],
            [["Has_Children"], "children_mean_ai"],
        ],
    ),
    TransformSpec(
        spec_id="D2",
        spec_name="D2",
        trans_verb=["derive"],
        branches=[Branch(dependencies=["D1"])],
        code="""
df['income_divide_half_if_children_ai'] = df.apply(lambda x: x['income_mean_ai'] / 2 if x['children_mean_ai'] else x['income_mean_ai'], axis=1)
""",
        input_cols_to_output_col_mapping=[
            [
                ["income_mean_ai", "children_mean_ai"],
                "income_divide_half_if_children_ai",
            ],
        ],
    ),
]


GND_TRUTH_SPECS = [
    TransformSpec(
        spec_id=ROOT_SPEC_ID,
        spec_name=ROOT_SPEC_NAME,
    ),
    TransformSpec(
        spec_id="F1",
        spec_name="F1",
        trans_verb=["filter"],
        code="""df = df[df['Income'] > 50000]
""",
        input_cols_to_output_col_mapping=[
            [["Income"], ""],
        ],
        branches=[Branch(dependencies=[ROOT_SPEC_ID])],
    ),
    TransformSpec(
        spec_id="GROUPBY",
        spec_name="GROUPBY",
        trans_verb=["groupby"],
        code="""df = df.groupby(['Education'])""",
        input_cols_to_output_col_mapping=[
            [["Education"], ""],
        ],
        branches=[Branch(dependencies=[ROOT_SPEC_ID]), Branch(dependencies=["F1"])],
    ),
    TransformSpec(
        spec_id="D1",
        spec_name="D1",
        trans_verb=["derive"],
        code="""
df = df_grp.agg(income_mean=('Income', 'mean'), age_mean=('Age', 'mean'), children_mode=('Has_Children', lambda x: x.mode()[0])).reset_index()
print(df.shape)
""",
        input_cols_to_output_col_mapping=[
            [["Income"], "income_mean"],
            [["Age"], "age_mean"],
            [["Has_Children"], "children_mode"],
        ],
        branches=[Branch(dependencies=["GROUPBY"])],
    ),
    TransformSpec(
        spec_id="D2",
        spec_name="D2",
        trans_verb=["derive"],
        branches=[Branch(dependencies=["D1"])],
        code="""
df['income_times_age'] = df['income_mean'] * df['age_mean']
df['income_divide_half_if_children'] = df.apply(lambda x: x['income_mean'] / 2 if x['children_mode'] else x['income_mean'], axis=1)
""",
        input_cols_to_output_col_mapping=[
            [["income_mean", "age_mean"], "income_times_age"],
            [["income_mean", "children_mode"], "income_divide_half_if_children"],
        ],
    ),
]


GND_TRUTH_SPECS_W_BRANCH = [
    TransformSpec(
        spec_id=ROOT_SPEC_ID,
        spec_name=ROOT_SPEC_NAME,
    ),
    TransformSpec(
        spec_id="F1",
        spec_name="F1",
        trans_verb=["filter"],
        code="""df = df[df['Income'] > 50000]""",
        input_cols_to_output_col_mapping=[
            [["Income"], ""],
        ],
        branches=[Branch(dependencies=[ROOT_SPEC_ID])],
    ),
    TransformSpec(
        spec_id="F2",
        spec_name="F2",
        trans_verb=["filter"],
        branches=[Branch(dependencies=["F1"]), Branch(dependencies=[ROOT_SPEC_ID])],
        code="df = df[df['Age'] < 50]",
        input_cols_to_output_col_mapping=[
            [["Age"], ""],
        ],
    ),
    TransformSpec(
        spec_id="GROUPBY",
        spec_name="GROUPBY",
        trans_verb=["groupby"],
        code="""df = df.groupby(['Education'])""",
        input_cols_to_output_col_mapping=[
            [["Education"], ""],
        ],
        branches=[
            Branch(dependencies=[ROOT_SPEC_ID]),
            Branch(dependencies=["F1"]),
            Branch(dependencies=["F2"]),
        ],
    ),
    TransformSpec(
        spec_id="D1",
        spec_name="D1",
        trans_verb=["derive"],
        code="""
df = df_grp.agg(income_mean=('Income', 'mean'), age_mean=('Age', 'mean'), children_mode=('Has_Children', lambda x: x.mode()[0])).reset_index()
print(df.shape)
""",
        input_cols_to_output_col_mapping=[
            [["Income"], "income_mean"],
            [["Age"], "age_mean"],
            [["Has_Children"], "children_mode"],
        ],
        branches=[Branch(dependencies=["GROUPBY"])],
    ),
    TransformSpec(
        spec_id="D2",
        spec_name="D2",
        trans_verb=["derive"],
        branches=[Branch(dependencies=["D1"])],
        code="""
df['income_times_age'] = df['income_mean'] * df['age_mean']
df['income_divide_half_if_children'] = df.apply(lambda x: x['income_mean'] / 2 if x['children_mode'] else x['income_mean'], axis=1)
""",
        input_cols_to_output_col_mapping=[
            [["income_mean", "age_mean"], "income_times_age"],
            [["income_mean", "children_mode"], "income_divide_half_if_children"],
        ],
    ),
]
