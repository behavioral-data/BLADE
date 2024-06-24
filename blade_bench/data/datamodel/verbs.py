TRANSFORM_VERBS = [
    "derive",
    "filter",
    "slice",
    "groupby",
    "deduplicate",
    "impute",
    # "pivot",
    "orderby",
    # "count",
    "rollup",
    # "join",
    "other",
    "other_impact_rows",
    "other_impact_cols",
]

TRANSFORM_VERBS_W_OUTPUT_COL = ["derive", "rollup", "impute", "other_impact_cols"]
VERBS_AFFECTING_WHOLE_DF = [
    "groupby",
    "filter",
    "other_impact_rows",
]
