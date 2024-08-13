import collections
from dataclasses import MISSING
import hashlib
import json
import string
from typing import (
    Any,
    List,
    Literal,
    Dict,
    Optional,
    Set,
    Tuple,
    Union,
)
import uuid
from pydantic import BaseModel, Field, field_validator
import networkx as nx

ROOT_SPEC_ID = "ROOT_SPEC_ID"
ROOT_SPEC_NAME = "ROOT"
LEAF_SPEC_ID = "LEAF_NODE"
LEAF_SPEC_NAME = "CUR_NODE"
DERIVED_COL_PREFIX = "🟩 "
POST_GROUPBY_TRANS_VERB = "post_groupby"

TRANSFORM_SPEC_COLUMN_NAME = "transform_spec_json"
MODEL_SPEC_COLUMN_NAME = "model_spec_json"
CONCEPTUAL_VAR_SPEC_COLUMN_NAME = "conceptual_spec_json"


DELETED_NODE_STR = "DELETED_NODE"


def replace_spaces_with_underscore(text: str, max_words=10):
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Split the text into words
    words = text.split()

    # If there are more than 10 words, truncate the list and add "etc"
    if len(words) > max_words:
        words = words[:max_words]
        words.append("etc")

    # Join the words with underscores
    modified_text = "_".join(words)
    return modified_text.lower()


class BaseSpec(BaseModel):
    spec_id: str = Field(default=None, exclude=True)
    time_stamp: Optional[str] = Field(default=None, exclude=True)

    def __init__(self, **data):
        if "spec_id" not in data or data["spec_id"] is None:
            data["spec_id"] = uuid.uuid4().hex
        super().__init__(**data)

    def __eq__(self, other):
        return self.spec_id == other.spec_id

    @property
    def column_name(self) -> str:
        raise NotImplementedError

    @property
    def to_df_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class SpecWithTags(BaseSpec):
    tags: Optional[List[str]] = []
    read_only_tags: Optional[List[str]] = []  # tags that cannot be removed by the user
    is_negative: Optional[bool] = False


class SpecWithSpecName(SpecWithTags):
    specification: str = ""
    spec_name: str = ""

    @field_validator("spec_name")
    def get_spec_name(cls, spec_name: str, values: Dict[str, Any]) -> str:
        if spec_name == "" and "specification" in values.data:
            return replace_spaces_with_underscore(values.data["specification"])
        return spec_name


class Branch(BaseModel):
    dependencies: List[str] = []
    condition: str = ""


class TransformSpec(SpecWithSpecName):
    trans_verb: List[
        Union[
            Literal[
                "derive",
                "filter",
                "group",
                "mutate",
                "select",
                "summarize",
                "deduplicate",
            ],
            str,
        ]
    ] = []
    code: str = ""
    rationale: str = ""
    conditions: str = ""
    input_cols_to_output_col_mapping: Optional[List[Tuple[List[str], str]]] = []
    branches: List[Branch] = []
    dependencies: List[str] = Field(default=[], exclude=True)

    @property
    def has_deleted_condition(self) -> bool:
        return any(DELETED_NODE_STR in branch.condition for branch in self.branches)

    @property
    def output_cols(self) -> List[str]:
        return [
            output_col
            for _, output_col in self.input_cols_to_output_col_mapping
            if output_col
        ]

    @property
    def input_cols(self) -> List[str]:
        return [
            input_col
            for input_cols, _ in self.input_cols_to_output_col_mapping
            for input_col in input_cols
        ]

    @property
    def column_name(self) -> str:
        return TRANSFORM_SPEC_COLUMN_NAME

    def to_df_dict(self) -> Dict[str, Any]:
        return {
            "transform_spec_json": [self.json()],
            "spec_id": [self.spec_id],
        }

    def validate_msgs(self, existing_specs: List = None) -> Dict[str, str]:
        """Sample validation for the demo. Use your preferred framework instead."""
        result = {}
        if self.spec_name == MISSING or self.spec_name == "":
            result["spec_name"] = "specification name must be non-empty"
        elif existing_specs and any(
            spec.spec_name == self.spec_name for spec in existing_specs
        ):
            result["spec_name"] = (
                f"specification **{self.spec_name}** already exists. See **Summary** page for more details."
            )

        if self.trans_verb == MISSING or self.trans_verb == []:
            result["trans_verb"] = "Must include at least one transform verb"

        if self.trans_verb == "derive" and (
            self.output_cols == MISSING or self.output_cols == []
        ):
            result["output_cols"] = (
                "Must include at least one output column for a derive transformation"
            )
        return result


class ModelSpec(SpecWithSpecName):
    rationale: Optional[str] = None
    dependencies: List[str] = []
    code: Optional[str] = None
    associated_columns_derived_spec_ids: Optional[List[str]] = []
    associated_columns_orig: Optional[List[str]] = []
    associated_columns_derived: Optional[List[str]] = []  # just the column names now
    associated_columns_leaf_spec_ids: Optional[List[str]] = []
    associated_specified_spec_ids: Optional[List[str]] = []
    annotator: Optional[str] = None

    @property
    def column_name(self) -> str:
        return MODEL_SPEC_COLUMN_NAME

    def to_df_dict(self) -> Dict[str, Any]:
        return {
            "model_spec_json": [self.json()],
            "spec_id": [self.spec_id],
        }

    def validate_msgs(self, existing_specs: List = None) -> Dict[str, str]:
        """Sample validation for the demo. Use your preferred framework instead."""
        result = {}
        if self.specification == MISSING or self.specification == "":
            result["specification"] = "model specification must be non-empty"
        if self.code == MISSING or self.code == "":
            result["code"] = (
                "please include the code you used to specify this model. For example (unrelated to the current spec)\n```python\nmodel = smf.ols(formula='rater_combined ~ red_cards_total + rating + player_win_rate', data=data),\n```"
            )
        return result


class ConceptualVarSpec(SpecWithTags):
    variable_type: Literal["IV", "DV", "Control"]
    concept: str
    final_columns_derived_spec_ids: Optional[List[str]] = []
    final_columns_orig: Optional[List[str]] = []
    final_columns_derived: Optional[List[str]] = []  # just the column names now
    final_columns_leaf_spec_ids: Optional[List[str]] = []
    column_rationales: Optional[Dict[str, str]] = {}
    interaction: bool
    rationale: str
    on: str = ""
    random_effect: Optional[bool] = False
    annotator: Optional[str] = None

    def __str__(self):
        if self.variable_type != "Control":
            return f"{self.variable_type}: {self.concept}"
        else:
            if self.interaction:
                return f"Moderator: {self.concept} interacting with {self.on} variable"
            else:
                return f"Control: {self.concept}."

    @property
    def column_name(self) -> str:
        return CONCEPTUAL_VAR_SPEC_COLUMN_NAME

    def to_df_dict(self) -> Dict[str, Any]:
        return {
            "conceptual_spec_json": [self.json()],
            "spec_id": [self.spec_id],
        }

    def transform_spec_ids_to_cols(
        self, id_to_spec: Dict[str, TransformSpec]
    ) -> Dict[str, Set[str]]:
        ret = collections.defaultdict(set)
        for spec_id in self.final_columns_derived_spec_ids:
            tspec = id_to_spec.get(spec_id)
            if tspec:
                for col in tspec.output_cols:
                    if DERIVED_COL_PREFIX + col in self.final_columns_derived:
                        ret[spec_id].add(col)
        return ret

    def cols_to_transform_spec_ids(
        self, id_to_spec: Dict[str, TransformSpec]
    ) -> Dict[str, Set[str]]:
        ret = collections.defaultdict(set)
        for col in self.final_columns_derived:
            for spec_id, tspec in id_to_spec.items():
                col_no_prefix = col[len(DERIVED_COL_PREFIX) :]
                if col_no_prefix in tspec.output_cols:
                    ret[col_no_prefix].add(spec_id)
        return dict(ret)

    def validate_msgs(self, existing_vars: list) -> Dict[str, str]:
        """Sample validation for the demo. Use your preferred framework instead."""

        def check_number(s):
            try:
                int(s)
            except ValueError:
                try:
                    float(s)
                except ValueError:
                    return False
                else:
                    return True
            else:
                return True

        def check_column_rationales():
            cols = []
            for col in self.final_columns_derived + self.final_columns_orig:
                if not self.column_rationales.get(col):
                    cols.append(col)
            return cols

        result = {}
        cols_without_rationales = check_column_rationales()
        if self.concept == MISSING or self.concept == "":
            result["concept"] = "Variable conceptual specification must be non-empty"
        elif check_number(self.concept):
            result["concept"] = "Variable conceptual specification must be non-numeric"
        elif self.variable_type == "Control" and (
            self.rationale == MISSING or self.rationale == ""
        ):
            result["rationale"] = "Please provide a rationale for the control variable"
        elif len(cols_without_rationales) > 0:
            result["column_rationales"] = (
                f"Please provide a rationale for the following column(s): {', '.join(cols_without_rationales)}"
            )
        else:
            exists = any(var.concept == self.concept for var in existing_vars)
            if exists:
                result["concept"] = (
                    f"Variable conceptual specification **{self.concept}** already exists. See **Summary** page for more details."
                )
        return result


if __name__ == "__main__":
    pass
