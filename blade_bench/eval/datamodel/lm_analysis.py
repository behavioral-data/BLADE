from collections import defaultdict
from typing import Dict, List, Literal, Union, Optional
import pickle
from pydantic import BaseModel, Field
import networkx as nx

from blade_bench.data.datamodel import (
    ConceptualVarSpec,
    ModelSpec,
    TransformDatasetState,
)
from blade_bench.parse_code import (
    extract_code_inside_functions_and_func_names,
    get_function_arg_name,
    replace_variable_name,
)


class IVar(BaseModel):
    description: str = Field(
        ..., title="Description of the independent variable variable"
    )

    def __str__(self):
        return f"IV: {self.description}."


class DVar(BaseModel):
    description: str = Field(
        ..., title="Description of the dependent variable variable"
    )

    def __str__(self):
        return f"DV: {self.description}."


class ControlVar(BaseModel):
    description: str = Field(..., title="Description of the control variable variable")
    is_moderator: bool = Field(
        ...,
        title="Whether the variable is a moderator.",
    )
    moderator_on: Optional[str] = Field(
        ...,
        title="The variable that the control variable is moderating. Only applicable for control variables that are moderators.",
        nullable=True,
    )

    def __str__(self):
        if self.is_moderator:
            return f"Moderator: {self.description} interacting with {self.moderator_on} variable"
        else:
            return f"Control: {self.description}."


class IVarWithCol(BaseModel):
    description: str = Field(
        ..., title="Description of the independent variable variable"
    )
    columns: List[str] = Field(
        ...,
        title="The column(s) in the FINAL dataframe used in the STATISTICAL MODEL that corresponds to the independent variable",
    )

    def __str__(self):
        return f"IV: {self.description}."

    def to_cvar_spec(self) -> ConceptualVarSpec:
        return ConceptualVarSpec(
            variable_type="IV",
            concept=self.description,
            final_columns_derived=self.columns,
            interaction=False,
            rationale="",
        )


class DVarWithCol(BaseModel):
    description: str = Field(
        ..., title="Description of the dependent variable variable"
    )
    columns: List[str] = Field(
        ...,
        title="The column(s) in the FINAL dataframe used in the STATISTICAL MODEL that corresponds to the dependent variable",
    )

    def __str__(self):
        return f"DV: {self.description}."

    def to_cvar_spec(self) -> ConceptualVarSpec:
        return ConceptualVarSpec(
            variable_type="DV",
            concept=self.description,
            final_columns_derived=self.columns,
            interaction=False,
            rationale="",
        )


class ControlVarWithCol(BaseModel):
    description: str = Field(..., title="Description of the control variable variable")
    is_moderator: bool = Field(
        ...,
        title="Whether the variable is a moderator.",
    )
    moderator_on: Optional[str] = Field(
        "",
        title="The variable that the control variable is moderating. Only applicable for control variables that are moderators.",
    )
    columns: List[str] = Field(
        ...,
        title="The column(s) in the FINAL dataframe used in the STATISTICAL MODEL that corresponds to the control variable",
    )

    def __str__(self):
        if self.is_moderator:
            return f"Moderator: {self.description} interacting with {self.moderator_on} variable"
        else:
            return f"Control: {self.description}."

    def to_cvar_spec(self) -> ConceptualVarSpec:
        return ConceptualVarSpec(
            variable_type="Control",
            concept="" if self.description is None else self.description,
            interaction=bool(self.is_moderator),
            on="" if self.moderator_on is None else self.moderator_on,
            final_columns_derived=self.columns,
            rationale="",
        )


class AgentCVarsWithCol(BaseModel):
    ivs: List[IVarWithCol] = Field(..., title="Independent variables")
    dv: DVarWithCol = Field(..., title="Dependent variable")
    controls: List[ControlVarWithCol] = Field(..., title="Control variables")

    def to_cvar_specs(self) -> Dict[str, ConceptualVarSpec]:
        cv_specs = {}
        for iv in self.ivs:
            cv_spec = iv.to_cvar_spec()
            cv_specs[cv_spec.spec_id] = cv_spec
        for control in self.controls:
            cv_spec = control.to_cvar_spec()
            cv_specs[cv_spec.spec_id] = cv_spec
        cv_spec = self.dv.to_cvar_spec()
        cv_specs[cv_spec.spec_id] = cv_spec
        return cv_specs


class AgentCVars(BaseModel):
    ivs: List[IVar] = Field(..., title="Independent variables")
    dv: DVar = Field(..., title="Dependent variable")
    controls: List[ControlVar] = Field(..., title="Control variables")


class AnalysisCode(BaseModel):
    transform_code: str = Field(..., title="The code that transforms the data")
    m_code: str = Field(..., title="The code for the statistical modeling")


class AnalysisCodeWithCvarMapping(BaseModel):
    transform_code: str = Field(..., title="The code that transforms the data")
    m_code: str = Field(..., title="The code for the statistical modeling")
    col_to_cvar_mapping: Dict[str, int] = Field(
        ..., title="Mapping of columns to conceptual variables"
    )


class ModelAndColumns(BaseModel):
    m_spec: str = Field(..., title="The model specification")
    m_columns: List[str] = Field(..., title="The columns used in the model")

    def to_model_spec(self) -> ModelSpec:
        return ModelSpec(
            specification=self.m_spec,
            associated_columns_derived=self.m_columns,
        )


class EntireAnalysis(BaseModel):
    cvars: AgentCVarsWithCol = Field(..., title="Conceptual variables")
    transform_code: str = Field(..., title="The code that transforms the data")
    m_code: str = Field(..., title="The code for the statistical modeling")

    @property
    def transform_code_inside_func(self) -> str:
        try:
            function_codes, function_names = (
                extract_code_inside_functions_and_func_names(self.transform_code)
            )
            if not any("transform" in name for name in function_names):
                return self.transform_code
            else:
                var_name = get_function_arg_name(self.transform_code, "transform")
                if var_name and var_name != "df":
                    return replace_variable_name(function_codes[0], var_name, "df")
            return function_codes[0]
        except SyntaxError:
            return self.transform_code

    @property
    def model_code_inside_func(self) -> str:
        try:
            function_codes, function_names = (
                extract_code_inside_functions_and_func_names(self.m_code)
            )
            if not any("model" in name for name in function_names):
                return self.m_code
            return function_codes[0]
        except SyntaxError:
            return self.m_code


class EntireAnalysisProcessed(BaseModel):
    cv_specs: Dict[str, ConceptualVarSpec]
    m_specs: Dict[str, ModelSpec]
    transform_state: Union[TransformDatasetState, None] = None
    agent_cvars: AgentCVarsWithCol
    m_code_and_cols: ModelAndColumns

    def get_model_associated_cvars(
        self,
    ) -> Dict[str, List[ConceptualVarSpec]]:
        mspec_id_to_cvars = defaultdict(list)
        if isinstance(self.transform_state, TransformDatasetState):
            col_associated_og_cols = self.get_col_asssociated_orig_cols()
        else:
            col_associated_og_cols = {}
        for model_spec in self.m_specs.values():
            added = set()
            # matched cvar to model spec
            for col in model_spec.associated_columns_derived:
                for cvar in self.cv_specs.values():
                    if col in cvar.final_columns_derived and col not in added:
                        added.add(col)
                        if cvar not in mspec_id_to_cvars[model_spec.spec_id]:
                            mspec_id_to_cvars[model_spec.spec_id].append(cvar)

                # if cvar is og cols and model is derived cols
                if col in col_associated_og_cols:
                    og_cols = set(col_associated_og_cols[col])
                    for cvar in self.cv_specs.values():
                        cvar_cols = set(cvar.final_columns_derived)
                        if cvar_cols.issubset(og_cols) and col not in added:
                            added.add(col)
                            if cvar not in mspec_id_to_cvars[model_spec.spec_id]:
                                mspec_id_to_cvars[model_spec.spec_id].append(cvar)
                # if cvar is derived cols and model is og cols
                if col_associated_og_cols:
                    for cvar in self.cv_specs.values():
                        cvar_cols = set(cvar.final_columns_derived)
                        for col_cv in cvar_cols:
                            if col_cv in col_associated_og_cols:
                                cols_cv = col_associated_og_cols[col_cv]
                                if (
                                    len(cols_cv) == 1
                                    and col in cols_cv
                                    and col not in added
                                ):
                                    added.add(col)
                                    if (
                                        cvar
                                        not in mspec_id_to_cvars[model_spec.spec_id]
                                    ):
                                        mspec_id_to_cvars[model_spec.spec_id].append(
                                            cvar
                                        )

        return mspec_id_to_cvars

    def get_col_asssociated_orig_cols(self):
        col_g = self.transform_state.graphs[0].col_graph.nx_g
        nid_to_col = dict(
            list(
                list(self.transform_state.graphs.values())[0].col_graph.nx_g.nodes(
                    data="col"
                )
            )
        )

        def get_root_ancestors(node):
            ret = []
            for ancestor in nx.ancestors(col_g, node):
                if col_g.in_degree(ancestor) == 0:
                    ret.append(col_g.nodes[ancestor]["col"])
            return ret

        return {col: get_root_ancestors(nid) for nid, col in nid_to_col.items()}

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
