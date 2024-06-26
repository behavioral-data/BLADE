import json
from typing import Any, Dict, Literal, Union
from blade_bench.data.datamodel import ConceptualVarSpec
from blade_bench.llms import (
    OpenAIGenConfig,
    GeminiGenConfig,
    AnthropicGenConfig,
)
from blade_bench.data.annotation import AnnotationDBData
from ..llm.conceptual_var_similarity import ConceptualVarSimilarity
from .base import BaseMatcher
from ..datamodel import MatchedCvars, AgentCVarsWithCol

from blade_bench.utils import get_dataset_info_path
from blade_bench.data.dataset import get_dataset_info, DatasetInfo


class CVarMatcher(BaseMatcher):
    def __init__(
        self,
        dataset_name: str,
        llm_config: Union[OpenAIGenConfig, GeminiGenConfig, AnthropicGenConfig] = None,
    ):
        super().__init__(dataset_name)
        self.dinfo_path = get_dataset_info_path(dataset_name)
        self.dinfo: DatasetInfo = get_dataset_info(dataset_name)
        if llm_config is None:
            self.cvar_compare_llm = ConceptualVarSimilarity.init_from_base_llm_config()
        else:
            self.cvar_compare_llm = ConceptualVarSimilarity.init_from_llm_config(
                llm_config
            )

    def match_annotator_data(
        self, adata1: AnnotationDBData, adata2: AnnotationDBData
    ) -> Dict[str, MatchedCvars]:
        cv_specs_by_type1, cv_specs_by_type2 = (
            adata1.get_cvars_by_type(),
            adata2.get_cvars_by_type(),
        )
        res = {}
        for vtype, v in cv_specs_by_type1.items():
            if vtype not in cv_specs_by_type2:
                continue
            res[vtype] = self.match_cv_spec(v, cv_specs_by_type2[vtype])
        return res

    def match_with_llm(
        self, adata: AnnotationDBData, llm: AgentCVarsWithCol
    ) -> Dict[Literal["Control", "IV", "DV", "Moderator"], MatchedCvars]:
        cv_specs_by_type = adata.get_cvars_by_type()
        res = {}
        for vtype, cv_specs in cv_specs_by_type.items():
            v_gnd = [str(cv_spec) for cv_spec in cv_specs.values()]
            if vtype == "Control":
                v_llm = [str(var) for var in llm.controls if not var.is_moderator]
            elif vtype == "Moderator":
                v_llm = [str(var) for var in llm.controls if var.is_moderator]
            elif vtype == "IV":
                v_llm = [str(var) for var in llm.ivs]
            elif vtype == "DV":
                v_llm = [str(llm.dv)]
            if not v_llm:
                continue
            res[vtype] = self.cvar_compare_llm.match_variables_str(
                dataset_info=self.dinfo,
                var_list1=v_gnd,
                var_list2=v_llm,
            )
        return res

    def match_cv_spec(
        self,
        cv_specs1: Dict[str, ConceptualVarSpec],
        cv_specs2: Dict[str, ConceptualVarSpec],
    ):
        c1 = list(cv_specs1.values())
        c2 = list(cv_specs2.values())

        res = self.cvar_compare_llm.match_variables_cvarspec(
            dataset_info=self.dinfo,
            var_list1=c1,
            var_list2=c2,
        )
        return res


if __name__ == "__main__":
    pass
