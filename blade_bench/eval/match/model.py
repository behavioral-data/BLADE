import json
from typing import Any, Dict, List, Tuple, Union
from collections import defaultdict
from blade_bench.data.datamodel import ModelSpec
from blade_bench.data.annotation import AnnotationDBData

from blade_bench.llms import (
    OpenAIGenConfig,
    GeminiGenConfig,
    AnthropicGenConfig,
)
from blade_bench.llms.base import TextGenerator
from blade_bench.llms.datamodel.gen_config import GenConfig, LLMHistory
from ..llm import ConceptualVarSimilarity, StatsModelSimilarity
from ..llm.model_similarity import StatsModelSimilarity
from .base import BaseMatcher
from ..datamodel import MatchedModels, MatchModel, MatchedCvars, EntireAnalysisProcessed

from blade_bench.utils import get_dataset_info_path
from blade_bench.data.dataset import load_dataset_info, DatasetInfo


class StatsModelMatcher(BaseMatcher):
    def __init__(
        self,
        dataset_name: str,
        llm_config: GenConfig = None,
        llm_history: LLMHistory = None,
        text_gen: TextGenerator = None,
    ):
        super().__init__(dataset_name)
        self.dinfo_path = get_dataset_info_path(dataset_name)
        self.dinfo: DatasetInfo = load_dataset_info(dataset_name)
        self.__init_llms(llm_config, llm_history, text_gen)

    def __init_llms(
        self, llm_config: GenConfig, llm_history: LLMHistory, text_gen: TextGenerator
    ):
        if llm_config is None and text_gen is None:
            raise ValueError("llm_config or text_gen must be provided")
        if text_gen is not None:
            self.smodel_compare_llm = StatsModelSimilarity(
                text_gen, history=llm_history
            )
            self.cvar_compare_llm = ConceptualVarSimilarity(
                text_gen, history=llm_history
            )
        else:
            self.smodel_compare_llm = StatsModelSimilarity.init_from_llm_config(
                llm_config, history=llm_history
            )
            self.cvar_compare_llm = ConceptualVarSimilarity.init_from_llm_config(
                llm_config, history=llm_history
            )

    def match_annotator_data(
        self, adata1: AnnotationDBData, adata2: AnnotationDBData
    ) -> MatchedModels:
        model_specs1, mspec_id_to_cvars1 = (
            adata1.m_specs,
            adata1.get_model_associated_cvars(),
        )
        model_specs2, mspec_id_to_cvars2 = (
            adata2.m_specs,
            adata2.get_model_associated_cvars(),
        )
        m1 = list(model_specs1.values())
        m2 = list(model_specs2.values())
        llm_compare = self.smodel_compare_llm.match_models(m1, m2)

        matched_models = MatchedModels(
            input_models1=m1,
            input_models2=m2,
            mspec_id_to_cvars1=mspec_id_to_cvars1,
            mspec_id_to_cvars2=mspec_id_to_cvars2,
            matched=llm_compare,
        )

        self.get_matched_cvars_for_model_match(matched_models)
        return matched_models

        # TODO beyond just the model spec name/code -> we can also match on the associated cvars, or the associated exact transform values
        # TODO need to match on associated_columns_leaf_spec_ids, or associated_specified_spec_ids

    def get_matched_cvars_for_model_match(
        self,
        matched_models: MatchedModels,
    ):
        for match_model in matched_models.matched.values():
            mspec1, mspec2 = match_model.model1, match_model.model2
            cvars1 = matched_models.mspec_id_to_cvars1[mspec1.spec_id]
            cvars2 = matched_models.mspec_id_to_cvars2.get(mspec2.spec_id, [])
            if not cvars2:
                match_res = {
                    "Control": MatchedCvars(
                        input_vars1=cvars1, input_vars2=[], matched={}
                    )
                }
            else:
                cvars_by_type1 = defaultdict(list)
                cvars_by_type2 = defaultdict(list)
                for cvar in cvars1:
                    cvars_by_type1[cvar.variable_type].append(cvar)

                for cvar in cvars2:
                    cvars_by_type2[cvar.variable_type].append(cvar)

                matched_cvars = {}
                for cvars1_type, cvars1_list in cvars_by_type1.items():
                    cvars2_list = cvars_by_type2.get(cvars1_type, [])
                    if not cvars2_list:
                        continue
                    match_res = self.cvar_compare_llm.match_variables_cvarspec(
                        dataset_info=self.dinfo,
                        var_list1=cvars1_list,
                        var_list2=cvars2_list,
                    )
                    matched_cvars[cvars1_type] = match_res
                match_model.matched_cvars = matched_cvars

    def get_transforms_from_spec(self, spec: ModelSpec):
        tspecified = spec.associated_specified_spec_ids
        if tspecified:
            pass

    def match_with_llm(
        self, adata: AnnotationDBData, llm: EntireAnalysisProcessed
    ) -> MatchedModels:
        model_specs1 = adata.m_specs
        mspec_id_to_cvars1 = adata.get_model_associated_cvars()

        model_specs2 = llm.m_specs
        mspec_id_to_cvars2 = llm.get_model_associated_cvars()

        m1 = list(model_specs1.values())

        mname_to_mspecs = defaultdict(list)
        for mspec in m1:
            mname_to_mspecs[mspec.specification.strip().lower()].append(mspec)
        m1_unique = [mname_to_mspecs[mname][0] for mname in mname_to_mspecs]
        m2 = list(model_specs2.values())

        llm_compare = self.smodel_compare_llm.match_models(m1_unique, m2)
        all_matched = {}
        for k, v in llm_compare.items():
            m1_spec = m1_unique[k[0] - 1]
            all_matched[k] = v
            for other_m1_spec in mname_to_mspecs[m1_spec.specification.strip().lower()][
                1:
            ]:
                all_matched[(m1.index(other_m1_spec) + 1, k[1])] = MatchModel(
                    model1=m1_spec,
                    model2=v.model2,
                    rationale="",
                )

        matched_models = MatchedModels(
            input_models1=m1,
            input_models2=m2,
            mspec_id_to_cvars1=mspec_id_to_cvars1,
            mspec_id_to_cvars2=mspec_id_to_cvars2,
            matched=all_matched,
            matched_unique=llm_compare,
            input_models1_unique=m1_unique,
        )
        self.get_matched_cvars_for_model_match(matched_models)
        return matched_models


if __name__ == "__main__":
    pass
