from typing import Dict, Literal, Union

from blade_bench.data.annotation import AnnotationDBData
from blade_bench.llms.base import TextGenerator
from blade_bench.llms.datamodel.gen_config import LLMHistory
from ..datamodel import (
    MatchedModels,
    MatchModel,
    MatchTransforms,
    MatchedAnnotations,
    MatchedCvars,
    MatchedTSpecs,
    EntireAnalysisProcessed,
)

from .conceptual_variable import CVarMatcher
from .model import StatsModelMatcher
from .transform import TransformMatcher

from blade_bench.llms import (
    OpenAIGenConfig,
    GeminiGenConfig,
    AnthropicGenConfig,
)


class SubmissionMatch:
    def __init__(
        self,
        dataset_name: str,
        llm_config: Union[OpenAIGenConfig, GeminiGenConfig, AnthropicGenConfig] = None,
        llm_history: LLMHistory = None,
        text_gen: TextGenerator = None,
        gnd_truth: AnnotationDBData = None,
        submission: EntireAnalysisProcessed = None,
    ):
        self.gnd_truth = gnd_truth
        self.submission: EntireAnalysisProcessed = submission
        self.transform_matcher = TransformMatcher(dataset_name=dataset_name)
        self.cvar_matcher = CVarMatcher(
            dataset_name,
            llm_config=llm_config,
            llm_history=llm_history,
            text_gen=text_gen,
        )
        self.smodel_matcher = StatsModelMatcher(
            dataset_name,
            llm_config=llm_config,
            llm_history=llm_history,
            text_gen=text_gen,
        )

    async def match_transforms(
        self,
        gnd_truth: AnnotationDBData = None,
        submission: EntireAnalysisProcessed = None,
    ) -> MatchTransforms:
        if submission:
            self.submission = submission
        if gnd_truth:
            self.gnd_truth = gnd_truth

        return await self.transform_matcher.match_with_llm(
            self.gnd_truth, self.submission.transform_state
        )

    def match_cvars(
        self,
        gnd_truth: AnnotationDBData = None,
        submission: EntireAnalysisProcessed = None,
    ) -> Dict[Literal["Control", "IV", "DV", "Moderator"], MatchedCvars]:
        if submission:
            self.submission = submission
        if gnd_truth:
            self.gnd_truth = gnd_truth

        return self.cvar_matcher.match_with_llm(
            self.gnd_truth, self.submission.agent_cvars
        )

    def match_statsmodel(
        self,
        gnd_truth: AnnotationDBData = None,
        submission: EntireAnalysisProcessed = None,
    ) -> MatchedModels:
        if submission:
            self.submission = submission
        if gnd_truth:
            self.gnd_truth = gnd_truth

        return self.smodel_matcher.match_with_llm(self.gnd_truth, self.submission)

    async def match_all(
        self, gnd_truth: AnnotationDBData, submission: EntireAnalysisProcessed = None
    ) -> MatchedAnnotations:
        if submission:
            self.submission = submission
        if gnd_truth:
            self.gnd_truth = gnd_truth

        if submission.transform_state is None:
            transform_match = MatchTransforms(
                transform_state1=None,
                transform_state2=None,
                matched_tspecs=MatchedTSpecs(
                    vspecs1=[],
                    vspecs2=[],
                    gspecs1=[],
                    gspecs2=[],
                    cat_specs1=[],
                    cat_specs2=[],
                ),
            )
        else:
            transform_match = await self.match_transforms()
        cvar_match = self.match_cvars()
        smodel_match = self.match_statsmodel()
        return MatchedAnnotations(
            matched_models=smodel_match,
            matched_transforms=transform_match,
            matched_cvars=cvar_match,
        )


if __name__ == "__main__":
    pass
