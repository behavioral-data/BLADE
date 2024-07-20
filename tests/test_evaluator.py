from unittest.mock import MagicMock, Mock, patch, AsyncMock
import pytest

from blade_bench.eval.datamodel.result import EvalResults
from blade_bench.eval.evaluator import Evaluator
from blade_bench.eval.exceptions import (
    LMSubmissionConversionError,
    LMSubmissionExecutionError,
    LoadGroundTruthError,
    MatchAnnotationsError,
    GetMetricsError,
)
from blade_bench.eval.convert import Convert
from blade_bench.eval.match.match_submission import SubmissionMatch
from blade_bench.llms.datamodel.gen_config import OpenAIGenConfig, TextGenConfig
from blade_bench.llms.textgen_openai import OpenAITextGenerator
from blade_bench.eval.datamodel import (
    EvalResult,
    RunResultModes,
    EntireAnalysis,
    EntireAnalysisProcessed,
    DatasetSubmission,
)

from .mock_data.hurricane_analysis import (
    HURRICANE_ANALYSIS,
    HURRICANE_ANALYSES_SUBMISSION,
)


analysis = EntireAnalysis(**HURRICANE_ANALYSIS)
dataset_submission = DatasetSubmission(**HURRICANE_ANALYSES_SUBMISSION)


# used to load these objets in Pydantic for EvalResult
class AnalysisMock(MagicMock, EntireAnalysis):
    pass


class AnalysisProcessedMock(MagicMock, EntireAnalysisProcessed):
    pass


@pytest.fixture
def textgen():
    textgen_config = TextGenConfig(n=1, temperature=0, max_tokens=1000, use_cache=True)
    config = OpenAIGenConfig(
        provider="azureopenai",
        api_key_env_name="OPENAI_AZURE_AGENTBENCH_EVAL_KEY",
        api_base="https://aagenteval.openai.azure.com/",
        api_version="2024-05-01-preview",
        deployment="gpt-4o-eval",
        model="gpt-4o",
        textgen_config=textgen_config,
    )
    return OpenAITextGenerator(config=config)


@pytest.fixture
def evaluator(textgen):
    return Evaluator(submission=dataset_submission, text_gen=textgen)


@pytest.mark.asyncio
async def test_process_error(evaluator):
    with patch.object(
        Convert, "convert_entire_analysis", side_effect=Exception("Error")
    ) as mock_method:
        with pytest.raises(LMSubmissionConversionError):
            await evaluator.process_analysis(analysis)


@pytest.mark.asyncio
async def test_process_analysis_error(evaluator):
    with patch.object(
        Convert,
        "get_state_data_from_code",
        side_effect=LMSubmissionExecutionError("testing"),
    ) as mock_method:
        assert evaluator.transform_run_result is None
        lm_state_data = await evaluator.process_analysis(analysis)
        assert evaluator.transform_run_result is not None
        assert lm_state_data.transform_state is None


@pytest.mark.asyncio
async def test_load_ground_truth_error(evaluator):
    with patch(
        "blade_bench.eval.evaluator.load_ground_truth",
    ) as mock_method:
        mock_method.side_effect = Exception("Error")
        with pytest.raises(LoadGroundTruthError):
            _ = await evaluator.load_ground_truth()
            mock_method.assert_called_once()


@pytest.mark.asyncio
async def test_match_annotations_error(evaluator):
    with patch.object(
        SubmissionMatch,
        "match_all",
        side_effect=Exception("Error"),
    ) as mock_method:
        mock_ground_truth = Mock()
        mock_analysis_processed = Mock()
        with pytest.raises(MatchAnnotationsError):
            _ = await evaluator.match_annotations(
                mock_ground_truth, mock_analysis_processed
            )


@pytest.mark.asyncio
async def test_get_metrics_error(evaluator):
    mock_metrics = Mock()
    with patch(
        "blade_bench.eval.evaluator.get_metrics_from_match_obj",
        side_effect=Exception("Error"),
    ) as mock_method:
        with pytest.raises(GetMetricsError):
            _ = await evaluator.get_metrics(mock_metrics)
            mock_method.assert_called_once()


@pytest.mark.asyncio
async def test_run_eval_handle_error(evaluator):
    with patch.object(
        Evaluator, "process_analysis", return_value=AnalysisProcessedMock()
    ) as mock_process:
        with patch.object(
            Evaluator, "load_ground_truth", return_value=Mock()
        ) as mock_load:
            with patch.object(
                SubmissionMatch,
                "match_all",
                side_effect=Exception("Error"),
            ) as mock_match:
                result: EvalResult = await evaluator.run_eval(AnalysisMock())
                assert result.analysis_processed is not None
                assert result.matched_annotations is None
                assert result.metrics is None
                assert (
                    result.eval_run_result.res_type
                    == RunResultModes.MATCHING_FAILED.value
                )


@pytest.mark.asyncio
async def test_evaluator_on_analyses(evaluator):
    result: EvalResults = await evaluator.run_eval_on_analyses()
    assert len(result.results) == len(dataset_submission.analyses)
    for res in result.results:
        assert res.analysis_processed is not None
        assert res.matched_annotations is not None
        assert res.metrics is not None


if __name__ == "__main__":
    pytest.main([__file__])
