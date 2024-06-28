from typing import Literal
from blade_bench.eval.datamodel.run import RunResultModes


class BladeBenchError(Exception):
    pass


class RunError(BladeBenchError):
    message: str
    res_type: RunResultModes

    def __init__(self, message: str, res_type: RunResultModes = None):
        super().__init__(message)
        self.message = message
        if res_type is not None:
            self.res_type = res_type


class LMGenerationError(RunError):
    res_type: Literal[RunResultModes.LM_GENERATION_FAILED] = (
        RunResultModes.LM_GENERATION_FAILED
    )


class LMSubmissionExecutionError(RunError):
    res_type: Literal[RunResultModes.LM_SUBMISSION_EXECUTION_FAILED] = (
        RunResultModes.LM_SUBMISSION_EXECUTION_FAILED
    )


class LMSubmissionConversionError(RunError):
    res_type: Literal[RunResultModes.LM_SUBMISSION_CONVERSION_FAILED] = (
        RunResultModes.LM_SUBMISSION_CONVERSION_FAILED
    )


class LMSubmissionEmptyError(RunError):
    res_type: Literal[RunResultModes.LM_SUBMISSION_TRANSFORM_CODE_EMPTY] = (
        RunResultModes.LM_SUBMISSION_TRANSFORM_CODE_EMPTY
    )


class LoadGroundTruthError(RunError):
    res_type: Literal[RunResultModes.LOAD_GND_TRUTH_FAILED] = (
        RunResultModes.LOAD_GND_TRUTH_FAILED
    )


class MatchAnnotationsError(RunError):
    res_type: Literal[RunResultModes.MATCHING_FAILED] = RunResultModes.MATCHING_FAILED


class GetMetricsError(RunError):
    res_type: Literal[RunResultModes.GETTING_METRICS_FAILED] = (
        RunResultModes.GETTING_METRICS_FAILED
    )


if __name__ == "__main__":
    try:
        raise LMGenerationError(
            "Submission failed", RunResultModes.LM_GENERATION_FAILED
        )
    except LMGenerationError as e:
        print(e.message)
        print(e.res_type)
        print("here")
