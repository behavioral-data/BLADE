from typing import Dict
from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis
from blade_bench.eval.datamodel.submission import DatasetSubmission
from blade_bench.logger import logger
from tqdm import tqdm


def eval_submissions_across_datasets(
    datasets_submissions: Dict[str, DatasetSubmission]
):
    for dataset_name, submission in tqdm(
        datasets_submissions.items(), desc="Evaluating datasets"
    ):
        eval_submission(submission)
    # TODO gather results on these


def eval_submission(submission: DatasetSubmission):
    analyses = submission.analyses
    num_analyses = len(analyses)
    logger.info(f"Got {num_analyses} analyses for dataset {submission.dataset_name}")
    for analysis in analyses:
        eval_analysis(analysis)
    # TODO gather results on these


def eval_analysis(analysis: EntireAnalysis):
    logger.info(f"Running evaluation for analysis {analysis.analysis_id}")
    pass


async def run(self):
    analysis_processed = await self.process_analysis(analysis)
    if isinstance(analysis_processed, RunResults):
        return analysis_processed
    gnd_truth = await self.load_ground_truth()
    if isinstance(gnd_truth, RunResults):
        return gnd_truth
    matched_annotations = await self.match_annotations(gnd_truth, analysis_processed)
    if isinstance(matched_annotations, RunResults):
        return matched_annotations
    logger.success(f"Got matched annotations")

    match_metrics = await self.get_metrics(matched_annotations)
    if isinstance(match_metrics, RunResults):
        return match_metrics
