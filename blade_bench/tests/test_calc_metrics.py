import json
import os.path as osp

import pytest

from blade_bench.eval.datamodel.result import EvalResults
from blade_bench.eval.datamodel.run import MetricsAcrossRuns
from blade_bench.eval.metrics.calc_metrics import CalcSubmissionMetrics

cur_folder = osp.dirname(osp.abspath(__file__))
mock_data_folder = osp.join(cur_folder, "mock_data")


@pytest.fixture
def eval_results():
    with open(osp.join(mock_data_folder, "eval_res.json"), "r") as f:
        eval_results = EvalResults(**json.load(f))
    return eval_results


def test_calc_metrics(eval_results):
    calc = CalcSubmissionMetrics(submission=eval_results)
    diversity = calc.calculate_diversity(ks=[5, 8, 10], num_samples=100)
    metrics: MetricsAcrossRuns = calc.calculate_metrics()
    assert metrics is not None
    assert len(metrics.num_cvars2) == len(eval_results.results)
    assert len(metrics.num_match_gspec2) == len(eval_results.results)
    assert len(metrics.num_match_vspec2) == len(eval_results.results)
    assert len(metrics.num_tspecs2) == len(eval_results.results)
    assert len(metrics.num_match_vspec) == len(eval_results.results)
    assert len(metrics.num_match_model_cvar) == len(eval_results.results)


if __name__ == "__main__":
    pytest.main([__file__])
