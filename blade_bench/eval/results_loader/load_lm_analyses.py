from glob import glob
import json
from typing import List
from blade_bench.eval.datamodel.lm_analysis import EntireAnalysis
from blade_bench.eval.datamodel.submission import DatasetSubmission


def load_lm_analyses_glob(file_glob_str: str, dataset_name: str):
    file_paths = glob(file_glob_str)
    return load_lm_analyses(file_paths, dataset_name)


def load_lm_analyses(file_paths: List[str], dataset_name: str):
    ret = []
    for file_path in file_paths:
        analysis = EntireAnalysis(**json.load(open(file_path)))
        ret.append(analysis)
    return DatasetSubmission(dataset_name=dataset_name, analyses=ret)
