import asyncio
import json
import os.path as osp

import pandas as pd
from blade_bench.logger import logger
from blade_bench.data.annotation import AnnotationDBData, get_annotation_data_from_df
from blade_bench.data.datamodel.mcq import MCQDatasetSimple
from blade_bench.data.dataset import list_datasets
from blade_bench.utils import (
    get_dataset_dir,
    get_dataset_annotations_path,
    get_dataset_csv_path,
    get_dataset_mcq_path,
)
import nest_asyncio

GROUND_TRUTH_FNAME = "annotations.pkl"


def load_mcq_dataset(dataset_name: str) -> MCQDatasetSimple:
    mcq_path = get_dataset_mcq_path(dataset_name)
    return MCQDatasetSimple(**json.load(open(mcq_path)))


def load_ground_truth(
    dataset_name: str, execute_output_dir: str = ".", load_from_pkl: bool = True
) -> AnnotationDBData:
    """
    Executes code in the local environment based on ground truth annotations to get
    all intermediate transform states.
    """

    adata_path = osp.join(get_dataset_dir(dataset_name), GROUND_TRUTH_FNAME)
    if not osp.exists(adata_path) or not load_from_pkl:
        logger.info(f"Loading ground truth transform states for {dataset_name}")
        gnd_truth_path = get_dataset_annotations_path(dataset_name)
        df = pd.read_csv(gnd_truth_path)
        adata = get_annotation_data_from_df(df)
        nest_asyncio.apply()
        asyncio.run(
            adata.get_state_data(
                get_dataset_csv_path(dataset_name),
                save_path=execute_output_dir,
                timeout=60,
            )
        )
        adata.save(adata_path)
    adata = AnnotationDBData.load(adata_path)
    return adata


def check_and_load_all_ground_truth():
    dataset_names = list_datasets()
    for dataset_name in dataset_names:
        load_ground_truth(dataset_name)
    logger.success("All ground truth loaded.")


if __name__ == "__main__":
    check_and_load_all_ground_truth()
