import asyncio
import os.path as osp

import pandas as pd
from blade_bench.data.annotation import AnnotationDBData, get_annotation_data_from_df
from blade_bench.utils import (
    get_dataset_dir,
    get_dataset_annotations_path,
    get_dataset_csv_path,
)

GROUND_TRUTH_FNAME = "annotations.pkl"


def load_ground_truth_data(
    dataset_name: str, execute_output_dir: str, load_from_pkl: bool = True
) -> AnnotationDBData:
    adata_path = osp.join(get_dataset_dir(dataset_name), GROUND_TRUTH_FNAME)
    if not osp.exists(adata_path) or not load_from_pkl:
        gnd_truth_path = get_dataset_annotations_path(dataset_name)
        df = pd.read_csv(gnd_truth_path)
        adata = get_annotation_data_from_df(df)
        asyncio.run(
            adata.get_state_data(
                get_dataset_csv_path(dataset_name),
                save_path=execute_output_dir,
            )
        )
        adata.save(adata_path)
    adata = AnnotationDBData.load(adata_path)
    return adata


if __name__ == "__main__":
    adata = load_ground_truth_data("hurricane", ".", load_from_pkl=False)
    print("here")
