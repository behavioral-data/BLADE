import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict
from blade_bench.data.annotation import get_annotation_data_from_df
from blade_bench.utils import (
    get_dataset_info_path,
    get_datasets_dir,
    get_dataset_csv_path,
    get_dataset_annotations_path,
)


class DatasetInfo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    research_questions: List[str]
    data_desc: Optional[Dict[str, Any]] = None
    df: Optional[pd.DataFrame] = None

    @property
    def research_question(self):
        return self.research_questions[0]

    @property
    def data_desc_no_desc(self):
        if self.data_desc is None:
            return None
        ret = {**self.data_desc}
        ret.pop("dataset_description", None)
        return ret

    @property
    def data_desc_no_semantic_type(self):
        if self.data_desc is None:
            return None
        ret = {**self.data_desc}
        for f in ret["fields"]:
            f["properties"].pop("semantic_type", None)
        return ret

    @property
    def data_desc_no_desc_no_semantic_type(self):
        if self.data_desc is None:
            return None
        ret = {**self.data_desc}
        ret.pop("dataset_description", None)
        for f in ret["fields"]:
            f["properties"].pop("semantic_type", None)
        return ret


def list_datasets():
    datasets_dir = get_datasets_dir()
    return [
        d
        for d in os.listdir(datasets_dir)
        if osp.isdir(osp.join(datasets_dir, d))
        if d != "toy" and osp.exists(osp.join(datasets_dir, d, "annotations.csv"))
    ]


def list_datasets_mcq():
    datasets_dir = get_datasets_dir()
    return [
        d
        for d in os.listdir(datasets_dir)
        if osp.isdir(osp.join(datasets_dir, d))
        if osp.exists(osp.join(datasets_dir, d, "mcq_dataset.json"))
    ]


def load_dataset_info(dataset: str, load_df=False):
    data_info_path = get_dataset_info_path(dataset)
    if not osp.exists(data_info_path):
        raise FileNotFoundError(f"Dataset info file not found: {data_info_path}")
    dinfo = DatasetInfo(**json.load(open(data_info_path)))
    if load_df:
        df_path = get_dataset_csv_path(dataset)
        df = pd.read_csv(df_path)
        dinfo.df = df
    return dinfo


def gen_datasets_jsonl():
    ret = []
    for dataset in list_datasets():
        dinfo = load_dataset_info(dataset)
        annotation_path = get_dataset_annotations_path(dataset)
        df = pd.read_csv(annotation_path)
        adata = get_annotation_data_from_df(df)

        ret.append(
            {
                "dataset": dataset,
                "research_question": dinfo.research_questions[0],
                "dinfo": dinfo.model_dump_json(),
                "model_specs": json.dumps(
                    {k: v.model_dump_json() for k, v in adata.m_specs.items()}
                ),
                "transform_specs": json.dumps(
                    {k: v.model_dump_json() for k, v in adata.transform_specs.items()}
                ),
                "cv_specs": json.dumps(
                    {k: v.model_dump_json() for k, v in adata.cv_specs.items()}
                ),
            }
        )
    return ret