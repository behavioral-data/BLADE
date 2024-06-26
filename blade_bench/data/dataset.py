import json
import os
import os.path as osp
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from blade_bench.utils import get_dataset_info_path, get_datasets_dir


class DatasetInfo(BaseModel):
    research_questions: List[str]
    data_desc: Optional[Dict[str, Any]] = None

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
    return [d for d in os.listdir(datasets_dir) if osp.isdir(osp.join(datasets_dir, d))]


def get_dataset_info(dataset: str):
    data_info_path = get_dataset_info_path(dataset)
    if not osp.exists(data_info_path):
        raise FileNotFoundError(f"Dataset info file not found: {data_info_path}")
    return DatasetInfo(**json.load(open(data_info_path)))
