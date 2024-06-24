from abc import ABC, abstractmethod
from typing import Any

from blade_bench.data.annotation import AnnotationDBData
from blade_bench.utils import get_dataset_csv_path


class BaseMatcher(ABC):
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.dataset_path = get_dataset_csv_path(dataset_name)

    @abstractmethod
    def match_with_llm(self, adata: AnnotationDBData, llm: Any):
        pass
