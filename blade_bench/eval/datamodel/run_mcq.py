from typing import Dict
from pydantic import BaseModel, computed_field


class MCQMetrics(BaseModel):
    num_correct_transforms: int
    num_correct_cvars: int
    num_total_transforms: int
    num_total_cvars: int

    @computed_field
    def num_correct(self) -> int:
        return self.num_correct_transforms + self.num_correct_cvars
    
    @computed_field
    def num_total(self) -> int:
        return self.num_total_transforms + self.num_total_cvars
    
    @computed_field
    @property
    def transform_accuracy(self) -> float:
        return self.num_correct_transforms / max(self.num_total_transforms, 1)

    @computed_field
    @property
    def cvar_accuracy(self) -> float:
        return self.num_correct_cvars / max(self.num_total_cvars, 1)

    @computed_field
    @property
    def total_accuracy(self) -> float:
        return (self.num_correct_transforms + self.num_correct_cvars) / max(
            self.num_total_transforms + self.num_total_cvars, 1
        )


class MCQMetricsAcrossDatasets(BaseModel):
    dataset_metrics: Dict[str, MCQMetrics]

    @computed_field
    @property
    def num_correct(self) -> int:
        return sum([v.num_correct for v in self.dataset_metrics.values()])
    
    @computed_field
    @property
    def num_total(self) -> int:
        return sum([v.num_total for v in self.dataset_metrics.values()])
    
    @computed_field
    @property
    def num_datasets(self) -> int:
        return len(self.dataset_metrics)

    @computed_field
    @property
    def avg_transform_accuracy(self) -> float:
        return (
            sum([v.transform_accuracy for v in self.dataset_metrics.values()])
            / self.num_datasets
        )

    @computed_field
    @property
    def avg_cvar_accuracy(self) -> float:
        return (
            sum([v.cvar_accuracy for v in self.dataset_metrics.values()])
            / self.num_datasets
        )

    @computed_field
    @property
    def avg_total_accuracy(self) -> float:
        return (
            sum([v.total_accuracy for v in self.dataset_metrics.values()])
            / self.num_datasets
        )
