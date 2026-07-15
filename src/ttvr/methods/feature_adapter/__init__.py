"""Class-agnostic adaptation of frozen vision-language image features."""

from .model import ResidualFeatureAdapter, similarity_logits
from .tasks import PreparedFeatureTask, build_feature_task, stable_taxon_partition
from .training import (
    AdapterFitResult,
    AdapterRefitResult,
    AdapterTrainConfig,
    FeatureTask,
    TaskScore,
    ValidationSnapshot,
    evaluate_tasks,
    fit_feature_adapter,
    refit_feature_adapter,
    sample_class_balanced_indices,
    score_task,
)

__all__ = [
    "AdapterFitResult",
    "AdapterRefitResult",
    "AdapterTrainConfig",
    "FeatureTask",
    "PreparedFeatureTask",
    "ResidualFeatureAdapter",
    "TaskScore",
    "ValidationSnapshot",
    "evaluate_tasks",
    "fit_feature_adapter",
    "refit_feature_adapter",
    "build_feature_task",
    "sample_class_balanced_indices",
    "score_task",
    "similarity_logits",
    "stable_taxon_partition",
]
