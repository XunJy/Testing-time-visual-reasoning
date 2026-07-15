"""Frozen-feature supervised residual heads."""

from .training import (
    HeadSearchResult,
    HeadSelection,
    HeadTrial,
    LinearFeatureHead,
    ResidualHeadSearchConfig,
    StratifiedSplit,
    ValidationScore,
    combine_residual_logits,
    evaluate_logits,
    head_logits_from_state,
    refit_feature_head,
    search_feature_head,
    stable_split_score,
    stratified_hash_split,
)

__all__ = [
    "HeadSearchResult",
    "HeadSelection",
    "HeadTrial",
    "LinearFeatureHead",
    "ResidualHeadSearchConfig",
    "StratifiedSplit",
    "ValidationScore",
    "combine_residual_logits",
    "evaluate_logits",
    "head_logits_from_state",
    "refit_feature_head",
    "search_feature_head",
    "stable_split_score",
    "stratified_hash_split",
]
