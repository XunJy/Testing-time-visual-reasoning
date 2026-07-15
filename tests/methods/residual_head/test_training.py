from __future__ import annotations

import pytest
import torch

from ttvr.methods.residual_head import (
    LinearFeatureHead,
    ResidualHeadSearchConfig,
    StratifiedSplit,
    combine_residual_logits,
    head_logits_from_state,
    refit_feature_head,
    search_feature_head,
    stratified_hash_split,
)


def test_stratified_hash_split_is_balanced_disjoint_and_stable() -> None:
    labels = torch.arange(5).repeat_interleave(10)
    keys = [f"sample-{index}" for index in range(labels.numel())]

    first = stratified_hash_split(
        labels,
        keys,
        validation_per_class=2,
        seed=2026,
    )
    second = stratified_hash_split(
        labels,
        keys,
        validation_per_class=2,
        seed=2026,
    )

    assert torch.equal(first.fit_indices, second.fit_indices)
    assert torch.equal(first.validation_indices, second.validation_indices)
    assert first.fit_indices.numel() == 40
    assert first.validation_indices.numel() == 10
    assert set(first.fit_indices.tolist()).isdisjoint(first.validation_indices.tolist())
    covered = set(first.fit_indices.tolist()).union(first.validation_indices.tolist())
    assert covered == set(range(50))
    validation_labels = labels.index_select(0, first.validation_indices)
    assert torch.bincount(validation_labels, minlength=5).tolist() == [2, 2, 2, 2, 2]


def test_residual_alpha_zero_and_zero_initialisation_recover_baseline() -> None:
    generator = torch.Generator().manual_seed(9)
    features = torch.randn(4, 7, generator=generator)
    baseline = torch.randn(4, 5, generator=generator)
    head = LinearFeatureHead(feature_dim=7, class_count=5)

    residual = head(features)

    assert torch.count_nonzero(residual) == 0
    assert torch.equal(combine_residual_logits(baseline, residual, 1.0), baseline)
    assert torch.equal(combine_residual_logits(baseline, torch.ones_like(baseline), 0.0), baseline)


def test_linear_and_residual_heads_learn_separable_frozen_features() -> None:
    class_count = 5
    examples_per_class = 12
    labels = torch.arange(class_count).repeat_interleave(examples_per_class)
    features = torch.eye(class_count).repeat_interleave(examples_per_class, dim=0)
    keys = [f"class-{label}-sample-{index}" for index, label in enumerate(labels.tolist())]
    split = stratified_hash_split(
        labels,
        keys,
        validation_per_class=2,
        seed=3,
    )
    config = ResidualHeadSearchConfig(
        validation_per_class=2,
        max_epochs=40,
        patience=10,
        batch_size=16,
        learning_rates=(0.1,),
        weight_decays=(0.0,),
        alpha_grid=(0.0, 1.0),
        seed=3,
    )

    linear = search_feature_head(
        features,
        labels,
        split,
        config,
        mode="linear",
    )
    residual = search_feature_head(
        features,
        labels,
        split,
        config,
        mode="residual",
        base_logits=torch.zeros(labels.numel(), class_count),
    )

    assert linear.selection.validation.top1 == 100.0
    assert residual.selection.validation.top1 == 100.0
    state = refit_feature_head(
        features,
        labels,
        linear.selection,
        config,
    )
    logits = head_logits_from_state(features, state, class_count=class_count)
    assert torch.equal(logits.argmax(dim=1), labels)


def test_training_modes_reject_incompatible_base_logits() -> None:
    features = torch.eye(5).repeat_interleave(3, dim=0)
    labels = torch.arange(5).repeat_interleave(3)
    keys = [f"sample-{index}" for index in range(labels.numel())]
    split = stratified_hash_split(
        labels,
        keys,
        validation_per_class=1,
        seed=1,
    )
    config = ResidualHeadSearchConfig(
        validation_per_class=1,
        max_epochs=1,
        patience=1,
        batch_size=8,
        learning_rates=(0.1,),
        weight_decays=(0.0,),
        alpha_grid=(0.0, 1.0),
        seed=1,
    )

    with pytest.raises(ValueError, match="must not receive"):
        search_feature_head(
            features,
            labels,
            split,
            config,
            mode="linear",
            base_logits=torch.zeros(15, 5),
        )
    with pytest.raises(ValueError, match="requires"):
        search_feature_head(
            features,
            labels,
            split,
            config,
            mode="residual",
        )


def test_search_rejects_incomplete_split() -> None:
    labels = torch.arange(5).repeat_interleave(3)
    features = torch.eye(5).repeat_interleave(3, dim=0)
    split = StratifiedSplit(
        fit_indices=torch.arange(0, 10, dtype=torch.long),
        validation_indices=torch.arange(10, 14, dtype=torch.long),
    )
    config = ResidualHeadSearchConfig(
        validation_per_class=1,
        max_epochs=1,
        patience=1,
        learning_rates=(0.1,),
        weight_decays=(0.0,),
        alpha_grid=(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="cover every sample exactly once"):
        search_feature_head(features, labels, split, config, mode="linear")


def test_grid_order_does_not_change_search_result() -> None:
    labels = torch.arange(5).repeat_interleave(6)
    features = torch.eye(5).repeat_interleave(6, dim=0)
    keys = [f"sample-{index}" for index in range(labels.numel())]
    split = stratified_hash_split(labels, keys, validation_per_class=1, seed=4)
    common = {
        "validation_per_class": 1,
        "max_epochs": 3,
        "patience": 2,
        "batch_size": 8,
        "alpha_grid": (0.0, 1.0),
        "seed": 4,
    }

    first = search_feature_head(
        features,
        labels,
        split,
        ResidualHeadSearchConfig(
            learning_rates=(0.01, 0.1),
            weight_decays=(0.0, 0.01),
            **common,
        ),
        mode="linear",
    )
    second = search_feature_head(
        features,
        labels,
        split,
        ResidualHeadSearchConfig(
            learning_rates=(0.1, 0.01),
            weight_decays=(0.01, 0.0),
            **common,
        ),
        mode="linear",
    )

    assert first.selection == second.selection
    assert first.trials == second.trials
    assert all(
        torch.equal(first.validation_state[name], second.validation_state[name])
        for name in first.validation_state
    )
