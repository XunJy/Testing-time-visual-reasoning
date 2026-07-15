from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from ttvr.methods.feature_adapter import (
    AdapterTrainConfig,
    FeatureTask,
    ResidualFeatureAdapter,
    fit_feature_adapter,
    refit_feature_adapter,
    sample_class_balanced_indices,
    score_task,
    similarity_logits,
)


def _unit_vectors(degrees: list[float]) -> torch.Tensor:
    radians = torch.tensor(degrees, dtype=torch.float32) * math.pi / 180.0
    return torch.stack((radians.cos(), radians.sin()), dim=1)


def _rotated_task(
    name: str,
    class_directions: list[float],
    *,
    image_rotation: float,
    examples_per_class: int = 8,
) -> FeatureTask:
    prototypes = _unit_vectors(class_directions)
    features = _unit_vectors(
        [direction + image_rotation for direction in class_directions]
    ).repeat_interleave(examples_per_class, dim=0)
    labels = torch.arange(len(class_directions)).repeat_interleave(examples_per_class)
    return FeatureTask(
        name=name,
        features=features,
        labels=labels,
        text_prototypes=prototypes,
    )


def test_zero_initialisation_is_identity_and_vocabulary_agnostic() -> None:
    torch.manual_seed(4)
    adapter = ResidualFeatureAdapter(feature_dim=7, hidden_dim=3)
    features = torch.randn(5, 7)

    adapted = adapter(features)

    torch.testing.assert_close(adapted, functional.normalize(features, dim=1))
    assert torch.count_nonzero(adapter.residual(features)) == 0
    assert similarity_logits(adapted, torch.randn(2, 7), logit_scale=3.0).shape == (5, 2)
    assert similarity_logits(adapted, torch.randn(19, 7), logit_scale=3.0).shape == (5, 19)
    assert sum(parameter.numel() for parameter in adapter.parameters()) == (7 * 3 + 3) + (
        3 * 7 + 7
    )


def test_class_balanced_sampler_does_not_follow_image_frequency() -> None:
    labels = torch.tensor([0] + [1] * 2 + [2] * 100, dtype=torch.long)
    task = FeatureTask(
        name="imbalanced",
        features=torch.randn(labels.numel(), 4),
        labels=labels,
        text_prototypes=torch.randn(3, 4),
    )
    first_generator = torch.Generator().manual_seed(73)
    second_generator = torch.Generator().manual_seed(73)

    first = sample_class_balanced_indices(
        task,
        batch_size=6_000,
        generator=first_generator,
    )
    second = sample_class_balanced_indices(
        task,
        batch_size=6_000,
        generator=second_generator,
    )
    sampled_counts = torch.bincount(labels.index_select(0, first), minlength=3)

    assert torch.equal(first, second)
    assert int(sampled_counts.max() - sampled_counts.min()) < 150
    assert sampled_counts.sum().item() == 6_000


def test_evaluation_is_invariant_to_positive_feature_scaling() -> None:
    features = torch.tensor(
        [
            [1.0, 0.2],
            [-0.3, 1.0],
            [-1.0, -0.2],
            [0.3, -1.0],
        ]
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    original = FeatureTask("original", features, labels, prototypes)
    scaled = FeatureTask(
        "scaled",
        features * torch.tensor([[0.1], [7.0], [19.0], [0.4]]),
        labels,
        prototypes,
    )
    adapter = ResidualFeatureAdapter(feature_dim=2, hidden_dim=3)
    with torch.no_grad():
        adapter.down.weight.copy_(torch.tensor([[0.5, -0.2], [0.1, 0.7], [-0.4, 0.3]]))
        adapter.down.bias.copy_(torch.tensor([0.2, -0.1, 0.4]))
        adapter.up.weight.copy_(torch.tensor([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]]))
        adapter.up.bias.copy_(torch.tensor([0.05, -0.03]))

    first = score_task(adapter, original, logit_scale=11.0, device=torch.device("cpu"))
    second = score_task(adapter, scaled, logit_scale=11.0, device=torch.device("cpu"))

    assert first.top1 == second.top1
    assert first.class_balanced_top1 == second.class_balanced_top1
    assert first.cross_entropy == pytest.approx(second.cross_entropy, abs=1e-6)


def test_shared_adapter_learns_rotation_and_transfers_to_unseen_vocabulary() -> None:
    train = _rotated_task(
        "source-cardinal-species",
        [0.0, 90.0, 180.0, 270.0],
        image_rotation=60.0,
    )
    validation = _rotated_task(
        "unseen-diagonal-species",
        [45.0, 135.0, 225.0, 315.0],
        image_rotation=60.0,
    )
    torch.manual_seed(99)
    adapter = ResidualFeatureAdapter(feature_dim=2, hidden_dim=16)
    initial = score_task(
        adapter,
        validation,
        logit_scale=20.0,
        device=torch.device("cpu"),
    )
    config = AdapterTrainConfig(
        steps=80,
        validation_interval=10,
        patience_intervals=3,
        batch_size=32,
        learning_rate=0.01,
        weight_decay=0.0,
        identity_weight=0.0,
        logit_scale=20.0,
        seed=7,
    )

    result = fit_feature_adapter(
        adapter,
        (train,),
        (validation,),
        config,
        device="cpu",
    )
    restored = ResidualFeatureAdapter(feature_dim=2, hidden_dim=16)
    restored.load_state_dict(result.state_dict)
    final = score_task(
        restored,
        validation,
        logit_scale=20.0,
        device=torch.device("cpu"),
    )

    assert initial.top1 == 0.0
    assert result.best_validation.tasks[0].name == "unseen-diagonal-species"
    assert result.best_validation.tasks[0].top1 == 100.0
    assert final.top1 == 100.0
    assert result.best_step > 0
    assert sum(result.train_task_draws.values()) < config.steps
    assert all(value.device.type == "cpu" for value in result.state_dict.values())


def test_fit_seed_covers_initialisation_sampling_and_updates() -> None:
    train = _rotated_task(
        "train",
        [0.0, 90.0, 180.0, 270.0],
        image_rotation=60.0,
        examples_per_class=3,
    )
    validation = _rotated_task(
        "validation",
        [45.0, 135.0, 225.0, 315.0],
        image_rotation=60.0,
        examples_per_class=2,
    )
    config = AdapterTrainConfig(
        steps=6,
        validation_interval=3,
        patience_intervals=3,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        identity_weight=0.0,
        logit_scale=20.0,
        seed=41,
    )
    torch.manual_seed(1)
    first_adapter = ResidualFeatureAdapter(feature_dim=2, hidden_dim=7)
    torch.manual_seed(9_999)
    second_adapter = ResidualFeatureAdapter(feature_dim=2, hidden_dim=7)

    first = fit_feature_adapter(
        first_adapter,
        (train,),
        (validation,),
        config,
        device="cpu",
    )
    second = fit_feature_adapter(
        second_adapter,
        (train,),
        (validation,),
        config,
        device="cpu",
    )

    assert first.history == second.history
    assert first.train_task_draws == second.train_task_draws
    assert first.state_dict.keys() == second.state_dict.keys()
    for name in first.state_dict:
        assert torch.equal(first.state_dict[name], second.state_dict[name])


def test_refit_seed_covers_initialisation_sampling_and_exact_step_count() -> None:
    train = _rotated_task(
        "all-source-species",
        [0.0, 90.0, 180.0, 270.0],
        image_rotation=35.0,
        examples_per_class=3,
    )
    config = AdapterTrainConfig(
        steps=99,
        validation_interval=3,
        patience_intervals=3,
        batch_size=8,
        learning_rate=0.01,
        weight_decay=0.0,
        identity_weight=0.1,
        logit_scale=20.0,
        seed=123,
    )
    torch.manual_seed(1)
    first_adapter = ResidualFeatureAdapter(feature_dim=2, hidden_dim=7)
    torch.manual_seed(9_999)
    second_adapter = ResidualFeatureAdapter(feature_dim=2, hidden_dim=7)

    first = refit_feature_adapter(
        first_adapter,
        (train,),
        config,
        steps=7,
        device="cpu",
    )
    second = refit_feature_adapter(
        second_adapter,
        (train,),
        config,
        steps=7,
        device="cpu",
    )

    assert first.steps == second.steps == 7
    assert first.train_task_draws == second.train_task_draws == {
        "all-source-species": 7
    }
    assert first.state_dict.keys() == second.state_dict.keys()
    for name in first.state_dict:
        assert torch.equal(first.state_dict[name], second.state_dict[name])


def test_feature_task_rejects_a_declared_class_without_images() -> None:
    with pytest.raises(ValueError, match="every local class"):
        FeatureTask(
            name="missing-class",
            features=torch.randn(4, 3),
            labels=torch.tensor([0, 0, 2, 2]),
            text_prototypes=torch.randn(3, 3),
        )
