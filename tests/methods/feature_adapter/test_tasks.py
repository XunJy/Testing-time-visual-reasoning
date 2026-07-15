from __future__ import annotations

import pytest
import torch

from ttvr.methods.feature_adapter import build_feature_task, stable_taxon_partition


def test_stable_taxon_partition_is_order_independent_disjoint_and_complete() -> None:
    taxon_ids = tuple(f"taxon-{index:02d}" for index in range(10))

    first_train, first_validation = stable_taxon_partition(
        taxon_ids,
        validation_fraction=0.2,
        salt="locked-protocol",
    )
    second_train, second_validation = stable_taxon_partition(
        tuple(reversed(taxon_ids)),
        validation_fraction=0.2,
        salt="locked-protocol",
    )

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 8
    assert len(first_validation) == 2
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train).union(first_validation) == set(taxon_ids)
    assert first_train == tuple(sorted(first_train))
    assert first_validation == tuple(sorted(first_validation))


@pytest.mark.parametrize(
    ("fraction", "expected_validation"),
    ((0.001, 1), (0.999, 4)),
)
def test_stable_taxon_partition_keeps_both_partitions_nonempty(
    fraction: float,
    expected_validation: int,
) -> None:
    training, validation = stable_taxon_partition(
        ("a", "b", "c", "d", "e"),
        validation_fraction=fraction,
        salt="edge-case",
    )

    assert len(validation) == expected_validation
    assert len(training) + len(validation) == 5


@pytest.mark.parametrize(
    ("taxon_ids", "fraction", "salt"),
    (
        (("only-one",), 0.2, "salt"),
        (("duplicate", "duplicate"), 0.2, "salt"),
        (("a", "b"), 0.0, "salt"),
        (("a", "b"), 1.0, "salt"),
        (("a", "b"), 0.2, ""),
    ),
)
def test_stable_taxon_partition_rejects_invalid_protocol_inputs(
    taxon_ids: tuple[str, ...],
    fraction: float,
    salt: str,
) -> None:
    with pytest.raises(ValueError):
        stable_taxon_partition(
            taxon_ids,
            validation_fraction=fraction,
            salt=salt,
        )


def test_build_feature_task_filters_rows_and_remaps_labels_without_reordering() -> None:
    local_taxon_ids = ("taxon-c", "taxon-a", "taxon-b", "taxon-d")
    features = torch.arange(18, dtype=torch.float32).reshape(6, 3).requires_grad_()
    labels = torch.tensor([3, 0, 2, 1, 3, 2], dtype=torch.long)
    prototypes = {
        "taxon-a": torch.tensor([1.0, 0.0, 0.0], requires_grad=True),
        "taxon-b": torch.tensor([0.0, 1.0, 0.0], requires_grad=True),
        "taxon-d": torch.tensor([0.0, 0.0, 1.0], requires_grad=True),
    }

    prepared = build_feature_task(
        "filtered-source",
        features,
        labels,
        local_taxon_ids,
        prototypes,
        included_taxon_ids={"taxon-a", "taxon-b", "taxon-d"},
        excluded_taxon_ids={"taxon-b"},
    )

    assert prepared.taxon_ids == ("taxon-a", "taxon-d")
    assert prepared.source_indices.tolist() == [0, 3, 4]
    assert prepared.task.labels.tolist() == [1, 0, 1]
    torch.testing.assert_close(
        prepared.task.features,
        features.detach().index_select(0, prepared.source_indices),
    )
    torch.testing.assert_close(
        prepared.task.text_prototypes,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert not prepared.task.features.requires_grad
    assert not prepared.task.text_prototypes.requires_grad
    assert prepared.task.features.is_contiguous()
    assert prepared.task.labels.is_contiguous()
    assert prepared.task.text_prototypes.is_contiguous()


def test_build_feature_task_requires_prototypes_only_for_retained_taxa() -> None:
    features = torch.eye(3).repeat_interleave(2, dim=0)
    labels = torch.arange(3).repeat_interleave(2)
    prototype_by_taxon = {
        "taxon-a": torch.tensor([1.0, 0.0, 0.0]),
        "taxon-b": torch.tensor([0.0, 1.0, 0.0]),
    }

    prepared = build_feature_task(
        "two-of-three",
        features,
        labels,
        ("taxon-a", "taxon-b", "taxon-c"),
        prototype_by_taxon,
        excluded_taxon_ids={"taxon-c"},
    )

    assert prepared.taxon_ids == ("taxon-a", "taxon-b")
    assert prepared.task.size == 4
    with pytest.raises(ValueError, match="missing text prototypes"):
        build_feature_task(
            "missing-retained-prototype",
            features,
            labels,
            ("taxon-a", "taxon-b", "taxon-c"),
            prototype_by_taxon,
            excluded_taxon_ids={"taxon-a"},
        )


def test_build_feature_task_rejects_filters_with_fewer_than_two_classes() -> None:
    with pytest.raises(ValueError, match="retain at least two"):
        build_feature_task(
            "one-class",
            torch.eye(3),
            torch.arange(3),
            ("a", "b", "c"),
            {"a": torch.eye(3)[0], "b": torch.eye(3)[1], "c": torch.eye(3)[2]},
            included_taxon_ids={"a"},
        )


def test_build_feature_task_excludes_rows_without_mutating_source_index_space() -> None:
    features = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    prototypes = {
        "a": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "b": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "c": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }

    prepared = build_feature_task(
        "deduplicated",
        features,
        labels,
        ("a", "b", "c"),
        prototypes,
        excluded_source_indices={0, 3},
    )

    assert prepared.taxon_ids == ("a", "b")
    assert prepared.source_indices.tolist() == [1, 2]
    assert prepared.task.labels.tolist() == [0, 1]


@pytest.mark.parametrize("indices", ({-1}, {3}, {0.5}))
def test_build_feature_task_rejects_invalid_excluded_source_indices(
    indices: set[object],
) -> None:
    with pytest.raises(ValueError, match="excluded_source_indices"):
        build_feature_task(
            "bad-indices",
            torch.eye(3),
            torch.arange(3),
            ("a", "b", "c"),
            {"a": torch.eye(3)[0], "b": torch.eye(3)[1], "c": torch.eye(3)[2]},
            excluded_source_indices=indices,  # type: ignore[arg-type]
        )
