from __future__ import annotations

import pytest
import torch

from ttvr.metrics import (
    compute_topk_accuracy,
    compute_transfer_counts,
    ordered_predictions,
    topk_hits,
)


def test_ordered_predictions_uses_stable_descending_order() -> None:
    logits = torch.tensor(
        [
            [0.5, 0.5, -1.0],
            [-2.0, 3.0, 1.0],
        ]
    )

    predictions = ordered_predictions(logits)

    assert torch.equal(predictions, torch.tensor([[0, 1, 2], [1, 2, 0]]))


def test_ordered_predictions_maps_shared_candidates_back_to_global_labels() -> None:
    logits = torch.tensor(
        [
            [-1.0, 3.0, 2.0],
            [5.0, 5.0, 0.0],
        ]
    )
    candidate_ids = torch.tensor([17, 4, 99])

    predictions = ordered_predictions(logits, candidate_ids)

    assert torch.equal(predictions, torch.tensor([[4, 99, 17], [17, 4, 99]]))


def test_ordered_predictions_supports_per_image_candidate_mapping() -> None:
    logits = torch.tensor([[0.1, 0.9, 0.4], [0.8, 0.2, 0.3]])
    candidate_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])

    predictions = ordered_predictions(logits, candidate_ids)

    assert torch.equal(predictions, torch.tensor([[11, 12, 10], [20, 22, 21]]))


def test_topk_hits_returns_one_boolean_per_sample() -> None:
    predictions = torch.tensor(
        [
            [5, 1, 9],
            [0, 1, 2],
            [4, 3, 8],
        ]
    )
    targets = torch.tensor([1, 2, 7])

    assert torch.equal(topk_hits(predictions, targets, k=1), torch.tensor([False] * 3))
    assert torch.equal(
        topk_hits(predictions, targets, k=2),
        torch.tensor([True, False, False]),
    )
    assert torch.equal(
        topk_hits(predictions, targets, k=3),
        torch.tensor([True, True, False]),
    )


def test_topk_accuracy_reports_counts_and_percentages() -> None:
    predictions = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 5, 4],
            [0, 1, 3, 4, 5, 2],
        ]
    )
    targets = torch.tensor([0, 5, 2])

    metrics = compute_topk_accuracy(predictions, targets)

    assert metrics.total == 3
    assert metrics.top1_correct == 1
    assert metrics.top5_correct == 2
    assert metrics.top1 == pytest.approx(100 / 3)
    assert metrics.top5 == pytest.approx(200 / 3)
    assert metrics.to_dict()["top1_correct"] == 1


def test_transfer_counts_partition_all_samples() -> None:
    targets = torch.tensor([0, 1, 2, 3, 4])
    baseline_predictions = torch.tensor(
        [
            [0, 5],  # both correct
            [9, 1],  # recovered
            [2, 8],  # degraded
            [8, 3],  # recovered
            [7, 4],  # both wrong
        ]
    )
    fudd_predictions = torch.tensor(
        [
            [0, 5],
            [1, 9],
            [9, 2],
            [3, 8],
            [6, 4],
        ]
    )

    counts = compute_transfer_counts(baseline_predictions, fudd_predictions, targets)

    assert counts.total == 5
    assert counts.both_correct == 1
    assert counts.recovered == 2
    assert counts.degraded == 1
    assert counts.both_wrong == 1
    assert (
        counts.both_correct + counts.recovered + counts.degraded + counts.both_wrong == counts.total
    )


def test_transfer_counts_accept_top1_vectors() -> None:
    targets = torch.tensor([3, 4])
    baseline_predictions = torch.tensor([3, 0])
    fudd_predictions = torch.tensor([0, 4])

    counts = compute_transfer_counts(baseline_predictions, fudd_predictions, targets)

    assert counts.both_correct == 0
    assert counts.recovered == 1
    assert counts.degraded == 1
    assert counts.both_wrong == 0
