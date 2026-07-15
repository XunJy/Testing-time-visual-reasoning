"""Pure ranking, accuracy, and prediction-transfer metrics shared by methods."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _as_batched(values: Tensor, *, name: str) -> tuple[Tensor, bool]:
    if values.ndim == 1:
        return values.unsqueeze(0), True
    if values.ndim == 2:
        return values, False
    raise ValueError(f"{name} must be one- or two-dimensional")


def ordered_predictions(
    logits: Tensor,
    candidate_ids: Tensor | None = None,
) -> Tensor:
    """Sort logits and return globally meaningful class ids.

    Args:
        logits: ``[batch, classes]`` scores, or one unbatched score vector.
        candidate_ids: Optional global labels for logit columns.  A shared
            ``[classes]`` vector or per-example ``[batch, classes]`` matrix is
            accepted.  When omitted, columns are already global class ids.

    Ties retain the original column order through a stable sort.  This makes
    CPU/GPU smoke tests deterministic and ensures local FuDD logits map back to
    the baseline's global labels.
    """

    batched_logits, was_unbatched = _as_batched(logits, name="logits")
    order = torch.argsort(
        batched_logits,
        dim=-1,
        descending=True,
        stable=True,
    )

    if candidate_ids is None:
        candidates = torch.arange(
            batched_logits.shape[1],
            device=batched_logits.device,
            dtype=torch.long,
        ).expand_as(order)
    else:
        candidates = candidate_ids.to(
            device=batched_logits.device,
            dtype=torch.long,
        )
        if candidates.ndim == 1:
            if candidates.shape[0] != batched_logits.shape[1]:
                raise ValueError("candidate_ids length must match logits columns")
            candidates = candidates.unsqueeze(0).expand_as(order)
        elif candidates.shape != batched_logits.shape:
            raise ValueError("candidate_ids must have shape [classes] or [batch, classes]")

    predictions = torch.gather(candidates, dim=1, index=order)
    return predictions.squeeze(0) if was_unbatched else predictions


def topk_hits(predictions: Tensor, targets: Tensor, k: int) -> Tensor:
    """Return a boolean per-example mask for top-``k`` correctness."""

    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape [batch, ranked_classes]")
    targets = targets.reshape(-1).to(device=predictions.device)
    if targets.shape[0] != predictions.shape[0]:
        raise ValueError("targets length must match predictions batch size")
    if k > predictions.shape[1]:
        raise ValueError(f"k={k} exceeds the {predictions.shape[1]} available predictions")
    return predictions[:, :k].eq(targets.unsqueeze(1)).any(dim=1)


def accuracy_percentage(hits: Tensor) -> float:
    """Convert a non-empty boolean hit mask to a Python percentage."""

    if hits.ndim != 1 or hits.numel() == 0:
        raise ValueError("hits must be a non-empty one-dimensional tensor")
    return 100.0 * float(hits.float().mean().item())


@dataclass(frozen=True, slots=True)
class TopKAccuracy:
    """Top-1/top-5 counts plus JSON-safe percentage properties."""

    total: int
    top1_correct: int
    top5_correct: int

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("total must be positive")
        if not 0 <= self.top1_correct <= self.top5_correct <= self.total:
            raise ValueError("Top-k counts are inconsistent")

    @property
    def top1(self) -> float:
        return 100.0 * self.top1_correct / self.total

    @property
    def top5(self) -> float:
        return 100.0 * self.top5_correct / self.total

    def to_dict(self) -> dict[str, int | float]:
        return {
            "total": self.total,
            "top1_correct": self.top1_correct,
            "top5_correct": self.top5_correct,
            "top1": self.top1,
            "top5": self.top5,
        }


def compute_topk_accuracy(predictions: Tensor, targets: Tensor) -> TopKAccuracy:
    """Compute top-1 and top-5 counts from ranked global predictions."""

    top1 = topk_hits(predictions, targets, 1)
    top5 = topk_hits(predictions, targets, 5)
    return TopKAccuracy(
        total=int(top1.numel()),
        top1_correct=int(top1.sum().item()),
        top5_correct=int(top5.sum().item()),
    )


@dataclass(frozen=True, slots=True)
class TransferCounts:
    """Four-way top-1 transition table from baseline to FuDD."""

    total: int
    both_correct: int
    recovered: int
    degraded: int
    both_wrong: int

    def __post_init__(self) -> None:
        values = (
            self.both_correct,
            self.recovered,
            self.degraded,
            self.both_wrong,
        )
        if self.total <= 0 or any(value < 0 for value in values):
            raise ValueError("Transfer counts must be non-negative with total > 0")
        if sum(values) != self.total:
            raise ValueError("Transfer categories must sum to total")

    def _percentage(self, count: int) -> float:
        return 100.0 * count / self.total

    def to_dict(self) -> dict[str, int | float]:
        """Return counts and percentages using unambiguous key suffixes."""

        return {
            "total": self.total,
            "both_correct": self.both_correct,
            "recovered": self.recovered,
            "degraded": self.degraded,
            "both_wrong": self.both_wrong,
            "both_correct_percent": self._percentage(self.both_correct),
            "recovered_percent": self._percentage(self.recovered),
            "degraded_percent": self._percentage(self.degraded),
            "both_wrong_percent": self._percentage(self.both_wrong),
        }


def _top1_labels(predictions: Tensor, *, name: str) -> Tensor:
    if predictions.ndim == 1:
        return predictions
    if predictions.ndim == 2 and predictions.shape[1] >= 1:
        return predictions[:, 0]
    raise ValueError(f"{name} must have shape [batch] or [batch, ranked_classes]")


def compute_transfer_counts(
    baseline_predictions: Tensor,
    fudd_predictions: Tensor,
    targets: Tensor,
) -> TransferCounts:
    """Count top-1 samples preserved, recovered, degraded, or still wrong."""

    baseline_top1 = _top1_labels(
        baseline_predictions,
        name="baseline_predictions",
    )
    fudd_top1 = _top1_labels(fudd_predictions, name="fudd_predictions")
    targets = targets.reshape(-1).to(device=baseline_top1.device)
    fudd_top1 = fudd_top1.to(device=baseline_top1.device)
    if not (baseline_top1.shape == fudd_top1.shape == targets.shape and targets.numel() > 0):
        raise ValueError("Predictions and targets must have the same non-empty batch")

    baseline_correct = baseline_top1.eq(targets)
    fudd_correct = fudd_top1.eq(targets)
    both_correct = baseline_correct & fudd_correct
    recovered = ~baseline_correct & fudd_correct
    degraded = baseline_correct & ~fudd_correct
    both_wrong = ~baseline_correct & ~fudd_correct

    return TransferCounts(
        total=int(targets.numel()),
        both_correct=int(both_correct.sum().item()),
        recovered=int(recovered.sum().item()),
        degraded=int(degraded.sum().item()),
        both_wrong=int(both_wrong.sum().item()),
    )
