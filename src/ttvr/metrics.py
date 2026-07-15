"""Pure ranking, accuracy, and prediction-transfer metrics shared by methods."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

EXACT_MCNEMAR_ALGORITHM = "exact-two-sided-binomial-mcnemar"
PAIRED_BOOTSTRAP_ALGORITHM = "paired-nonparametric-percentile-bootstrap-top1-accuracy-gain"


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


def _paired_correctness(
    baseline_correct: Tensor,
    comparison_correct: Tensor,
) -> tuple[Tensor, Tensor]:
    """Validate and copy paired Top-1 correctness vectors to CPU."""

    for values, name in (
        (baseline_correct, "baseline_correct"),
        (comparison_correct, "comparison_correct"),
    ):
        if values.ndim != 1 or values.numel() == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
        if values.dtype is not torch.bool:
            raise ValueError(f"{name} must have boolean dtype")
    if baseline_correct.shape != comparison_correct.shape:
        raise ValueError("Paired correctness tensors must have the same shape")
    return (
        baseline_correct.detach().to(device="cpu").contiguous(),
        comparison_correct.detach().to(device="cpu").contiguous(),
    )


@dataclass(frozen=True, slots=True)
class ExactMcNemarResult:
    """Exact two-sided McNemar result for paired Top-1 correctness."""

    total: int
    recovered: int
    degraded: int
    p_value: float

    @property
    def discordant(self) -> int:
        return self.recovered + self.degraded

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "algorithm": EXACT_MCNEMAR_ALGORITHM,
            "total": self.total,
            "recovered": self.recovered,
            "degraded": self.degraded,
            "discordant": self.discordant,
            "p_value": self.p_value,
        }


def _exact_two_sided_binomial_p_value(successes: int, failures: int) -> float:
    trials = successes + failures
    if trials == 0:
        return 1.0
    lower = min(successes, failures)
    coefficient = 1
    lower_tail_numerator = 1
    for k in range(1, lower + 1):
        coefficient = coefficient * (trials - k + 1) // k
        lower_tail_numerator += coefficient
    return min(1.0, (2 * lower_tail_numerator) / (1 << trials))


def exact_mcnemar_test(
    baseline_correct: Tensor,
    comparison_correct: Tensor,
) -> ExactMcNemarResult:
    """Run the exact two-sided McNemar test on paired Top-1 outcomes.

    The null distribution conditions on discordant pairs and treats recovery
    versus degradation as a fair binomial outcome.  No asymptotic correction
    or chi-squared approximation is used.
    """

    baseline, comparison = _paired_correctness(baseline_correct, comparison_correct)
    recovered = int((~baseline & comparison).sum().item())
    degraded = int((baseline & ~comparison).sum().item())
    return ExactMcNemarResult(
        total=baseline.numel(),
        recovered=recovered,
        degraded=degraded,
        p_value=_exact_two_sided_binomial_p_value(recovered, degraded),
    )


@dataclass(frozen=True, slots=True)
class PairedBootstrapGain:
    """Percentile bootstrap interval for a paired Top-1 accuracy gain."""

    total: int
    baseline_accuracy_percent: float
    comparison_accuracy_percent: float
    gain_pp: float
    confidence_level: float
    ci_lower_pp: float
    ci_upper_pp: float
    reps: int
    seed: int
    chunk_size: int

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "algorithm": PAIRED_BOOTSTRAP_ALGORITHM,
            "total": self.total,
            "baseline_accuracy_percent": self.baseline_accuracy_percent,
            "comparison_accuracy_percent": self.comparison_accuracy_percent,
            "gain_pp": self.gain_pp,
            "confidence_level": self.confidence_level,
            "ci_lower_pp": self.ci_lower_pp,
            "ci_upper_pp": self.ci_upper_pp,
            "reps": self.reps,
            "seed": self.seed,
            "chunk_size": self.chunk_size,
        }


def paired_bootstrap_accuracy_gain(
    baseline_correct: Tensor,
    comparison_correct: Tensor,
    *,
    confidence_level: float = 0.95,
    reps: int = 10_000,
    seed: int = 2026,
    chunk_size: int = 256,
) -> PairedBootstrapGain:
    """Estimate a paired nonparametric percentile CI for Top-1 gain.

    Samples, rather than the two systems independently, are resampled with
    replacement.  Work is performed on CPU in bounded chunks so the default
    10,000 replicates do not allocate a ``[reps, samples]`` tensor at once.
    """

    if not 0.0 < confidence_level < 1.0 or not math.isfinite(confidence_level):
        raise ValueError("confidence_level must be finite and strictly between zero and one")
    if isinstance(reps, bool) or not isinstance(reps, int) or reps <= 0:
        raise ValueError("reps must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    baseline, comparison = _paired_correctness(baseline_correct, comparison_correct)
    total = baseline.numel()
    baseline_count = int(baseline.sum().item())
    comparison_count = int(comparison.sum().item())
    differences = comparison.to(torch.float64) - baseline.to(torch.float64)
    replicate_gains = torch.empty(reps, dtype=torch.float64, device="cpu")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    for start in range(0, reps, chunk_size):
        stop = min(start + chunk_size, reps)
        indices = torch.randint(
            total,
            (stop - start, total),
            generator=generator,
            device="cpu",
        )
        replicate_gains[start:stop] = differences[indices].mean(dim=1).mul_(100.0)

    alpha = 1.0 - confidence_level
    quantiles = torch.quantile(
        replicate_gains,
        torch.tensor([alpha / 2.0, 1.0 - alpha / 2.0], dtype=torch.float64),
        interpolation="linear",
    )
    return PairedBootstrapGain(
        total=total,
        baseline_accuracy_percent=100.0 * baseline_count / total,
        comparison_accuracy_percent=100.0 * comparison_count / total,
        gain_pp=100.0 * (comparison_count - baseline_count) / total,
        confidence_level=confidence_level,
        ci_lower_pp=float(quantiles[0].item()),
        ci_upper_pp=float(quantiles[1].item()),
        reps=reps,
        seed=seed,
        chunk_size=chunk_size,
    )
