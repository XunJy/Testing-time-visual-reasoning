"""Supervised linear heads trained on frozen vision-language features."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ...metrics import TopKAccuracy, compute_topk_accuracy, ordered_predictions

HeadMode = Literal["linear", "residual"]


@dataclass(frozen=True, slots=True)
class ResidualHeadSearchConfig:
    """Pre-registered search space for frozen-feature head tuning."""

    validation_per_class: int = 6
    max_epochs: int = 200
    patience: int = 20
    batch_size: int = 256
    learning_rates: tuple[float, ...] = (1e-3, 3e-3, 1e-2)
    weight_decays: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2)
    alpha_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.validation_per_class <= 0:
            raise ValueError("validation_per_class must be positive")
        if self.max_epochs <= 0 or self.patience <= 0 or self.batch_size <= 0:
            raise ValueError("Epoch, patience, and batch sizes must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        for name, values in (
            ("learning_rates", self.learning_rates),
            ("weight_decays", self.weight_decays),
            ("alpha_grid", self.alpha_grid),
        ):
            if not values or any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"{name} must contain finite non-negative values")
        if 0.0 not in self.alpha_grid:
            raise ValueError("alpha_grid must include 0.0 to recover the frozen baseline")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("learning_rates", "weight_decays", "alpha_grid"):
            values[key] = list(values[key])
        return values


@dataclass(frozen=True, slots=True)
class StratifiedSplit:
    """Indices selected without consulting the official test split."""

    fit_indices: Tensor
    validation_indices: Tensor

    def __post_init__(self) -> None:
        for name, indices in (
            ("fit_indices", self.fit_indices),
            ("validation_indices", self.validation_indices),
        ):
            if indices.ndim != 1 or indices.dtype != torch.long:
                raise ValueError(f"{name} must be a one-dimensional int64 tensor")
            if indices.device.type != "cpu":
                raise ValueError(f"{name} must be stored on CPU")
            if indices.numel() != torch.unique(indices).numel():
                raise ValueError(f"{name} must not contain duplicate indices")
        if self.fit_indices.numel() == 0 or self.validation_indices.numel() == 0:
            raise ValueError("Both split partitions must be non-empty")
        fit = set(self.fit_indices.tolist())
        validation = set(self.validation_indices.tolist())
        if fit.intersection(validation):
            raise ValueError("Fit and validation partitions must be disjoint")


@dataclass(frozen=True, slots=True)
class ValidationScore:
    total: int
    top1_correct: int
    top5_correct: int
    cross_entropy: float

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
            "cross_entropy": self.cross_entropy,
        }


@dataclass(frozen=True, slots=True)
class HeadTrial:
    mode: HeadMode
    learning_rate: float
    weight_decay: float
    epochs_ran: int
    selected_epoch: int
    selected_alpha: float
    validation: ValidationScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs_ran": self.epochs_ran,
            "selected_epoch": self.selected_epoch,
            "selected_alpha": self.selected_alpha,
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HeadSelection:
    mode: HeadMode
    learning_rate: float
    weight_decay: float
    epoch: int
    alpha: float
    validation: ValidationScore

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epoch": self.epoch,
            "alpha": self.alpha,
            "validation": self.validation.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HeadSearchResult:
    selection: HeadSelection
    validation_state: dict[str, Tensor]
    trials: tuple[HeadTrial, ...]


class LinearFeatureHead(nn.Module):
    """A zero-initialised affine classifier over frozen image features."""

    def __init__(self, feature_dim: int, class_count: int) -> None:
        super().__init__()
        if feature_dim <= 0 or class_count <= 1:
            raise ValueError("feature_dim must be positive and class_count must exceed one")
        self.linear = nn.Linear(feature_dim, class_count)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)


def stable_split_score(seed: int, sample_key: str) -> str:
    """Return the version-independent hash used to assign validation rows."""

    if seed < 0 or not sample_key:
        raise ValueError("seed must be non-negative and sample_key must not be empty")
    return hashlib.sha256(f"{seed}\0{sample_key}".encode()).hexdigest()


def stratified_hash_split(
    labels: Tensor,
    sample_keys: Sequence[str],
    *,
    validation_per_class: int,
    seed: int,
) -> StratifiedSplit:
    """Select a stable fixed number of validation examples from each class."""

    labels = labels.detach().cpu().long().reshape(-1)
    if labels.numel() != len(sample_keys) or labels.numel() == 0:
        raise ValueError("labels and sample_keys must have the same non-zero length")
    if validation_per_class <= 0:
        raise ValueError("validation_per_class must be positive")
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("sample_keys must be unique")

    fit: list[int] = []
    validation: list[int] = []
    for class_id in sorted(set(labels.tolist())):
        class_indices = torch.where(labels == class_id)[0].tolist()
        if len(class_indices) <= validation_per_class:
            raise ValueError(f"Class {class_id} does not have enough fit examples")
        ranked = sorted(
            class_indices,
            key=lambda index: (stable_split_score(seed, sample_keys[index]), index),
        )
        validation.extend(ranked[:validation_per_class])
        fit.extend(ranked[validation_per_class:])
    return StratifiedSplit(
        fit_indices=torch.tensor(sorted(fit), dtype=torch.long),
        validation_indices=torch.tensor(sorted(validation), dtype=torch.long),
    )


def combine_residual_logits(base_logits: Tensor, residual_logits: Tensor, alpha: float) -> Tensor:
    """Combine frozen zero-shot and learned residual logits."""

    if base_logits.shape != residual_logits.shape or base_logits.ndim != 2:
        raise ValueError("base_logits and residual_logits must have equal 2D shapes")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative")
    return base_logits + alpha * residual_logits


def _validate_training_inputs(
    features: Tensor,
    labels: Tensor,
    base_logits: Tensor | None,
    mode: HeadMode,
) -> tuple[Tensor, Tensor, Tensor | None]:
    if mode not in ("linear", "residual"):
        raise ValueError("mode must be 'linear' or 'residual'")
    features = features.detach().cpu().float().contiguous()
    labels = labels.detach().cpu().long().reshape(-1).contiguous()
    if features.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels are misaligned")
    if features.shape[0] == 0 or not bool(torch.isfinite(features).all()):
        raise ValueError("features must be non-empty and finite")
    class_count = int(labels.max().item()) + 1
    if labels.min().item() < 0 or class_count < 5:
        raise ValueError("labels must contain at least five non-negative class ids")
    if mode == "residual":
        if base_logits is None:
            raise ValueError("Residual training requires base_logits")
        base_logits = base_logits.detach().cpu().float().contiguous()
        if base_logits.shape != (features.shape[0], class_count):
            raise ValueError("base_logits shape does not match samples and classes")
        if not bool(torch.isfinite(base_logits).all()):
            raise ValueError("base_logits must be finite")
    elif base_logits is not None:
        raise ValueError("Linear-only training must not receive base_logits")
    return features, labels, base_logits


def _validate_split_for_training(
    split: StratifiedSplit,
    labels: Tensor,
    validation_per_class: int,
) -> None:
    sample_count = labels.numel()
    covered = torch.cat((split.fit_indices, split.validation_indices))
    if (
        covered.numel() != sample_count
        or covered.min().item() < 0
        or covered.max().item() >= sample_count
        or not torch.equal(torch.sort(covered).values, torch.arange(sample_count))
    ):
        raise ValueError("Split partitions must cover every sample exactly once")
    class_count = int(labels.max().item()) + 1
    validation_counts = torch.bincount(
        labels.index_select(0, split.validation_indices),
        minlength=class_count,
    )
    if validation_counts.tolist() != [validation_per_class] * class_count:
        raise ValueError("Validation partition must contain validation_per_class samples per class")


def _score(logits: Tensor, labels: Tensor) -> ValidationScore:
    predictions = ordered_predictions(logits)
    metrics = compute_topk_accuracy(predictions, labels)
    return ValidationScore(
        total=metrics.total,
        top1_correct=metrics.top1_correct,
        top5_correct=metrics.top5_correct,
        cross_entropy=float(functional.cross_entropy(logits, labels).item()),
    )


def _selection_key(score: ValidationScore) -> tuple[int, int, float]:
    return (score.top1_correct, score.top5_correct, -score.cross_entropy)


def _cpu_state(head: LinearFeatureHead) -> dict[str, Tensor]:
    return {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def _validation_candidates(
    mode: HeadMode,
    residual_logits: Tensor,
    labels: Tensor,
    base_logits: Tensor | None,
    alpha_grid: Sequence[float],
) -> list[tuple[float, ValidationScore]]:
    if mode == "linear":
        return [(1.0, _score(residual_logits, labels))]
    assert base_logits is not None
    return [
        (alpha, _score(combine_residual_logits(base_logits, residual_logits, alpha), labels))
        for alpha in alpha_grid
    ]


def search_feature_head(
    features: Tensor,
    labels: Tensor,
    split: StratifiedSplit,
    config: ResidualHeadSearchConfig,
    *,
    mode: HeadMode,
    base_logits: Tensor | None = None,
    device: str | torch.device = "cpu",
    trial_callback: Callable[[HeadTrial], None] | None = None,
) -> HeadSearchResult:
    """Select head hyperparameters using only the supplied fit/validation split."""

    features, labels, base_logits = _validate_training_inputs(features, labels, base_logits, mode)
    _validate_split_for_training(split, labels, config.validation_per_class)
    resolved_device = torch.device(device)
    class_count = int(labels.max().item()) + 1
    fit_x = features.index_select(0, split.fit_indices).to(resolved_device)
    fit_y = labels.index_select(0, split.fit_indices).to(resolved_device)
    val_x = features.index_select(0, split.validation_indices).to(resolved_device)
    val_y = labels.index_select(0, split.validation_indices).to(resolved_device)
    fit_base = (
        None
        if base_logits is None
        else base_logits.index_select(0, split.fit_indices).to(resolved_device)
    )
    val_base = (
        None
        if base_logits is None
        else base_logits.index_select(0, split.validation_indices).to(resolved_device)
    )

    trials: list[HeadTrial] = []
    selected: HeadSelection | None = None
    selected_state: dict[str, Tensor] | None = None
    selected_key: tuple[int, int, float] | None = None
    for learning_rate in sorted(config.learning_rates):
        for weight_decay in sorted(config.weight_decays):
            torch.manual_seed(config.seed)
            head = LinearFeatureHead(features.shape[1], class_count).to(resolved_device)
            optimizer = torch.optim.AdamW(
                head.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
            generator = torch.Generator().manual_seed(config.seed)
            trial_best_key: tuple[int, int, float] | None = None
            trial_best_epoch = 0
            trial_best_alpha = 1.0
            trial_best_score: ValidationScore | None = None
            trial_best_state = _cpu_state(head)
            epochs_ran = 0

            for epoch in range(config.max_epochs + 1):
                head.eval()
                with torch.inference_mode():
                    residual = head(val_x)
                    candidates = _validation_candidates(
                        mode,
                        residual,
                        val_y,
                        val_base,
                        sorted(config.alpha_grid),
                    )
                for alpha, score in candidates:
                    key = _selection_key(score)
                    if trial_best_key is None or key > trial_best_key:
                        trial_best_key = key
                        trial_best_epoch = epoch
                        trial_best_alpha = alpha
                        trial_best_score = score
                        trial_best_state = _cpu_state(head)
                if epoch == config.max_epochs:
                    break
                if epoch - trial_best_epoch >= config.patience:
                    break

                head.train()
                order = torch.randperm(fit_x.shape[0], generator=generator)
                for start in range(0, fit_x.shape[0], config.batch_size):
                    rows = order[start : start + config.batch_size].to(resolved_device)
                    logits = head(fit_x.index_select(0, rows))
                    if mode == "residual":
                        assert fit_base is not None
                        logits = combine_residual_logits(
                            fit_base.index_select(0, rows), logits, 1.0
                        )
                    loss = functional.cross_entropy(logits, fit_y.index_select(0, rows))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                epochs_ran = epoch + 1

            assert trial_best_key is not None and trial_best_score is not None
            trial = HeadTrial(
                mode=mode,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs_ran=epochs_ran,
                selected_epoch=trial_best_epoch,
                selected_alpha=trial_best_alpha,
                validation=trial_best_score,
            )
            trials.append(trial)
            if trial_callback is not None:
                trial_callback(trial)
            if selected_key is None or trial_best_key > selected_key:
                selected_key = trial_best_key
                selected = HeadSelection(
                    mode=mode,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    epoch=trial_best_epoch,
                    alpha=trial_best_alpha,
                    validation=trial_best_score,
                )
                selected_state = trial_best_state

    assert selected is not None and selected_state is not None
    return HeadSearchResult(
        selection=selected,
        validation_state=selected_state,
        trials=tuple(trials),
    )


def refit_feature_head(
    features: Tensor,
    labels: Tensor,
    selection: HeadSelection,
    config: ResidualHeadSearchConfig,
    *,
    base_logits: Tensor | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """Refit a selected head on all official training examples."""

    features, labels, base_logits = _validate_training_inputs(
        features, labels, base_logits, selection.mode
    )
    resolved_device = torch.device(device)
    class_count = int(labels.max().item()) + 1
    x = features.to(resolved_device)
    y = labels.to(resolved_device)
    base = None if base_logits is None else base_logits.to(resolved_device)
    torch.manual_seed(config.seed)
    head = LinearFeatureHead(features.shape[1], class_count).to(resolved_device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=selection.learning_rate,
        weight_decay=selection.weight_decay,
    )
    generator = torch.Generator().manual_seed(config.seed)
    for _epoch in range(selection.epoch):
        order = torch.randperm(x.shape[0], generator=generator)
        for start in range(0, x.shape[0], config.batch_size):
            rows = order[start : start + config.batch_size].to(resolved_device)
            logits = head(x.index_select(0, rows))
            if selection.mode == "residual":
                assert base is not None
                logits = combine_residual_logits(base.index_select(0, rows), logits, 1.0)
            loss = functional.cross_entropy(logits, y.index_select(0, rows))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return _cpu_state(head)


def head_logits_from_state(
    features: Tensor,
    state: dict[str, Tensor],
    *,
    class_count: int,
    batch_size: int = 1024,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Apply a serialised linear head to frozen features in bounded batches."""

    features = features.detach().cpu().float()
    if features.ndim != 2 or features.shape[0] == 0 or batch_size <= 0:
        raise ValueError("features must be a non-empty matrix and batch_size positive")
    resolved_device = torch.device(device)
    head = LinearFeatureHead(features.shape[1], class_count).to(resolved_device)
    head.load_state_dict(state, strict=True)
    head.eval()
    outputs: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, features.shape[0], batch_size):
            outputs.append(head(features[start : start + batch_size].to(resolved_device)).cpu())
    return torch.cat(outputs, dim=0)


def evaluate_logits(logits: Tensor, labels: Tensor) -> TopKAccuracy:
    """Compute the shared top-1/top-5 metrics for a full classifier."""

    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("logits and labels are misaligned")
    return compute_topk_accuracy(ordered_predictions(logits), labels)
