"""Dataset-balanced fitting for a class-agnostic feature adapter."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from .model import ResidualFeatureAdapter, similarity_logits


@dataclass(frozen=True, slots=True)
class FeatureTask:
    """One dataset-local label space and its frozen text prototypes."""

    name: str
    features: Tensor
    labels: Tensor
    text_prototypes: Tensor

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task name must not be empty")
        if self.features.ndim != 2 or self.features.shape[0] == 0:
            raise ValueError("features must be a non-empty matrix")
        if self.labels.ndim != 1 or self.labels.shape[0] != self.features.shape[0]:
            raise ValueError("labels must align with features")
        if self.labels.dtype != torch.long:
            raise ValueError("labels must use torch.long")
        if self.text_prototypes.ndim != 2:
            raise ValueError("text_prototypes must be a matrix")
        if self.text_prototypes.shape[1] != self.features.shape[1]:
            raise ValueError("image and text feature dimensions must match")
        if self.text_prototypes.shape[0] < 2:
            raise ValueError("each task must contain at least two classes")
        labels_cpu = self.labels.detach().cpu()
        if labels_cpu.min().item() < 0 or labels_cpu.max().item() >= self.class_count:
            raise ValueError("labels must index text_prototypes")
        class_counts = torch.bincount(labels_cpu, minlength=self.class_count)
        if class_counts.numel() != self.class_count or not bool(class_counts.gt(0).all()):
            raise ValueError("every local class must have at least one image")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("features must be finite")
        if not bool(torch.isfinite(self.text_prototypes).all()):
            raise ValueError("text prototypes must be finite")

    @property
    def size(self) -> int:
        return self.features.shape[0]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[1]

    @property
    def class_count(self) -> int:
        return self.text_prototypes.shape[0]


@dataclass(frozen=True, slots=True)
class AdapterTrainConfig:
    """Locked optimisation settings for one adapter trial."""

    steps: int = 20_000
    validation_interval: int = 250
    patience_intervals: int = 8
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    identity_weight: float = 0.1
    logit_scale: float = 100.0
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.validation_interval <= 0 or self.patience_intervals <= 0:
            raise ValueError("step, validation, and patience values must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("identity_weight", self.identity_weight),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if not math.isfinite(self.logit_scale) or self.logit_scale <= 0:
            raise ValueError("logit_scale must be finite and positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TaskScore:
    name: str
    size: int
    class_count: int
    top1: float
    class_balanced_top1: float
    cross_entropy: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationSnapshot:
    step: int
    macro_class_balanced_top1: float
    tasks: tuple[TaskScore, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "macro_class_balanced_top1": self.macro_class_balanced_top1,
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass(frozen=True, slots=True)
class AdapterFitResult:
    state_dict: dict[str, Tensor]
    best_step: int
    best_validation: ValidationSnapshot
    history: tuple[ValidationSnapshot, ...]
    train_task_draws: dict[str, int]


@dataclass(frozen=True, slots=True)
class AdapterRefitResult:
    state_dict: dict[str, Tensor]
    steps: int
    train_task_draws: dict[str, int]


def _validate_tasks(tasks: tuple[FeatureTask, ...], *, role: str) -> int:
    if not tasks:
        raise ValueError(f"at least one {role} task is required")
    names = [task.name for task in tasks]
    if len(names) != len(set(names)):
        raise ValueError(f"{role} task names must be unique")
    dimensions = {task.feature_dim for task in tasks}
    if len(dimensions) != 1:
        raise ValueError(f"all {role} tasks must use one feature dimension")
    return dimensions.pop()


def _class_indices(task: FeatureTask) -> tuple[Tensor, ...]:
    labels = task.labels.detach().cpu()
    return tuple(torch.where(labels == class_id)[0] for class_id in range(task.class_count))


def sample_class_balanced_indices(
    task: FeatureTask,
    *,
    batch_size: int,
    generator: torch.Generator,
    indices_by_class: tuple[Tensor, ...] | None = None,
) -> Tensor:
    """Draw classes uniformly, then draw one image uniformly within each class."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    groups = _class_indices(task) if indices_by_class is None else indices_by_class
    if len(groups) != task.class_count or any(group.numel() == 0 for group in groups):
        raise ValueError("indices_by_class must cover every local class")
    selected_classes = torch.randint(
        task.class_count,
        (batch_size,),
        generator=generator,
    )
    selected: list[int] = []
    for class_id in selected_classes.tolist():
        candidates = groups[class_id]
        offset = int(torch.randint(candidates.numel(), (), generator=generator).item())
        selected.append(int(candidates[offset].item()))
    return torch.tensor(selected, dtype=torch.long)


def _task_probabilities(tasks: tuple[FeatureTask, ...]) -> Tensor:
    """Use sqrt(N) task weights so large datasets help without dominating."""

    weights = torch.tensor([math.sqrt(task.size) for task in tasks], dtype=torch.double)
    return weights / weights.sum()


def score_task(
    adapter: ResidualFeatureAdapter,
    task: FeatureTask,
    *,
    logit_scale: float,
    device: torch.device,
    batch_size: int = 4096,
) -> TaskScore:
    """Evaluate one dataset, including a species-balanced accuracy."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    correct_chunks: list[Tensor] = []
    losses: list[Tensor] = []
    labels_cpu = task.labels.detach().cpu()
    prototypes = functional.normalize(task.text_prototypes.detach().float(), dim=1).to(device)
    adapter.eval()
    with torch.inference_mode():
        for start in range(0, task.size, batch_size):
            stop = min(start + batch_size, task.size)
            labels = labels_cpu[start:stop].to(device)
            frozen = functional.normalize(
                task.features[start:stop].detach().float().to(device), dim=1
            )
            adapted = adapter(frozen)
            logits = similarity_logits(adapted, prototypes, logit_scale=logit_scale)
            correct_chunks.append(logits.argmax(dim=1).eq(labels).cpu())
            losses.append(functional.cross_entropy(logits, labels, reduction="sum").cpu())
    correct = torch.cat(correct_chunks)
    per_class = [
        correct[labels_cpu == class_id].float().mean()
        for class_id in range(task.class_count)
    ]
    return TaskScore(
        name=task.name,
        size=task.size,
        class_count=task.class_count,
        top1=100.0 * float(correct.float().mean().item()),
        class_balanced_top1=100.0 * float(torch.stack(per_class).mean().item()),
        cross_entropy=float(torch.stack(losses).sum().item() / task.size),
    )


def evaluate_tasks(
    adapter: ResidualFeatureAdapter,
    tasks: tuple[FeatureTask, ...],
    *,
    step: int,
    logit_scale: float,
    device: torch.device,
) -> ValidationSnapshot:
    """Score validation datasets with equal weight per dataset."""

    scores = tuple(
        score_task(adapter, task, logit_scale=logit_scale, device=device) for task in tasks
    )
    macro = sum(score.class_balanced_top1 for score in scores) / len(scores)
    return ValidationSnapshot(step=step, macro_class_balanced_top1=macro, tasks=scores)


def fit_feature_adapter(
    adapter: ResidualFeatureAdapter,
    train_tasks: tuple[FeatureTask, ...],
    validation_tasks: tuple[FeatureTask, ...],
    config: AdapterTrainConfig,
    *,
    device: str | torch.device | None = None,
) -> AdapterFitResult:
    """Fit one shared adapter over independent dataset-local vocabularies."""

    train_dim = _validate_tasks(train_tasks, role="training")
    validation_dim = _validate_tasks(validation_tasks, role="validation")
    if train_dim != validation_dim or adapter.feature_dim != train_dim:
        raise ValueError("adapter, training, and validation feature dimensions must match")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    # The caller necessarily constructs the module before entering this
    # function. Reset it after seeding so the trial seed covers initialisation,
    # not just sampling and optimiser updates.
    adapter.reset_parameters()
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    adapter = adapter.to(resolved_device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    probabilities = _task_probabilities(train_tasks)
    grouped_indices = tuple(_class_indices(task) for task in train_tasks)
    prototypes = tuple(
        functional.normalize(task.text_prototypes.detach().float(), dim=1).to(resolved_device)
        for task in train_tasks
    )
    task_draws = {task.name: 0 for task in train_tasks}

    initial = evaluate_tasks(
        adapter,
        validation_tasks,
        step=0,
        logit_scale=config.logit_scale,
        device=resolved_device,
    )
    history = [initial]
    best = initial
    best_state = copy.deepcopy(adapter.state_dict())
    stale_intervals = 0

    adapter.train()
    for step in range(1, config.steps + 1):
        task_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        task = train_tasks[task_index]
        task_draws[task.name] += 1
        indices = sample_class_balanced_indices(
            task,
            batch_size=config.batch_size,
            generator=generator,
            indices_by_class=grouped_indices[task_index],
        )
        frozen = task.features.index_select(0, indices).detach().float().to(resolved_device)
        frozen = functional.normalize(frozen, dim=1)
        labels = task.labels.index_select(0, indices).detach().to(resolved_device)
        adapted = adapter(frozen)
        logits = similarity_logits(
            adapted,
            prototypes[task_index],
            logit_scale=config.logit_scale,
        )
        classification_loss = functional.cross_entropy(logits, labels)
        identity_loss = (1.0 - (adapted * frozen).sum(dim=1)).mean()
        loss = classification_loss + config.identity_weight * identity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        should_validate = step % config.validation_interval == 0 or step == config.steps
        if not should_validate:
            continue
        snapshot = evaluate_tasks(
            adapter,
            validation_tasks,
            step=step,
            logit_scale=config.logit_scale,
            device=resolved_device,
        )
        history.append(snapshot)
        if snapshot.macro_class_balanced_top1 > best.macro_class_balanced_top1 + 1e-12:
            best = snapshot
            best_state = copy.deepcopy(adapter.state_dict())
            stale_intervals = 0
        else:
            stale_intervals += 1
        if stale_intervals >= config.patience_intervals:
            break
        adapter.train()

    cpu_state = {name: value.detach().cpu().clone() for name, value in best_state.items()}
    return AdapterFitResult(
        state_dict=cpu_state,
        best_step=best.step,
        best_validation=best,
        history=tuple(history),
        train_task_draws=task_draws,
    )


def refit_feature_adapter(
    adapter: ResidualFeatureAdapter,
    train_tasks: tuple[FeatureTask, ...],
    config: AdapterTrainConfig,
    *,
    steps: int,
    device: str | torch.device | None = None,
) -> AdapterRefitResult:
    """Refit on every source species for a validation-selected step count.

    Hyperparameters and ``steps`` must already have been selected without any
    target data.  No validation is consulted here, so the final fit can use the
    source species that were held out during model selection.
    """

    feature_dim = _validate_tasks(train_tasks, role="refit training")
    if adapter.feature_dim != feature_dim or steps < 0:
        raise ValueError("adapter dimension or refit step count is invalid")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    adapter.reset_parameters()
    adapter = adapter.to(resolved_device)
    if steps == 0:
        return AdapterRefitResult(
            state_dict={
                name: value.detach().cpu().clone()
                for name, value in adapter.state_dict().items()
            },
            steps=0,
            train_task_draws={task.name: 0 for task in train_tasks},
        )

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    probabilities = _task_probabilities(train_tasks)
    grouped_indices = tuple(_class_indices(task) for task in train_tasks)
    prototypes = tuple(
        functional.normalize(task.text_prototypes.detach().float(), dim=1).to(resolved_device)
        for task in train_tasks
    )
    task_draws = {task.name: 0 for task in train_tasks}
    adapter.train()
    for _ in range(steps):
        task_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
        task = train_tasks[task_index]
        task_draws[task.name] += 1
        indices = sample_class_balanced_indices(
            task,
            batch_size=config.batch_size,
            generator=generator,
            indices_by_class=grouped_indices[task_index],
        )
        frozen = task.features.index_select(0, indices).detach().float().to(resolved_device)
        frozen = functional.normalize(frozen, dim=1)
        labels = task.labels.index_select(0, indices).detach().to(resolved_device)
        adapted = adapter(frozen)
        logits = similarity_logits(
            adapted,
            prototypes[task_index],
            logit_scale=config.logit_scale,
        )
        classification_loss = functional.cross_entropy(logits, labels)
        identity_loss = (1.0 - (adapted * frozen).sum(dim=1)).mean()
        loss = classification_loss + config.identity_weight * identity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return AdapterRefitResult(
        state_dict={
            name: value.detach().cpu().clone()
            for name, value in adapter.state_dict().items()
        },
        steps=steps,
        train_task_draws=task_draws,
    )
