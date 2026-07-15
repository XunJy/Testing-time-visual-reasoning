"""Model-backed single-template and top-k FuDD evaluation on CUB."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from ...data.cub import CUB200Dataset, prepare_cub, validate_class_name_alignment
from ...metrics import (
    TopKAccuracy,
    TransferCounts,
    compute_topk_accuracy,
    compute_transfer_counts,
    ordered_predictions,
    topk_hits,
)
from ...models.base import ProgressCallback, TextFeatureTable, VisionLanguageBackend
from ...models.clip import CLIPBackend
from .config import FuDDConfig
from .prompts import (
    CubPromptRepository,
    download_official_prompts,
    load_official_prompts,
)


@dataclass(frozen=True, slots=True)
class FuDDMetrics(TopKAccuracy):
    """FuDD accuracy plus recall of the baseline candidate set."""

    candidate_hits: int

    def __post_init__(self) -> None:
        TopKAccuracy.__post_init__(self)
        if not self.top5_correct <= self.candidate_hits <= self.total:
            raise ValueError("Candidate hits must cover every correct FuDD top-5")

    @property
    def candidate_recall(self) -> float:
        """Percentage whose target occurs in the baseline top-k candidates."""

        return 100.0 * self.candidate_hits / self.total

    def to_dict(self) -> dict[str, int | float]:
        values = TopKAccuracy.to_dict(self)
        values.update(
            {
                "candidate_hits": self.candidate_hits,
                "candidate_recall": self.candidate_recall,
            }
        )
        return values


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Per-image evidence needed to audit both FuDD classification rounds."""

    sample_index: int
    image_id: int
    relative_path: str
    target_class_id: int
    baseline_topk_class_ids: tuple[int, ...]
    fudd_ranked_class_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.sample_index < 0 or self.image_id <= 0:
            raise ValueError("Sample and image ids must be non-negative/positive")
        if not self.relative_path:
            raise ValueError("relative_path must not be empty")
        if self.target_class_id < 0:
            raise ValueError("target_class_id must be non-negative")
        if len(self.baseline_topk_class_ids) < 5:
            raise ValueError("At least five baseline candidates are required")
        if len(self.baseline_topk_class_ids) != len(self.fudd_ranked_class_ids):
            raise ValueError("Baseline and FuDD rankings must have equal length")
        if len(set(self.baseline_topk_class_ids)) != len(self.baseline_topk_class_ids):
            raise ValueError("Baseline candidates must be unique")
        if set(self.baseline_topk_class_ids) != set(self.fudd_ranked_class_ids):
            raise ValueError("FuDD must only reorder the baseline candidate set")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "image_id": self.image_id,
            "relative_path": self.relative_path,
            "target_class_id": self.target_class_id,
            "baseline_top1_class_id": self.baseline_topk_class_ids[0],
            "baseline_topk_class_ids": list(self.baseline_topk_class_ids),
            "fudd_top1_class_id": self.fudd_ranked_class_ids[0],
            "fudd_ranked_class_ids": list(self.fudd_ranked_class_ids),
            "target_in_baseline_topk": (self.target_class_id in self.baseline_topk_class_ids),
            "baseline_top1_correct": (self.baseline_topk_class_ids[0] == self.target_class_id),
            "fudd_top1_correct": (self.fudd_ranked_class_ids[0] == self.target_class_id),
        }


@dataclass(frozen=True, slots=True)
class CandidateRerankResult:
    """Candidate-aligned FuDD scores and their globally labelled ranking."""

    candidate_class_ids: Tensor
    scores: Tensor
    ranked_class_ids: Tensor

    def __post_init__(self) -> None:
        if not (
            self.candidate_class_ids.ndim == 2
            and self.scores.ndim == 2
            and self.ranked_class_ids.ndim == 2
        ):
            raise ValueError("Candidate rerank tensors must be two-dimensional")
        if not (
            self.candidate_class_ids.shape
            == self.scores.shape
            == self.ranked_class_ids.shape
        ):
            raise ValueError("Candidate rerank tensors must have identical shapes")
        if self.scores.numel() == 0 or not bool(torch.isfinite(self.scores).all()):
            raise ValueError("Candidate rerank scores must be non-empty and finite")


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Agreement between the batched implementation and official-style loop."""

    samples: int
    matching_predictions: int
    max_abs_prototype_difference: float

    def __post_init__(self) -> None:
        if self.samples <= 0 or not 0 <= self.matching_predictions <= self.samples:
            raise ValueError("Invalid parity counts")
        if self.max_abs_prototype_difference < 0:
            raise ValueError("Prototype difference must be non-negative")

    @property
    def passed(self) -> bool:
        return self.matching_predictions == self.samples

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "samples": self.samples,
            "matching_predictions": self.matching_predictions,
            "max_abs_prototype_difference": self.max_abs_prototype_difference,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Complete JSON-serialisable summary of one CUB evaluation."""

    config: Mapping[str, Any]
    num_samples: int
    baseline: TopKAccuracy
    fudd: FuDDMetrics
    transfers: TransferCounts
    parity: ParityReport
    prompt_digest: str
    dataset_fingerprint: str
    feature_dtype: str
    predictions: tuple[PredictionRecord, ...]

    def __post_init__(self) -> None:
        if self.num_samples != len(self.predictions):
            raise ValueError("Prediction count must equal num_samples")
        if self.num_samples != self.baseline.total or self.num_samples != self.fudd.total:
            raise ValueError("Metric totals must equal num_samples")
        if self.num_samples != self.transfers.total:
            raise ValueError("Transfer total must equal num_samples")
        if len({record.sample_index for record in self.predictions}) != self.num_samples:
            raise ValueError("Prediction sample indices must be unique")
        if len({record.image_id for record in self.predictions}) != self.num_samples:
            raise ValueError("Prediction image ids must be unique")
        baseline_hits = [
            record.baseline_topk_class_ids[0] == record.target_class_id
            for record in self.predictions
        ]
        fudd_hits = [
            record.fudd_ranked_class_ids[0] == record.target_class_id for record in self.predictions
        ]
        baseline_top5 = sum(
            record.target_class_id in record.baseline_topk_class_ids[:5]
            for record in self.predictions
        )
        fudd_top5 = sum(
            record.target_class_id in record.fudd_ranked_class_ids[:5]
            for record in self.predictions
        )
        candidate_hits = sum(
            record.target_class_id in record.baseline_topk_class_ids for record in self.predictions
        )
        expected_metrics = (
            sum(baseline_hits),
            baseline_top5,
            sum(fudd_hits),
            fudd_top5,
            candidate_hits,
        )
        actual_metrics = (
            self.baseline.top1_correct,
            self.baseline.top5_correct,
            self.fudd.top1_correct,
            self.fudd.top5_correct,
            self.fudd.candidate_hits,
        )
        if expected_metrics != actual_metrics:
            raise ValueError("Per-image predictions do not reproduce aggregate metrics")
        expected_transfers = (
            sum(before and after for before, after in zip(baseline_hits, fudd_hits, strict=True)),
            sum(
                not before and after for before, after in zip(baseline_hits, fudd_hits, strict=True)
            ),
            sum(
                before and not after for before, after in zip(baseline_hits, fudd_hits, strict=True)
            ),
            sum(
                not before and not after
                for before, after in zip(baseline_hits, fudd_hits, strict=True)
            ),
        )
        actual_transfers = (
            self.transfers.both_correct,
            self.transfers.recovered,
            self.transfers.degraded,
            self.transfers.both_wrong,
        )
        if expected_transfers != actual_transfers:
            raise ValueError("Per-image predictions do not reproduce transfer counts")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": dict(self.config),
            "num_samples": self.num_samples,
            "baseline": self.baseline.to_dict(),
            "fudd": self.fudd.to_dict(),
            "transfers": self.transfers.to_dict(),
            "parity": self.parity.to_dict(),
            "prompt_digest": self.prompt_digest,
            "dataset_fingerprint": self.dataset_fingerprint,
            "feature_dtype": self.feature_dtype,
            "prediction_count": len(self.predictions),
        }

    def write_predictions_jsonl(self, path: Path | str) -> tuple[Path, str]:
        """Atomically create a non-overwriting JSONL record and return its SHA-256."""

        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite predictions: {output_path}")
        temporary = output_path.with_name(f"{output_path.name}.tmp-{os.getpid()}")
        temporary.unlink(missing_ok=True)
        digest = hashlib.sha256()
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                for record in self.predictions:
                    line = json.dumps(
                        record.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    encoded = f"{line}\n".encode()
                    handle.write(encoded.decode())
                    digest.update(encoded)
            os.link(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return output_path, digest.hexdigest()


def _slug(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return compact or hashlib.sha256(value.encode()).hexdigest()[:12]


def _batched_baseline_predictions(
    image_features: Tensor,
    class_features: Tensor,
    *,
    batch_size: int,
    device: torch.device,
    progress: ProgressCallback | None,
) -> Tensor:
    predictions: list[Tensor] = []
    total = image_features.shape[0]
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = image_features[start:stop].to(device, non_blocking=True)
            logits = batch @ class_features.t()
            predictions.append(ordered_predictions(logits).cpu())
            if progress is not None:
                progress("baseline", stop, total)
    return torch.cat(predictions, dim=0)


def _required_followup_texts(
    candidate_rows: list[list[int]],
    prompts: CubPromptRepository,
    *,
    progress: ProgressCallback | None,
) -> tuple[str, ...]:
    required: dict[str, None] = {}
    total = len(candidate_rows)
    update_every = max(1, total // 100)
    for index, candidates in enumerate(candidate_rows, start=1):
        groups = prompts.prompts_for_candidates(candidates)
        for group in groups:
            for text in group:
                required.setdefault(text, None)
        if progress is not None and (index == total or index % update_every == 0):
            progress("candidate_prompts", index, total)
    return tuple(required)


def _batched_fudd_predictions(
    image_features: Tensor,
    candidates: Tensor,
    candidate_rows: list[list[int]],
    prompts: CubPromptRepository,
    backend: VisionLanguageBackend,
    text_table: TextFeatureTable,
    *,
    batch_size: int,
    progress: ProgressCallback | None,
) -> Tensor:
    """Compatibility wrapper around :func:`rerank_candidates`."""

    del candidate_rows
    return rerank_candidates(
        image_features,
        candidates,
        prompts,
        backend,
        text_table=text_table,
        batch_size=batch_size,
        progress=progress,
    ).ranked_class_ids


def rerank_candidates(
    image_features: Tensor,
    candidates: Tensor,
    prompts: CubPromptRepository,
    backend: VisionLanguageBackend,
    *,
    text_table: TextFeatureTable | None = None,
    batch_size: int = 32,
    progress: ProgressCallback | None = None,
) -> CandidateRerankResult:
    """Rerank per-image candidate classes with FuDD prompt prototypes.

    ``scores`` remains aligned with the input candidate columns.  Rankings use
    the corresponding global class ids and never introduce a new candidate.
    When no text table is supplied, the exact required prompt union is encoded
    once before scoring.
    """

    if image_features.ndim != 2 or image_features.shape[0] == 0:
        raise ValueError("image_features must have shape [samples, dimensions]")
    if candidates.ndim != 2 or candidates.shape[0] == 0:
        raise ValueError("candidates must have shape [samples, classes]")
    if image_features.shape[0] != candidates.shape[0]:
        raise ValueError("image_features and candidates must have the same sample count")
    if candidates.shape[1] < 2:
        raise ValueError("FuDD requires at least two candidates per image")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if candidates.dtype == torch.bool or candidates.dtype.is_floating_point:
        raise TypeError("candidates must contain integer class ids")

    candidate_ids = candidates.detach().to(device="cpu", dtype=torch.long).contiguous()
    if bool((candidate_ids < 0).any()) or bool((candidate_ids >= prompts.class_count).any()):
        raise ValueError("Candidate class id is outside the prompt repository")
    for row in candidate_ids.tolist():
        if len(set(row)) != len(row):
            raise ValueError("Candidate class ids must be unique within each image")
    candidate_rows = candidate_ids.tolist()

    if text_table is None:
        required_texts = _required_followup_texts(
            candidate_rows,
            prompts,
            progress=progress,
        )
        text_table = backend.build_text_feature_table(
            required_texts,
            progress=progress,
        )

    prediction_batches: list[Tensor] = []
    score_batches: list[Tensor] = []
    total, top_k = candidate_ids.shape
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            groups = [
                group
                for candidate_row in candidate_rows[start:stop]
                for group in prompts.prompts_for_candidates(candidate_row)
            ]
            class_features = backend.pool_prompt_groups(groups, text_table)
            class_features = class_features.reshape(
                stop - start,
                top_k,
                class_features.shape[-1],
            )
            batch_images = image_features[start:stop].to(
                backend.device,
                non_blocking=True,
            )
            # This is the batched equivalent of the official per-image
            # ``image_feature @ candidate_embeddings.T`` computation.
            logits = torch.bmm(
                class_features,
                batch_images.unsqueeze(2),
            ).squeeze(2)
            global_candidates = candidate_ids[start:stop].to(backend.device)
            score_batches.append(logits.cpu())
            prediction_batches.append(ordered_predictions(logits, global_candidates).cpu())
            if progress is not None:
                progress("fudd", stop, total)
    return CandidateRerankResult(
        candidate_class_ids=candidate_ids,
        scores=torch.cat(score_batches, dim=0),
        ranked_class_ids=torch.cat(prediction_batches, dim=0),
    )


def _check_official_reference_parity(
    image_features: Tensor,
    candidates: Tensor,
    candidate_rows: list[list[int]],
    fudd_predictions: Tensor,
    prompts: CubPromptRepository,
    backend: VisionLanguageBackend,
    text_table: TextFeatureTable,
    *,
    sample_count: int,
) -> ParityReport:
    """Compare batched pooling/ranking with the authors' per-image formulation."""

    checked = min(sample_count, image_features.shape[0])
    if checked <= 0:
        raise ValueError("sample_count must be positive")
    matching_predictions = 0
    max_difference = 0.0
    with torch.inference_mode():
        for index in range(checked):
            groups = prompts.prompts_for_candidates(candidate_rows[index])
            reference_prototypes: list[Tensor] = []
            for group in groups:
                row_ids = torch.tensor(
                    [text_table.index[text] for text in group],
                    device=text_table.features.device,
                )
                rows = text_table.features.index_select(0, row_ids)
                reference_prototypes.append(functional.normalize(rows.mean(dim=0), dim=0))
            reference = torch.stack(reference_prototypes)
            batched = backend.pool_prompt_groups(groups, text_table)
            difference = float((reference - batched).abs().max().item())
            max_difference = max(max_difference, difference)

            image = image_features[index].to(backend.device, non_blocking=True)
            logits = image @ reference.t()
            reference_prediction = ordered_predictions(
                logits,
                candidates[index].to(backend.device),
            ).cpu()
            if torch.equal(reference_prediction, fudd_predictions[index]):
                matching_predictions += 1

    report = ParityReport(
        samples=checked,
        matching_predictions=matching_predictions,
        max_abs_prototype_difference=max_difference,
    )
    if not report.passed:
        raise RuntimeError(
            "Batched FuDD predictions disagree with the official-style "
            f"reference loop on {checked - matching_predictions}/{checked} samples"
        )
    return report


def evaluate_cub(
    dataset: CUB200Dataset,
    prompts: CubPromptRepository,
    backend: VisionLanguageBackend,
    config: FuDDConfig,
    *,
    max_samples: int | None = None,
    parity_samples: int = 8,
    progress: ProgressCallback | None = None,
) -> EvaluationReport:
    """Evaluate a single-template baseline and FuDD reranking on CUB.

    The image encoder runs exactly once per sample.  The single-template class
    features are computed once, and all differential prompt features required
    by the selected examples are encoded in large batches and reused.

    ``max_samples`` takes a deterministic prefix of the official test split and
    is intended only for smoke tests.  Leave it as ``None`` for the reproducible
    full 5,794-image result.
    """

    if dataset.split != "test":
        raise ValueError("FuDD CUB reproduction requires split='test'")
    if dataset.transform is None:
        raise ValueError("Dataset transform must be the model backend's preprocess transform")
    if prompts.class_count != 200 or prompts.pair_count != 19_900:
        raise ValueError("The official 200-class, 19,900-pair prompts are required")
    if backend.model_name != config.model_name:
        raise ValueError("Backend model_name does not match FuDDConfig")
    if backend.text_batch_size != config.text_batch_size:
        raise ValueError("Backend text_batch_size does not match FuDDConfig")
    if backend.precision != config.precision:
        raise ValueError("Backend precision does not match FuDDConfig")
    if config.is_official_clip_reproduction and backend.feature_dtype_name != "torch.float32":
        raise ValueError("The official FuDD protocol requires FP32 CLIP features")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    if parity_samples <= 0:
        raise ValueError("parity_samples must be positive")
    validate_class_name_alignment(dataset.class_names, prompts.class_names)

    torch.manual_seed(config.seed)
    if backend.device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    dataset_fingerprint = dataset.fingerprint
    effective_samples = min(len(dataset), max_samples or len(dataset))
    image_cache_path: Path | None = None
    if config.cache_dir is not None:
        precision = _slug(f"{backend.precision}-{backend.feature_dtype_name}")
        model = _slug(config.model_name)
        identity = hashlib.sha256(backend.cache_identity.encode()).hexdigest()[:16]
        text_cache_path = (
            config.cache_dir
            / "text_features"
            / f"{model}-{identity}-{precision}-{prompts.source_digest[:16]}.pt"
        )
        backend.configure_text_cache(text_cache_path)
        sample_tag = "all" if effective_samples == len(dataset) else str(effective_samples)
        image_cache_path = (
            config.cache_dir
            / "image_features"
            / (
                f"cub-test-{model}-{identity}-{precision}-"
                f"{dataset_fingerprint[:16]}-{sample_tag}.pt"
            )
        )

    image_feature_set = backend.encode_dataset(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_samples=max_samples,
        seed=config.seed,
        cache_path=image_cache_path,
        cache_tag=(f"cub:test:{dataset_fingerprint}:{effective_samples}:{backend.cache_identity}"),
        progress=progress,
    )

    baseline_groups = prompts.single_template_prompts()
    baseline_features = backend.pool_prompt_groups(baseline_groups)
    baseline_predictions = _batched_baseline_predictions(
        image_feature_set.features,
        baseline_features,
        batch_size=config.batch_size,
        device=backend.device,
        progress=progress,
    )
    candidates = baseline_predictions[:, : config.top_k].contiguous()
    candidate_rows = candidates.tolist()

    required_texts = _required_followup_texts(
        candidate_rows,
        prompts,
        progress=progress,
    )
    differential_table = backend.build_text_feature_table(
        required_texts,
        progress=progress,
    )
    # Persist immediately after the expensive text pass so a later interruption
    # does not discard completed GPU work.
    backend.save_text_cache()

    fudd_predictions = _batched_fudd_predictions(
        image_feature_set.features,
        candidates,
        candidate_rows,
        prompts,
        backend,
        differential_table,
        batch_size=config.batch_size,
        progress=progress,
    )
    parity = _check_official_reference_parity(
        image_feature_set.features,
        candidates,
        candidate_rows,
        fudd_predictions,
        prompts,
        backend,
        differential_table,
        sample_count=parity_samples,
    )

    labels = image_feature_set.labels
    baseline_metrics = compute_topk_accuracy(baseline_predictions, labels)
    fudd_accuracy = compute_topk_accuracy(fudd_predictions, labels)
    candidate_hits = int(topk_hits(baseline_predictions, labels, config.top_k).sum().item())
    fudd_metrics = FuDDMetrics(
        total=fudd_accuracy.total,
        top1_correct=fudd_accuracy.top1_correct,
        top5_correct=fudd_accuracy.top5_correct,
        candidate_hits=candidate_hits,
    )
    transfers = compute_transfer_counts(
        baseline_predictions,
        fudd_predictions,
        labels,
    )
    prediction_records: list[PredictionRecord] = []
    for row_index, sample_index in enumerate(image_feature_set.sample_indices.tolist()):
        sample = dataset.samples[sample_index]
        target_class_id = int(labels[row_index].item())
        if sample.target != target_class_id:
            raise RuntimeError("Cached labels do not align with CUB sample metadata")
        prediction_records.append(
            PredictionRecord(
                sample_index=sample_index,
                image_id=sample.image_id,
                relative_path=sample.relative_path.as_posix(),
                target_class_id=target_class_id,
                baseline_topk_class_ids=tuple(
                    int(value) for value in candidates[row_index].tolist()
                ),
                fudd_ranked_class_ids=tuple(
                    int(value) for value in fudd_predictions[row_index].tolist()
                ),
            )
        )
    return EvaluationReport(
        config=config.to_dict(),
        num_samples=image_feature_set.size,
        baseline=baseline_metrics,
        fudd=fudd_metrics,
        transfers=transfers,
        parity=parity,
        prompt_digest=prompts.source_digest,
        dataset_fingerprint=dataset_fingerprint,
        feature_dtype=backend.feature_dtype_name,
        predictions=tuple(prediction_records),
    )


def run_clip_cub_experiment(
    config: FuDDConfig,
    *,
    max_samples: int | None = None,
    parity_samples: int = 8,
    download_data: bool = True,
    download_prompts: bool = True,
    verify_images: bool = True,
    progress: ProgressCallback | None = None,
) -> EvaluationReport:
    """Run the complete CLIP + FuDD preparation and evaluation pipeline."""

    if download_prompts:
        download_official_prompts(config.prompt_root)
    prompt_repository = load_official_prompts(config.prompt_root)
    model_cache_dir = None if config.cache_dir is None else config.cache_dir / "models"
    backend = CLIPBackend(
        model_name=config.model_name,
        device=config.device,
        precision=config.precision,
        text_batch_size=config.text_batch_size,
        model_cache_dir=model_cache_dir,
    )
    dataset = prepare_cub(
        config.data_root,
        transform=backend.preprocess,
        download=download_data,
        verify_images=verify_images,
        split="test",
    )
    return evaluate_cub(
        dataset,
        prompt_repository,
        backend,
        config,
        max_samples=max_samples,
        parity_samples=parity_samples,
        progress=progress,
    )
