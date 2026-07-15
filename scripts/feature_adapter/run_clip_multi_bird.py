#!/usr/bin/env python3
"""Fit a class-agnostic CLIP adapter on BirdMix and evaluate untouched CUB."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ttvr.data.bird_manifest import (
    BirdSample,
    BirdTaxon,
    ManifestBirdDataset,
    load_samples,
    load_taxa,
)
from ttvr.data.cub import CUB_TEST_COUNT, CUB200Dataset, prepare_cub
from ttvr.data.cub_taxonomy import build_cub_birdnet_crosswalk, read_cub_class_names
from ttvr.experiments import (
    atomic_torch_save,
    create_run_directory,
    sha256_file,
    write_checksums,
    write_json,
    write_jsonl,
)
from ttvr.methods.feature_adapter import (
    AdapterTrainConfig,
    FeatureTask,
    PreparedFeatureTask,
    ResidualFeatureAdapter,
    build_feature_task,
    fit_feature_adapter,
    refit_feature_adapter,
    stable_taxon_partition,
)
from ttvr.metrics import (
    compute_topk_accuracy,
    compute_transfer_counts,
    exact_mcnemar_test,
    ordered_predictions,
    paired_bootstrap_accuracy_gain,
)
from ttvr.models.cached import normalise_features, torch_load_tensors
from ttvr.models.clip import CLIPBackend


@dataclass(frozen=True, slots=True)
class SourceSpec:
    dataset_id: str
    root: Path
    samples: Path
    taxa: Path
    train_cache: Path
    validation_cache: Path | None


@dataclass(frozen=True, slots=True)
class CachedSplit:
    dataset: ManifestBirdDataset
    features: torch.Tensor
    labels: torch.Tensor
    cache_path: Path


@dataclass(frozen=True, slots=True)
class LoadedSource:
    spec: SourceSpec
    samples: tuple[BirdSample, ...]
    taxa: tuple[BirdTaxon, ...]
    train: CachedSplit
    validation: CachedSplit | None


@dataclass(frozen=True, slots=True)
class DuplicateAuditConfig:
    """Non-destructive manifest audit plus an optional exact-byte filter."""

    exact_sha256_policy: str = "audit-only"
    perceptual_hash_policy: str = "report-only"
    perceptual_hamming_threshold: int = 0

    def __post_init__(self) -> None:
        if self.exact_sha256_policy not in {"audit-only", "drop-later-source"}:
            raise ValueError(
                "duplicate_audit.exact_sha256_policy must be audit-only or drop-later-source"
            )
        if self.perceptual_hash_policy != "report-only":
            raise ValueError("duplicate_audit.perceptual_hash_policy must remain report-only")
        if (
            type(self.perceptual_hamming_threshold) is not int
            or not 0 <= self.perceptual_hamming_threshold <= 8
        ):
            raise ValueError(
                "duplicate_audit.perceptual_hamming_threshold must be an integer "
                "between zero and eight"
            )

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _source_indices_sha256(indices: torch.Tensor) -> str:
    """Hash a one-dimensional source-index vector with a portable encoding."""

    values = indices.detach().cpu().long().contiguous()
    if values.ndim != 1 or bool(values.lt(0).any()):
        raise ValueError("source indices must be a non-negative vector")
    digest = hashlib.sha256(b"ttvr-source-indices-little-endian-int64-v1\0")
    for start in range(0, values.numel(), 8_192):
        chunk = values[start : start + 8_192].tolist()
        digest.update(struct.pack(f"<{len(chunk)}q", *chunk))
    return digest.hexdigest()


def _source_task_membership_row(
    prepared: PreparedFeatureTask,
    *,
    dataset_id: str,
    role: str,
    source_index_segments: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Describe exact task taxa and fingerprint its source-row membership."""

    if role not in {"selection_train", "unseen_validation", "refit"}:
        raise ValueError(f"Unknown source task role: {role}")
    indices = prepared.source_indices.detach().cpu().long().contiguous()
    if indices.ndim != 1 or indices.numel() != prepared.task.size:
        raise RuntimeError("Prepared task indices do not align with task features")
    index_space_size = 0
    for segment in source_index_segments:
        start = int(segment["start"])
        stop = int(segment["stop"])
        if start != index_space_size or stop <= start:
            raise ValueError("source index segments must be non-empty and contiguous")
        index_space_size = stop
    if index_space_size == 0:
        raise ValueError("source index space must be non-empty")
    if indices.numel() and (
        int(indices.min().item()) < 0 or int(indices.max().item()) >= index_space_size
    ):
        raise RuntimeError("Prepared task indices escape their declared index space")
    if not torch.equal(indices, torch.unique(indices, sorted=True)):
        raise RuntimeError("Prepared task indices must be unique and sorted")
    if len(prepared.taxon_ids) != prepared.task.class_count:
        raise RuntimeError("Prepared task taxa do not align with its text prototypes")
    return {
        "dataset_id": dataset_id,
        "task_name": prepared.task.name,
        "role": role,
        "sample_count": prepared.task.size,
        "class_count": prepared.task.class_count,
        "taxon_ids": prepared.taxon_ids,
        "source_indices_count": indices.numel(),
        "source_indices_sha256": _source_indices_sha256(indices),
        "source_indices_encoding": "little-endian-int64-v1",
        "source_index_space_size": index_space_size,
        "source_index_segments": source_index_segments,
    }


def _resolve_one_glob(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    if any(character in str(path) for character in "*?["):
        matches = sorted(path.parent.glob(path.name))
        if len(matches) != 1:
            raise RuntimeError(f"Cache glob must resolve exactly once: {path} -> {matches}")
        return matches[0]
    return path


def _source_specs(
    path: Path,
    project_root: Path,
) -> tuple[tuple[SourceSpec, ...], float]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("source config must contain a non-empty sources list")

    enabled_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("every source config entry must be an object")
        enabled = row.get("enabled", True)
        if type(enabled) is not bool:
            raise ValueError("source enabled flags must be boolean")
        if enabled:
            enabled_rows.append(row)
        elif not isinstance(row.get("dataset_id"), str) or not row["dataset_id"].strip():
            raise ValueError("disabled source placeholders still need a dataset_id")
    if not enabled_rows:
        raise ValueError("source config must enable at least one source")

    top_level_fraction = value.get("validation_taxon_fraction")
    legacy_fractions = {
        float(row["validation_taxon_fraction"])
        for row in enabled_rows
        if row.get("validation_taxon_fraction") is not None
    }
    if top_level_fraction is not None and legacy_fractions:
        raise ValueError("validation_taxon_fraction must appear only at source-config top level")
    if top_level_fraction is None:
        if len(legacy_fractions) != 1:
            raise ValueError("source config needs one top-level validation_taxon_fraction")
        validation_taxon_fraction = next(iter(legacy_fractions))
    else:
        validation_taxon_fraction = float(top_level_fraction)
    if not 0.0 < validation_taxon_fraction < 1.0:
        raise ValueError("validation_taxon_fraction must be between zero and one")

    result: list[SourceSpec] = []
    for row in enabled_rows:
        validation_value = row.get("validation_cache")
        result.append(
            SourceSpec(
                dataset_id=str(row["dataset_id"]),
                root=_resolve_one_glob(project_root, str(row["root"])),
                samples=_resolve_one_glob(project_root, str(row["samples"])),
                taxa=_resolve_one_glob(project_root, str(row["taxa"])),
                train_cache=_resolve_one_glob(project_root, str(row["train_cache"])),
                validation_cache=(
                    None
                    if validation_value is None
                    else _resolve_one_glob(project_root, str(validation_value))
                ),
            )
        )
    if len({source.dataset_id for source in result}) != len(result):
        raise ValueError("source dataset ids must be unique")
    return tuple(result), validation_taxon_fraction


def _duplicate_audit_config(path: Path) -> DuplicateAuditConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("source config must be a JSON object")
    raw = value.get("duplicate_audit")
    if raw is None:
        return DuplicateAuditConfig()
    if not isinstance(raw, dict):
        raise ValueError("duplicate_audit must be an object")
    expected = {
        "exact_sha256_policy",
        "perceptual_hash_policy",
        "perceptual_hamming_threshold",
    }
    missing = sorted(expected - raw.keys())
    unknown = sorted(raw.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"duplicate_audit keys do not match schema; missing={missing}, unknown={unknown}"
        )
    return DuplicateAuditConfig(
        exact_sha256_policy=raw["exact_sha256_policy"],
        perceptual_hash_policy=raw["perceptual_hash_policy"],
        perceptual_hamming_threshold=raw["perceptual_hamming_threshold"],
    )


def _disabled_source_rows(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value.get("sources") if isinstance(value, dict) else None
    if not isinstance(rows, list):
        return []
    result: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("enabled", True) is not False:
            continue
        result.append(
            {
                "dataset_id": str(row["dataset_id"]),
                "reason": str(row.get("reason", "disabled source placeholder")),
            }
        )
    return result


def _validated_aligned_feature_tensors(
    payload: dict[str, Any],
    *,
    expected_labels: torch.Tensor,
    feature_dim: int,
    context: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fail closed on malformed, misaligned, or unnormalised feature rows."""

    features = payload.get("features")
    labels = payload.get("labels")
    indices = payload.get("sample_indices")
    if not all(isinstance(value, torch.Tensor) for value in (features, labels, indices)):
        raise RuntimeError(f"Feature cache tensors are missing: {context}")
    assert isinstance(features, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    sample_count = expected_labels.numel()
    if features.shape != (sample_count, feature_dim) or not features.is_floating_point():
        raise RuntimeError(
            f"Feature cache has invalid feature shape or dtype: {context}; "
            f"expected ({sample_count}, {feature_dim}) floating-point, "
            f"found {tuple(features.shape)} {features.dtype}"
        )
    if labels.shape != (sample_count,) or labels.dtype != torch.long:
        raise RuntimeError(f"Feature cache labels have invalid shape or dtype: {context}")
    if indices.shape != (sample_count,) or indices.dtype != torch.long:
        raise RuntimeError(f"Feature cache indices have invalid shape or dtype: {context}")
    if not torch.equal(indices, torch.arange(sample_count, dtype=torch.long)):
        raise RuntimeError(f"Feature cache indices do not align: {context}")
    if not torch.equal(labels, expected_labels):
        raise RuntimeError(f"Feature cache labels do not align: {context}")
    for start in range(0, sample_count, 8_192):
        chunk = features[start : start + 8_192].float()
        if not bool(torch.isfinite(chunk).all()):
            raise RuntimeError(f"Feature cache contains non-finite values: {context}")
        norms = torch.linalg.vector_norm(chunk, dim=1)
        if not torch.allclose(
            norms,
            torch.ones_like(norms),
            atol=5e-4,
            rtol=5e-4,
        ):
            maximum_error = float((norms - 1.0).abs().max().item())
            raise RuntimeError(
                f"Feature cache rows are not unit normalised: {context}; "
                f"maximum norm error={maximum_error:.8g}"
            )
    return (
        features.detach().cpu().float().contiguous(),
        labels.detach().cpu().long().contiguous(),
    )


def _load_cached_split(
    spec: SourceSpec,
    samples: tuple[BirdSample, ...],
    taxa: tuple[BirdTaxon, ...],
    *,
    split: str,
    cache_path: Path,
    backend: CLIPBackend,
) -> CachedSplit:
    dataset = ManifestBirdDataset(
        spec.root,
        samples,
        taxa,
        splits=(split,),
        verify_images=False,
    )
    payload = torch_load_tensors(cache_path)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid feature cache: {cache_path}")
    expected_tag = f"{spec.dataset_id}:{split}:{dataset.fingerprint}:{backend.cache_identity}"
    if (
        payload.get("format") != 2
        or payload.get("cache_identity") != backend.cache_identity
        or payload.get("model_name") != backend.model_name
        or payload.get("precision") != backend.precision
        or payload.get("dtype") != backend.feature_dtype_name
        or payload.get("cache_tag") != expected_tag
        or payload.get("sample_count") != len(dataset)
    ):
        raise RuntimeError(f"Feature cache metadata does not match manifest: {cache_path}")
    expected_labels = torch.tensor(dataset.targets, dtype=torch.long)
    features, labels = _validated_aligned_feature_tensors(
        payload,
        expected_labels=expected_labels,
        feature_dim=768,
        context=str(cache_path),
    )
    return CachedSplit(
        dataset=dataset,
        features=features,
        labels=labels,
        cache_path=cache_path,
    )


def _validated_cub_target_cache(
    payload: Any,
    cub_dataset: CUB200Dataset,
    backend: CLIPBackend,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the frozen CUB test cache before any target evaluation."""

    if len(cub_dataset) != CUB_TEST_COUNT:
        raise RuntimeError(
            f"CUB target dataset must contain {CUB_TEST_COUNT} test images; "
            f"found {len(cub_dataset)}"
        )
    if not isinstance(payload, dict):
        raise RuntimeError("Invalid CUB feature cache payload")
    expected_tag = f"cub:test:{cub_dataset.fingerprint}:{backend.cache_identity}"
    if (
        payload.get("format") != 2
        or payload.get("cache_identity") != backend.cache_identity
        or payload.get("model_name") != backend.model_name
        or payload.get("precision") != backend.precision
        or payload.get("dtype") != backend.feature_dtype_name
        or payload.get("cache_tag") != expected_tag
        or payload.get("sample_count") != CUB_TEST_COUNT
    ):
        raise RuntimeError("CUB target cache metadata does not match dataset and backend")
    expected_labels = torch.tensor(cub_dataset.targets, dtype=torch.long)
    return _validated_aligned_feature_tensors(
        payload,
        expected_labels=expected_labels,
        feature_dim=768,
        context="CUB target cache",
    )


def _load_sources(
    specs: tuple[SourceSpec, ...],
    backend: CLIPBackend,
) -> tuple[LoadedSource, ...]:
    result: list[LoadedSource] = []
    for spec in specs:
        samples = load_samples(spec.samples)
        taxa = load_taxa(spec.taxa)
        if {sample.dataset_id for sample in samples} != {spec.dataset_id}:
            raise RuntimeError(f"Configured id does not match manifest: {spec.dataset_id}")
        train = _load_cached_split(
            spec,
            samples,
            taxa,
            split="train",
            cache_path=spec.train_cache,
            backend=backend,
        )
        validation = None
        if spec.validation_cache is not None:
            validation = _load_cached_split(
                spec,
                samples,
                taxa,
                split="validation",
                cache_path=spec.validation_cache,
                backend=backend,
            )
        result.append(
            LoadedSource(
                spec=spec,
                samples=samples,
                taxa=taxa,
                train=train,
                validation=validation,
            )
        )
    return tuple(result)


def _sample_occurrence(
    sample: BirdSample,
    *,
    manifest_index: int,
    eligible_for_fit: bool,
) -> dict[str, Any]:
    return {
        "dataset_id": sample.dataset_id,
        "source_sample_id": sample.source_sample_id,
        "source_split": sample.source_split,
        "taxon_id": sample.taxon_id,
        "manifest_index": manifest_index,
        "eligible_for_fit": eligible_for_fit,
    }


def _phash_chunk_keys(value: int, threshold: int) -> tuple[tuple[int, int], ...]:
    """Partition 64 bits so hashes within ``threshold`` share one chunk."""

    parts = threshold + 1
    base, remainder = divmod(64, parts)
    keys: list[tuple[int, int]] = []
    shift = 0
    for part in range(parts):
        width = base + int(part < remainder)
        keys.append((part, (value >> shift) & ((1 << width) - 1)))
        shift += width
    return tuple(keys)


def _near_phash_rows(
    occurrences_by_phash: dict[str, list[dict[str, Any]]],
    *,
    threshold: int,
) -> list[dict[str, Any]]:
    """Enumerate cross-source pHash candidates without quadratic all-pairs work."""

    if threshold <= 0:
        return []
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    result: list[dict[str, Any]] = []
    dataset_ids = {
        phash: {str(row["dataset_id"]) for row in rows}
        for phash, rows in occurrences_by_phash.items()
    }
    for phash in sorted(occurrences_by_phash):
        value = int(phash, 16)
        keys = _phash_chunk_keys(value, threshold)
        candidates = {candidate for key in keys for candidate in buckets.get(key, ())}
        for candidate in sorted(candidates):
            distance = (value ^ int(candidate, 16)).bit_count()
            if distance > threshold:
                continue
            left_sources = dataset_ids[candidate]
            right_sources = dataset_ids[phash]
            if not any(left != right for left in left_sources for right in right_sources):
                continue
            result.append(
                {
                    "phash_a": candidate,
                    "phash_b": phash,
                    "hamming_distance": distance,
                    "dataset_ids_a": sorted(left_sources),
                    "dataset_ids_b": sorted(right_sources),
                    "occurrence_count_a": len(occurrences_by_phash[candidate]),
                    "occurrence_count_b": len(occurrences_by_phash[phash]),
                    "action": "report-only",
                }
            )
        for key in keys:
            buckets[key].append(phash)
    return result


def _audit_cross_source_duplicates(
    sources: tuple[LoadedSource, ...],
    config: DuplicateAuditConfig,
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    """Audit all manifests and return active rows excluded by exact-byte policy.

    Canonical manifests are never rewritten.  Source ``test`` rows can appear
    in the report, but only rows present in the configured train/validation
    caches are eligible for exact-byte filtering or model fitting.
    """

    active_ids: dict[str, set[str]] = {}
    source_order = {source.spec.dataset_id: index for index, source in enumerate(sources)}
    sha_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    phash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in sources:
        active = {sample.source_sample_id for sample in source.train.dataset.samples}
        if source.validation is not None:
            active.update(sample.source_sample_id for sample in source.validation.dataset.samples)
        active_ids[source.spec.dataset_id] = active
        for index, sample in enumerate(source.samples):
            occurrence = _sample_occurrence(
                sample,
                manifest_index=index,
                eligible_for_fit=sample.source_sample_id in active,
            )
            sha_groups[sample.sha256].append(occurrence)
            if sample.phash:
                phash_groups[sample.phash].append(occurrence)

    dropped: dict[str, set[str]] = {source.spec.dataset_id: set() for source in sources}
    exact_rows: list[dict[str, Any]] = []
    for sha256, occurrences in sorted(sha_groups.items()):
        if len({str(row["dataset_id"]) for row in occurrences}) < 2:
            continue
        ordered = sorted(
            occurrences,
            key=lambda row: (
                source_order[str(row["dataset_id"])],
                int(row["manifest_index"]),
            ),
        )
        active = [row for row in ordered if bool(row["eligible_for_fit"])]
        canonical = active[0] if active else None
        dropped_rows: list[dict[str, Any]] = []
        active_datasets = {str(row["dataset_id"]) for row in active}
        if (
            config.exact_sha256_policy == "drop-later-source"
            and canonical is not None
            and len(active_datasets) >= 2
        ):
            canonical_dataset = str(canonical["dataset_id"])
            for row in active:
                if str(row["dataset_id"]) == canonical_dataset:
                    continue
                dataset_id = str(row["dataset_id"])
                source_sample_id = str(row["source_sample_id"])
                dropped[dataset_id].add(source_sample_id)
                dropped_rows.append(row)
        exact_rows.append(
            {
                "sha256": sha256,
                "occurrences": ordered,
                "active_occurrence_count": len(active),
                "canonical_active_occurrence": canonical,
                "dropped_active_occurrences": dropped_rows,
                "taxon_conflict": len({str(row["taxon_id"]) for row in ordered}) > 1,
                "action": config.exact_sha256_policy,
            }
        )

    identical_phash_rows: list[dict[str, Any]] = []
    for phash, occurrences in sorted(phash_groups.items()):
        if len({str(row["dataset_id"]) for row in occurrences}) < 2:
            continue
        identical_phash_rows.append(
            {
                "phash": phash,
                "occurrences": sorted(
                    occurrences,
                    key=lambda row: (
                        source_order[str(row["dataset_id"])],
                        int(row["manifest_index"]),
                    ),
                ),
                "taxon_conflict": len({str(row["taxon_id"]) for row in occurrences}) > 1,
                "action": "report-only",
            }
        )

    near_phash_rows = _near_phash_rows(
        phash_groups,
        threshold=config.perceptual_hamming_threshold,
    )
    dropped_counts = {dataset_id: len(sample_ids) for dataset_id, sample_ids in dropped.items()}
    audit = {
        "schema_version": 1,
        "scope": "all canonical rows reported; only configured train/validation rows fit",
        "config": config.to_dict(),
        "source_order": [source.spec.dataset_id for source in sources],
        "exact_sha256_cross_source_groups": exact_rows,
        "identical_phash_cross_source_groups": identical_phash_rows,
        "near_phash_cross_source_pairs": near_phash_rows,
        "summary": {
            "canonical_manifest_samples": sum(len(source.samples) for source in sources),
            "fit_eligible_samples_before_exact_filter": sum(
                len(values) for values in active_ids.values()
            ),
            "exact_sha256_cross_source_group_count": len(exact_rows),
            "identical_phash_cross_source_group_count": len(identical_phash_rows),
            "near_phash_cross_source_pair_count": len(near_phash_rows),
            "exact_duplicate_rows_dropped_by_source": dropped_counts,
            "exact_duplicate_rows_dropped_total": sum(dropped_counts.values()),
            "perceptual_duplicate_rows_dropped_total": 0,
        },
    }
    return audit, dropped


def _prototype_map(
    backend: CLIPBackend,
    taxa_by_id: dict[str, BirdTaxon],
) -> tuple[dict[str, torch.Tensor], list[dict[str, str]]]:
    taxon_ids = tuple(sorted(taxa_by_id))
    groups: list[tuple[str, str]] = []
    rows: list[dict[str, str]] = []
    for taxon_id in taxon_ids:
        taxon = taxa_by_id[taxon_id]
        prompts = (
            f"a photo of the bird species called {taxon.common_name}.",
            f"a photo of the bird species {taxon.scientific_name}.",
        )
        groups.append(prompts)
        rows.extend(
            {
                "taxon_id": taxon_id,
                "scientific_name": taxon.scientific_name,
                "common_name": taxon.common_name,
                "prompt": prompt,
            }
            for prompt in prompts
        )
    table = backend.build_text_feature_table(prompt for group in groups for prompt in group)
    prototypes = backend.pool_prompt_groups(groups, table).detach().cpu().float()
    return (
        {taxon_id: prototypes[index] for index, taxon_id in enumerate(taxon_ids)},
        rows,
    )


def _excluded_dataset_indices(
    split: CachedSplit,
    dropped_source_sample_ids: set[str],
) -> set[int]:
    return {
        index
        for index, sample in enumerate(split.dataset.samples)
        if sample.source_sample_id in dropped_source_sample_ids
    }


def _usable_split_taxa(
    split: CachedSplit,
    *,
    excluded_taxa: set[str],
    dropped_source_sample_ids: set[str],
) -> set[str]:
    return {
        sample.taxon_id
        for sample in split.dataset.samples
        if sample.taxon_id not in excluded_taxa
        and sample.source_sample_id not in dropped_source_sample_ids
    }


def _combined_refit_rows(
    source: LoadedSource,
    *,
    dropped_source_sample_ids: set[str],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[str, ...],
    tuple[dict[str, Any], ...],
    set[int],
]:
    """Merge cached splits while remapping their split-local label spaces."""

    splits: list[tuple[str, CachedSplit]] = [("train", source.train)]
    if source.validation is not None:
        splits.append(("validation", source.validation))
    combined_taxa = tuple(
        sorted({taxon_id for _, split in splits for taxon_id in split.dataset.taxon_ids})
    )
    combined_label = {taxon_id: index for index, taxon_id in enumerate(combined_taxa)}
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    segments: list[dict[str, Any]] = []
    excluded_indices: set[int] = set()
    offset = 0
    for split_name, split in splits:
        feature_rows.append(split.features)
        remap = torch.tensor(
            [combined_label[taxon_id] for taxon_id in split.dataset.taxon_ids],
            dtype=torch.long,
        )
        label_rows.append(remap.index_select(0, split.labels))
        stop = offset + len(split.dataset)
        segments.append(
            {
                "split": split_name,
                "start": offset,
                "stop": stop,
                "cache_path": str(split.cache_path.resolve()),
            }
        )
        excluded_indices.update(
            offset + index
            for index in _excluded_dataset_indices(
                split,
                dropped_source_sample_ids,
            )
        )
        offset = stop
    return (
        torch.cat(feature_rows, dim=0),
        torch.cat(label_rows, dim=0),
        combined_taxa,
        tuple(segments),
        excluded_indices,
    )


def _role_audit(
    *,
    status: str,
    eligible_taxa: set[str],
    prepared: PreparedFeatureTask | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": status,
        "eligible_taxon_count": len(eligible_taxa),
        "eligible_taxa": sorted(eligible_taxa),
        "task_sample_count": 0,
        "task_class_count": 0,
        "task_taxa": [],
    }
    if prepared is not None:
        row.update(
            {
                "task_sample_count": prepared.task.size,
                "task_class_count": prepared.task.class_count,
                "task_taxa": list(prepared.taxon_ids),
            }
        )
    return row


def _prepare_source_tasks(
    sources: tuple[LoadedSource, ...],
    backend: CLIPBackend,
    *,
    excluded_taxa: set[str],
    validation_taxon_fraction: float,
    duplicate_excluded_sample_ids: dict[str, set[str]] | None = None,
) -> tuple[
    tuple[FeatureTask, ...],
    tuple[FeatureTask, ...],
    tuple[FeatureTask, ...],
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    list[dict[str, str]],
    tuple[str, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    duplicate_excluded = duplicate_excluded_sample_ids or {}
    taxa_by_id: dict[str, BirdTaxon] = {}
    exclusion_rows: list[dict[str, Any]] = []
    for source in sources:
        source_taxa = {taxon.taxon_id: taxon for taxon in source.taxa}
        dropped = duplicate_excluded.get(source.spec.dataset_id, set())
        cached_splits = (source.train,) + (
            () if source.validation is None else (source.validation,)
        )
        retained = set().union(
            *(
                _usable_split_taxa(
                    split,
                    excluded_taxa=excluded_taxa,
                    dropped_source_sample_ids=dropped,
                )
                for split in cached_splits
            )
        )
        for taxon_id in retained:
            taxon = source_taxa[taxon_id]
            previous = taxa_by_id.get(taxon_id)
            if previous is not None and previous != taxon:
                raise RuntimeError(f"Conflicting taxonomy rows for {taxon_id}")
            taxa_by_id[taxon_id] = taxon
        counts: dict[str, int] = {}
        for sample in source.samples:
            if sample.taxon_id in excluded_taxa:
                counts[sample.taxon_id] = counts.get(sample.taxon_id, 0) + 1
        for taxon_id, count in sorted(counts.items()):
            exclusion_rows.append(
                {
                    "dataset_id": source.spec.dataset_id,
                    "taxon_id": taxon_id,
                    "sample_count": count,
                    "reason": "strict_cub_target_taxon",
                }
            )
    prototypes, prompt_rows = _prototype_map(backend, taxa_by_id)

    _, global_validation_taxa = stable_taxon_partition(
        tuple(sorted(taxa_by_id)),
        validation_fraction=validation_taxon_fraction,
        salt="ttvr-birdmix-global-source-species-validation-v1",
    )
    global_validation_set = set(global_validation_taxa)

    train_tasks: list[FeatureTask] = []
    validation_tasks: list[FeatureTask] = []
    refit_tasks: list[FeatureTask] = []
    membership_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    for source in sources:
        dropped = duplicate_excluded.get(source.spec.dataset_id, set())
        usable_train_taxa = _usable_split_taxa(
            source.train,
            excluded_taxa=excluded_taxa,
            dropped_source_sample_ids=dropped,
        )
        selection_train_taxa = usable_train_taxa - global_validation_set
        train_segment = (
            {
                "split": "train",
                "start": 0,
                "stop": len(source.train.dataset),
                "cache_path": str(source.train.cache_path.resolve()),
            },
        )
        train_prepared: PreparedFeatureTask | None = None
        train_status = "omitted-fewer-than-two-eligible-taxa"
        if len(selection_train_taxa) >= 2:
            train_prepared = build_feature_task(
                f"{source.spec.dataset_id}:selection-train",
                source.train.features,
                source.train.labels,
                source.train.dataset.taxon_ids,
                prototypes,
                included_taxon_ids=selection_train_taxa,
                excluded_source_indices=_excluded_dataset_indices(source.train, dropped),
            )
            train_tasks.append(train_prepared.task)
            membership_rows.append(
                _source_task_membership_row(
                    train_prepared,
                    dataset_id=source.spec.dataset_id,
                    role="selection_train",
                    source_index_segments=train_segment,
                )
            )
            train_status = "included"

        validation_split = source.validation or source.train
        usable_validation_taxa = _usable_split_taxa(
            validation_split,
            excluded_taxa=excluded_taxa,
            dropped_source_sample_ids=dropped,
        )
        source_validation_taxa = usable_validation_taxa & global_validation_set
        validation_source_split = "validation" if source.validation is not None else "train"
        validation_segment = (
            {
                "split": validation_source_split,
                "start": 0,
                "stop": len(validation_split.dataset),
                "cache_path": str(validation_split.cache_path.resolve()),
            },
        )
        validation_prepared: PreparedFeatureTask | None = None
        validation_status = "omitted-fewer-than-two-eligible-taxa"
        if len(source_validation_taxa) >= 2:
            validation_prepared = build_feature_task(
                f"{source.spec.dataset_id}:unseen-species-validation",
                validation_split.features,
                validation_split.labels,
                validation_split.dataset.taxon_ids,
                prototypes,
                included_taxon_ids=source_validation_taxa,
                excluded_source_indices=_excluded_dataset_indices(
                    validation_split,
                    dropped,
                ),
            )
            validation_tasks.append(validation_prepared.task)
            membership_rows.append(
                _source_task_membership_row(
                    validation_prepared,
                    dataset_id=source.spec.dataset_id,
                    role="unseen_validation",
                    source_index_segments=validation_segment,
                )
            )
            validation_status = "included"

        (
            refit_features,
            refit_labels,
            refit_taxon_ids,
            refit_segments,
            refit_excluded_indices,
        ) = _combined_refit_rows(
            source,
            dropped_source_sample_ids=dropped,
        )
        refit_usable_taxa = set().union(
            usable_train_taxa,
            *(
                ()
                if source.validation is None
                else (
                    _usable_split_taxa(
                        source.validation,
                        excluded_taxa=excluded_taxa,
                        dropped_source_sample_ids=dropped,
                    ),
                )
            ),
        )
        refit_prepared: PreparedFeatureTask | None = None
        refit_status = "omitted-fewer-than-two-eligible-taxa"
        if len(refit_usable_taxa) >= 2:
            refit_prepared = build_feature_task(
                f"{source.spec.dataset_id}:refit-all-source-species",
                refit_features,
                refit_labels,
                refit_taxon_ids,
                prototypes,
                included_taxon_ids=refit_usable_taxa,
                excluded_source_indices=refit_excluded_indices,
            )
            refit_tasks.append(refit_prepared.task)
            membership_rows.append(
                _source_task_membership_row(
                    refit_prepared,
                    dataset_id=source.spec.dataset_id,
                    role="refit",
                    source_index_segments=refit_segments,
                )
            )
            refit_status = "included"

        all_split_counts = Counter(sample.source_split for sample in source.samples)
        active_ids = {sample.source_sample_id for sample in source.train.dataset.samples}
        if source.validation is not None:
            active_ids.update(
                sample.source_sample_id for sample in source.validation.dataset.samples
            )
        partition_rows.append(
            {
                "dataset_id": source.spec.dataset_id,
                "canonical_manifest_split_counts": dict(sorted(all_split_counts.items())),
                "configured_cache_splits": [
                    "train",
                    *([] if source.validation is None else ["validation"]),
                ],
                "official_test_manifest_samples": all_split_counts.get("test", 0),
                "official_test_samples_used": 0,
                "fit_eligible_samples_before_filters": len(active_ids),
                "strict_cub_excluded_active_samples": sum(
                    sample.source_sample_id in active_ids and sample.taxon_id in excluded_taxa
                    for sample in source.samples
                ),
                "exact_duplicate_excluded_active_samples": len(dropped),
                "selection_train": _role_audit(
                    status=train_status,
                    eligible_taxa=selection_train_taxa,
                    prepared=train_prepared,
                ),
                "unseen_validation": _role_audit(
                    status=validation_status,
                    eligible_taxa=source_validation_taxa,
                    prepared=validation_prepared,
                ),
                "refit": _role_audit(
                    status=refit_status,
                    eligible_taxa=refit_usable_taxa,
                    prepared=refit_prepared,
                ),
            }
        )

    if not train_tasks:
        raise RuntimeError("No source retains a two-class selection-training task")
    if not validation_tasks:
        raise RuntimeError(
            "Global validation split produced no source task with at least two held-out taxa"
        )
    if not refit_tasks:
        raise RuntimeError("No source retains a two-class refit task")
    selection_taxa = {
        taxon_id
        for row in membership_rows
        if row["role"] == "selection_train"
        for taxon_id in row["taxon_ids"]
    }
    used_validation_taxa = {
        taxon_id
        for row in membership_rows
        if row["role"] == "unseen_validation"
        for taxon_id in row["taxon_ids"]
    }
    if not used_validation_taxa <= global_validation_set:
        raise RuntimeError("Validation task escaped the global held-out taxa")
    if selection_taxa & global_validation_set:
        raise RuntimeError("A globally held-out taxon entered selection training")
    return (
        tuple(train_tasks),
        tuple(validation_tasks),
        tuple(refit_tasks),
        prototypes,
        exclusion_rows,
        prompt_rows,
        global_validation_taxa,
        membership_rows,
        partition_rows,
    )


def _predict_cub(
    adapter: ResidualFeatureAdapter,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline: list[torch.Tensor] = []
    adapted: list[torch.Tensor] = []
    text = normalise_features(text_features.float()).to(device)
    adapter.eval()
    with torch.inference_mode():
        for start in range(0, image_features.shape[0], batch_size):
            frozen = normalise_features(image_features[start : start + batch_size].float()).to(
                device
            )
            baseline.append(ordered_predictions(frozen @ text.t())[:, :5].cpu())
            adapted.append(ordered_predictions(adapter(frozen) @ text.t())[:, :5].cpu())
    return torch.cat(baseline), torch.cat(adapted)


def _species_cluster_bootstrap(
    baseline_correct: torch.Tensor,
    adapted_correct: torch.Tensor,
    labels: torch.Tensor,
    *,
    reps: int,
    seed: int,
) -> dict[str, float | int]:
    class_ids = tuple(sorted(set(labels.tolist())))
    per_class_delta = torch.tensor(
        [
            (
                adapted_correct[labels == class_id].float().mean()
                - baseline_correct[labels == class_id].float().mean()
            ).item()
            for class_id in class_ids
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(len(class_ids), (reps, len(class_ids)), generator=generator)
    gains = per_class_delta.index_select(0, draws.reshape(-1)).reshape(reps, -1).mean(dim=1)
    return {
        "clusters": len(class_ids),
        "reps": reps,
        "seed": seed,
        "mean_gain_percentage_points": 100.0 * float(per_class_delta.mean().item()),
        "ci95_low_percentage_points": 100.0 * float(torch.quantile(gains, 0.025).item()),
        "ci95_high_percentage_points": 100.0 * float(torch.quantile(gains, 0.975).item()),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": _package_version("torchvision"),
        "clip": _package_version("clip"),
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
    }


def _model_runtime_config(backend: CLIPBackend) -> dict[str, Any]:
    """Serialize the identity verified by this model-consuming interpreter."""

    installation = backend.openai_clip_installation
    checkpoint = backend.openai_clip_checkpoint
    if checkpoint is None:
        raise RuntimeError("The formal CLIP run requires a verified local checkpoint")
    return {
        "cache_identity": backend.cache_identity,
        "clip_distribution": installation.distribution,
        "clip_version": installation.version,
        "clip_repository_url": installation.repository_url,
        "clip_commit": installation.commit_id,
        "checkpoint_path": str(checkpoint.path),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_size_bytes": checkpoint.size_bytes,
    }


def _source_digest_paths(
    project_root: Path,
    dataset_ids: tuple[str, ...] = (),
) -> tuple[Path, ...]:
    """Return every implementation file that can define the audited run."""

    paths: tuple[Path, ...] = (
        Path(__file__).resolve(),
        project_root / "scripts/feature_adapter/cache_clip_manifest_features.py",
        project_root / "scripts/feature_adapter/verify_clip_runtime.py",
        project_root / "src/ttvr/data/bird_manifest.py",
        project_root / "src/ttvr/data/birdnet.py",
        project_root / "src/ttvr/data/cub.py",
        project_root / "src/ttvr/data/cub_taxonomy.py",
        project_root / "src/ttvr/data/inat2021.py",
        project_root / "src/ttvr/experiments/artifacts.py",
        project_root / "src/ttvr/methods/feature_adapter/model.py",
        project_root / "src/ttvr/methods/feature_adapter/tasks.py",
        project_root / "src/ttvr/methods/feature_adapter/training.py",
        project_root / "src/ttvr/metrics.py",
        project_root / "src/ttvr/models/cached.py",
        project_root / "src/ttvr/models/clip.py",
    )
    if any(dataset_id.startswith("big-bird-") for dataset_id in dataset_ids):
        paths += (
            project_root / "src/ttvr/data/bird_crops.py",
            project_root / "src/ttvr/data/big_bird.py",
        )
    if any(dataset_id.startswith("visual-wetlandbirds-") for dataset_id in dataset_ids):
        paths += (
            project_root / "src/ttvr/data/bird_crops.py",
            project_root / "src/ttvr/data/visual_wetlandbirds.py",
        )
    if any(dataset_id.startswith("usgs-aerial-avian-") for dataset_id in dataset_ids):
        paths += (
            project_root / "src/ttvr/data/bird_crops.py",
            project_root / "src/ttvr/data/bird_source_archive.py",
            project_root / "src/ttvr/data/birdnet_lock.py",
            project_root / "src/ttvr/data/usgs_aerial_avian.py",
        )
    if any(dataset_id.startswith("nm-uas-waterfowl-") for dataset_id in dataset_ids):
        paths += (
            project_root / "src/ttvr/data/bird_crops.py",
            project_root / "src/ttvr/data/bird_source_archive.py",
            project_root / "src/ttvr/data/birdnet_lock.py",
            project_root / "src/ttvr/data/nm_uas_waterfowl.py",
        )
    return tuple(sorted(set(paths), key=lambda value: str(value)))


def _source_digest(
    project_root: Path,
    dataset_ids: tuple[str, ...] = (),
) -> str:
    paths = _source_digest_paths(project_root, dataset_ids)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--cub-data-root", type=Path, default=project_root / "data")
    parser.add_argument(
        "--cub-class-names",
        type=Path,
        default=project_root / "data/fudd_official/cub_class_names.json",
    )
    parser.add_argument("--birdnet-csv", type=Path, required=True)
    parser.add_argument("--cub-feature-cache", type=Path, required=True)
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=project_root / ".cache/fudd_clip_cub/models",
    )
    parser.add_argument(
        "--text-cache",
        type=Path,
        default=project_root / ".cache/feature_adapter_clip_multi_bird/text_features.pt",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help=("Immutable run directory root. Defaults to experiments/<experiment-id>/runs."),
    )
    parser.add_argument(
        "--experiment-id",
        default="06_feature_adapter_clip_multi_bird",
    )
    parser.add_argument(
        "--protocol",
        default="external-only-strict-cub-v1",
    )
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--patience-intervals", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--identity-weight", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kind", default="external-strict")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    identifier_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if (
        not args.experiment_id
        or not args.experiment_id[0].isalnum()
        or ".." in args.experiment_id
        or any(character not in identifier_characters for character in args.experiment_id)
    ):
        raise ValueError("experiment-id must be a filesystem-safe identifier")
    if not isinstance(args.protocol, str) or not args.protocol.strip():
        raise ValueError("protocol must be non-empty")
    runs_root = args.runs_root or (project_root / "experiments" / args.experiment_id / "runs")
    device = torch.device(args.device)
    class_names = read_cub_class_names(args.cub_class_names)
    crosswalk = build_cub_birdnet_crosswalk(args.cub_class_names, args.birdnet_csv)
    excluded_taxa = {f"birdnet:{value}" for value in crosswalk.excluded_birdnet_ids(range(200))}
    specs, validation_taxon_fraction = _source_specs(args.source_config, project_root)
    duplicate_config = _duplicate_audit_config(args.source_config)
    disabled_sources = _disabled_source_rows(args.source_config)
    train_config = AdapterTrainConfig(
        steps=args.steps,
        validation_interval=args.validation_interval,
        patience_intervals=args.patience_intervals,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        identity_weight=args.identity_weight,
        logit_scale=100.0,
        seed=args.seed,
    )
    backend = CLIPBackend(
        device=device,
        precision="fp32",
        model_cache_dir=args.model_cache_dir,
        text_cache_path=args.text_cache,
    )
    resolved_config = {
        "experiment": args.experiment_id,
        "protocol": args.protocol,
        "model": "OpenAI CLIP ViT-L/14@336px fp32",
        "model_runtime": _model_runtime_config(backend),
        "method": {
            "feature_dim": 768,
            "hidden_dim": args.hidden_dim,
            "architecture": "normalize(x + W_up(GELU(W_down(x))))",
        },
        "training": train_config.to_dict(),
        "source_config": str(args.source_config.resolve()),
        "source_config_sha256": sha256_file(args.source_config),
        "validation_taxon_fraction": validation_taxon_fraction,
        "duplicate_audit": duplicate_config.to_dict(),
        "disabled_source_placeholders": disabled_sources,
        "sources": [
            {
                "dataset_id": spec.dataset_id,
                "source_metadata": str((spec.samples.parent / "source.json").resolve()),
                "source_metadata_sha256": sha256_file(spec.samples.parent / "source.json"),
                "samples": str(spec.samples),
                "samples_sha256": sha256_file(spec.samples),
                "taxa": str(spec.taxa),
                "taxa_sha256": sha256_file(spec.taxa),
                "train_cache": str(spec.train_cache),
                "train_cache_sha256": sha256_file(spec.train_cache),
                "validation_cache": (
                    None if spec.validation_cache is None else str(spec.validation_cache)
                ),
                "validation_cache_sha256": (
                    None if spec.validation_cache is None else sha256_file(spec.validation_cache)
                ),
            }
            for spec in specs
        ],
        "cub_feature_cache": str(args.cub_feature_cache.resolve()),
        "cub_feature_cache_sha256": sha256_file(args.cub_feature_cache),
        "cub_crosswalk_digest": crosswalk.digest,
        "source_code_sha256": _source_digest(
            project_root,
            tuple(spec.dataset_id for spec in specs),
        ),
        "seed": args.seed,
    }
    run_dir = create_run_directory(runs_root, resolved_config, kind=args.kind)
    write_json(run_dir / "config.json", resolved_config)
    write_json(run_dir / "environment.json", _environment())
    write_json(
        run_dir / "run_started.json",
        {"state": "running", "run_directory": str(run_dir)},
    )
    write_json(
        run_dir / "cub_taxonomy_crosswalk.json",
        {
            "protocol": crosswalk.protocol,
            "birdnet_csv_sha256": crosswalk.birdnet_csv_sha256,
            "digest": crosswalk.digest,
            "status_counts": crosswalk.status_counts,
            "entries": [asdict(entry) for entry in crosswalk.entries],
        },
    )

    sources = _load_sources(specs, backend)
    duplicate_audit, duplicate_excluded_sample_ids = _audit_cross_source_duplicates(
        sources,
        duplicate_config,
    )
    write_json(run_dir / "cross_source_duplicate_audit.json", duplicate_audit)
    (
        train_tasks,
        validation_tasks,
        refit_tasks,
        source_prototypes,
        exclusions,
        source_prompts,
        validation_taxa,
        source_task_membership,
        source_partition_audit,
    ) = _prepare_source_tasks(
        sources,
        backend,
        excluded_taxa=excluded_taxa,
        validation_taxon_fraction=validation_taxon_fraction,
        duplicate_excluded_sample_ids=duplicate_excluded_sample_ids,
    )
    backend.save_text_cache()
    write_jsonl(run_dir / "exclusion_manifest.jsonl", exclusions)
    write_jsonl(run_dir / "source_prompt_manifest.jsonl", source_prompts)
    write_jsonl(
        run_dir / "source_validation_taxa.jsonl",
        ({"taxon_id": taxon_id} for taxon_id in validation_taxa),
    )
    write_jsonl(run_dir / "source_task_membership.jsonl", source_task_membership)
    write_jsonl(run_dir / "source_partition_audit.jsonl", source_partition_audit)
    atomic_torch_save(
        {
            "taxon_ids": tuple(sorted(source_prototypes)),
            "features": torch.stack(
                [source_prototypes[taxon_id] for taxon_id in sorted(source_prototypes)]
            ),
        },
        run_dir / "source_text_prototypes.pt",
    )

    adapter = ResidualFeatureAdapter(768, args.hidden_dim)
    fit = fit_feature_adapter(
        adapter,
        train_tasks,
        validation_tasks,
        train_config,
        device=device,
    )
    atomic_torch_save(
        {
            "state_dict": fit.state_dict,
            "feature_dim": 768,
            "hidden_dim": args.hidden_dim,
            "best_step": fit.best_step,
            "training": train_config.to_dict(),
        },
        run_dir / "selection_adapter.pt",
    )
    adapter = ResidualFeatureAdapter(768, args.hidden_dim)
    refit = refit_feature_adapter(
        adapter,
        refit_tasks,
        train_config,
        steps=fit.best_step,
        device=device,
    )
    adapter.load_state_dict(refit.state_dict)
    adapter.to(device)
    atomic_torch_save(
        {
            "state_dict": refit.state_dict,
            "feature_dim": 768,
            "hidden_dim": args.hidden_dim,
            "refit_steps": refit.steps,
            "training": train_config.to_dict(),
        },
        run_dir / "adapter.pt",
    )
    write_json(
        run_dir / "source_validation.json",
        {
            "best_step": fit.best_step,
            "best": fit.best_validation.to_dict(),
            "history": [snapshot.to_dict() for snapshot in fit.history],
            "train_task_draws": fit.train_task_draws,
            "train_tasks": [
                {"name": task.name, "samples": task.size, "classes": task.class_count}
                for task in train_tasks
            ],
            "validation_tasks": [
                {"name": task.name, "samples": task.size, "classes": task.class_count}
                for task in validation_tasks
            ],
            "refit_tasks": [
                {"name": task.name, "samples": task.size, "classes": task.class_count}
                for task in refit_tasks
            ],
            "refit_steps": refit.steps,
            "refit_task_draws": refit.train_task_draws,
        },
    )

    # Target class names are encoded only after fitting and locking the adapter.
    target_prompts = [f"a photo of a {class_name}." for class_name in class_names]
    target_text = backend.encode_texts(target_prompts).detach().cpu().float()
    backend.save_text_cache()
    write_jsonl(
        run_dir / "target_prompt_manifest.jsonl",
        (
            {"class_id": class_id, "class_name": class_name, "prompt": prompt}
            for class_id, (class_name, prompt) in enumerate(
                zip(class_names, target_prompts, strict=True)
            )
        ),
    )
    atomic_torch_save(
        {"class_names": class_names, "prompts": target_prompts, "features": target_text},
        run_dir / "target_text_prototypes.pt",
    )

    cub_dataset = prepare_cub(
        args.cub_data_root,
        split="test",
        download=False,
        verify_images=False,
    )
    cub_payload = torch_load_tensors(args.cub_feature_cache)
    cub_features, cub_labels = _validated_cub_target_cache(
        cub_payload,
        cub_dataset,
        backend,
    )
    baseline_predictions, adapted_predictions = _predict_cub(
        adapter,
        cub_features,
        target_text,
        device=device,
    )
    baseline_metrics = compute_topk_accuracy(baseline_predictions, cub_labels)
    adapted_metrics = compute_topk_accuracy(adapted_predictions, cub_labels)
    if baseline_metrics.top1_correct != 3_671:
        raise RuntimeError(
            "Historical CLIP baseline parity failed: "
            f"expected 3671, found {baseline_metrics.top1_correct}"
        )
    baseline_correct = baseline_predictions[:, 0].eq(cub_labels)
    adapted_correct = adapted_predictions[:, 0].eq(cub_labels)
    cluster_bootstrap = _species_cluster_bootstrap(
        baseline_correct,
        adapted_correct,
        cub_labels,
        reps=10_000,
        seed=args.seed,
    )
    comparison = {
        "transfers": compute_transfer_counts(
            baseline_predictions, adapted_predictions, cub_labels
        ).to_dict(),
        "mcnemar": exact_mcnemar_test(baseline_correct, adapted_correct).to_dict(),
        "paired_image_bootstrap": paired_bootstrap_accuracy_gain(
            baseline_correct,
            adapted_correct,
            reps=10_000,
            seed=args.seed,
        ).to_dict(),
        "species_cluster_bootstrap": cluster_bootstrap,
    }
    result = {
        "baseline": baseline_metrics.to_dict(),
        "adapted": adapted_metrics.to_dict(),
        "gain_top1_percentage_points": adapted_metrics.top1 - baseline_metrics.top1,
        "comparison": comparison,
        "strict_transfer_criterion_passed": (
            float(cluster_bootstrap["ci95_low_percentage_points"]) > 0.0
        ),
        "excluded_target_taxa": len(excluded_taxa),
        "source_dataset_count": len(sources),
        "disabled_source_placeholder_count": len(disabled_sources),
        "source_validation_task_count": len(validation_tasks),
        "source_validation_omitted_count": sum(
            row["unseen_validation"]["status"] != "included" for row in source_partition_audit
        ),
        "exact_duplicate_images_excluded": duplicate_audit["summary"][
            "exact_duplicate_rows_dropped_total"
        ],
        "source_selection_training_images": sum(task.size for task in train_tasks),
        "source_refit_images": sum(task.size for task in refit_tasks),
        "source_validation_images": sum(task.size for task in validation_tasks),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "best_step": fit.best_step,
    }
    write_json(run_dir / "result.json", result)
    prediction_rows = []
    for index, sample in enumerate(cub_dataset.samples):
        label = int(cub_labels[index].item())
        baseline_top5 = baseline_predictions[index].tolist()
        adapted_top5 = adapted_predictions[index].tolist()
        prediction_rows.append(
            {
                "index": index,
                "image_id": sample.image_id,
                "relative_path": sample.relative_path.as_posix(),
                "label": label,
                "label_name": class_names[label],
                "baseline_top5": baseline_top5,
                "baseline_top5_names": [class_names[value] for value in baseline_top5],
                "adapted_top5": adapted_top5,
                "adapted_top5_names": [class_names[value] for value in adapted_top5],
                "baseline_correct": bool(baseline_correct[index].item()),
                "adapted_correct": bool(adapted_correct[index].item()),
            }
        )
    write_jsonl(run_dir / "cub_predictions.jsonl", prediction_rows)

    source_manifest_root = run_dir / "source_manifests"
    for source in sources:
        destination = source_manifest_root / source.spec.dataset_id
        destination.mkdir(parents=True)
        shutil.copy2(source.spec.samples.parent / "source.json", destination / "source.json")
        shutil.copy2(source.spec.samples, destination / "samples.jsonl")
        shutil.copy2(source.spec.taxa, destination / "taxa.jsonl")
    write_json(
        run_dir / "run_complete.json",
        {"state": "complete", "result": result},
    )
    write_checksums(run_dir)
    print(json.dumps({"run_directory": str(run_dir), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
