"""Readable construction of dataset-local feature tasks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from .training import FeatureTask


@dataclass(frozen=True, slots=True)
class PreparedFeatureTask:
    """A training task plus the identities retained by its local labels."""

    task: FeatureTask
    taxon_ids: tuple[str, ...]
    source_indices: Tensor


def stable_taxon_partition(
    taxon_ids: tuple[str, ...],
    *,
    validation_fraction: float,
    salt: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split species deterministically without looking at any image."""

    if len(taxon_ids) < 2 or len(set(taxon_ids)) != len(taxon_ids):
        raise ValueError("taxon_ids must contain at least two unique values")
    if not 0.0 < validation_fraction < 1.0 or not salt:
        raise ValueError("validation_fraction and salt are invalid")
    validation_count = max(1, min(len(taxon_ids) - 1, round(len(taxon_ids) * validation_fraction)))
    ranked = sorted(
        taxon_ids,
        key=lambda taxon_id: hashlib.sha256(f"{salt}\0{taxon_id}".encode()).hexdigest(),
    )
    validation = tuple(sorted(ranked[:validation_count]))
    training = tuple(sorted(ranked[validation_count:]))
    return training, validation


def build_feature_task(
    name: str,
    features: Tensor,
    labels: Tensor,
    local_taxon_ids: tuple[str, ...],
    prototype_by_taxon: dict[str, Tensor],
    *,
    included_taxon_ids: set[str] | None = None,
    excluded_taxon_ids: set[str] | None = None,
    excluded_source_indices: set[int] | None = None,
) -> PreparedFeatureTask:
    """Filter rows/taxa and remap source-local labels to a compact vocabulary.

    ``excluded_source_indices`` is intentionally an index-space operation.  It
    lets an experiment remove exact cross-source duplicate images without
    mutating a canonical source manifest or its frozen feature cache.
    """

    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must be aligned")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long")
    if not local_taxon_ids or len(set(local_taxon_ids)) != len(local_taxon_ids):
        raise ValueError("local_taxon_ids must be non-empty and unique")
    labels_cpu = labels.detach().cpu()
    if labels_cpu.min().item() < 0 or labels_cpu.max().item() >= len(local_taxon_ids):
        raise ValueError("labels do not index local_taxon_ids")
    included = set(local_taxon_ids) if included_taxon_ids is None else set(included_taxon_ids)
    excluded = set() if excluded_taxon_ids is None else set(excluded_taxon_ids)
    excluded_indices = (
        set() if excluded_source_indices is None else set(excluded_source_indices)
    )
    if any(
        type(index) is not int or index < 0 or index >= labels_cpu.numel()
        for index in excluded_indices
    ):
        raise ValueError("excluded_source_indices must index the aligned source rows")
    eligible_rows = torch.ones(labels_cpu.numel(), dtype=torch.bool)
    if excluded_indices:
        eligible_rows[torch.tensor(sorted(excluded_indices), dtype=torch.long)] = False
    present_source_labels = set(labels_cpu[eligible_rows].tolist())
    selected_taxa = tuple(
        taxon_id
        for source_label, taxon_id in enumerate(local_taxon_ids)
        if source_label in present_source_labels
        and taxon_id in included
        and taxon_id not in excluded
    )
    if len(selected_taxa) < 2:
        raise ValueError("taxon filters must retain at least two classes")
    missing_prototypes = set(selected_taxa) - set(prototype_by_taxon)
    if missing_prototypes:
        raise ValueError(f"missing text prototypes for taxa: {sorted(missing_prototypes)[:5]}")

    selected_set = set(selected_taxa)
    remap = torch.full((len(local_taxon_ids),), -1, dtype=torch.long)
    compact_taxa: list[str] = []
    for source_label, taxon_id in enumerate(local_taxon_ids):
        if taxon_id in selected_set:
            remap[source_label] = len(compact_taxa)
            compact_taxa.append(taxon_id)
    compact_labels = remap.index_select(0, labels_cpu)
    mask = compact_labels.ge(0) & eligible_rows
    source_indices = torch.where(mask)[0]
    filtered_features = features.detach().cpu().index_select(0, source_indices).contiguous()
    filtered_labels = compact_labels.index_select(0, source_indices).contiguous()
    text_prototypes = torch.stack(
        [prototype_by_taxon[taxon_id].detach().cpu() for taxon_id in compact_taxa]
    ).contiguous()
    task = FeatureTask(
        name=name,
        features=filtered_features,
        labels=filtered_labels,
        text_prototypes=text_prototypes,
    )
    return PreparedFeatureTask(
        task=task,
        taxon_ids=tuple(compact_taxa),
        source_indices=source_indices,
    )
