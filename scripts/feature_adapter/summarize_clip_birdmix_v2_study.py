#!/usr/bin/env python3
"""Fail-closed audit and three-seed summary for the CLIP BirdMix-v2 study."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ttvr.data.cub import CUB_CLASS_COUNT, CUB_TEST_COUNT, CubSample, prepare_cub
from ttvr.data.cub_taxonomy import (
    CubBirdNetCrosswalk,
    build_cub_birdnet_crosswalk,
    read_cub_class_names,
)
from ttvr.metrics import exact_mcnemar_test
from ttvr.models.cached import torch_load_tensors
from ttvr.models.clip import (
    DEFAULT_CLIP_MODEL,
    OPENAI_CLIP_COMMIT,
    OPENAI_CLIP_REPOSITORY,
    VIT_L14_336_CHECKPOINT_FILENAME,
    VIT_L14_336_CHECKPOINT_SHA256,
)

if __package__:
    from scripts.feature_adapter.run_clip_birdmix_v2_study import (
        SourceTrial,
        StudyConfig,
        StudyConfigError,
        load_study_config,
    )
    from scripts.feature_adapter.summarize_clip_birdmix_study import (
        _REQUIRED_RUN_FILES,
        StudySummaryError,
        _read_json_object,
        _resolved_path,
        _sha256_file,
        _summarize_result,
        _validate_run_directory_name,
        _verify_checksums,
    )
else:
    from run_clip_birdmix_v2_study import (  # type: ignore[no-redef]
        SourceTrial,
        StudyConfig,
        StudyConfigError,
        load_study_config,
    )
    from summarize_clip_birdmix_study import (  # type: ignore[no-redef]
        _REQUIRED_RUN_FILES,
        StudySummaryError,
        _read_json_object,
        _resolved_path,
        _sha256_file,
        _summarize_result,
        _validate_run_directory_name,
        _verify_checksums,
    )

_EXPECTED_MODEL = "OpenAI CLIP ViT-L/14@336px fp32"
_EXPECTED_FEATURE_DIM = 768
_EXPECTED_ARCHITECTURE = "normalize(x + W_up(GELU(W_down(x))))"
_EXPECTED_CLIP_DISTRIBUTION = "clip"
_EXPECTED_CLIP_VERSION = "1.0"
_EXPECTED_CHECKPOINT_SIZE_BYTES = 934_088_680
_EXPECTED_CUB_BASELINE_TOP1_CORRECT = 3_671
_MODEL_RUNTIME_KEYS = {
    "cache_identity",
    "clip_distribution",
    "clip_version",
    "clip_repository_url",
    "clip_commit",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
}
_MAX_PREDICTION_LINE_BYTES = 64 * 1024
_PREDICTION_KEYS = {
    "index",
    "image_id",
    "relative_path",
    "label",
    "label_name",
    "baseline_top5",
    "baseline_top5_names",
    "adapted_top5",
    "adapted_top5_names",
    "baseline_correct",
    "adapted_correct",
}
_THREE_SEED_T_CRITICAL_95 = 4.302652729911275
_CI_METHOD = (
    "two-sided Student t interval over the three preregistered seeds; "
    "df=2; t(0.975,2)=4.302652729911275; endpoints are not clipped"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REQUIRED_V2_RUN_FILES = (
    *_REQUIRED_RUN_FILES,
    "cub_taxonomy_crosswalk.json",
    "source_text_prototypes.pt",
    "target_text_prototypes.pt",
)
_SOURCE_SNAPSHOT_KEYS = {
    "dataset_id",
    "source_metadata",
    "source_metadata_sha256",
    "samples",
    "samples_sha256",
    "taxa",
    "taxa_sha256",
    "train_cache",
    "train_cache_sha256",
    "validation_cache",
    "validation_cache_sha256",
}


def _expected_training(config: StudyConfig, *, seed: int) -> dict[str, Any]:
    common = config.common
    return {
        "steps": common.steps,
        "validation_interval": common.validation_interval,
        "patience_intervals": common.patience_intervals,
        "batch_size": common.batch_size,
        "learning_rate": common.learning_rate,
        "weight_decay": common.weight_decay,
        "identity_weight": common.identity_weight,
        "logit_scale": 100.0,
        "seed": seed,
    }


def _source_protocol_values(trial: SourceTrial) -> tuple[float, dict[str, Any]]:
    value = _read_json_object(trial.source_config, context="v2 source config")
    fraction = value.get("validation_taxon_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise StudySummaryError("source validation_taxon_fraction must be numeric")
    duplicate = value.get("duplicate_audit")
    if not isinstance(duplicate, dict):
        raise StudySummaryError("source duplicate_audit must be an object")
    return float(fraction), duplicate


def _resolve_one_source_path(
    project_root: Path,
    value: object,
    *,
    context: str,
) -> Path:
    path = _resolved_path(value, context=context)
    raw = Path(str(value)).expanduser()
    if not raw.is_absolute():
        path = (project_root / raw).resolve()
    if any(character in str(path) for character in "*?["):
        matches = sorted(path.parent.glob(path.name))
        if len(matches) != 1:
            raise StudySummaryError(f"{context} must resolve exactly once: {path} -> {matches}")
        path = matches[0].resolve()
    return path


def _require_sha256(value: object, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise StudySummaryError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _string_sequence(value: object, *, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise StudySummaryError(f"{context} must be a non-empty string sequence")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise StudySummaryError(f"{context} contains an invalid string")
    return tuple(value)


def _prototype_content_digest(
    path: Path,
    *,
    kind: str,
    locked_class_names: tuple[str, ...] | None = None,
) -> str:
    """Hash validated prototype metadata and tensor content, not torch.save bytes."""

    try:
        payload = torch_load_tensors(path)
    except Exception as error:
        raise StudySummaryError(f"Cannot safely load {kind} prototypes: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise StudySummaryError(f"{kind} prototype payload must be an object: {path}")
    if kind == "source":
        expected_keys = {"taxon_ids", "features"}
        if set(payload) != expected_keys:
            raise StudySummaryError(f"source prototype payload has the wrong keys: {path}")
        taxon_ids = _string_sequence(payload.get("taxon_ids"), context="source taxon_ids")
        if taxon_ids != tuple(sorted(set(taxon_ids))):
            raise StudySummaryError("source prototype taxon_ids must be sorted and unique")
        metadata: dict[str, Any] = {"kind": kind, "taxon_ids": taxon_ids}
        row_count = len(taxon_ids)
    elif kind == "target":
        expected_keys = {"class_names", "prompts", "features"}
        if set(payload) != expected_keys:
            raise StudySummaryError(f"target prototype payload has the wrong keys: {path}")
        class_names = _string_sequence(payload.get("class_names"), context="target class_names")
        prompts = _string_sequence(payload.get("prompts"), context="target prompts")
        if len(class_names) != 200 or len(set(class_names)) != 200:
            raise StudySummaryError("target prototypes must contain 200 unique class names")
        if locked_class_names is not None and class_names != locked_class_names:
            raise StudySummaryError(
                "target prototype class names differ from the locked CUB class names"
            )
        if prompts != tuple(f"a photo of a {name}." for name in class_names):
            raise StudySummaryError("target prototype prompts do not match the locked template")
        metadata = {
            "kind": kind,
            "class_names": class_names,
            "prompts": prompts,
        }
        row_count = len(class_names)
    else:
        raise StudySummaryError(f"Unknown prototype kind: {kind}")

    features = payload.get("features")
    if not (
        isinstance(features, torch.Tensor)
        and features.layout == torch.strided
        and features.dtype == torch.float32
        and features.shape == (row_count, _EXPECTED_FEATURE_DIM)
    ):
        raise StudySummaryError(
            f"{kind} prototype features must have shape "
            f"({row_count}, {_EXPECTED_FEATURE_DIM}) and dtype torch.float32"
        )
    canonical = features.detach().cpu().contiguous()
    if not bool(torch.isfinite(canonical).all()):
        raise StudySummaryError(f"{kind} prototype features contain non-finite values")
    norms = torch.linalg.vector_norm(canonical, dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5):
        raise StudySummaryError(f"{kind} prototype features are not unit-normalised")

    digest = hashlib.sha256(b"ttvr-prototype-content-v1\0")
    digest.update(
        json.dumps(
            metadata,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(b"\0dtype=float32\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(canonical.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare decoded JSON values without bool/int or int/float coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(  # type: ignore[arg-type]
            _strict_json_equal(actual[key], value)  # type: ignore[index]
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(  # type: ignore[arg-type]
            _strict_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)  # type: ignore[arg-type]
        )
    return actual == expected


def _canonical_row_digest(rows: list[tuple[object, ...]], *, domain: bytes) -> str:
    digest = hashlib.sha256(domain + b"\0")
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _decode_prediction_object(raw: bytes, *, path: Path, line_number: int) -> dict[str, Any]:
    if len(raw) > _MAX_PREDICTION_LINE_BYTES:
        raise StudySummaryError(
            f"CUB prediction line exceeds {_MAX_PREDICTION_LINE_BYTES} bytes: {path}:{line_number}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StudySummaryError(
            f"CUB prediction line is not UTF-8: {path}:{line_number}"
        ) from error
    if not text.strip():
        raise StudySummaryError(f"Blank CUB prediction line: {path}:{line_number}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as error:
        raise StudySummaryError(
            f"Malformed CUB prediction JSON: {path}:{line_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise StudySummaryError(f"CUB prediction row must be an object: {path}:{line_number}")
    return value


def _validated_top5(
    row: dict[str, Any],
    *,
    prefix: str,
    class_names: tuple[str, ...],
    context: str,
) -> tuple[int, ...]:
    raw_ids = row.get(f"{prefix}_top5")
    if (
        not isinstance(raw_ids, list)
        or len(raw_ids) != 5
        or any(
            type(class_id) is not int or not 0 <= class_id < CUB_CLASS_COUNT for class_id in raw_ids
        )
        or len(set(raw_ids)) != 5
    ):
        raise StudySummaryError(f"{context}.{prefix}_top5 must contain five unique CUB ids")
    expected_names = [class_names[class_id] for class_id in raw_ids]
    if not _strict_json_equal(row.get(f"{prefix}_top5_names"), expected_names):
        raise StudySummaryError(
            f"{context}.{prefix}_top5_names do not match the locked CUB class names"
        )
    return tuple(raw_ids)


def _species_cluster_bootstrap_from_predictions(
    baseline_correct: torch.Tensor,
    adapted_correct: torch.Tensor,
    labels: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float | int]:
    class_ids = tuple(sorted(set(labels.tolist())))
    if class_ids != tuple(range(CUB_CLASS_COUNT)):
        raise StudySummaryError("CUB predictions do not cover every locked target class")
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
    draws = torch.randint(
        len(class_ids),
        (10_000, len(class_ids)),
        generator=generator,
    )
    gains = per_class_delta.index_select(0, draws.reshape(-1)).reshape(10_000, -1).mean(dim=1)
    return {
        "clusters": len(class_ids),
        "reps": 10_000,
        "seed": seed,
        "mean_gain_percentage_points": 100.0 * float(per_class_delta.mean().item()),
        "ci95_low_percentage_points": 100.0 * float(torch.quantile(gains, 0.025).item()),
        "ci95_high_percentage_points": 100.0 * float(torch.quantile(gains, 0.975).item()),
    }


def _validate_prediction_artifact(
    path: Path,
    result: dict[str, Any],
    *,
    class_names: tuple[str, ...],
    canonical_samples: tuple[CubSample, ...],
    canonical_targets: tuple[int, ...],
    seed: int,
) -> dict[str, str]:
    """Read every CUB prediction safely and derive all reported outcome counts."""

    if len(canonical_samples) != CUB_TEST_COUNT or len(canonical_targets) != CUB_TEST_COUNT:
        raise StudySummaryError(
            f"Canonical CUB test dataset must contain exactly {CUB_TEST_COUNT} samples and targets"
        )

    identities: list[tuple[object, ...]] = []
    baseline_rows: list[tuple[object, ...]] = []
    labels: list[int] = []
    baseline_predictions: list[tuple[int, ...]] = []
    adapted_predictions: list[tuple[int, ...]] = []
    baseline_flags: list[bool] = []
    adapted_flags: list[bool] = []
    image_ids: set[int] = set()
    relative_paths: set[str] = set()
    with path.open("rb") as handle:
        for line_number in range(1, CUB_TEST_COUNT + 2):
            raw = handle.readline(_MAX_PREDICTION_LINE_BYTES + 1)
            if not raw:
                break
            row = _decode_prediction_object(raw, path=path, line_number=line_number)
            context = f"CUB prediction row {line_number}"
            if set(row) != _PREDICTION_KEYS:
                raise StudySummaryError(f"{context} has the wrong fields")
            index = row.get("index")
            if type(index) is not int or index != line_number - 1:
                raise StudySummaryError(f"{context}.index is not the exact dataset index")
            if index >= CUB_TEST_COUNT:
                raise StudySummaryError(
                    f"CUB prediction artifact contains more than {CUB_TEST_COUNT} rows: {path}"
                )
            canonical_sample = canonical_samples[index]
            canonical_label = canonical_targets[index]
            if canonical_sample.target != canonical_label:
                raise StudySummaryError(
                    f"Canonical CUB sample and target disagree at dataset index {index}"
                )
            image_id = row.get("image_id")
            if (
                type(image_id) is not int
                or image_id != canonical_sample.image_id
                or image_id in image_ids
            ):
                raise StudySummaryError(
                    f"{context}.image_id does not match the canonical CUB test sample"
                )
            image_ids.add(image_id)
            relative_path = row.get("relative_path")
            if (
                not isinstance(relative_path, str)
                or relative_path != canonical_sample.relative_path.as_posix()
                or relative_path in relative_paths
            ):
                raise StudySummaryError(
                    f"{context}.relative_path does not match the canonical CUB test sample"
                )
            relative_paths.add(relative_path)
            label = row.get("label")
            if type(label) is not int or label != canonical_label:
                raise StudySummaryError(
                    f"{context}.label does not match the canonical CUB test target"
                )
            if row.get("label_name") != class_names[label]:
                raise StudySummaryError(f"{context}.label_name disagrees with the locked class")
            baseline_top5 = _validated_top5(
                row,
                prefix="baseline",
                class_names=class_names,
                context=context,
            )
            adapted_top5 = _validated_top5(
                row,
                prefix="adapted",
                class_names=class_names,
                context=context,
            )
            baseline_correct = row.get("baseline_correct")
            adapted_correct = row.get("adapted_correct")
            if type(baseline_correct) is not bool or baseline_correct != (
                baseline_top5[0] == label
            ):
                raise StudySummaryError(f"{context}.baseline_correct is inconsistent")
            if type(adapted_correct) is not bool or adapted_correct != (adapted_top5[0] == label):
                raise StudySummaryError(f"{context}.adapted_correct is inconsistent")
            identities.append((index, image_id, relative_path, label, class_names[label]))
            baseline_rows.append((baseline_top5, tuple(row["baseline_top5_names"])))
            labels.append(label)
            baseline_predictions.append(baseline_top5)
            adapted_predictions.append(adapted_top5)
            baseline_flags.append(baseline_correct)
            adapted_flags.append(adapted_correct)

    if len(labels) != CUB_TEST_COUNT:
        raise StudySummaryError(
            f"CUB prediction artifact must contain exactly {CUB_TEST_COUNT} rows; "
            f"found {len(labels)}: {path}"
        )

    baseline_top1 = sum(baseline_flags)
    adapted_top1 = sum(adapted_flags)
    if baseline_top1 != _EXPECTED_CUB_BASELINE_TOP1_CORRECT:
        raise StudySummaryError(
            "Frozen CLIP historical CUB baseline Top-1 count must be "
            f"{_EXPECTED_CUB_BASELINE_TOP1_CORRECT}; found {baseline_top1}"
        )
    baseline_top5 = sum(
        label in predicted for label, predicted in zip(labels, baseline_predictions, strict=True)
    )
    adapted_top5 = sum(
        label in predicted for label, predicted in zip(labels, adapted_predictions, strict=True)
    )

    def metric(top1_correct: int, top5_correct: int) -> dict[str, int | float]:
        return {
            "total": CUB_TEST_COUNT,
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "top1": 100.0 * top1_correct / CUB_TEST_COUNT,
            "top5": 100.0 * top5_correct / CUB_TEST_COUNT,
        }

    expected_baseline = metric(baseline_top1, baseline_top5)
    expected_adapted = metric(adapted_top1, adapted_top5)
    if not _strict_json_equal(result.get("baseline"), expected_baseline):
        raise StudySummaryError("result.baseline does not exactly match cub_predictions.jsonl")
    if not _strict_json_equal(result.get("adapted"), expected_adapted):
        raise StudySummaryError("result.adapted does not exactly match cub_predictions.jsonl")

    both_correct = sum(
        left and right for left, right in zip(baseline_flags, adapted_flags, strict=True)
    )
    recovered = sum(
        not left and right for left, right in zip(baseline_flags, adapted_flags, strict=True)
    )
    degraded = sum(
        left and not right for left, right in zip(baseline_flags, adapted_flags, strict=True)
    )
    both_wrong = CUB_TEST_COUNT - both_correct - recovered - degraded

    def percentage(count: int) -> float:
        return 100.0 * count / CUB_TEST_COUNT

    expected_transfers = {
        "total": CUB_TEST_COUNT,
        "both_correct": both_correct,
        "recovered": recovered,
        "degraded": degraded,
        "both_wrong": both_wrong,
        "both_correct_percent": percentage(both_correct),
        "recovered_percent": percentage(recovered),
        "degraded_percent": percentage(degraded),
        "both_wrong_percent": percentage(both_wrong),
    }
    comparison = result.get("comparison")
    if not isinstance(comparison, dict):
        raise StudySummaryError("result.comparison must be an object")
    if not _strict_json_equal(comparison.get("transfers"), expected_transfers):
        raise StudySummaryError(
            "result.comparison.transfers does not exactly match cub_predictions.jsonl"
        )

    baseline_tensor = torch.tensor(baseline_flags, dtype=torch.bool)
    adapted_tensor = torch.tensor(adapted_flags, dtype=torch.bool)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    expected_mcnemar = exact_mcnemar_test(baseline_tensor, adapted_tensor).to_dict()
    if not _strict_json_equal(comparison.get("mcnemar"), expected_mcnemar):
        raise StudySummaryError("result.comparison.mcnemar does not match CUB predictions")
    expected_cluster = _species_cluster_bootstrap_from_predictions(
        baseline_tensor,
        adapted_tensor,
        labels_tensor,
        seed=seed,
    )
    if not _strict_json_equal(comparison.get("species_cluster_bootstrap"), expected_cluster):
        raise StudySummaryError(
            "result.comparison.species_cluster_bootstrap does not match CUB predictions"
        )
    expected_strict_pass = float(expected_cluster["ci95_low_percentage_points"]) > 0.0
    if result.get("strict_transfer_criterion_passed") is not expected_strict_pass:
        raise StudySummaryError(
            "result.strict_transfer_criterion_passed disagrees with the species-cluster CI"
        )
    return {
        "cub_identity_and_labels_digest": _canonical_row_digest(
            identities,
            domain=b"ttvr-cub-prediction-identity-v1",
        ),
        "baseline_top5_digest": _canonical_row_digest(
            baseline_rows,
            domain=b"ttvr-cub-baseline-top5-v1",
        ),
    }


def _expected_crosswalk_artifact(crosswalk: CubBirdNetCrosswalk) -> dict[str, Any]:
    value = {
        "protocol": crosswalk.protocol,
        "birdnet_csv_sha256": crosswalk.birdnet_csv_sha256,
        "digest": crosswalk.digest,
        "status_counts": crosswalk.status_counts,
        "entries": [asdict(entry) for entry in crosswalk.entries],
    }
    return json.loads(json.dumps(value, ensure_ascii=False))


def _validate_source_snapshots(
    value: dict[str, Any],
    trial: SourceTrial,
    *,
    project_root: Path,
    context: str,
) -> None:
    source_config = _read_json_object(trial.source_config, context="v2 source config")
    configured = source_config.get("sources")
    if not isinstance(configured, list):
        raise StudySummaryError("v2 source config sources must be a list")
    expected_rows = [
        row for row in configured if isinstance(row, dict) and row.get("enabled", True) is not False
    ]
    snapshots = value.get("sources")
    if not isinstance(snapshots, list) or len(snapshots) != len(expected_rows):
        raise StudySummaryError(f"{context}.sources does not match enabled sources")

    for index, (configured_row, snapshot) in enumerate(zip(expected_rows, snapshots, strict=True)):
        row_context = f"{context}.sources[{index}]"
        if not isinstance(snapshot, dict) or set(snapshot) != _SOURCE_SNAPSHOT_KEYS:
            raise StudySummaryError(f"{row_context} has the wrong snapshot fields")
        if snapshot.get("dataset_id") != configured_row.get("dataset_id"):
            raise StudySummaryError(f"{row_context} has the wrong dataset id")
        expected_paths = {
            key: _resolve_one_source_path(
                project_root,
                configured_row[key],
                context=f"source config {configured_row.get('dataset_id')}.{key}",
            )
            for key in ("samples", "taxa", "train_cache")
        }
        validation_value = configured_row.get("validation_cache")
        expected_paths["validation_cache"] = (
            None
            if validation_value is None
            else _resolve_one_source_path(
                project_root,
                validation_value,
                context=(f"source config {configured_row.get('dataset_id')}.validation_cache"),
            )
        )
        expected_paths["source_metadata"] = expected_paths["samples"].parent / "source.json"

        for key, expected_path in expected_paths.items():
            digest_key = f"{key}_sha256"
            if expected_path is None:
                if snapshot.get(key) is not None or snapshot.get(digest_key) is not None:
                    raise StudySummaryError(f"{row_context}.{key} must be null")
                continue
            actual_path = _resolved_path(snapshot.get(key), context=f"{row_context}.{key}")
            if actual_path != expected_path.resolve():
                raise StudySummaryError(f"{row_context}.{key} has the wrong path")
            _require_sha256(snapshot.get(digest_key), context=f"{row_context}.{digest_key}")


def _seed_invariant_config(value: dict[str, Any]) -> dict[str, Any]:
    invariant = json.loads(json.dumps(value, ensure_ascii=False))
    invariant.pop("seed", None)
    invariant.pop("kind", None)
    training = invariant.get("training")
    if isinstance(training, dict):
        training.pop("seed", None)
    return invariant


def _validate_run_config(
    value: dict[str, Any],
    study: StudyConfig,
    trial: SourceTrial,
    *,
    seed: int,
    run_directory: Path,
    project_root: Path,
    expected_crosswalk_digest: str,
) -> None:
    context = f"run config in {run_directory}"
    if value.get("experiment") != study.experiment_id:
        raise StudySummaryError(f"{context} has the wrong experiment")
    if value.get("protocol") != study.protocol:
        raise StudySummaryError(f"{context} has the wrong protocol")
    if value.get("model") != _EXPECTED_MODEL:
        raise StudySummaryError(f"{context} has the wrong model")
    model_runtime = value.get("model_runtime")
    expected_model_runtime = {
        "cache_identity": f"openai-clip:{DEFAULT_CLIP_MODEL}@{OPENAI_CLIP_COMMIT}",
        "clip_distribution": _EXPECTED_CLIP_DISTRIBUTION,
        "clip_version": _EXPECTED_CLIP_VERSION,
        "clip_repository_url": OPENAI_CLIP_REPOSITORY,
        "clip_commit": OPENAI_CLIP_COMMIT,
        "checkpoint_path": str(
            (study.common.model_cache_dir / VIT_L14_336_CHECKPOINT_FILENAME).resolve()
        ),
        "checkpoint_sha256": VIT_L14_336_CHECKPOINT_SHA256,
        "checkpoint_size_bytes": _EXPECTED_CHECKPOINT_SIZE_BYTES,
    }
    if (
        not isinstance(model_runtime, dict)
        or set(model_runtime) != _MODEL_RUNTIME_KEYS
        or not _strict_json_equal(model_runtime, expected_model_runtime)
    ):
        raise StudySummaryError(f"{context} has the wrong locked CLIP runtime identity")
    if type(value.get("seed")) is not int or value["seed"] != seed:
        raise StudySummaryError(f"{context} has the wrong seed")

    source_path = _resolved_path(
        value.get("source_config"),
        context=f"{context}.source_config",
    )
    if source_path != trial.source_config.resolve():
        raise StudySummaryError(f"{context} has the wrong source config")
    if value.get("source_config_sha256") != _sha256_file(trial.source_config):
        raise StudySummaryError(f"{context} has the wrong source config digest")

    cub_cache = _resolved_path(
        value.get("cub_feature_cache"),
        context=f"{context}.cub_feature_cache",
    )
    if cub_cache != study.common.cub_feature_cache.resolve():
        raise StudySummaryError(f"{context} has the wrong CUB feature cache")
    _require_sha256(
        value.get("cub_feature_cache_sha256"),
        context=f"{context}.cub_feature_cache_sha256",
    )

    if value.get("training") != _expected_training(study, seed=seed):
        raise StudySummaryError(f"{context} does not match common training settings")
    expected_method = {
        "feature_dim": _EXPECTED_FEATURE_DIM,
        "hidden_dim": study.common.hidden_dim,
        "architecture": _EXPECTED_ARCHITECTURE,
    }
    if value.get("method") != expected_method:
        raise StudySummaryError(f"{context} has the wrong adapter configuration")

    validation_fraction, duplicate_audit = _source_protocol_values(trial)
    if value.get("validation_taxon_fraction") != validation_fraction:
        raise StudySummaryError(f"{context} has the wrong validation taxon fraction")
    if value.get("duplicate_audit") != duplicate_audit:
        raise StudySummaryError(f"{context} has the wrong duplicate-audit settings")
    crosswalk_digest = _require_sha256(
        value.get("cub_crosswalk_digest"), context=f"{context}.cub_crosswalk_digest"
    )
    if crosswalk_digest != expected_crosswalk_digest:
        raise StudySummaryError(f"{context} has the wrong canonical CUB crosswalk digest")
    _require_sha256(
        value.get("source_code_sha256"),
        context=f"{context}.source_code_sha256",
    )
    _validate_source_snapshots(
        value,
        trial,
        project_root=project_root,
        context=context,
    )


def _load_complete_run(
    run_directory: Path,
    study: StudyConfig,
    trial: SourceTrial,
    *,
    seed: int,
    kind: str,
    project_root: Path,
    class_names: tuple[str, ...],
    canonical_samples: tuple[CubSample, ...],
    canonical_targets: tuple[int, ...],
    expected_crosswalk: dict[str, Any],
) -> dict[str, Any]:
    config = _read_json_object(run_directory / "config.json", context="run config")
    _validate_run_directory_name(run_directory, kind=kind, config=config)
    _validate_run_config(
        config,
        study,
        trial,
        seed=seed,
        run_directory=run_directory,
        project_root=project_root,
        expected_crosswalk_digest=str(expected_crosswalk["digest"]),
    )
    _verify_checksums(run_directory)
    checksum_lines = (run_directory / "checksums.sha256").read_text(encoding="utf-8").splitlines()
    for name in _REQUIRED_V2_RUN_FILES:
        if name == "checksums.sha256":
            continue
        if not any(line.endswith(f"  {name}") for line in checksum_lines):
            raise StudySummaryError(
                f"Checksum manifest does not cover required v2 artifact: {run_directory / name}"
            )

    result = _read_json_object(run_directory / "result.json", context="run result")
    complete = _read_json_object(
        run_directory / "run_complete.json",
        context="run completion record",
    )
    if complete.get("state") != "complete":
        raise StudySummaryError(f"Run is not marked complete: {run_directory}")
    if complete.get("result") != result:
        raise StudySummaryError(
            f"run_complete.json does not embed result.json exactly: {run_directory}"
        )
    run_summary = _summarize_result(
        result,
        trial_id=trial.trial_id,
        seed=seed,
        kind=kind,
        run_directory=run_directory,
    )
    source_validation = _read_json_object(
        run_directory / "source_validation.json",
        context="source validation record",
    )
    best_step = run_summary["best_step"]
    if (
        type(source_validation.get("best_step")) is not int
        or source_validation["best_step"] != best_step
    ):
        raise StudySummaryError(
            f"source_validation.best_step disagrees with result.best_step: {run_directory}"
        )
    if (
        type(source_validation.get("refit_steps")) is not int
        or source_validation["refit_steps"] != best_step
    ):
        raise StudySummaryError(
            f"source_validation.refit_steps disagrees with result.best_step: {run_directory}"
        )
    crosswalk = _read_json_object(
        run_directory / "cub_taxonomy_crosswalk.json",
        context="CUB taxonomy crosswalk",
    )
    if not _strict_json_equal(crosswalk, expected_crosswalk):
        raise StudySummaryError(
            f"CUB crosswalk artifact is not the canonical locked crosswalk: {run_directory}"
        )
    prediction_audit = _validate_prediction_artifact(
        run_directory / "cub_predictions.jsonl",
        result,
        class_names=class_names,
        canonical_samples=canonical_samples,
        canonical_targets=canonical_targets,
        seed=seed,
    )
    run_summary["_audit"] = {
        "config": config,
        "cub_taxonomy_crosswalk_sha256": _sha256_file(
            run_directory / "cub_taxonomy_crosswalk.json"
        ),
        "source_text_prototypes_content_sha256": _prototype_content_digest(
            run_directory / "source_text_prototypes.pt",
            kind="source",
        ),
        "target_text_prototypes_content_sha256": _prototype_content_digest(
            run_directory / "target_text_prototypes.pt",
            kind="target",
            locked_class_names=class_names,
        ),
        **prediction_audit,
    }
    return run_summary


def _find_exactly_one_run(
    runs_root: Path,
    study: StudyConfig,
    trial: SourceTrial,
    *,
    seed: int,
    project_root: Path,
    class_names: tuple[str, ...],
    canonical_samples: tuple[CubSample, ...],
    canonical_targets: tuple[int, ...],
    expected_crosswalk: dict[str, Any],
) -> dict[str, Any]:
    kind = f"{study.study_id}-{trial.trial_id}-seed{seed}"
    name_pattern = re.compile(rf"\d{{8}}T\d{{6}}\.\d{{6}}Z-{re.escape(kind)}-[0-9a-f]{{10}}\Z")
    candidates = tuple(
        path
        for path in sorted(runs_root.iterdir())
        if path.is_dir() and name_pattern.fullmatch(path.name) is not None
    )
    if not candidates:
        raise StudySummaryError(f"Missing run for trial={trial.trial_id}, seed={seed}")
    if len(candidates) != 1:
        raise StudySummaryError(
            f"Duplicate runs for trial={trial.trial_id}, seed={seed}: "
            f"{[path.name for path in candidates]}"
        )
    run_directory = candidates[0]
    missing = [name for name in _REQUIRED_V2_RUN_FILES if not (run_directory / name).is_file()]
    if missing:
        raise StudySummaryError(
            f"Incomplete run for trial={trial.trial_id}, seed={seed}: "
            f"{run_directory.name}; missing={missing}"
        )
    return _load_complete_run(
        run_directory,
        study,
        trial,
        seed=seed,
        kind=kind,
        project_root=project_root,
        class_names=class_names,
        canonical_samples=canonical_samples,
        canonical_targets=canonical_targets,
        expected_crosswalk=expected_crosswalk,
    )


def _three_seed_metric(
    rows: list[dict[str, Any]],
    *,
    value_for_row: Any,
) -> dict[str, Any]:
    values = [float(value_for_row(row)) for row in rows]
    if len(values) != 3:
        raise StudySummaryError(
            f"Three-seed summary requires exactly three values, found {len(values)}"
        )
    if any(not math.isfinite(value) for value in values):
        raise StudySummaryError("Three-seed summary values must be finite")
    mean = statistics.fmean(values)
    sample_std = statistics.stdev(values)
    margin = _THREE_SEED_T_CRITICAL_95 * sample_std / math.sqrt(3.0)
    return {
        "unit": "percentage_points",
        "per_seed": [
            {"seed": int(row["seed"]), "value": value}
            for row, value in zip(rows, values, strict=True)
        ],
        "mean": mean,
        "sample_standard_deviation": sample_std,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "ci95_method": _CI_METHOD,
    }


def _validate_baseline_parity(rows: list[dict[str, Any]], *, trial_id: str) -> None:
    baselines = {
        (
            int(row["baseline"]["total"]),
            int(row["baseline"]["top1_correct"]),
            float(row["baseline"]["top1"]),
        )
        for row in rows
    }
    if len(baselines) != 1:
        raise StudySummaryError(f"Frozen CLIP baseline differs across seeds for trial={trial_id}")


def _recorded_input_digests(config: dict[str, Any]) -> dict[Path, str]:
    result: dict[Path, str] = {}

    def add(path_value: object, digest_value: object, *, context: str) -> None:
        path = _resolved_path(path_value, context=f"{context}.path")
        digest = _require_sha256(digest_value, context=f"{context}.sha256")
        previous = result.get(path)
        if previous is not None and previous != digest:
            raise StudySummaryError(
                f"Conflicting recorded digests for input file {path}: {previous} != {digest}"
            )
        result[path] = digest

    add(
        config.get("cub_feature_cache"),
        config.get("cub_feature_cache_sha256"),
        context="CUB feature cache",
    )
    sources = config.get("sources")
    if not isinstance(sources, list):
        raise StudySummaryError("run config sources must be a list")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise StudySummaryError(f"run config sources[{index}] must be an object")
        for key in ("source_metadata", "samples", "taxa", "train_cache"):
            add(
                source.get(key),
                source.get(f"{key}_sha256"),
                context=f"sources[{index}].{key}",
            )
        if source.get("validation_cache") is not None:
            add(
                source.get("validation_cache"),
                source.get("validation_cache_sha256"),
                context=f"sources[{index}].validation_cache",
            )
    return result


def _verify_stable_digest_map(
    inputs: dict[Path, str],
) -> list[dict[str, str]]:
    verified: list[dict[str, str]] = []
    for path, expected in sorted(inputs.items(), key=lambda item: str(item[0])):
        try:
            before = path.stat()
        except OSError as error:
            raise StudySummaryError(f"Recorded input no longer exists: {path}") from error
        actual = _sha256_file(path)
        try:
            after = path.stat()
        except OSError as error:
            raise StudySummaryError(f"Recorded input disappeared while hashing: {path}") from error
        before_state = (before.st_size, before.st_mtime_ns)
        after_state = (after.st_size, after.st_mtime_ns)
        if before_state != after_state:
            raise StudySummaryError(f"Input changed while being audited: {path}")
        if actual != expected:
            raise StudySummaryError(f"Recorded input digest no longer matches: {path}")
        verified.append({"path": str(path), "sha256": actual})
    return verified


def _audit_three_seed_identity(
    run_rows: list[dict[str, Any]],
    study: StudyConfig,
) -> list[dict[str, str]]:
    target_digests: set[str] = set()
    cub_identity_digests: set[str] = set()
    baseline_top5_digests: set[str] = set()
    all_inputs: dict[Path, str] = {}
    for trial in study.source_trials:
        rows = [row for row in run_rows if row["trial_id"] == trial.trial_id]
        if len(rows) != 3:
            raise StudySummaryError(
                f"Expected three completed runs for trial={trial.trial_id}, found {len(rows)}"
            )
        configs = [row["_audit"]["config"] for row in rows]
        invariant_configs = [_seed_invariant_config(config) for config in configs]
        if any(config != invariant_configs[0] for config in invariant_configs[1:]):
            raise StudySummaryError(f"Run configs differ beyond seed for trial={trial.trial_id}")
        source_digests = {
            str(row["_audit"]["source_text_prototypes_content_sha256"]) for row in rows
        }
        if len(source_digests) != 1:
            raise StudySummaryError(
                f"Source text prototypes differ across seeds for trial={trial.trial_id}"
            )
        crosswalk_digests = {str(row["_audit"]["cub_taxonomy_crosswalk_sha256"]) for row in rows}
        if len(crosswalk_digests) != 1:
            raise StudySummaryError(
                f"CUB crosswalk artifacts differ across seeds for trial={trial.trial_id}"
            )
        target_digests.update(
            str(row["_audit"]["target_text_prototypes_content_sha256"]) for row in rows
        )
        trial_identity_digests = {
            str(row["_audit"]["cub_identity_and_labels_digest"]) for row in rows
        }
        if len(trial_identity_digests) != 1:
            raise StudySummaryError(
                f"CUB image identities or labels differ across seeds for trial={trial.trial_id}"
            )
        trial_baseline_digests = {str(row["_audit"]["baseline_top5_digest"]) for row in rows}
        if len(trial_baseline_digests) != 1:
            raise StudySummaryError(
                f"Frozen CLIP baseline Top-5 rows differ across seeds for trial={trial.trial_id}"
            )
        cub_identity_digests.update(trial_identity_digests)
        baseline_top5_digests.update(trial_baseline_digests)
        for path, digest in _recorded_input_digests(configs[0]).items():
            previous = all_inputs.get(path)
            if previous is not None and previous != digest:
                raise StudySummaryError(f"Trials record conflicting input digests for {path}")
            all_inputs[path] = digest

    if len(target_digests) != 1:
        raise StudySummaryError("Target text prototypes differ across preregistered runs")
    if len(cub_identity_digests) != 1:
        raise StudySummaryError("CUB image identities or labels differ across trials")
    if len(baseline_top5_digests) != 1:
        raise StudySummaryError("Frozen CLIP baseline Top-5 rows differ across trials")
    return _verify_stable_digest_map(all_inputs)


def summarize_study(
    study_config: Path | str,
    runs_root: Path | str | None = None,
    *,
    project_root: Path | str,
) -> dict[str, Any]:
    """Return the deterministic BirdMix-v2 audit summary without writing files."""

    root = Path(project_root).expanduser().resolve()
    config_path = Path(study_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        study = load_study_config(config_path, root)
    except StudyConfigError as error:
        raise StudySummaryError(str(error)) from error
    if len(study.seeds) != 3:
        raise StudySummaryError(
            f"BirdMix-v2 summary requires exactly three preregistered seeds; "
            f"found {len(study.seeds)}"
        )
    try:
        class_names = read_cub_class_names(study.common.cub_class_names)
        crosswalk = build_cub_birdnet_crosswalk(
            study.common.cub_class_names,
            study.common.birdnet_csv,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise StudySummaryError(f"Cannot rebuild the locked CUB crosswalk: {error}") from error
    expected_crosswalk = _expected_crosswalk_artifact(crosswalk)
    try:
        canonical_cub = prepare_cub(
            study.common.cub_data_root,
            split="test",
            download=False,
            verify_images=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise StudySummaryError(
            f"Cannot rebuild the canonical CUB test dataset: {error}"
        ) from error
    if len(canonical_cub) != CUB_TEST_COUNT:
        raise StudySummaryError(
            f"Canonical CUB test dataset must contain exactly {CUB_TEST_COUNT} samples; "
            f"found {len(canonical_cub)}"
        )
    canonical_samples = tuple(canonical_cub.samples)
    canonical_targets = canonical_cub.targets

    runs = study.common.runs_root if runs_root is None else Path(runs_root).expanduser()
    if not runs.is_absolute():
        runs = root / runs
    if not runs.is_dir():
        raise StudySummaryError(f"Runs root does not exist: {runs}")

    run_rows = [
        _find_exactly_one_run(
            runs,
            study,
            trial,
            seed=seed,
            project_root=root,
            class_names=class_names,
            canonical_samples=canonical_samples,
            canonical_targets=canonical_targets,
            expected_crosswalk=expected_crosswalk,
        )
        for seed in study.seeds
        for trial in study.source_trials
    ]
    verified_inputs = _audit_three_seed_identity(run_rows, study)
    trial_rows: list[dict[str, Any]] = []
    for trial in study.source_trials:
        rows = [row for row in run_rows if row["trial_id"] == trial.trial_id]
        _validate_baseline_parity(rows, trial_id=trial.trial_id)
        trial_rows.append(
            {
                "trial_id": trial.trial_id,
                "seed_count": 3,
                "baseline_top1": _three_seed_metric(
                    rows,
                    value_for_row=lambda row: row["baseline"]["top1"],
                ),
                "adapted_top1": _three_seed_metric(
                    rows,
                    value_for_row=lambda row: row["adapted"]["top1"],
                ),
                "gain_top1": _three_seed_metric(
                    rows,
                    value_for_row=lambda row: row["gain_top1_percentage_points"],
                ),
                "all_seeds_positive_gain": all(
                    float(row["gain_top1_percentage_points"]) > 0.0 for row in rows
                ),
                "all_seeds_strict_transfer_criterion_passed": all(
                    bool(row["strict_transfer_criterion_passed"]) for row in rows
                ),
            }
        )

    for row in run_rows:
        row.pop("_audit")

    return {
        "schema_version": 2,
        "experiment_id": study.experiment_id,
        "protocol": study.protocol,
        "study_id": study.study_id,
        "study_config": str(config_path.resolve()),
        "study_config_sha256": _sha256_file(config_path),
        "runs_root": str(runs.resolve()),
        "selection_policy": "exactly-one-complete-run-per-preregistered-trial-seed",
        "target_based_run_selection": False,
        "aggregation_policy": {
            "seed_count": 3,
            "point_estimate": "arithmetic mean over preregistered seeds",
            "ci95": _CI_METHOD,
        },
        "audit": {
            "seed_invariant_run_configs": True,
            "source_and_target_text_prototypes_identical_across_seeds": True,
            "cub_crosswalk_artifacts_identical_across_seeds": True,
            "cub_crosswalk_matches_locked_class_names_and_birdnet_csv": True,
            "target_prototypes_match_locked_cub_class_names": True,
            "cub_predictions_recomputed_from_exactly_5794_rows": True,
            "cub_identity_labels_and_baseline_top5_identical_across_seeds": True,
            "recorded_input_digests_identical_across_seeds": True,
            "recorded_inputs_rehashed_without_size_or_mtime_change": True,
            "verified_input_file_count": len(verified_inputs),
            "verified_inputs": verified_inputs,
        },
        "run_count": len(run_rows),
        "runs": run_rows,
        "trials": trial_rows,
    }


def _summary_bytes(summary: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _summary_digest(summary: dict[str, Any]) -> str:
    canonical = json.dumps(
        summary,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def write_summary(
    summary: dict[str, Any],
    *,
    output_root: Path | str | None = None,
    output: Path | str | None = None,
    timestamp: dt.datetime | None = None,
) -> Path:
    """Write once to an explicit file or a new timestamp-and-digest directory."""

    if (output_root is None) == (output is None):
        raise StudySummaryError("Choose exactly one of output_root or output")
    payload = _summary_bytes(summary)
    if output is not None:
        target = Path(output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                handle.write(payload)
        except FileExistsError as error:
            raise StudySummaryError(f"Refusing to overwrite summary: {target}") from error
        return target

    assert output_root is not None
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    moment = timestamp or dt.datetime.now(dt.timezone.utc)
    if moment.tzinfo is None:
        raise StudySummaryError("Summary timestamp must be timezone-aware")
    stamp = moment.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    digest = _summary_digest(summary)
    directory = root / f"{stamp}-summary-{digest[:10]}"
    try:
        directory.mkdir()
    except FileExistsError as error:
        raise StudySummaryError(f"Refusing to overwrite summary directory: {directory}") from error
    target = directory / "summary.json"
    with target.open("xb") as handle:
        handle.write(payload)
    checksum = hashlib.sha256(payload).hexdigest()
    with (directory / "checksums.sha256").open("x", encoding="utf-8") as handle:
        handle.write(f"{checksum}  summary.json\n")
    return target


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=(
            project_root / "experiments/07_feature_adapter_clip_birdmix_v2_cub/configs/"
            "clip_birdmix_v2_study.json"
        ),
    )
    parser.add_argument("--runs-root", type=Path)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-root", type=Path)
    destination.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    try:
        summary = summarize_study(
            args.study_config,
            args.runs_root,
            project_root=project_root,
        )
        if args.output is not None:
            output = write_summary(summary, output=args.output)
        else:
            output_root = args.output_root or (Path(summary["runs_root"]) / "summaries")
            output = write_summary(summary, output_root=output_root)
    except StudySummaryError as error:
        raise SystemExit(f"BirdMix-v2 summary failed: {error}") from error
    print(json.dumps({"summary": str(output)}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
