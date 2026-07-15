#!/usr/bin/env python3
"""Train and evaluate linear/residual heads on frozen CLIP CUB features."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from ttvr import (
    CLIPBackend,
    compute_topk_accuracy,
    compute_transfer_counts,
    load_official_prompts,
    ordered_predictions,
    prepare_cub,
    validate_class_name_alignment,
)
from ttvr.methods.residual_head import (
    HeadTrial,
    ResidualHeadSearchConfig,
    combine_residual_logits,
    evaluate_logits,
    head_logits_from_state,
    refit_feature_head,
    search_feature_head,
    stable_split_score,
    stratified_hash_split,
)
from ttvr.metrics import exact_mcnemar_test, paired_bootstrap_accuracy_gain

MODEL_NAME = "ViT-L/14@336px"
MODEL_PRECISION = "fp32"
SEED = 2026
PARENT_RUN_ID = "20260714T185902.445729Z-full-3b975c99f4"
PARENT_PREDICTIONS_SHA256 = "5ff4b21fd7827cf1aee21642947c662e6567caca79e090d9767ba33e77ce512e"


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            encoded = f"{line}\n".encode()
            handle.write(encoded.decode())
            digest.update(encoded)
    return digest.hexdigest()


def _write_checksums(run_dir: Path) -> Path:
    output = run_dir / "checksums.sha256"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    lines = [
        f"{_sha256(path)}  {path.relative_to(run_dir).as_posix()}"
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _atomic_torch_save(value: Any, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_digest(project_root: Path) -> str:
    files = (
        project_root / "pyproject.toml",
        Path(__file__).resolve(),
        project_root / "scripts/residual_head/cache_clip_cub_features.py",
        project_root / "src/ttvr/__init__.py",
        project_root / "src/ttvr/data/cub.py",
        project_root / "src/ttvr/metrics.py",
        project_root / "src/ttvr/models/__init__.py",
        project_root / "src/ttvr/models/base.py",
        project_root / "src/ttvr/models/cached.py",
        project_root / "src/ttvr/models/clip.py",
        project_root / "src/ttvr/methods/fudd/prompts.py",
        project_root / "src/ttvr/methods/residual_head/__init__.py",
        project_root / "src/ttvr/methods/residual_head/training.py",
    )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(project_root).as_posix()):
        digest.update(path.relative_to(project_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
        "pillow": _package_version("pillow"),
        "clip": _package_version("clip"),
        "cuda_available": cuda,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32 if cuda else None,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32 if cuda else None,
    }


def _feature_cache_path(
    cache_dir: Path,
    split: str,
    dataset_fingerprint: str,
    backend: CLIPBackend,
) -> Path:
    identity = hashlib.sha256(backend.cache_identity.encode()).hexdigest()[:16]
    model = _slug(backend.model_name)
    precision = _slug(f"{backend.precision}-{backend.feature_dtype_name}")
    return (
        cache_dir
        / "image_features"
        / (f"cub-{split}-{model}-{identity}-{precision}-{dataset_fingerprint[:16]}-all.pt")
    )


class ProgressPrinter:
    def __init__(self) -> None:
        self._last: dict[tuple[str, int], int] = {}

    def __call__(self, stage: str, completed: int, total: int) -> None:
        key = (stage, total)
        previous = self._last.get(key, 0)
        if completed == total or completed - previous >= 256:
            print(f"[{stage}] {completed:,}/{total:,}", flush=True)
            self._last[key] = completed


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=project_root / "data")
    parser.add_argument("--prompt-root", type=Path, default=project_root / "data/fudd_official")
    parser.add_argument("--cache-dir", type=Path, default=project_root / ".cache/fudd_clip_cub")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=project_root / "experiments/05_residual_head_clip_cub/runs",
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-device", default="cuda")
    return parser.parse_args()


def _metric_payload(metrics: Any) -> dict[str, int | float]:
    return metrics.to_dict()


def _print_trial(trial: HeadTrial) -> None:
    print("[trial] " + json.dumps(trial.to_dict(), sort_keys=True), flush=True)


def _batched_similarity_logits(
    image_features: torch.Tensor,
    class_features: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match experiment 01's CUDA similarity and ranking path exactly."""

    if (
        image_features.ndim != 2
        or class_features.ndim != 2
        or image_features.shape[1] != class_features.shape[1]
        or batch_size <= 0
    ):
        raise ValueError("Feature matrices or similarity batch size are invalid")
    scores: list[torch.Tensor] = []
    rankings: list[torch.Tensor] = []
    class_features = class_features.to(device)
    with torch.inference_mode():
        for start in range(0, image_features.shape[0], batch_size):
            batch = image_features[start : start + batch_size].to(device, non_blocking=True)
            logits = batch @ class_features.t()
            scores.append(logits.cpu())
            rankings.append(ordered_predictions(logits).cpu())
    return torch.cat(scores), torch.cat(rankings)


def _validate_feature_alignment(feature_set: Any, dataset: Any) -> None:
    expected_indices = torch.arange(len(dataset), dtype=torch.long)
    expected_labels = torch.tensor([sample.target for sample in dataset.samples], dtype=torch.long)
    if not torch.equal(feature_set.sample_indices, expected_indices):
        raise RuntimeError("Cached sample indices do not align with the dataset")
    if not torch.equal(feature_set.labels, expected_labels):
        raise RuntimeError("Cached labels do not align with the dataset")
    if not bool(torch.isfinite(feature_set.features).all()):
        raise RuntimeError("Cached image features contain non-finite values")
    norms = torch.linalg.vector_norm(feature_set.features.float(), dim=1)
    if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4)):
        raise RuntimeError("Cached image features are not L2-normalised")


def _comparison_payload(
    baseline_predictions: torch.Tensor,
    comparison_predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    baseline_correct = baseline_predictions[:, 0].eq(labels)
    comparison_correct = comparison_predictions[:, 0].eq(labels)
    return {
        "transfers": compute_transfer_counts(
            baseline_predictions, comparison_predictions, labels
        ).to_dict(),
        "mcnemar": exact_mcnemar_test(baseline_correct, comparison_correct).to_dict(),
        "paired_bootstrap": paired_bootstrap_accuracy_gain(
            baseline_correct,
            comparison_correct,
            reps=10_000,
            seed=SEED,
        ).to_dict(),
    }


def _parent_baseline_top10(parent_path: Path, dataset: Any) -> list[list[int]]:
    if _sha256(parent_path) != PARENT_PREDICTIONS_SHA256:
        raise RuntimeError("Canonical parent prediction digest does not match")
    rows: list[list[int]] = []
    with parent_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            if index >= len(dataset):
                raise RuntimeError("Canonical parent has too many prediction rows")
            sample = dataset.samples[index]
            identity = (
                record.get("sample_index") == index
                and record.get("image_id") == sample.image_id
                and record.get("relative_path") == sample.relative_path.as_posix()
                and record.get("target_class_id") == sample.target
            )
            if not identity:
                raise RuntimeError(f"Canonical parent row {index} is misaligned")
            rows.append(record["baseline_topk_class_ids"])
    if len(rows) != len(dataset):
        raise RuntimeError("Canonical parent prediction count does not match test data")
    return rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA image encoding requested but unavailable")
    if args.head_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA head training requested but unavailable")
    execution = {
        "data_root": str(args.data_root.expanduser().resolve()),
        "prompt_root": str(args.prompt_root.expanduser().resolve()),
        "cache_dir": str(args.cache_dir.expanduser().resolve()),
        "runs_root": str(args.runs_root.expanduser().resolve()),
        "image_batch_size": args.image_batch_size,
        "num_workers": args.num_workers,
        "image_device": args.device,
        "head_device": args.head_device,
        "seed": SEED,
    }
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True

    source_digest = _source_digest(project_root)
    created = _utc_now()
    run_id = created.strftime("%Y%m%dT%H%M%S.%fZ") + f"-full-{source_digest[:10]}"
    run_dir = args.runs_root.expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "run_state.json"
    initial_state = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "RUNNING",
        "created_at_utc": created.isoformat(),
        "source_digest": source_digest,
        "execution": execution,
    }
    _write_json(state_path, initial_state)

    try:
        prompts = load_official_prompts(args.prompt_root)
        backend = CLIPBackend(
            model_name=MODEL_NAME,
            device=args.device,
            precision=MODEL_PRECISION,
            text_batch_size=256,
            model_cache_dir=args.cache_dir / "models",
        )
        train_dataset = prepare_cub(
            args.data_root,
            transform=backend.preprocess,
            download=True,
            verify_images=False,
            split="train",
        )
        validate_class_name_alignment(train_dataset.class_names, prompts.class_names)
        train_cache = _feature_cache_path(
            args.cache_dir, "train", train_dataset.fingerprint, backend
        )
        train_features = backend.encode_dataset(
            train_dataset,
            batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            seed=SEED,
            cache_path=train_cache,
            cache_tag=(f"cub:train:{train_dataset.fingerprint}:{backend.cache_identity}"),
            progress=ProgressPrinter(),
        )
        _validate_feature_alignment(train_features, train_dataset)
        text_prototypes = backend.pool_prompt_groups(prompts.single_template_prompts()).detach()
        logit_scale = float(
            backend.model.logit_scale.exp().detach().float().clamp(max=100).cpu().item()
        )
        train_similarities, _train_rankings = _batched_similarity_logits(
            train_features.features,
            text_prototypes,
            batch_size=args.image_batch_size,
            device=backend.device,
        )
        train_base_logits = train_similarities.mul(logit_scale)

        sample_keys = [
            f"{sample.image_id}:{sample.relative_path.as_posix()}"
            for sample in train_dataset.samples
        ]
        search_config = ResidualHeadSearchConfig(seed=SEED)
        split = stratified_hash_split(
            train_features.labels,
            sample_keys,
            validation_per_class=search_config.validation_per_class,
            seed=SEED,
        )
        split_rows = []
        validation_set = set(split.validation_indices.tolist())
        for index, sample in enumerate(train_dataset.samples):
            split_rows.append(
                {
                    "sample_index": index,
                    "image_id": sample.image_id,
                    "relative_path": sample.relative_path.as_posix(),
                    "target_class_id": sample.target,
                    "partition": "validation" if index in validation_set else "fit",
                    "split_score_sha256": stable_split_score(SEED, sample_keys[index]),
                }
            )
        split_digest = _write_jsonl(run_dir / "split_manifest.jsonl", split_rows)

        print("[head] searching linear probe", flush=True)
        linear_search = search_feature_head(
            train_features.features,
            train_features.labels,
            split,
            search_config,
            mode="linear",
            device=args.head_device,
            trial_callback=_print_trial,
        )
        print(json.dumps(linear_search.selection.to_dict(), indent=2), flush=True)
        print("[head] searching residual head", flush=True)
        residual_search = search_feature_head(
            train_features.features,
            train_features.labels,
            split,
            search_config,
            mode="residual",
            base_logits=train_base_logits,
            device=args.head_device,
            trial_callback=_print_trial,
        )
        print(json.dumps(residual_search.selection.to_dict(), indent=2), flush=True)

        print("[head] refitting selected heads on all training images", flush=True)
        linear_state = refit_feature_head(
            train_features.features,
            train_features.labels,
            linear_search.selection,
            search_config,
            device=args.head_device,
        )
        residual_state = refit_feature_head(
            train_features.features,
            train_features.labels,
            residual_search.selection,
            search_config,
            base_logits=train_base_logits,
            device=args.head_device,
        )

        # The official test split is first touched by the head-selection code
        # only after both hyperparameter searches and full-train refits finish.
        test_dataset = prepare_cub(
            args.data_root,
            transform=backend.preprocess,
            download=True,
            verify_images=False,
            split="test",
        )
        validate_class_name_alignment(test_dataset.class_names, prompts.class_names)
        test_cache = _feature_cache_path(args.cache_dir, "test", test_dataset.fingerprint, backend)
        test_features = backend.encode_dataset(
            test_dataset,
            batch_size=args.image_batch_size,
            num_workers=args.num_workers,
            seed=SEED,
            cache_path=test_cache,
            cache_tag=f"cub:test:{test_dataset.fingerprint}:{backend.cache_identity}",
            progress=ProgressPrinter(),
        )
        _validate_feature_alignment(test_features, test_dataset)
        test_similarities, baseline_predictions = _batched_similarity_logits(
            test_features.features,
            text_prototypes,
            batch_size=args.image_batch_size,
            device=backend.device,
        )
        test_base_logits = test_similarities.mul(logit_scale)
        linear_logits = head_logits_from_state(
            test_features.features,
            linear_state,
            class_count=prompts.class_count,
            device=args.head_device,
        )
        residual_only_logits = head_logits_from_state(
            test_features.features,
            residual_state,
            class_count=prompts.class_count,
            device=args.head_device,
        )
        residual_logits = combine_residual_logits(
            test_base_logits,
            residual_only_logits,
            residual_search.selection.alpha,
        )

        linear_predictions = ordered_predictions(linear_logits)
        residual_predictions = ordered_predictions(residual_logits)
        labels = test_features.labels
        baseline_metrics = compute_topk_accuracy(baseline_predictions, labels)
        linear_metrics = evaluate_logits(linear_logits, labels)
        residual_metrics = evaluate_logits(residual_logits, labels)

        parent_predictions_path = (
            project_root / "experiments/01_fudd_clip_cub/runs" / PARENT_RUN_ID / "predictions.jsonl"
        )
        parent_top10 = _parent_baseline_top10(parent_predictions_path, test_dataset)
        baseline_top10 = baseline_predictions[:, :10].tolist()
        checks = {
            "official_train_size": train_features.size == 5_994,
            "official_test_size": test_features.size == 5_794,
            "fit_size": split.fit_indices.numel() == 4_794,
            "validation_size": split.validation_indices.numel() == 1_200,
            "validation_six_per_class": (
                torch.bincount(
                    train_features.labels.index_select(0, split.validation_indices),
                    minlength=200,
                ).tolist()
                == [6] * 200
            ),
            "alpha_zero_in_search_grid": 0.0 in search_config.alpha_grid,
            "parent_baseline_top10_exact_match": baseline_top10 == parent_top10,
            "parent_baseline_top1_exact_match": baseline_metrics.top1_correct == 3_671,
            "parent_baseline_top5_exact_match": baseline_metrics.top5_correct == 5_338,
            "head_parameter_count": (
                linear_state["linear.weight"].numel() + linear_state["linear.bias"].numel()
                == 153_800
            ),
        }
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"Integrity checks failed: {failed}")

        trials = [trial.to_dict() for trial in (*linear_search.trials, *residual_search.trials)]
        trials_digest = _write_jsonl(run_dir / "trials.jsonl", trials)
        checkpoint_path = run_dir / "heads.pt"
        _atomic_torch_save(
            {
                "format": 1,
                "model_name": MODEL_NAME,
                "cache_identity": backend.cache_identity,
                "feature_dim": train_features.features.shape[1],
                "class_count": prompts.class_count,
                "search_config": search_config.to_dict(),
                "linear_selection": linear_search.selection.to_dict(),
                "residual_selection": residual_search.selection.to_dict(),
                "linear_validation_state": linear_search.validation_state,
                "residual_validation_state": residual_search.validation_state,
                "linear_full_train_state": linear_state,
                "residual_full_train_state": residual_state,
            },
            checkpoint_path,
        )

        prediction_rows: list[dict[str, Any]] = []
        for index, sample in enumerate(test_dataset.samples):
            prediction_rows.append(
                {
                    "sample_index": index,
                    "image_id": sample.image_id,
                    "relative_path": sample.relative_path.as_posix(),
                    "target_class_id": sample.target,
                    "baseline_top10_class_ids": baseline_predictions[index, :10].tolist(),
                    "linear_top10_class_ids": linear_predictions[index, :10].tolist(),
                    "residual_top10_class_ids": residual_predictions[index, :10].tolist(),
                    "baseline_top1_correct": (
                        int(baseline_predictions[index, 0].item()) == sample.target
                    ),
                    "linear_top1_correct": (
                        int(linear_predictions[index, 0].item()) == sample.target
                    ),
                    "residual_top1_correct": (
                        int(residual_predictions[index, 0].item()) == sample.target
                    ),
                }
            )
        predictions_digest = _write_jsonl(run_dir / "predictions.jsonl", prediction_rows)

        completed = _utc_now()
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "PASS",
            "created_at_utc": created.isoformat(),
            "completed_at_utc": completed.isoformat(),
            "scientific_scope": {
                "name": "exploratory_supervised_frozen_feature_head_tuning",
                "clip_backbone_frozen": True,
                "test_used_for_model_selection": False,
                "pristine_confirmatory_claim_allowed": False,
            },
            "source": {
                "project_source_sha256": source_digest,
                "parent_run_id": PARENT_RUN_ID,
                "parent_predictions_sha256": _sha256(parent_predictions_path),
                "prompt_digest": prompts.source_digest,
                "cache_identity": backend.cache_identity,
                "train_dataset_fingerprint": train_dataset.fingerprint,
                "test_dataset_fingerprint": test_dataset.fingerprint,
                "train_feature_cache_sha256": _sha256(train_cache),
                "test_feature_cache_sha256": _sha256(test_cache),
            },
            "environment": _environment(),
            "protocol": {
                "execution": execution,
                "model_name": MODEL_NAME,
                "precision": MODEL_PRECISION,
                "logit_scale": logit_scale,
                "head_parameter_count": 153_800,
                "search_config": search_config.to_dict(),
                "fit_samples": split.fit_indices.numel(),
                "validation_samples": split.validation_indices.numel(),
                "full_train_refit_samples": train_features.size,
                "test_samples": test_features.size,
            },
            "selection": {
                "linear": linear_search.selection.to_dict(),
                "residual": residual_search.selection.to_dict(),
            },
            "test": {
                "baseline": _metric_payload(baseline_metrics),
                "linear": _metric_payload(linear_metrics),
                "residual": _metric_payload(residual_metrics),
                "linear_vs_baseline": _comparison_payload(
                    baseline_predictions, linear_predictions, labels
                ),
                "residual_vs_baseline": _comparison_payload(
                    baseline_predictions, residual_predictions, labels
                ),
            },
            "artifacts": {
                "split_manifest": {
                    "path": "split_manifest.jsonl",
                    "sha256": split_digest,
                    "rows": len(split_rows),
                },
                "trials": {
                    "path": "trials.jsonl",
                    "sha256": trials_digest,
                    "rows": len(trials),
                },
                "heads": {
                    "path": checkpoint_path.name,
                    "sha256": _sha256(checkpoint_path),
                },
                "predictions": {
                    "path": "predictions.jsonl",
                    "sha256": predictions_digest,
                    "rows": len(prediction_rows),
                },
            },
            "checks": checks,
        }
        _write_json(run_dir / "result.json", result)
        (run_dir / "environment_pip_freeze.txt").write_text(
            subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout,
            encoding="utf-8",
        )
        _write_json(
            state_path,
            {
                **initial_state,
                "status": "PASS",
                "completed_at_utc": completed.isoformat(),
                "result": "result.json",
            },
            overwrite=True,
        )
        checksum_path = _write_checksums(run_dir)
    except BaseException as error:
        _write_json(
            state_path,
            {
                **initial_state,
                "status": "FAILED",
                "failed_at_utc": _utc_now().isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
            },
            overwrite=True,
        )
        raise

    print(
        "PASS "
        f"baseline={baseline_metrics.top1:.4f}% "
        f"linear={linear_metrics.top1:.4f}% "
        f"residual={residual_metrics.top1:.4f}%",
        flush=True,
    )
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Checksums: {checksum_path}", flush=True)


if __name__ == "__main__":
    main()
