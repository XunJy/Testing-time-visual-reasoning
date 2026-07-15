from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.feature_adapter import summarize_clip_birdmix_v2_study as summary
from ttvr.data.cub import CUB_IMAGE_COUNT, CUB_TEST_COUNT
from ttvr.data.cub_taxonomy import (
    CUB_SCIENTIFIC_NAME_OVERRIDES,
    build_cub_birdnet_crosswalk,
    read_cub_class_names,
)
from ttvr.metrics import exact_mcnemar_test

EXPERIMENT_ID = "07_feature_adapter_clip_birdmix_v2_cub"
PROTOCOL = "external-only-strict-cub-birdmix-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_CLASS_NAMES = PROJECT_ROOT / "data/fudd_official/cub_class_names.json"

_BIRDNET_COLUMNS = (
    "birdnet_id",
    "scientific_name",
    "common_name",
    "common_name_alt",
    "taxon_group",
    "record_type",
    "scientific_name_aliases",
    "common_name_aliases",
    "common_name_en",
)
_ADDITIONAL_SPLIT_TAXA = (
    ("Larus smithsonianus", "American Herring Gull"),
    ("Colibri cyanotus", "Lesser Violetear"),
    ("Setophaga petechia", "Mangrove Warbler"),
    ("Troglodytes musculus", "Southern House Wren"),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_digest(value: dict[str, Any]) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


def _write_checksums(run_directory: Path) -> None:
    checksum_path = run_directory / "checksums.sha256"
    checksum_path.unlink(missing_ok=True)
    lines = [
        f"{_sha256(path)}  {path.relative_to(run_directory).as_posix()}"
        for path in sorted(run_directory.rglob("*"))
        if path.is_file() and path != checksum_path
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _birdnet_row(
    birdnet_id: str,
    scientific_name: str,
    common_name: str,
) -> dict[str, str]:
    return {
        "birdnet_id": birdnet_id,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "common_name_alt": "",
        "taxon_group": "Aves",
        "record_type": "species",
        "scientific_name_aliases": "",
        "common_name_aliases": "",
        "common_name_en": common_name,
    }


def _write_complete_birdnet_csv(path: Path, class_names_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for class_id, class_name in enumerate(read_cub_class_names(class_names_path)):
        if class_name == "Sayornis":
            continue
        scientific_name = CUB_SCIENTIFIC_NAME_OVERRIDES.get(
            class_name,
            f"Syntheticus{class_id} example",
        )
        common_name = (
            f"Reviewed canonical name {class_id}"
            if class_name in CUB_SCIENTIFIC_NAME_OVERRIDES
            else class_name
        )
        rows.append(_birdnet_row(f"BN{class_id:05d}", scientific_name, common_name))
    for offset, (scientific_name, common_name) in enumerate(_ADDITIONAL_SPLIT_TAXA):
        rows.append(_birdnet_row(f"BNS{offset:04d}", scientific_name, common_name))
    for offset, (scientific_name, common_name) in enumerate(
        (
            ("Sayornis nigricans", "Black Phoebe"),
            ("Sayornis phoebe", "Eastern Phoebe"),
            ("Sayornis saya", "Say's Phoebe"),
        )
    ):
        rows.append(_birdnet_row(f"BNG{offset:04d}", scientific_name, common_name))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_BIRDNET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_canonical_cub_metadata(root: Path) -> None:
    dataset_root = root / "CUB_200_2011"
    dataset_root.mkdir(parents=True)
    (dataset_root / "classes.txt").write_text(
        "".join(
            f"{class_id} {class_id:03d}.Synthetic_CUB_class_{class_id:03d}\n"
            for class_id in range(1, 201)
        ),
        encoding="utf-8",
    )
    images: list[str] = []
    labels: list[str] = []
    splits: list[str] = []
    for index in range(CUB_IMAGE_COUNT):
        image_id = index + 1
        label = index % 200
        if index < CUB_TEST_COUNT:
            relative_path = f"synthetic/class-{label:03d}/image-{index:04d}.jpg"
            split = 0
        else:
            relative_path = f"synthetic-train/class-{label:03d}/image-{index:05d}.jpg"
            split = 1
        images.append(f"{image_id} {relative_path}\n")
        labels.append(f"{image_id} {label + 1}\n")
        splits.append(f"{image_id} {split}\n")
    (dataset_root / "images.txt").write_text("".join(images), encoding="utf-8")
    (dataset_root / "image_class_labels.txt").write_text(
        "".join(labels),
        encoding="utf-8",
    )
    (dataset_root / "train_test_split.txt").write_text(
        "".join(splits),
        encoding="utf-8",
    )


def _make_project(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Path]]:
    project_root = tmp_path / "project"
    config_root = project_root / "experiments" / EXPERIMENT_ID / "configs"
    config_root.mkdir(parents=True)
    source_root = project_root / "data/source"
    source_root.mkdir(parents=True)
    class_names_path = project_root / "data/cub-class-names.json"
    shutil.copyfile(PROJECT_CLASS_NAMES, class_names_path)
    _write_canonical_cub_metadata(project_root / "data")
    _write_complete_birdnet_csv(project_root / "data/birdnet.csv", class_names_path)
    manifest_root = project_root / "manifests/source-a"
    manifest_root.mkdir(parents=True)
    inputs = {
        "source_metadata": manifest_root / "source.json",
        "samples": manifest_root / "samples.jsonl",
        "taxa": manifest_root / "taxa.jsonl",
        "train_cache": project_root / "cache/source-a-train.pt",
        "validation_cache": project_root / "cache/source-a-validation.pt",
        "cub_cache": project_root / "cache/cub-test.pt",
    }
    for key, path in inputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {key}\n".encode())

    source_path = config_root / "birdmix_v2.json"
    _write_json(
        source_path,
        {
            "validation_taxon_fraction": 0.1,
            "duplicate_audit": {
                "exact_sha256_policy": "drop-later-source",
                "perceptual_hash_policy": "report-only",
                "perceptual_hamming_threshold": 4,
            },
            "sources": [
                {
                    "dataset_id": "source-a",
                    "root": str(source_root.relative_to(project_root)),
                    "samples": str(inputs["samples"].relative_to(project_root)),
                    "taxa": str(inputs["taxa"].relative_to(project_root)),
                    "train_cache": str(inputs["train_cache"].relative_to(project_root)),
                    "validation_cache": str(inputs["validation_cache"].relative_to(project_root)),
                }
            ],
        },
    )
    runs_root = project_root / "runs" / EXPERIMENT_ID
    runs_root.mkdir(parents=True)
    study_path = config_root / "study.json"
    _write_json(
        study_path,
        {
            "schema_version": 2,
            "experiment_id": EXPERIMENT_ID,
            "protocol": PROTOCOL,
            "study_id": "synthetic-v2",
            "seeds": [11, 12, 13],
            "source_configs": [
                {
                    "trial_id": "all-sources",
                    "path": str(source_path.relative_to(project_root)),
                }
            ],
            "common": {
                "cub_data_root": "data",
                "cub_class_names": "data/cub-class-names.json",
                "birdnet_csv": "data/birdnet.csv",
                "cub_feature_cache": str(inputs["cub_cache"].relative_to(project_root)),
                "model_cache_dir": "cache/models",
                "text_cache": "cache/text.pt",
                "runs_root": str(runs_root.relative_to(project_root)),
                "steps": 100,
                "validation_interval": 10,
                "patience_intervals": 3,
                "batch_size": 16,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "identity_weight": 0.1,
                "hidden_dim": 8,
                "device": "cpu",
            },
        },
    )
    return project_root, study_path, runs_root, inputs


def _prediction_rows(
    class_names: tuple[str, ...],
    *,
    delta_correct: int,
    baseline_top1_correct: int = 3_671,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    degraded = 2
    recovered = degraded + delta_correct
    rows: list[dict[str, Any]] = []
    labels: list[int] = []
    baseline_flags: list[bool] = []
    adapted_flags: list[bool] = []
    for index in range(CUB_TEST_COUNT):
        label = index % len(class_names)
        wrong = tuple((label + offset) % len(class_names) for offset in range(1, 6))
        baseline_correct = index < baseline_top1_correct
        adapted_correct = baseline_correct
        if index < degraded:
            adapted_correct = False
        elif baseline_top1_correct <= index < baseline_top1_correct + recovered:
            adapted_correct = True

        baseline_top5_hit = index < 5_000
        adapted_top5_hit = index < 5_001
        baseline_top5 = (
            (label, *wrong[:4])
            if baseline_correct
            else ((wrong[0], label, *wrong[1:4]) if baseline_top5_hit else wrong)
        )
        adapted_top5 = (
            (label, *wrong[:4])
            if adapted_correct
            else ((wrong[0], label, *wrong[1:4]) if adapted_top5_hit else wrong)
        )
        rows.append(
            {
                "index": index,
                "image_id": index + 1,
                "relative_path": f"synthetic/class-{label:03d}/image-{index:04d}.jpg",
                "label": label,
                "label_name": class_names[label],
                "baseline_top5": list(baseline_top5),
                "baseline_top5_names": [class_names[value] for value in baseline_top5],
                "adapted_top5": list(adapted_top5),
                "adapted_top5_names": [class_names[value] for value in adapted_top5],
                "baseline_correct": baseline_correct,
                "adapted_correct": adapted_correct,
            }
        )
        labels.append(label)
        baseline_flags.append(baseline_correct)
        adapted_flags.append(adapted_correct)
    return (
        rows,
        torch.tensor(labels, dtype=torch.int64),
        torch.tensor(baseline_flags, dtype=torch.bool),
        torch.tensor(adapted_flags, dtype=torch.bool),
    )


def _result(
    delta_correct: int,
    *,
    seed: int,
    class_names: tuple[str, ...],
    baseline_top1_correct: int = 3_671,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows, labels, baseline_flags, adapted_flags = _prediction_rows(
        class_names,
        delta_correct=delta_correct,
        baseline_top1_correct=baseline_top1_correct,
    )
    baseline_correct = int(baseline_flags.sum().item())
    adapted_correct = int(adapted_flags.sum().item())
    both_correct = int((baseline_flags & adapted_flags).sum().item())
    recovered = int((~baseline_flags & adapted_flags).sum().item())
    degraded = int((baseline_flags & ~adapted_flags).sum().item())
    both_wrong = CUB_TEST_COUNT - both_correct - recovered - degraded

    def metric(correct: int, top5_correct: int) -> dict[str, int | float]:
        return {
            "total": CUB_TEST_COUNT,
            "top1_correct": correct,
            "top5_correct": top5_correct,
            "top1": 100.0 * correct / CUB_TEST_COUNT,
            "top5": 100.0 * top5_correct / CUB_TEST_COUNT,
        }

    def percent(count: int) -> float:
        return 100.0 * count / CUB_TEST_COUNT

    species_cluster = summary._species_cluster_bootstrap_from_predictions(
        baseline_flags,
        adapted_flags,
        labels,
        seed=seed,
    )
    result = {
        "baseline": metric(baseline_correct, 5_000),
        "adapted": metric(adapted_correct, 5_001),
        "gain_top1_percentage_points": 100.0 * delta_correct / CUB_TEST_COUNT,
        "comparison": {
            "transfers": {
                "total": CUB_TEST_COUNT,
                "both_correct": both_correct,
                "recovered": recovered,
                "degraded": degraded,
                "both_wrong": both_wrong,
                "both_correct_percent": percent(both_correct),
                "recovered_percent": percent(recovered),
                "degraded_percent": percent(degraded),
                "both_wrong_percent": percent(both_wrong),
            },
            "mcnemar": exact_mcnemar_test(baseline_flags, adapted_flags).to_dict(),
            "species_cluster_bootstrap": species_cluster,
        },
        "strict_transfer_criterion_passed": (
            float(species_cluster["ci95_low_percentage_points"]) > 0.0
        ),
        "best_step": 40,
    }
    return result, rows


def _make_run(
    project_root: Path,
    runs_root: Path,
    inputs: dict[str, Path],
    *,
    seed: int,
    delta_correct: int,
    ordinal: int,
    baseline_top1_correct: int = 3_671,
) -> Path:
    source_config = project_root / f"experiments/{EXPERIMENT_ID}/configs/birdmix_v2.json"
    class_names_path = project_root / "data/cub-class-names.json"
    birdnet_csv = project_root / "data/birdnet.csv"
    class_names = read_cub_class_names(class_names_path)
    crosswalk = build_cub_birdnet_crosswalk(class_names_path, birdnet_csv)
    source_snapshot = {
        "dataset_id": "source-a",
        "source_metadata": str(inputs["source_metadata"].resolve()),
        "source_metadata_sha256": _sha256(inputs["source_metadata"]),
        "samples": str(inputs["samples"].resolve()),
        "samples_sha256": _sha256(inputs["samples"]),
        "taxa": str(inputs["taxa"].resolve()),
        "taxa_sha256": _sha256(inputs["taxa"]),
        "train_cache": str(inputs["train_cache"].resolve()),
        "train_cache_sha256": _sha256(inputs["train_cache"]),
        "validation_cache": str(inputs["validation_cache"].resolve()),
        "validation_cache_sha256": _sha256(inputs["validation_cache"]),
    }
    config = {
        "experiment": EXPERIMENT_ID,
        "protocol": PROTOCOL,
        "model": "OpenAI CLIP ViT-L/14@336px fp32",
        "model_runtime": {
            "cache_identity": (
                "openai-clip:ViT-L/14@336px@a1d071733d7111c9c014f024669f959182114e33"
            ),
            "clip_distribution": "clip",
            "clip_version": "1.0",
            "clip_repository_url": "https://github.com/openai/CLIP.git",
            "clip_commit": "a1d071733d7111c9c014f024669f959182114e33",
            "checkpoint_path": str((project_root / "cache/models/ViT-L-14-336px.pt").resolve()),
            "checkpoint_sha256": (
                "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
            ),
            "checkpoint_size_bytes": 934_088_680,
        },
        "method": {
            "feature_dim": 768,
            "hidden_dim": 8,
            "architecture": "normalize(x + W_up(GELU(W_down(x))))",
        },
        "training": {
            "steps": 100,
            "validation_interval": 10,
            "patience_intervals": 3,
            "batch_size": 16,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "identity_weight": 0.1,
            "logit_scale": 100.0,
            "seed": seed,
        },
        "source_config": str(source_config.resolve()),
        "source_config_sha256": _sha256(source_config),
        "validation_taxon_fraction": 0.1,
        "duplicate_audit": {
            "exact_sha256_policy": "drop-later-source",
            "perceptual_hash_policy": "report-only",
            "perceptual_hamming_threshold": 4,
        },
        "disabled_source_placeholders": [],
        "sources": [source_snapshot],
        "cub_feature_cache": str(inputs["cub_cache"].resolve()),
        "cub_feature_cache_sha256": _sha256(inputs["cub_cache"]),
        "cub_crosswalk_digest": crosswalk.digest,
        "source_code_sha256": "2" * 64,
        "seed": seed,
    }
    kind = f"synthetic-v2-all-sources-seed{seed}"
    name = f"20260715T1000{ordinal:02d}.000000Z-{kind}-{_config_digest(config)}"
    run_directory = runs_root / name
    run_directory.mkdir()
    result, prediction_rows = _result(
        delta_correct,
        seed=seed,
        class_names=class_names,
        baseline_top1_correct=baseline_top1_correct,
    )
    _write_json(run_directory / "config.json", config)
    _write_json(run_directory / "result.json", result)
    _write_json(run_directory / "run_complete.json", {"state": "complete", "result": result})
    _write_json(
        run_directory / "source_validation.json",
        {"best_step": 40, "refit_steps": 40},
    )
    _write_json(
        run_directory / "cub_taxonomy_crosswalk.json",
        {
            "protocol": crosswalk.protocol,
            "birdnet_csv_sha256": crosswalk.birdnet_csv_sha256,
            "digest": crosswalk.digest,
            "status_counts": crosswalk.status_counts,
            "entries": [asdict(entry) for entry in crosswalk.entries],
        },
    )
    (run_directory / "adapter.pt").write_bytes(f"adapter seed {seed}".encode())
    source_features = torch.zeros((2, 768), dtype=torch.float32)
    source_features[0, 0] = 1.0
    source_features[1, 1] = 1.0
    target_features = torch.zeros((200, 768), dtype=torch.float32)
    target_features[torch.arange(200), torch.arange(200)] = 1.0
    legacy_serialization = seed == 12
    torch.save(
        {
            "taxon_ids": ("taxon:a", "taxon:b"),
            "features": source_features,
        },
        run_directory / "source_text_prototypes.pt",
        _use_new_zipfile_serialization=not legacy_serialization,
    )
    torch.save(
        {
            "class_names": class_names,
            "prompts": [f"a photo of a {name}." for name in class_names],
            "features": target_features,
        },
        run_directory / "target_text_prototypes.pt",
        _use_new_zipfile_serialization=not legacy_serialization,
    )
    (run_directory / "cub_predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in prediction_rows
        ),
        encoding="utf-8",
    )
    _write_checksums(run_directory)
    return run_directory


def _make_all_runs(
    project_root: Path,
    runs_root: Path,
    inputs: dict[str, Path],
    *,
    baseline_top1_correct: int = 3_671,
) -> dict[int, Path]:
    result = {}
    pairs = zip((11, 12, 13), (2, 4, 6), strict=True)
    for ordinal, (seed, delta) in enumerate(pairs, start=1):
        result[seed] = _make_run(
            project_root,
            runs_root,
            inputs,
            seed=seed,
            delta_correct=delta,
            ordinal=ordinal,
            baseline_top1_correct=baseline_top1_correct,
        )
    return result


def _rewrite_config(
    run_directory: Path,
    change: Callable[[dict[str, Any]], None],
) -> Path:
    config_path = run_directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    change(config)
    _write_json(config_path.with_suffix(".new"), config)
    config_path.unlink()
    config_path.with_suffix(".new").rename(config_path)
    new_name = f"{run_directory.name.rsplit('-', 1)[0]}-{_config_digest(config)}"
    rewritten = run_directory.rename(run_directory.with_name(new_name))
    _write_checksums(rewritten)
    return rewritten


def test_v2_summary_reports_three_seed_mean_and_student_t_ci(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    assert _sha256(runs[11] / "source_text_prototypes.pt") != _sha256(
        runs[12] / "source_text_prototypes.pt"
    )

    first = summary.summarize_study(
        study_path,
        runs_root,
        project_root=project_root,
    )
    second = summary.summarize_study(
        study_path,
        runs_root,
        project_root=project_root,
    )

    assert first == second
    assert first["run_count"] == 3
    assert first["target_based_run_selection"] is False
    assert [row["seed"] for row in first["runs"]] == [11, 12, 13]
    trial = first["trials"][0]
    baseline = 100.0 * 3_671 / CUB_TEST_COUNT
    assert trial["baseline_top1"]["mean"] == baseline
    assert trial["baseline_top1"]["ci95_low"] == baseline
    assert math.isclose(
        trial["adapted_top1"]["mean"],
        100.0 * (3_671 + 4) / CUB_TEST_COUNT,
    )
    gain = trial["gain_top1"]
    expected_mean = 100.0 * 4 / CUB_TEST_COUNT
    expected_std = 100.0 * 2 / CUB_TEST_COUNT
    assert math.isclose(gain["mean"], expected_mean)
    assert math.isclose(gain["sample_standard_deviation"], expected_std)
    margin = 4.302652729911275 * expected_std / math.sqrt(3.0)
    assert math.isclose(gain["ci95_low"], expected_mean - margin)
    assert math.isclose(gain["ci95_high"], expected_mean + margin)
    assert "Student t" in gain["ci95_method"]
    assert first["audit"]["seed_invariant_run_configs"] is True
    assert first["audit"]["cub_predictions_recomputed_from_exactly_5794_rows"] is True
    assert first["audit"]["verified_input_file_count"] == 6


def test_missing_and_incomplete_runs_fail_closed(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    shutil.rmtree(runs[13])
    with pytest.raises(summary.StudySummaryError, match="Missing run"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)

    runs[13] = _make_run(
        project_root,
        runs_root,
        inputs,
        seed=13,
        delta_correct=6,
        ordinal=3,
    )
    (runs[13] / "target_text_prototypes.pt").unlink()
    with pytest.raises(summary.StudySummaryError, match="Incomplete run"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_duplicate_is_rejected_even_when_one_candidate_is_incomplete(
    tmp_path: Path,
) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    _make_all_runs(project_root, runs_root, inputs)
    duplicate = _make_run(
        project_root,
        runs_root,
        inputs,
        seed=11,
        delta_correct=100,
        ordinal=9,
    )
    (duplicate / "adapter.pt").unlink()

    with pytest.raises(summary.StudySummaryError, match="Duplicate runs"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda value: value.__setitem__("experiment", "wrong"), "wrong experiment"),
        (lambda value: value.__setitem__("protocol", "wrong"), "wrong protocol"),
        (lambda value: value.__setitem__("model", "wrong"), "wrong model"),
        (
            lambda value: value["model_runtime"].__setitem__("clip_commit", "0" * 40),
            "locked CLIP runtime identity",
        ),
        (lambda value: value.__setitem__("seed", 999), "wrong seed"),
        (
            lambda value: value.__setitem__("source_config_sha256", "0" * 64),
            "source config digest",
        ),
        (
            lambda value: value["training"].__setitem__("batch_size", 999),
            "common training settings",
        ),
        (
            lambda value: value.__setitem__(
                "cub_feature_cache", str(Path(value["cub_feature_cache"]).with_name("wrong.pt"))
            ),
            "wrong CUB feature cache",
        ),
    ],
)
def test_run_identity_mismatches_fail_closed(
    tmp_path: Path,
    change: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    _rewrite_config(runs[11], change)

    with pytest.raises(summary.StudySummaryError, match=message):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_seed_invariant_config_and_prototype_content_are_enforced(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    _rewrite_config(
        runs[12],
        lambda value: value.__setitem__("source_code_sha256", "3" * 64),
    )
    with pytest.raises(summary.StudySummaryError, match="differ beyond seed"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)

    project_root, study_path, runs_root, inputs = _make_project(tmp_path / "second")
    runs = _make_all_runs(project_root, runs_root, inputs)
    prototype_path = runs[13] / "source_text_prototypes.pt"
    payload = torch.load(prototype_path, map_location="cpu", weights_only=True)
    payload["features"] = payload["features"].clone()
    payload["features"][0].zero_()
    payload["features"][0, 2] = 1.0
    torch.save(payload, prototype_path)
    _write_checksums(runs[13])
    with pytest.raises(summary.StudySummaryError, match="prototypes differ"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_tampered_predictions_fail_after_checksums_are_rewritten(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    path = runs[11] / "cub_predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    label = row["label"]
    tail = [value for value in row["adapted_top5"] if value != label]
    row["adapted_top5"] = [label, *tail]
    class_names = read_cub_class_names(project_root / "data/cub-class-names.json")
    row["adapted_top5_names"] = [class_names[value] for value in row["adapted_top5"]]
    row["adapted_correct"] = True
    lines[0] = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_checksums(runs[11])

    with pytest.raises(summary.StudySummaryError, match="result.adapted"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


@pytest.mark.parametrize("case", ["identity", "baseline_top5"])
def test_cross_seed_prediction_rows_must_be_identical(
    tmp_path: Path,
    case: str,
) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    path = runs[12] / "cub_predictions.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[100])
    if case == "identity":
        row["image_id"] = 99_999
        message = "canonical CUB test sample"
    else:
        row["baseline_top5"][1], row["baseline_top5"][2] = (
            row["baseline_top5"][2],
            row["baseline_top5"][1],
        )
        row["baseline_top5_names"][1], row["baseline_top5_names"][2] = (
            row["baseline_top5_names"][2],
            row["baseline_top5_names"][1],
        )
        message = "baseline Top-5 rows differ"
    lines[100] = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_checksums(runs[12])

    with pytest.raises(summary.StudySummaryError, match=message):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("index", "exact dataset index"),
        ("image_id", "canonical CUB test sample"),
        ("relative_path", "canonical CUB test sample"),
        ("label", "canonical CUB test target"),
    ],
)
def test_all_three_seeds_cannot_share_tampered_cub_identity(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    class_names = read_cub_class_names(project_root / "data/cub-class-names.json")
    for run_directory in runs.values():
        path = run_directory / "cub_predictions.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        row = json.loads(lines[100])
        if field == "index":
            row[field] = 999
        elif field == "image_id":
            row[field] = 99_999
        elif field == "relative_path":
            row[field] = "synthetic/forged/image.jpg"
        else:
            row[field] = (row[field] + 1) % 200
            row["label_name"] = class_names[row[field]]
        lines[100] = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _write_checksums(run_directory)

    with pytest.raises(summary.StudySummaryError, match=message):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_self_consistent_historical_baseline_count_tampering_fails(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    _make_all_runs(
        project_root,
        runs_root,
        inputs,
        baseline_top1_correct=3_670,
    )

    with pytest.raises(summary.StudySummaryError, match="historical CUB baseline Top-1"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_target_prototypes_must_use_the_locked_class_names(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    path = runs[13] / "target_text_prototypes.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    arbitrary = tuple(f"arbitrary-target-{index:03d}" for index in range(200))
    payload["class_names"] = arbitrary
    payload["prompts"] = tuple(f"a photo of a {name}." for name in arbitrary)
    torch.save(payload, path)
    _write_checksums(runs[13])

    with pytest.raises(summary.StudySummaryError, match="locked CUB class names"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_empty_self_consistent_crosswalk_is_rejected(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    original = json.loads((runs[11] / "cub_taxonomy_crosswalk.json").read_text(encoding="utf-8"))
    empty_payload = {
        "protocol": original["protocol"],
        "birdnet_csv_sha256": original["birdnet_csv_sha256"],
        "entries": [],
    }
    empty_digest = hashlib.sha256(
        json.dumps(
            empty_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    rewritten = _rewrite_config(
        runs[11],
        lambda value: value.__setitem__("cub_crosswalk_digest", empty_digest),
    )
    _write_json(
        rewritten / "cub_taxonomy_crosswalk.json",
        {
            **empty_payload,
            "digest": empty_digest,
            "status_counts": {},
        },
    )
    _write_checksums(rewritten)

    with pytest.raises(summary.StudySummaryError, match="canonical CUB crosswalk"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_refit_step_and_current_input_digest_are_enforced(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root, inputs)
    validation_path = runs[12] / "source_validation.json"
    value = json.loads(validation_path.read_text(encoding="utf-8"))
    value["refit_steps"] = 39
    validation_path.unlink()
    _write_json(validation_path, value)
    _write_checksums(runs[12])
    with pytest.raises(summary.StudySummaryError, match="refit_steps"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)

    project_root, study_path, runs_root, inputs = _make_project(tmp_path / "second")
    _make_all_runs(project_root, runs_root, inputs)
    inputs["train_cache"].write_bytes(b"changed after all runs")
    with pytest.raises(summary.StudySummaryError, match="digest no longer matches"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("keys", "wrong keys"),
        ("metadata", "invalid string"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("finite", "non-finite"),
        ("norm", "unit-normalised"),
    ],
)
def test_prototype_content_digest_validates_metadata_and_tensor(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    features = torch.zeros((2, 768), dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 1] = 1.0
    payload: dict[str, Any] = {
        "taxon_ids": ("taxon:a", "taxon:b"),
        "features": features,
    }
    if case == "keys":
        payload["extra"] = True
    elif case == "metadata":
        payload["taxon_ids"] = ("taxon:a", "")
    elif case == "shape":
        payload["features"] = features[:, :-1]
    elif case == "dtype":
        payload["features"] = features.double()
    elif case == "finite":
        payload["features"] = features.clone()
        payload["features"][0, 0] = float("nan")
    elif case == "norm":
        payload["features"] = features * 2.0
    path = tmp_path / f"{case}.pt"
    torch.save(payload, path)

    with pytest.raises(summary.StudySummaryError, match=message):
        summary._prototype_content_digest(path, kind="source")


def test_target_prototype_digest_locks_class_count_and_prompt_template(
    tmp_path: Path,
) -> None:
    class_names = tuple(f"class-{index:03d}" for index in range(200))
    features = torch.zeros((200, 768), dtype=torch.float32)
    features[torch.arange(200), torch.arange(200)] = 1.0
    path = tmp_path / "target.pt"
    torch.save(
        {
            "class_names": class_names,
            "prompts": ["wrong prompt", *[f"a photo of a {name}." for name in class_names[1:]]],
            "features": features,
        },
        path,
    )

    with pytest.raises(summary.StudySummaryError, match="locked template"):
        summary._prototype_content_digest(path, kind="target")


def test_summary_output_is_write_once_and_commits_to_digest(tmp_path: Path) -> None:
    project_root, study_path, runs_root, inputs = _make_project(tmp_path)
    _make_all_runs(project_root, runs_root, inputs)
    value = summary.summarize_study(study_path, runs_root, project_root=project_root)
    moment = dt.datetime(2026, 7, 15, 12, 0, tzinfo=dt.timezone.utc)

    output = summary.write_summary(
        value,
        output_root=tmp_path / "summaries",
        timestamp=moment,
    )
    assert output.name == "summary.json"
    assert output.parent.name.startswith("20260715T120000.000000Z-summary-")
    assert json.loads(output.read_text(encoding="utf-8")) == value
    checksum = (output.parent / "checksums.sha256").read_text(encoding="utf-8")
    assert checksum == f"{_sha256(output)}  summary.json\n"
    with pytest.raises(summary.StudySummaryError, match="overwrite summary directory"):
        summary.write_summary(
            value,
            output_root=tmp_path / "summaries",
            timestamp=moment,
        )

    explicit = tmp_path / "explicit.json"
    summary.write_summary(value, output=explicit)
    with pytest.raises(summary.StudySummaryError, match="overwrite summary"):
        summary.write_summary(value, output=explicit)
