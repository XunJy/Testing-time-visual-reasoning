from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts.feature_adapter import summarize_clip_birdmix_study as summary


def _write_json(path: Path, value: Any) -> None:
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
    lines = [
        f"{_sha256(path)}  {path.relative_to(run_directory).as_posix()}"
        for path in sorted(run_directory.rglob("*"))
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (run_directory / "checksums.sha256").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _make_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_root = tmp_path / "project"
    config_root = (
        project_root / "experiments/06_feature_adapter_clip_multi_bird/configs"
    )
    config_root.mkdir(parents=True)
    source_rows = []
    for trial_id in ("inat-only", "birdmix-v1"):
        source_path = config_root / f"{trial_id}.json"
        _write_json(
            source_path,
            {
                "validation_taxon_fraction": 0.1,
                "sources": [
                    {
                        "dataset_id": f"synthetic-{trial_id}",
                        "root": f"data/{trial_id}",
                        "samples": f"data/{trial_id}/samples.jsonl",
                        "taxa": f"data/{trial_id}/taxa.jsonl",
                        "train_cache": f"cache/{trial_id}-train.pt",
                    }
                ],
            },
        )
        source_rows.append(
            {
                "trial_id": trial_id,
                "path": str(source_path.relative_to(project_root)),
            }
        )

    runs_root = project_root / "runs"
    runs_root.mkdir()
    study_path = config_root / "study.json"
    _write_json(
        study_path,
        {
            "schema_version": 1,
            "study_id": "synthetic-study",
            "seeds": [11, 12, 13],
            "source_configs": source_rows,
            "common": {
                "cub_data_root": "data",
                "cub_class_names": "data/cub-class-names.json",
                "birdnet_csv": "data/birdnet.csv",
                "cub_feature_cache": "cache/cub-test.pt",
                "model_cache_dir": "cache/models",
                "text_cache": "cache/text.pt",
                "runs_root": "runs",
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
    return project_root, study_path, runs_root


def _result(delta_correct: int, *, strict: bool) -> dict[str, Any]:
    total = 1_000
    baseline_correct = 600
    adapted_correct = baseline_correct + delta_correct
    degraded = 2 + max(-delta_correct, 0)
    recovered = degraded + delta_correct
    return {
        "baseline": {
            "total": total,
            "top1_correct": baseline_correct,
            "top5_correct": 900,
            "top1": 100.0 * baseline_correct / total,
            "top5": 90.0,
        },
        "adapted": {
            "total": total,
            "top1_correct": adapted_correct,
            "top5_correct": 901,
            "top1": 100.0 * adapted_correct / total,
            "top5": 90.1,
        },
        "gain_top1_percentage_points": (
            100.0 * adapted_correct / total - 100.0 * baseline_correct / total
        ),
        "comparison": {
            "transfers": {
                "total": total,
                "both_correct": baseline_correct - degraded,
                "recovered": recovered,
                "degraded": degraded,
                "both_wrong": total - (baseline_correct - degraded) - recovered - degraded,
            }
        },
        "strict_transfer_criterion_passed": strict,
        "best_step": 40,
    }


def _make_run(
    project_root: Path,
    runs_root: Path,
    *,
    trial_id: str,
    seed: int,
    delta_correct: int,
    strict: bool,
    timestamp_suffix: int,
) -> Path:
    source_config = (
        project_root
        / "experiments/06_feature_adapter_clip_multi_bird/configs"
        / f"{trial_id}.json"
    )
    config = {
        "experiment": "06_feature_adapter_clip_multi_bird",
        "protocol": "external-only-strict-cub-v1",
        "method": {"hidden_dim": 8},
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
        "cub_feature_cache": str((project_root / "cache/cub-test.pt").resolve()),
        "seed": seed,
    }
    kind = f"synthetic-study-{trial_id}-seed{seed}"
    name = (
        f"20260715T1000{timestamp_suffix:02d}.000000Z-{kind}-"
        f"{_config_digest(config)}"
    )
    run_directory = runs_root / name
    run_directory.mkdir()
    result = _result(delta_correct, strict=strict)
    _write_json(run_directory / "config.json", config)
    _write_json(run_directory / "result.json", result)
    _write_json(run_directory / "run_complete.json", {"state": "complete", "result": result})
    _write_json(run_directory / "source_validation.json", {"best_step": 40})
    (run_directory / "adapter.pt").write_bytes(b"synthetic adapter bytes")
    (run_directory / "cub_predictions.jsonl").write_text(
        '{"index":0,"baseline_correct":true,"adapted_correct":true}\n',
        encoding="utf-8",
    )
    _write_checksums(run_directory)
    return run_directory


def _make_all_runs(project_root: Path, runs_root: Path) -> dict[tuple[str, int], Path]:
    deltas = {
        "inat-only": (2, 4, 6),
        "birdmix-v1": (3, -1, 5),
    }
    result = {}
    ordinal = 0
    for seed_index, seed in enumerate((11, 12, 13)):
        for trial_id in ("inat-only", "birdmix-v1"):
            ordinal += 1
            delta = deltas[trial_id][seed_index]
            result[(trial_id, seed)] = _make_run(
                project_root,
                runs_root,
                trial_id=trial_id,
                seed=seed,
                delta_correct=delta,
                strict=delta >= 2,
                timestamp_suffix=ordinal,
            )
    return result


def test_summary_is_deterministic_and_reports_each_preregistered_cell(
    tmp_path: Path,
) -> None:
    project_root, study_path, runs_root = _make_project(tmp_path)
    _make_all_runs(project_root, runs_root)

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
    assert first["run_count"] == 6
    assert first["target_based_run_selection"] is False
    assert [(row["seed"], row["trial_id"]) for row in first["runs"]] == [
        (11, "inat-only"),
        (11, "birdmix-v1"),
        (12, "inat-only"),
        (12, "birdmix-v1"),
        (13, "inat-only"),
        (13, "birdmix-v1"),
    ]
    first_run = first["runs"][0]
    assert first_run["baseline"]["top1"] == 60.0
    assert first_run["adapted"]["top1"] == 60.2
    assert first_run["recovered"] == 4
    assert first_run["degraded"] == 2
    assert first_run["best_step"] == 40
    assert first_run["strict_transfer_criterion_passed"] is True

    by_trial = {row["trial_id"]: row for row in first["trials"]}
    inat = by_trial["inat-only"]
    assert math.isclose(inat["gain_top1_percentage_points"]["mean"], 0.4)
    assert math.isclose(inat["gain_top1_percentage_points"]["sample_std"], 0.2)
    assert math.isclose(inat["gain_top1_percentage_points"]["min"], 0.2)
    assert math.isclose(inat["gain_top1_percentage_points"]["max"], 0.6)
    assert inat["all_seeds_positive_gain"] is True
    assert inat["all_seeds_strict_transfer_criterion_passed"] is True
    birdmix = by_trial["birdmix-v1"]
    assert birdmix["all_seeds_positive_gain"] is False
    assert birdmix["all_seeds_strict_transfer_criterion_passed"] is False


def test_missing_complete_run_fails_closed(tmp_path: Path) -> None:
    project_root, study_path, runs_root = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root)
    (runs[("birdmix-v1", 13)] / "checksums.sha256").unlink()

    with pytest.raises(summary.StudySummaryError, match="Missing complete run"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_duplicate_complete_runs_fail_instead_of_selecting_by_target_gain(
    tmp_path: Path,
) -> None:
    project_root, study_path, runs_root = _make_project(tmp_path)
    _make_all_runs(project_root, runs_root)
    _make_run(
        project_root,
        runs_root,
        trial_id="inat-only",
        seed=11,
        delta_correct=100,
        strict=True,
        timestamp_suffix=50,
    )

    with pytest.raises(summary.StudySummaryError, match="Duplicate complete runs"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_artifact_tampering_fails_checksum_verification(tmp_path: Path) -> None:
    project_root, study_path, runs_root = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root)
    result_path = runs[("inat-only", 12)] / "result.json"
    result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(summary.StudySummaryError, match="Checksum mismatch"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)


def test_checksum_manifest_must_cover_key_audit_artifacts(tmp_path: Path) -> None:
    project_root, study_path, runs_root = _make_project(tmp_path)
    runs = _make_all_runs(project_root, runs_root)
    checksum_path = runs[("inat-only", 13)] / "checksums.sha256"
    lines = [
        line
        for line in checksum_path.read_text(encoding="utf-8").splitlines()
        if not line.endswith("  adapter.pt")
    ]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(summary.StudySummaryError, match="does not cover required"):
        summary.summarize_study(study_path, runs_root, project_root=project_root)
