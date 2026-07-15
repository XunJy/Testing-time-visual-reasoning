#!/usr/bin/env python3
"""Audit and summarize one completed preregistered CLIP BirdMix study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

if __package__:
    from scripts.feature_adapter.run_clip_birdmix_study import (
        SourceTrial,
        StudyConfig,
        StudyConfigError,
        load_study_config,
    )
else:
    from run_clip_birdmix_study import (  # type: ignore[no-redef]
        SourceTrial,
        StudyConfig,
        StudyConfigError,
        load_study_config,
    )

_REQUIRED_RUN_FILES = (
    "config.json",
    "result.json",
    "run_complete.json",
    "source_validation.json",
    "adapter.pt",
    "cub_predictions.jsonl",
    "checksums.sha256",
)
_SHA256_LINE = re.compile(r"([0-9a-f]{64})  (.+)\Z")


class StudySummaryError(ValueError):
    """The preregistered study does not have one auditable run per cell."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudySummaryError(f"Cannot read {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StudySummaryError(f"{context} must be a JSON object: {path}")
    return value


def _canonical_config_digest(config: dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()[:10]


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


def _resolved_path(value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StudySummaryError(f"{context} must be a non-empty path string")
    return Path(value).expanduser().resolve()


def _validate_run_config(
    value: dict[str, Any],
    study: StudyConfig,
    trial: SourceTrial,
    *,
    seed: int,
    run_directory: Path,
) -> None:
    context = f"run config in {run_directory}"
    if value.get("experiment") != "06_feature_adapter_clip_multi_bird":
        raise StudySummaryError(f"{context} has the wrong experiment")
    if value.get("protocol") != "external-only-strict-cub-v1":
        raise StudySummaryError(f"{context} has the wrong protocol")
    if type(value.get("seed")) is not int or value["seed"] != seed:
        raise StudySummaryError(f"{context} has the wrong seed")
    if _resolved_path(value.get("source_config"), context=f"{context}.source_config") != (
        trial.source_config.resolve()
    ):
        raise StudySummaryError(f"{context} has the wrong source config")
    if value.get("source_config_sha256") != _sha256_file(trial.source_config):
        raise StudySummaryError(f"{context} has the wrong source config digest")
    if _resolved_path(
        value.get("cub_feature_cache"),
        context=f"{context}.cub_feature_cache",
    ) != study.common.cub_feature_cache.resolve():
        raise StudySummaryError(f"{context} has the wrong CUB feature cache")
    if value.get("training") != _expected_training(study, seed=seed):
        raise StudySummaryError(f"{context} does not match preregistered training settings")
    method = value.get("method")
    if not isinstance(method, dict) or method.get("hidden_dim") != study.common.hidden_dim:
        raise StudySummaryError(f"{context} has the wrong adapter hidden dimension")


def _validate_run_directory_name(
    run_directory: Path,
    *,
    kind: str,
    config: dict[str, Any],
) -> None:
    digest = _canonical_config_digest(config)
    pattern = re.compile(
        rf"\d{{8}}T\d{{6}}\.\d{{6}}Z-{re.escape(kind)}-{digest}\Z"
    )
    if pattern.fullmatch(run_directory.name) is None:
        raise StudySummaryError(
            f"Run directory name does not commit to its kind/config: {run_directory}"
        )


def _verify_checksums(run_directory: Path) -> None:
    checksum_path = run_directory / "checksums.sha256"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise StudySummaryError(f"Cannot read checksums: {checksum_path}: {error}") from error
    if not lines:
        raise StudySummaryError(f"Checksum manifest is empty: {checksum_path}")

    recorded: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise StudySummaryError(
                f"Malformed checksum line {line_number} in {checksum_path}"
            )
        expected, relative_name = match.groups()
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts or relative_name in recorded:
            raise StudySummaryError(
                f"Unsafe or duplicate checksum path in {checksum_path}: {relative_name}"
            )
        artifact = run_directory / relative
        if not artifact.is_file():
            raise StudySummaryError(f"Checksum references a missing file: {artifact}")
        actual = _sha256_file(artifact)
        if actual != expected:
            raise StudySummaryError(f"Checksum mismatch for {artifact}")
        recorded[relative_name] = expected

    required_entries = {name for name in _REQUIRED_RUN_FILES if name != "checksums.sha256"}
    missing_entries = sorted(required_entries - recorded.keys())
    if missing_entries:
        raise StudySummaryError(
            f"Checksum manifest does not cover required artifacts in {run_directory}: "
            f"{missing_entries}"
        )


def _finite_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StudySummaryError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise StudySummaryError(f"{context} must be finite")
    return number


def _metric_summary(value: object, *, context: str) -> dict[str, int | float]:
    if not isinstance(value, dict):
        raise StudySummaryError(f"{context} must be an object")
    total = value.get("total")
    correct = value.get("top1_correct")
    if type(total) is not int or total <= 0:
        raise StudySummaryError(f"{context}.total must be a positive integer")
    if type(correct) is not int or not 0 <= correct <= total:
        raise StudySummaryError(f"{context}.top1_correct is invalid")
    top1 = _finite_number(value.get("top1"), context=f"{context}.top1")
    expected = 100.0 * correct / total
    if not math.isclose(top1, expected, rel_tol=0.0, abs_tol=1e-10):
        raise StudySummaryError(f"{context}.top1 does not match its count")
    return {"total": total, "top1_correct": correct, "top1": top1}


def _summarize_result(
    value: dict[str, Any],
    *,
    trial_id: str,
    seed: int,
    kind: str,
    run_directory: Path,
) -> dict[str, Any]:
    baseline = _metric_summary(value.get("baseline"), context="result.baseline")
    adapted = _metric_summary(value.get("adapted"), context="result.adapted")
    if baseline["total"] != adapted["total"]:
        raise StudySummaryError("Baseline and adapted totals differ")
    gain = _finite_number(
        value.get("gain_top1_percentage_points"),
        context="result.gain_top1_percentage_points",
    )
    if not math.isclose(
        gain,
        float(adapted["top1"]) - float(baseline["top1"]),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise StudySummaryError("Stored gain does not match adapted minus baseline Top-1")

    comparison = value.get("comparison")
    transfers = comparison.get("transfers") if isinstance(comparison, dict) else None
    if not isinstance(transfers, dict):
        raise StudySummaryError("result.comparison.transfers must be an object")
    recovered = transfers.get("recovered")
    degraded = transfers.get("degraded")
    if type(recovered) is not int or recovered < 0:
        raise StudySummaryError("result.comparison.transfers.recovered is invalid")
    if type(degraded) is not int or degraded < 0:
        raise StudySummaryError("result.comparison.transfers.degraded is invalid")
    if recovered - degraded != adapted["top1_correct"] - baseline["top1_correct"]:
        raise StudySummaryError("Transfer counts disagree with the Top-1 correct counts")

    best_step = value.get("best_step")
    if type(best_step) is not int or best_step < 0:
        raise StudySummaryError("result.best_step must be a non-negative integer")
    strict_pass = value.get("strict_transfer_criterion_passed")
    if type(strict_pass) is not bool:
        raise StudySummaryError(
            "result.strict_transfer_criterion_passed must be boolean"
        )
    return {
        "trial_id": trial_id,
        "seed": seed,
        "kind": kind,
        "run_directory": run_directory.name,
        "baseline": baseline,
        "adapted": adapted,
        "gain_top1_percentage_points": gain,
        "recovered": recovered,
        "degraded": degraded,
        "best_step": best_step,
        "strict_transfer_criterion_passed": strict_pass,
    }


def _load_complete_run(
    run_directory: Path,
    study: StudyConfig,
    trial: SourceTrial,
    *,
    seed: int,
    kind: str,
) -> dict[str, Any]:
    config = _read_json_object(run_directory / "config.json", context="run config")
    _validate_run_directory_name(run_directory, kind=kind, config=config)
    _validate_run_config(config, study, trial, seed=seed, run_directory=run_directory)
    _verify_checksums(run_directory)

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
    return _summarize_result(
        result,
        trial_id=trial.trial_id,
        seed=seed,
        kind=kind,
        run_directory=run_directory,
    )


def _find_one_complete_run(
    runs_root: Path,
    study: StudyConfig,
    trial: SourceTrial,
    *,
    seed: int,
) -> dict[str, Any]:
    kind = f"{study.study_id}-{trial.trial_id}-seed{seed}"
    name_pattern = re.compile(
        rf"\d{{8}}T\d{{6}}\.\d{{6}}Z-{re.escape(kind)}-[0-9a-f]{{10}}\Z"
    )
    candidates = tuple(
        path
        for path in sorted(runs_root.iterdir())
        if path.is_dir() and name_pattern.fullmatch(path.name) is not None
    )
    complete_candidates = tuple(
        path
        for path in candidates
        if all((path / name).is_file() for name in _REQUIRED_RUN_FILES)
    )
    if not complete_candidates:
        incomplete = {
            path.name: [
                name for name in _REQUIRED_RUN_FILES if not (path / name).is_file()
            ]
            for path in candidates
        }
        raise StudySummaryError(
            f"Missing complete run for trial={trial.trial_id}, seed={seed}; "
            f"incomplete_candidates={incomplete}"
        )
    if len(complete_candidates) != 1:
        raise StudySummaryError(
            f"Duplicate complete runs for trial={trial.trial_id}, seed={seed}: "
            f"{[path.name for path in complete_candidates]}"
        )
    return _load_complete_run(
        complete_candidates[0],
        study,
        trial,
        seed=seed,
        kind=kind,
    )


def summarize_study(
    study_config: Path | str,
    runs_root: Path | str,
    *,
    project_root: Path | str,
) -> dict[str, Any]:
    """Return a deterministic audit summary without modifying run artifacts."""

    root = Path(project_root).expanduser().resolve()
    config_path = Path(study_config).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    runs = Path(runs_root).expanduser()
    if not runs.is_absolute():
        runs = root / runs
    if not runs.is_dir():
        raise StudySummaryError(f"Runs root does not exist: {runs}")
    try:
        study = load_study_config(config_path, root)
    except StudyConfigError as error:
        raise StudySummaryError(str(error)) from error

    run_rows = [
        _find_one_complete_run(runs, study, trial, seed=seed)
        for seed in study.seeds
        for trial in study.source_trials
    ]
    trial_rows = []
    for trial in study.source_trials:
        rows = [row for row in run_rows if row["trial_id"] == trial.trial_id]
        gains = [float(row["gain_top1_percentage_points"]) for row in rows]
        trial_rows.append(
            {
                "trial_id": trial.trial_id,
                "seed_count": len(rows),
                "gain_top1_percentage_points": {
                    "mean": statistics.fmean(gains),
                    "sample_std": statistics.stdev(gains) if len(gains) > 1 else None,
                    "min": min(gains),
                    "max": max(gains),
                },
                "all_seeds_positive_gain": all(gain > 0.0 for gain in gains),
                "all_seeds_strict_transfer_criterion_passed": all(
                    bool(row["strict_transfer_criterion_passed"]) for row in rows
                ),
            }
        )

    return {
        "schema_version": 1,
        "study_id": study.study_id,
        "study_config_sha256": _sha256_file(config_path),
        "selection_policy": "exactly-one-preregistered-run-per-trial-seed",
        "target_based_run_selection": False,
        "run_count": len(run_rows),
        "runs": run_rows,
        "trials": trial_rows,
    }


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=(
            project_root
            / "experiments/06_feature_adapter_clip_multi_bird/configs/"
            "clip_birdmix_preregistered_v1.json"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=project_root / "experiments/06_feature_adapter_clip_multi_bird/runs",
    )
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
    except StudySummaryError as error:
        raise SystemExit(f"Study summary failed: {error}") from error
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
