#!/usr/bin/env python3
"""Validate and launch the immutable CLIP BirdMix-v2 trial matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SLUG = re.compile(r"[a-z0-9][a-z0-9-]*\Z")
_EXPERIMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_V2_EXPERIMENT_ID = "07_feature_adapter_clip_birdmix_v2_cub"
_V2_PROTOCOL = "external-only-strict-cub-birdmix-v2"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "experiment_id",
    "protocol",
    "study_id",
    "seeds",
    "source_configs",
    "common",
}
_SOURCE_TRIAL_KEYS = {"trial_id", "path"}
_COMMON_PATH_KEYS = {
    "cub_data_root",
    "cub_class_names",
    "birdnet_csv",
    "cub_feature_cache",
    "model_cache_dir",
    "text_cache",
    "runs_root",
}
_COMMON_POSITIVE_INTEGER_KEYS = {
    "steps",
    "validation_interval",
    "patience_intervals",
    "batch_size",
    "hidden_dim",
}
_COMMON_FLOAT_KEYS = {"learning_rate", "weight_decay", "identity_weight"}
_COMMON_KEYS = _COMMON_PATH_KEYS | _COMMON_POSITIVE_INTEGER_KEYS | _COMMON_FLOAT_KEYS | {
    "device"
}
_ENABLED_SOURCE_KEYS = {
    "enabled",
    "dataset_id",
    "root",
    "samples",
    "taxa",
    "train_cache",
    "validation_cache",
}
_DISABLED_SOURCE_KEYS = {"enabled", "dataset_id", "reason"}


class StudyConfigError(ValueError):
    """The v2 study cannot be launched without completing its locked inputs."""


@dataclass(frozen=True, slots=True)
class SourceTrial:
    trial_id: str
    source_config: Path


@dataclass(frozen=True, slots=True)
class CommonInputs:
    cub_data_root: Path
    cub_class_names: Path
    birdnet_csv: Path
    cub_feature_cache: Path
    model_cache_dir: Path
    text_cache: Path
    runs_root: Path
    steps: int
    validation_interval: int
    patience_intervals: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    identity_weight: float
    hidden_dim: int
    device: str


@dataclass(frozen=True, slots=True)
class StudyConfig:
    experiment_id: str
    protocol: str
    study_id: str
    seeds: tuple[int, ...]
    source_trials: tuple[SourceTrial, ...]
    common: CommonInputs


@dataclass(frozen=True, slots=True)
class TrialCommand:
    trial_id: str
    seed: int
    argv: tuple[str, ...]


def _require_exact_keys(value: dict[str, Any], expected: set[str], *, context: str) -> None:
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise StudyConfigError(
            f"{context} keys do not match schema; missing={missing}, unknown={unknown}"
        )


def _read_json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyConfigError(f"Cannot read {context}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise StudyConfigError(f"{context} must be a JSON object: {path}")
    return value


def _resolve(project_root: Path, value: object, *, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StudyConfigError(f"{context} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def _validated_source_rows(trial: SourceTrial) -> tuple[dict[str, Any], ...]:
    value = _read_json_object(
        trial.source_config,
        context=f"source config for {trial.trial_id}",
    )
    if set(value) != {"validation_taxon_fraction", "duplicate_audit", "sources"}:
        _require_exact_keys(
            value,
            {"validation_taxon_fraction", "duplicate_audit", "sources"},
            context=f"source config for {trial.trial_id}",
        )
    validation_fraction = value["validation_taxon_fraction"]
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise StudyConfigError("validation_taxon_fraction must be between zero and one")
    duplicate = value["duplicate_audit"]
    if not isinstance(duplicate, dict):
        raise StudyConfigError("duplicate_audit must be an object")
    _require_exact_keys(
        duplicate,
        {
            "exact_sha256_policy",
            "perceptual_hash_policy",
            "perceptual_hamming_threshold",
        },
        context="duplicate_audit",
    )
    if duplicate["exact_sha256_policy"] != "drop-later-source":
        raise StudyConfigError("v2 locks exact_sha256_policy to drop-later-source")
    if duplicate["perceptual_hash_policy"] != "report-only":
        raise StudyConfigError("perceptual hash candidates must remain report-only")
    threshold = duplicate["perceptual_hamming_threshold"]
    if type(threshold) is not int or not 0 <= threshold <= 8:
        raise StudyConfigError("perceptual_hamming_threshold must be in [0, 8]")

    rows = value["sources"]
    if not isinstance(rows, list) or not rows:
        raise StudyConfigError("source config must contain source rows")
    enabled_count = 0
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context = f"{trial.trial_id}.sources[{index}]"
        if not isinstance(row, dict):
            raise StudyConfigError(f"{context} must be an object")
        enabled = row.get("enabled", True)
        if type(enabled) is not bool:
            raise StudyConfigError(f"{context}.enabled must be boolean")
        expected = _ENABLED_SOURCE_KEYS if enabled else _DISABLED_SOURCE_KEYS
        required = expected - ({"enabled", "validation_cache"} if enabled else set())
        missing = sorted(required - row.keys())
        unknown = sorted(row.keys() - expected)
        if missing or unknown:
            raise StudyConfigError(
                f"{context} keys do not match schema; missing={missing}, unknown={unknown}"
            )
        if not isinstance(row.get("dataset_id"), str) or not row["dataset_id"].strip():
            raise StudyConfigError(f"{context}.dataset_id must be non-empty")
        if enabled:
            enabled_count += 1
            for key in ("root", "samples", "taxa", "train_cache"):
                if not isinstance(row[key], str) or not row[key].strip():
                    raise StudyConfigError(f"{context}.{key} must be non-empty")
            validation_cache = row.get("validation_cache")
            if validation_cache is not None and (
                not isinstance(validation_cache, str) or not validation_cache.strip()
            ):
                raise StudyConfigError(f"{context}.validation_cache is invalid")
        elif not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise StudyConfigError(f"{context}.reason must explain the placeholder")
        result.append(row)
    if enabled_count == 0:
        raise StudyConfigError("at least one source must be enabled")
    dataset_ids = [str(row["dataset_id"]) for row in result]
    if len(dataset_ids) != len(set(dataset_ids)):
        raise StudyConfigError("source dataset ids must be unique, including placeholders")
    return tuple(result)


def load_study_config(path: Path | str, project_root: Path | str) -> StudyConfig:
    root = Path(project_root).expanduser().resolve()
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    value = _read_json_object(config_path, context="v2 study config")
    _require_exact_keys(value, _TOP_LEVEL_KEYS, context="v2 study config")
    if type(value["schema_version"]) is not int or value["schema_version"] != 2:
        raise StudyConfigError("schema_version must be the integer 2")
    experiment_id = value["experiment_id"]
    if not isinstance(experiment_id, str) or _EXPERIMENT.fullmatch(experiment_id) is None:
        raise StudyConfigError("experiment_id must be a safe identifier")
    if experiment_id != _V2_EXPERIMENT_ID:
        raise StudyConfigError(f"this launcher is locked to {_V2_EXPERIMENT_ID}")
    protocol = value["protocol"]
    if not isinstance(protocol, str) or not protocol.strip():
        raise StudyConfigError("protocol must be non-empty")
    if protocol != _V2_PROTOCOL:
        raise StudyConfigError(f"this launcher is locked to {_V2_PROTOCOL}")
    study_id = value["study_id"]
    if not isinstance(study_id, str) or _SLUG.fullmatch(study_id) is None:
        raise StudyConfigError("study_id must be a lowercase hyphenated slug")

    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise StudyConfigError("seeds must be a non-empty list")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise StudyConfigError("every seed must be a non-negative integer")
    if len(seeds) != len(set(seeds)):
        raise StudyConfigError("seeds must be unique")

    trial_values = value["source_configs"]
    if not isinstance(trial_values, list) or not trial_values:
        raise StudyConfigError("source_configs must be a non-empty list")
    trials: list[SourceTrial] = []
    for index, row in enumerate(trial_values):
        if not isinstance(row, dict):
            raise StudyConfigError(f"source_configs[{index}] must be an object")
        _require_exact_keys(row, _SOURCE_TRIAL_KEYS, context=f"source_configs[{index}]")
        trial_id = row["trial_id"]
        if not isinstance(trial_id, str) or _SLUG.fullmatch(trial_id) is None:
            raise StudyConfigError(f"source_configs[{index}].trial_id is not a slug")
        trial = SourceTrial(
            trial_id=trial_id,
            source_config=_resolve(root, row["path"], context=f"source_configs[{index}].path"),
        )
        _validated_source_rows(trial)
        trials.append(trial)
    if len({trial.trial_id for trial in trials}) != len(trials):
        raise StudyConfigError("trial ids must be unique")
    if len({trial.source_config for trial in trials}) != len(trials):
        raise StudyConfigError("source config paths must be unique")

    common_value = value["common"]
    if not isinstance(common_value, dict):
        raise StudyConfigError("common must be an object")
    _require_exact_keys(common_value, _COMMON_KEYS, context="common")
    for key in _COMMON_POSITIVE_INTEGER_KEYS:
        if type(common_value[key]) is not int or common_value[key] <= 0:
            raise StudyConfigError(f"common.{key} must be a positive integer")
    for key in _COMMON_FLOAT_KEYS:
        number = common_value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise StudyConfigError(f"common.{key} must be numeric")
        if not math.isfinite(float(number)) or float(number) < 0.0:
            raise StudyConfigError(f"common.{key} must be finite and non-negative")
    if float(common_value["learning_rate"]) == 0.0:
        raise StudyConfigError("common.learning_rate must be positive")
    if not isinstance(common_value["device"], str) or not common_value["device"].strip():
        raise StudyConfigError("common.device must be non-empty")
    common = CommonInputs(
        **{
            key: _resolve(root, common_value[key], context=f"common.{key}")
            for key in _COMMON_PATH_KEYS
        },
        **{key: int(common_value[key]) for key in _COMMON_POSITIVE_INTEGER_KEYS},
        **{key: float(common_value[key]) for key in _COMMON_FLOAT_KEYS},
        device=common_value["device"],
    )
    if experiment_id not in common.runs_root.parts:
        raise StudyConfigError(
            "common.runs_root must be an experiment-07-specific directory"
        )
    return StudyConfig(
        experiment_id=experiment_id,
        protocol=protocol,
        study_id=study_id,
        seeds=tuple(seeds),
        source_trials=tuple(trials),
        common=common,
    )


def build_trial_commands(
    config: StudyConfig,
    project_root: Path | str,
    *,
    python_executable: Path | str = sys.executable,
) -> tuple[TrialCommand, ...]:
    root = Path(project_root).expanduser().resolve()
    runner = root / "scripts/feature_adapter/run_clip_multi_bird.py"
    python_path = Path(python_executable).expanduser()
    if not python_path.is_absolute() and len(python_path.parts) > 1:
        python_path = root / python_path
    python_command = str(python_path)
    common = config.common
    result: list[TrialCommand] = []
    for seed in config.seeds:
        for trial in config.source_trials:
            kind = f"{config.study_id}-{trial.trial_id}-seed{seed}"
            argv = (
                python_command,
                "-u",
                str(runner),
                "--experiment-id",
                config.experiment_id,
                "--protocol",
                config.protocol,
                "--source-config",
                str(trial.source_config),
                "--cub-data-root",
                str(common.cub_data_root),
                "--cub-class-names",
                str(common.cub_class_names),
                "--birdnet-csv",
                str(common.birdnet_csv),
                "--cub-feature-cache",
                str(common.cub_feature_cache),
                "--model-cache-dir",
                str(common.model_cache_dir),
                "--text-cache",
                str(common.text_cache),
                "--runs-root",
                str(common.runs_root),
                "--steps",
                str(common.steps),
                "--validation-interval",
                str(common.validation_interval),
                "--patience-intervals",
                str(common.patience_intervals),
                "--batch-size",
                str(common.batch_size),
                "--learning-rate",
                str(common.learning_rate),
                "--weight-decay",
                str(common.weight_decay),
                "--identity-weight",
                str(common.identity_weight),
                "--hidden-dim",
                str(common.hidden_dim),
                "--seed",
                str(seed),
                "--device",
                common.device,
                "--kind",
                kind,
            )
            result.append(TrialCommand(trial_id=trial.trial_id, seed=seed, argv=argv))
    return tuple(result)


def _resolve_one_source_path(project_root: Path, value: object, *, context: str) -> Path:
    path = _resolve(project_root, value, context=context)
    if any(character in str(path) for character in "*?["):
        matches = sorted(path.parent.glob(path.name))
        if len(matches) != 1:
            raise StudyConfigError(
                f"{context} must resolve exactly once: {path} -> {matches}"
            )
        path = matches[0]
    return path


def preflight_runtime_inputs(config: StudyConfig, project_root: Path | str) -> None:
    root = Path(project_root).expanduser().resolve()
    files = {
        "runner": root / "scripts/feature_adapter/run_clip_multi_bird.py",
        "CUB class names": config.common.cub_class_names,
        "BirdNET taxonomy CSV": config.common.birdnet_csv,
        "CUB feature cache": config.common.cub_feature_cache,
    }
    directories = {
        "CUB data root": config.common.cub_data_root,
        "model cache directory": config.common.model_cache_dir,
    }
    for label, path in files.items():
        if not path.is_file():
            raise StudyConfigError(f"Missing {label}: {path}")
    for label, path in directories.items():
        if not path.is_dir():
            raise StudyConfigError(f"Missing {label}: {path}")
    for trial in config.source_trials:
        for index, row in enumerate(_validated_source_rows(trial)):
            if row.get("enabled", True) is False:
                continue
            context = f"{trial.trial_id}.sources[{index}]"
            resolved = {
                key: _resolve_one_source_path(root, row[key], context=f"{context}.{key}")
                for key in ("root", "samples", "taxa", "train_cache")
            }
            if not resolved["root"].is_dir():
                raise StudyConfigError(f"Missing source directory: {resolved['root']}")
            for key in ("samples", "taxa", "train_cache"):
                if not resolved[key].is_file():
                    raise StudyConfigError(f"Missing source file: {resolved[key]}")
            source_metadata = resolved["samples"].parent / "source.json"
            if not source_metadata.is_file():
                raise StudyConfigError(f"Missing source metadata: {source_metadata}")
            if row.get("validation_cache") is not None:
                validation = _resolve_one_source_path(
                    root,
                    row["validation_cache"],
                    context=f"{context}.validation_cache",
                )
                if not validation.is_file():
                    raise StudyConfigError(f"Missing source file: {validation}")


def run_commands(
    commands: tuple[TrialCommand, ...],
    project_root: Path | str,
    *,
    dry_run: bool,
) -> None:
    root = Path(project_root).expanduser().resolve()
    environment = os.environ.copy()
    source_root = str(root / "src")
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited else os.pathsep.join((source_root, inherited))
    )
    for index, command in enumerate(commands, start=1):
        print(
            f"[{index}/{len(commands)}] trial={command.trial_id} seed={command.seed}\n"
            f"{shlex.join(command.argv)}",
            flush=True,
        )
        if not dry_run:
            subprocess.run(command.argv, cwd=root, check=True, env=environment)


def _parse_args(project_root: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-config",
        type=Path,
        default=(
            project_root
            / "experiments/07_feature_adapter_clip_birdmix_v2_cub/configs/"
            "clip_birdmix_v2_study.json"
        ),
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    args = _parse_args(project_root)
    try:
        config = load_study_config(args.study_config, project_root)
        commands = build_trial_commands(
            config,
            project_root,
            python_executable=args.python,
        )
        if not args.dry_run:
            preflight_runtime_inputs(config, project_root)
        run_commands(commands, project_root, dry_run=args.dry_run)
    except StudyConfigError as error:
        raise SystemExit(f"BirdMix-v2 preflight failed: {error}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
