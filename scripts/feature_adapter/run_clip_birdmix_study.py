#!/usr/bin/env python3
"""Run the preregistered CLIP BirdMix study as six immutable trials."""

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
_TOP_LEVEL_KEYS = {"schema_version", "study_id", "seeds", "source_configs", "common"}
_SOURCE_KEYS = {"trial_id", "path"}
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


class StudyConfigError(ValueError):
    """The study cannot be launched without changing or completing its config."""


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
    study_id: str
    seeds: tuple[int, ...]
    source_trials: tuple[SourceTrial, ...]
    common: CommonInputs


@dataclass(frozen=True, slots=True)
class TrialCommand:
    trial_id: str
    seed: int
    argv: tuple[str, ...]


def _resolve(project_root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StudyConfigError(f"{field} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


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


def _validated_source_rows(trial: SourceTrial) -> tuple[dict[str, Any], ...]:
    """Validate the source-config contract without resolving its data paths."""

    value = _read_json_object(
        trial.source_config,
        context=f"source config for {trial.trial_id}",
    )
    validation_fraction = value.get("validation_taxon_fraction")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(float(validation_fraction))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise StudyConfigError(
            f"{trial.source_config} needs validation_taxon_fraction between zero and one"
        )
    rows = value.get("sources")
    if not isinstance(rows, list) or not rows:
        raise StudyConfigError(
            f"Source config for {trial.trial_id} has no non-empty sources list"
        )
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        context = f"{trial.trial_id}.sources[{index}]"
        if not isinstance(row, dict):
            raise StudyConfigError(f"{context} must be an object")
        for key in ("dataset_id", "root", "samples", "taxa", "train_cache"):
            if key not in row:
                raise StudyConfigError(f"{context}.{key} is required")
        for key in ("dataset_id", "root", "samples", "taxa", "train_cache"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise StudyConfigError(f"{context}.{key} must be a non-empty string")
        if row.get("validation_cache") is not None and (
            not isinstance(row["validation_cache"], str)
            or not row["validation_cache"].strip()
        ):
            raise StudyConfigError(
                f"{context}.validation_cache must be null or a non-empty string"
            )
        result.append(row)
    dataset_ids = [str(row["dataset_id"]) for row in result]
    if len(set(dataset_ids)) != len(dataset_ids):
        raise StudyConfigError(f"Source dataset ids must be unique in {trial.source_config}")
    return tuple(result)


def load_study_config(path: Path | str, project_root: Path | str) -> StudyConfig:
    """Parse the locked study schema without importing torch or the GPU runner."""

    root = Path(project_root).expanduser().resolve()
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    value = _read_json_object(config_path, context="study config")
    _require_exact_keys(value, _TOP_LEVEL_KEYS, context="study config")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise StudyConfigError("schema_version must be the integer 1")

    study_id = value["study_id"]
    if not isinstance(study_id, str) or _SLUG.fullmatch(study_id) is None:
        raise StudyConfigError("study_id must be a lowercase hyphenated slug")

    seed_values = value["seeds"]
    if not isinstance(seed_values, list) or not seed_values:
        raise StudyConfigError("seeds must be a non-empty list")
    if any(type(seed) is not int or seed < 0 for seed in seed_values):
        raise StudyConfigError("every seed must be a non-negative integer")
    if len(set(seed_values)) != len(seed_values):
        raise StudyConfigError("seeds must be unique")

    source_values = value["source_configs"]
    if not isinstance(source_values, list) or not source_values:
        raise StudyConfigError("source_configs must be a non-empty list")
    source_trials: list[SourceTrial] = []
    for index, row in enumerate(source_values):
        if not isinstance(row, dict):
            raise StudyConfigError(f"source_configs[{index}] must be an object")
        _require_exact_keys(row, _SOURCE_KEYS, context=f"source_configs[{index}]")
        trial_id = row["trial_id"]
        if not isinstance(trial_id, str) or _SLUG.fullmatch(trial_id) is None:
            raise StudyConfigError(f"source_configs[{index}].trial_id is not a slug")
        source_trials.append(
            SourceTrial(
                trial_id=trial_id,
                source_config=_resolve(
                    root,
                    row["path"],
                    field=f"source_configs[{index}].path",
                ),
            )
        )
    if len({trial.trial_id for trial in source_trials}) != len(source_trials):
        raise StudyConfigError("source trial ids must be unique")
    if len({trial.source_config for trial in source_trials}) != len(source_trials):
        raise StudyConfigError("source config paths must be unique")
    for trial in source_trials:
        _validated_source_rows(trial)

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
    device = common_value["device"]
    if not isinstance(device, str) or not device.strip():
        raise StudyConfigError("common.device must be a non-empty string")

    common_paths = {
        key: _resolve(root, common_value[key], field=f"common.{key}")
        for key in _COMMON_PATH_KEYS
    }
    common = CommonInputs(
        **common_paths,
        **{key: int(common_value[key]) for key in _COMMON_POSITIVE_INTEGER_KEYS},
        **{key: float(common_value[key]) for key in _COMMON_FLOAT_KEYS},
        device=device,
    )
    return StudyConfig(
        study_id=study_id,
        seeds=tuple(seed_values),
        source_trials=tuple(source_trials),
        common=common,
    )


def build_trial_commands(
    config: StudyConfig,
    project_root: Path | str,
    *,
    python_executable: Path | str = sys.executable,
) -> tuple[TrialCommand, ...]:
    """Construct every seed-by-source command with no implicit runner arguments."""

    root = Path(project_root).expanduser().resolve()
    runner = root / "scripts/feature_adapter/run_clip_multi_bird.py"
    python_path = Path(python_executable).expanduser()
    if not python_path.is_absolute():
        python_path = root / python_path
    common = config.common
    result: list[TrialCommand] = []
    for seed in config.seeds:
        for trial in config.source_trials:
            kind = f"{config.study_id}-{trial.trial_id}-seed{seed}"
            argv = (
                str(python_path),
                "-u",
                str(runner),
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
    if not isinstance(value, str):
        raise StudyConfigError(f"{context} must be a path string")
    path = _resolve(project_root, value, field=context)
    if any(character in str(path) for character in "*?["):
        matches = sorted(path.parent.glob(path.name))
        if len(matches) != 1:
            raise StudyConfigError(
                f"{context} must resolve to exactly one file: {path} -> {matches}"
            )
        path = matches[0]
    return path


def preflight_runtime_inputs(config: StudyConfig, project_root: Path | str) -> None:
    """Check all six trials' shared and source inputs before starting the first."""

    root = Path(project_root).expanduser().resolve()
    runner = root / "scripts/feature_adapter/run_clip_multi_bird.py"
    required_files = {
        "runner": runner,
        "CUB class names": config.common.cub_class_names,
        "BirdNET taxonomy CSV": config.common.birdnet_csv,
        "CUB feature cache": config.common.cub_feature_cache,
    }
    required_directories = {
        "CUB data root": config.common.cub_data_root,
        "model cache directory": config.common.model_cache_dir,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise StudyConfigError(f"Missing {label}: {path}")
    for label, path in required_directories.items():
        if not path.is_dir():
            raise StudyConfigError(f"Missing {label}: {path}")
    for output in (config.common.text_cache, config.common.runs_root):
        if output.exists() and output.is_dir() != (output == config.common.runs_root):
            raise StudyConfigError(f"Output path has the wrong type: {output}")
        if output.parent.exists() and not output.parent.is_dir():
            raise StudyConfigError(f"Output parent is not a directory: {output.parent}")

    for trial in config.source_trials:
        for index, row in enumerate(_validated_source_rows(trial)):
            context = f"{trial.trial_id}.sources[{index}]"
            resolved: dict[str, Path] = {}
            for key in ("root", "samples", "taxa", "train_cache"):
                path = _resolve_one_source_path(root, row[key], context=f"{context}.{key}")
                resolved[key] = path
                expected_directory = key == "root"
                if expected_directory and not path.is_dir():
                    raise StudyConfigError(f"Missing source directory: {path}")
                if not expected_directory and not path.is_file():
                    raise StudyConfigError(f"Missing source file: {path}")
            source_metadata = resolved["samples"].parent / "source.json"
            if not source_metadata.is_file():
                raise StudyConfigError(f"Missing source metadata: {source_metadata}")
            validation_cache = row.get("validation_cache")
            if validation_cache is not None:
                path = _resolve_one_source_path(
                    root,
                    validation_cache,
                    context=f"{context}.validation_cache",
                )
                if not path.is_file():
                    raise StudyConfigError(f"Missing source file: {path}")


def run_commands(
    commands: tuple[TrialCommand, ...],
    project_root: Path | str,
    *,
    dry_run: bool,
) -> None:
    """Run sequentially; inherited stdio streams logs and check=True stops on failure."""

    root = Path(project_root).expanduser().resolve()
    environment = os.environ.copy()
    source_root = str(root / "src")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not inherited_pythonpath
        else os.pathsep.join((source_root, inherited_pythonpath))
    )
    for index, command in enumerate(commands, start=1):
        printable = shlex.join(command.argv)
        print(
            f"[{index}/{len(commands)}] trial={command.trial_id} seed={command.seed}\n"
            f"{printable}",
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
            / "experiments/06_feature_adapter_clip_multi_bird/configs/"
            "clip_birdmix_preregistered_v1.json"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter used for each core-runner subprocess.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the study schema and print all commands without checking data caches.",
    )
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
        raise SystemExit(f"Study preflight failed: {error}") from error
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error


if __name__ == "__main__":
    main()
