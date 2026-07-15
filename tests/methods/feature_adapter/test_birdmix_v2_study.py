from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.feature_adapter import run_clip_birdmix_v2_study as study

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments/07_feature_adapter_clip_birdmix_v2_cub"
STUDY_CONFIG = EXPERIMENT_ROOT / "configs/clip_birdmix_v2_study.json"
DRIVE_ROOT = "/content/drive/MyDrive/Testing-Time Visual Reasoning"


def _option(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_locked_v2_study_builds_three_isolated_experiment_commands() -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(
        config,
        PROJECT_ROOT,
        python_executable=".venv/bin/python",
    )

    assert config.experiment_id == "07_feature_adapter_clip_birdmix_v2_cub"
    assert config.protocol == "external-only-strict-cub-birdmix-v2"
    assert config.seeds == (2026, 2027, 2028)
    assert [command.seed for command in commands] == [2026, 2027, 2028]
    assert all(command.trial_id == "all-enabled-sources" for command in commands)
    for command in commands:
        assert _option(command.argv, "--experiment-id") == config.experiment_id
        assert _option(command.argv, "--protocol") == config.protocol
        assert _option(command.argv, "--runs-root") == str(config.common.runs_root)
        assert "06_feature_adapter_clip_multi_bird/runs" not in command.argv


def test_v2_source_config_enables_all_six_audited_sources() -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    rows = study._validated_source_rows(config.source_trials[0])

    enabled = [row for row in rows if row.get("enabled", True)]
    disabled = [row for row in rows if row.get("enabled", True) is False]
    assert [row["dataset_id"] for row in enabled] == [
        "inaturalist-2021-mini-aves",
        "birdnet-taxonomy-v0.3-jul2026",
        "big-bird-v2026.03.21-bbox-crops",
        "visual-wetlandbirds-v4-stride10-crops",
        "usgs-aerial-avian-2023-publisher-crops",
        "nm-uas-waterfowl-expert-consensus-v1",
    ]
    assert disabled == []


def test_v2_drive_paths_use_the_existing_space_named_root() -> None:
    study_value = json.loads(STUDY_CONFIG.read_text(encoding="utf-8"))
    source_value = json.loads(
        (EXPERIMENT_ROOT / "configs/birdmix_v2.json").read_text(encoding="utf-8")
    )
    drive_paths = [
        study_value["common"]["text_cache"],
        study_value["common"]["runs_root"],
        *(
            row[key]
            for row in source_value["sources"]
            for key in ("samples", "taxa", "train_cache", "validation_cache")
            if key in row
        ),
    ]

    assert drive_paths
    assert all(path.startswith(f"{DRIVE_ROOT}/") for path in drive_paths)
    assert not any("Testing-Time-Visual-Reasoning" in path for path in drive_paths)
    readme = (EXPERIMENT_ROOT / "README.md").read_text(encoding="utf-8")
    assert DRIVE_ROOT in readme
    assert "Testing-Time-Visual-Reasoning" not in readme


def test_v2_config_rejects_an_enabled_source_without_locked_paths(
    tmp_path: Path,
) -> None:
    source = json.loads(
        (EXPERIMENT_ROOT / "configs/birdmix_v2.json").read_text(encoding="utf-8")
    )
    source["sources"][-1].pop("train_cache")
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    trial = study.SourceTrial("invalid-placeholder", source_path)

    with pytest.raises(study.StudyConfigError, match="missing"):
        study._validated_source_rows(trial)


def test_v2_json_schemas_are_valid_json_and_lock_versions() -> None:
    source_schema = json.loads(
        (EXPERIMENT_ROOT / "schemas/source_config.schema.json").read_text(
            encoding="utf-8"
        )
    )
    study_schema = json.loads(
        (EXPERIMENT_ROOT / "schemas/study.schema.json").read_text(encoding="utf-8")
    )

    assert source_schema["$schema"].endswith("2020-12/schema")
    assert study_schema["properties"]["schema_version"]["const"] == 2
    assert study_schema["properties"]["experiment_id"]["const"] == (
        "07_feature_adapter_clip_birdmix_v2_cub"
    )


def test_v2_dry_run_does_not_start_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(config, PROJECT_ROOT, python_executable="python3")
    assert all(command.argv[0] == "python3" for command in commands)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not start a subprocess")

    monkeypatch.setattr(study.subprocess, "run", unexpected)
    study.run_commands(commands, PROJECT_ROOT, dry_run=True)

    output = capsys.readouterr().out
    assert output.count("run_clip_multi_bird.py") == 3
    assert output.count("07_feature_adapter_clip_birdmix_v2_cub") >= 3
