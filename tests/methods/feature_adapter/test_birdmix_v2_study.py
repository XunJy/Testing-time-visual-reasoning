from __future__ import annotations

import json
from dataclasses import replace
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
    source = json.loads((EXPERIMENT_ROOT / "configs/birdmix_v2.json").read_text(encoding="utf-8"))
    source["sources"][-1].pop("train_cache")
    source_path = tmp_path / "sources.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    trial = study.SourceTrial("invalid-placeholder", source_path)

    with pytest.raises(study.StudyConfigError, match="missing"):
        study._validated_source_rows(trial)


def test_v2_json_schemas_are_valid_json_and_lock_versions() -> None:
    source_schema = json.loads(
        (EXPERIMENT_ROOT / "schemas/source_config.schema.json").read_text(encoding="utf-8")
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


def test_v2_main_dry_run_does_not_acquire_gpu_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    monkeypatch.setattr(
        study,
        "_parse_args",
        lambda project_root: study.argparse.Namespace(
            study_config=STUDY_CONFIG,
            python=Path("python3"),
            gpu_lock=tmp_path / "must-not-be-created.lock",
            dry_run=True,
        ),
    )
    monkeypatch.setattr(
        study,
        "run_commands",
        lambda commands, project_root, *, dry_run: calls.append(dry_run),
    )
    monkeypatch.setattr(
        study,
        "run_formal_matrix",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("dry-run must not enter the formal lock")
        ),
    )

    study.main()

    assert calls == [True]
    assert not (tmp_path / "must-not-be-created.lock").exists()


def test_v2_preflight_runs_in_selected_python_and_pins_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    text_cache = tmp_path / "text.pt"
    text_cache.write_bytes(b"validated by the test double")
    common = replace(
        config.common,
        model_cache_dir=tmp_path,
        text_cache=text_cache,
    )
    selected_python = tmp_path / "selected-python"
    calls: dict[str, object] = {}
    identity = {
        "cache_identity": ("openai-clip:ViT-L/14@336px@a1d071733d7111c9c014f024669f959182114e33"),
        "clip_commit": "a1d071733d7111c9c014f024669f959182114e33",
        "checkpoint_sha256": ("3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"),
    }

    def run(command: tuple[str, ...], **kwargs: object) -> object:
        calls["command"] = command
        calls["kwargs"] = kwargs
        return study.subprocess.CompletedProcess(command, 0, stdout=json.dumps(identity))

    monkeypatch.setattr(study.subprocess, "run", run)

    actual = study._preflight_clip_runtime(
        common,
        tmp_path,
        python_executable=selected_python,
    )

    command = calls["command"]
    assert isinstance(command, tuple)
    assert command[0] == str(selected_python)
    assert command[1].endswith("scripts/feature_adapter/verify_clip_runtime.py")
    assert command[-4:] == (
        "--model-cache-dir",
        str(tmp_path),
        "--text-cache",
        str(text_cache),
    )
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["cwd"] == tmp_path
    assert actual == identity


def test_v2_preflight_wraps_clip_identity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    common = replace(config.common, model_cache_dir=tmp_path, text_cache=tmp_path / "new.pt")

    def fail(command: tuple[str, ...], **kwargs: object) -> object:
        raise study.subprocess.CalledProcessError(
            1,
            command,
            stderr="wrong commit",
        )

    monkeypatch.setattr(study.subprocess, "run", fail)

    with pytest.raises(study.StudyConfigError, match="wrong commit"):
        study._preflight_clip_runtime(
            common,
            tmp_path,
            python_executable=tmp_path / "selected-python",
        )


def test_formal_matrix_holds_gpu_lock_across_preflight_and_all_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(config, PROJECT_ROOT)
    lock_path = tmp_path / "gpu.lock"
    events: list[str] = []

    def assert_locked(stage: str) -> None:
        with pytest.raises(study.GPULockBusyError, match="lock is busy"):
            with study.exclusive_gpu_lock(lock_path, purpose="competing job"):
                pass
        events.append(stage)

    monkeypatch.setattr(
        study,
        "preflight_runtime_inputs",
        lambda config, project_root, *, python_executable: assert_locked("preflight"),
    )
    monkeypatch.setattr(
        study,
        "run_commands",
        lambda commands, project_root, *, dry_run: assert_locked("commands"),
    )

    study.run_formal_matrix(
        config,
        commands,
        PROJECT_ROOT,
        gpu_lock=lock_path,
        python_executable="selected-python",
    )

    assert events == ["preflight", "commands"]
    with study.exclusive_gpu_lock(lock_path, purpose="after matrix"):
        pass
