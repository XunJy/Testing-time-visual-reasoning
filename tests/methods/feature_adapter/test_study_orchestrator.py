from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.feature_adapter import run_clip_birdmix_study as study

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_CONFIG = (
    PROJECT_ROOT
    / "experiments/06_feature_adapter_clip_multi_bird/configs/"
    "clip_birdmix_preregistered_v1.json"
)


def _option(argv: tuple[str, ...], name: str) -> str:
    index = argv.index(name)
    return argv[index + 1]


def test_preregistered_config_builds_two_trials_for_each_locked_seed() -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(
        config,
        PROJECT_ROOT,
        python_executable=".venv/bin/python",
    )

    assert config.seeds == (2026, 2027, 2028)
    assert tuple(trial.trial_id for trial in config.source_trials) == (
        "inat-only",
        "birdmix-v1",
    )
    assert [(command.seed, command.trial_id) for command in commands] == [
        (2026, "inat-only"),
        (2026, "birdmix-v1"),
        (2027, "inat-only"),
        (2027, "birdmix-v1"),
        (2028, "inat-only"),
        (2028, "birdmix-v1"),
    ]


def test_commands_pass_every_common_input_explicitly() -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(config, PROJECT_ROOT, python_executable="python")

    for command in commands:
        argv = command.argv
        assert argv[:3] == (
            str(PROJECT_ROOT / "python"),
            "-u",
            str(PROJECT_ROOT / "scripts/feature_adapter/run_clip_multi_bird.py"),
        )
        assert _option(argv, "--source-config").endswith(".json")
        assert _option(argv, "--cub-data-root") == str(config.common.cub_data_root)
        assert _option(argv, "--cub-class-names") == str(config.common.cub_class_names)
        assert _option(argv, "--birdnet-csv") == str(config.common.birdnet_csv)
        assert _option(argv, "--cub-feature-cache") == str(config.common.cub_feature_cache)
        assert _option(argv, "--model-cache-dir") == str(config.common.model_cache_dir)
        assert _option(argv, "--text-cache") == str(config.common.text_cache)
        assert _option(argv, "--runs-root") == str(config.common.runs_root)
        assert _option(argv, "--steps") == "20000"
        assert _option(argv, "--validation-interval") == "250"
        assert _option(argv, "--patience-intervals") == "8"
        assert _option(argv, "--batch-size") == "256"
        assert _option(argv, "--learning-rate") == "0.0003"
        assert _option(argv, "--weight-decay") == "0.0001"
        assert _option(argv, "--identity-weight") == "0.1"
        assert _option(argv, "--hidden-dim") == "128"
        assert _option(argv, "--seed") == str(command.seed)
        assert _option(argv, "--device") == "cuda"
        assert _option(argv, "--kind").endswith(
            f"-{command.trial_id}-seed{command.seed}"
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unexpected": True}), "unknown"),
        (lambda value: value.update({"seeds": [2026, 2026]}), "unique"),
        (lambda value: value["common"].update({"batch_size": 0}), "positive"),
    ],
)
def test_config_validation_fails_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    value = json.loads(STUDY_CONFIG.read_text(encoding="utf-8"))
    mutation(value)  # type: ignore[operator]
    path = tmp_path / "study.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(study.StudyConfigError, match=message):
        study.load_study_config(path, PROJECT_ROOT)


def test_dry_run_prints_commands_without_starting_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(config, PROJECT_ROOT, python_executable="python")

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not start a subprocess")

    monkeypatch.setattr(study.subprocess, "run", unexpected)
    study.run_commands(commands, PROJECT_ROOT, dry_run=True)

    output = capsys.readouterr().out
    assert output.count("run_clip_multi_bird.py") == 6


def test_subprocess_failure_stops_later_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    config = study.load_study_config(STUDY_CONFIG, PROJECT_ROOT)
    commands = study.build_trial_commands(config, PROJECT_ROOT, python_executable="python")
    calls: list[tuple[str, ...]] = []

    def fail_second(argv: tuple[str, ...], **kwargs: object) -> None:
        calls.append(argv)
        assert kwargs["cwd"] == PROJECT_ROOT
        assert kwargs["check"] is True
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert str(PROJECT_ROOT / "src") in environment["PYTHONPATH"].split(":")
        if len(calls) == 2:
            raise subprocess.CalledProcessError(9, argv)

    monkeypatch.setattr(study.subprocess, "run", fail_second)

    with pytest.raises(subprocess.CalledProcessError):
        study.run_commands(commands, PROJECT_ROOT, dry_run=False)
    assert len(calls) == 2
